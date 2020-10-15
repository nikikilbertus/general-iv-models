"""Data loading and pre-processing utilities."""

from typing import Tuple, Callable, Sequence, Text, Dict, Union

import os

from absl import logging

import jax.numpy as np
from jax import random

import numpy as onp
import pandas as pd
from scipy.stats import norm

import utils


DataSynth = Tuple[Dict[Text, Union[np.ndarray, float, None]],
                  np.ndarray, np.ndarray]
DataReal = Dict[Text, Union[np.ndarray, float, None]]
ArrayTup = Tuple[np.ndarray, np.ndarray]

Equations = Dict[Text, Callable[..., np.ndarray]]


# =============================================================================
# NOISE SOURCES
# =============================================================================


def std_normal_1d(key: np.ndarray, num: int) -> np.ndarray:
  """Generate a Gaussian for the confounder."""
  return random.normal(key, (num,))


def std_normal_2d(key: np.ndarray, num: int) -> ArrayTup:
  """Generate a multivariate Gaussian for the noises e_X, e_Y."""
  key1, key2 = random.split(key)
  return random.normal(key1, (num,)), random.normal(key2, (num,))


# =============================================================================
# SYNTHETIC STRUCTURAL EQUATIONS
# =============================================================================


structural_equations = {
  "lin1": {
    "noise": std_normal_2d,
    "confounder": std_normal_1d,
    "f_z": std_normal_1d,
    "f_x": lambda z, c, ex: 0.5 * z + 3 * c + ex,
    "f_y": lambda x, c, ey: x - 6 * c + ey,
  },
  "lin2": {
    "noise": std_normal_2d,
    "confounder": std_normal_1d,
    "f_z": std_normal_1d,
    "f_x": lambda z, c, ex: 3.0 * z + 0.5 * c + ex,
    "f_y": lambda x, c, ey: x - 6 * c + ey,
  },
  "quad1": {
    "noise": std_normal_2d,
    "confounder": std_normal_1d,
    "f_z": std_normal_1d,
    "f_x": lambda z, c, ex: 0.5 * z + 3 * c + ex,
    "f_y": lambda x, c, ey: 0.3 * x ** 2 - 1.5 * x * c + ey,
  },
  "quad2": {
    "noise": std_normal_2d,
    "confounder": std_normal_1d,
    "f_z": std_normal_1d,
    "f_x": lambda z, c, ex: 3.0 * z + 0.5 * c + ex,
    "f_y": lambda x, c, ey: 0.3 * x ** 2 - 1.5 * x * c + ey,
  },
}


# =============================================================================
# DATA GENERATORS
# =============================================================================

def whiten(
  inputs: Dict[Text, np.ndarray]
) -> Dict[Text, Union[float, np.ndarray, None]]:
  """Whiten each input."""
  res = {}
  for k, v in inputs.items():
    if v is not None:
      mu = np.mean(v, 0)
      std = np.maximum(np.std(v, 0), 1e-7)
      res[k + "_mu"] = mu
      res[k + "_std"] = std
      res[k] = (v - mu) / std
    else:
      res[k] = v
  return res


def whiten_with_mu_std(val: np.ndarray, mu: float, std: float) -> np.ndarray:
  return (val - mu) / std


def get_synth_data(
  key: np.ndarray,
  num: int,
  equations: Text,
  num_xstar: int = 100,
  external_equations: Equations = None,
  disconnect_instrument: bool = False
) -> DataSynth:
  """Generate some synthetic data.

    Args:
      key: A JAX random key.
      num: The number of examples to generate.
      equations: Which structural equations to choose for x and y. Default: 1
      num_xstar: Size of grid for interventions on x.
      external_equations: A dictionary that must contain the keys 'f_x' and
        'f_y' mapping to callables as values that take two np.ndarrays as
        arguments and produce another np.ndarray. These are the structural
        equations for X and Y in the graph Z -> X -> Y.
        If this argument is not provided, the `equation` argument selects
        structural equations from the pre-defined dict `structural_equations`.
      disconnect_instrument: Whether to regenerate random (standard Gaussian)
        values for the instrument after the data has been generated. This
        serves for diagnostic purposes, i.e., looking at the same x, y data,

    Returns:
      A 3-tuple (values, xstar, ystar) consisting a dictionary `values`
          containing values for x, y, z, confounder, ex, ey as well as two
          array xstar, ystar containing values for the true cause-effect.
  """
  if external_equations is not None:
    eqs = external_equations
  elif equations == "np":
    return get_newey_powell(key, num, num_xstar)
  else:
    eqs = structural_equations[equations]

  key, subkey = random.split(key)
  ex, ey = eqs["noise"](subkey, num)
  key, subkey = random.split(key)
  confounder = eqs["confounder"](subkey, num)
  key, subkey = random.split(key)
  z = eqs["f_z"](subkey, num)
  x = eqs["f_x"](z, confounder, ex)
  y = eqs["f_y"](x, confounder, ey)

  values = whiten({'x': x, 'y': y, 'z': z, 'confounder': confounder,
                   'ex': ex, 'ey': ey})

  # Evaluate E[ Y | do(x^*)] empirically
  xmin, xmax = np.min(x), np.max(x)
  xstar = np.linspace(xmin, xmax, num_xstar)
  ystar = []
  for _ in range(500):
    key, subkey = random.split(key)
    tmpey = eqs["noise"](subkey, num_xstar)[1]
    key, subkey = random.split(key)
    tmpconf = eqs["confounder"](subkey, num_xstar)
    tmp_ystar = whiten_with_mu_std(
      eqs["f_y"](xstar, tmpconf, tmpey), values["y_mu"], values["y_std"])
    ystar.append(tmp_ystar)
  ystar = np.array(ystar)
  xstar = whiten_with_mu_std(xstar, values["x_mu"], values["x_std"])
  if disconnect_instrument:
    key, subkey = random.split(key)
    values['z'] = random.normal(subkey, shape=z.shape)
  return values, xstar, ystar


def get_colonial_origins(data_dir: Text = "../data") -> DataReal:
  """Load data from colonial origins paper of Acemoglu."""
  stata_path = os.path.join(data_dir, "colonial_origins", "data.dta")
  df = pd.read_stata(stata_path)
  ycol = 'logpgp95'
  zcol = 'logem4'
  xcol = 'avexpr'
  df = df[[zcol, xcol, ycol]].dropna()
  z, x, y = df[zcol].values, df[xcol].values, df[ycol].values
  data = {'x': x, 'y': y, 'z': z, 'confounder': None, 'ex': None, 'ey': None}
  return whiten(data)


def get_newey_powell(key: np.ndarray,
                     num: int,
                     num_xstar: int = 100) -> DataSynth:
  """Get simulated Newey Powell (sigmoid design) data from KIV paper."""
  def np_true(vals: np.ndarray):
    return np.log(np.abs(16. * vals - 8) + 1) * np.sign(vals - 0.5)
  xstar = np.linspace(0, 1, num_xstar)
  ystar = np_true(xstar)

  mu = np.zeros(3)
  sigma = np.array([[1., 0.5, 0.], [0.5, 1., 0.], [0., 0., 1.]])
  r = random.multivariate_normal(key, mu, sigma, shape=(num,))
  u, t, w = r[:, 0], r[:, 1], r[:, 2]
  x = w + t
  x = norm.cdf(x / np.sqrt(2.))
  z = norm.cdf(w)
  e = u
  y = np_true(x) + e
  values = whiten({'x': x, 'y': y, 'z': z, 'ex': e, 'ey': e})
  xstar = whiten_with_mu_std(xstar, values['x_mu'], values['x_std'])
  ystar = whiten_with_mu_std(ystar, values['y_mu'], values['y_std'])
  values['confounder'] = None
  return values, xstar, ystar


# =============================================================================
# DISCRETIZATION AND CDF HANDLING
# =============================================================================


def ecdf(vals: np.ndarray, num_points: int = None) -> ArrayTup:
  """Evaluate the empirical distribution function on fixed number of points."""
  if num_points is None:
    num_points = len(vals)
  cdf = np.linspace(0, 1, num_points)
  t = np.quantile(vals, cdf)
  return t, cdf


def cdf_inv(vals: np.ndarray,
            num_points: int = None) -> Callable[..., np.ndarray]:
  """Compute an interpolation function of the (empirical) inverse cdf."""
  t, cdf = ecdf(vals, num_points)
  return lambda x: onp.interp(x, cdf, t)


def get_cdf_invs(val: np.ndarray,
                 bin_ids: np.ndarray,
                 num_z: int) -> Sequence[Callable[..., np.ndarray]]:
  """Compute a list of interpolated inverse CDFs of val at each z in Z grid."""
  cdf_invs = []
  for i in range(num_z):
    cdf_invs.append(cdf_inv(val[bin_ids == i]))
  return cdf_invs


def get_z_bin_assigment(z: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
  """Assignment of values in z to the respective bin in z_grid."""
  bins = np.concatenate((np.array([-np.inf]),
                         z_grid[1:-1],
                         np.array([np.inf])))
  hist = onp.digitize(z, bins=bins, right=True) - 1
  return hist


def get_x_samples(x: np.ndarray,
                  bin_ids: np.ndarray,
                  num_z: int,
                  num_sample: int) -> ArrayTup:
  """Pre-compute samples from p(x | z^{(i)}) for each gridpoint zi."""
  x_cdf_invs = get_cdf_invs(x, bin_ids, num_z)
  tmp = np.linspace(0, 1, num_sample + 2)[1:-1]
  tmp0 = utils.normal_cdf_inv(tmp, np.array([0]), np.array([0]))
  return tmp0, np.array([x_cdf_inv(tmp) for x_cdf_inv in x_cdf_invs])


def get_y_pre(y: np.ndarray,
              bin_ids: np.ndarray,
              num_z: int,
              num_points: int) -> np.ndarray:
  """Compute the grid of y points for constraint approach y."""
  y_cdf_invs = get_cdf_invs(y, bin_ids, num_z)
  grid = np.linspace(0, 1, num_points + 2)[1:-1]
  return np.array([y_cdf_inv(grid) for y_cdf_inv in y_cdf_invs])


def make_zgrid_and_binids(z: np.ndarray, num_z: int) -> ArrayTup:
  """Discretize instrument Z and assign all z points to corresponding bins."""
  if num_z <= 0:
    logging.info("Discrete instrument specified, checking for values.")
    z_grid = np.sort(onp.unique(z))
    if len(z_grid) > 50:
      logging.info("Found more than 50 unique values for z. This is not a "
                   "discrete instrument. Aborting!")
      raise RuntimeError("Discrete instrument specified, but not found.")
    logging.info(f"Found {len(z_grid)} unique values for discrete instrument.")
    bin_ids = - onp.ones_like(z)
    for i, zpoint in enumerate(z_grid):
      bin_ids[z == zpoint] = i
    if onp.any(bin_ids < 0):
      raise ValueError(f"Found negative value in bin_ids. "
                       "Couldn't discretize instrument.")
    bin_ids = np.array(bin_ids).astype(int)
  else:
    z_grid = ecdf(z, num_z + 1)[0]
    bin_ids = get_z_bin_assigment(z, z_grid)
    z_grid = (z_grid[:-1] + z_grid[1:]) / 2
  return z_grid, bin_ids
