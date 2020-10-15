"""Main entry point."""

from typing import Tuple, Text

import json
import os

from absl import app
from absl import flags
from absl import logging

from datetime import datetime

import jax.numpy as np
from jax import random, value_and_grad, jit
import jax.experimental.optimizers as optim
from jax.ops import index_update
import numpy as onp
from scipy.interpolate import UnivariateSpline

from tqdm import tqdm

import data
import kiv
import plotting
import utils

Params = Tuple[np.ndarray, np.ndarray, np.ndarray]

# ------------------------------- DATASET -------------------------------------
flags.DEFINE_enum("dataset", "synthetic",
                  ("synthetic", "nlsym", "colonial_origins"),
                  "Which dataset to use.")
flags.DEFINE_string("equations", "lin1",
                    "Which structural equations to use in synthetic setting.")
flags.DEFINE_bool("disconnect_instrument", False,
                  "Whether to resample independent values for instrument. "
                  "(Mostly for debugging.)")
flags.DEFINE_integer("num_data", 5_000,
                     "The number of observations in the synthetic dataset.")
# ---------------------------- APPROXIMATIONS ---------------------------------
flags.DEFINE_enum("response_type", "poly", ("poly", "gp", "mlp"),
                  "Basis response functions (polynomials or GP samples).")
flags.DEFINE_integer("num_xstar", 15,
                     "Number of x values at which to evaluate the objective.")
flags.DEFINE_integer("dim_theta", 2,
                     "The dimension of the parameter theta. This is also the "
                     "number of response basis functions to use.")
flags.DEFINE_integer("num_z", 20,
                     "The number of grid points for the instrument Z.")
# ---------------------------- OPTIMIZATION -----------------------------------
flags.DEFINE_integer("num_rounds", 150,
                     "Number of rounds in the augmented Lagrangian.")
flags.DEFINE_integer("opt_steps", 30,
                     "Number of gradient updates per optimization subproblem.")
flags.DEFINE_integer("bs", 1024,
                     "Number of examples for MC estimates of the objective.")
flags.DEFINE_integer("bs_constr", 4096,
                     "Number of examples for MC estimates of the constraints.")
# ---------------------------- CONSTRAINT -------------------------------------
flags.DEFINE_float("slack", 0.2,
                   "Fractional tolerance for the constraints.")
flags.DEFINE_float("slack_abs", 0.2,
                   "Additional absolute tolerance for the constraints.")
# --------------------- LEARNING RATE & MOMENTUM ------------------------------
flags.DEFINE_float("lr", 0.001,
                   "The (initial) learning rate for the optimization.")
flags.DEFINE_integer("decay_steps", 1000,
                     "Number of decay steps for the learning rate schedule.")
flags.DEFINE_float("decay_rate", 1.0,
                   "The decay rate for the learning rate schedule.")
flags.DEFINE_bool("staircase", False,
                  "Whether to use staircases in the learning rate schedule")
flags.DEFINE_float("momentum", 0.9,
                   "The momentum parameter for the SGD optimizer.")
# ---------------------------- SCHEDULES --------------------------------------
flags.DEFINE_float("tau_init", 0.1,
                   "The initial value of the temperature parameter.")
flags.DEFINE_float("tau_factor", 1.08,
                   "The factor by which tau is multiplied each round.")
flags.DEFINE_float("tau_max", 10.0,
                   "The maximum temperature tau.")
# ---------------------------- INPUT/OUTPUT -----------------------------------
flags.DEFINE_string("data_dir", "../data/",
                    "Directory of the input data.")
flags.DEFINE_string("output_dir", "../results/",
                    "Path to the output directory (for results).")
flags.DEFINE_string("output_name", "",
                    "Name for result folder. Use timestamp if empty.")
flags.DEFINE_bool("plot_init", False,
                  "Whether to plot data and initialization visuals.")
flags.DEFINE_bool("plot_intermediate", False,
                  "Whether to plot results from individual runs.")
flags.DEFINE_bool("plot_final", True,
                  "Whether to plot final aggregate results.")
flags.DEFINE_bool("store_data", False,
                  "Whether to store data, intermediate and final results and "
                  "baseline results.")
# ---------------------------- COMPARISONS ------------------------------------
flags.DEFINE_bool("run_2sls", True,
                  "Whether to run two stage least squares as comparison.")
flags.DEFINE_bool("run_kiv", True,
                  "Whether to run kernel instrumental variable as comparison.")
# ------------------------------ MISC -----------------------------------------
flags.DEFINE_integer("seed", 0, "The random seed.")
FLAGS = flags.FLAGS


# =============================================================================
# RHS CONSTRAINT FUNCTIONS THAT MUST BE OVERWRITTEN
# =============================================================================

@jit
def get_phi(y: np.ndarray) -> np.ndarray:
  """The phis for the constraints."""
  return np.array([np.mean(y, axis=-1), np.var(y, axis=-1)]).T


@jit
def get_rhs(thetahat: np.ndarray, xhats_pre: np.ndarray) -> np.ndarray:
  """Construct the RHS for the second approach (unsing basis functions phi)."""
  return get_phi(response(thetahat, xhats_pre))


def make_constraint_lhs(y: np.ndarray,
                        bin_ids: np.ndarray,
                        z_grid: np.ndarray) -> np.ndarray:
  """Get the LHS of the constraints."""
  # Use indicator annealing approach
  logging.info(f"Setup {FLAGS.num_z * 2} constraints...")
  tmp = []
  for i in range(FLAGS.num_z):
    tmp.append(get_phi(y[bin_ids == i]))
  tmp = np.array(tmp)
  # Smoothen LHS constraints with UnivariateSpline smoothing
  logging.info(f"Smoothen constraints with splines. Fixed factor: 0.2 ...")
  lhs = []
  for i in range(tmp.shape[-1]):
    spl = UnivariateSpline(z_grid, tmp[:, i], s=0.2)
    lhs.append(spl(z_grid))
  lhs = np.array(lhs).T
  return lhs


@jit
def response_poly(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
  """The response function."""
  return np.polyval(theta, x)


# Must be overwritten with one of the available response functions
# noinspection PyUnusedLocal
@jit
def response(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
  """The response function."""
  return np.empty((0,))


# =============================================================================
# OPTIMIZATION (AUGMENTED LAGRANGIAN)
# =============================================================================

@jit
def get_constraint_term(constr: np.ndarray,
                        lmbda: np.ndarray,
                        tau: float) -> float:
  """Compute the sum of \psi(c_i, \lambda, \tau) for the Lagrangian."""
  case1 = - lmbda * constr + 0.5 * tau * constr**2
  case2 = - 0.5 * lmbda**2 / tau
  psi = np.where(tau * constr <= lmbda, case1, case2)
  return np.sum(psi)


@jit
def update_lambda(constr: np.ndarray,
                  lmbda: np.ndarray,
                  tau: float) -> np.ndarray:
    """Update Lagrangian parameters lambda."""
    return np.maximum(lmbda - tau * constr, 0)


@jit
def make_cholesky_factor(l_param: np.ndarray) -> np.ndarray:
  """Get the actual cholesky factor from our parameterization of L."""
  lmask = np.tri(l_param.shape[0])
  lmask = index_update(lmask, (0, 0), 0)
  tmp = l_param * lmask
  idx = np.diag_indices(l_param.shape[0])
  return index_update(tmp, idx, np.exp(tmp[idx]))


@jit
def make_correlation_matrix(l_param: np.ndarray) -> np.ndarray:
    """Get correlation matrix from our parameterization of L."""
    chol = make_cholesky_factor(l_param)
    return chol @ chol.T


@jit
def objective_rhs_psisum_constr(
    key: np.ndarray,
    params: Params,
    lmbda: np.ndarray,
    tau: float,
    lhs: np.ndarray,
    slack: np.ndarray,
    xstar: float,
    tmp_pre: np.ndarray,
    xhats_pre: np.ndarray,
) -> Tuple[float, np.ndarray, float, np.ndarray]:
  """Estimate the objective, RHS, psisum (constraint term), and constraints.

  Refer to the docstring of `lagrangian` for a description of the arguments.
  """
  # (k+1, k+1), (k,), (k,)
  L, mu, log_sigma = params
  n = tmp_pre.shape[-1]
  # (k, n)
  tmp = random.normal(key, (FLAGS.dim_theta, n))
  # (k+1, n)
  tmp = np.concatenate((tmp_pre, tmp), axis=0)
  # (k+1, n) add initial dependence
  tmp = utils.std_normal_cdf(make_cholesky_factor(L) @ tmp)
  # (k, n) get thetas with current means and variances
  thetahat = utils.normal_cdf_inv(tmp[1:, :], mu, log_sigma)
  # (1,) main objective <- (n,) <- (k, n), ()
  obj = np.mean(response(thetahat, np.array(xstar)))
  # (m, l) computes rhs for all z
  rhs = get_rhs(thetahat, xhats_pre)
  # (m * l,) constraints (with tolerances)
  constr = slack - np.ravel(np.abs(lhs - rhs))
  # (1,) constraint term of lagrangian
  psisum = get_constraint_term(constr, lmbda, tau)
  return obj, rhs, psisum, constr


@jit
def lagrangian(key: np.ndarray,
               params: Params,
               lmbda: np.ndarray,
               tau: float,
               lhs: np.ndarray,
               slack: np.ndarray,
               xstar: float,
               tmp_pre: np.ndarray,
               xhats_pre: np.ndarray,
               sign: float = 1.) -> float:
  """Estimate the Lagrangian at a given \eta.

  For given $\eta$ compute MC estimate of the Lagrangian with samples from
  $p(\theta | x, z)$, which are used for the constraints, but also
  (marginalized) for the main objective.

  Args:
      key: Key for the random number generator.
      params: A 3-tuple with the parameters to optimize consisting of
          L: Lower triangular matrix from which we compute the Cholesky factor.
              (Not the Cholesky factor itself!).
              Dimension: (DIM_THETA + 1, DIM_THETA + 1)
          mu: The means of the (Gaussian) marginals of the thetas.
              Dimension: (DIM_THETA, )
          log_sigma: The log of the standard deviations of the (Gaussian)
              marginals of the thetas. (Use log to ensure they're positive).
              Dimension: (DIM_THETA, )
      lmbda: The Lagrangian multipliers lambda. Dimension: (NUM_Z * NUM_PHI, )
      tau: The temperature parameter for the augmented Lagrangian approach.
      lhs: The LHS of the constraints. Dimension: (NUM_Z, NUM_PHI)
      slack: The tolerance for how well the constraints must be satisfied.
          Dimension: (NUM_Z * NUM_PHI, )
      xstar: The interventional value of x in the objective.
      tmp_pre: Precomputed standard Guassian distributed values (for x).
          Dimension: (1, num_sample)
      xhats_pre: Precomputed samples following p(x | zi) for the zi in the
          Z grid (corresponding to the values in tmp_pre).
          Dimension: (NUM_Z, num_sample)
      sign: Either -1 or 1. If sign == 1, we are computing a lower bound.
          If sign == -1, we are computing an upper bound.

  Returns:
      a scalar estimate of the Lagrangian at the given eta and xstar
  """
  obj, _, psisum, _ = objective_rhs_psisum_constr(
      key, params, lmbda, tau, lhs, slack, xstar, tmp_pre, xhats_pre)
  return sign * obj + psisum


def init_params(key: np.ndarray) -> Params:
  """Initiliaze the optimization parameters."""
  key, subkey = random.split(key)
  # init diagonal at 0, because it will be exponentiated
  L = 0.05 * np.tri(FLAGS.dim_theta + 1, k=-1)
  L *= random.normal(subkey, (FLAGS.dim_theta + 1, FLAGS.dim_theta + 1))
  corr = make_correlation_matrix(L)
  assert np.all(np.isclose(np.linalg.cholesky(corr),
                           make_cholesky_factor(L))), "not PSD"
  key, subkey = random.split(key)
  if FLAGS.response_type == "poly":
    mu = 0.001 * random.normal(subkey, (FLAGS.dim_theta,))
    log_sigma = np.array([np.log(1. / (i + 1))
                          for i in range(FLAGS.dim_theta)])
  elif FLAGS.response_type == "gp":
    mu = np.ones(FLAGS.dim_theta) / FLAGS.dim_theta
    log_sigma = 0.5 * np.ones(FLAGS.dim_theta)
  else:
    mu = 0.01 * random.normal(subkey, (FLAGS.dim_theta,))
    log_sigma = 0.5 * np.ones(FLAGS.dim_theta)
  params = (L, mu, log_sigma)
  return params


lagrangian_value_and_grad = jit(value_and_grad(lagrangian, argnums=1))


def run_optim(key: np.ndarray,
              lhs: np.ndarray,
              tmp: np.ndarray,
              xhats: np.ndarray,
              tmp_c: np.ndarray,
              xhats_c: np.ndarray,
              xstar: float,
              bound: Text,
              out_dir: Text,
              x: np.ndarray,
              y: np.ndarray) -> Tuple[int, float, float, int, float, float]:
  """Run optimization (either lower or upper) for a single xstar."""
  # Directory setup
  # ---------------------------------------------------------------------------
  out_dir = os.path.join(out_dir, f"{bound}-xstar_{xstar}")
  if FLAGS.store_data:
    logging.info(f"Current run output directory: {out_dir}...")
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

  # Init optim params
  # ---------------------------------------------------------------------------
  logging.info(f"Initialize parameters L, mu, log_sigma, lmbda, tau, slack...")
  key, subkey = random.split(key)
  params = init_params(subkey)

  for parname, param in zip(['L', 'mu', 'log_sigma'], params):
    logging.info(f"Parameter {parname}: {param.shape}")
    logging.info(f"  -> {parname}: {param}")

  tau = FLAGS.tau_init
  logging.info(f"Initial tau = {tau}")
  fin_tau = np.minimum(FLAGS.tau_factor**FLAGS.num_rounds * tau, FLAGS.tau_max)
  logging.info(f"Final tau = {fin_tau}")

  # Set constraint approach and slacks
  # ---------------------------------------------------------------------------
  slack = FLAGS.slack * np.ones(FLAGS.num_z * 2)
  lmbda = np.zeros(FLAGS.num_z * 2)
  logging.info(f"Lambdas: {lmbda.shape}")

  logging.info(f"Fractional tolerance (slack) for constraints = {FLAGS.slack}")
  logging.info(f"Set relative slack variables...")
  slack *= np.abs(lhs.ravel())
  logging.info(f"Set minimum slack to {FLAGS.slack_abs}...")
  slack = np.maximum(FLAGS.slack_abs, slack)
  logging.info(f"Slack {slack.shape}")
  logging.info(f"Actual slack min: {np.min(slack)}, max: {np.max(slack)}")

  # Setup optimizer
  # ---------------------------------------------------------------------------
  logging.info(f"Vanilla SGD with init_lr={FLAGS.lr}...")
  logging.info(f"Set learning rate schedule")
  step_size = optim.inverse_time_decay(
    FLAGS.lr, FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.staircase)
  init_fun, update_fun, get_params = optim.sgd(step_size)

  logging.info(f"Init state for JAX optimizer (including L, mu, log_sigma)...")
  state = init_fun(params)

  # Setup result dict
  # ---------------------------------------------------------------------------
  logging.info(f"Initialize dictionary for results...")
  results = {
    "mu": [],
    "sigma": [],
    "cholesky_factor": [],
    "tau": [],
    "lambda": [],
    "objective": [],
    "constraint_term": [],
    "rhs": []
  }
  if FLAGS.plot_intermediate:
    results["grad_norms"] = []
    results["lagrangian"] = []

  logging.info(f"Evaluate at xstar={xstar}...")

  logging.info(f"Evaluate {bound} bound...")
  sign = 1 if bound == "lower" else -1

  # ===========================================================================
  # OPTIMIZATION LOOP
  # ===========================================================================
  # One-time logging before first step
  # ---------------------------------------------------------------------------
  key, subkey = random.split(key)
  obj, rhs, psisum, constr = objective_rhs_psisum_constr(
    subkey, get_params(state), lmbda, tau, lhs, slack, xstar, tmp_c, xhats_c)
  results["objective"].append(obj)
  results["constraint_term"].append(psisum)
  results["rhs"].append(rhs)

  logging.info(f"Objective: scalar")
  logging.info(f"RHS: {rhs.shape}")
  logging.info(f"Sum over Psis: scalar")
  logging.info(f"Constraint: {constr.shape}")

  tril_idx = np.tril_indices(FLAGS.dim_theta + 1)
  count = 0
  logging.info(f"Start optimization loop...")
  for _ in tqdm(range(FLAGS.num_rounds)):
    # log current parameters
    # -------------------------------------------------------------------------
    results["lambda"].append(lmbda)
    results["tau"].append(tau)
    cur_L, cur_mu, cur_logsigma = get_params(state)
    cur_chol = make_cholesky_factor(cur_L)[tril_idx].ravel()[1:]
    results["mu"].append(cur_mu)
    results["sigma"].append(np.exp(cur_logsigma))
    results["cholesky_factor"].append(cur_chol)

    subkeys = random.split(key, num=FLAGS.opt_steps + 1)
    key = subkeys[0]
    # inner optimization for subproblem
    # -------------------------------------------------------------------------
    for j in range(FLAGS.opt_steps):
      v, grads = lagrangian_value_and_grad(
        subkeys[j + 1], get_params(state), lmbda, tau, lhs, slack, xstar,
        tmp, xhats, sign)
      state = update_fun(count, grads, state)
      count += 1
      if FLAGS.plot_intermediate:
        results["lagrangian"].append(v)
        results["grad_norms"].append([np.linalg.norm(grad) for grad in grads])

    # post inner optimization logging
    # -------------------------------------------------------------------------
    key, subkey = random.split(key)
    obj, rhs, psisum, constr = objective_rhs_psisum_constr(
      subkey, get_params(state), lmbda, tau, lhs, slack, xstar, tmp_c, xhats_c)
    results["objective"].append(obj)
    results["constraint_term"].append(psisum)
    results["rhs"].append(rhs)

    # update lambda, tau
    # -------------------------------------------------------------------------
    lmbda = update_lambda(constr, lmbda, tau)
    tau = np.minimum(tau * FLAGS.tau_factor, FLAGS.tau_max)

  # Convert and store results
  # ---------------------------------------------------------------------------
  logging.info(f"Finished optimization loop...")

  logging.info(f"Convert all results to numpy arrays...")
  results = {k: np.array(v) for k, v in results.items()}

  logging.info(f"Add final parameters and lhs to results...")
  L, mu, log_sigma = get_params(state)
  results["final_L"] = L
  results["final_mu"] = mu
  results["final_log_sigma"] = log_sigma
  results["lhs"] = lhs

  if FLAGS.store_data:
    logging.info(f"Save result data to...")
    result_path = os.path.join(out_dir, "results.npz")
    onp.savez(result_path, **results)

  # Generate and store plots
  # ---------------------------------------------------------------------------
  if FLAGS.plot_intermediate:
    fig_dir = os.path.join(out_dir, "figures")
    logging.info(f"Generate and save all plots at {fig_dir}...")
    plotting.plot_all(results, x, y, response, fig_dir)

  # Compute last valid and last satisfied
  # ---------------------------------------------------------------------------
  maxabsdiff = np.array([np.max(np.abs(lhs - r)) for r in results["rhs"]])
  fin_i = np.sum(~np.isnan(results["objective"])) - 1
  logging.info(f"Final non-nan objective at {fin_i}.")
  fin_obj = results["objective"][fin_i]
  fin_maxabsdiff = maxabsdiff[fin_i]

  sat_i = [np.all((np.abs((lhs - r) / lhs) < FLAGS.slack) |
                  (np.abs(lhs - r) < FLAGS.slack_abs))
           for r in results["rhs"]]
  sat_i = np.where(sat_i)[0]

  if len(sat_i) > 0:
    sat_i = sat_i[-1]
    logging.info(f"Final satisfied constraint at {sat_i}.")
    sat_obj = results["objective"][sat_i]
    sat_maxabsdiff = maxabsdiff[sat_i]
  else:
    sat_i = -1
    logging.info(f"Constraints were never satisfied.")
    sat_obj, sat_maxabsdiff = np.nan, np.nan

  logging.info("Finished run.")
  return fin_i, fin_obj, fin_maxabsdiff, sat_i, sat_obj, sat_maxabsdiff


# =============================================================================
# MAIN
# =============================================================================

def main(_):
  # ---------------------------------------------------------------------------
  # Directory setup, save flags, set random seed
  # ---------------------------------------------------------------------------
  FLAGS.alsologtostderr = True

  if FLAGS.output_name == "":
    dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  else:
    dir_name = FLAGS.output_name
  out_dir = os.path.join(os.path.abspath(FLAGS.output_dir), dir_name)
  logging.info(f"Save all output to {out_dir}...")
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  FLAGS.log_dir = out_dir
  logging.get_absl_handler().use_absl_log_file(program_name="run")

  logging.info("Save FLAGS (arguments)...")
  with open(os.path.join(out_dir, 'flags.json'), 'w') as fp:
    json.dump(FLAGS.flag_values_dict(), fp, sort_keys=True, indent=2)

  logging.info(f"Set random seed {FLAGS.seed}...")
  key = random.PRNGKey(FLAGS.seed)

  # ---------------------------------------------------------------------------
  # Load and store data
  # ---------------------------------------------------------------------------
  logging.info(f"Get dataset: {FLAGS.dataset}")
  if FLAGS.dataset == "synthetic":
    logging.info(f"Generate synthetic data (n={FLAGS.num_data}) "
                 f"using equations {FLAGS.equations}...")
    key, subkey = random.split(key)
    dat, data_xstar, data_ystar = data.get_synth_data(
      subkey, FLAGS.num_data, FLAGS.equations,
      disconnect_instrument=FLAGS.disconnect_instrument)
  elif FLAGS.dataset == "colonial_origins":
    dat = data.get_colonial_origins(FLAGS.data_dir)
    data_xstar, data_ystar = None, None
  else:
    raise ValueError(f"Unknown dataset {FLAGS.dataset}")

  for k, v in dat.items():
    if v is not None:
      logging.info(f'{k}: {v.shape}')

  if FLAGS.store_data:
    logging.info(f"Store data...")
    result_path = os.path.join(out_dir, "data.npz")
    onp.savez(result_path, **dat, xstar=data_xstar, ystar=data_ystar)

  x, y, z, ex, ey = dat['x'], dat['y'], dat['z'], dat['ex'], dat['ey']
  confounder = dat['confounder']

  # ---------------------------------------------------------------------------
  # Discretize z, generate LHS and reusable x samples
  # ---------------------------------------------------------------------------
  logging.info(f"Discretize Z and bin datapoints (num_z={FLAGS.num_z})...")
  z_grid, bin_ids = data.make_zgrid_and_binids(z, FLAGS.num_z)
  if len(z_grid) != FLAGS.num_z:
    FLAGS.num_z = len(z_grid)
  logging.info(f"Updated num_z to {FLAGS.num_z}")

  # Set global functions depending on FLAGS
  # ---------------------------------------------------------------------------
  logging.info(f"Use response type {FLAGS.response_type}...")
  basis_predict = None
  global response
  if FLAGS.response_type == "poly":
    response = response_poly
  elif FLAGS.response_type == "gp":
    basis_predict = utils.get_gp_prediction(x, y, n_samples=FLAGS.dim_theta)

    @jit
    def response_gp(theta: np.ndarray, _x: np.ndarray) -> np.ndarray:
      _x = np.atleast_1d(_x)
      if _x.ndim == 1:
        # (n,) <- (1, k) @ (k, n)
        return (basis_predict(_x) @ theta).squeeze()
      else:
        # (n_constr, n) <- (n_constr, n, k) @ (k, n)
        return np.einsum('ijk,kj->ij', basis_predict(_x), theta)

    response = response_gp
  elif FLAGS.response_type == "mlp":
    key, subkey = random.split(key)
    basis_predict = utils.fit_mlp(subkey, x[:, np.newaxis], y,
                                  n_samples=FLAGS.dim_theta)

    @jit
    def response_mlp(theta: np.ndarray, _x: np.ndarray) -> np.ndarray:
      _x = np.atleast_2d(_x)
      if _x.shape[0] == 1:
        # (n,) <- (1, k) @ (k, n)
        return (basis_predict(_x) @ theta).squeeze()
      else:
        # (n_constr, n) <- (n_constr, n, k) @ (k, n)
        return np.einsum('ijk,kj->ij', basis_predict(_x[:, :, None]), theta)

    response = response_mlp
  else:
    raise NotImplementedError(f"Unknown response_type {FLAGS.response_type}.")

  logging.info(f"Make LHS of constraints ...")
  lhs = make_constraint_lhs(y, bin_ids, z_grid)
  logging.info(f"LHS: {lhs.shape} ...")

  logging.info(f"Generate fixed x samples for objective {FLAGS.bs}...")
  tmp, xhats = data.get_x_samples(x, bin_ids, FLAGS.num_z, FLAGS.bs)
  logging.info(f"tmp: {tmp.shape}...")
  logging.info(f"xhats: {xhats.shape}...")

  logging.info(f"Generate fixed x samples for constraint {FLAGS.bs_constr}...")
  tmp_c, xhats_c = data.get_x_samples(x, bin_ids, FLAGS.num_z, FLAGS.bs_constr)
  logging.info(f"tmp_c: {tmp_c.shape}...")
  logging.info(f"xhats_c: {xhats_c.shape}...")

  xmin, xmax = np.min(x), np.max(x)
  xstar_grid = np.linspace(xmin, xmax, FLAGS.num_xstar + 1)
  xstar_grid = (xstar_grid[:-1] + xstar_grid[1:]) / 2

  # ---------------------------------------------------------------------------
  # Plot data and initialization
  # ---------------------------------------------------------------------------
  if FLAGS.plot_init:
    logging.info(f"Plot data and discretization visuals...")
    plotting.plot_all_init(z, x, y, confounder, ex, ey, xhats_c, z_grid,
                           bin_ids, lhs, basis_func=basis_predict,
                           base_dir=out_dir)
  else:
    logging.info(f"Skip plots of data and discretization visuals...")

  # ---------------------------------------------------------------------------
  # Allocate memory for aggregate results
  # ---------------------------------------------------------------------------
  final = {
    "indices": np.zeros((FLAGS.num_xstar, 2), dtype=np.int32),
    "objective": np.zeros((FLAGS.num_xstar, 2)),
    "maxabsdiff": np.zeros((FLAGS.num_xstar, 2)),
  }
  satis = {
    "indices": np.zeros((FLAGS.num_xstar, 2), dtype=np.int32),
    "objective": np.zeros((FLAGS.num_xstar, 2)),
    "maxabsdiff": np.zeros((FLAGS.num_xstar, 2)),
  }

  # ---------------------------------------------------------------------------
  # Main loops over xstar and bounds
  # ---------------------------------------------------------------------------
  for i, xstar in enumerate(xstar_grid):
    for j, bound in enumerate(["lower", "upper"]):
      logging.info(f"Run xstar={xstar}, bound={bound}...")
      vis = "=" * 10
      logging.info(f"{vis} {i * 2 + j + 1}/{2 * FLAGS.num_xstar} {vis}")
      fin_i, fin_obj, fin_diff, sat_i, sat_obj, sat_diff = run_optim(
        key, lhs, tmp, xhats, tmp_c, xhats_c, xstar, bound, out_dir,
        x, y)
      final["indices"] = index_update(final["indices"], (i, j), fin_i)
      final["objective"] = index_update(final["objective"], (i, j), fin_obj)
      final["maxabsdiff"] = index_update(final["maxabsdiff"], (i, j), fin_diff)
      satis["indices"] = index_update(satis["indices"], (i, j), sat_i)
      satis["objective"] = index_update(satis["objective"], (i, j), sat_obj)
      satis["maxabsdiff"] = index_update(satis["maxabsdiff"], (i, j), sat_diff)

  # ---------------------------------------------------------------------------
  # Comparison methods
  # ---------------------------------------------------------------------------
  if FLAGS.run_2sls:
    logging.info(f"Compute 2SLS regression...")
    coeff_2sls = utils.two_stage_least_squares(z, x, y)
    if FLAGS.store_data:
      result_path = os.path.join(out_dir, "coeff_2sls.npz")
      onp.savez(result_path, coeff_2sls=coeff_2sls)
  else:
    coeff_2sls = None

  if FLAGS.run_kiv:
    logging.info(f"Compute KIV regression...")
    x_kiv, y_kiv = kiv.fit_kiv(z, x, y)
    if FLAGS.store_data:
      result_path = os.path.join(out_dir, "kiv_results.npz")
      onp.savez(result_path, x_star=x_kiv, y_star=y_kiv)
  else:
    x_kiv, y_kiv = None, None

  # ---------------------------------------------------------------------------
  # Store basis functions
  # ---------------------------------------------------------------------------
  if FLAGS.response_type != "poly":
    basis_x = np.linspace(xmin, xmax, 100)
    basis_y = basis_predict(basis_x).squeeze()
    if FLAGS.store_data:
      logging.info(f"Store the basis functions...")
      result_path = os.path.join(out_dir, "basis_functions.npz")
      onp.savez(result_path, x=basis_x, y=basis_y)

  # ---------------------------------------------------------------------------
  # Store aggregate results
  # ---------------------------------------------------------------------------
  if FLAGS.store_data:
    logging.info(f"Store indices, bounds, constraint diffs at final non-nan.")
    result_path = os.path.join(out_dir, "final_nonnan.npz")
    onp.savez(result_path, xstar_grid=xstar_grid, **final)

    logging.info(f"Store indices, bounds, constraint diffs at last satisfied.")
    result_path = os.path.join(out_dir, "final_satisfied.npz")
    onp.savez(result_path, xstar_grid=xstar_grid, **satis)

  # ---------------------------------------------------------------------------
  # Plot aggregate results
  # ---------------------------------------------------------------------------
  if FLAGS.plot_final:
    logging.info(f"Plot final aggregate results...")
    plotting.plot_all_final(
      final, satis, x, y, xstar_grid, data_xstar, data_ystar,
      coeff_2sls=coeff_2sls, x_kiv=x_kiv, y_kiv=y_kiv, base_dir=out_dir)

  logging.info(f"DONE")


if __name__ == "__main__":
  app.run(main)
