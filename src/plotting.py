"""Plotting functionality."""

from typing import Text, Optional, Dict

import os

from collections import Counter
import numpy as onp

import jax.numpy as np

import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt


# =============================================================================
# MATPLOTLIB STYLING SETTINGS
# =============================================================================

matplotlib.rcdefaults()

rc('text', usetex=True)
rc('font',  size='16', family='serif', serif=['Palatino'])
rc('figure', titlesize='20')  # fontsize of the figure title
rc('axes', titlesize='20')     # fontsize of the axes title
rc('axes', labelsize='18')    # fontsize of the x and y labels
rc('legend', fontsize='18')    # legend fontsize
rc('xtick', labelsize='18')    # fontsize of the tick labels
rc('ytick', labelsize='18')    # fontsize of the tick labels

rc('axes', xmargin=0)
rc('lines', linewidth=3)
rc('lines', markersize=10)
rc('grid', color='grey', linestyle='solid', linewidth=0.5)
titlekws = dict(y=1.0)

FIGSIZE = (9, 6)
data_kwargs = dict(alpha=0.5, s=5, marker='.', c='grey', label="data")


# =============================================================================
# DECORATORS & GENERAL FUNCTIONALITY
# =============================================================================

def empty_fig_on_failure(func):
  """Decorator for individual plot functions to return empty fig on failure."""
  def applicator(*args, **kwargs):
    # noinspection PyBroadException
    try:
      return func(*args, **kwargs)
    except Exception:  # pylint: disable=bare-except
      return plt.figure()
  return applicator


def save_plot(figure: plt.Figure, path: Text):
  """Store a figure in a given location on disk."""
  if path is not None:
    figure.savefig(path, bbox_inches="tight", format="pdf")
    plt.close(figure)


# =============================================================================
# FINAL AGGREGATE RESULTS
# =============================================================================

@empty_fig_on_failure
def plot_final_max_abs_diff(xstar: np.ndarray, maxabsdiff: np.ndarray):
  fig = plt.figure()
  plt.semilogy(xstar, maxabsdiff[:, 0], 'g--x', label="lower", lw=2)
  plt.semilogy(xstar, maxabsdiff[:, 1], 'r--x', label="upper", lw=2)
  plt.xlabel("x")
  plt.ylabel(f"$\max |LHS - RHS|$")
  plt.title(f"Final maximum violation of constraints")
  plt.legend()
  return fig


@empty_fig_on_failure
def plot_final_bounds(x: np.ndarray,
                      y: np.ndarray,
                      xstar: np.ndarray,
                      bounds: np.ndarray,
                      data_xstar: np.ndarray,
                      data_ystar: np.ndarray,
                      coeff_2sls: np.ndarray = None,
                      x_kiv: np.ndarray = None,
                      y_kiv: np.ndarray = None) -> plt.Figure:
  fig = plt.figure()
  plt.scatter(x, y, **data_kwargs)
  plt.plot(xstar, bounds[:, 0], 'g--x', label="lower", lw=2, markersize=10)
  plt.plot(xstar, bounds[:, 1], 'r--x', label="upper", lw=2, markersize=10)
  if data_xstar is not None and data_ystar is not None:
    if data_ystar.ndim > 1:
      data_ystar = data_ystar.mean(0)
    plt.plot(data_xstar, data_ystar, label=f"$E[Y | do(x^*)]$", lw=2)
  if coeff_2sls is not None:
    tt = np.linspace(np.min(x), np.max(x), 10)
    y_2sls = coeff_2sls[0] + coeff_2sls[1] * tt
    plt.plot(tt, y_2sls, ls='dotted', c="tab:purple", lw=2, label="2sls")
  if x_kiv is not None and y_kiv is not None:
    plt.plot(x_kiv, y_kiv, ls='dashdot', c="tab:olive", lw=2, label="KIV")

  def get_limits(vals):
    lo = np.min(vals)
    hi = np.max(vals)
    extend = (hi - lo) / 15.
    return lo - extend, hi + extend

  plt.xlim(get_limits(x))
  plt.ylim(get_limits(y))
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title("Lower and upper bound on actual effect")
  plt.legend()
  return fig


# =============================================================================
# INDIVIDUAL RUN RESULTS
# =============================================================================

@empty_fig_on_failure
def plot_lagrangian(values: np.ndarray) -> plt.Figure:
  fig = plt.figure()
  plt.plot(values)
  plt.xlabel("update steps")
  plt.ylabel("Lagrangian")
  plt.title(f"Overall Lagrangian")
  return fig


@empty_fig_on_failure
def plot_max_sq_lhs_rhs(lhs: np.ndarray, rhs: np.ndarray) -> plt.Figure:
  fig = plt.figure()
  tt = np.array([np.max((lhs - r)**2) for r in rhs])
  plt.semilogy(tt)
  plt.xlabel("optimization rounds")
  plt.ylabel("(LHS - RHS)^2")
  plt.title(f"(LHS - RHS)^2")
  return fig


@empty_fig_on_failure
def plot_max_abs_lhs_rhs(lhs: np.ndarray, rhs: np.ndarray) -> plt.Figure:
  fig = plt.figure()
  tt = np.array([np.max(np.abs(lhs - r)) for r in rhs])
  plt.semilogy(tt)
  plt.xlabel("optimization rounds")
  plt.ylabel("max(|LHS - RHS|)")
  plt.title(f"max(|LHS - RHS|)")
  return fig


@empty_fig_on_failure
def plot_max_rel_abs_lhs_rhs(lhs: np.ndarray, rhs: np.ndarray) -> plt.Figure:
  fig = plt.figure()
  tt = np.array([np.max(np.abs((lhs - r) / lhs)) for r in rhs])
  plt.semilogy(tt)
  plt.xlabel("optimization rounds")
  plt.ylabel("max(|LHS - RHS| / |LHS|)")
  plt.title(f"max(|LHS - RHS| / |LHS|)")
  return fig


@empty_fig_on_failure
def plot_abs_lhs_rhs(lhs: np.ndarray, rhs: np.ndarray) -> plt.Figure:
  fig = plt.figure()
  tt = np.array([np.abs(lhs - r) for r in rhs])
  for i in range(len(lhs)):
    plt.semilogy(tt[:, i], label=f'{i + 1}')
  plt.xlabel("optimization rounds")
  plt.ylabel("|LHS - RHS|")
  plt.title(f"individual |LHS - RHS|")
  plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
  return fig


@empty_fig_on_failure
def plot_min_max_rhs(rhs: np.ndarray) -> plt.Figure:
  fig = plt.figure()
  tt = np.array([(np.min(r), np.max(r)) for r in rhs])
  plt.plot(tt)
  plt.xlabel("optimization rounds")
  plt.ylabel("RHS min and max")
  plt.title(f"min and max of RHS")
  return fig


@empty_fig_on_failure
def plot_grad_norms(grad_norms: np.ndarray) -> plt.Figure:
  fig = plt.figure()
  grad_norms = np.array(grad_norms)
  plt.semilogy(grad_norms)
  plt.xlabel("update steps")
  plt.ylabel("norm of gradients")
  plt.legend(["L", "mu", "log_sigma"])
  plt.title(f"Gradient norms (w.r.t. $L$, $\mu$, $\log(\sigma)$)")
  return fig


@empty_fig_on_failure
def plot_mu(mus: np.ndarray) -> plt.Figure:
  fig = plt.figure()
  plt.plot(np.array(mus), '-x')
  plt.xlabel("optimization rounds")
  plt.ylabel(f"$\mu$")
  plt.title(f"Means $\mu$ of $\\theta$s")
  return fig


@empty_fig_on_failure
def plot_sigma(sigmas: np.ndarray) -> plt.Figure:
  fig = plt.figure()
  plt.plot(np.array(sigmas), '-x')
  plt.xlabel("optimization rounds")
  plt.ylabel(f"$\sigma$")
  plt.title(f"Stddevs $\sigma$ of $\\theta$s")
  return fig


@empty_fig_on_failure
def plot_mu_and_sigma(mus: np.ndarray, sigmas: np.ndarray) -> plt.Figure:
  fig = plt.figure()
  epochs = np.arange(mus.shape[0])
  mus = np.array(mus)
  sigmas = np.array(sigmas)
  for i in range(mus.shape[1]):
    mu = mus[:, i]
    sigma = sigmas[:, i]
    plt.fill_between(epochs, mu - sigma, mu + sigma, alpha=0.3)
    plt.plot(epochs, mu, '-x')
  plt.xlabel("optimization rounds")
  plt.ylabel(f"$\mu$")
  plt.title(f"Means $\mu$ of $\\theta$s")
  return fig


@empty_fig_on_failure
def plot_cholesky_factor(ls: np.ndarray) -> plt.Figure:
  fig = plt.figure()
  plt.plot(ls, '-x')
  plt.xlabel("optimization rounds")
  plt.ylabel(f"entries of $L$")
  plt.title(f"entries of the Cholesky factor $L$")
  return fig


@empty_fig_on_failure
def plot_tau(taus: np.ndarray) -> plt.Figure:
  fig = plt.figure()
  plt.plot(taus, "-x")
  plt.xlabel("optimization rounds")
  plt.ylabel(f"temperature $\\tau$")
  plt.title(f"temperature parameter $\\tau$")
  return fig


@empty_fig_on_failure
def plot_rho(rhos: np.ndarray) -> plt.Figure:
  fig = plt.figure()
  plt.plot(rhos, "-x")
  plt.xlabel("optimization rounds")
  plt.ylabel(f"annealing $\\rho$")
  plt.title(f"annealing parameter $\\rho$")
  return fig


@empty_fig_on_failure
def plot_lambda(lmbdas: np.ndarray) -> plt.Figure:
  fig = plt.figure()
  plt.semilogy(lmbdas, "-x")
  plt.xlabel("optimization rounds")
  plt.ylabel(f"multipliers $\lambda$")
  plt.title(f"Lagrange multipliers $\lambda$")
  return fig


@empty_fig_on_failure
def plot_objective(objectives: np.ndarray) -> plt.Figure:
  fig = plt.figure()
  plt.plot(objectives)
  plt.xlabel("optimization rounds")
  plt.ylabel(f"objective value")
  plt.title("Objective")
  return fig


@empty_fig_on_failure
def plot_constraint_term(constrs: np.ndarray) -> plt.Figure:
  fig = plt.figure()
  plt.semilogy(constrs)
  plt.xlabel("optimization rounds")
  plt.ylabel(f"constraint term")
  plt.title("Constraint term")
  return fig


@empty_fig_on_failure
def plot_mean_response(mus: np.ndarray,
                       x: np.ndarray,
                       y: np.ndarray,
                       response) -> plt.Figure:
  fig = plt.figure()
  plt.plot(x, y, '.', alpha=0.3, label='data')
  xx = np.linspace(np.min(x), np.max(x), 100)
  yy = []
  for x in xx:
    yy.append(response(mus[-1, :], x))
  yy = np.array(yy).squeeze()
  plt.plot(xx, yy, label='mean response')
  plt.xlabel("x")
  plt.ylabel("y")
  plt.title("Mean response")
  return fig


# =============================================================================
# DATA AND PREPROCESSING
# =============================================================================

# @empty_fig_on_failure
def plot_data(z: np.ndarray,
              x: np.ndarray,
              y: np.ndarray,
              confounder: np.ndarray,
              ex: np.ndarray,
              ey: np.ndarray) -> plt.Figure:

  def corr_label(_x, _y):
    return f'$\\rho = $ {onp.corrcoef(_x, _y)[0, 1]:.02f}'

  fig, axs = plt.subplots(3, 3, figsize=(15, 10))
  if ex is not None:
    axs[0, 0].plot(ex, x, '.', label=corr_label(ex, x))
    axs[0, 0].set_xlabel("noise ex")
    axs[0, 0].set_ylabel("treatment x")
    axs[0, 0].legend()
  axs[0, 1].plot(z, x, '.', label=corr_label(z, x))
  axs[0, 1].set_xlabel("instrument z")
  axs[0, 1].set_ylabel("treatment x")
  axs[0, 1].legend()
  if confounder is not None:
    axs[0, 2].plot(confounder, x, '.', label=corr_label(confounder, x))
    axs[0, 2].set_xlabel("confounder")
    axs[0, 2].set_ylabel("treatment x")
    axs[0, 2].legend()

  if ey is not None:
    axs[1, 0].plot(ey, y, '.', label=corr_label(ey, y))
    axs[1, 0].set_xlabel("noise ey")
    axs[1, 0].set_ylabel("outcome y")
    axs[1, 0].legend()
  axs[1, 1].plot(x, y, '.', label=corr_label(x, y))
  axs[1, 1].set_xlabel("treatment x")
  axs[1, 1].set_ylabel("outcome y")
  axs[1, 1].legend()
  if confounder is not None:
    axs[1, 2].plot(confounder, y, '.', label=corr_label(confounder, y))
    axs[1, 2].set_xlabel("confounder")
    axs[1, 2].set_ylabel("outcome y")
    axs[1, 2].legend()

  if ey is not None and ex is not None:
    axs[2, 0].plot(ex, ey, '.', label=corr_label(ex, ey))
    axs[2, 0].set_xlabel("noise ex")
    axs[2, 0].set_ylabel("noise ey")
    axs[2, 0].legend()
  axs[2, 1].plot(z, y, '.', label=corr_label(z, y))
  axs[2, 1].set_xlabel("instrument z")
  axs[2, 1].set_ylabel("outcome y")
  axs[2, 1].legend()

  return fig


@empty_fig_on_failure
def plot_bin_hist(bin_ids: np.ndarray) -> plt.Figure:
  fig = plt.figure()
  tt = np.array(list(Counter(bin_ids).items()))
  plt.bar(tt[:, 0], tt[:, 1])
  plt.xlabel("bins")
  plt.ylabel("Number of data points")
  plt.title("Distribution of datapoints into z-bins")
  return fig


@empty_fig_on_failure
def plot_bin_assignment(z: np.ndarray,
                        val: np.ndarray,
                        z_grid: np.ndarray,
                        bin_ids: np.ndarray,
                        ylabel: Text) -> plt.Figure:
  fig = plt.figure()
  num_z = len(z_grid)
  for i in range(num_z):
    plt.plot(z[bin_ids == i], val[bin_ids == i], '.')
  for zi in z_grid:
    plt.axvline(zi, c='k', lw=0.5)
  plt.xlabel('z')
  plt.ylabel(ylabel)
  plt.title("Bin assignment and z-grid lines")
  return fig


@empty_fig_on_failure
def plot_hist_at_z(y: np.ndarray, bin_ids: np.ndarray, idx: int) -> plt.Figure:
  fig = plt.figure()
  plt.hist(y[bin_ids == idx], bins=30)
  plt.xlabel('y')
  mean = np.mean(y[bin_ids == idx])
  var = np.var(y[bin_ids == idx])
  plt.title(
    f"mean {mean:.2f} and variance {var:.2f} for z bin {idx}")
  return fig


@empty_fig_on_failure
def plot_y_with_constraints(z: np.ndarray,
                            y: np.ndarray,
                            z_grid: np.ndarray,
                            lhs: np.ndarray) -> plt.Figure:
  fig = plt.figure()
  lo = lhs[:, 0] - lhs[:, 1]
  hi = lhs[:, 0] + lhs[:, 1]
  plt.fill_between(z_grid, lo, hi, alpha=0.5, color='r')
  plt.plot(z, y, '.', alpha=0.3)
  plt.plot(z_grid, lhs[:, 0])
  plt.xlabel('z')
  plt.ylabel('y')
  plt.title("Datapoints with mean and variance from LHS constraints")
  return fig


@empty_fig_on_failure
def plot_inverse_cdfs(x_cdf_invs) -> plt.Figure:
  fig = plt.figure()
  t = np.linspace(0, 1, 50)
  for i, invcdf in enumerate(x_cdf_invs):
      plt.plot(t, invcdf(t), label=f"i: {i}")
  plt.ylabel("x")
  plt.xlabel("CDF")
  plt.title("Inverse CDFs of x for different z in grid")
  return fig


@empty_fig_on_failure
def plot_xhats_distr(x: np.ndarray, xhats: np.ndarray) -> plt.Figure:
  fig = plt.figure()
  plt.hist(xhats.ravel(), bins=50, density=True, alpha=0.3, label="sampled x")
  plt.hist(x, bins=50, density=True, alpha=0.3, label="actual x (data)")
  plt.legend()
  plt.xlabel("x")
  plt.ylabel("density")
  plt.title("Distribution of pre-sampled and actual x")
  return fig


@empty_fig_on_failure
def plot_discrepancy_x(z: np.ndarray,
                       x: np.ndarray,
                       xhats: np.ndarray,
                       z_grid: np.ndarray) -> plt.Figure:
  # Check where the discrepancy between real x and sampled x comes from
  fig = plt.figure()
  middle = np.mean(xhats, axis=-1)
  delta = np.std(xhats, axis=-1)
  plt.plot(z, x, '.', alpha=0.2, label='data')
  plt.fill_between(z_grid, middle - delta, middle + delta,
                   alpha=0.5, color='r', label='var samples')
  plt.plot(z_grid, middle, label='mean samples')
  plt.xlabel('z')
  plt.ylabel('x')
  plt.legend()
  return fig


# @empty_fig_on_failure
def plot_basis_samples(basis_func, x: np.ndarray, y: np.ndarray) -> plt.Figure:
  fig = plt.figure()
  xx = np.linspace(np.min(x), np.max(x), 200)
  ys = basis_func(xx).squeeze()
  plt.plot(x, y, '.', alpha=0.2, label='data')
  plt.plot(xx, ys, label='basis funcs')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.legend()
  return fig


# =============================================================================
# PLOT ALL
# =============================================================================

def plot_all_init(z: np.ndarray,
                  x: np.ndarray,
                  y: np.ndarray,
                  confounder: np.ndarray,
                  ex: np.ndarray,
                  ey: np.ndarray,
                  xhats: np.ndarray,
                  z_grid: np.ndarray,
                  bin_ids: np.ndarray,
                  lhs: np.ndarray,
                  basis_func=None,
                  base_dir: Optional[Text] = None):
  """Call all relevant plotting functions for initialization and data."""
  if base_dir is not None and not os.path.exists(base_dir):
    os.makedirs(base_dir)

  num_z = len(z_grid)

  path = os.path.join(base_dir, f"data.pdf")
  save_plot(plot_data(z, x, y, confounder, ex, ey), path)

  path = os.path.join(base_dir, f"bin_histogram.pdf")
  save_plot(plot_bin_hist(bin_ids), path)

  path = os.path.join(base_dir, f"bin_assignment_x.pdf")
  save_plot(plot_bin_assignment(z, x, z_grid, bin_ids, 'x'), path)

  path = os.path.join(base_dir, f"bin_assignment_y.pdf")
  save_plot(plot_bin_assignment(z, y, z_grid, bin_ids, 'y'), path)

  path = os.path.join(base_dir, f"y_hist_last_z.pdf")
  save_plot(plot_hist_at_z(y, bin_ids, num_z - 1), path)

  path = os.path.join(base_dir, f"y_with_constraints.pdf")
  save_plot(plot_y_with_constraints(z, y, z_grid, lhs), path)

  path = os.path.join(base_dir, f"xhat_distribution.pdf")
  save_plot(plot_xhats_distr(x, xhats), path)

  path = os.path.join(base_dir, f"discrepancy_x.pdf")
  save_plot(plot_discrepancy_x(z, x, xhats, z_grid), path)

  # path = os.path.join(base_dir, f"inverse_cdfs.pdf")
  # save_plot(plot_inverse_cdfs(x_cdf_invs), path)

  if basis_func is not None:
    path = os.path.join(base_dir, f"basis_functions.pdf")
    save_plot(plot_basis_samples(basis_func, x, y), path)


def plot_all_final(final: Dict[Text, np.ndarray],
                   satisfied: Dict[Text, np.ndarray],
                   x: np.ndarray,
                   y: np.ndarray,
                   xstar_grid: np.ndarray,
                   data_xstar: np.ndarray,
                   data_ystar: np.ndarray,
                   coeff_2sls: np.ndarray = None,
                   x_kiv: np.ndarray = None,
                   y_kiv: np.ndarray = None,
                   base_dir: Optional[Text] = None):
  """Call all relevant plotting functions for final aggregate results."""
  if base_dir is not None and not os.path.exists(base_dir):
    os.makedirs(base_dir)

  # To also show the last valid (non-nan) bound regardless of whether they
  # satisfied the constraints, use mode="non-nan", results=satisfied instead.
  mode = "satisfied"
  results = satisfied
  result_path = os.path.join(base_dir, f"final_{mode}_bounds.pdf")
  save_plot(plot_final_bounds(x, y, xstar_grid, results["objective"],
                              data_xstar, data_ystar, coeff_2sls,
                              x_kiv, y_kiv),
            result_path)

  # Uncomment to show maximum absolute violation of exact constraints
  # result_path = os.path.join(base_dir, f"final_{mode}_maxabsdiff.pdf")
  # save_plot(plot_final_max_abs_diff(xstar_grid, results["maxabsdiff"]),
  #           result_path)


def plot_all(results,
             x: np.ndarray,
             y: np.ndarray,
             response,
             base_dir: Optional[Text] = None, suffix: Text = ""):
  """Call all relevant plotting functions.

  Args:
      results: The results dictionary.
      x: The x values of the original data.
      y: The y values of the original data.
      response: The response function.
      base_dir: The path where to store the figures. If `None` don't save the
            figures to disk.
      suffix: An optional suffix to each filename stored by this function.
  """
  if base_dir is not None and not os.path.exists(base_dir):
    os.makedirs(base_dir)

  def get_filename(base: Text, fname: Text):
    return None if base is None else os.path.join(base, fname)

  suff = "_" + suffix if suffix else suffix

  name = "lagrangian{}.pdf".format(suff)
  save_plot(plot_lagrangian(results["lagrangian"]),
            get_filename(base_dir, name))

  name = "grad_norms{}.pdf".format(suff)
  save_plot(plot_grad_norms(results["grad_norms"]),
            get_filename(base_dir, name))

  name = "mu{}.pdf".format(suff)
  save_plot(plot_mu(results["mu"]),
            get_filename(base_dir, name))

  name = "sigma{}.pdf".format(suff)
  save_plot(plot_sigma(results["sigma"]),
            get_filename(base_dir, name))

  name = "mu_and_sigma{}.pdf".format(suff)
  save_plot(plot_mu_and_sigma(results["mu"], results["sigma"]),
            get_filename(base_dir, name))

  name = "cholesky_factor{}.pdf".format(suff)
  save_plot(plot_cholesky_factor(results["cholesky_factor"]),
            get_filename(base_dir, name))

  name = "tau{}.pdf".format(suff)
  save_plot(plot_tau(results["tau"]),
            get_filename(base_dir, name))

  name = "rho{}.pdf".format(suff)
  save_plot(plot_rho(results["rho"]),
            get_filename(base_dir, name))

  name = "lambda{}.pdf".format(suff)
  save_plot(plot_lambda(results["lambda"]),
            get_filename(base_dir, name))

  name = "objective{}.pdf".format(suff)
  save_plot(plot_objective(results["objective"]),
            get_filename(base_dir, name))

  name = "constraint_term{}.pdf".format(suff)
  save_plot(plot_constraint_term(results["constraint_term"]),
            get_filename(base_dir, name))

  name = "mean_response{}.pdf".format(suff)
  save_plot(plot_mean_response(results["mu"], x, y, response),
            get_filename(base_dir, name))

  name = "max_abs_lhs_rhs{}.pdf".format(suff)
  save_plot(plot_max_abs_lhs_rhs(results["lhs"], results["rhs"]),
            get_filename(base_dir, name))

  name = "max_rel_abs_lhs_rhs{}.pdf".format(suff)
  save_plot(plot_max_rel_abs_lhs_rhs(results["lhs"], results["rhs"]),
            get_filename(base_dir, name))

  name = "abs_lhs_rhs{}.pdf".format(suff)
  save_plot(plot_abs_lhs_rhs(results["lhs"], results["rhs"]),
            get_filename(base_dir, name))
