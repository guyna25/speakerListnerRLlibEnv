# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
pyMCCM.py

Updated and Enhanced version of OpenAI Multi-Agent Deep Deterministic Policy Gradient (MADDPG) Algorithm
(https://github.com/openai/maddpg)
"""

import rpy2.robjects.packages as rpack
import rpy2.robjects as ro
import rpy2.robjects.vectors as rvec
import skccm
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2020, Multi-Agent Deep Deterministic Policy Gradient'
__credits__ = ['Rolando Fernandez', 'OpenAI']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rolando.fernandez1.civ@mail.mil'
__status__ = 'Dev'

# Basic R utils
r_base = rpack.importr("base")
r_utils = rpack.importr("utils")
r_stats = rpack.importr("stats")

# Make sure that multispatialCCM is installed
r_utils.chooseCRANmirror(ind=1)  # select the first mirror in the list
r_packnames = ('multispatialCCM', 'Rmisc')  # Essential packages
to_install = [x for x in r_packnames if not rpack.isinstalled(x)]

if len(to_install) > 0:
    print("Installing Packages: {}".format(to_install))
    r_utils.install_packages(rvec.StrVector(to_install))

mccm = rpack.importr("multispatialCCM")


def make_test_data(rx_a=3.72, rx_b=3.72, b_ab=0.2, b_ba=0.01, t=20, obs=10, seed=12345):
    """
    Creates a test data-set based on a coupled logistic map.

    This function generates a data-set suitable for testing pyMCCM. The data is
    generated from a coupled logistic map, which is known to be an accurate
    model for population dynamics between one predatory and one prey species
    that coexist within an ecosystem. The coupled logistic map is characterized
    by a shape parameter 'r' and a coupling strength 'b'.

    Args:
        rx_a (float): Shape parameter for vector a_vec [0:4)
        rx_b (float): Shape parameter for vector b_vec [0:4)
        b_ab (float): Coupling strength for a_vec->b_vec
        b_ba (float): Coupling strength for b_vec->a_vec
        t (int): Number time steps per observation
        obs (int): Number of observations (i.e., initial conditions) to generate
        seed (int): Random seed to use (for recreating specific trajectories)

    Returns:
        (np.array) Test data-set
    """
    np.random.seed(seed)

    # Generator function for coupled-logistic dynamics
    def coupled_logistic(t, xa_0, xb_0, rx_a, rx_b, b_ab, b_ba):
        tt = 0
        while tt < t:
            yield (xa_0, xb_0)
            xa_0 = xa_0 * (rx_a - rx_a * xa_0 - b_ba * xb_0)
            xb_0 = xb_0 * (rx_b - rx_b * xb_0 - b_ab * xa_0)
            tt += 1

    result = np.empty((0, t, 2))

    # Populate a data-set for a random set of initial conditions
    for ob in range(obs):
        xa_0 = np.random.random() / 2
        xb_0 = np.random.random() / 2
        temp = np.array([[xA, xB] for xA, xB in coupled_logistic(t, xa_0, xb_0, rx_a, rx_b, b_ab, b_ba)])
        result = np.append(result, [temp], axis=0)

    # Return data-set
    return result


def _to_mccm_vec(x):
    """
    Compose vector into format for multispatialCCM.R

    Args:
        x (np.array): Data vector

    Returns:
        (rvec.FloatVector) Data in R format
    """
    mccm_x = np.array([[np.nan] + list(x) for x in list(x)])
    mccm_x = mccm_x.ravel()

    return rvec.FloatVector(mccm_x)


def _get_time_delay(x):
    """
    Estimate accurate time-delay for attractor reconstruction.

    This function attempts to estimate the necessary time-delay for
    reconstructing the attractor dynamics of a system characterized by an
    observed time-series. Recreating attractor dynamics in this way is based on
    Taken's Embedding Theorem.

    Note: This method functions problematically for systems with an embedding
    dimension less than 3, and an expected time-delay of 1 (e.g., coupled
    logistic map), as such systems are not truely chaotic. Therefore, this method
    should be used with caution especially if there is a reason to suspect
    that the system in question may not be high dimensional or chaotic.

    Args:
        x (np.array): The vector to be evaluated

    Returns:
        (float) Estimated time-delay
    """

    def find_minima(x, num_minima=None, ignore_first=True):
        """
        Find the first n local minima of a vector.

        Args:
            x (np.array): Vector to be searched
            num_mimima (int): Number of local minima to return
            ignore_first (boolean): Drop the first local minimum found

        Returns:
            The local minima of a vector
        """

        if not isinstance(x, type(np.array([]))):
            x = np.array(x)

        if len(x.shape) > 1.0 and x.shape[0] < x.shape[1]:
            x.reshape((x.shape[1], x.shape[0]))

        zero_crosses = np.zeros(x.shape)
        for i, val in enumerate(np.diff(x)):
            if np.sign(val) != np.sign(x[i - 1]):
                zero_crosses[i] = 1
        minima = np.zeros(zero_crosses.shape)

        for i, val in enumerate(zero_crosses.tolist()):
            if val != zero_crosses[i - 1] and val == 0.0:
                minima[i] = 1.0

        if num_minima == None:
            return np.where(minima == 1.0)[0].tolist()

        if num_minima >= len(np.where(minima == 1.0)[0].tolist()):
            return np.where(minima == 1.0)[0].tolist()

        if num_minima > 0:
            return np.where(minima == 1.0)[0].tolist()[num_minima - 1]

        return None

    # Strip nan's from vector to be tested
    x = x[np.logical_not(np.isnan(x))]

    l = x.shape[0]

    # Create an Embed object from skccm
    e = skccm.Embed(x)

    # Evaulate the loss of mutual information for time-lags 0:l-1
    mi = e.mutual_information(int(l - 1))

    # Identify local minima to find the first minimum of lagged mutual information
    lag = find_minima(mi, 1)

    # Return estimated time-delay
    if type(lag) == list:
        lag = lag[0]

    return lag


def _get_embedding_dim(x, e_max, tau, predstep):
    """
    Estimate the embedding dimension of the hypothesized attractor.

    Args:
        x (rvec.FloatVector): Vector to be embedded
        e_max (int): Maximum embedding dimension to test (e_max < len(x))
        tau (float): Time-delay for attractor reconstruction
        predstep (int): Prediction step

    Returns:
        (int) The embedding dimension that yields the best predictions
    """
    if e_max <= 2:
        # Embedding dimension must be at least 2
        return (np.array([e_max]))

    # Iteratively test embedding dimension to find the one with the best characterization of the data
    embedding_dims = np.empty((0, 2))
    for embedding_dim in range(2, e_max + 1):
        # Boot-strap embedding dimension from multispatialCCM.R
        dim = [embedding_dim, mccm.SSR_pred_boot(A=x, E=embedding_dim, predstep=predstep, tau=tau).rx2("rho")[0]]

        # Keep a record of tried dimensions
        embedding_dims = np.append(embedding_dims, [dim], axis=0)

    # Return the embedding dimension that yields the best predictions
    embedding_dims = embedding_dims[embedding_dims[:, 1] == np.max(embedding_dims[:, 1]), 0]

    return embedding_dims


def _check_auto_predictability(x, embedding_dim, tau, predsteplist=None):
    """
    Check data validity for MCCM via auto-predictability

    This function evaluates whether or not the data used is valid to be tested
    by MCCM. The primary concerns are non-linearity and local periodicity.
    Issues warnings if these features are detected, but also returns the
    validity test results for analysis offline.

    Args:
        x (rvec.FloatVector): Vector to be tested
        embedding_dim (int): Embedding dimension
        tau (float): Time-delay for attractor reconstruction
        predsteplist (list): List of temporal distances for evaluating prediction

    Returns:
        (dict) Results of validity testing
    """
    # Set default prediction step list if not provided
    if predsteplist is None:
        predsteplist = list(range(1, 11))

    # Auto-predictability test from multispatialCCM.R
    signal_out_r = mccm.SSR_check_signal(A=x, E=embedding_dim, tau=tau, predsteplist=rvec.IntVector(predsteplist))

    # Prediction strength (rho) over increasing temporal distance (predstep)
    rho_pred = np.array(signal_out_r.rx2("predatout"))

    # Slope and p-value (linear regression) of rho over temporal distance
    rho_slope = np.array(signal_out_r.rx2("rho_pre_slope"))

    if rho_slope[0] >= 0:
        # If prediction strength (rho) remains the same or increases with temporal distance,
        # a basic tenant of non-linearity has been violated
        warnings.warn("Prediction increases with historical distance. Data may not be non-linear...")

    if np.max(rho_pred) < 0.2:
        # If the highest prediction strength is fairly low, there may be too much noise
        # in the data for MCCM to provide a valid result
        warnings.warn(
            "Corrlation coefficient for short time steps (predictive validity) is below 0.2. "
            "Excessive stochasitic noise may be present...")

    if np.min(rho_pred) < (rho_pred[1, 0] - 0.2) and np.min(rho_pred) < (rho_pred[1, -1] - 0.2):
        # If prediction strength (rho) dips and then rises again, the data is likely locally periodic.
        # This is highly problematic for accurate MCCM
        warnings.warn("Possible periodicity detected...")

    # Return results of validity testing
    signal_out = {"rho predicted": rho_pred,
                  "rho test": rho_slope}

    return signal_out


def get_score(a_vec, b_vec, e_max=3, estimate_dim=True, tau=None, iterations=10, predstep=10, full_out=False, show_plot=True,
              clock=False):
    """
    Get MCCM score for time-series A and B

    This function operates in either a 'full' or 'short' mode to return the
    measured causal influence of two time-series on one another. In the short
    mode, it only returns basic measures for the influence of A on B. In the
    'full' mode, it returns the full MCCM models for each time-series, as well
    as validity tests, and the opportunity for a visualization of the MCCM
    causal curves.

    Args:
        a_vec (np.array): he primary vector being analyzed
        b_vec (np.array): the secondary vector with which 'A' may have a causal link
        e_max (int): The maximum embedding dimension to test
        estimate_dim (boolean): Whether or not to estimate the embedding dimension
        tau (float): Time-lag to use for attractor reconstruction
        iterations (int): Number of boot-strap iterations to run
        predstep (int): How far ahead to look when testing auto-predictability
        full_out (boolean): Whether or not to use full mode or short mode
        show_plot (boolean): Whether or not to display MCCM curves (only in full mode)
        clock (boolean): Whether or not to print out the process time

    Returns:
        (dict) A dictionary containing the MCCM models, their validity,
               the statistical evaluation of causal relationship, and plot.
    """
    t_start = time.time()

    # Cast numpy vectors to r-vector form needed for multispaitalCCM.R
    a_ccm = _to_mccm_vec(a_vec)
    b_ccm = _to_mccm_vec(b_vec)

    # If no time-delay is provided, estimate it
    tau_time = time.time()
    if tau == None:
        tau1 = _get_time_delay(a_vec[0])
        tau2 = _get_time_delay(b_vec[0])
        if tau1 != tau2:
            tau = min(tau1, tau2)
        else:
            tau = tau1
    tau_time = time.time() - tau_time

    # Estimate embedding dimension if indicated
    e_time = time.time()
    if estimate_dim:
        e_a = _get_embedding_dim(a_ccm, e_max, tau, predstep)
        e_b = _get_embedding_dim(b_ccm, e_max, tau, predstep)
    else:
        # Otherwise, take e_max to be the embedding dimension
        e_a, e_b = e_max, e_max
    e_time = time.time() - e_time

    # Compute short return values (i.e., does b_vec cause a_vec)
    m_time = time.time()
    b_causes_a = mccm.CCM_boot(b_ccm, a_ccm, e_b[0], tau=tau, iterations=iterations)  # MCCM model b_vec->a_vec
    m_time = time.time() - m_time

    # Python dictionary of returned data.frame from multispatialCCM.R
    mccm_ba = {key: np.array(b_causes_a.rx2(key)) for key in b_causes_a.names}

    # Statistical evaluation of causal relationship (i.e., is it more than just correlation?)
    p_ba = mccm.ccmtest(b_causes_a, b_causes_a)[0]

    if not full_out:
        # Return short version, indicating only if a causal relationship exists (p < 0.05) and the final strength (rho).
        out = np.array([mccm_ba["rho"][-1], p_ba])
        if clock:
            print("Time Taken: {}s".format(np.round(time.time() - t_start, 3)))

        return out
    else:
        # Evaluate data validity for MCCM via auto-predictability
        v_time = time.time()
        a_valid = _check_auto_predictability(a_ccm, e_a[0], tau, list(range(1, predstep + 1)))
        b_valid = _check_auto_predictability(b_ccm, e_b[0], tau, list(range(1, predstep + 1)))
        v_time = time.time() - v_time

        # If full mode is used, compute the additional model of a_vec->b_vec (i.e., does a_vec cause b_vec)
        # and return a dictionary of both models and validity tests

        #  MCCM model a_vec->b_vec
        a_causes_b = mccm.CCM_boot(a_ccm, b_ccm, e_a[0], tau=tau, iterations=iterations)

        # Python dictionary of returned data.frame from multispatialCCM.R
        mccm_ab = {key: np.array(a_causes_b.rx2(key)) for key in a_causes_b.names}

        # Statistical evaluation of causal relationship
        p_ab = mccm.ccmtest(a_causes_b, a_causes_b)[0]

        # Python dictionary output
        out = {"AB Model": mccm_ab,
               "BA Model": mccm_ba,
               "a_vec Validity": a_valid,
               "b_vec Validity": b_valid,
               "p test": np.array([p_ab, p_ba])}

        if show_plot:
            # Plotting
            fig = plt.subplot(111)
            fig.set_ylim(-0.1, 1.0)
            fig.fill_between(mccm_ab["Lobs"], mccm_ab["rho"] - mccm_ab["sdevrho"], mccm_ab["rho"] + mccm_ab["sdevrho"],
                             alpha=0.2)
            fig.plot(mccm_ab["Lobs"], mccm_ab["rho"], label="a_vec causes b_vec, p={}".format(np.round(p_ab, 4)))

            fig.fill_between(mccm_ba["Lobs"], mccm_ba["rho"] - mccm_ba["sdevrho"], mccm_ba["rho"] + mccm_ba["sdevrho"],
                             alpha=0.2)
            fig.plot(mccm_ba["Lobs"], mccm_ba["rho"], label="b_vec causes a_vec, p={}".format(np.round(p_ba, 4)))
            fig.legend()

            out["Plot"] = fig

        if clock:
            print("Tau Time: {}s".format(np.round(tau_time, 3)))
            print("Embed Time: {}s".format(np.round(e_time, 3)))
            print("Validate Time: {}s".format(np.round(v_time, 3)))
            print("Model Time: {}s".format(np.round(m_time, 3)))
            print("Total Time Taken: {}s".format(np.round(time.time() - t_start, 3)))

        return out


if __name__ == "__main__":
    # Test pyMCCM
    data = make_test_data(obs=10, t=25, b_ab=0.05, b_ba=0.2)
    a = data[:, :, 0]
    b = data[:, :, 1]
    print(get_score(a, b, full_out=False, show_plot=True))
