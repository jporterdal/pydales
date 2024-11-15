import numpy as np
from math import gamma
import itertools as it
import problems, constraints_handler


class IterationState():
    """Convenience class for quickly encapsulating a set of related values within a single iteration of an ES algorithm.

    Ex:
        history = list()
        for i in range(100):
            x = i + 1
            y = i * 2
            z = i**3
            history.append(IterationState(x=x, y=y, z=z))

        s = history[9]
        print(s.x + s.y + s.z)
    """

    def __init__(self, **kwargs):
        for kw in kwargs.keys():
            setattr(self, kw, kwargs[kw])


def matlab_vec(x):
    """Convenience function for copy/pasting NP 2-D vectors into Matlab.
    """

    rows = list()
    for r in range(x.shape[0]):
        rows.append("[" + ", ".join([str(xr) for xr in x[r,:]]) + "]")

    print("[", end="")
    print("; ".join(rows), end="")
    print("]")


def dal_es(prob, x, sigma, max_steps, f_target, delta):
    """Straightforward implementation of dAL-ES from "Direct Augmented Lagrangian Evolution Strategies" by
    Porter & Arnold, GECCO'24, and associated Matlab code.

    By default, manages and updates its own evolution strategy internally. Setting 'use_simple_es' to True will import
    and use the local constraints_handler.SimpleES object-oriented implementation of an ES.

    The flow within each iteration is summarized by:
      generate offspring y
      evaluate against f(y) and g(y)/h(y)
          collect in fy and ghy, respectively
      calculate internals: M, q, R
      update_W
      update Lagrange/penalty values and determine Lagrangian: alpha, pi_inv, ell1, ell2
      update stepsize sigma
      evaluate L(y), depending on state of W

    :param prob: optimization problem represented by instance defined using problems.Problem class
    :param x: initial vector (n-dimensional) in search space
    :param sigma: initial step-size (scalar) for ES
    :param max_steps: maximum number of iterations
    :param f_target: algorithm will stop when f(y) <= f_target for a feasible offspring y
    :param delta: constraint tolerance such that g(y) <= delta will be considered feasible
    :return (rec, flag, history): within this tuple, 'rec' gives (f_evals, f_best) for each iteration and 'flag'
      indicates if f_best <= f_target (both are identical to those returned by Matlab version); 'history' is a list of
      IterationState instances giving a fuller recording of the algorithm's run.
    """

    rng = np.random.default_rng()
    history = list()

    def update_history():
        history.append(
            IterationState(
                t=t,
                ghy=ghy,
                x=x,
                sigma=sigma,
                zavg=zavg,
                alpha=alpha,
                fy=fy,
                y=y,
                W=W,
                f_best=f_best,
                x_best=x_best,
                evals=f_evals,
                max_vio=max_vio(ghy),
                stop=(t >= max_steps, f_best <= f_target),
            )
        )

    def max_vio(ghy):
        # TODO: Could streamline instead of considering separately
        vio = np.zeros((1, ghy.shape[1]))
        if l > 0:  # inequality constraints
            ##vio = np.max((vio, np.max(ghy[:l, :])))
            vio = np.max(np.vstack((vio, np.max(ghy[:l, :], axis=0))), axis=0)
        if m > 0:  # equality constraints
            ##vio = np.max((vio, np.max(np.abs(ghy[l:, :]))))
            vio = np.max(np.vstack((vio, np.max(ghy[l:, :], axis=0))), axis=0)
        return vio

    # Uses 'delta' parameter
    def update_best(fy, ghy):
        """Internally tracks the best individual (x_best) and associated objective function score (f_best). Only
        considers individuals that are feasible with tolerance 'delta'.
        """

        nonlocal f_best, x_best

        not_feasible = np.logical_not(max_vio(ghy) <= delta)

        fy_cmp = np.copy(fy)
        fy_cmp[0][not_feasible] = np.inf
        if np.any(fy_cmp < f_best):
            kk = np.argmin(fy_cmp)
            f_best = fy_cmp[0][kk]
            x_best = y[:, kk]


    def var_coeff(xi):
        return np.std(xi, axis=1) / np.mean(xi, axis=1)

    def update_W(W):
        """Updates the working set W.
        """

        nu = var_coeff(ghy)

        # Generate column vector with 1 where  0 <= nu <= theta  in first (L) entries or
        #  abs(nu) < theta  for remainder (M) entries
        severe = np.append(
            np.logical_and(
                0 <= nu[:l],
                nu[:l] < theta
            ),
            np.abs(nu[l:]) < theta,
        )
        # Recall:  L entries are inequalities, M entries are equalities


        # generate column vector which is 1 if (already in W or (inequality and nu > 0) or (equality)) AND not
        #  already in severe
        potential_less_severe = np.logical_or(W,
                                      np.append(nu[:l] > 0,
                                                np.ones((m,1)).astype('bool')
                                                ),
                                      )
        less_severe = np.logical_and(potential_less_severe, np.logical_not(severe))
        # Result here is that equalities are always included as at least a less_severe entry

        # Select a maximal subset of severely violated constraints
        W0 = np.zeros(l+m).astype('bool')  # initialize W0, flat to match shape of W
        found_one = False
        kk = min(n, np.sum(severe))

        while kk > 0 and not found_one:
            #  choose 'kk' of the severe indices; after transpose, this will be a matrix of kk rows, each column being
            #   a distinct combination of 'kk' of the indices
            ww = np.array(list(
                it.combinations(np.flatnonzero(severe), kk)
            )).T

            num_ww_cols = ww.shape[1]
            ww = ww[:, np.random.permutation(range(num_ww_cols))]  # Randomize the columns of ww
            for col_ww in range(num_ww_cols):
                W0 = np.zeros(l+m).astype('bool')  # re-initialize W0
                W0[ww[:, col_ww]] = True  # set entries to 1 using choice of 'kk' 'severe' indices, taken from ww

                # check condition number of smoothed approximation to covar(g,g); if not too large,
                #  found_one = true and we break out of both loops
                if np.linalg.cond(R[np.ix_(W0, W0)]) < 1.0e+16:
                    found_one = True
                    break

            # End result is we EXHAUSTIVELY check all possible combinations of severely violated constraints and accept
            #  the first one that allows the condition variable to be "not too large".

            kk = kk - 1

        # Add as many less severely violated constraints as possible
        W1 = np.copy(W0)
        found_one = False
        kk = min(n-np.sum(W0), np.sum(less_severe))

        while kk > 0 and not found_one:
            #  choose 'kk' of the less_severe indices; after transpose, each column will be a distinct combination
            #   of the indices
            ww = np.array(list(
                it.combinations(np.flatnonzero(less_severe), kk)
            )).T
            num_ww_cols = ww.shape[1]
            ww = ww[:, np.random.permutation(range(num_ww_cols))]  # Randomize the columns of ww
            for col_ww in range(num_ww_cols):
                W1 = np.copy(W0)
                W1[ww[:, col_ww]] = True

                if np.linalg.cond(R[np.ix_(W1, W1)]) > 1.0e+16:  # Condition number too large
                    continue

                a = np.zeros((l+m, 1))  # Matlab typo here fixed, from (l+1) to (l+m)
                a[W1] = np.linalg.solve(-R[np.ix_(W1, W1)], q[W1])  # Lagrange multipliers for constraints in W1
                a[W0] = 0  # Zero multipliers for constraints also in W0 (severe)
                if np.min(a[:l]) >= 0:
                    found_one = True
                    break

            kk = kk - 1

        # TODO: what do we want to do here if found_one == False yet kk <= 0 ? This means we found at least one set of
        #  constraints for which the condition number was not 'too large', yet for all of these the associated
        #  Lagrange multipliers had at least one negative value. Is this possible?
        return W1


    # -----------------------------------------------------------------------------------------------------------------
    #  Begin main section of code
    # -----------------------------------------------------------------------------------------------------------------

    n, l, m, f, g, h = prob.unpack()

    # Set the ES parameters
    lamb = 4 + int(np.floor(3 * np.log(n)))
    w = np.array([np.log((lamb + 1)/2) - np.log(np.arange(1, lamb+1))])
    w[w < 0] = 0
    w = w / np.sum(w)
    mu_eff = 1 / np.sum(w**2)
    c = (mu_eff + 2) / (n + mu_eff + 5)
    d = 1 + 2 * max(0, np.sqrt((mu_eff - 1)/(n + 1)) - 1) + c
    e = np.sqrt(2) * gamma((n+1)/2) / gamma(n/2)
    theta = 0.25

    # Initialize
    y = np.zeros((n, lamb))
    zavg = np.copy(y)
    fy = np.zeros((1, lamb))
    ghy = np.ones((l+m, lamb))
    q = np.zeros((l+m, 1))
    R = np.eye((l+m))
    s = np.zeros((n, 1))
    W = np.zeros(l+m).astype(bool)  # NB: 'flat' array since we use W for indexing, not matrix arithmetic

    rec = list()
    f_evals = 0
    x_best = np.nan  # set from offspring vectors (y)
    f_best = np.inf
    t = 0

    es = constraints_handler.SimpleES(x0=x.flatten(), sigma0=sigma )
    use_simple_es = False


    # ----- Main loop
    while t < max_steps and f_best > f_target:
        t = t + 1

        # Generate and evaluate offspring
        if use_simple_es:
            y = es.ask()
            for k in range(0, lamb):
                fy[0][k] = f(y[k])
                ghy[:, [k]] = np.vstack((g(y[k]), h(y[k])))
            y = np.vstack(y).T
        else:
            z = rng.standard_normal((n, lamb))
            for k in range(0, lamb):
                y[:,[k]] = x + sigma * z[:,[k]]
                try:
                    fy[0][k] = f(y[:, [k]])
                except DeprecationWarning:  # debug
                    print(k)
                    print(y[:, [k]])
                    print(fy)
                    raise

                ghy[:,[k]] = np.vstack( (g(y[:,[k]]), h(y[:,[k]])) )  # Xi

        f_evals = f_evals + lamb
        update_best(fy, ghy)
        rec.append(np.array([f_evals, f_best]))

        # Accumulate derivative related information
        M = np.cov(np.vstack((fy, ghy)).T, rowvar=False)  # Imitating Matlab
        # since in Matlab:  "columns represent random variables and whose rows represent observations"
        q = (1-c) * q + c * M[1:,[0]] / sigma**2
        R = (1-c) * R + c * M[1:, 1:] / sigma**2

        # Potentially update the working set
        W = update_W(W)

        # Determine Lagrange and penalty values
        alpha = np.linalg.solve(-R[np.ix_(W, W)], q[W])
        pi_inv = R[np.ix_(W, W)]
        Xi_W = ghy[W, :]  # Xi(W)
        ell1 = fy + alpha.T @ Xi_W
        ell2 = np.array([np.diag(Xi_W.T @ np.linalg.solve(pi_inv, Xi_W)).T])  # take diagonal as a row vector

        if np.sum(np.abs(var_coeff(ghy)) > theta) > n + 1 and var_coeff(ell2) > theta:
            sigma = sigma / 2
            update_history()
            if use_simple_es:
                es.sigma = sigma / 2
            continue

        # Determine Lagrangian that will be used to evaluate
        if np.sum(W) == 0:
            ly = fy
        elif np.sum(W) == n or var_coeff(ell2) < theta:
            here_count = here_count + 1  # debug
            ly = ell2
        else:
            CC = np.cov(ell1, ell2)
            eta = np.sqrt(2 * CC[0, 0] / CC[1, 1])
            # Probably doesn't matter, but np.cov() uses different normalization than np.var(); could instead do:
            #eta = np.sqrt(2 * np.var(ell1) / np.var(ell2))
            ly = ell1 + eta * ell2

        if use_simple_es:
            es.tell(y.flatten(), ly.flatten())
            # Copy out updated values from ES for local use
            x = es.mean
            sigma = es.sigma
            s = es.s
        else:
            # Select and recombine
            idx = np.argsort(ly).flatten()
            zavg = z[:, idx] @ w.T

            # Update x, s, and sigma
            x = x + sigma * zavg
            s = (1-c) * s + np.sqrt(mu_eff * c * (2-c)) * zavg
            sigma = sigma * np.exp(c * (np.linalg.norm(s, ord=2)/e - 1) / d)

        update_history()

    return rec, f_best <= f_target, history


# For Matlab:
# format long g
# [xBest, fEvals, exitflag, rec] = dAL_ES(@g03, ones(10,1)*50, 1, 4000, -1.0 + 1.0e-8, 1.0e-06);

# For Python:
# exec(open("dal_es.py").read())
if __name__ == "__main__":
    import sys

    np.set_printoptions(suppress=True, linewidth=np.nan, threshold=sys.maxsize, precision=18)

    # Initialize problem
    prob = problems.g06()
    fopt = prob.fopt
    ftarget = fopt + 1.0e-8 * abs(fopt)
    lamb = 4 + int(np.floor(3 * np.log(prob.n)))

    total_evals = 0
    unsuccesses = 0
    runs = 15

    for i in range(runs):
        print(f"Run {i+1} starting")
        rec, flag, r = dal_es(
            prob,
            np.vstack([1.0] * prob.n),
            1.0,
            (int)(100000/lamb),
            ftarget,
            1.0e-6
        )
        if r[-1].stop[1]:
            total_evals = total_evals + r[-1].evals
        else:
            unsuccesses = unsuccesses + 1

    print("Avg. evals:", total_evals / runs)
    print("Unsuccessful runs:", unsuccesses)

    if 11 < 3:
        rec, flag, r = dal_es(
            prob,
            np.vstack([1.0] * prob.n),
            1.0,
            (int)(100000 / lamb),
            ftarget,
            1.0e-6
        )
        print("Final x: ")
        print(r[-1].x)
        print("Final f(x): ")
        print(prob.f(r[-1].x))
        print("Best y: ")
        print(r[-1].x_best)
        print("Best f(y): ")
        print(r[-1].f_best)
        print("Actual f_opt:")
        print(prob.fopt)
        print("Evaluation count:")
        print(r[-1].evals)
        print()

