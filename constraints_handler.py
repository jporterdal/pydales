import numpy as np
from math import gamma
import itertools as it
import cma


class SimpleES(cma.interfaces.OOOptimizer):
    """ A simple implementation of an evolution strategy that mimics the ask/tell interface used by
    CMAEvolutionStrategy.

    Note that most other internals are not (or are minimally) implemented.

    Example, given f and g representing objective and constraint functions, respectively:
    >>> es = cma.SimpleES([1.0] * 3, 1)
    >>> y = es.ask()
    >>> es.tell(y, f(y), g(y))
    >>> print(es.x_best)
    """
    def __init__(self, x0, sigma0, inopts = None):
        inopts = {} if inopts is None else inopts
        self.opts = cma.CMAOptions(inopts).complement()

        self.rng = np.random.default_rng()
        self._stopdict = cma.evolution_strategy._CMAStopDict()


        self.countiter = 0
        self.x = np.array(x0)
        self.sigma = sigma0
        self.f_best = np.inf
        self.x_best = self.x
        self.evals = 0

        self.n = self.N = n = len(x0)

        # Set the ES parameters
        self.lamb = lamb = 4 + int(np.floor(3 * np.log(n)))
        w = np.array([np.log((lamb + 1) / 2) - np.log(np.arange(1, lamb + 1))])
        w[w < 0] = 0
        w = w / np.sum(w)
        self.w = w
        self.mu_eff = mu_eff = 1 / np.sum(w ** 2)
        self.c = c = (mu_eff + 2) / (n + mu_eff + 5)
        self.d = d = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c
        self.e = np.sqrt(2) * gamma((n + 1) / 2) / gamma(n / 2)
        self.theta = 0.25
        self.s = np.zeros((n, 1))

        # Initialize
        self.y = [np.zeros(n)] * lamb
        self.z = [np.zeros(n)] * lamb


    @property
    def result(self):
        return cma.evolution_strategy.CMAEvolutionStrategyResult(
            self.x_best,
            self.f_best,
            self.evals,
            self.evals,
            self.countiter,
            0,
            0,
            self.stop()
        )

    @property
    def mean(self):
        return np.copy(self.x)

    def update_best(self, y, fy, filter_best=None):
        """Updates the internal records for best individual (x_best) and associated objective function
        score (f_best). If filter_best is used, interpret as a Boolean index vector that indicates which
        individuals in 'y' are feasible. The result is that an entry marked as False will not be
        considered for the best individual records.
        """

        if filter_best is None:
            filter_best = [True] * len(fy)
        fy = np.array(np.copy(fy))
        fy[np.logical_not(filter_best)] = np.inf
        if np.any(fy < self.f_best):
            kk = np.argmin(fy)
            self.f_best = fy[kk]
            self.x_best = y[kk]

    def stop(self):
        """Attempts to mimic the stop() function of CMAEvolutionStrategy that returns True when the ES
        detects it should stop.
        """

        if self.countiter > 0 and self.opts['termination_callback']:
            self.callbackstop = cma.utilities.utils.ListOfCallables(self.opts['termination_callback'])(self)
        self._stopdict._get_value = None  # Hack-y; adapted directly from CMAEvolutionStrategy implementation
        res = self._stopdict(self, True)

        return res


    def ask(self):
        """Returns a list() of np.array()'s representing offspring vectors calculated using standard
        normal vectors 'z' centered at the internal centroid 'x' and scaled by step-size 'sigma'. The vectors 'z' are
        saved internally to be used when ordering offspring.
        """

        self.z = [self.rng.standard_normal(self.n) for _ in range(self.lamb)]

        return [self.x + self.sigma * zz for zz in self.z]


    def tell(self, X, F, filter_feasible=None):
        """Updates internal parameters and centroid based on passed individuals (X) and objective function scores (F).
        Marking individuals as infeasible (using filter_feasible) only removes them from consideration when
        updating the best x and best f(x) scores.

        ***** NB: For simplicity, at present this IGNORES the parameter X and assumes the scores in F are given in the
        same order as returned by self.ask()

        @param X: list of 1-dim np.array() representing offspring vectors
        @param F: list of scalars representing F(x) for each of the x in X
        @param filter_best: optional list of Boolean values indicating which entries are feasible
        """

        self.update_best(X, F, filter_feasible)
        self.evals = self.evals + self.lamb

        idx = np.argsort(F).flatten()


        z_sorted = np.array(self.z).T[:, idx]
        zavg = z_sorted @ self.w.T

        # Update x, s, and sigma
        self.x = self.x + self.sigma * zavg.flatten()
        self.s = (1 - self.c) * self.s + np.sqrt(self.mu_eff * self.c * (2 - self.c)) * zavg
        self.sigma = self.sigma * np.exp(self.c * (np.linalg.norm(self.s, ord=2) / self.e - 1) / self.d)


class DirectAugmentedLagrangian():
    """Object-oriented implementation of the Direct Augmented Lagrangian approach of Porter and Arnold (GECCO'24)
    intended to mirror the pycma interface used for AugmentedLagrangian. Some differences are apparent. Notably,
    dAL makes direct changes to the underlying evolution strategy so stores a reference to its ES object internally.

    Example, given f and g representing objective and constraint functions, respectively::
    >>> y = es.ask()  # Generate offspring
    >>> eva = PopulationEvaluator(f, g)(y)  # Evaluate offspring and store results
    >>> dal.update(eva.F, eva.G)  # Update dAL internals
    >>> dal(eva.X, eva.F, eva.G)  # Determine and evaluate Lagrangian then update ES
    """

    def __init__(self, dimension, es, equality=False):
        self._equality = np.array(equality, dtype=bool)
        self._dimension = self.n = n = dimension
        self.l = l = np.sum(np.logical_not(self._equality))  # inequalities
        self.m = m = np.sum(self._equality)  # equalities

        self.es = es  # Expects, eg, instance of class cma.CMAEvolutionStrategy

        # Set the ES parameters
        self.lamb = 4 + int(np.floor(3 * np.log(n)))
        w = np.array([np.log((self.lamb + 1) / 2) - np.log(np.arange(1, self.lamb + 1))])
        w[w < 0] = 0
        self.w = w / np.sum(w)
        self.mu_eff = 1 / np.sum(w ** 2)
        self.c = (self.mu_eff + 2) / (n + self.mu_eff + 5)
        self.theta = 0.25

        # Initialize
        self.fy = np.zeros((1, self.lamb))
        self.ghy = np.ones((l + m, self.lamb))
        self.q = np.zeros((l + m, 1))
        self.R = np.eye((l + m))
        self.W = np.zeros(l + m).astype(bool)


    def update_W(self):
        """Update the working set W.
        """

        nu = self.var_coeff(self.ghy)

        # Generate column vector with 1 where  0 <= nu <= theta  in first (L) entries or
        #  abs(nu) < theta  for remainder (M) entries
        severe = np.append(
            np.logical_and(
                0 <= nu[:self.l],
                nu[:self.l] < self.theta
            ),
            np.abs(nu[self.l:]) < self.theta,
        )
        # Recall:  L entries are inequalities, M entries are equalities

        # Generate column vector which is 1 if (already in W or (inequality and nu > 0) or (equality)) AND not already
        #  in severe
        potential_less_severe = np.logical_or(self.W,
                                      np.append(nu[:self.l] > 0,
                                                np.ones((self.m,1)).astype('bool')
                                                ),
                                      )
        less_severe = np.logical_and(potential_less_severe, np.logical_not(severe))
        # Result here is that equalities are always included as at least a less_severe entry


        # Select a maximal subset of severely violated constraints
        W0 = np.zeros(self.l+self.m).astype('bool')  # initialize W0, flat to match shape of W
        found_one = False
        kk = min(self.n, np.sum(severe))

        while kk > 0 and not found_one:
            # Choose 'kk' of the severe indices; after transpose, this will be a matrix of kk rows, each column being
            #  a distinct combination of 'kk' of the indices
            ww = np.array(list(
                it.combinations(np.flatnonzero(severe), kk)
            )).T

            num_ww_cols = ww.shape[1]
            ww = ww[:, np.random.permutation(range(num_ww_cols))]  # Randomize the columns of ww
            for col_ww in range(num_ww_cols):
                W0 = np.zeros(self.l+self.m).astype('bool')  # re-initialize W0
                W0[ww[:, col_ww]] = True  # set entries to 1 using choice of 'kk' 'severe' indices, taken from ww

                # Check condition variable of smoothed approximation to covar(g,g); if not too large,
                #  found_one = true and we break out of both loops
                if np.linalg.cond(self.R[np.ix_(W0, W0)]) < 1.0e+16:
                    found_one = True
                    break

            # End result is we EXHAUSTIVELY check all possible combinations of severely violated constraints and accept
            #  the first one that allows the condition variable to be "not too large".

            kk = kk - 1

        # Add as many less severely violated constraints as possible
        W1 = np.copy(W0)
        found_one = False
        kk = min(self.n-np.sum(W0), np.sum(less_severe))

        while kk > 0 and not found_one:
            ww = np.array(list(
                it.combinations(np.flatnonzero(less_severe), kk)
            )).T
            num_ww_cols = ww.shape[1]
            ww = ww[:, np.random.permutation(range(num_ww_cols))]  # Randomize the columns of ww
            for col_ww in range(num_ww_cols):
                W1 = np.copy(W0)
                W1[ww[:, col_ww]] = True

                if np.linalg.cond(self.R[np.ix_(W1, W1)]) > 1.0e+16:
                    continue

                a = np.zeros((self.l+self.m, 1))  # typo fix
                a[W1] = np.linalg.solve(-self.R[np.ix_(W1, W1)], self.q[W1])
                a[W0] = 0
                if np.min(a[:self.l]) >= 0:  # Check Lagrange multipliers
                    found_one = True
                    break

            kk = kk - 1

        self.W = np.copy(W1)

    def _reorder_gh(self, gh):
        """Convenience method for mirroring interface used by pycma's AugmentedLagrangian.

          Takes a matrix of constraint values with columns corresponding to g(y) and h(y) according to internal
         'equality' array and returns a matrix with rows corresponding to g(y) and h(y) with all inequalities
          ordered to appear before equalities. In transferring rows to columns (representing evaluations for individual
          points), the given order is preserved.
          """

        ghy = np.zeros(self.ghy.shape)

        # Assume that gh is list of col vectors, as returned by eg [gh_flat(x) for x in X] from an instance of Problem()
        gh = np.hstack(gh)
        # So here, gh is a matrix with rows corresponding to constraints and columns to individual offspring

        try:
            assert ghy.shape == gh.shape
        except AssertionError:
            print("(internal) ghy.shape = ", ghy.shape)
            print("(passed) gh.shape = ", gh.shape)
            raise

        ghy[:self.l, :] = gh[np.logical_not(self._equality), :]
        ghy[self.l:, :] = gh[self._equality, :]

        return ghy

    def update(self, F, G):
        """Update Lagrange multipliers and other internals based on f/g evaluations of population, given by lists
        F and G.
        """

        fy = np.hstack(F)
        self.ghy = ghy = self._reorder_gh(G)

        # After reordering, each column of ghy represents the constraint values (g+h) for a single offspring y, so
        #  matrix Xi from dAL-ES.
        self.F_out = fy
        self.G_out = ghy

        # Accumulate derivative related information
        M = np.cov(np.vstack((fy, ghy)).T, rowvar=False)  # Imitating Matlab
        # since in Matlab:  "columns represent random variables and whose rows represent observations"
        self.q = (1 - self.c) * self.q + self.c * M[1:, [0]] / self.es.sigma ** 2
        self.R = (1 - self.c) * self.R + self.c * M[1:, 1:] / self.es.sigma ** 2

        self.update_W()

    def __call__(self, Y, F, G):
        """Use passed population of points (Y), objective function evaluations (F), and constraint function
        evaluations (G) along with internal values (assumed already set by calling .update(F,G)) to evaluate Lagrangian 
        L(Y) on population of points. Will internally call es.tell() with appropriate values, if needed.
        """
        fy = np.hstack(F)  # Make column vector
        ghy = self._reorder_gh(G)


        # Determine Lagrange and penalty values
        try:
            alpha = np.linalg.solve(-self.R[np.ix_(self.W, self.W)], self.q[self.W])
        except np.linalg.LinAlgError:  # we should not see this
            print(self.R)
            print(-self.R[np.ix_(self.W, self.W)])
            print(self.W)
            print(self.q)
            raise
        pi_inv = self.R[np.ix_(self.W, self.W)]

        ell1 = fy + alpha.T @ ghy[self.W, :]
        ell2 = np.array(
            [np.diag(ghy[self.W, :].T @ np.linalg.solve(pi_inv, ghy[self.W, :])).T]
        )  # take diagonal as a row vector


        if np.sum(np.abs(self.var_coeff(self.ghy)) > self.theta) > self.n + 1 and self.var_coeff(ell2) > self.theta:
            self.es.sigma = self.es.sigma / 2  # directly modifies stepsize within ES without calling tell()
            return

        if np.sum(self.W) == 0:
            ly = fy
        elif np.sum(self.W) == self.n or self.var_coeff(ell2) < self.theta:
            ly = ell2
        else:
            CC = np.cov(ell1, ell2)
            eta = np.sqrt(2 * CC[0, 0] / CC[1, 1])
            ly = ell1 + eta * ell2

        return self.es.tell(Y, ly.flatten())

    # Unused - safe to remove?
    def max_vio(self, ghy):
        vio = np.zeros((1, ghy.shape[1]))
        if self.l > 0:  # inequality constraints
            vio = np.max(np.vstack((vio, np.max(ghy[:self.l, :], axis=0))), axis=0)
        if self.m > 0:  # equality constraints
            vio = np.max(np.vstack((vio, np.max(ghy[self.l:, :], axis=0))), axis=0)
        return vio

    def var_coeff(self, xi):
        return np.std(xi, axis=1) / np.mean(xi, axis=1)

# Bare-bones unit tests for dAL and ES implementations.
def init_for_test():
    equality = [1, 0, 1, 0, 1, 1, 0]
    es = cma.CMAEvolutionStrategy(4 * [1], 1, {'termination_callback': lambda es: sum(es.mean ** 2) < 1e-1})
    dal = DirectAugmentedLagrangian(4, es, equality)
    return es, dal, equality

def test_dal_init():
    es, dal, equality = init_for_test()

    assert np.all(dal._equality == np.array(equality, dtype=bool))
    assert dal.lamb == 8
    assert dal.fy.shape == (1,8)
    assert dal.ghy.shape == (len(equality), dal.lamb)

def test_dal_ask():
    es, dal, equality = init_for_test()

    y = es.ask()
    assert len(y) == dal.lamb
    assert len(y[-1]) == es.N == dal.n

def test_functions():
    return lambda x: np.sum(x), lambda x: np.vstack([6 * x[0], 5 * x[1], 4 * x[0], 3 * x[1], 2 * x[0], 1 * x[1], 0])

def test_dal_update():
    es, dal, equality = init_for_test()
    f, g = test_functions()

    eva = PopulationEvaluator(f, g)(es.ask())
    dal.update(eva.F, eva.G)

    assert len(eva.F) == len(eva.G) == dal.lamb
    assert dal.q.shape == (len(equality), 1)
    assert dal.R.shape == (len(equality), len(equality))

def test_dal_call():
    es, dal, equality = init_for_test()
    f, g = test_functions()

    eva = PopulationEvaluator(f, g)(es.ask())
    dal.update(eva.F, eva.G)

    dal(eva.X, eva.F, eva.G)

    assert es.sigma != 1


# exec(open("constraints_handler.py").read())
if __name__ == "__main__":
    import sys, problems
    from cma.constraints_handler import AugmentedLagrangian, PopulationEvaluator

    np.set_printoptions(suppress=True, linewidth=np.nan, threshold=sys.maxsize)

    do_tests = False
    if do_tests:
        print("\n----- ----- Tests starting ----- \n")
        test_dal_init()
        test_dal_ask()
        test_dal_update()
        test_dal_call()
        print("\n----- ----- Tests finished ----- \n\n")

    prob = problems.g06()
    f, g = prob.f_flat, prob.gh_flat
    fopt = prob.fopt

    # TODO: passing this function as an option is not doing what I expected it to be doing.
    def termination_callback(e):
        ftarget = fopt + 1.0e-8 * abs(fopt)
        delta = 0.25

        x_best = es.best.x
        return f(x_best) <= ftarget and np.max(x_best) <= 1.0e-6

    es = cma.CMAEvolutionStrategy(prob.n * [1], 1, {'termination_callback': termination_callback})
    #es = SimpleES(prob.n * [1], 1, {'termination_callback': lambda es: sum(es.mean ** 2) < 1e-8})
    al = AugmentedLagrangian(es.N)

    dal = DirectAugmentedLagrangian(es.N, es, prob.equality)

    #al.set_coefficients(eva.F, eva.G)
    #al.update(eva.m['f'], eva.m['g'])
    #es.tell(eva.X, [f + sum(al(g)) for f, g in zip(eva.F, eva.G)])

    while not es.stop() and es.result.evaluations <= 5000:
        y = es.ask()
        eva = PopulationEvaluator(f, g)(y)  # , m=es.mean

        dal.update(eva.F, eva.G)
        dal(eva.X, eva.F, eva.G)
