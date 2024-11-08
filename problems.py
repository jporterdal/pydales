import numpy as np

class Problem():
    def __init__(self, n=0, l=0, m=0, fopt=np.nan):
        self.n = n
        self.l = l
        self.m = m
        self.fopt = fopt

    def f(self, x):
        raise NotImplementedError

    def g(self, x):
        if self.l == 0:
            return np.empty((0, 0))
        else:
            raise NotImplementedError

    def h(self, x):
        if self.m == 0:
            return np.empty((0, 0))
        else:
            raise NotImplementedError

    def unpack(self):
        # n, l, m, f, g, h,
        return self.n, self.l, self.m, self.f, self.g, self.h


class g03(Problem):
    def __init__(self):
        super().__init__(n=10, l=20, m=1, fopt=-1.00050010001000)

    def f(self, x):
        return -10**5 * np.prod(x)
    def g(self, x):
        return np.vstack((
            -np.eye(self.n) @ x,
            np.eye(self.n) @ x - np.ones((10,1)),
        ))
    def h(self, x):
        return x.T @ x - 1


class g04(Problem):
    def __init__(self):
        super().__init__(n=5, l=16, m=0, fopt=-30665.53867178332)

    def _h1(self, x):
        return 85.334407 + 0.0056858 * x[1] * x[4] + 0.0006262 * x[0] * x[3] - 0.0022053 * x[2] * x[4]  # coeff corrected to 0.0006262

    def _h2(self, x):
        return 80.51249 + 0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + 0.0021813 * x[2] ** 2

    def _h3(self, x):
        return 9.300961 + 0.0047026 * x[2] * x[4] + 0.0012547 * x[0] * x[2] + 0.0019085 * x[2] * x[3]

    def g(self, x):
        v = np.zeros(self.l)

        v[0] = self._h1(x) - 92
        v[1] = -self._h1(x)

        v[2] = self._h2(x) - 110
        v[3] = 90 - self._h2(x)

        v[4] = self._h3(x) - 25
        v[5] = 20 - self._h3(x)

        v[6] = 78. - x[0]
        v[7] = x[0] - 102.
        v[8] = 33. - x[1]
        v[9] = x[1] - 45.
        v[10] = 27. - x[2]
        v[11] = x[2] - 45.
        v[12] = 27. - x[3]
        v[13] = x[3] - 45.
        v[14] = 27. - x[4]
        v[15] = x[4] - 45.

        return np.vstack(v)

    def f(self, x):
        5.3578547 * x[2] ** 2 + 0.8356891 * x[0] * x[4] + 37.293239 * x[0] - 40792.141



class g05(Problem):
    def __init__(self):
        super().__init__(n=4, l=10, m=3, fopt=5126.4967140071)

    def f(self, x):
        return 3*x[0] + 0.000001 * x[0]**3 + 2*x[1] + (0.000002/3.) * x[1]**3

    def g(self, x):
        v = np.zeros(self.l)

        v[0] = -x[3] + x[2] - 0.55
        v[1] = -x[2] + x[3] - 0.55

        v[2] = -x[0]
        v[3] = x[0] - 1200
        v[4] = -x[1]
        v[5] = x[1] - 1200
        v[6] = -0.55 - x[2]
        v[7] = x[2] - 0.55
        v[8] = -0.55 - x[3]
        v[9] = x[3] - 0.55

        return np.vstack(v)

    def h(self, x):
        v = np.zeros(self.l)

        v[0] = 1000 * np.sin(-x[2] - 0.25) + 1000 * np.sin(-x[3] - 0.25) + 894.8 - x[0]
        v[1] = 1000 * np.sin(x[2] - 0.25) + 1000 * np.sin(x[2] - x[3] - 0.25) + 894.8 - x[1]
        v[2] = 1000 * np.sin(x[3] - 0.25) + 1000 * np.sin(x[3] - x[2] - 0.25) + 1294.8

        return np.vstack(v)


class g06(Problem):
    def __init__(self):
        super().__init__(n=2, l=6, m=0, fopt=-6961.81381)

    def f(self, x):
        return (x[0] - 10.0)**3 + (x[1] - 20.0)**3

    def g(self, x):
        v = np.zeros(self.l)

        v[0] = (-(x[0] - 5.0) ** 2 - (x[1] - 5.0) ** 2 + 100.0)
        v[1] = ((x[0] - 6.0) ** 2 + (x[1] - 5.0) ** 2 - 82.81)
        v[2] = 13.0 - x[0]
        v[3] = x[0] - 100.0
        v[4] = -x[1]
        v[5] = x[1] - 100.0

        return np.vstack(v)


class g07(Problem):
    def __init__(self):
        super().__init__(n=10, l=28, m=0, fopt=24.3062091)

    def f(self, x):
        return x[0]**2 + x[1]**2 + x[0]*x[1] - 14*x[0] - 16*x[1] + (x[2] - 10)**2 + 4*(x[3]-5)**2 + (x[4] - 3)**2 + 2*(x[5] - 1)**2 + 5*x[6]**2 + 7*(x[7] - 11)**2 + 2*(x[8] - 10)**2 + (x[9] - 7)**2 + 45

    def g(self, x):
        v = np.zeros(self.l)

        v[0] = 4 * x[0] + 5 * x[1] - 3 * x[6] + 9 * x[7] - 105
        v[1] = 10 * x[0] - 8 * x[1] - 17 * x[6] + 2 * x[7]
        v[2] = -8 * x[0] + 2 * x[1] + 5 * x[8] - 2 * x[9] - 12
        v[3] = -3 * x[0] + 6 * x[1] + 12 * (x[8] - 8) ** 2 - 7 * x[9]
        v[4] = 3 * (x[0] - 2) ** 2 + 4 * (x[1] - 3) ** 2 + 2 * x[2] ** 2 - 7 * x[3] - 120
        v[5] = x[0] ** 2 + 2 * (x[1] - 2) ** 2 - 2 * x[0] * x[1] + 14 * x[4] - 6 * x[5]
        v[6] = 5 * x[0] ** 2 + 8 * x[1] + (x[2] - 6) ** 2 - 2 * x[3] - 40
        v[7] = (x[0] - 8) ** 2 + 4 * (x[1] - 4) ** 2 + 6 * x[4] ** 2 - 2 * x[5] - 60

        v[8] = -x[0] - 10
        v[9] = x[0] - 10
        v[10] = -x[1] - 10
        v[11] = x[1] - 10
        v[12] = -x[2] - 10
        v[13] = x[2] - 10
        v[14] = -x[3] - 10
        v[15] = x[3] - 10
        v[16] = -x[4] - 10
        v[17] = x[4] - 10
        v[18] = -x[5] - 10
        v[19] = x[5] - 10
        v[20] = -x[6] - 10
        v[21] = x[6] - 10
        v[22] = -x[7] - 10
        v[23] = x[7] - 10
        v[24] = -x[8] - 10
        v[25] = x[8] - 10
        v[26] = -x[9] - 10
        v[27] = x[9] - 10

        return np.vstack(v)



class g09(Problem):
    def __init__(self):
        super().__init__(n=7, l=18, m=0, fopt=680.630057)

    def f(self, x):
        return (x[0] - 10)**2 + 5*(x[1] - 12)**2 + x[2]**4 + 3*(x[3] - 11)**2 + 10*x[4]**6 + 7*x[5]**2 + x[6]**4 - 4*x[5]*x[6] - 10*x[5] - 8*x[6]

    def g(self, x):
        v = np.zeros(self.l)

        v[0] = -127 + 2 * x[0] ** 2 + 3 * x[1] ** 4 + x[2] + 4 * x[3] ** 2 + 5 * x[4]
        v[1] = -196 + 23 * x[0] + x[1] ** 2 + 6 * x[5] ** 2 - 8 * x[6]
        v[2] = -282 + 7 * x[0] + 3 * x[1] + 10 * x[2] ** 2 + x[3] - x[4]
        v[3] = 4 * x[0] ** 2 + x[1] ** 2 - 3 * x[0] * x[1] + 2 * x[2] ** 2 + 5 * x[5] - 11 * x[6]

        v[4] = -x[0] - 10
        v[5] = x[0] - 10
        v[6] = -x[1] - 10
        v[7] = x[1] - 10
        v[8] = -x[2] - 10
        v[9] = x[2] - 10
        v[10] = -x[3] - 10
        v[11] = x[3] - 10
        v[12] = -x[4] - 10
        v[13] = x[4] - 10
        v[14] = -x[5] - 10
        v[15] = x[5] - 10
        v[16] = -x[6] - 10
        v[17] = x[6] - 10


        return np.vstack(v)


class g10(Problem):
    def __init__(self):
        super().__init__(n=8, l=22, m=0, fopt=7049.2480)

    def f(self, x):
        return x[0] + x[1] + x[2]

    def g(self, x):
        v = np.zeros(self.l)

        v[0] = 1 * (0.0025 * (x[3] + x[5]) - 1)
        v[1] = 1 * (0.0025 * (x[4] + x[6] - x[3]) - 1)
        v[2] = 1 * (0.01 * (x[7] - x[4]) - 1)
        v[3] = -x[0] * x[5] + 833.33252 * x[3] + 100 * x[0] - 83333.333
        v[4] = -x[1] * x[6] + 1250 * x[4] + x[1] * x[3] - 1250 * x[3]
        v[5] = -x[2] * x[7] + 1250000 + x[2] * x[4] - 2500 * x[4]

        v[6] = 100 - x[0]
        v[7] = x[0] - 10000
        v[8] = 1000 - x[1]
        v[9] = x[1] - 10000
        v[10] = 1000 - x[2]
        v[11] = x[2] - 10000
        v[12] = 10 - x[3]
        v[13] = x[3] - 1000
        v[14] = 10 - x[4]
        v[15] = x[4] - 1000
        v[16] = 10 - x[5]
        v[17] = x[5] - 1000
        v[18] = 10 - x[6]
        v[19] = x[6] - 1000
        v[20] = 10 - x[7]
        v[21] = x[7] - 1000


        return np.vstack(v)


class g11(Problem):
    def __init__(self):
        super().__init__(n=2, l=4, m=1, fopt=0.75)

    def f(self, x):
        return x[0]**2 + (x[1] - 1)**2

    def g(self, x):
        v = np.zeros(self.l)

        v[0] = x[1] - x[0] ** 2

        v[1] = -1 - x[0]
        v[2] = x[0] - 1
        v[3] = -1 - x[1]
        v[4] = x[1] - 1

        return np.vstack(v)

    def h(self, x):
        v = np.zeros(self.m)

        v[0] = -(x[1] - x[0] ** 2)

        return np.vstack(v)


class hps240(Problem):
    def __init__(self):
        super().__init__(n=5, l=6, m=0, fopt=-5000)

    def f(self, x):
        return -np.sum(x)

    def g(self, x):
        v = np.zeros(self.l)

        v[0] = np.sum([(i + 1 + 9) * xi for i, xi in enumerate(x)]) - 50000

        v[1] = -x[0]
        v[2] = -x[1]
        v[3] = -x[2]
        v[4] = -x[3]
        v[5] = -x[4]

        return np.vstack(v)


class hps241(Problem):
    def __init__(self):
        super().__init__(n=5, l=6, m=0, fopt=-125000/7.)

    def f(self, x):
        return -np.sum([(i+1)*xi for i,xi in enumerate(x)])

    def g(self, x):
        v = np.zeros(self.l)

        v[0] = np.sum([(i + 1 + 9) * xi for i, xi in enumerate(x)]) - 50000

        v[1] = -x[0]
        v[2] = -x[1]
        v[3] = -x[2]
        v[4] = -x[3]
        v[5] = -x[4]

        return np.vstack(v)


class km(Problem):
    def __init__(self, n):
        super().__init__(n=n, l=n**2, m=0, fopt=n**3)

    def f(self, x):
        return x.flatten()[-1]

