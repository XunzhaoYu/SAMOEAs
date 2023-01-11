import numpy as np
from pyDOE import lhs
#from pymoo.core.problem import Problem
#from pymoo.problems.many import generic_sphere, get_ref_dirs
#from pymoo.util.function_loader import load_function
#from pymoo.util.misc import powerset


class WFG:
    def __init__(self, n_var, n_obj, k=None, l=None, **kwargs):
        self.n_var = n_var
        self.n_obj = n_obj
        self.lowerbound = 0.0
        self.upperbound = 2 * np.arange(1, n_var + 1).astype(float)

        self.S = np.arange(2, 2 * self.n_obj + 1, 2).astype(float)
        self.A = np.ones(self.n_obj - 1)

        if k:
            self.k = k
        else:
            """
            if n_obj == 2:
                self.k = 4
            else:
                self.k = 2 * (n_obj - 1)
            """
            self.k = self.n_obj-1

        if l:
            self.l = l
        else:
            self.l = n_var - self.k
        print("k:", self.k, ".  l:", self.l)

        self.validate(self.l, self.k, self.n_obj)

    def get_bounds(self, upper=True):
        if upper:
            return self.upperbound.copy()
        else:
            return self.lowerbound.copy()

    def validate(self, l, k, n_obj):
        if n_obj < 2:
            raise ValueError('WFG problems must have two or more objectives.')
        if not k % (n_obj - 1) == 0:
            raise ValueError('Position parameter (k) must be divisible by number of objectives minus one.')
        #if k < 4:
        #    raise ValueError('Position parameter (k) must be greater or equal than 4.')
        if (k + l) < n_obj:
            raise ValueError('Sum of distance and position parameters must be greater than num. of objs. (k + l >= M).')

    def _post(self, t, a):
        x = []
        for i in range(t.shape[1] - 1):
            x.append(np.maximum(t[:, -1], a[i]) * (t[:, i] - 0.5) + 0.5)
        x.append(t[:, -1])
        return np.column_stack(x)

    def _calculate(self, x, s, h):
        return x[:, -1][:, None] + s * np.column_stack(h)

    def _rand_optimal_position(self, n):
        return np.random.random((n, self.k))

    def _positional_to_optimal(self, K):
        suffix = np.full((len(K), self.l), 0.35)
        X = np.column_stack([K, suffix])
        return X * self.upperbound

    """
    def _calc_pareto_set_extremes(self):
        ps = np.ones((2 ** self.k, self.k))
        for i, s in enumerate(powerset(np.arange(self.k))):
            ps[i, s] = 0
        return self._positional_to_optimal(ps)

    def _calc_pareto_set_interior(self, n_points):
        return self._positional_to_optimal(self._rand_optimal_position(n_points))

    def _calc_pareto_set(self, n_points=500, *args, **kwargs):
        extremes = self._calc_pareto_set_extremes()
        interior = self._calc_pareto_set_interior(n_points - len(extremes))
        return np.row_stack([extremes, interior])

    def _calc_pareto_front(self, ref_dirs=None, n_iterations=200, points_each_iteration=200, *args, **kwargs):
        pf = self.evaluate(self._calc_pareto_set_extremes(), return_values_of=["F"])

        if ref_dirs is None:
            ref_dirs = get_ref_dirs(self.n_obj)

        for k in range(n_iterations):
            _pf = self.evaluate(self._calc_pareto_set_interior(points_each_iteration), return_values_of=["F"])
            pf = np.row_stack([pf, _pf])

            ideal, nadir = pf.min(axis=0), pf.max(axis=0)

            N = (pf - ideal) / (nadir-ideal)
            dist_matrix = load_function("calc_perpendicular_distance")(N, ref_dirs)

            closest = np.argmin(dist_matrix, axis=0)
            pf = pf[closest]

        pf = pf[np.lexsort(pf.T[::-1])]
        return pf
    """

class WFG1(WFG):

    @staticmethod
    def t1(x, n, k):
        x[:, k:n] = _transformation_shift_linear(x[:, k:n], 0.35)
        return x

    @staticmethod
    def t2(x, n, k):
        x[:, k:n] = _transformation_bias_flat(x[:, k:n], 0.8, 0.75, 0.85)
        return x

    @staticmethod
    def t3(x, n):
        x[:, :n] = _transformation_bias_poly(x[:, :n], 0.02)
        return x

    @staticmethod
    def t4(x, m, n, k):
        w = np.arange(2, 2 * n + 1, 2)
        gap = k // (m - 1)
        t = []
        for m in range(1, m):
            _y = x[:, (m - 1) * gap: (m * gap)]
            _w = w[(m - 1) * gap: (m * gap)]
            t.append(_reduction_weighted_sum(_y, _w))
        t.append(_reduction_weighted_sum(x[:, k:n], w[k:n]))
        return np.column_stack(t)

    def evaluate(self, x):
        y = x / self.upperbound
        y = WFG1.t1(y, self.n_var, self.k)
        y = WFG1.t2(y, self.n_var, self.k)
        y = WFG1.t3(y, self.n_var)
        y = WFG1.t4(y, self.n_obj, self.n_var, self.k)

        y = self._post(y, self.A)

        h = [_shape_convex(y[:, :-1], m + 1) for m in range(self.n_obj - 1)]
        h.append(_shape_mixed(y[:, 0], alpha=1.0, A=5.0))

        return self._calculate(y, self.S, h)

    def _rand_optimal_position(self, n):
        return np.power(np.random.random((n, self.k)), 50.0)


class WFG2(WFG):

    def validate(self, l, k, n_obj):
        super().validate(l, k, n_obj)
        if validate_wfg2_wfg3(l):
            self.k = 2 * self.k
            self.l = self.n_var - self.k
            print("new k:", self.k, ".  l:", self.l)


    @staticmethod
    def t2(x, n, k):
        y = [x[:, i] for i in range(k)]

        l = n - k
        ind_non_sep = k + l // 2

        i = k + 1
        while i <= ind_non_sep:
            head = k + 2 * (i - k) - 2
            tail = k + 2 * (i - k)
            y.append(_reduction_non_sep(x[:, head:tail], 2))
            i += 1

        return np.column_stack(y)

    @staticmethod
    def t3(x, m, n, k):
        ind_r_sum = k + (n - k) // 2
        gap = k // (m - 1)

        t = [_reduction_weighted_sum_uniform(x[:, (m - 1) * gap: (m * gap)]) for m in range(1, m)]
        t.append(_reduction_weighted_sum_uniform(x[:, k:ind_r_sum]))

        return np.column_stack(t)

    def evaluate(self, x):
        y = x / self.upperbound
        y = WFG1.t1(y, self.n_var, self.k)
        y = WFG2.t2(y, self.n_var, self.k)
        y = WFG2.t3(y, self.n_obj, self.n_var, self.k)
        y = self._post(y, self.A)

        h = [_shape_convex(y[:, :-1], m + 1) for m in range(self.n_obj - 1)]
        h.append(_shape_disconnected(y[:, 0], alpha=1.0, beta=1.0, A=5.0))

        return self._calculate(y, self.S, h)


class WFG3(WFG):

    def __init__(self, n_var, n_obj, k=None, **kwargs):
        super().__init__(n_var, n_obj, k=k, **kwargs)
        self.A[1:] = 0

    def validate(self, l, k, n_obj):
        super().validate(l, k, n_obj)
        if validate_wfg2_wfg3(l):
            self.k = 2 * self.k
            self.l = self.n_var - self.k
            print("new k:", self.k, ".  l:", self.l)

    def evaluate(self, x):
        y = x / self.upperbound
        y = WFG1.t1(y, self.n_var, self.k)
        y = WFG2.t2(y, self.n_var, self.k)
        y = WFG2.t3(y, self.n_obj, self.n_var, self.k)
        y = self._post(y, self.A)

        h = [_shape_linear(y[:, :-1], m + 1) for m in range(self.n_obj)]

        return self._calculate(y, self.S, h)

    # def _calc_pareto_front(self, ref_dirs=None):
    #     if ref_dirs is None:
    #         ref_dirs = get_ref_dirs(self.n_obj)
    #     return ref_dirs * self.S


class WFG4(WFG):

    @staticmethod
    def t1(x):
        return _transformation_shift_multi_modal(x, 30.0, 10.0, 0.35)

    @staticmethod
    def t2(x, m, k):
        gap = k // (m - 1)
        t = [_reduction_weighted_sum_uniform(x[:, (m - 1) * gap: (m * gap)]) for m in range(1, m)]
        t.append(_reduction_weighted_sum_uniform(x[:, k:]))
        return np.column_stack(t)

    def evaluate(self, x):
        y = x / self.upperbound
        y = WFG4.t1(y)
        y = WFG4.t2(y, self.n_obj, self.k)
        y = self._post(y, self.A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        return self._calculate(y, self.S, h)

    # def _calc_pareto_front(self, ref_dirs=None):
    #     if ref_dirs is None:
    #         ref_dirs = get_ref_dirs(self.n_obj)
    #     return generic_sphere(ref_dirs) * self.S


class WFG5(WFG):

    @staticmethod
    def t1(x):
        return _transformation_param_deceptive(x, A=0.35, B=0.001, C=0.05)

    def evaluate(self, x):
        y = x / self.upperbound
        y = WFG5.t1(y)
        y = WFG4.t2(y, self.n_obj, self.k)
        y = self._post(y, self.A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        return self._calculate(y, self.S, h)

    # def _calc_pareto_front(self, ref_dirs=None):
    #     if ref_dirs is None:
    #         ref_dirs = get_ref_dirs(self.n_obj)
    #     return generic_sphere(ref_dirs) * self.S


class WFG6(WFG):

    @staticmethod
    def t2(x, m, n, k):
        gap = k // (m - 1)
        t = [_reduction_non_sep(x[:, (m - 1) * gap: (m * gap)], gap) for m in range(1, m)]
        t.append(_reduction_non_sep(x[:, k:], n - k))
        return np.column_stack(t)

    def evaluate(self, x):
        y = x / self.upperbound
        y = WFG1.t1(y, self.n_var, self.k)
        y = WFG6.t2(y, self.n_obj, self.n_var, self.k)
        y = self._post(y, self.A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        return self._calculate(y, self.S, h)

    # def _calc_pareto_front(self, ref_dirs=None):
    #     if ref_dirs is None:
    #         ref_dirs = get_ref_dirs(self.n_obj)
    #     return generic_sphere(ref_dirs) * self.S


class WFG7(WFG):

    @staticmethod
    def t1(x, k):
        for i in range(k):
            aux = _reduction_weighted_sum_uniform(x[:, i + 1:])
            x[:, i] = _transformation_param_dependent(x[:, i], aux)
        return x

    def evaluate(self, x):
        y = x / self.upperbound
        y = WFG7.t1(y, self.k)
        y = WFG1.t1(y, self.n_var, self.k)
        y = WFG4.t2(y, self.n_obj, self.k)
        y = self._post(y, self.A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        return self._calculate(y, self.S, h)

    # def _calc_pareto_front(self, ref_dirs=None):
    #     if ref_dirs is None:
    #         ref_dirs = get_ref_dirs(self.n_obj)
    #     return generic_sphere(ref_dirs) * self.S


class WFG8(WFG):

    @staticmethod
    def t1(x, n, k):
        ret = []
        for i in range(k, n):
            aux = _reduction_weighted_sum_uniform(x[:, :i])
            ret.append(_transformation_param_dependent(x[:, i], aux, A=0.98 / 49.98, B=0.02, C=50.0))
        return np.column_stack(ret)

    def evaluate(self, x):
        y = x / self.upperbound
        y[:, self.k:self.n_var] = WFG8.t1(y, self.n_var, self.k)
        y = WFG1.t1(y, self.n_var, self.k)
        y = WFG4.t2(y, self.n_obj, self.k)
        y = self._post(y, self.A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        return self._calculate(y, self.S, h)

    def _positional_to_optimal(self, K):
        k, l = self.k, self.l

        for i in range(k, k + l):
            u = K.sum(axis=1) / K.shape[1]
            tmp1 = np.abs(np.floor(0.5 - u) + 0.98 / 49.98)
            tmp2 = 0.02 + 49.98 * (0.98 / 49.98 - (1.0 - 2.0 * u) * tmp1)
            suffix = np.power(0.35, np.power(tmp2, -1.0))

            K = np.column_stack([K, suffix[:, None]])

        ret = K * (2 * (np.arange(self.n_var) + 1))
        return ret


class WFG9(WFG):

    @staticmethod
    def t1(x, n):
        ret = []
        for i in range(0, n - 1):
            aux = _reduction_weighted_sum_uniform(x[:, i + 1:])
            ret.append(_transformation_param_dependent(x[:, i], aux))
        return np.column_stack(ret)

    @staticmethod
    def t2(x, n, k):
        a = [_transformation_shift_deceptive(x[:, i], 0.35, 0.001, 0.05) for i in range(k)]
        b = [_transformation_shift_multi_modal(x[:, i], 30.0, 95.0, 0.35) for i in range(k, n)]
        return np.column_stack(a + b)

    @staticmethod
    def t3(x, m, n, k):
        gap = k // (m - 1)
        t = [_reduction_non_sep(x[:, (m - 1) * gap: (m * gap)], gap) for m in range(1, m)]
        t.append(_reduction_non_sep(x[:, k:], n - k))
        return np.column_stack(t)

    def evaluate(self, x):
        y = x / self.upperbound
        y[:, :self.n_var - 1] = WFG9.t1(y, self.n_var)
        y = WFG9.t2(y, self.n_var, self.k)
        y = WFG9.t3(y, self.n_obj, self.n_var, self.k)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        return self._calculate(y, self.S, h)

    def _positional_to_optimal(self, K):
        k, l = self.k, self.l

        suffix = np.full((len(K), self.l), 0.0)
        X = np.column_stack([K, suffix])
        X[:, self.k + self.l - 1] = 0.35

        for i in range(self.k + self.l - 2, self.k - 1, -1):
            m = X[:, i + 1:k + l]
            val = m.sum(axis=1) / m.shape[1]
            X[:, i] = 0.35 ** ((0.02 + 1.96 * val) ** -1)

        ret = X * (2 * (np.arange(self.n_var) + 1))
        return ret

    """
    def _calc_pareto_front(self, ref_dirs=None):
        if ref_dirs is None:
            ref_dirs = get_ref_dirs(self.n_obj)
        return generic_sphere(ref_dirs) * self.S
    """


# ---------------------------------------------------------------------------------------------------------
# TRANSFORMATIONS
# ---------------------------------------------------------------------------------------------------------


def _transformation_shift_linear(value, shift=0.35):
    return correct_to_01(np.fabs(value - shift) / np.fabs(np.floor(shift - value) + shift))


def _transformation_shift_deceptive(y, A=0.35, B=0.005, C=0.05):
    tmp1 = np.floor(y - A + B) * (1.0 - C + (A - B) / B) / (A - B)
    tmp2 = np.floor(A + B - y) * (1.0 - C + (1.0 - A - B) / B) / (1.0 - A - B)
    ret = 1.0 + (np.fabs(y - A) - B) * (tmp1 + tmp2 + 1.0 / B)
    return correct_to_01(ret)


def _transformation_shift_multi_modal(y, A, B, C):
    tmp1 = np.fabs(y - C) / (2.0 * (np.floor(C - y) + C))
    tmp2 = (4.0 * A + 2.0) * np.pi * (0.5 - tmp1)
    ret = (1.0 + np.cos(tmp2) + 4.0 * B * np.power(tmp1, 2.0)) / (B + 2.0)
    return correct_to_01(ret)


def _transformation_bias_flat(y, a, b, c):
    ret = a + np.minimum(0, np.floor(y - b)) * (a * (b - y) / b) \
          - np.minimum(0, np.floor(c - y)) * ((1.0 - a) * (y - c) / (1.0 - c))
    return correct_to_01(ret)


def _transformation_bias_poly(y, alpha):
    return correct_to_01(y ** alpha)


def _transformation_param_dependent(y, y_deg, A=0.98 / 49.98, B=0.02, C=50.0):
    aux = A - (1.0 - 2.0 * y_deg) * np.fabs(np.floor(0.5 - y_deg) + A)
    ret = np.power(y, B + (C - B) * aux)
    return correct_to_01(ret)


def _transformation_param_deceptive(y, A=0.35, B=0.001, C=0.05):
    tmp1 = np.floor(y - A + B) * (1.0 - C + (A - B) / B) / (A - B)
    tmp2 = np.floor(A + B - y) * (1.0 - C + (1.0 - A - B) / B) / (1.0 - A - B)
    ret = 1.0 + (np.fabs(y - A) - B) * (tmp1 + tmp2 + 1.0 / B)
    return correct_to_01(ret)


# ---------------------------------------------------------------------------------------------------------
# REDUCTION
# ---------------------------------------------------------------------------------------------------------


def _reduction_weighted_sum(y, w):
    return correct_to_01(np.dot(y, w) / w.sum())


def _reduction_weighted_sum_uniform(y):
    return correct_to_01(y.mean(axis=1))


def _reduction_non_sep(y, A):
    n, m = y.shape
    val = np.ceil(A / 2.0)

    num = np.zeros(n)
    for j in range(m):
        num += y[:, j]
        for k in range(A - 1):
            num += np.fabs(y[:, j] - y[:, (1 + j + k) % m])

    denom = m * val * (1.0 + 2.0 * A - 2 * val) / A

    return correct_to_01(num / denom)


# ---------------------------------------------------------------------------------------------------------
# SHAPE
# ---------------------------------------------------------------------------------------------------------


def _shape_concave(x, m):
    M = x.shape[1]
    if m == 1:
        ret = np.prod(np.sin(0.5 * x[:, :M] * np.pi), axis=1)
    elif 1 < m <= M:
        ret = np.prod(np.sin(0.5 * x[:, :M - m + 1] * np.pi), axis=1)
        ret *= np.cos(0.5 * x[:, M - m + 1] * np.pi)
    else:
        ret = np.cos(0.5 * x[:, 0] * np.pi)
    return correct_to_01(ret)


def _shape_convex(x, m):
    M = x.shape[1]
    if m == 1:
        ret = np.prod(1.0 - np.cos(0.5 * x[:, :M] * np.pi), axis=1)
    elif 1 < m <= M:
        ret = np.prod(1.0 - np.cos(0.5 * x[:, :M - m + 1] * np.pi), axis=1)
        ret *= 1.0 - np.sin(0.5 * x[:, M - m + 1] * np.pi)
    else:
        ret = 1.0 - np.sin(0.5 * x[:, 0] * np.pi)
    return correct_to_01(ret)


def _shape_linear(x, m):
    M = x.shape[1]
    if m == 1:
        ret = np.prod(x, axis=1)
    elif 1 < m <= M:
        ret = np.prod(x[:, :M - m + 1], axis=1)
        ret *= 1.0 - x[:, M - m + 1]
    else:
        ret = 1.0 - x[:, 0]
    return correct_to_01(ret)


def _shape_mixed(x, A=5.0, alpha=1.0):
    aux = 2.0 * A * np.pi
    ret = np.power(1.0 - x - (np.cos(aux * x + 0.5 * np.pi) / aux), alpha)
    return correct_to_01(ret)


def _shape_disconnected(x, alpha=1.0, beta=1.0, A=5.0):
    aux = np.cos(A * np.pi * x ** beta)
    return correct_to_01(1.0 - x ** alpha * aux ** 2)


# ---------------------------------------------------------------------------------------------------------
# UTIL
# ---------------------------------------------------------------------------------------------------------

def validate_wfg2_wfg3(l):
    if not l % 2 == 0:
        #raise ValueError('In WFG2/WFG3 the distance-related parameter (l) must be divisible by 2.')
        print("In WFG2/WFG3 the distance-related parameter (l) must be divisible by 2.")
        print("Set new k and l ...")
        return True
    else:
        return False

def correct_to_01(X, epsilon=1.0e-10):
    X[np.logical_and(X < 0, X >= 0 - epsilon)] = 0
    X[np.logical_and(X > 1, X <= 1 + epsilon)] = 1
    return X





def WFG_validation(WFG_list):
    if 1 in WFG_list:
        print('--- WFG 1 ---')
        dataset = WFG1(10, 3)
        X = np.array([[1.08854981319285, 2.88336864817126, 2.26151969048427, 6.85587897325909, 5.50774672114278, 11.3619491740763, 0.993607643502324, 12.7476499626573, 9.51749373544387, 13.9469154321725],
                      [0.604916530645588, 2.83243236846361, 1.08564315191318, 1.65613202529189, 9.92817774785589, 8.67400816509106, 10.6090373570489, 2.20562483724899, 12.0117687538961, 2.33741938579107],
                      [1.26359517475495, 1.04213391542253, 4.80089481701664, 1.31305713165933, 9.23718752328934, 6.94795393584, 2.53950542445972, 5.07151257421173, 17.2228709914341, 0.9626573771487],
                      [0.153089588517859, 3.48656692679636, 0.675164689470663, 0.0280924154361591, 4.48425908131889, 10.7442702543093, 3.92083395044034, 2.71173650041344, 9.99813377374578, 17.1970834883952]])
        Y = np.array([[2.92779802578131, 0.986101160484812, 0.987627609921421],
                      [2.89581163838436, 0.991072950155688, 1.00352028156932],
                      [2.87555463689288, 0.989861755617231, 0.987186651340053],
                      [2.81997578277204, 0.985180606846278, 1.09475578600025]])
        print(dataset.evaluate(X) - Y)

    if 2 in WFG_list:
        print('--- WFG 2 ---')
        dataset = WFG2(10, 3)
        X = np.array([[1.51215634670685, 1.98046188620202, 2.17123205516798, 4.28272264683346, 1.67560302649847, 7.45865072083838, 10.3456568199683, 3.17408245839211, 17.5922307989805, 2.22789613281489],
                      [1.65724228844236, 3.75148713154759, 3.80920112440229, 2.04674050857133, 4.71394745021335, 8.0987099046684, 2.21089005303561, 4.37336956825761, 13.0245498011878, 7.09552477899624]])
        Y = np.array([[0.823269169947225, 1.21047059380468, 3.7645144707503],
                      [1.7835950584571, 0.472532451724152, 2.42582512027814]])
        print(dataset.evaluate(X) - Y)

    if 3 in WFG_list:
        print('--- WFG 3 ---')
        dataset = WFG3(10, 3)
        X = np.array([[1.38663349883148, 1.39095701793336, 1.00651400424944, 1.08124749578659, 3.08488862377431, 7.97168781965395, 7.76075416049597, 2.66837163922627, 5.08502619704711, 15.0825267506388],
                      [0.857279299089974, 2.90178703928795, 0.562662124363031, 1.62200945196832, 9.21877546951233, 9.18323121613658, 12.9452537606931, 9.4066087893712, 11.1373467719423, 11.1228130773289],
                      [0.972828288318792, 3.30043205456895, 1.59130876555149, 1.25130637703317, 9.72493287754122, 1.25826747229887, 5.77792241569192, 12.8549248376846, 13.5182364945479, 13.0712479627488],
                      [0.05787696298673, 2.73844307514897, 3.35587141949041, 2.27927716922222, 1.05777033975633, 6.60828440953838, 10.2633126093307, 10.7126816889197, 15.6329278855848, 3.34290219455068]])
        Y = np.array([[1.06023017837464, 2.04814437461039, 2.30521208140447],
                      [1.12327366377001, 1.2143893538016, 4.01028813045063],
                      [1.2509300110279, 1.18625072894685, 3.66233319316531],
                      [0.510007955859936, 0.523689192220033, 6.30235283702863]])
        print(dataset.evaluate(X) - Y)

    if 4 in WFG_list:
        print('--- WFG 4 ---')
        dataset = WFG4(10, 3)
        X = np.array([[1.13783890382196, 0.39981342549122, 2.57104400359446, 7.38059326152385, 0.18024697236177, 4.76511856888059, 4.94868612529733, 5.03603867466566, 1.57950371631846, 5.02059681386812],
                      [1.24461287529011, 3.47327010872662, 5.43623388146076, 7.94615451354421, 5.40004134634819, 1.01695950794045, 5.48432969385307, 1.62513156036691, 5.43756079090007, 16.845612279212],
                      [1.3483917307458, 3.40633721753899, 5.78151771907546, 2.63492324427142, 6.94416921281742, 2.88016099935476, 13.1911070274896, 2.97957306442058, 3.87758428471048, 11.4314968749729],
                      [1.04757801036099, 1.88111694935307, 2.00402875380894, 5.70334301516702, 4.55333751888472, 8.27804323815833, 6.40662120721998, 8.26948687327702, 0.229783478365364, 13.1661986496426]])
        Y = np.array([[0.706332603289956, 1.14447455569412, 6.03537463248557],
                      [0.963434680277201, 1.09292165133283, 6.05832165357606],
                      [1.07075915988437, 1.19846384116053, 5.82348456594644],
                      [0.334946480439561, 0.839576787685893, 6.19071079640542]])
        print(dataset.evaluate(X) - Y)

    if 5 in WFG_list:
        print('--- WFG 5 ---')
        dataset = WFG5(10, 3)
        X = np.array([[1.2658018033216, 3.18868341877624, 3.21674728712595, 2.08766437576511, 1.87500134447649, 9.21098472567939, 2.30814691679358, 1.25584817131949, 17.7385278296678, 8.30370524977232],
                      [0.188427418427115, 1.8744784818475, 0.633157170511586, 2.73768679269978, 1.38430792507739, 7.15562649914803, 3.38867613467205, 12.2754868226584, 16.3339183981048, 9.32069651971608],
                      [1.30733068788624, 3.23996382627474, 1.51298734605049, 0.151738627922504, 7.10136607495888, 3.49080201399634, 12.3541209340065, 11.1733430579877, 4.44885294202544, 4.28396803065948],
                      [1.92366471786647, 2.5530647566848, 3.12059081912017, 6.06933290222744, 2.93628043954894, 10.7610911050643, 6.94289071867055, 9.16338291470144, 11.2676918682856, 10.6531655297223]])
        Y = np.array([[1.34888749224017, 3.24939082730017, 4.14487961342243],
                      [1.45525021079554, 1.05768676090736, 5.88104647591294],
                      [1.28454039468884, 3.19897625033758, 4.37452809221445],
                      [0.886565707929169, 1.03136575496366, 6.5423541997296]])
        print(dataset.evaluate(X) - Y)

    if 6 in WFG_list:
        print('--- WFG 6 ---')
        dataset = WFG6(10, 3)
        X = np.array([[1.94501871330945, 1.86960168990496, 1.96989048063627, 3.06779485183013, 4.84162319219383, 4.75928430895196, 5.85331053617453, 13.2513660474365, 0.510286690310382, 18.6552194512127],
                      [1.6382070126678, 1.92461292772802, 0.334566489815851, 3.72761590768986, 9.41104341445888, 5.74945077283758, 12.9618313158316, 0.717866352348218, 2.64811675978753, 14.0007923108559],
                      [0.939109813727533, 1.6506765026082, 1.71570362522649, 3.45508684333413, 4.53345202891128, 6.5766957587481, 13.8121946947571, 7.14665236217724, 17.2382558951885, 18.4541611792558],
                      [1.34095292693237, 2.08669096099881, 4.28146989537444, 0.297012501508765, 6.9633337218723, 10.9762367807178, 3.81984309372725, 15.7681794973395, 8.90932551566349, 9.68168696095493]])
        Y = np.array([[2.1088891146428, 3.7368890934722, 1.0291775489388],
                      [2.00093495832413, 3.47839027226041, 2.36626738194932],
                      [1.59266942450381, 2.92495354073471, 5.22121627754014],
                      [1.99075376399652, 3.09350686154877, 3.68953206238677]])
        print(dataset.evaluate(X) - Y)

    if 7 in WFG_list:
        print('--- WFG 7 ---')
        dataset = WFG7(10, 3)
        X = np.array([[0.337549438578628, 1.55921659024094, 4.94553476741995, 1.6283190933529, 4.19639449341452, 2.66295392887256, 6.3802812867656, 2.50662348019979, 11.3208749535504, 13.7398730938539],
                      [0.487202151742022, 0.964386388124292, 0.741882978573107, 3.3763597352786, 4.67878671847323, 3.79175127742979, 2.80345466394784, 3.99691703515509, 12.596150343545, 10.389220448287],
                      [0.431067575693884, 0.313436740339362, 4.54058194298759, 6.63495882913101, 5.36173766848859, 2.56462047169617, 1.6572427810141, 12.4029269120389, 13.4746843302705, 1.79733800927311],
                      [0.866704508375235, 0.0563285441947447, 2.59927499409445, 4.61660841796661, 8.94030029806247, 11.8391383384409, 2.20248998409654, 3.91420272443612, 6.8389824559699, 9.59707339826101]])
        Y = np.array([[0.806046258534732, 1.40520487476877, 6.09953804366995],
                      [0.865223755462074, 2.15545034407944, 5.39011799683968],
                      [0.5998915531353, 2.0762744421912, 6.15862584417774],
                      [0.42389787520902, 3.07132814633459, 4.92165209479412]])
        print(dataset.evaluate(X) - Y)

"""
WFG_list = [1, 2, 3, 4, 5, 6, 7]
#WFG_list = [1, 2, 3, 4]
WFG_validation(WFG_list)
"""
n_vars = 10
n_objs = 4
dataset = WFG2(n_vars, n_objs)
X = lhs(n_vars, 4)
print(dataset.evaluate(X))
#"""

# 获取Pareto Front
# 画图