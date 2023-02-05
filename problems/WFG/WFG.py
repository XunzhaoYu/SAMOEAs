# -*- coding: UTF-8 -*-
import numpy as np
from pyDOE import lhs


desired_width = 160
np.set_printoptions(linewidth=desired_width)
np.set_printoptions(precision=8, suppress=True)

""" Written by Xun-Zhao Yu (yuxunzhao@gmail.com). Last update: 2023-Feb-04.
WFG Multi-/Many-Objective Optimization Benchmark functions.

Example:
dataset = WFG1(config)
x, y = dataset.sample(n_samples=100)

Return:
x shape: (n_samples, n_vars)
y shape: (n_samples, n_objs)
"""
class WFG:
    def __init__(self, config, k=None, l=None):
        self.n_vars = config['x_dim']
        self.n_objs = config['y_dim']
        self.lowerbound = np.zeros(self.n_vars)
        self.upperbound = 2 * np.arange(1, self.n_vars + 1).astype(float)

        self.S = np.arange(2, 2 * self.n_objs + 1, 2).astype(float)
        self.A = np.ones(self.n_objs - 1)

        if k:  # k should be dividable by (n_objs - 1)
            self.k = k
        else:
            self.k = self.n_objs-1

        if l:
            self.l = l
        else:
            self.l = self.n_vars - self.k
        print("k:", self.k, ".  l:", self.l)

        self.validate(self.l, self.k, self.n_objs)

    def get_bounds(self, upper=True):
        if upper:
            return self.upperbound.copy()
        else:
            return self.lowerbound.copy()

    def validate(self, l, k, n_obj):
        if n_obj < 2:
            raise ValueError('WFG problems must have two or more objectives.')
        if not k % (n_obj - 1) == 0:
            raise ValueError('Position parameter (k) must be divisible by number of objectives (n_objs) minus one.')
        if (k + l) < n_obj:
            raise ValueError('Sum of distance and position parameters must be greater than num. of objs. (k + l >= n_objs).')

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

    def evaluate(self, x):
        return

    def sample(self, n_samples):
        """
        :param n_samples: The number of samples collected from a WFG function. Type: int
        :return: The sampled data x and evaluated fitness y.
        """
        x = lhs(self.n_vars, n_samples)
        y = self.evaluate(x)
        return x, y


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
        """
        :param x: The set of samples to be evaluated. Type: 2darray. Shape: (n_samples, n_vars).
        :return: The set of evaluated samples. Type: 2darray. Shape: (n_samples, n_objs).
        """
        y = x / self.upperbound
        y = WFG1.t1(y, self.n_vars, self.k)
        y = WFG1.t2(y, self.n_vars, self.k)
        y = WFG1.t3(y, self.n_vars)
        y = WFG1.t4(y, self.n_objs, self.n_vars, self.k)
        y = self._post(y, self.A)
        h = [_shape_convex(y[:, :-1], m + 1) for m in range(self.n_objs - 1)]
        h.append(_shape_mixed(y[:, 0], alpha=1.0, A=5.0))
        return self._calculate(y, self.S, h)

    def _rand_optimal_position(self, n):
        return np.power(np.random.random((n, self.k)), 50.0)


class WFG2(WFG):
    def validate(self, l, k, n_obj):
        super().validate(l, k, n_obj)
        if validate_wfg2_wfg3(l):
            self.k = 2 * self.k
            self.l = self.n_vars - self.k
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
        """
        :param x: The set of samples to be evaluated. Type: 2darray. Shape: (n_samples, n_vars).
        :return: The set of evaluated samples. Type: 2darray. Shape: (n_samples, n_objs).
        """
        y = x / self.upperbound
        y = WFG1.t1(y, self.n_vars, self.k)
        y = WFG2.t2(y, self.n_vars, self.k)
        y = WFG2.t3(y, self.n_objs, self.n_vars, self.k)
        y = self._post(y, self.A)

        h = [_shape_convex(y[:, :-1], m + 1) for m in range(self.n_objs - 1)]
        h.append(_shape_disconnected(y[:, 0], alpha=1.0, beta=1.0, A=5.0))

        return self._calculate(y, self.S, h)


class WFG3(WFG):
    def __init__(self, config):
        super().__init__(config)
        self.A[1:] = 0

    def validate(self, l, k, n_obj):
        super().validate(l, k, n_obj)
        if validate_wfg2_wfg3(l):
            self.k = 2 * self.k
            self.l = self.n_vars - self.k
            print("new k:", self.k, ".  l:", self.l)

    def evaluate(self, x):
        """
        :param x: The set of samples to be evaluated. Type: 2darray. Shape: (n_samples, n_vars).
        :return: The set of evaluated samples. Type: 2darray. Shape: (n_samples, n_objs).
        """
        y = x / self.upperbound
        y = WFG1.t1(y, self.n_vars, self.k)
        y = WFG2.t2(y, self.n_vars, self.k)
        y = WFG2.t3(y, self.n_objs, self.n_vars, self.k)
        y = self._post(y, self.A)

        h = [_shape_linear(y[:, :-1], m + 1) for m in range(self.n_objs)]

        return self._calculate(y, self.S, h)


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
        """
        :param x: The set of samples to be evaluated. Type: 2darray. Shape: (n_samples, n_vars).
        :return: The set of evaluated samples. Type: 2darray. Shape: (n_samples, n_objs).
        """
        y = x / self.upperbound
        y = WFG4.t1(y)
        y = WFG4.t2(y, self.n_objs, self.k)
        y = self._post(y, self.A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_objs)]

        return self._calculate(y, self.S, h)


class WFG5(WFG):
    @staticmethod
    def t1(x):
        return _transformation_param_deceptive(x, A=0.35, B=0.001, C=0.05)

    def evaluate(self, x):
        """
        :param x: The set of samples to be evaluated. Type: 2darray. Shape: (n_samples, n_vars).
        :return: The set of evaluated samples. Type: 2darray. Shape: (n_samples, n_objs).
        """
        y = x / self.upperbound
        y = WFG5.t1(y)
        y = WFG4.t2(y, self.n_objs, self.k)
        y = self._post(y, self.A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_objs)]

        return self._calculate(y, self.S, h)


class WFG6(WFG):
    @staticmethod
    def t2(x, m, n, k):
        gap = k // (m - 1)
        t = [_reduction_non_sep(x[:, (m - 1) * gap: (m * gap)], gap) for m in range(1, m)]
        t.append(_reduction_non_sep(x[:, k:], n - k))
        return np.column_stack(t)

    def evaluate(self, x):
        """
        :param x: The set of samples to be evaluated. Type: 2darray. Shape: (n_samples, n_vars).
        :return: The set of evaluated samples. Type: 2darray. Shape: (n_samples, n_objs).
        """
        y = x / self.upperbound
        y = WFG1.t1(y, self.n_vars, self.k)
        y = WFG6.t2(y, self.n_objs, self.n_vars, self.k)
        y = self._post(y, self.A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_objs)]
        return self._calculate(y, self.S, h)


class WFG7(WFG):
    @staticmethod
    def t1(x, k):
        for i in range(k):
            aux = _reduction_weighted_sum_uniform(x[:, i + 1:])
            x[:, i] = _transformation_param_dependent(x[:, i], aux)
        return x

    def evaluate(self, x):
        """
        :param x: The set of samples to be evaluated. Type: 2darray. Shape: (n_samples, n_vars).
        :return: The set of evaluated samples. Type: 2darray. Shape: (n_samples, n_objs).
        """
        y = x / self.upperbound
        y = WFG7.t1(y, self.k)
        y = WFG1.t1(y, self.n_vars, self.k)
        y = WFG4.t2(y, self.n_objs, self.k)
        y = self._post(y, self.A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_objs)]

        return self._calculate(y, self.S, h)


class WFG8(WFG):
    @staticmethod
    def t1(x, n, k):
        ret = []
        for i in range(k, n):
            aux = _reduction_weighted_sum_uniform(x[:, :i])
            ret.append(_transformation_param_dependent(x[:, i], aux, A=0.98 / 49.98, B=0.02, C=50.0))
        return np.column_stack(ret)

    def evaluate(self, x):
        """
        :param x: The set of samples to be evaluated. Type: 2darray. Shape: (n_samples, n_vars).
        :return: The set of evaluated samples. Type: 2darray. Shape: (n_samples, n_objs).
        """
        y = x / self.upperbound
        y[:, self.k:self.n_vars] = WFG8.t1(y, self.n_vars, self.k)
        y = WFG1.t1(y, self.n_vars, self.k)
        y = WFG4.t2(y, self.n_objs, self.k)
        y = self._post(y, self.A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_objs)]

        return self._calculate(y, self.S, h)

    def _positional_to_optimal(self, K):
        k, l = self.k, self.l

        for i in range(k, k + l):
            u = K.sum(axis=1) / K.shape[1]
            tmp1 = np.abs(np.floor(0.5 - u) + 0.98 / 49.98)
            tmp2 = 0.02 + 49.98 * (0.98 / 49.98 - (1.0 - 2.0 * u) * tmp1)
            suffix = np.power(0.35, np.power(tmp2, -1.0))

            K = np.column_stack([K, suffix[:, None]])

        ret = K * (2 * (np.arange(self.n_vars) + 1))
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
        """
        :param x: The set of samples to be evaluated. Type: 2darray. Shape: (n_samples, n_vars).
        :return: The set of evaluated samples. Type: 2darray. Shape: (n_samples, n_objs).
        """
        y = x / self.upperbound
        y[:, :self.n_vars - 1] = WFG9.t1(y, self.n_vars)
        y = WFG9.t2(y, self.n_vars, self.k)
        y = WFG9.t3(y, self.n_objs, self.n_vars, self.k)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_objs)]

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

        ret = X * (2 * (np.arange(self.n_vars) + 1))
        return ret


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
        print("In WFG2/WFG3 the distance-related parameter (l) must be divisible by 2.")
        print("Set new k and l ...")
        return True
    else:
        return False


def correct_to_01(X, epsilon=1.0e-10):
    X[np.logical_and(X < 0, X >= 0 - epsilon)] = 0
    X[np.logical_and(X > 1, X <= 1 + epsilon)] = 1
    return X


