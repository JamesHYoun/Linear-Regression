import matplotlib.pyplot as plt
import numpy as np
from sympy import *
import time
from sympy.parsing.sympy_parser import rationalize


class LinReg:

    def __init__(self, data, learn_rate, reg_coeff, eps):
        self.num_weights = len(data[0])
        self.num_data = len(data[1])
        self.data_x = data[0]
        self.data_y = data[1]
        self.bias = 0.5
        self.weights = np.array([0.5 for _ in range(self.num_weights)])
        self.learn_rate = learn_rate
        self.reg_coeff = reg_coeff
        self.eps = eps

    def hypo(self):
        return self.bias + np.matmul(self.weights, self.data_x)

    def cost_func(self):
        diff = self.hypo() - self.data_y
        cost = (1/(2*self.num_data)) * np.dot(diff, diff) + self.reg_coeff * np.dot(self.weights, self.weights)
        return cost

    def gradient(self):
        diff = self.hypo() - self.data_y
        grad_b = (1/self.num_data) * (np.matmul(np.ones(self.num_data), np.array([diff]).T).T)[0]
        grad_w = (1/self.num_data) * (np.matmul(self.data_x, np.array([diff]).T).T)[0] + 2 * self.reg_coeff * self.weights
        return grad_b, grad_w

    def grad_desc(self):
        curr_cost = self.cost_func()
        prev_cost = 2 * curr_cost
        i = 0
        while abs(curr_cost - prev_cost) > self.eps * prev_cost:
            grad = self.gradient()
            self.bias -= self.learn_rate * grad[0]
            self.weights -= self.learn_rate * grad[1]
            prev_cost = curr_cost
            curr_cost = self.cost_func()

    def normalize(self):
        self.data_x_org = self.data_x
        data_x = [[] for _ in range(self.num_weights)]
        params = [_ for _ in range(self.num_weights)]
        for i in range(self.num_weights):
            row = self.data_x[i]
            mean, std = np.mean(row), np.std(row)
            data_x[i] = [(x - mean) / std for x in row]
            params[i] = (mean, std)
        self.data_x = data_x
        self.params = params

    def revert(self):
        diff = 0
        for i in range(self.num_weights):
            mean, std = self.params[i][0], self.params[i][1]
            self.weights[i] /= std
            diff += self.weights[i] * mean
        self.bias -= diff
        self.data_x = self.data_x_org

    def main(self):
        self.normalize()
        self.grad_desc()
        self.revert()
        return self.bias, self.weights


class LearnRate:

    def __init__(self, spread, data, reg_coeff, eps, num_iter=1):
        self.spread = spread
        self.data = data
        self.reg_coeff = reg_coeff
        self.eps = eps
        self.num_iter = num_iter

    def get_ratio(self, learn_rate):
        linreg = LinReg(self.data, learn_rate, self.reg_coeff, self.eps)
        linreg.normalize()
        cost_initial = linreg.cost_func()
        for i in range(self.num_iter):
            grad = linreg.gradient()
            linreg.bias -= learn_rate * grad[0]
            linreg.weights -= learn_rate * grad[1]
        linreg.revert()
        return linreg.cost_func() / cost_initial

    def get_learn_rate(self):
        rate = 0
        min_rate, max_rate = self.spread[0], self.spread[1]
        while abs(max_rate - min_rate) > 1e-9 * (self.spread[1] - self.spread[0]):
            rate = (min_rate + max_rate) / 2
            left = rate - (max_rate - min_rate) * 0.1
            right = rate + (max_rate - min_rate) * 0.1
            ratio = self.get_ratio(10**rate)
            ratio_l = self.get_ratio(10**left)
            ratio_r = self.get_ratio(10**right)
            if ratio_l < ratio and ratio < ratio_r:
                max_rate = rate
            if ratio_l > ratio and ratio > ratio_r:
                min_rate = rate
            if ratio_l >= ratio and ratio <= ratio_r:
                min_rate = left
                max_rate = right
        return 10**rate


# SET-UP POLYNOMIAL

def polynomial_matrix(x_list, order):
    return np.array([[x**i for x in x_list] for i in range(1, order+1)])

def get_data_y(func, x_list, mean, std):
    data_y = np.array([0.0 for _ in range(len(x_list))])
    for i in range(len(x_list)):
        data_y[i] = func.subs(x, x_list[i]) + np.random.normal(mean, std)
    return data_y

order = 3
num_weights = order
num_data = 11
x_start = -5
x_end = 5

x_list = np.linspace(x_start, x_end, num=num_data)
data_x = polynomial_matrix(x_list, order)

x = symbols('x')
f = 0
coeffs = [int(np.random.normal(0, 15)) for _ in range(order + 1)]
print(f'coefficients: {coeffs}')
for i in range(order + 1):
    f += coeffs[i] * x**i
data_y = get_data_y(f, x_list, 0, 5 * order**2)


# LINEAR REGRESSION

eps = 10**(-6)
reg_coeff = 1e-2
learn_rate = LearnRate((-12, 0), (data_x, data_y), eps, 10).get_learn_rate()
# learn_rate = 2e-6
print(f'learn_rate: {learn_rate}')
linreg = LinReg((data_x, data_y), learn_rate, reg_coeff, eps)
start_time = time.time()
bias, weights = linreg.main()
end_time = time.time()
print(f'Time taken: {end_time - start_time}')
print(f'bias: {bias}\nweights: {weights}')


# PLOTTING

plt.plot(x_list, data_y, 'o')
plt.plot(x_list, linreg.hypo())
plt.show()
