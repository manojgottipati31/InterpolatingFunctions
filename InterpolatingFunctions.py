import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, solve, Eq
import sympy
import math
sympy.init_printing(use_latex=False)

#############################
# Helper Functions

def lagrange_polynomial(base_points):
    def basis(k, x):
        """ Calculate the k-th Lagrange basis polynomial at x. """
        result = 1
        for i, (xi, _) in enumerate(base_points):
            if i != k:
                result *= (x - xi) / (base_points[k][0] - xi)
        return result

    def polynomial(x):
        """ Evaluate the Lagrange polynomial at x. """
        return sum(basis(k, x) * yi for k, (_, yi) in enumerate(base_points))

    return polynomial

def newton_interpolation(base_points):
    n = len(base_points)
    # Initialize the divided difference table
    divided_diff_table = [[0 for _ in range(n)] for _ in range(n)]

    # Extract x and y values
    x_values = [point[0] for point in base_points]
    y_values = [point[1] for point in base_points]

    # Fill the first column of the table with y values
    for i in range(n):
        divided_diff_table[i][0] = y_values[i]

    # Compute the divided differences
    for j in range(1, n):
        for i in range(n - j):
            divided_diff_table[i][j] = (divided_diff_table[i + 1][j - 1] - divided_diff_table[i][j - 1]) / (x_values[i + j] - x_values[i])

    # Define the Newton polynomial function
    def polynomial(x):
        total_sum = divided_diff_table[0][0]
        for i in range(1, n):
            term = divided_diff_table[0][i]
            for j in range(i):
                term *= (x - x_values[j])
            total_sum += term
        return total_sum

    return polynomial


def cubic_poly(base_points):
    # Define symbols
    def get_coeff(base_points):
        x = symbols('x')
        a, b, c, d = symbols('a b c d', cls=sympy.IndexedBase)

        # Set up the system of equations based on the polynomial passing through the base points
        equations = []

        # Construct the spline polynomials
        spline_polynomials = []
        for i in range(len(base_points) - 1):
            xi, yi = base_points[i]
            xi1, yi1 = base_points[i+1]
            poly = a[i]*x**3 + b[i]*x**2 + c[i]*x + d[i]
            spline_polynomials.append(poly)
            # Equations for the polynomial passing through the base points
            equations.append(Eq(poly.subs(x, xi), yi))
            equations.append(Eq(poly.subs(x, xi1), yi1))

        # Derivatives must be equal at the internal points
        for i in range(len(base_points) - 2):
            # First derivative continuity
            equations.append(Eq(spline_polynomials[i].diff(x).subs(x, base_points[i+1][0]),
                                spline_polynomials[i+1].diff(x).subs(x, base_points[i+1][0])))
            # Second derivative continuity
            equations.append(Eq(spline_polynomials[i].diff(x, x).subs(x, base_points[i+1][0]),
                                spline_polynomials[i+1].diff(x, x).subs(x, base_points[i+1][0])))

        # Boundary conditions for natural spline (second derivative at end points is zero)
        equations.append(Eq(spline_polynomials[0].diff(x, x).subs(x, base_points[0][0]), 0))
        equations.append(Eq(spline_polynomials[-1].diff(x, x).subs(x, base_points[-1][0]), 0))

        # Solve the system of equations for the coefficients
        solutions = solve(equations)

        # Output the spline polynomials with found coefficients
        spline_polynomials_with_coefficients = [sp.subs(solutions) for sp in spline_polynomials]
        numerical_solutions = {var: val.evalf() for var, val in solutions.items()}
        final_dict = {}
        for key, value in numerical_solutions.items():
            final_dict[str(key)] = float(value)
        return final_dict

    def find_interval(num, sorted_list):
        for i in range(len(sorted_list) - 1):
            if sorted_list[i] <= num < sorted_list[i + 1]:
                s = sorted_list[i]
                b = sorted_list[i + 1]
                s_id = sorted_list.index(s)
                b_id = sorted_list.index(b)
                return s_id,b_id 
        return None

    def polynomial(x_val_list):
        coefficients = get_coeff(base_points)
        fx_list = []
        for i in range(len(x_val_list)):
            x_val = x_val_list[i]
            if x_val!=base_points[-1][0]:
                small, big = find_interval(x_val, sorted([i for i,j in base_points]))
            else: 
                small = len(base_points)-2
            a = coefficients['a['+str(small)+']']
            b = coefficients['b['+str(small)+']']
            c = coefficients['c['+str(small)+']']
            d = coefficients['d['+str(small)+']']
            fx = a*(x_val**3) + b*(x_val**2) + c*(x_val**1) + d*(x_val**0)
            fx_list.append(fx)
        return fx_list

    return polynomial

##############################################################################
# 1.
# Helper functions (invert_matrix, mul_matA_matB, get_coeff)
def invert_matrix(A):
    n = len(A)
    
    if any(len(row) != n for row in A):
        return ValueError("not sq mat")

    augmented = [row + [0] * i + [1] + [0] * (n - i - 1) for i, row in enumerate(A)]

    for i in range(n):

        pivot = augmented[i][i]
        if pivot == 0:
            return ValueError("singular mat")
        for j in range(2*n):
            augmented[i][j] /= pivot
        for k in range(n):
            if k != i:
                ratio = augmented[k][i]
                for j in range(2*n):
                    augmented[k][j] -= ratio * augmented[i][j]

    inverse = [row[n:] for row in augmented]
    return inverse


def mul_matA_matB(mat_A, mat_B):
    rowsA = len(mat_A)
    colsA = len(mat_A[0])
    rowsB = len(mat_B)
    colsB = len(mat_B[0])

    if colsA != rowsB:
        return 'error'
    result = [[0 for _ in range(colsB)] for _ in range(rowsA)]

    for i in range(rowsA):
        for j in range(colsB):
            for k in range(colsA):
                result[i][j] += mat_A[i][k] * mat_B[k][j]

    return result


# X = [1,2,3,4]
# Y = [[1],[7],[26],[69]]
def get_coeff(X,Y):

    mat_X = []
    for i in range(len(X)):
        row = [X[i]**(j+1) for j in range(len(X)-1)]
        row.insert(0, 1)
        mat_X.append(row)

    mat_X_transpose =  [[mat_X[j][i] for j in range(len(mat_X))] for i in range(len(mat_X[0]))]

    X_transp_X = mul_matA_matB(mat_X_transpose, mat_X)
    X_transp_X_inv = invert_matrix(X_transp_X)
    X_transp_yObs = mul_matA_matB(mat_X_transpose, Y)
    beta = mul_matA_matB(X_transp_X_inv, X_transp_yObs)

    return beta


def intrapolate_Y(X,beta):
    return [sum([(beta[i][0]*(X[j]**i)) for i in range(len(X))]) for j in range(len(X))]
    # return [beta[0][0] + beta[1][0]*x + beta[2][0]*(x**2) + beta[3][0]*(X[i]**3) for i in range(len(X))]


# test HW03 problem 4
X_data = [1,-1,3]
Y_data = [[0],[-3],[-4]]
betas = get_coeff(X_data,Y_data)
est_Y = intrapolate_Y(X_data,betas)
print("Actual Y:",Y_data)
print("Intrapolated Y:",est_Y)
print('betas b0,b1..:',betas)

# test HW03 problem 6
X_data = [1,-2,0,3,-1,7]
Y_data = [[-2],[-56],[-2],[4],[-16],[376]]
betas = get_coeff(X_data,Y_data)
est_Y = intrapolate_Y(X_data,betas)
print("Actual Y:",Y_data)
print("Intrapolated Y:",est_Y)
print('betas b0,b1..:',betas)

##############################################################################
# 2.
def plot_func(X,Y,cs,lg):
# Plot the spline and the original data points
    xs = np.linspace(X.min(), X.max(), 100)
    ys_cs = cs(xs)
    ys_lg = [lg(x) for x in xs]

    plt.figure(figsize=(8, 4))
    plt.plot(X, Y, 'o', label='data points')
    plt.plot(xs, ys_cs, label='cubic spline',color='blue')
    plt.plot(xs, ys_lg, label='lagrange poly',color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


# modify y5
def modify_fifth_element(Y, epsilon):
    Y[4] = 5 + epsilon
    return Y


X_data = [i+1 for i in range(10)]
Y_data = [[i+1] for i in range(10)]
Y_data_list = [Y_data[i][0] for i in range(len(Y_data))]


# no change in Y
# cs_obj = CubicSpline(X_data, Y_data_list, bc_type='natural')
lg_obj = lagrange_polynomial(list(zip(X_data,Y_data_list)))
cs_obj = cubic_poly(list(zip(X_data,Y_data_list)))
plot_func(np.array(X_data),np.array(Y_data_list),cs_obj,lg_obj)


# for eps = 0.1
Y_data_list_eps = modify_fifth_element(Y_data_list,0.1)
# cs_obj = CubicSpline(X_data, Y_data_list_eps, bc_type='natural')
cs_obj = cubic_poly(list(zip(X_data,Y_data_list)))
lg_obj = lagrange_polynomial(list(zip(X_data,Y_data_list)))
plot_func(np.array(X_data),np.array(Y_data_list_eps),cs_obj,lg_obj)


# for eps = 0.3
Y_data_list_eps = modify_fifth_element(Y_data_list,0.3)
cs_obj = cubic_poly(list(zip(X_data,Y_data_list)))
lg_obj = lagrange_polynomial(list(zip(X_data,Y_data_list)))
plot_func(np.array(X_data),np.array(Y_data_list_eps),cs_obj,lg_obj)


# for eps = 0.6
Y_data_list_eps = modify_fifth_element(Y_data_list,0.6)
cs_obj = cubic_poly(list(zip(X_data,Y_data_list)))
lg_obj = lagrange_polynomial(list(zip(X_data,Y_data_list)))
plot_func(np.array(X_data),np.array(Y_data_list_eps),cs_obj,lg_obj)


# comment: cubic = lagrange for epsilon = 0. as epsilon increases, the
# divergence in the intrapolating ploynomials increases.
# lg_obj = lagrange(X_data, Y_data_list)

##############################################################################
# 5. 
def f(x):
    return 1/(1+(x**2))
    # return 1/(1+(x**2))

def get_base_point(a,b,n,i):
    return a + ((b-a)/n)*i
    

def get_data(min_val,max_val,n):
    # X = np.linspace(min_val, max_val, n+1)
    X = [get_base_point(min_val,max_val,n,i) for i in range(n+1)]
    Y = [f(x) for x in X]

    return X, Y

def plot_func(X,Y,lg):
# Plot the spline and the original data points

    xs = np.linspace(X.min(), X.max(), 100)
    ys_act = [f(x) for x in xs]
    ys_lg = lg(xs)
    er = [(ys_act[i] - ys_lg[i]) for i in range(len(ys_act))]
    plt.figure(figsize=(8, 4))
    plt.plot(list(X), list(Y), 'o', label='data points')
    plt.plot(list(xs), list(ys_act), '-', label='actual f',color='yellow')
    plt.plot(list(xs), list(ys_lg), label='lagrange poly',color='blue')
    plt.plot(list(xs), list(er), label='error',color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

X_data, Y_data = get_data(-5,5,10)
lg_obj = lagrange_polynomial(list(zip(X_data,Y_data)))
plot_func(np.array(X_data),np.array(Y_data),lg_obj)

# comments: largest error approx @ x=-4.75 and x = 4.75 with
# approx error = 4

##############################################################################
# 6. Using Chebyshev points

def get_base_point(a,b,n,i):
    return (a+b)/2 + ((b-a)/2)*(math.cos(math.radians( ((2*i+1) / 2*(n+1))*(math.pi) )))
    
def get_data(min_val,max_val,n):
    X = [get_base_point(min_val,max_val,i,n) for i in range(n+1)]
    Y = [f(x) for x in X]
    return X, Y

X_data, Y_data = get_data(-5,5,10)
lg_obj = lagrange_polynomial(list(zip(X_data,Y_data)))
plot_func(np.array(X_data),np.array(Y_data),lg_obj)

# summary: when the data points are taken at local max/min, the error
# seems to be smaller. This may be because the  intrapolating poly
# is able to capture direction change leaing to a close hugging of
# the actual function
##############################################################################

# 8.
def plot_func(X,Y,cs,lg,new):
# Plot the spline and the original data points

    xs = np.linspace(X.min(), X.max(), 100)
    # ys_act = [f(x) for x in xs]
    ys_lg = lg(xs)
    ys_cs = cs(xs)
    ys_new = new(xs)
    # er = [(ys_act[i] - ys_lg[i]) for i in range(len(ys_act))]
    plt.figure(figsize=(8, 4))
    plt.plot(list(X), list(Y), 'o', label='data points')
    plt.plot(list(xs), list(ys_cs), '-', label='cubic',color='yellow')
    plt.plot(list(xs), list(ys_lg), label='lagrange',color='blue')
    plt.plot(list(xs), list(ys_new), label='newton',color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

aapl = pd.read_csv(r'/Users/manojgottipati/Desktop/MQF/Rutgers Coursework/Term3/NumAl/Assignments/AAPL.csv')
aapl['Date'] = [i+1 for i in range(len(aapl))]
base_points = list(zip(list(aapl['Date']),list(aapl['Close'])))
# Y_data_list_eps = modify_fifth_element(Y_data_list,0.3)
cs_obj = cubic_poly(base_points)
lg_obj = lagrange_polynomial(base_points)
new_obj = newton_interpolation(base_points)
plot_func(np.array(list(aapl['Date'])),np.array(list(aapl['Close'])),cs_obj,lg_obj,new_obj)

##############################################################################
# 9.
def find_interval(num, sorted_list):
    for i in range(len(sorted_list) - 1):
        if sorted_list[i] <= num < sorted_list[i + 1]:
            s = sorted_list[i]
            b = sorted_list[i + 1]
            s_id = sorted_list.index(s)
            b_id = sorted_list.index(b)
            return s_id,b_id 
    return None


def hermit(X,degree):
    x = symbols('x')
    f_values = [np.exp(xi**2 / 10) for xi in X]
    f_prime_values = [np.exp(xi**2 / 10)*(x/5) for xi in X]
    base_points = list(zip(X,f_values))

    deg=degree-1
    spline_polynomials = []
    phi_list = []
    psi_list = []
    coeff_dict = {}
    for i in range(1,len(base_points)):
        xi = base_points[i][0]
        xi1 = base_points[i-1][0]
        yi = f_values[i]
        y_primei = f_prime_values[i]
        delta_x = xi - xi1

        
        b = (1-2*(x-xi)/delta_x) * (1/(delta_x**2))
        c = (1/(delta_x**2))
        phi = ((x-xi1)**deg) * (b)
        psi = ((x-xi1)**deg) * (x-xi) * c


        wx = 0
        for d in range(degree):
            wx += phi.subs(x, (x-xi)/delta_x)*yi + psi.subs(x, (x-xi)/delta_x)*delta_x*y_primei
        spline_polynomials.append(wx)

        coeff_dict['a'+str(i)] = (wx.diff(x).diff(x).diff(x).subs(x,0))/6
        coeff_dict['b'+str(i)] = (wx.diff(x).diff(x).subs(x,0))/2
        coeff_dict['c'+str(i)] = (wx.diff(x).subs(x,0))/1
        coeff_dict['d'+str(i)] = (wx.subs(x,0))

    return coeff_dict

def intrapolate(X,x,degree,cubic_dict):
    small, big = find_interval(x, sorted(X))
    return cubic_dict['a'+str(big)]*(x**degree) + cubic_dict['b'+str(big)]*(x**(degree-1)) + cubic_dict['c'+str(big)]*(x**(degree-2)) + cubic_dict['d'+str(big)]*(x**(degree-3))


X_list = [1, 1.5]
cubic_dict = hermit(X_list,3)
intrapolate(X_list,1,3,cubic_dict)

X_list = [1, 2, 3]
cubic_dict = hermit(X_list,5)

