import numpy as np 
import matplotlib.pyplot as plt

def interval_length(knots):
    """ 
        Computes the length of each interval [x_{j-1}, x_j]. 
        For convenience in indexing we pad h[0] = 0.
    """
    h = knots - (np.roll(knots, 1))
    h[0] = 0
    return h

def draw_graph(coeff_matrix, knots):
    n = len(knots) - 1
    S = np.array([])
    x = np.array([])
    for j in range((n-1)+1):
        xj = knots[j]
        coeffs = coeff_matrix[j]
        xSample = np.linspace(xj, knots[j+1], 50)
        res = coeffs[3]
        for i in range(2, 0-1, -1):
            res = res * (xSample - xj) + coeffs[i] 
        x = np.hstack((x, xSample))
        S = np.hstack((S, res))
    return x, S
            
def find_idx(knots, z):
    n = len(knots) - 1
    if z < knots[0] or z > knots[-1]:
        raise KeyError
    left = 0
    right = n
    while (right - left) > 1 :
        mid = (left + right) // 2
        if knots[mid] > z :
            right = mid
        else :
            left = mid
    return left            

def eval_p(coeff_matrix, knots, z):
    """ 
       
    """
    j = find_idx(knots, z)
    xj = knots[j]
    coeffs = coeff_matrix[j]

    res = coeffs[3]
    for i in range(2, 0-1, -1):
        res = res * (z - xj) + coeffs[i] 
    
    return res

def error_analysis(f, coeff_matrix, knots, targets):
    errors = 0 * targets
    for i in range(len(targets)):
        z = targets[i]
        spline_value = eval_p(coeff_matrix, knots, z)
        true_value = f(z)
        errors[i] = true_value - spline_value
    return errors

def moment_relation_scalar(y, h):
    n = len(h) - 1
    d = np.zeros(n+1)
    for j in range(1, (n-1)+1):
        divdiff1 = (y[j+1] - y[j  ]) / h[j+1]
        divdiff0 = (y[j  ] - y[j-1]) / h[j  ]
        d[j] = 6 * (divdiff1 - divdiff0) / (h[j] + h[j+1])
    return d

def moment_relation_matrix(h):
    """
        
    """
    n = len(h) - 1
    A = 2 * np.eye(n+1)
    for j in range(1, (n-1)+1):
        lambda_j = h[j+1] / (h[j] + h[j+1])
        mu_j = 1 - lambda_j
        A[j][j-1] = mu_j
        A[j][j+1] = lambda_j
    
    return A
   
def compute_coeffs(M, y, h):
    n = len(h) - 1
    coeff_matrix = np.zeros((n , 4))
    for j in range((n-1)+1):
        alpha_j = y[j]
        gamma_j = (1/2) * M[j]
        delta_j = (M[j+1] - M[j]) / (6 * h[j+1])
        tmp1 = (y[j+1] - y[j]) / h[j+1]
        tmp0 = (2*M[j] + M[j+1]) * h[j+1] / 6
        beta_j = tmp1 - tmp0
        coeffs = np.array([alpha_j, beta_j, gamma_j, delta_j])
        coeff_matrix[j] = coeffs

    return coeff_matrix
    

fig = plt.figure()    # Preparation to plot functions

a, b = -1, 1
f = lambda x: 1 / (1 + 25 * x**2)

targets = np.linspace(a, b, 40+1)
n_list = [4, 10, 20]

for n in n_list:

    knots = np.linspace(a, b, n + 1)
    y = f(knots)
    h = interval_length(knots)

    A = moment_relation_matrix(h)
    d = moment_relation_scalar(y, h)

    M = np.linalg.solve(A, d)

    coeff_matrix = compute_coeffs(M, y, h)
    print("Coefficients of the cubic spline when n = %d: " %(n))
    print(coeff_matrix, end = "\n\n")

    errors = error_analysis(f, coeff_matrix, knots, targets)

    plt.plot(targets, errors, label = "n = %d" %(n))
    plt.legend()
   
plt.title('Error Plots')
plt.show()    # Show all plotted graphs 
