import numpy as np 
#import matplotlib.pyplot as plt

def gamma_j(j_plus_1):
    """
        $gamma_{j+1}$ defined in the recurrence relation in 
        Theorem 3.6.3. Used in the construction of the 
        tridiagonal matrix $J_n$.
        We use /j_plus_1/ to make notation consistent with the textbook.
    """
    j = j_plus_1 - 1
    gamma = j / np.sqrt(4*j*j - 1)
    return gamma

def construct_Jn(n):
    """
        Constructs the tridiagonal matrix $J_n$ (equation 3.6.19)
        which enables us to compute the weights and nodes we need.
        $delta_j$ is always 0, so we only need $gamma_j$.
    """
    J = np.zeros((n, n))
    for i in range(n-1):
        J[i  ][i+1] = gamma_j(i+2)
        J[i+1][i  ] = gamma_j(i+2)
    return J

def weights_and_nodes(n):
    """
        Returns the precomputed weights and nodes to be used 
        in the computation of Gaussian Quadrature using weight
        function 1. 
    """
    Jn = construct_Jn(n)
    eigval, eigvec = np.linalg.eig(Jn)

    # /eigvec/ is a matrix where the columns are unit norm 
    # eigenvectors of /Jn/, i-th column corresponding to /eigval[i]/.
    # The factor 2 comes from the fact that $(p_0, p_0) = 2$.

    nodes = eigval
    weights = 2 * np.square(eigvec[0])
    return (weights, nodes)

def eval_quadrature(f, a, b, n):
    """
        Evaluates the /n/-point Gaussian Quadrature of /f/
        on the interval [/a/, /b/] with the weight function 1.
    """
    if (a, b) != (-1, 1):
        # It is in general a bad idea to compare two floating point numbers
        # in this fashion, but here we do so to endorse change of variables.
        g = lambda t: f( (t+1) * (b-a) / 2 + a ) 
        factor = (b-a) / 2
    else :
        g = f
        factor = 1

    weights, nodes = weights_and_nodes(n)
    res = 0

    for i in range(n):
        res = res + weights[i] * g(nodes[i])
    
    return factor * res

def eval_composite_quad(f, a, b, n, sub):
    """
        Evaluates the composite Gaussian Quadrature of /f/
        on the interval [/a/, /b/] with the weight function 1.
        We use /n/-point quadrature on each subintervals, 
        and /sub/ is the number of subintervals we are using.
    """
    points = np.linspace(a, b, sub + 1)
    res = 0
    for i in range(sub):
        quad = eval_quadrature(f, points[i], points[i+1], n)
        res = res + quad
    return res

print()
#===============================#
 
print("Example 1")
a, b = -1, 1
f = lambda x: np.exp(x**2) * np.log(2 - x)

four_point_quad = eval_quadrature(f, a, b, 4)
two_point_comp_quad = eval_composite_quad(f, a, b, 2, 2)
print("Four point quadrature:")
print("\t", four_point_quad)
print("Composite two point quadrature on two subintervals:")
print("\t", two_point_comp_quad)

print()
#===============================#

print("Example 2")
a, b = 1, 3
f = lambda x: 1 / np.sqrt(x**4 + 1)

four_point_quad = eval_quadrature(f, a, b, 4)
two_point_comp_quad = eval_composite_quad(f, a, b, 2, 2)
print("Four point quadrature:")
print("\t", four_point_quad)
print("Composite two point quadrature on two subintervals:")
print("\t", two_point_comp_quad)
