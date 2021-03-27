import numpy as np 
import matplotlib.pyplot as plt

def numsteps(a, b, h):
    """ 
        Compute the number of steps. There are (numsteps + 1) node points.
    """
    return( round((b-a)/h) )

def get_divdiff(f, absc):
    """
        Compute the divided difference scheme table. Returns it as an array.
    """
    x = absc
    fk = f(x)

    arr = np.zeros((k+1, k+1))
    arr[0] = fk
    for i in range(1, k + 1):
        for j in range(i, k + 1):
            arr[i][j] = (arr[i-1][j] - arr[i-1][j-1]) / (x[j] - x[j-i])

    return arr

def eval_p(divdiff, absc, z):
    """ 
        Evaluates the interpolating polynomial at a given point z, from the
        divided difference table and abscissae using Horner's scheme.
    """
    x = absc
    k = np.shape(divdiff)[0] - 1
    res = divdiff[k][k]
    for i in range(k-1, 0 - 1, -1):
        res = res * (z-x[i]) + divdiff[i][i]
    return res

def construct_interp_poly(divdiff, absc):
    """
        Constructs a list containing the coefficients of the interpolating
        polynomial which can be computed from the divided difference scheme.
        The i-th element in the list is the i-th degree term coefficient.
    """
    x = absc
    k = np.shape(divdiff)[0] - 1
    p = [divdiff[k][k]]
    for i in range(k-1, 0 - 1, -1):
        p = [0] + p
        for j in range(k-i):
            p[j] = p[j] - x[i] * p[j+1]
        p[0] = p[0] + divdiff[i][i]
    return p
    

fig = plt.figure()    # Preparation to plot functions

###-------------------------------------###
###----------- First Example -----------###
###-------------------------------------###

a, b, h = 0, 2, 0.25
f = lambda x: np.sin(x)

k = numsteps(a, b, h)
abscissa = np.linspace(a, b, k + 1)
divdiff = get_divdiff(f, abscissa)

p1 = construct_interp_poly(divdiff, abscissa)
print("Coefficients of the interpolating polynomial in Example 1 :")
print(p1); print()

xSample = np.linspace(a, b, 250)    # Sample points to plot
fSample = f(xSample)
pSample = eval_p(divdiff, abscissa, xSample)
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(xSample, fSample, label = 'f')
ax1.plot(xSample, pSample, label = 'p')
ax1.set_title('Example 1')
ax1.legend()

###-------------------------------------###
###----------- Second Example ----------###
###-------------------------------------###

a, b, h = -5, 5, 1
f = lambda x: 1/(1+x**2)

k = numsteps(a, b, h)
abscissa = np.linspace(a, b, k + 1)
divdiff = get_divdiff(f, abscissa)

p2 = construct_interp_poly(divdiff, abscissa)
print("Coefficients of the interpolating polynomial in Example 2 :")
print(p2); print()

xSample = np.linspace(a, b, 250)    # Sample points to plot
fSample = f(xSample)
pSample = eval_p(divdiff, abscissa, xSample)
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(xSample, fSample, label = 'f')
ax2.plot(xSample, pSample, label = 'p')
ax2.set_title('Example 2')
ax2.legend()

###-------------------------------------###
###----------- Third Example -----------###
###-------------------------------------###

a, b, h = 0, 1, 0.1
f = lambda x: np.sqrt(x)

k = numsteps(a, b, h)
abscissa = np.linspace(a, b, k + 1)
divdiff = get_divdiff(f, abscissa)

p3 = construct_interp_poly(divdiff, abscissa)
print("Coefficients of the interpolating polynomial in Example 3 :")
print(p3); print()

xSample = np.linspace(a, b, 250)    # Sample points to plot
fSample = f(xSample)
pSample = eval_p(divdiff, abscissa, xSample)
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(xSample, fSample, label = 'f')
ax3.plot(xSample, pSample, label = 'p')
ax3.set_title('Example 3')
ax3.legend()

plt.show()    # Show all plotted graphs
