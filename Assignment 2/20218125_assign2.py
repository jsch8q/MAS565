import numpy as np 
from numpy.fft import fft
import matplotlib.pyplot as plt

def numsteps(a, b, h):
    """ 
        Compute the number of steps. There are (numsteps + 1) node points.
    """
    return( round((b-a)/h) )

def eval_p(A, B, z):
    """ 
        Evaluates the trigonometric interpolating polynomial 
        at a given point z, from the coefficients A and B. 
    """
    if len(A) == len(B):  # corresponds to N odd
        N = 2 * len(A) - 1
        M = N // 2
        res = A[0] / 2
        for h in range(1, M+1):
            res = res + (A[h] * np.cos(h * z) + B[h] * np.sin(h * z))
    else :  # corresponds to N even
        N = 2 * len(A) - 2
        M = N // 2
        res = (A[0] + A[M] * np.cos(M * z)) / 2
        for h in range(1, M):
            res = res + (A[h] * np.cos(h * z) + B[h] * np.sin(h * z))

    return res

def construct_interp_trig_poly(beta):
    """
        Constructs a trigonometric interpolating polynomial from the 
        coefficients of the phase polynomial, beta. Returns two lists, 
        each corresponding to A_h and B_h of the trigonometric poly.
        If N is even, length of A and B are different. B[0] is padded.
    """
    N = len(beta)
    if (N % 2) :  # N is odd
        M = N // 2
        A = [0] * M
        B = [0] * M
        A[0] = 2 * beta[0]
        for h in range(1, M + 1):
            A[h] = beta[h] + beta[N-h]
            B[h] = 1j * (beta[h] - beta[N-h])
    else :  # N is even
        M = N // 2 
        A = [0] * (M+1)
        B = [0] *  M
        A[0] = 2 * beta[0]
        for h in range(1, M):
            A[h] = beta[h] + beta[N-h]
            B[h] = 1j * (beta[h] - beta[N-h])
        A[M] = 2 * beta[M]
    
    return np.real(A), np.real(B)  
    # A and B are anyhow real. We force taking real parts 
    # so that the program knows that they are indeed real.
    

fig = plt.figure()    # Preparation to plot functions

###-------------------------------------###
###----------- First Example -----------###
###-------------------------------------###

a, b, h = -np.pi, np.pi, np.pi/8
f = lambda x: x**2 * np.cos(x)

N = numsteps(a, b, h)
# In FFT the last point in linspace should be excluded
abscissa = np.linspace(a, b, N + 1)[:-1]  
ordinate = f(abscissa)

## Up to this point, to compute the ordinates we had to work
## on the interval [a,b]. However to compute the trigonometric
## interpolating polynomial we should forget about the given 
## interval [a,b], and pretend as if the given interval is 
## [0, 2\pi]. As a result the trigonometric interpolating 
## polynomial will be so that the (true) domain [a,b] is scaled
## and translated into [0, 2\pi]. 

beta = (1/N) * fft(ordinate)
A, B = construct_interp_trig_poly(beta)

print("Coefficients of the trig. interp. polynomial in Example 1 :")
print("A : ", A); print("B : ", B[1:]); print()

xSample = np.linspace(a, b, 250)    # Sample points to plot
fSample = f(xSample)
## p has a scaled and translated domain, so in order to evaluate 
## p as we wish, the sample points must be scaled accordingly. 
scaled_xSample = 2 * np.pi * (xSample - a) / (b - a)
pSample = eval_p(A, B, scaled_xSample)

ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(xSample, fSample, label = 'f')
ax1.plot(xSample, pSample, label = 'p')
ax1.set_title('Example 1')
ax1.legend()

###-------------------------------------###
###----------- Second Example ----------###
###-------------------------------------###

a, b, h = -5, 5, 1
f = lambda x: 1 / (1 + x**2) 

## Do the exact same routine as the first example

N = numsteps(a, b, h)
abscissa = np.linspace(a, b, N + 1)[:-1]   
ordinate = f(abscissa)

beta = (1/N) * fft(ordinate)
A, B = construct_interp_trig_poly(beta)

print("Coefficients of the trig. interp. polynomial in Example 2 :")
print("A : ", A); print("B : ", B[1:]); print()

xSample = np.linspace(a, b, 250) 
fSample = f(xSample) 
scaled_xSample = 2 * np.pi * (xSample - a) / (b - a)
pSample = eval_p(A, B, scaled_xSample)

ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(xSample, fSample, label = 'f')
ax2.plot(xSample, pSample, label = 'p')
ax2.set_title('Example 2')
ax2.legend()

plt.show()    # Show all plotted graphs
