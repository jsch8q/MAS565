import numpy as np 
import matplotlib.pyplot as plt

def sample_points(window, n = 50):
    """
        Generate an array that contains sample points. These points
        will be used to see how the behavior of the Newton's method
        differs according to different initial points. 
    """
    lo, hi = window
    a, b = 0, 0
    if lo is None:
        if hi is None:
            a, b = -5, 5
        else :
            a, b = hi - 10, hi
    else :
        if hi is None:
            a, b = lo, lo + 10
        else :
            a, b = lo, hi
    return np.linspace(a, b, n)

def newton_method(f, fp, samples, epsilon = 10**-6, max_depth = 1000):
    """
        Applies Newton's method to sample points /sample/ given the 
        function /f/ and its derivative /fp/. Tolerance is given as the
        argument /epsilon/, and the maximum number of iterations to try
        is given as the argument /max_depth/.
    """
    for i in range(len(samples)):
        x = samples[i]
        termination_flag = False
        for _ in range(max_depth):
            if (fp(x) == 0):
                # It is in general a bad idea to do such a comparison.
                # However we do so to endorse Newton's method to continue.
                # Break the loop with the flag down, indicating an error.
                break
            y = x - f(x) / fp(x)  
            # y is the result of one Newton's method iteration
            if abs(y) < epsilon:  
                #  if y is too close to 0
                if abs(f(y)) < epsilon:
                    #  is y=0 actually a root of f?
                    termination_flag = True
                    break
                else :
                    #  if not, continue.
                    x = y
            else :
                #  if y is sufficiently far from 0,
                #  we (can) examine the relative error
                if abs(x-y)/abs(y) < epsilon:
                    #  stopping criterion met?
                    termination_flag = True
                    break
                else:
                    #  if not, continue.
                    x = y
        if termination_flag : 
            # Newton's method has found the root.
            roots[i] = y
        else :
            # Newton's method has failed, even after /max_depth/ iterations.
            roots[i] = np.NaN
    return roots
        
def find_root_in_window(roots, window):
    """
        The reason why we need this function is implementation-specific.
        Details are on the report, but loosely speaking, because we don't 
        make an effort to find a good approximation of the root to use as
        the initial point for Newton's method, we need an extra processing 
        step to find out which of the found roots is the desired one.  
    """
    a, b = window
    if a is None:
        a = -np.Infinity
    if b is None:
        b =  np.Infinity
    root = np.NaN
    for candidate in roots:
        if np.isnan(candidate):
            continue
        elif candidate > a and candidate < b:
            root = candidate
            break
        else:
            continue
    return root        

print()
fig = plt.figure()  
#===============================#
 
print("Example 1")

window = (None, None)
samples = sample_points(window)
roots = 0 * samples

epsilon = 10**-6
f = lambda x : x + np.exp(-x**2)*np.cos(x)
fp = lambda x : 1 - np.exp(-x**2)*np.sin(x) - 2*x*np.exp(-x**2)*np.cos(x)

roots = newton_method(f, fp, samples, epsilon)
root = find_root_in_window(roots, window)

ax0 = fig.add_subplot(2, 2, 1)
ax0.plot(samples, roots, 'o', label = 'roots')
ax0.set_title('Example 1')
ax0.set_ylim([-1,0])  # an ad-hoc patch to remove unexpected matplotlib bug
ax0.legend()
ax0.grid(True)

print(root)
#===============================#

print("Example 2")

window = (1, 6)
samples = sample_points(window)
roots = 0 * samples

epsilon = 10**-6
f = lambda x : np.cos(x)**2 - np.sin(x)
fp = lambda x : -np.sin(2*x) - np.cos(x)

roots = newton_method(f, fp, samples, epsilon)
root = find_root_in_window(roots, window)

ax1 = fig.add_subplot(2, 2, 2)
ax1.plot(samples, roots, 'o', label = 'roots')
ax1.set_title('Example 2')
ax1.legend()
ax1.grid(True)

print(root)

#===============================#

print("Example 3")
window = (-1, 1)
samples = sample_points(window)
roots = 0 * samples

epsilon = 10**-6
f = lambda x : 2*x**5 - 7*x**3 + 3*x - 1
fp = lambda x : 10*x**4 - 21*x**2 + 3

roots = newton_method(f, fp, samples, epsilon)
root = find_root_in_window(roots, window)

ax2 = fig.add_subplot(2, 2, 3)
ax2.plot(samples, roots, 'o', label = 'roots')
ax2.set_title('Example 3')
ax2.legend()
ax2.grid(True)

print(root)

# A plot that shows what root has been found according to the initial point
plt.show()  
