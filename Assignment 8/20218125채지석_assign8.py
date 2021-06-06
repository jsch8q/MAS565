import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size

def make_tridiag(main_diag, upper, lower):
    """
        Construct a tridiagonal matrix from given three vectors. 
    """
    res = np.diag(main_diag) + np.diag(upper, 1) + np.diag(lower, -1)
    return res

def make_Tn(n):
    """
        Construct T_n as indicated in the assignment. 
    """
    if n <= 0 :
        raise KeyError
    
    off_diag = -1 * np.ones(n-1)
    main_diag = 4 * np.ones(n)
    res = make_tridiag(main_diag, off_diag, off_diag)

    return res

def make_An(n):
    """
        Construct a block matrix A_n as indicated in the assignment. 
    """
    if n <= 0 :
        raise KeyError

    Tn = make_Tn(n)
    if n == 1 :
        return Tn

    I = np.eye(n)
    O = np.zeros((n,n))

    res = np.block([Tn, -I] + [O] * (n-2))  # first 'row'
    for i in range(1, n-1):
        tmp = np.block([O] * (i-1) + [-I, Tn, -I] + [O] * (n-2-i))
        res = np.vstack((res, tmp))
    tmp = tmp = np.block([O] * (n-2) + [-I, Tn])
    res = np.vstack((res, tmp))  # last 'row'
    return res    

def make_ej(j, n):
    """
        Construct the j-th canonical basis vector in the C^n.
    """
    res = np.hstack((np.zeros(j-1), [1], np.zeros(n-j)))
    return res

def residual(v, Q):
    """
        Compute the residual vector when orthogonally projecting v
        onto span(Q). In this program we only condsider the case where
        all columns of Q are orthonormal, so we can simplify our task.
    """
    # proj = Q @ np.linalg.inv(Q.T @ Q) @ Q.T
    proj = Q @ Q.T
    return v - proj @ v

def vec_transpose(v):
    """
        In numpy, vectors are really vectors, that is, they are
        not the same with n*1 matrices. Therefore, to 'transpose' 
        a vector we need special treatment. 
    """
    vT = np.reshape(v.copy(), (-1, 1))
    return vT

fig = plt.figure()  # Preparation to plot eigenvalues

A = make_An(7)
eps = 10**-6  # set the tolerance to determine stopping criteria.
n = np.size(A, 0)  # dimension we are dealing with

## Assignment 7 : Construct Tridiagonal matrix similar to An

gamma = []
delta = []
gamma_i = 0
q = make_ej(1, n)
q_prev = np.zeros(n)

i = 1
queue = 1

delta.append(q.T @ A @ q)
Q = vec_transpose(q)

while i < n :
    i = i + 1
    delta_i = q.T @ A @ q
    delta.append(delta_i)
    r_i = A @ q - delta_i * q - gamma_i * q_prev
    gamma_i = np.linalg.norm(r_i)
    
    if abs(gamma_i) < eps:
        if i <= n:
            gamma_i = 0
            gamma.append(gamma_i)
            new_q = np.zeros(0)
            while np.linalg.norm(new_q) < eps:
                if queue > n :
                    raise KeyError
                queue = queue + 1
                test = make_ej(queue, n)
                new_q = residual(test, Q)
            q = new_q / np.linalg.norm(new_q)
            Q = np.hstack((Q, vec_transpose(q)))
        else :
            break
    else :
        gamma.append(gamma_i)
        q_prev = q
        q = r_i / gamma_i
        Q = np.hstack((Q, vec_transpose(q)))

tridiag = make_tridiag(delta, gamma, gamma)

eigval, _ = np.linalg.eig(tridiag)
eigval = np.sort(eigval)[::-1]

ax1 = fig.add_subplot(1, 2, 1)
plt.xlabel("index")
plt.ylabel("eigenvalues")
ax1.scatter([i for i in range(len(eigval))], eigval, label = "eig")
ax1.legend()


#-------------------------------------------------------------------#
#                     Assignment 8 starts here!                     #
#-------------------------------------------------------------------#


def my_qr(A):
    """
        I am not sure if implementing the QR decomposition is also a
        part of the assignment or not, so I just made one myself. 
        This uses Householder reflections to compute a QR decomposition 
        of the given matrix A, but it is of course much more ineffecient  
        than the numpy built-in function /*numpy.linalg.qr*/.
    """
    n = np.size(A, 0)
    Q = np.eye(n)
    R = A
    for k in range(n):
        x = R[k:n, k]
        x_size = np.size(x)
        x_norm = np.linalg.norm(x)
        hh_vec = x + np.sign(x[0]) * x_norm * make_ej(1, x_size)
        hh_vec = hh_vec / np.linalg.norm(hh_vec)
        for j in range(k, n):
            R[k:n, j] = R[k:n, j] - 2*np.outer(hh_vec,hh_vec) @ R[k:n,j]
        refl = np.hstack((np.zeros(k), hh_vec))
        Qk = np.eye(n) - 2 * np.outer(refl, refl) 
        Q = Q @ Qk

    return Q, R

max_iter = 1000
# QR method may not converge in general (although, not in this case).
# Hence, maximum number of iterations must be set.

A_k = tridiag
for k in range(max_iter):
    Q, R = np.linalg.qr(A_k)  # For speed, it is recommended to use this.
    #Q, R = my_qr(A_k)  # This is roughly 200 times slower, per iteration. 
    A_next = R @ Q
    if np.linalg.norm(A_next) < eps:
        rel_err = 0
    else :
        rel_err = np.linalg.norm(A_next - A_k) / np.linalg.norm(A_next)
    A_k = A_next
    if rel_err < eps:
        break

eig_diag = np.diag(A_k)

print("Maximum off-diagonal:")
print("\t", np.amax(np.diag(eig_diag) - A_k))
    # The smaller this value, the higher probability of convergence.

eigval_qr = np.sort(eig_diag)[::-1]
ax2 = fig.add_subplot(1, 2, 2)
plt.xlabel("index")
plt.ylabel("eigenvalues")
ax2.scatter( [i for i in range(len(eigval_qr))], eigval_qr,\
             label = "QR" )
ax2.legend()

print("Maximum relative error of results from QR method:")
print("\t", np.amax(abs(eigval- eigval_qr) / eigval))

plt.show()