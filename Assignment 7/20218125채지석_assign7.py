import numpy as np
import matplotlib.pyplot as plt

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


A = make_An(7)
eps = 10**-6  # set the tolerance to determine stopping criteria.

## Task 1 : Use Power method to find lambda_max

n = np.size(A, 0)  # dimension we are dealing with
yk = make_ej(1, n)
rk = (yk.T @ A @ yk) / (np.dot(yk, yk))
rel_err = np.finfo(float).max  # virtually making a do-while statement

while rel_err > eps or abs(rk) < eps:
    yk = A @ yk / np.linalg.norm(A @ yk)
    rk_new = (yk.T @ A @ yk) / (np.dot(yk, yk))
    rel_err = abs(rk_new - rk) / abs(rk)
    rk = rk_new

print("Maximum eigenvalue computed using Power Method:")
print("\t", rk)


## Task 2-1 : Use Lanczos method to compute a tridiag matrix similar to An

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
            # The Lanczos method has terminated before computing 
            # the full tridiagonal matrix. We should find a new 
            # vector which is orthogonal to q_i's computed, in 
            # order to re-initiate the process. 
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
            # The Lanczos method has terminated with computing 
            # the full tridiagonal matrix.
            break
    else :
        gamma.append(gamma_i)
        q_prev = q
        q = r_i / gamma_i
        Q = np.hstack((Q, vec_transpose(q)))

tridiag = make_tridiag(delta, gamma, gamma)

## Task 2-2 : Compute the eigvals of the tridiag matrix and plot them

eigval, _ = np.linalg.eig(tridiag)
eigval = np.sort(eigval)[::-1]

print("Maximum eigenvalue computed using Lanczos Method:")
print("\t", eigval[0])

print("Maximum eigenvalue computed using internal method eig:")
print("\t", np.amax(np.linalg.eig(A)[0]))

plt.xlabel("index")
plt.ylabel("eigenvalues")
plt.scatter([i for i in range(len(eigval))], eigval, label = "eigenvalues")
plt.show()
