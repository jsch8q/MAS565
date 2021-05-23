import numpy as np
import sympy as sp

def make_Tn(n):
    if n <= 0 :
        raise KeyError
    
    off_main_diag = [-1 for _ in range(n-1)]
    upper = np.diag(off_main_diag,  1)
    lower = np.diag(off_main_diag, -1)
    return 4*np.eye(n) + upper + lower

def make_An(n):
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

n = 7
A = sp.Matrix(np.array(make_An(n), dtype = int))
e1 = sp.Matrix(np.array(np.hstack(([1], np.zeros(n**2-1))), dtype = int))

K = e1
for i in range(1, 49):
    e1 = A * e1
    K = K.col_insert(i, e1)

print(K.rank())
