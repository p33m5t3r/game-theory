import numpy as np
import scipy.optimize as opt
import random

vector = np.ndarray
matrix = np.ndarray

def check_opt_pure_strategy(A: matrix):
    min_rows = [(i, j, A[i, j]) for i, j in enumerate(np.argmin(A, axis=1))]
    max_cols = [(i, j, A[i, j]) for j, i in enumerate(np.argmax(A, axis=0))]
    maxmin = max(min_rows, key=lambda x: x[2])
    minmax = min(max_cols, key=lambda x: x[2])

    if maxmin[2] != minmax[2]:
        return None
    
    x_star = np.zeros(A.shape[0])
    y_star = np.zeros(A.shape[1])
    x_star[maxmin[0]] = 1
    y_star[minmax[0]] = 1
    value = maxmin[2]
    return {"value": value, "x": x_star, "y": y_star}

def try_solve_invertable(A: matrix):
    n_rows, n_cols = A.shape
    if n_rows != n_cols:
        return None
    n = n_rows

    def try_find_value(B, r=0, tries=10):
        if tries == 0:
            return None
        
        B = B + r
        B_inv = np.linalg.inv(B)
        d = np.ones(B.shape[0]) @ B_inv @ np.ones(B.shape[0]).T
        if d == 0:
            r = random.randint(0, np.max(B))
            try_find_value(B, r, tries - 1)

        return (r, B_inv, d)

    A_local = A.copy()
    maybe_value = try_find_value(A_local)
    if maybe_value is None:
        return None

    r, A_inv, d = maybe_value[0], maybe_value[1], maybe_value[2]
    A_local += r
    v = 1 / d
    y = v * (A_inv @ np.ones(n).T)
    x = v * (np.ones(n) @ A_inv)

    if np.all(x >= 0) and np.all(y >= 0):
        return {"value": v - r, "x": x, "y": y}

def lp_solve_for(A: np.ndarray, _for='x'):
    A_local = A.copy()  
    n_rows, n_cols = A_local.shape
    min_aij = A_local.min()
    
    if min_aij < 0:
        A_local += abs(min_aij)

    if _for == 'x':
        A_ub = -A_local.T
        c = np.ones(n_rows)
        b_ub = -np.ones(n_cols)
        res = opt.linprog(c=c, A_ub=A_ub, b_ub=b_ub)
        value = res.fun
    else:
        A_ub = A_local
        c = -np.ones(n_cols)
        b_ub = np.ones(n_rows)
        res = opt.linprog(c=c, A_ub=A_ub, b_ub=b_ub)
        value = -res.fun

    xs_final = res.x / value
    v_final = (1 / value) + (min_aij if min_aij < 0 else 0)
    return v_final, xs_final
   
def lp_solve(A: matrix) -> dict:
    vx, x = lp_solve_for(A, _for='x')
    vy, y = lp_solve_for(A, _for='y')
    if not np.isclose(vx, vy):
        print("critical fuckup") 
        print(f"vx = {vx}, vy = {vy}")

    return {"value": vx, "x": x, "y": y}


def solve_game(A: matrix):
    # check pure strategies
    opt_soln = check_opt_pure_strategy(A)
    if opt_soln:
        print("found pure optimal strategy")
        return opt_soln

    # try solving if nice and nxn
    inverse_soln = try_solve_invertable(A)
    if inverse_soln:
        print("solved with inverse method")
        return inverse_soln
    
    # brute force linear programming
    return lp_solve(A)


# x=1xn -> col player strategy
# x=nx1 -> row
def best_response_to(A: matrix, x: vector):

    z = np.zeros_like(x)
    if x.shape[0] < x.shape[1]:
        z[0][np.argmax(np.array([row @ x[0] for row in A]))] = 1
    else:
        z[np.argmin(np.array([col @ x for col in A.T]))][0] = 1
   
    return z.T



# A = np.array([ [1,2,3], [3,1,2], [2,3,1] ])
# A = np.array([[2,-1], [-1, 1]])
# A = np.array([ [0,0,0,1], [0,0,1,-2], [0,1,-2,3], [1,-2,3,-4] ])
# A = np.array([[2,0], [0,2]])

# pure
# A = np.array([ [3,4,5], [2,-1,-1], [1, -1, -1] ])
# A = np.array([ [4,1,2], [7,2,2], [5,2,8] ])

# inverse-solvable
# A = np.array([ [0,1,-2], [1,-2,3], [-2,3,-4] ]) + 5

# battalions
# A = np.array([ [-1,-1,1,1], [1,-1,1,1], [1,-1,-1,1], [1,1,-1,1], [1,1,-1,-1] ])

# AK 1x 1v1 bet/fold poker
# A = np.array([ [1/2, 0], [0,1] ])

# A = np.array([ [4, 3], [1, 5] ])
# A = np.array([ [4,3,3], [1,6,5], [2,4,3] ])
# A = np.array([ [-2,3,5,-2], [3,-4,1,-6], [-5,3,2,-1], [-1,-3,2,2] ])
# A = np.array([ [3,-2,4,7], [-2,8,4,0] ])
A = np.array([[3,-2], [-2,1]])
print(solve_game(A))
