
import numpy as np
from rl_project.policies import policy_matrices, q_from_v, greedy_policy


def solve_policy_direct(P, R, pi, gamma):
    """Exact V^pi by solving (I - gamma P^pi) V = R^pi."""
    P_pi, R_pi = policy_matrices(P, R, pi)
    # Your code goes here: -------------------------------------
    I = np.eye(P_pi.shape[0])
    V_pi = np.linalg.solve(I - gamma * P_pi, R_pi)
    return V_pi
    # ----------------------------------------------------------

def iterative_policy_evaluation(P, R, pi, gamma, tol=1e-10, max_iter=100_000):
    P_pi, R_pi = policy_matrices(P, R, pi)
    V = np.zeros(P.shape[0])
    deltas = []

    for k in range(max_iter):
        # Your code goes here: -------------------------------------
        V_new = R_pi + gamma * P_pi @ V
        delta = np.max(np.abs(V_new - V))
        deltas.append(delta)
        V = V_new
        if delta < tol:
            break
        # ----------------------------------------------------------

    return V, k + 1, deltas

def policy_iteration(P, R, gamma, max_iter=1_000):
    pi = np.ones(R.shape) / R.shape[1]      # start from the random policy

    for k in range(max_iter):
        # Your code goes here: -------------------------------------
        V = iterative_policy_evaluation(P, R, pi, gamma)[0]
        pi = greedy_policy(q_from_v(P, R, V, gamma))   
        # ----------------------------------------------------------

    return V, pi, k + 1

def value_iteration(P, R, gamma, tol=1e-10, max_iter=100_000):
    V = np.zeros(R.shape[0])
    deltas = []

    for k in range(max_iter):
        # Your code goes here: -------------------------------------
        V_new = np.max(q_from_v(P, R, V, gamma), axis=1)
        delta = np.max(np.abs(V_new - V))
        deltas.append(delta)
        V = V_new
        if delta < tol:
            break
        # ----------------------------------------------------------

    pi = greedy_policy(q_from_v(P, R, V, gamma))
    return V, pi, k + 1, deltas