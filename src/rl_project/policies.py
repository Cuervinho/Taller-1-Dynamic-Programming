from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np

class Policy(ABC):
    @abstractmethod
    def get_action(self, state):
        pass


class RandomPolicy(Policy):
    def __init__(self, actions_cardinality):
        self.actions_cardinality = actions_cardinality
    
    def get_action(self, state):
        # Your code goes here: -------------------------------------
        return np.random.randint(self.actions_cardinality)
        # ----------------------------------------------------------

class ValueIterationPolicy(Policy):
    def __init__(self, policy, env):
        self.policy = policy
        self.env = env
 
    def get_action(self, state):
        state_idx = self.env.state_to_idx[state]
        return np.argmax(self.policy[state_idx])

def rollout(env: gym.Env, policy: RandomPolicy, render=True):
    frames =  []

    initial_state, _ = env.reset()

    states  = [ initial_state ]
    actions = []
    rewards = []

    done = False
    while not done:
        if render:
            frames.append(env.render())
        
        # Your code goes here: -------------------------------------
        action = policy.get_action(states[-1])
        state, reward, terminated, truncated, _ = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        done = terminated or truncated
        # ----------------------------------------------------------
    
    if render:
        return frames, states, actions, rewards
    else:
        return states, actions, rewards

def policy_matrices(P, R, pi):
    """Average the dynamics over the policy: returns (P_pi, R_pi)."""
    # Your code goes here: -------------------------------------
    P_pi = np.einsum('sa,sat->st', pi, P)
    R_pi = np.einsum('sa,sa->s', pi, R)
    return P_pi, R_pi
    # ----------------------------------------------------------

def q_from_v(P, R, V, gamma):
    """One-step lookahead: Q[s, a] from V."""
    # Your code goes here: -------------------------------------
    Q = R + gamma * P @ V
    return Q
    # ----------------------------------------------------------


def greedy_policy(Q):
    """Deterministic greedy policy, as a one-hot matrix."""
    # Your code goes here: -------------------------------------
    pi = np.zeros_like(Q)
    best_actions = np.argmax(Q, axis=1)
    pi[np.arange(Q.shape[0]), best_actions] = 1.0
    return pi
    # ----------------------------------------------------------


