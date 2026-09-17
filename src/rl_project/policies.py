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