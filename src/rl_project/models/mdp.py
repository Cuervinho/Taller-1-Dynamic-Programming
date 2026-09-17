from rl_project.envs.milan_taxy import MilanTaxiEnv
import numpy as np

class MDP:
    def __init__(self, rows: int = 6, cols: int = 6, p: float = 0.2):
        
        self.env = MilanTaxiEnv(rows=rows, cols=cols, p=p)
        self.num_states = self.env.num_states
        self.num_actions: int = self.env.action_space.n
        self.p = p
        # self.P es un diccionario que mapea (state, action) a una lista de (probability, next_state, reward, terminated)
        self.P = {}
        self.R = {}
        self.build_transition_matrix_P()

    def build_transition_matrix_P(self):
        for s, state in enumerate(self.env.all_states):
            row, col, pass_id, dest_id = state
            self.P[s] = {a: [] for a in range(self.num_actions)}
            self.R[s] = {}
            self.R[s] = {a: 0 for a in range(self.num_actions)}
            for a in range(self.num_actions):
                # Transición terminal: el pasajero esta en el taxi y se encuentra en el destino
                # La recompensa es 0 o 20 en este caso?
                if pass_id == dest_id:
                    self.P[s][a].append((1.0, s, 0, True))
                    self.R[s][a] = 0   
                    continue
                if a < 4:  # Movimiento
                    self.movement_transition(s, row, col, pass_id, dest_id, a)
                else: # Acción de dropoff o pickup
                    new_row, new_col, new_pass_idx, dest_idx, reward, dest_reached = self.env._transitions(row, col, pass_id, dest_id, a)
                    next_state = self.env.state_to_idx[(new_row, new_col, new_pass_idx, dest_idx)]
                    self.P[s][a].append((1.0, next_state, reward, dest_reached))
                    self.R[s][a] = reward

    def movement_transition(self, s, row, col, pass_id, dest_id, action):
        matrix_transition = {}
        other_actions = [a for a in range(4) if a != action]
        prob = 1 -self.p 
        prob_other = self.p / 3

        actions_probabilities = [(action, prob)] + [(a, prob_other) for a in other_actions]

        for a, p in actions_probabilities:
            new_row, new_col, new_pass_idx, new_dest_idx, reward, dest_reached = self.env._transitions(row, col, pass_id, dest_id, a)
            next_state = self.env.state_to_idx[(new_row, new_col, new_pass_idx, new_dest_idx)]
            if next_state not in matrix_transition:
                matrix_transition[next_state] = {"prob": p, "reward": reward, "dest_reached": dest_reached}
            else:
                matrix_transition[next_state]["prob"] += p

        for next_state, values in matrix_transition.items():
            self.P[s][action].append((values["prob"], next_state, values["reward"], values["dest_reached"]))



def build_model(self, terminated=True):
    """Return (P, R) with an absorbing state at index env.nS."""
    nS, nA = self.num_states, self.num_actions
    N = nS + 1
    P = np.zeros((N, nA, N))
    R = np.zeros((N, nA))

    for s in range(nS):
        for a in range(nA):
            for prob, s2, r, done in self.P[s][a]:
                if terminated and r == -100:
                    done = True
                # Your code goes here: -------------------------------------
                next_state = nS if done else s2
                P[s, a, next_state] += prob
                R[s, a] += prob * r
                # ----------------------------------------------------------

    P[nS, :, nS] = 1.0   # the absorbing state loops to itself with reward 0
    return P, R



