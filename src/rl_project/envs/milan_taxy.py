"""
Este codigo implementa un entorno de taxi en la ciudad de Milán utilizando la biblioteca Gym. 
El entorno simula un taxi que puede moverse en una cuadrícula de mxn, recoger pasajeros y dejarlos en sus destinos.
Se modifica el tamaño de la grilla, las transiciones y las recompensas.
"""

from gymnasium.envs.registration import register
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display
import numpy as np
import gymnasium as gym
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
from abc import ABC, abstractmethod
import seaborn as sns
from PIL import Image
import urllib.request
from io import BytesIO
from IPython.display import clear_output
from gymnasium import spaces

SOUTH, NORTH, EAST, WEST, PICKUP, DROPOFF = 0, 1, 2, 3, 4, 5

INTERNAL_WALLS = [
    ((0, 1), (0, 2)), ((1, 1), (1, 2)),
    ((3, 0), (3, 1)), ((4, 0), (4, 1)),
    ((3, 2), (3, 3)), ((4, 2), (4, 3))
]

def check_wall(row, col, new_row, new_col):
    """
    Returns True if there is a wall between the current position (row, col) and the new position (new_row, new_col)."""
    return ((row, col), (new_row, new_col)) in INTERNAL_WALLS or \
        ((new_row, new_col), (row, col)) in INTERNAL_WALLS

LOCS = [(0, 0), (0, 4), (4, 0), (4, 3)]

# If pass_idx is equal to PASS_IN_TAXI, osea 4 , it means the passenger is in the taxi.
PASS_IN_TAXI = 4

class MilanTaxiEnv(gym.Env):
    def __init__(self, rows: int = 6, cols: int = 6, max_steps: int = 100, p: float = 0.2):
        super().__init__()
        self.time_step = 0
        self.lastaction = None
        self._delivered_passengers = 0
        self.rows = rows
        self.cols = cols
        self.max_steps = max_steps

        # Mapeo explicito de tupla -> índice y viceversa para vectorizar Bellman
        self.all_states: List[Tuple[int, int, int, int]] = [
            (r, c, p, d)
            for r in range(self.rows)
            for c in range(self.cols)
            for p in range(5)  # 0-3: en mapa, 4: en taxi
            for d in range(4)  # 0-3: destino
        ]
        # Estados van desde 0 hasta N-1, donde N = rows * cols * 5 * 4
        self.state_to_idx: Dict[Tuple[int, int, int, int], int] = {
            s: idx for idx, s in enumerate(self.all_states)
        }

        self.num_states = len(self.all_states)
        self.observation_space = spaces.Discrete(self.num_states)
        self.action_space = spaces.Discrete(6)

        self.state: Optional[Tuple[int, int, int, int]] = None

    def reset(self):
        self._delivered_passengers = 0
        self.time_step = 0
        self.lastaction = None

        # Your code goes here: -------------------------------------
        taxi_row = np.random.randint(self.rows)
        taxi_col = np.random.randint(self.cols)
        pass_idx, dest_idx = self._spawn_new_passenger()
        # ----------------------------------------------------------

        self.state = (taxi_row, taxi_col, pass_idx, dest_idx)

        return self.state, {}

    def step(self, action):
        row, col, pass_idx, dest_idx = self.state

        actual_action = action
        if action < 4 and np.random.rand() < self.p:
            others_action = [a for a in range(4) if a != action]
            actual_action = np.random.choice(others_action)

        new_row, new_col, new_pass_idx, new_dest_idx, reward, dest_reached = \
            self._transitions(row, col, pass_idx, dest_idx, actual_action)

        self.lastaction = action

        self.time_step += 1
        truncated = self.time_step >= self.max_steps
        
        if dest_reached:
            self._delivered_passengers += 1
            if not truncated:
                new_pass_idx, new_dest_idx = self._spawn_new_passenger()

        self.state = (new_row, new_col, new_pass_idx, new_dest_idx)

        return self.state, reward, False, truncated, {"delivered_passengers": self._delivered_passengers}

    def _spawn_new_passenger(self):
        pickup, dropoff = np.random.choice(len(LOCS), size=2, replace=False)
        return int(pickup), int(dropoff)

    def _transitions(self, row, col, pass_idx, dest_idx, action):
        """
        Returns the reward -1 if the taxi moves to a new position, -10 if the taxi tries to pick up or drop off a passenger in the wrong location, 
        and +20 if the taxi successfully drops off a passenger at their destination.
        """
        new_row, new_col = row, col
        new_pass_idx = pass_idx
        dest_reached = False
        reward = -1

        # Your code goes here: -------------------------------------
        # Norte: arriba, Sur: abajo, Este: derecha, West: izquierda\n"
        if action == NORTH:
            target_row = max(0, row - 1)
            new_row = target_row if check_wall(row, col, target_row, col) == False else row
        elif action == SOUTH:
            target_row = min(self.rows - 1, row + 1)
            new_row = target_row if check_wall(row, col, target_row, col) == False else row
        elif action == EAST:
            target_col = min(self.cols - 1, col + 1)
            new_col = target_col if check_wall(row, col, row, target_col) == False else col
        elif action == WEST:
            target_col = max(0, col - 1)
            new_col = target_col if check_wall(row, col, row, target_col) == False else col
        elif action == PICKUP:
            if pass_idx != PASS_IN_TAXI and (row, col) == LOCS[pass_idx]:
                new_pass_idx = PASS_IN_TAXI
            else:
                reward = -10
        elif action == DROPOFF:
            if pass_idx == PASS_IN_TAXI and (row, col) == LOCS[dest_idx]:
                reward = 20
                dest_reached = True
            else:
                reward = -10
        # ----------------------------------------------------------

        return new_row, new_col, new_pass_idx, dest_idx, reward, dest_reached

    def render(self):
        return render_taxi(self.state, self.lastaction)

    def close(self):
        close_taxi_render()