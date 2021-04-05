from helper_functions import get_action_probabilities, get_immediate_rewards, get_string_direction
from constants import ROWS, COLS, GAMMA

class Agent:
    def __init__(self, pos, grid, coord):
        self.pos = pos
        self.index = None
        self.actions = {0, 1, 2, 3}  # up, right, down, left
        self.grid = grid
        self.coord = coord
        self.transition_probabilities = None
        self.immediate_rewards = None
        self.values = None

    def setup_values(self):
        values = dict()
        for y in range(ROWS):
            for x in range(COLS):
                for action in self.actions:
                    if not self.coord[(x, y)].is_obstacle:
                        values[(x, y), action] = 0.0
        return values

    def initialize(self):
        self.transition_probabilities = get_action_probabilities(grid_nodes=self.grid, coord=self.coord, actions=self.actions)
        self.immediate_rewards = get_immediate_rewards(grid_nodes=self.grid, coord=self.coord, actions=self.actions)
        self.values = self.setup_values()

    def print_valid_directions(self, num_rows, num_cols, coord):  # (col, row)
        col, row = self.pos
        if row-1 >= 0:  # UP
            print(col, row-1, "can go up")
            print(coord[(col, row-1)].index)
        if col+1 < num_cols:  # RIGHT
            print(col+1, row, "can go right")
            print(coord[(col+1, row)].index)
        if row+1 < num_rows:  # DOWN
            print(col, row+1, "can go down")
            print(coord[(col, row+1)].index)
        if col-1 >= 0:  # LEFT
            print(col-1, row, "can go left")
            print(coord[(col-1, row)].index)
        print()

    # From Deep Reinforcement Learning book (Packt)
    def select_best_action(self, state_prime):
        best_action, best_value = None, None
        for action in self.actions:
            action_value = self.values[(state_prime, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def get_next_state(self, state, action):
        col, row = state.pos
        new_pos = state.pos
        if action == 0:
            new_pos = (col, row-1)
        elif action == 1:
            new_pos = (col+1, row)
        elif action == 2:
            new_pos = (col, row+1)
        elif action == 3:
            new_pos = (col-1, row)
        if new_pos in self.coord:
            return self.coord[new_pos]
        else:
            return state

    def explore(self, iterations):
        state = self.coord[(0, 0)]
        for _ in range(iterations):
            pos = state.pos
            best_action = self.select_best_action(pos)
            new_state = self.get_next_state(state, best_action)
            state = new_state

    def value_iteration(self):
        for state in self.grid: # goes through all possible states
            pos = state.pos
            utility = 0.0
            for action in self.actions:
                value = 0.0
                next_state = self.transition_probabilities[pos, action]
                for state_prime, proba in next_state.items():
                    reward_key = (pos, action, state_prime)
                    if reward_key in self.immediate_rewards:
                        immediate_reward = self.immediate_rewards[reward_key]
                        best_action = self.select_best_action(state_prime=state_prime)
                        current_value = proba * (immediate_reward + (GAMMA * self.values[(state_prime, best_action)]))
                        value += current_value
                        utility = value
                self.values[(pos, action)] = value
            # update for the window
            state.utility = utility
            pos = state.pos
            best_action = self.select_best_action(pos)
            state.direction = get_string_direction(best_action)
