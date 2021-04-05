from constants import WIDTH_DIM, HEIGHT_DIM, ROWS, COLS
from colors import Color

def get_immediate_rewards(grid_nodes, coord, actions):
    rewards_dict = dict()
    for node in grid_nodes:
        col, row = node.pos
        for action in actions:
            if action == 0:  # up
                if (col, row - 1) in coord and not coord[(col, row - 1)].is_obstacle:
                    rewards_dict[(col, row), action, (col, row - 1)] = coord[(col, row - 1)].cost
            elif action == 1:  # right
                if (col + 1, row) in coord and not coord[(col + 1, row)].is_obstacle:
                    rewards_dict[(col, row), action, (col + 1, row)] = coord[(col + 1, row)].cost
            elif action == 2:  # down
                if (col, row + 1) in coord and not coord[(col, row + 1)].is_obstacle:
                    rewards_dict[(col, row), action, (col, row + 1)] = coord[(col, row + 1)].cost
            elif action == 3:  # left
                if (col - 1, row) in coord and not coord[(col - 1, row)].is_obstacle:
                    rewards_dict[(col, row), action, (col - 1, row)] = coord[(col - 1, row)].cost
    return rewards_dict


def get_action_probabilities(grid_nodes, coord, actions):  # intended .8, else .1
    transition_proba = dict()
    for node in grid_nodes:
        col, row = node.pos
        for action in actions:
            if action == 0:  # Up
                invalid = 0.0
                transition_proba[(col, row), action] = dict()
                if (col, row - 1) in coord and not coord[(col, row - 1)].is_obstacle:
                    transition_proba[(col, row), action][(col, row - 1)] = 0.8
                else:
                    invalid += 0.8
                if (col + 1, row) in coord and not coord[(col + 1, row)].is_obstacle:
                    transition_proba[(col, row), action][(col + 1, row)] = 0.1
                else:
                    invalid += 0.1
                if (col - 1, row) in coord and not coord[(col - 1, row)].is_obstacle:
                    transition_proba[(col, row), action][(col - 1, row)] = 0.1
                else:
                    invalid += 0.1
                if invalid > 0.0:
                    transition_proba[(col, row), action][(col, row)] = invalid
            if action == 1:  # Right
                invalid = 0.0
                transition_proba[(col, row), action] = dict()
                if (col + 1, row) in coord and not coord[(col + 1, row)].is_obstacle:
                    transition_proba[(col, row), action][(col + 1, row)] = 0.8
                else:
                    invalid += 0.8
                if (col, row - 1) in coord and not coord[(col, row - 1)].is_obstacle:
                    transition_proba[(col, row), action][(col, row - 1)] = 0.1
                else:
                    invalid += 0.1
                if (col, row + 1) in coord and not coord[(col, row + 1)].is_obstacle:
                    transition_proba[(col, row), action][(col, row + 1)] = 0.1
                else:
                    invalid += 0.1
                if invalid > 0.0:
                    transition_proba[(col, row), action][(col, row)] = invalid
            if action == 2:  # Down
                invalid = 0.0
                transition_proba[(col, row), action] = dict()
                if (col, row + 1) in coord and not coord[(col, row + 1)].is_obstacle:
                    transition_proba[(col, row), action][(col, row + 1)] = 0.8
                else:
                    invalid += 0.8
                if (col + 1, row) in coord and not coord[(col + 1, row)].is_obstacle:
                    transition_proba[(col, row), action][(col + 1, row)] = 0.1
                else:
                    invalid += 0.1
                if (col - 1, row) in coord and not coord[(col - 1, row)].is_obstacle:
                    transition_proba[(col, row), action][(col - 1, row)] = 0.1
                else:
                    invalid += 0.1
                if invalid > 0.0:
                    transition_proba[(col, row), action][(col, row)] = invalid
            if action == 3:  # Left
                invalid = 0.0
                transition_proba[(col, row), action] = dict()
                if (col - 1, row) in coord and not coord[(col - 1, row)].is_obstacle:
                    transition_proba[(col, row), action][(col - 1, row)] = 0.8
                else:
                    invalid += 0.8
                if (col, row - 1) in coord and not coord[(col, row - 1)].is_obstacle:
                    transition_proba[(col, row), action][(col, row - 1)] = 0.1
                else:
                    invalid += 0.1
                if (col, row + 1) in coord and not coord[(col, row + 1)].is_obstacle:
                    transition_proba[(col, row), action][(col, row + 1)] = 0.1
                else:
                    invalid += 0.1
                if invalid > 0.0:
                    transition_proba[(col, row), action][(col, row)] = invalid
    return transition_proba


def get_string_direction(action):
    if action == 0:
        return "up"
    elif action == 1:
        return "right"
    elif action == 2:
        return "down"
    else:
        return "left"

def transform_pos(pos):
    x, y = pos
    x = x // WIDTH_DIM
    y = y // HEIGHT_DIM
    return (x, y)

def check_directions(grid):
    directions = list()
    for node in grid:
        directions.append(node.direction)
    return directions

def make_path(pos, agent):
    print(pos)
    nodes_to_goal = list()
    path_color = Color.BLUE.value
    node = agent.coord[pos]
    if node.is_obstacle or node.is_pit:
        return None
    node.color = path_color
    nodes_to_goal.append(node)
    count = 0
    while not node.is_goal and count < (ROWS * COLS):
        action = agent.select_best_action(pos)
        col, row = pos
        if action == 0:
            pos = (col, row-1)
            if row-1 < 0:
                break
            node = agent.coord[pos]
            if node.is_obstacle or node.is_pit:
                break
            node.color = path_color
        elif action == 1:
            pos = (col+1, row)
            if col+1 >= COLS:
                break
            node = agent.coord[pos]
            if node.is_obstacle or node.is_pit:
                break
            node.color = path_color
        elif action == 2:
            pos = (col, row+1)
            if row+1 > ROWS:
                break
            node = agent.coord[pos]
            if node.is_obstacle or node.is_pit:
                break
            node.color = path_color
        elif action == 3:
            pos = (col-1, row)
            if col-1 < 0:
                break
            node = agent.coord[pos]
            if node.is_obstacle or node.is_pit:
                break
            node.color = path_color
        if node.is_goal:
            break
        nodes_to_goal.append(node)
        count += 1
        print(action)
    return nodes_to_goal
