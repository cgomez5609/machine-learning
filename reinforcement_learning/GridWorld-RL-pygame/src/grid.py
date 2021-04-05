import pygame

from colors import Color
from node import Node
from constants import WIN_SIZE, ROWS, COLS

class Grid:
    def __init__(self, window):
        self.window = window
        self.coord = dict()
        self.grid = self.create_grid()

    def create_grid(self):  # (col, row)
        grid = list()
        index = 0
        width_dim = WIN_SIZE // COLS
        height_dim = WIN_SIZE // ROWS
        for row in range(ROWS):
            for col in range(COLS):
                x = col
                y = row
                node = Node(col, row, width_dim, height_dim, index)
                self.coord[(x, y)] = node
                index += 1
                grid.append(node)
        return grid

    def draw_grid(self):
        gap_r = WIN_SIZE // ROWS
        gap_c = WIN_SIZE // COLS
        color = Color.PURPLE.value
        for i in range(ROWS):
            pygame.draw.line(self.window, color, (0, i * gap_r), (WIN_SIZE, i * gap_r))
            for j in range(COLS):
                pygame.draw.line(self.window, color, (j * gap_c, 0), (j * gap_c, WIN_SIZE))

    def draw_nodes(self):
        for node in self.grid:
            node.draw(self.window)
        self.draw_grid()