import pygame
import pygame.font

from colors import Color

class Node:
    def __init__(self, x, y, x_dim, y_dim, index):
        self.x = x
        self.y = y
        self.pos = (x, y)
        self.col = x * x_dim
        self.row = y * y_dim
        self.index = index
        self.color = Color.WHITE.value
        self.width = x_dim
        self.height = y_dim
        self.cost = -0.04
        self.is_obstacle = False
        self.font = pygame.font.SysFont('Arial', 12)
        self.utility = 0.0
        self.is_goal = False
        self.is_pit = False
        self.direction = "N/A"

    def make_obstacle(self):
        self.is_obstacle = True
        self.color = Color.GREY.value

    def make_goal(self):
        self.color = Color.GREEN.value
        self.is_goal = True
        self.direction = "GOAL"

    def make_pit(self):
        self.color = Color.RED.value
        self.is_pit = True
        self.direction = "PIT"

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.col, self.row, self.width, self.height), 0)
        if not self.is_goal and not self.is_pit and not self.is_obstacle:
            win.blit(self.font.render(self.direction, True, Color.BLACK.value, self.color), (self.col, self.row))