import pygame

from colors import Color
from helper_functions import transform_pos, make_path, check_directions
from grid import Grid
from agent import Agent
from constants import WIN_SIZE, FPS, GOAL_COST, PIT_COST


def main():
    pygame.init()
    window = pygame.display.set_mode((WIN_SIZE, WIN_SIZE))
    pygame.display.set_caption("Grid World (Click square to add Goal (GREEN))")
    window.fill(Color.WHITE.value)
    pygame.time.set_timer(pygame.USEREVENT, 2000)

    goal_node = None
    pit_node = None
    obstacles_placed = False
    is_running = True
    clock = pygame.time.Clock()
    path_list = None
    directions = None

    g = Grid(window=window)
    start = (0, 0)
    agent = Agent(pos=start, grid=g.grid, coord=g.coord)

    while is_running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if not obstacles_placed:
                        agent.initialize()
                        obstacles_placed = True
                        text = "Press SPACE to calculate the best policy"
                        pygame.display.set_caption(text)
                    else:
                        for i in range(20):
                            agent.value_iteration()
                            
                            if directions is not None:
                                new_dir = check_directions(agent.grid)
                                if directions == new_dir:
                                    text = f"Optimal policies found after {i} iterations - CLICK on NODE to see path"
                                    pygame.display.set_caption(text)
                                    break
                                else:
                                    directions = new_dir
                            else:
                                directions = check_directions(agent.grid)

                if event.key == pygame.K_RETURN:
                    pit_node = True
                    text = "Add obstacles to the grid (GREY) - then press SPACE"
                    pygame.display.set_caption(text)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    pos = pygame.mouse.get_pos()
                    pos = transform_pos(pos)
                    node = agent.coord[pos]
                    if not goal_node and not pit_node:
                        node.make_goal()
                        node.cost = GOAL_COST
                        goal_node = node
                        text = "Add pits to the grid (RED) - then press ENTER"
                        pygame.display.set_caption(text)
                    elif not pit_node and not node.is_goal:
                        node.make_pit()
                        node.cost = PIT_COST
                    elif not obstacles_placed and (node != goal_node and not node.is_pit):
                        node.make_obstacle()
                        node.cost = -100.0
                if event.button == 1 and obstacles_placed:
                    pos = pygame.mouse.get_pos()
                    pos = transform_pos(pos)
                    path_list = make_path(pos, agent)
            if event.type == pygame.USEREVENT and path_list is not None:
                for n in path_list:
                    n.color = Color.WHITE.value
                goal_node.color = Color.GREEN.value
                path_list = None

        g.draw_nodes()
        pygame.display.update()

    pygame.quit()


if __name__ == '__main__':
    main()


