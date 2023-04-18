import numpy as np
import random
import matplotlib.pyplot as plt

def generate_maze(MAZE_WIDTH, MAZE_HEIGHT):
    """
    Cette fonction permet de générer un tableau à deux dimension représentant un labyrinthe aléatoire.
    """

    # Define constants for the maze cells
    WALL = 1
    PATH = 0

    # Create a maze grid with all cells set to WALL
    maze = [[WALL for x in range(MAZE_WIDTH)] for y in range(MAZE_HEIGHT)]

    # Define a helper function to get a list of neighboring cells
    def get_neighbors(x, y):
        neighbors = []
        if x > 1:
            neighbors.append((x - 2, y))
        if x < MAZE_WIDTH - 2:
            neighbors.append((x + 2, y))
        if y > 1:
            neighbors.append((x, y - 2))
        if y < MAZE_HEIGHT - 2:
            neighbors.append((x, y + 2))
        return neighbors

    # Set a random starting point in the maze
    start_x = (MAZE_HEIGHT-1)
    start_y = (0)

    maze[start_y][start_x] = PATH

    # Create a list of cells that can be expanded from
    frontier = get_neighbors(start_x, start_y)

    # Expand cells in the frontier until the list is empty
    while frontier:
        # Choose a random cell from the frontier
        cell_x, cell_y = random.choice(frontier)
        frontier.remove((cell_x, cell_y))

        # If the cell is already a path cell, skip it
        if maze[cell_y][cell_x] == PATH:
            continue

        # Connect the cell to the nearest path cell
        neighbors = [(x, y) for x, y in get_neighbors(cell_x, cell_y) if maze[y][x] == PATH]
        if neighbors:
            path_x, path_y = random.choice(neighbors)
            maze[cell_y][cell_x] = PATH
            maze[(cell_y + path_y) // 2][(cell_x + path_x) // 2] = PATH

        # Add the neighboring cells of the current cell to the frontier
        frontier += get_neighbors(cell_x, cell_y)
    return np.array(maze)
    
def plot_maze(maze):
    """
    Cette fonction permet simplement d'afficher le labyrinthe
    """
    # Define the colors for the maze
    cmap = plt.cm.binary
    cmap.set_bad(color='black')
    cmap.set_over(color='green')

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Plot the maze
    ax.imshow(maze, cmap=cmap, interpolation='nearest')

    # Set the ticks and labels for the axes
    ax.set_xticks(np.arange(-.5, maze.shape[1], 1))
    ax.set_yticks(np.arange(-.5, maze.shape[0], 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', length=0)

    # Show the plot
    plt.show()
if __name__ == "__main__":
    plot_maze(generate_maze(21, 21))