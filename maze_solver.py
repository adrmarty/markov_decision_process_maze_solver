import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation

from pprint import pprint


maze = np.array([
    [0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0]
])
maze_width = len(maze[:, 0])
maze_height = len(maze[0::])

actions = ["up", "down", "left", "right"]


def shift(position, action):
    """
    Cette fonction permet d'obtenir un nouveau état à partir d'un état et d'une action
    """
    (x, y) = position
    if action == "up":
        return (x, y-1)
    if action == "down":
        return (x, y+1)
    if action == "left":
        return (x-1, y)
    if action == "right":
        return (x+1, y)


def get_admissible_actions(position):
    admissible_actions = actions[:]
    (x, y) = position
    if x == 0:
        admissible_actions.remove("left")
    if x == maze_width-1:
        admissible_actions.remove("right")
    if y == 0:
        admissible_actions.remove("up")
    if y == maze_height-1:
        admissible_actions.remove("down")
    return admissible_actions


def get_action_numbers(action_list):
    """
    Cette fonction permet d'obtenir le numéro associé à une action
    """
    return [actions.index(action) for action in action_list]

def get_path_from_Q(Q, start, end):
    X = start
    x_list = [X[0]]
    y_list = [X[1]]
    i = 1
    while X != end and i <= 100:
        (x, y) = X
        action_number = np.argmax(Q[x, y, :])
        X = shift(X, actions[action_number])
        x_list.append(X[0])
        y_list.append(X[1])
        i += 1
    return np.array(x_list), np.array(y_list)



def plot_maze(maze):
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

def plot_path(maze, Q):
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


    x, y = get_path_from_Q(Q, (0, 4), (4, 0))

    x = x-0.5
    y = y-0.5
    yaw = np.zeros(len(x))

    patch = patches.Rectangle((0, 0), 0, 0, fc='r')


    def init():
        ax.add_patch(patch)
        return patch,


    def animate(i):
        patch.set_width(1)
        patch.set_height(1)
        patch.set_xy([x[i], y[i]])
        patch._angle = -np.rad2deg(yaw[i])
        return patch,


    anim = animation.FuncAnimation(fig, animate,
                                init_func=init,
                                frames=len(x),
                                interval=500,
                                blit=True)
    plt.show()


Q = np.zeros(shape=(maze.shape[0], maze.shape[1], len(actions)))


def reward(case, a):
    x, y = case
    if maze[x, y] == 1:
        return -10000
    else:
        return 10


N = 1000  # Number of games
epsilon = 0.1
alpha = 0.5
gamma = 0.9

for t in range(N):
    X_s = (0, 4)
    s = 0
    while X_s != (maze.shape[0]-1, 0):
        admissible_actions = get_admissible_actions(X_s)

        u = np.random.uniform(0, 1)
        if epsilon <= u:
            action = np.random.choice(admissible_actions)
            action_number = actions.index(action)
        else:
            admissible_actions_numbers = get_action_numbers(admissible_actions)
            action_number = np.argmax(
                Q[X_s[0], X_s[1], admissible_actions_numbers])
            action = admissible_actions[action_number]
            action_number = actions.index(action)

        X_splus1 = shift(X_s, action)
        Q[X_s[0], X_s[1], action_number] = (1-alpha)*Q[X_s[0], X_s[1], action_number] + alpha*(reward(X_s, action) + np.max(Q[X_splus1[0], X_splus1[1], :]))
        s += 1
        X_s = X_splus1



plot_path(maze, Q)


