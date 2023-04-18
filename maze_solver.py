import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation

from maze_generator import generate_maze
from pprint import pprint

def my_argmax(array):
    """
    Cette fonction est similaire à np.argmax, mais lorsque il y a plusieurs argmax, on en choisi un aléatoirement
    avec une probabilité uniforme.
    """
    return np.random.choice(np.where(array == np.max(array))[0])


def shift(position, action):
    """
    Cette fonction permet d'obtenir un nouveau état à partir d'un état et d'une action
    """
    (y, x) = position
    if action == "up":
        return (y-1, x)
    if action == "down":
        return (y+1, x)
    if action == "left":
        return (y, x-1)
    if action == "right":
        return (y, x+1)
    

def get_admissible_actions(position):
    """
    Cette fonction permet de récupérer les actions admissibles d'un état. On enlèves les actions qui nous font sortir de la grille
    """
    admissible_actions = actions[:]
    (y, x) = position
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


def get_path_from_Q(Q, start, end, max_iteration=100):
    """
    Cette fonction permet de générer les coordonnés d'un trajet calculé à partir de la matrice de décision
    état-action Q
    """
    X = start
    x_list = [X[0]]
    y_list = [X[1]]
    i = 1
    while X != end and i <= max_iteration:
        (x, y) = X
        action_number = my_argmax(Q[x, y, :])
        X = shift(X, actions[action_number])
        x_list.append(X[0])
        y_list.append(X[1])
        i += 1
    return np.array(x_list), np.array(y_list)


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


def plot_path(maze, Q, max_iteration=200):
    """
    Cette fonction permet d'afficher le chemin que l'on obtient en prenant nos décisions à partir de la matrice Q
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

    y, x = get_path_from_Q(Q, start_case, end_case, max_iteration=max_iteration)

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
                                   interval=100,
                                   blit=True)
    anim.save("animation.gif")
    plt.show()


def reward(case, action):
    x, y = case
    if maze[x, y] == 1:
        return -np.inf
    elif shift(case, action) == end_case:
        return 100000
    else:
        return 1


maze = np.array([
    [0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0]])

maze = generate_maze(21, 21)

maze_width = len(maze[:, 0])
maze_height = len(maze[0::])

start_case = (maze_height-1, 0)
end_case = (0, maze_width-1)

actions = ["up", "down", "left", "right"]

# Initialization of the Q matrix
Q = np.zeros(shape=(maze.shape[0], maze.shape[1], len(actions)))

# Number of games / training
N = 1000  

epsilon = lambda x: 1 - 1/(x+2)
epsilon = lambda x: 0.3

alpha = 0.8
gamma = 0.95

for t in range(N):
    print(f"entrainement {t}/{N}")
    X_s = start_case
    s = 0
    while X_s != end_case:
        admissible_actions = get_admissible_actions(X_s)

        u = np.random.uniform(0, 1)
        if u <= epsilon(s):
            action = np.random.choice(admissible_actions)
            action_number = actions.index(action)
        else:
            admissible_actions_numbers = get_action_numbers(admissible_actions)
            action_number = my_argmax(
                Q[X_s[0], X_s[1], admissible_actions_numbers])
            action = admissible_actions[action_number]
            action_number = actions.index(action)

        X_splus1 = shift(X_s, action)
        Q[X_s[0], X_s[1], action_number] = (1-alpha)*Q[X_s[0], X_s[1], action_number] + alpha*(
            reward(X_s, action) + gamma*np.max(Q[X_splus1[0], X_splus1[1], :]))
        s += 1
        X_s = X_splus1

plot_path(maze, Q)
