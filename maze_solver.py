import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation

from maze_generator import generate_maze
from pprint import pprint

np.random.seed(10)

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


def get_path_from_Q(Q, start, end, max_iteration=1000):
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
        admissible_actions = get_admissible_actions((x, y))
        admissible_actions_numbers = get_action_numbers(admissible_actions)
        admissible_action_number = my_argmax(
            Q[x, y, admissible_actions_numbers])
        X = shift(X, admissible_actions[admissible_action_number])
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


def plot_path(maze, Q, max_iteration=1000, title=""):
    """
    Cette fonction permet d'afficher le chemin que l'on obtient en prenant nos décisions à partir de la matrice Q
    """
    # Création de la grille

    # Définition des couleurs
    colors = ['white', 'black', 'red']

    # Affichage de la grille avec les couleurs associées à chaque valeur
    fig, ax = plt.subplots()
    grid = np.copy(maze)
    cmap = plt.cm.colors.ListedColormap(colors)

    y_list, x_list = get_path_from_Q(Q, start_case, end_case, max_iteration=max_iteration)
    for i in range(len(x_list)):
        grid[y_list[i], x_list[i]] = 2
    

    ax.imshow(grid, cmap=cmap, interpolation='nearest')
    plt.title(title+f"\n Longueur du chemin : {len(x_list)} ")
    plt.show()


def plot_path_animation(maze, Q, max_iteration=1000):
    """
    Cette fonction permet d'afficher le chemin que l'on obtient en prenant nos décisions à partir de la matrice Q
    sous forme d'animation.
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

    y, x = get_path_from_Q(Q, start_case, end_case,
                           max_iteration=max_iteration)

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
        return 0


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

Q_HISTORY = {}
times = []
path_lenghts = []

# Number of games / training
N = 1000


def epsilon(x): return 1 - 1/(x+2)
def epsilon(x): return 0.3


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

    if t in [0, 10, 50, 100, 150, 200]:
        Q_HISTORY[t] = np.copy(Q)

    # à partir d'ici, on compte la longueur du chemin obtenu à partir de Q pour visual

    if t >= 1:
        x_list, y_list = get_path_from_Q(
            Q, start_case, end_case, max_iteration=10000)
        times.append(t)
        path_lenghts.append(len(x_list))

# On affiche la convergence du chemin vers le chemin optimal
plt.title("Convergence vers le chemin optimal")
plt.xlabel("Nombre d'entrainements")
plt.ylabel("Longueur du chemin parcouru")
plt.plot(times, path_lenghts)
plt.show()

# On affiche les chemins intermediares parcourus pendant l'entrainement
for t in Q_HISTORY.keys():
    title = f"Chemin à partir du {t}ième entrainement"
    if t == 1:
        title = "Chemin à partir du premier entrainement"
    if t == 0:
        title = "Chemin sans entrainement"
    plot_path(maze, Q_HISTORY[t], max_iteration=10000, title=title)

# On affiche l'animation du résultat final
plot_path_animation(maze, Q)


