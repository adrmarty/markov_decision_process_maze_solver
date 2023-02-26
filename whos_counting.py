import numpy as np
from pprint import pprint

# https://www.ceremade.dauphine.fr/~carlier/progdyn.pdf

# un état est un couple (grille, nombre tiré) :
# state = ([6, None, 5, None], 7)

# Une action est l'emplacement ou mettre le chiffre tiré :
# action = 1


N = 2
digits = [i for i in range(10)]
values = [None]+digits


def final_value(state):
    (grid, _) = state
    value = 0
    for i in range(len(grid)):
        if grid[N-i-1] != None:
            value += 10**i * grid[N-i-1]
    return value


def reward(state, action):
    (grid, digit) = state
    if not None in grid:
        return 0
    return digit*10**(N-action-1)


def empty_cases(state):
    (grid, _) = state
    return [i for i in range(len(grid)) if grid[i] == None]


def nonempty_cases(state):
    (grid, _) = state
    return [i for i in range(len(grid)) if grid[i] != None]


def expected_value(state, V):
    (grid, digit) = state
    nb_empty_cases = len(empty_cases(state))
    if nb_empty_cases == 1:
        values = [V[(tuple(grid[:i] + (digit,) + grid[i+1:]), None)]
                  for i in range(N) if grid[i] == None]
        return np.mean(values)

    values = [V[(tuple(grid[:i] + (digit,) + grid[i+1:]), j)]
              for i in range(N) for j in digits if grid[i] == None]
    return np.mean(values) if len(values) > 0 else 0


# Initialisation des valeurs finales
V = {}
for x in range(10**N):
    grid = tuple([int(str(x+10**(N))[i+1]) for i in range(N)])
    state = (grid, None)
    V[state] = final_value(state)
# Résolution du jeu par programmation dynamique (backward induction)
computed_keys = list(V.keys())
while not all(((None,)*N, digit) in computed_keys for digit in digits):
    computed_keys = list(V.keys())
    for state in computed_keys:
        (grid, digit) = state
        for case_number in nonempty_cases(state):
            previous_grid = tuple(
                grid[:case_number] + (None,) + grid[case_number+1:])
            for added_digit in digits:
                previous_state = (previous_grid, added_digit)
                if not previous_state in V.keys():
                    for i in empty_cases(previous_state):
                        for j in digits:
                            new_grid = tuple(
                                previous_grid[:i] + (added_digit,) + previous_grid[i+1:])
                            if len(empty_cases(previous_state)) == 1:
                                new_state = (new_grid, None)
                            else:
                                new_state = (new_grid, j)
                            V[previous_state] = max(
                                V.get(previous_state, 0), V[new_state] + reward(previous_state, i))

# Détermination de la politique optimale
policy = []
for state in V.keys():
    (grid, digit) = state
    best_value = -np.inf
    best_action = None
    for action in empty_cases(state):
        new_grid = tuple(grid[:action] + (digit,) + grid[action+1:])
        new_states = []
        for i in digits:
            if len([i for i in range(len(new_grid)) if new_grid[i] == None]) == 0:
                new_states.append((new_grid, None))
            else:
                new_states.append((new_grid, i))
        new_value = np.mean([V[new_state] for new_state in new_states])
        if new_value > best_value:
            best_value = new_value
            best_action = action
    policy.append((state, best_action))

# Affichage des résultats
print("Politique optimale :")
for state, best_action in policy:
    (grid, digit) = state
    if digit != None:
        print("Placez le chiffre {} en position {} pour l'état {}".format(
            digit, best_action+1, state))
