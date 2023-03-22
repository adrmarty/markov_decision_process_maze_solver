import numpy as np
from pprint import pprint

"""
Pour rendre la logique du code générique et adaptable à d'autres problèmes,
on définit un objet état que l'on manipulera dans les equations de Bellmann
"""


class Action:
    """
    Une action est simplement l'emplacement de la grille dans lequel on met le nombre tiré.
    """

    def __init__(self, grid_position) -> None:
        self.grid_position = grid_position

    def __repr__(self) -> str:
        return str(self.grid_position)


class State:
    """
    Un état est de la forme ([True, False, True], 4) où True 
    signifie que la case est libre et 4 est le tirage
    """
    digits = list(range(10))

    def __init__(self, grid, digit):

        self.N = len(grid)
        self.grid = grid
        self.digit = digit

    def __repr__(self) -> str:
        """
        On définit la facon dont on affiche un état dans la console
        """
        return f"(grid = {self.grid}, digit = {self.digit})"

    def __eq__(self, other):
        """
        On définit la facon dont on compare deux états
        """
        return (self.grid == other.grid and self.digit == other.digit)

    def __hash__(self):
        """
        On définit la facon dont l'objet est hashable, ce qui permet de l'utiliser dans un dictionnaire
        """
        return hash(str(self))

    @classmethod
    def get_inital_states(cls, N):
        return [State([True for _ in range(N)], digit) for digit in State.digits]

    @classmethod
    def get_final_states(cls, N):
        return [State([False for _ in range(N)], digit) for digit in State.digits]

    def get_next_states(self, action=None):
        grid = self.grid
        next_grids = []
        next_states = []

        if action == None:
            for i in [i for i in range(len(grid)) if grid[i] == True]:
                next_grid = grid[:]
                next_grid[i] == False
                next_grids.append(next_grid)
        else:
            next_grid = grid[:]
            next_grid[action.grid_position] = False
            next_grids = [next_grid]

        for next_grid in next_grids:
            for digit in State.digits:
                next_states.append(State(next_grid, digit))
        return next_states

    def get_previous_states(self):
        grid = self.grid
        previous_grids = []
        previous_states = []

        for i in [i for i in range(len(grid)) if grid[i] == False]:
            previous_grid = grid[:]
            previous_grid[i] = True
            previous_grids.append(previous_grid)

        for previous_grid in previous_grids:
            for digit in State.digits:
                previous_states.append(State(previous_grid, digit))
        return previous_states

    def reward(self, action):
        if len(self.get_admissible_actions()) == 0:
            return 0
        return self.digit * 10**(self.N-action.grid_position-1)

    def get_admissible_actions(self):
        return [Action(i) for i in range(self.N) if self.grid[i] == True]


# Résolution du jeu par programmation dynamique (backward induction)

N = 5
V = dict()

for final_state in State.get_final_states(N):
    V[final_state] = 0 # pas sûr de ca....

intial_states = State.get_inital_states(N)
computed_keys = list(V.keys())
while not all(initial_state in computed_keys for initial_state in intial_states):
    computed_keys = list(V.keys())

    states_to_compute = []
    for state in computed_keys:
        for state_to_compute in state.get_previous_states():
            if not state_to_compute in states_to_compute:
                states_to_compute.append(state_to_compute)

    for x in states_to_compute:
        temp_V = []
        for action in x.get_admissible_actions():
            reward = x.reward(action)
            next_values = [V[next_state]
                           for next_state in x.get_next_states(action=action)]
            # Hypothèse de loi uniforme ici
            temp_V.append(reward + np.mean(next_values))
        V[x] = np.max(temp_V)

intial_values = []
for iv in State.get_inital_states(N):
    intial_values.append(V[iv])
print("Nombre final moyen avec la politique optimale avec N =", N, ": ", np.mean(intial_values))
# Détermination de la politique optimale
policy = dict()
for state in V.keys():
    best_value = -np.inf
    best_action = None
    for action in state.get_admissible_actions():
        values = [V[next_state] + state.reward(action) for next_state in state.get_next_states(action = action)]
        value = np.mean(values)
        if value > best_value:
            best_value = value
            best_action = action
    if best_action != None:        
        policy[state] = best_action

# Affichage des résultats
print("Politique optimale :")
for state in policy.keys():
    (grid, digit) = state.grid, state.digit
    best_action = policy[state]
    print("Placez le chiffre {} en position {} pour l'état {}".format(digit, best_action.grid_position + 1, state))
