import numpy as np
import random
from pynput.keyboard import Key, Controller

from copy import deepcopy

from simpleoptions import BaseEnvironment
from passenger_instance import Passenger

#  The environment's layout, consisting of two 3x3 rooms connected by a small doorway.
#  The start is in the top-left, the goal is in the bottom-right.
#  Rewards of -1 per time-step, +10 for reaching the goal.
#  Deterministic actions for moving North, South, East, and West.
#  Moving into a wall causes the agent to stay in its current state, but time advances.
#  # # # # # # # # #
#  # S . . # P . . #
#  # . . . # . . . #
#  # . . . # . . . #
#  # . . . . . . . #
#  # . . . # . . . #
#  # . . . # . . . #
#  # . . . # . . G #
#  # # # # # # # # #
#
#  . = Passable Floor, # = Impassable Wall, S = start, G = Goal, P = Passenger

class SmallRoomsEnv(BaseEnvironment):
    ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "PICKUP", 5: "PUTDOWN"}
    ACTION_IDS = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3, "PICKUP": 4, "PUTDOWN": 5}

    def __init__(self):
        super().__init__()
        self.rooms = self._initialise_rooms()
        self.start_state = (1, 1)
        self.goal_state = (7, 7)
        self.current_state = None
        self.number_passenger = 4

    def get_state_space(self):
        return (8, 8)

    def reset(self):
        self.current_state = self.start_state
        self.passengers = []
        occupied_positions = set([self.start_state, self.goal_state])  # Keep track of already occupied positions # Keep track of already occupied positions

        for i in range(self.number_passenger):
            passenger = Passenger()
            passenger.label = f"P{i+1}"
            # Generate a unique position for each passenger
            while True:
                new_position = (random.randint(1, 7), random.randint(1, 7))
                if new_position not in occupied_positions:
                    passenger.position = new_position
                    occupied_positions.add(new_position)  # Mark this position as occupied
                    break
            self.passengers.append(passenger)
            
        return self.start_state

    def step(self, action):
        next_state = self._get_intended_cell(self.current_state, action)
        self.passenger_moves(action)
        self.current_state = next_state
        reward = -1

        if self.is_state_terminal(self.current_state):
            reward += 20
            print("Goal Reached")

        return self.current_state, reward, self.is_state_terminal(self.current_state), {}

    def get_action_space(self):
        return list([0, 1, 2, 3, 4, 5])

    def get_available_actions(self, state):
        return self.get_action_space()

    def is_state_terminal(self, state):
        return all(passenger.delivered for passenger in self.passengers)

    def get_initial_states(self):
        return deepcopy([self.start_state])

    def get_successors(self, state=None, actions=None):
        if state is None:
            state = self._get_intended_cell(self.current_state, action)
        if actions is None:
            actions = self.get_available_actions(state)

        successors = []
        for action in actions:
            successor_state = self._get_intended_cell(state, action)
            if successor_state not in successors:
                successors.append(successor_state)

        return successors

    def render(self, mode="human"):
        self.rooms[self.current_state] = "A"
        for passenger in self.passengers:
            self.rooms[passenger.position] = passenger.label
        self.rooms[self.goal_state] = "G"
        for i in range(self.rooms.shape[0]):
            print(" ".join(self.rooms[i]))
        print()
        self.rooms[self.current_state] = "."
        for passenger in self.passengers:
            self.rooms[passenger.position] = "."
        self.rooms[self.goal_state] = "."

    def close(self):
        pass

    def _initialise_rooms(self):
        rooms = np.full((9, 9), "#", dtype=str)
        rooms[1:-1, 1:-1] = "."
        rooms[:, 4] = "#"
        rooms[4, 4] = "."
        return rooms

    def passenger_moves(self, action):
        for passenger in self.passengers:
            if passenger.carrying:
                passenger.position = self._get_intended_cell(passenger.position, action)


    def _get_intended_cell(self, current_state, action):
        intended_next_state = current_state

        if self.ACTION_NAMES[action] == "UP":
            intended_next_state = (intended_next_state[0] - 1, intended_next_state[1])
        elif self.ACTION_NAMES[action] == "DOWN":
            intended_next_state = (intended_next_state[0] + 1, intended_next_state[1])
        elif self.ACTION_NAMES[action] == "LEFT":
            intended_next_state = (intended_next_state[0], intended_next_state[1] - 1)
        elif self.ACTION_NAMES[action] == "RIGHT":
            intended_next_state = (intended_next_state[0], intended_next_state[1] + 1)
        elif self.ACTION_NAMES[action] == "PICKUP":
            for passenger in self.passengers:
                if not passenger.carrying and self.current_state == passenger.position and not passenger.delivered:
                    passenger.carrying = True
                    break  # Assuming you can only pick up one passenger at a time
        elif self.ACTION_NAMES[action] == "PUTDOWN":
            for passenger in self.passengers:
                if passenger.carrying:
                    passenger.carrying = False
                    if passenger.position == self.goal_state:
                        passenger.delivered = True
                        print(f"Passenger {passenger.label} delivered!")
                    break  # Assuming you put down one passenger at a time

        if self.rooms[intended_next_state] == "#":
            intended_next_state = current_state

        return intended_next_state
'''
if __name__ == "__main__":
    env = SmallRoomsEnv()
    env.reset()

    action_input = {
        "0": 0,  # UP
        "1": 1,  # DOWN
        "2": 2,  # LEFT
        "3": 3,  # RIGHT
        "4": 4,  # PICKUP
        "5": 5   # PUTDOWN
    }

    while True:
        action = input("Enter action: ")
        action = action_input[action]
        state, reward, terminal, _ = env.step(action)
        print(f"Action: {env.ACTION_NAMES[action]}, Reward: {reward}")
        env.render()
        if terminal:
            print("Episode finished!")
            break

'''