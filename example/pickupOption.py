from simpleoptions import BaseOption
from small_rooms_env import SmallRoomsEnv
from collections import deque

# Define our option which takes our agent from any state in the SmallRooms gridworld environment to the passenger's location.

class PickupOption(BaseOption):
    def __init__(self):
        self.environment = SmallRoomsEnv()
        self.passenger_first_location = self.environment.passenger_first
        self.passenger_second_location = self.environment.passenger_second
        self.rooms = self.environment.rooms
        self.goal_state = self.environment.goal_state
        self.current_state = self.environment.current_state
    
    def initiation(self, state):
        # The option should be available in every state except the passenger's location.
        return state != self.passenger_first_location and state != self.passenger_second_location
    
    def termination(self, state):
        # The option should terminate when the agent reaches the passenger's location.
        if state == self.passenger_first_location or state == self.passenger_second_location:
            return 1.0 
        else: 
            return 0.0
    def is_goal_position(self, position):
        return self.goal_state == position
    
    def policy(self, start=None, end=None):
        # Simple policy to take the agent from the current state to the passenger's location using Lee's algorithm.
        if start is None:
            start = self.current_state
        if end is None:
            if not self.is_goal_position(self.passenger_first_location):
                end = self.passenger_first_location
            elif not self.is_goal_position(self.passenger_second_location):
                end = self.passenger_second_location
        matrix = self.rooms
        path = self.lee_algorithm(matrix, start, end)

        if not path or len(path) < 2:
            return SmallRoomsEnv.ACTION_IDS.get("STAY", 0)  # Return default action if no path is found or path is too short

        next_position = path[1]    

        if start[0] < next_position[0]:
            return SmallRoomsEnv.ACTION_IDS["DOWN"]
        elif start[0] > next_position[0]:
            return SmallRoomsEnv.ACTION_IDS["UP"]
        elif start[1] < next_position[1]:
            return SmallRoomsEnv.ACTION_IDS["RIGHT"]
        elif start[1] > next_position[1]:
            return SmallRoomsEnv.ACTION_IDS["LEFT"]
        elif next_position == path[-1]:
            return SmallRoomsEnv.ACTION_IDS["PICKUP"]
        return SmallRoomsEnv.ACTION_IDS.get("STAY", 0)  # Default action if no movement is needed

    def lee_algorithm(self, matrix, start, end):
        queue = deque()
        visited = set()
        distance = {start: 0}
        prev = {}

        queue.append(start)
        visited.add(start)

        while queue:
            node = queue.popleft()

            # Explore the neighboring nodes
            for neighbor in self.get_neighbors(matrix, node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    distance[neighbor] = distance[node] + 1
                    prev[neighbor] = node
                    queue.append(neighbor)

                if neighbor == end:
                    return self.get_shortest_path(prev, start, end)
        return None
    
    def get_neighbors(self, matrix, node):
        neighbors = []
        row, col = node

        # Check the top neighbor
        if row > 0 and matrix[row - 1][col] != "#":
            neighbors.append((row - 1, col))

        # Check the bottom neighbor
        if row < len(matrix) - 1 and matrix[row + 1][col] != "#":
            neighbors.append((row + 1, col))

        # Check the left neighbor
        if col > 0 and matrix[row][col - 1] != "#":
            neighbors.append((row, col - 1))

        # Check the right neighbor
        if col < len(matrix[0]) - 1 and matrix[row][col + 1] != "#":
            neighbors.append((row, col + 1))

        return neighbors
    
    def get_shortest_path(self, prev, start, end):
        path = []
        node = end

        while node != start:
            path.append(node)
            node = prev[node]

        path.append(start)
        path.reverse()

        return path

    def __str__(self):
        return "PickupOption"

    def __repr__(self):
        return "PickupOption"

    def __hash__(self):
        return hash(str(self))


