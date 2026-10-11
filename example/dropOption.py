from option import BaseOption
from small_rooms_env import SmallRoomsEnv
from collections import deque

class DropOption(BaseOption):
    def __init__(self, environment):
        super().__init__(is_primitive=False)
        self.environment = environment
        self.rooms = self.environment.rooms
        # Store all exit cells, e.g. [(6,5), (6,4), (6,3)]
        self.goal_states = self.environment.exit_states  
        
        self.previous_carrying_status = [False] * len(self.environment.blocks)
        self.block_locations = self.environment._get_storage_positions()  # Initialize block locations

    # --------------------------------------------------------------------------
    #   Utilities to track blocks / carrying status
    # --------------------------------------------------------------------------
    def update_block_locations(self):
        self.block_locations = self.environment.get_block_positions()

    def update_block_carrying(self):
        self.update_block_locations()
        current_carrying_status = self.environment.get_block_carrying_status()
        self.previous_carrying_status = current_carrying_status

    def is_carrying_block(self):
        self.update_block_carrying()
        return any(self.previous_carrying_status)

    # --------------------------------------------------------------------------
    #   Option: Initiation & Termination
    # --------------------------------------------------------------------------
    def initiation(self, state):
        """
        We initiate the DropOption if the agent is carrying a block.
        """
        return self.is_carrying_block()

    def termination(self, state):
        """
        We terminate if the agent is on ANY exit cell and is no longer carrying a block.
        """
        if state in self.goal_states and not self.is_carrying_block():
            return 1.0
        return 0.0

    # --------------------------------------------------------------------------
    #   Main Policy
    # --------------------------------------------------------------------------
    def policy(self, start=None, end=None):
        """
        1) Find the closest exit among self.goal_states.
        2) Use the Lee algorithm to get a path.
        3) Return the next action to move along that path.
        4) If no path, STAY.
        """
        self.update_block_locations()
        
        # Default the start to agent's current position.
        if start is None:
            start = self.environment.current_state

        # Step 1: pick the best exit cell (the one with the shortest path).
        #         If 'end' is provided externally, we could skip this; 
        #         but typically you want to find the closest exit among self.goal_states.
        if end is None:
            # Among all exits, pick whichever has the *shortest* path from `start`.
            best_path = None
            best_len = float('inf')
            for exit_cell in self.goal_states:
                path = self.lee_algorithm(self.rooms, start, exit_cell)
                if path is not None and len(path) < best_len:
                    best_len = len(path)
                    best_path = path
            
            # If no path to ANY exit, STAY
            if not best_path:
                return SmallRoomsEnv.ACTION_IDS.get("WAIT", 6)  # or "STAY" if you prefer
            
            path = best_path
        else:
            # If 'end' is specified, just use that
            path = self.lee_algorithm(self.rooms, start, end)
            if not path or len(path) < 2:
                return SmallRoomsEnv.ACTION_IDS.get("WAIT", 6)  # or "STAY"

        # Step 2: If the path length is at least 2, figure out the next step
        if len(path) < 2:
            return SmallRoomsEnv.ACTION_IDS.get("WAIT", 6)

        next_position = path[1]

        # Step 3: Convert the (row, col) difference to an action
        if next_position[0] > start[0]:
            return SmallRoomsEnv.ACTION_IDS["DOWN"]
        elif next_position[0] < start[0]:
            return SmallRoomsEnv.ACTION_IDS["UP"]
        elif next_position[1] > start[1]:
            return SmallRoomsEnv.ACTION_IDS["RIGHT"]
        elif next_position[1] < start[1]:
            return SmallRoomsEnv.ACTION_IDS["LEFT"]

        # If we somehow reach here, do WAIT
        return SmallRoomsEnv.ACTION_IDS.get("WAIT", 6)

    # --------------------------------------------------------------------------
    #   BFS (Lee's Algorithm) to find a path in the grid
    # --------------------------------------------------------------------------
    def lee_algorithm(self, matrix, start, end):
        """
        Standard BFS that returns a path from 'start' to 'end' (if any).
        """
        if start == end:
            return [start]
        
        queue = deque([start])
        visited = {start}
        prev = {}

        while queue:
            node = queue.popleft()
            for neighbor in self.get_neighbors(matrix, node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    prev[neighbor] = node
                    queue.append(neighbor)
                    if neighbor == end:
                        return self.reconstruct_path(prev, start, end)
        return None

    def get_neighbors(self, matrix, node):
        """
        Return all free (non-wall) neighbors up/down/left/right.
        """
        neighbors = []
        row, col = node
        rows = len(matrix)
        cols = len(matrix[0]) if rows > 0 else 0

        # Up
        if row > 0 and matrix[row - 1][col] != "#":
            neighbors.append((row - 1, col))
        # Down
        if row < rows - 1 and matrix[row + 1][col] != "#":
            neighbors.append((row + 1, col))
        # Left
        if col > 0 and matrix[row][col - 1] != "#":
            neighbors.append((row, col - 1))
        # Right
        if col < cols - 1 and matrix[row][col + 1] != "#":
            neighbors.append((row, col + 1))

        return neighbors

    def reconstruct_path(self, prev, start, end):
        path = []
        node = end
        while node != start:
            path.append(node)
            node = prev[node]
        path.append(start)
        path.reverse()
        return path

    # --------------------------------------------------------------------------
    #   Representation
    # --------------------------------------------------------------------------
    def __str__(self):
        return "DropOption"

    def __repr__(self):
        return "DropOption"

    def __hash__(self):
        return hash(str(self))
