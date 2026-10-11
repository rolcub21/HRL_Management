import time
import random
import numpy as np
from environment import BaseEnvironment

class Blocks:
    def __init__(self, label=None):
        self.label = label  # Unique name for each block
        self.position = None
        self.carrying = False
        self.delivered = False
        self.stored = False
        self.picked = False
        self.storage_location = None
        self.storage_duration = 0
        self.storage_start_time = None
        # New attribute: cleared indicates whether the block has been moved away (cleared) from an obstacle location.
        self.cleared = False
        self.hold = False            # If True, Manager will skip this block
        self.hold_counter = 0 
        #self.storage_steps_elapsed = 0
        #self.storage_steps_needed = 0 
        self.stored_time_step = None
        self.delivered_time_step = None

        self.storage_chosen_state  = None   # φ_s  just before the choice
        self.storage_chosen_idx    = None   # integer index of the chosen cell
        self.storage_reward_accum  = 0.0    # will hold cumulative γ-discounted reward

    def set_storage_location(self, location):
        self.storage_location = location

    def initialize_storage_time(self):
        """Initialize storage time randomly between 10 and 30 steps."""
        self.storage_steps_needed = random.randint(5, 15)
        self.storage_steps_elapsed = 0
        #print(f"Initialized storage time for block: {self.storage_duration} seconds")

    def start_storage_timer(self):
        """
        Reset the steps elapsed whenever the block is officially 'stored'.
        """
        self.storage_steps_elapsed = 0

    def is_storage_time_elapsed(self):
        """
        Check if the block has stayed in storage for the required *number of steps*.
        """
        return self.storage_steps_elapsed >= self.storage_steps_needed
    
    def increment_storage_steps(self):
        """
        Increment the storage steps counter by 1.
        Call this once per environment step if the block is currently 'stored' and not yet delivered.
        """
        self.storage_steps_elapsed += 1
    
    def get_remaining_storage_time(self):
        """
        Return how many storage steps remain (not used by the environment internally, but
        can be a helper if you want).
        """
        return self.storage_steps_needed - self.storage_steps_elapsed


    def mark_as_cleared(self):
        """Mark the block as cleared so that planning algorithms ignore it as an obstacle."""
        self.cleared = True

    def reset_cleared(self):
        """Reset the cleared flag (if needed)."""
        self.cleared = False

    def __repr__(self):
        """Return a readable representation of the block, including its cleared state."""
        return f"Block({self.label}, Pos={self.position}, cleared={self.cleared})"

    def __str__(self):
        """Return a human-readable string when printed."""
        return f"Block {self.label} at {self.position} (cleared={self.cleared})"
