import numpy as np
from environment import BaseEnvironment


class Passenger:
    def __init__(self):
        self.position = None
        self.carrying = False
        self.delivered = False
        self.label = None