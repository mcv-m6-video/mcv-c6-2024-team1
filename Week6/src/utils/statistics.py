""" This module contains classes and functions for statistics. """
class RollingMean:
    def __init__(self, window_size):
        self.window_size = window_size
        self.data = []

    def __call__(self, value):
        self.data.append(value)
        if len(self.data) > self.window_size:
            self.data.pop(0)
        
        if len(self.data) == 0:
            return None
        return sum(self.data) / len(self.data)

