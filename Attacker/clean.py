import numpy as np

class Clean:
    def __init__(self):
        pass

    def attack(self, x):
        x = np.array(x, dtype=np.float32) / 255
        return x
    
