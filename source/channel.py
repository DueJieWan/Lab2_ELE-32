import numpy as np


class Channel:
    def __init__(self, p: float = 0.5) -> None:
        self.error_probability = p

    def transmit(self, msg: np.ndarray) -> np.ndarray:
        corrupted = np.random.rand(*msg.shape) < self.error_probability
        return np.logical_xor(msg, corrupted)
