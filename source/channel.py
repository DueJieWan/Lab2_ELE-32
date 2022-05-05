import numpy as np


class Channel:
    def __init__(self, p: float = 0.5) -> None:
        self.error_probability = p

    def transmit(self, msg: np.ndarray) -> np.ndarray:
        i: int = 0
        while i < msg.shape[0]:
            corrupted = np.random.rand(*msg[i].shape) < self.error_probability
            msg[i] += corrupted
            i += 1
        return msg % 2
