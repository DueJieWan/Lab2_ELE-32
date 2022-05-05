from abc import ABC
import numpy as np
import numpy.polynomial.polynomial as poly


class Encoder(ABC):
    def __init__(self) -> None:
        super().__init__()

    def encode(self):
        raise NotImplementedError("Abstract method!")


class Hamming(Encoder):
    def __init__(self) -> None:
        super().__init__()
        self.G = [
            [1, 0, 0, 0, 1, 1, 1],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 1, 1, 0],
            [0, 0, 0, 1, 0, 1, 1],
        ]

    def encode(self, msg: np.ndarray) -> np.ndarray:
        return np.matmul(msg, self.G) % 2


class Custom(Encoder):
    def __init__(self) -> None:
        super().__init__()

    def encode(self, msg: np.ndarray, gD: np.ndarray) -> np.ndarray:
        n_message = msg.shape[0]
        i = 0
        encoded: np.ndarray = np.zeros(n_message, np.ndarray)
        while i < n_message:
            encoded[i] = poly.polymul(msg[i], gD)
            encoded[i] = encoded[i].astype(int)
            i += 1

        return encoded % 2
