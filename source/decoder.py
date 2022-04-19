from abc import ABC
import numpy as np


def array_logical_not(array, length):
    return array + np.ones(length) % 2


class Decoder(ABC):
    def __init__(self) -> None:
        super().__init__()

    def decode(self):
        raise NotImplementedError("Abstract method!")


class Hamming(Decoder):
    def __init__(self) -> None:
        super().__init__()
        self.HT = [
            [1, 1, 1],  # b1
            [1, 0, 1],  # b2
            [1, 1, 0],  # b3
            [0, 1, 1],  # b4
            [1, 0, 0],  # p1
            [0, 1, 0],  # p2
            [0, 0, 1],  # p3
        ]

    def decode(self, msg: np.ndarray) -> np.ndarray:
        N: int = msg.shape[0]
        s: np.ndarray = np.matmul(msg, self.HT) % 2
        e = np.zeros((N, 4))

        def nt(x):
            return array_logical_not(x, N)

        e[:, 0] = s[:, 0] * s[:, 1] * s[:, 2]  # b1 = s1.s2.s3
        e[:, 1] = s[:, 0] * s[:, 2] * nt(s[:, 1])  # b2 = s1.!s2.s3
        e[:, 2] = s[:, 0] * s[:, 1] * nt(s[:, 2])  # b3 = s1.s2.!s3
        e[:, 3] = s[:, 1] * s[:, 2] * nt(s[:, 0])  # b4 = !s1.s2.s3

        return (msg[:, :4] + e) % 2


class Custom(Decoder):
    def __init__(self) -> None:
        super().__init__()
        self.HT = [
            [1, 0, 1, 0],  # b1
            [1, 0, 0, 1],  # b2
            [1, 1, 0, 0],  # b3
            [0, 1, 0, 1],  # b4
            [0, 1, 1, 0],  # b5
            [0, 0, 1, 1],  # b6
            [1, 0, 0, 0],  # p1
            [0, 1, 0, 0],  # p2
            [0, 0, 1, 0],  # p3
            [0, 0, 0, 1],  # p4
        ]

    def decode(self, msg: np.ndarray) -> np.ndarray:
        N: int = msg.shape[0]
        s: np.ndarray = np.matmul(msg, self.HT) % 2
        e = np.zeros((N, 6))

        def nt(x):
            return array_logical_not(x, N)

        e[:, 0] = s[:, 0] * nt(s[:, 1]) * s[:, 2] * nt(s[:, 3])  # b1 = s1.!s2.s3.!s4
        e[:, 1] = s[:, 0] * nt(s[:, 1]) * nt(s[:, 2]) * s[:, 3]  # b2 = s1.!s2.!s3.s4
        e[:, 2] = s[:, 0] * s[:, 1] * nt(s[:, 2]) * nt(s[:, 3])  # b3 = s1.s2.!s3.!s4
        e[:, 3] = nt(s[:, 0]) * s[:, 1] * nt(s[:, 2]) * s[:, 3]  # b4 = !s1.s2.!s3.s4
        e[:, 4] = nt(s[:, 0]) * s[:, 1] * s[:, 2] * nt(s[:, 3])  # b5 = !s1.s2.s3.!s4
        e[:, 5] = nt(s[:, 0]) * nt(s[:, 1]) * s[:, 2] * s[:, 3]  # b6 = !s1.!s2.s3.s4

        return (msg[:, :6] + e) % 2
