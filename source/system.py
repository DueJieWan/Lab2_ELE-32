from encoder import Hamming as HammingEncoder, Custom as CustomEncoder
from channel import Channel
from decoder import Hamming as HammingDecoder, Custom as CustomDecoder
import numpy as np
import numpy.polynomial.polynomial as poly


class System:
    def __init__(self, p: float, pol: np.ndarray, n: int, k: int) -> None:
        self.gD = pol
        self.n = n
        self.k = k
        self.channel = Channel(p)
        self.encoder = CustomEncoder()
        self.decoder = CustomDecoder()

    def _encode_message(self, msg: np.ndarray) -> np.ndarray:
        return self.encoder.encode(msg, self.gD)

    def _transmit_message(self, msg: np.ndarray) -> np.ndarray:
        return self.channel.transmit(msg)

    def _decode_message(self, msg: np.ndarray) -> np.ndarray:
        return self.decoder.decode(msg, self.gD)

    def process_message(self, msg: np.ndarray) -> np.ndarray:
        encoded_msg = self._encode_message(msg)
        transmitted_msg = self._transmit_message(encoded_msg)
        decoded_msg = self._decode_message(transmitted_msg)
        return decoded_msg
