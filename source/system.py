from encoder import Hamming as HammingEncoder, Custom as CustomEncoder
from channel import Channel
from decoder import Hamming as HammingDecoder, Custom as CustomDecoder
import numpy as np


class System:
    def __init__(self, p: float = 0.0, type: str = "hamming") -> None:
        self.channel = Channel(p)
        if type == "hamming":
            self.encoder = HammingEncoder()
            self.decoder = HammingDecoder()
        elif type == "custom":
            self.encoder = CustomEncoder()
            self.decoder = CustomDecoder()
        else:
            raise NotImplementedError("There's no system with the requested type")

    def _encode_message(self, msg: np.ndarray) -> np.ndarray:
        return self.encoder.encode(msg)

    def _transmit_message(self, msg: np.ndarray) -> np.ndarray:
        return self.channel.transmit(msg)

    def _decode_message(self, msg: np.ndarray) -> np.ndarray:
        return self.decoder.decode(msg)

    def process_message(self, msg: np.ndarray) -> np.ndarray:
        encoded_msg = self._encode_message(msg)
        transmitted_msg = self._transmit_message(encoded_msg)
        decoded_msg = self._decode_message(transmitted_msg)
        return decoded_msg
