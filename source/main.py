import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import timeit as tm
from system import System
from ploting import plot_graph


def generate_bits(n_bits: int = 1_000_000, message_length: int = 4):
    return np.random.randint(0, 2, (np.floor(n_bits / message_length), message_length))


def calculate_bit_error_probability(original_msg: np.ndarray, decoded_msg: np.ndarray):
    diff_table: np.ndarray = original_msg != decoded_msg
    probability = sum(sum(diff_table)) / diff_table.size
    return probability


if __name__ == "__main__":
    start = tm.default_timer()
    types = ["hamming", "custom"]
    msg_lengths = {"hamming": 4, "custom": 6}
    # fmt: off
    p_list = [
        0.000002, 0.000005, 0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005,
        0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5
    ]
    # fmt: on
    """
    bit_errors = {"none": p_list}
    
    for type in types:
        bit_errors[type] = []

    for p_value in p_list:
        for type in types:
            original_msg = generate_bits(message_length=msg_lengths[type])
            system = System(p_value, type=type)
            final_msg = system.process_message(original_msg)
            bit_error = calculate_bit_error_probability(original_msg, final_msg)
            bit_errors[type].append(bit_error)

    stop = tm.default_timer()
    print("Execution time: ", stop - start)

    df = pd.DataFrame.from_dict(bit_errors)
    df.to_csv("lab1/results.csv")
    plot_graph(show_fig=False)
    
    """
