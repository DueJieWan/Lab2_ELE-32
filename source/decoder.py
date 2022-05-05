from abc import ABC
import numpy as np
import numpy.polynomial.polynomial as poly
from math import factorial


def array_logical_not(array, length):
    return array + np.ones(length) % 2


# transform an array of bits into decimal
def bit2dec(a: np.ndarray):
    i = 0
    b = np.copy(a)
    while i < len(a):
        b[i] *= pow(2, i)
        i += 1
    return sum(b)


# rotate a sD in relation to gD
def rotate_smol_poly(sD: np.ndarray, gD: np.ndarray) -> np.ndarray:
    NminusKminus1 = len(gD) - 1 - 1
    aD: np.ndarray = gD[0:NminusKminus1]

    if len(sD) == NminusKminus1 + 1:
        if sD[NminusKminus1]:
            sD[NminusKminus1] = 0
            sD = poly.polymulx(sD)
            sD = poly.polyadd(sD, aD)
            sD %= 2
        else:
            sD = poly.polymulx(sD)
    else:
        sD = poly.polymulx(sD)

    return sD


# rotate a polynomial
def rotate_poly(pol: np.ndarray) -> np.ndarray:
    if len(pol) > 1:
        highest_coef = pol[len(pol) - 1]
        return np.append(highest_coef, pol[0 : len(pol) - 1]).astype(int)
    else:
        return pol


# rotate a polynomial in relation to N
def rotate_big_poly(pol: np.ndarray, n: int) -> np.ndarray:
    if len(pol) == n:
        if pol[n - 1]:
            pol[n - 1] = 0
            pol = poly.polymulx(pol)
            pol[0] = 1
        else:
            pol = poly.polymulx(pol)
    else:
        pol = poly.polymulx(pol)
    return pol


# generate unique syndromes with 2 errors
def gen_uniq_syndromes_2err(gD: np.ndarray, n: int) -> dict:

    i = 1
    n_err = 2  # we will only correct up to 2 errors at once
    one_err_pattern = np.zeros(n)
    one_err_pattern[0] = 1  # [1, 0, .... 0]
    n_uniq_err = int(  # cyclical combination of 2 errors
        np.ceil(factorial(n - 1) / (factorial(n_err) * factorial(n - n_err)))
    )
    err_pattern: np.ndarray = np.zeros(n_uniq_err + 1, np.ndarray)
    pattern: np.ndarray
    pattern = one_err_pattern
    err_pattern[0] = one_err_pattern  # first error
    while i < n_uniq_err + 1:  # rest of the errors
        err_pattern[i] = np.append([1], pattern[0 : len(pattern) - 1])
        pattern = rotate_poly(pattern)
        i += 1

    n_error = 0
    sD_dict = {}  # store the pattern and symptom in a dictionary
    while n_error < len(err_pattern):
        sD = poly.polydiv(err_pattern[n_error], gD)[1] % 2
        dec = bit2dec(sD)  # keys are decimal version of symptom
        sD_dict[dec] = err_pattern[n_error].astype(int)
        n_error += 1

    return sD_dict


# fix word
def fix_word(
    n: int, k: int, uDlinha: np.ndarray, gD: np.ndarray, sDB_dict: dict
) -> np.ndarray:
    r = 0
    original = np.copy(uDlinha)
    sD = poly.polydiv(uDlinha, gD)[1] % 2
    while np.sum(sD):  # 1st step identify syndrome
        if bit2dec(sD) in sDB_dict:  # 2nd step found syndrome match in database
            sD[0] = np.logical_xor(sD[0], 1)
            uDlinha[0] = np.logical_xor(uDlinha[0], 1)
        else:  # 3rd step didn't find match
            if r > n:  # didn't find any match there are more than 2 errors
                sD = np.zeros(n)  # give up
                uDlinha = original  # restore original
                r = 0  # don't rotate
                break
            sD = rotate_smol_poly(sD, gD, n, k)  # rotate syndrome to find match
            uDlinha = rotate_big_poly(uDlinha, n)
            r += 1
    if r != 0:  # 4th step if rotated
        while r < n:  # complete the rotation to restore to original
            uDlinha = rotate_big_poly(uDlinha, n)
            r += 1
    uDfinal = poly.polydiv(uDlinha, gD)[0] % 2  # 5th step get original message
    return uDfinal.astype(int)


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

    def decode(self, msg: np.ndarray, n: int, k: int, gD: np.ndarray) -> np.ndarray:
        n_message: int = msg.shape[0]
        sB_dict = gen_uniq_syndromes_2err(gD, n)
        i = 0
        while i < n_message:
            msg[i] = fix_word(n, k, msg[i], gD, sB_dict)
            i += 1
        return msg
