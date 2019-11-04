import pickle
import numpy as np
import typing as tp


class RawRaman:
    def __init__(
        self,
        name: str,
        w: int,
        l: int,
        frequency: np.array,
        intensity: np.array,
    ):
        self.name = name
        self.width = w
        self.length = l
        self.raman = np.hstack((
            frequency,
            intensity,
        ))


def load_raman_pkl(path: str) -> tp.Generator[RawRaman, None, None]:
    with open(path, 'rb') as f:
        while True:
            try:
                yield RawRaman(**pickle.load(f))
            except EOFError:
                break
