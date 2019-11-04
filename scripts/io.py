import pickle
import numpy as np
import typing as tp

from scripts.smoothing import smooth_raman


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
        self.frequency = frequency
        self.intensity = intensity


def load_raman_pkl(path: str) -> tp.Generator[RawRaman, None, None]:
    with open(path, 'rb') as f:
        while True:
            try:
                yield RawRaman(**pickle.load(f))
            except EOFError:
                break


def load_smoothed_dataset(
        path: str,
        min_freq: float,
        max_freq: float,
        points: int,
        width: float
):
    X = []
    y = []

    for entry in load_raman_pkl(path):
        print(entry.name)
        smoothed = smooth_raman(
            min_freq=min_freq,
            max_freq=max_freq,
            points=points,
            width=width,
            frequency=entry.frequency,
            intensity=entry.intensity,
        )
        X.append(smoothed)
        y.append([entry.width, entry.length])

    return np.array(X), np.array(y)
