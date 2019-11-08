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


def _smooth_dataset(
        args: tp.Tuple[tp.Dict[str, tp.Any], RawRaman]
) -> tp.Tuple[np.ndarray, np.ndarray]:
    smoothing_kwargs, data = args

    print(data.name)
    return smooth_raman(
        **smoothing_kwargs,
        frequency=data.frequency,
        intensity=data.intensity,
    ), np.array([data.width, data.length])


# load and smooth a dataset
def load_smoothed_dataset(
        path: str,
        min_freq: float,
        max_freq: float,
        points: int,
        width: float,
        parallel: tp.Optional[int] = None,
) -> tp.Tuple[np.ndarray, np.ndarray]:
    from multiprocessing.pool import Pool

    with Pool(processes=parallel) as pool:
        mapped = pool.map(_smooth_dataset, [({
            'min_freq': min_freq,
            'max_freq': max_freq,
            'points': points,
            'width': width,
        }, entry) for entry in load_raman_pkl(path)])

    return np.array([x for x, _ in mapped]), \
        np.array([y for _, y in mapped])
