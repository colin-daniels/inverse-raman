import numpy as np
from sklearn.model_selection import train_test_split


# TODO: need to scale counts by intensity, as it is now they're all the same
#  due to the normalization step
def generate_sample_histogram(
        smoothed_data: np.ndarray,
        num_samples: int,
        random_state: np.random.RandomState,
):
    """Sample from the smoothed data as if it were a probability distribution"""
    # calculate the cdf values (making sure to normalize)
    cdf = smoothed_data.cumsum()
    cdf /= cdf[-1]  # note: all elements of cdf are in [0,1]

    # evaluate the inverse cdf `num_samples` times by linearly interpolating
    points = np.linspace(0, len(cdf), endpoint=True)
    values = np.interp(
        x=random_state.rand(num_samples),
        xp=cdf,
        fp=points[:-1],
    )

    # be lazy and get numpy to make a histogram for us
    return np.histogram(
        a=values,
        bins=points,
    )[0]


def fit_dataset(
        x: np.ndarray,
        y: np.ndarray,
):
    random = np.random.RandomState(42)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.25,
        random_state=random,
    )

    print(f"x (train): {x_train.shape}")
    print(f"y (train): {y_train.shape}")
    print()
    print(f"x (test): {x_test.shape}")
    print(f"y (test): {y_test.shape}")
