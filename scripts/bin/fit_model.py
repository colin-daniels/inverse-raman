import click

import numpy as np
from scripts.ml import fit_dataset


@click.command()
@click.argument("dataset")
def main(dataset: str):
    """Load a smoothed Raman dataset npz file and fit it."""
    dataset = np.load(dataset)
    fit_dataset(
        x=dataset["x"],
        y=dataset["y"],
    )


if __name__ == '__main__':
    main()
