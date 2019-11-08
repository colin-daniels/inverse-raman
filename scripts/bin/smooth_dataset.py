import click
import os

import numpy as np
from scripts.io import load_smoothed_dataset


@click.command()
@click.option("--min-freq", default=0.0, help="Minimum frequency.",
              show_default=True)
@click.option("--max-freq", default=2000.0, help="Maximum frequency.",
              show_default=True)
@click.option("-p", "--points", default=2000,
              help="Number of datapoints in output.",
              show_default=True)
@click.option("-o", "--output", required=True, type=str, metavar="PATH",
              help="Path to output smoothed dataset npz file.")
@click.option("-w", "--width", required=True, type=float,
              help="Smoothing width (FWHM).")
@click.option("--num-acoustic", default=3,
              help="Number of acoustic phonon modes for each input "
                   "(they will be ignored).",
              show_default=True)
@click.argument("input_dataset")
def main(
        min_freq: float,
        max_freq: float,
        points: int,
        width: float,
        num_acoustic: int,
        input_dataset: str,
        output: str,
):
    """Load a pkl Raman dataset, smooth it in parallel, and output to npz."""
    x, y = load_smoothed_dataset(
        input_dataset,
        min_freq=min_freq,
        max_freq=max_freq,
        points=points,
        width=width,
        num_acoustic=num_acoustic,
        parallel=os.cpu_count(),
    )
    np.savez_compressed(output, x=x, y=y)


if __name__ == '__main__':
    main()
