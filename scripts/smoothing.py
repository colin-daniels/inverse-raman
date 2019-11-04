import numpy as np
import json


def smooth_raman_json(min_freq, max_freq, points, width, filename):
    frequency = []
    intensity = []
    file = json.loads(open(filename).read())

    # Load the json file data into lists
    for i in range(len(file['frequency'])):
        # Check for the max and min frequency
        if max_freq >= file['frequency'][i] >= min_freq:
            frequency.append(file['frequency'][i])
            intensity.append(file['average-3d'][i])

    return smooth_raman(
        min_freq,
        max_freq,
        points,
        width,
        frequency=np.array(frequency),
        intensity=np.array(intensity),
    )


def smooth_raman(min_freq, max_freq, points, width, frequency, intensity):
    # Only select peaks within the given frequency range
    peaks_in_range = (min_freq <= frequency) & (frequency <= max_freq)
    intensity = intensity[peaks_in_range]
    frequency = frequency[peaks_in_range]

    # Lorentzian distribution smoothing for the data set
    frequency_smooth = np.linspace(min_freq, max_freq, points)
    intensity_smooth = np.zeros(points)

    gamma_sq = (0.5 * width)**2
    for i, f_eval in enumerate(frequency_smooth):
        intensity_smooth[i] = np.sum(
            intensity * gamma_sq / ((f_eval - frequency)**2 + gamma_sq)
        )

    return intensity_smooth
