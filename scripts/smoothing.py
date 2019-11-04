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

    # Lorentzian distribution smoothing for the data set
    frequency_smooth = np.linspace(min_freq, max_freq, points)
    intensity_smooth = np.zeros(points)
    for i, fi in enumerate(frequency_smooth):
        for j, fj in enumerate(frequency):
            height = intensity[j]
            intensity_smooth[i] += height * (0.5 * width)**2 / (
                        (fi - fj)**2 + (0.5 * width)**2)

    return intensity_smooth
