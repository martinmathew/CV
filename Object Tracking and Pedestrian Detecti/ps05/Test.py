import numpy as np

def get_error_metric(template, frame_cutout):
    """Returns the error metric used based on the similarity measure.

    Returns:
        float: similarity value.
    """
    sum1 = 0.0
    for x in (0, frame_cutout.shape[0] - template.shape[0]):
        for y in (0, frame_cutout.shape[1] - template.shape[1]):
            sum1 = sum1 + np.sum(template - frame_cutout[x:x + template.shape[0], y:y + template.shape[1]]) ** 2
    sum1 = sum1 / (template.shape[0] * template.shape[1])
    return np.exp(-sum1 / (2 * (10 ** 2)))

template, _ = np.mgrid[0:3, 0:2]
frame_cutout, _ = np.mgrid[0:6, 0:5]
get_error_metric(template, frame_cutout)