import cv2 as cv
import cv2
import numpy as np
import maxflow
import sys

INPUT_DIR = "input_img"
OUTPUT_DIR = "src/"


def ground_truth(filename):
    ground_truth_flowers, _ = load_pfm(open(filename, "rb"))
    ground_truth_flowers = ground_truth_flowers.T
    ground_truth_flowers = np.rot90(ground_truth_flowers)
    ground_truth_flowers = cv2.resize(ground_truth_flowers, (0, 0), fx=0.25, fy=0.25)
    ground_truth_flowers *= 0.25
    return ground_truth_flowers


def calculate_ssd(left_img, right_img, patch_size):
    left_img = resize(cv.cvtColor(left_img, cv.COLOR_BGR2GRAY))
    right_img = resize(cv.cvtColor(right_img, cv.COLOR_BGR2GRAY))

    depth_img = np.zeros(left_img.shape, dtype=np.uint8)
    start_row = int(patch_size / 2)
    start_col = int(patch_size / 2)
    for row in range(start_row, left_img.shape[0] - start_row):
        print(".", end="", flush=True)
        for col in range(start_col, left_img.shape[1] - start_col):
            left_patch = left_img[row - start_row:row + start_row, col - start_col:col + start_col]
            min = sys.float_info.max
            min_col = -1
            for col1 in range(start_col, right_img.shape[1] - start_col):
                right_patch = right_img[row - start_row:row + start_row, col1 - start_col:col1 + start_col]
                diff = right_patch - left_patch
                sqrd = diff * diff
                sqrd_sum = np.sum(sqrd)
                if sqrd_sum < min:
                    min = sqrd_sum
                    min_col = col1 - col
            depth_img[row, col] = abs(min_col)
    # normalizedImg = np.zeros(left_img.shape)
    # return cv.normalize(depth_img,  normalizedImg, 0, 255, cv.NORM_MINMAX)
    print("Finished")
    return cv.equalizeHist(depth_img)


VAR_ALPHA = 0
VAR_ABSENT = 1


def sumOfSquaredDifferences(image1, image2, w):
    disp = np.zeros(image1.shape)

    for y in range(w, len(image1) - w):
        for x in range(w, len(image1[y]) - w):
            res = cv2.matchTemplate(image1[y - w:y + w, :], image2[y - w:y + w, x - w:x + w], cv2.cv.CV_TM_SQDIFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            disp[y, x] = abs(x - min_loc[0])

    return disp


# Utility function
def normalize_and_scale(image_in, scale_range=(0, 255)):
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


def handle_occlusion_scale(image_in):
    locations = np.where(image_in == np.inf, 0, 1)
    image_out = np.where(image_in == np.inf, 0, image_in)
    average = sum(image_out) / np.sum(locations)
    image_out = np.where(locations == 0, average, image_in)
    disp = normalize_and_scale(image_out)
    disp = disp.astype(np.uint8)
    return cv.equalizeHist(disp)

def resize(img, scale_percent=50):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


'''
Load a PFM file into a Numpy array. Note that it will have
a shape of H x W, not W x H. Returns a tuple containing the
loaded image and the scale factor from the file.

Taken from https://gist.github.com/chpatrick/8935738
'''
'''
Load a PFM file into a Numpy array. Note that it will have
a shape of H x W, not W x H. Returns a tuple containing the
loaded image and the scale factor from the file.
'''


def load_pfm(file):
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    return np.reshape(data, shape), scale


'''
Save a Numpy array to a PFM file.
'''


def save_pfm(file, image, scale=1):
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image.tofile(file)


def graph_cut_energy_minimization(left_img, right_img):
    left_img = resize(cv.cvtColor(left_img, cv.COLOR_BGR2GRAY))
    right_img = resize(cv.cvtColor(right_img, cv.COLOR_BGR2GRAY))

    left_img = left_img.astype(np.float32)
    right_img = right_img.astype(np.float32)

    alpha_limit = 17

    prev_e = sys.maxsize
    disparity = np.full(left_img.shape, np.inf)
    K = calculate_k(alpha_limit, left_img, right_img)
    for it in range(10):
        alphas = np.random.permutation(alpha_limit)
        print(".", end="", flush=True)
        for alpha in alphas:

            g = maxflow.Graph[float](2, 2)
            varsO = []
            varsOnodes = []
            varsA = []
            varsAnodes = []

            for row in range(disparity.shape[0]):
                for col in range(disparity.shape[1]):
                    if disparity[row, col] == np.inf:
                        varsO.append(VAR_ABSENT)
                        varsOnodes.append(-1)
                        if col + alpha < left_img.shape[1]:
                            varsA.append({"cord": (row, col), "alpha": alpha})
                            varsAnodes.append(g.add_nodes(1))
                        else:
                            varsA.append(VAR_ABSENT)
                            varsAnodes.append(-1)
                    else:
                        if disparity[row, col] != alpha:
                            varsO.append({"cord": (row, col), "alpha": disparity[row, col]})
                            varsOnodes.append(g.add_nodes(1))
                            if col + alpha < left_img.shape[1]:
                                varsA.append({"cord": (row, col), "alpha": alpha})
                                varsAnodes.append(g.add_nodes(1))
                            else:
                                varsA.append(VAR_ABSENT)
                                varsAnodes.append(-1)
                        else:
                            varsA.append(VAR_ALPHA)
                            varsO.append(VAR_ALPHA)
                            varsAnodes.append(-1)
                            varsOnodes.append(-1)

            penalty = 0.
            for row in range(disparity.shape[0]):
                for col in range(disparity.shape[1]):
                    n = (disparity.shape[1] * row) + col
                    if is_var(varsA[n]):
                        col = varsA[n]["cord"]
                        a = varsA[n]["alpha"]
                        D = squared_dissimilarity(left_img, right_img, col, (col[0], int(col[1] + a))) - K
                        g.add_tedge(varsAnodes[n], D, 0)
                    if varsA[n] == VAR_ALPHA:
                        col = (row, col)
                        a = alpha
                        D = squared_dissimilarity(left_img, right_img, col, (col[0], int(col[1] + a))) - K
                        penalty += D
                    if is_var(varsO[n]):
                        col = varsO[n]["cord"]
                        a = varsO[n]["alpha"]
                        D = squared_dissimilarity(left_img, right_img, col, (col[0], int(col[1] + a))) - K
                        g.add_tedge(varsOnodes[n], 0, D)

            for row in range(disparity.shape[0]):
                for col in range(disparity.shape[1]):
                    i1 = (disparity.shape[1] * row) + col

                    if is_var(varsO[i1]) and is_var(varsA[i1]):
                        forbid01(g, varsOnodes[i1], varsAnodes[i1])
                        ia = int(varsO[i1]["alpha"])
                        c2 = col + ia - alpha
                        i2 = (disparity.shape[1] * row) + c2

                        forbid01(g, varsOnodes[i1], varsAnodes[i2])

                    indices = (col, row)

                    if indices[0] != disparity.shape[1] - 1:
                        r2 = row
                        c2 = col + (1)
                        if c2 + alpha < disparity.shape[1]:
                            i2 = (disparity.shape[1] * r2) + c2
                            pen = K / 5
                            d1 = int(left_img[row, col] - left_img[r2, c2])
                            d2 = int(right_img[row, col + alpha] - right_img[r2, c2 + alpha])
                            if max(d1, d2) < 8:
                                pen *= 3.

                            if is_var(varsA[i1]):
                                if is_var(varsA[i2]):
                                    pairwise_term(g, varsAnodes[i1], varsAnodes[i2], 0, pen, pen, 0)  # add term

                            if is_var(varsO[i1]):
                                if is_var(varsO[i2]):
                                    pairwise_term(g, varsAnodes[i1], varsAnodes[i2], 0, pen, pen, 0)  # add term

                            if is_var(varsA[i1]) and varsA[i2] == VAR_ALPHA:
                                g.add_tedge(varsAnodes[i1], 0, pen)
                            if is_var(varsA[i2]) and varsA[i1] == VAR_ALPHA:
                                g.add_tedge(varsAnodes[i2], 0, pen)

                            if is_var(varsO[i1]) and not is_var(varsO[i2]):
                                g.add_tedge(varsOnodes[i1], 0, pen)
                            if is_var(varsO[i2]) and not is_var(varsO[i1]):
                                g.add_tedge(varsOnodes[i2], 0, pen)

                        if indices[1] != disparity.shape[0] - 1:
                            r2 = row + 1
                            c2 = col
                            if c2 + alpha < disparity.shape[1]:
                                i2 = (disparity.shape[1] * r2) + c2
                                pen = K / 5
                                d1 = int(left_img[row, col] - left_img[r2, c2])
                                d2 = int(right_img[row, col + alpha] - right_img[r2, c2 + alpha])
                                if max(d1, d2) < 8:
                                    pen *= 3.

                                if is_var(varsA[i1]):
                                    if is_var(varsA[i2]):
                                        pairwise_term(g, varsAnodes[i1], varsAnodes[i2], 0, pen, pen,
                                                      0)

                                if is_var(varsO[i1]):
                                    if is_var(varsO[i2]):
                                        pairwise_term(g, varsAnodes[i1], varsAnodes[i2], 0, pen, pen,
                                                      0)

                                if is_var(varsA[i1]) and varsA[i2] == VAR_ALPHA:
                                    g.add_tedge(varsAnodes[i1], 0, pen)
                                if is_var(varsA[i2]) and varsA[i1] == VAR_ALPHA:
                                    g.add_tedge(varsAnodes[i2], 0, pen)

                                if is_var(varsO[i1]) and not is_var(varsO[i2]):
                                    g.add_tedge(varsOnodes[i1], 0, pen)
                                if is_var(varsO[i2]) and not is_var(varsO[i1]):
                                    g.add_tedge(varsOnodes[i2], 0, pen)

            flow = g.maxflow()
            energy = flow + penalty

            if energy < prev_e:
                prev_e = energy
                for row in range(disparity.shape[0]):
                    for col in range(disparity.shape[1]):
                        i = (disparity.shape[1] * row) + col

                        if varsOnodes[i] != -1 and g.get_segment(varsOnodes[i]) == 1:
                            disparity[row, col] = np.inf

                        if varsAnodes[i] != -1 and g.get_segment(varsAnodes[i]) == 1:
                            disparity[row, col] = alpha

    return handle_occlusion_scale(disparity)


def pairwise_term(g, n1, n2, A, B, C, D):
    g.add_tedge(n1, D, B)
    g.add_tedge(n2, 0, A - B)
    g.add_edge(n1, n2, 0, B + C - A - D)


def forbid01(g, n1, n2):
    g.add_edge(n1, n2, sys.maxsize, 0)


def is_var(v):
    return v != VAR_ALPHA and v != VAR_ABSENT


def calculate_k(alphaRange, image1, image2):
    K = int(alphaRange / 4)
    s = 0
    i = 0
    for r in range(alphaRange, image1.shape[0] - alphaRange):
        for c in range(alphaRange, image2.shape[1] - alphaRange):
            i += 1
            da_s = [squared_dissimilarity(image1, image2, (r, c), (r, c + alpha)) for alpha in range(alphaRange)]
            da_s.sort()
            s += da_s[K]
    return int(s / i)


def squared_dissimilarity(image1, image2, p1, p2):
    return (image1[p1] - image2[p2]) ** 2
