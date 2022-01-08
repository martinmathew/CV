import cv2
import numpy as np
import cv2 as cv
import math


def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def cluster(points, thr):
    dict = {}
    index = 0
    for i in range(len(points[0])):
        flat_list = [item for sublist in dict.values() for item in sublist]
        if i in flat_list:
            continue
        index = index + 1
        dict[index] = [i]
        a = (points[0][i], points[1][i])
        for j in range(i + 1, len(points[0])):
            b = (points[0][j], points[1][j])
            if distance(a, b) < thr:
                flat_list = [item for sublist in dict.values() for item in sublist]
                if j not in flat_list:
                    list = dict[index]
                    list.append(j)
                    dict[index] = list
    return dict


def getCenters(dict, point):
    centers = []
    for key in dict.keys():
        list = dict[key]
        x = 0
        y = 0
        for i in list:
            x = x + point[0][i]
            y = y + point[1][i]
        x = int(x / len(list))
        y = int(y / len(list))
        centers.append((x, y))
    return centers


def denoise(img_in):
    temp = cv.fastNlMeansDenoisingColored(img_in, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    # temp = cv.medianBlur(temp, 5)
    return temp


# Function documentation can be found in https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=createtrackbar

def linelength(line):
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]
    # print(line)
    if (x1 - x2) ** 2 + (y1 - y2) ** 2 < 0:
        return 999999999
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def intersection(a, b, c, d):
    a1 = b[1] - a[1]
    b1 = a[0] - b[0]

    c1 = a1 * (a[0]) + b1 * (a[1])

    a2 = d[1] - c[1]

    b2 = c[0] - d[0]

    c2 = a2 * (c[0]) + b2 * (c[1]);

    determinant = a1 * b2 - a2 * b1;

    if (determinant == 0):
        return None
    else:
        x = (b2 * c1 - b1 * c2) / determinant;

        y = (a1 * c2 - a2 * c1) / determinant;

        ax = linelength((a[0], a[1], int(x), int(y)))
        xb = linelength((int(x), int(y), b[0], b[1]))
        ab = linelength((a[0], a[1], b[0], b[1]))

        cx = linelength((c[0], c[1], int(x), int(y)))
        xd = linelength((int(x), int(y), d[0], d[1]))
        cd = linelength((c[0], c[1], d[0], d[1]))

        if abs(ax - ab / 2) <= 1.0 and abs(xb - ab / 2) <= 1.0 or abs(cx - cd / 2) <= 1.0 and abs(xd - cd / 2) <= 1.0:
            return (int(x), int(y))

        return None


def nothing(x):
    pass


def createImg():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (100, 100), 30, (255, 255, 255), 50)
    img = cv2.medianBlur(img, 5)
    return img


# img = cv2.imread('input_images/test_images/stop_249_149_background.png')
# img = cv2.imread('input_images/scene_stp_1.png')
# img = cv2.imread('input_images/scene_constr_1.png')
# img = cv2.imread('input_images/test_images/stop_249_149_blank.png')
# img = cv2.imread('input_images/test_images/no_entry_145_145_background.png')
#
# img = cv2.imread('input_images/test_images/no_entry_145_145_blank.png')
# img = cv2.imread('input_images/scene_all_signs.png')
# img = cv2.imread('input_images/scene_some_signs.png')

# img = cv2.imread('input_images/scene_some_signs_noisy.png')
# img = cv2.imread('input_images/sim_noisy_scene_1.jpg')
# img = cv2.imread('input_images/sim_noisy_scene_2.jpg')

img = cv2.imread('input_images/ps3-2-d_base.jpg')
template_img = cv2.imread('input_images/template.jpg')

rho = "rho"
threshold = "threshold"
min_line_length = "min_line_length"
max_line_gap = "max_line_gap"
window = "Params"
k = "k"
dk = "dilation kernel"
threshold1 = "threshold1"
threshold2 = "threshold2"
apertureSize = "apertureSize"
dp = "dp"
minDist = "minDist"
param1 = "param1"
param2 = "param2"
minRadius = "minRadius"
maxRadius = "maxRadius"
blockSize = "blockSize"
C = "C"
median_val = "median_val"
blk = "Block Size"

cv2.namedWindow(window)
# cv2.createTrackbar(k, window, 9, 100, nothing)

# cv2.createTrackbar(rho, window, 2, 100, nothing)
cv2.createTrackbar(threshold, window, 23, 100, nothing)
# cv2.createTrackbar(min_line_length, window, 15, 100, nothing)
# cv2.createTrackbar(max_line_gap, window, 53, 100, nothing)
# cv2.createTrackbar(threshold1, window, 15, 200, nothing)
# cv2.createTrackbar(threshold2, window, 45, 200, nothing)
# cv2.createTrackbar(apertureSize, window, 3, 7, nothing)

cv2.createTrackbar(dk, window, 1, 200, nothing)

cv2.createTrackbar(dp, window, 2, 100, nothing)
cv2.createTrackbar(minDist, window, 334, 500, nothing)
cv2.createTrackbar(param1, window, 39, 200, nothing)
cv2.createTrackbar(param2, window, 34, 200, nothing)
cv2.createTrackbar(minRadius, window, 22, 200, nothing)
cv2.createTrackbar(maxRadius, window, 37, 200, nothing)

# cv2.createTrackbar(blockSize, window, 11, 100, nothing)
# cv2.createTrackbar(C, window, 2, 100, nothing)

cv2.createTrackbar(blk, window, 2, 100, nothing)

while 1:
    temp = img.copy()
    temp1 = img.copy()
    template = template_img.copy()
    k1 = cv2.waitKey(1) & 0xFF

    if k1 == 27:
        break

    k = cv2.getTrackbarPos('k', 'Params')
    if k % 2 == 0:
        k = k + 1

    # rho = cv2.getTrackbarPos('rho', 'Params')

    # theta = cv2.getTrackbarPos('theta', 'Params')

    threshold = cv2.getTrackbarPos('threshold', 'Params')

    # min_line_length = cv2.getTrackbarPos('min_line_length', 'Params')

    # max_line_gap = cv2.getTrackbarPos('max_line_gap', 'Params')

    # threshold1 = cv2.getTrackbarPos('threshold1', 'Params')

    # threshold2 = cv2.getTrackbarPos('threshold2', 'Params')

    # apertureSize = cv2.getTrackbarPos('apertureSize', 'Params')

    dk = cv2.getTrackbarPos('dilation kernel', 'Params')

    blk = cv2.getTrackbarPos('Block Size', 'Params')

    dp = cv2.getTrackbarPos('dp', 'Params')
    minDist = cv2.getTrackbarPos('minDist', 'Params')
    param1 = cv2.getTrackbarPos('param1', 'Params')
    param2 = cv2.getTrackbarPos('param2', 'Params')
    minRadius = cv2.getTrackbarPos('minRadius', 'Params')
    maxRadius = cv2.getTrackbarPos('maxRadius', 'Params')
    #
    # blockSize = cv2.getTrackbarPos('blockSize', 'Params')
    # C = cv2.getTrackbarPos('C', 'Params')

    # if blockSize%2 == 0:
    #     blockSize = blockSize +1
    # if apertureSize % 2 == 0:
    #     apertureSize = apertureSize + 1
    #     if apertureSize > 7:
    #         apertureSize = 7

    if rho == -1:
        rho = 10
    if threshold == -1:
        threshold = 50
    if min_line_length == -1:
        min_line_length = 50
    if max_line_gap == -1:
        max_line_gap = 20

    lower_black = (150, 150, 150)
    upper_black = (255, 255, 255)
    temp = denoise(temp)
    mask = cv.inRange(temp, lower_black, upper_black)
    res = cv.bitwise_and(temp, temp, mask=mask)

    gray = cv.split(res)[2]
    # kernel = np.ones((0, 0), np.uint8)
    #
    # kernel1 = np.ones((dk, dk), np.uint8)
    # den = denoise(temp)
    # temp = denoise(temp)
    # gray = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
    _, gray = cv.threshold(gray, 180, 255, cv.THRESH_BINARY)
    # #
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                               param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    # template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    # _, template_gray = cv.threshold(template_gray, 127, 255, cv.THRESH_BINARY_INV)
    # temp = cv2.erode(gray, kernel1, iterations=1)
    #
    # temp = np.float32(temp)
    # free = threshold/1000
    # dst = cv.cornerHarris(temp, blk, 3, free)
    # res = np.where(dst == dst.max())
    #
    # cluster_dict = cluster(res, 30)
    # center = getCenters(cluster_dict, res)


    # if center is not None:
    #     print("Circle - {}".format(len(center)))
    #     for i in center:
    #         # draw the outer circle
    #         # cv2.circle(temp, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #         # draw the center of the circle
    #         cv2.circle(temp1, ( i[1],i[0]), 2, (0, 255, 0), 1)




    # _, temp = cv.threshold(temp, 127, 255, cv.THRESH_BINARY_INV)
    # temp = cv2.dilate(temp, kernel1, iterations=1)

    # ret, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # gray = cv2.medianBlur(gray, 7)

    #
    # temp = np.float32(gray)
    # dst = cv.cornerHarris(temp, blk, 3, 0.04)
    # res = np.where(dst == dst.max())
    #
    # cluster_dict = cluster(res, 30)
    # center = getCenters(cluster_dict, res)
    #
    #
    # if center is not None:
    #     print("Circle - {}".format(len(center)))
    #     for i in center:
    #         # draw the outer circle
    #         # cv2.circle(temp, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #         # draw the center of the circle
    #         cv2.circle(temp1, ( i[1],i[0]), 2, (0, 255, 0), 1)

    if circles is not None:
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(temp1, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(temp1, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow("Test", gray)
    cv2.imshow("Test1", temp1)
