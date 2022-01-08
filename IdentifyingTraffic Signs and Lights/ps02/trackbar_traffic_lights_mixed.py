import cv2
import cv2 as cv
import numpy as np
import math


# Function documentation can be found in https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=createtrackbar

def nothing(x):
    pass


def line_len(line):
    return math.sqrt((line[0][0] - line[0][2]) ** 2 + (line[0][1] - line[0][3]) ** 2)


def denoise(img_in):
    temp = cv.medianBlur(img_in, 5)
    dst = cv.fastNlMeansDenoisingColored(temp, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    return dst


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
#img = cv2.imread('input_images/scene_some_signs_noisy.png')
#img = cv2.imread('input_images/scene_all_signs_noisy.png')

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
median_blur = "median_blur"
max_length = "max_length"

cv2.namedWindow(window)
# cv2.createTrackbar(k, window, 9, 100, nothing)
# cv2.createTrackbar(rho, window, 1, 100, nothing)
# cv2.createTrackbar(threshold, window, 20, 100, nothing)
# cv2.createTrackbar(min_line_length, window, 20, 100, nothing)
# cv2.createTrackbar(max_line_gap, window, 20, 100, nothing)
cv2.createTrackbar(threshold1, window, 50, 200, nothing)
cv2.createTrackbar(threshold2, window, 150, 200, nothing)
cv2.createTrackbar(apertureSize, window, 3, 7, nothing)
# cv2.createTrackbar(dk, window, 4, 20, nothing)


cv2.createTrackbar(dp, window, 1, 100, nothing)
cv2.createTrackbar(minDist, window, 20, 100, nothing)
cv2.createTrackbar(param1, window, 50, 100, nothing)
cv2.createTrackbar(param2, window, 30, 100, nothing)
cv2.createTrackbar(minRadius, window, 0, 100, nothing)
cv2.createTrackbar(maxRadius, window, 0, 100, nothing)
# cv2.createTrackbar(blockSize, window, 11, 100, nothing)
# cv2.createTrackbar(C, window, 2, 100, nothing)
# cv2.createTrackbar(median_blur, window, 7, 100, nothing)
# cv2.createTrackbar(max_length, window, 1, 100, nothing)

while 1:
    temp = img.copy()

    k1 = cv2.waitKey(1) & 0xFF

    if k1 == 27:
        break

    k = cv2.getTrackbarPos('k', 'Params')
    if k % 2 == 0:
        k = k + 1

    # rho = cv2.getTrackbarPos('rho', 'Params')

    # theta = cv2.getTrackbarPos('theta', 'Params')

    # threshold = cv2.getTrackbarPos('threshold', 'Params')
    #
    # min_line_length = cv2.getTrackbarPos('min_line_length', 'Params')
    #
    # max_line_gap = cv2.getTrackbarPos('max_line_gap', 'Params')
    #
    threshold1 = cv2.getTrackbarPos('threshold1', 'Params')
    #
    threshold2 = cv2.getTrackbarPos('threshold2', 'Params')
    # #
    apertureSize = cv2.getTrackbarPos('apertureSize', 'Params')
    #
    # median_blur = cv2.getTrackbarPos('median_blur', 'Params')
    #
    # max_length = cv2.getTrackbarPos('max_length', 'Params')

    # dk = cv2.getTrackbarPos('dilation kernel', 'Params')

    dp = cv2.getTrackbarPos('dp', 'Params')
    minDist = cv2.getTrackbarPos('minDist', 'Params')
    param1 = cv2.getTrackbarPos('param1', 'Params')
    param2 = cv2.getTrackbarPos('param2', 'Params')
    minRadius = cv2.getTrackbarPos('minRadius', 'Params')
    maxRadius = cv2.getTrackbarPos('maxRadius', 'Params')
    #
    # blockSize = cv2.getTrackbarPos('blockSize', 'Params')
    # C = cv2.getTrackbarPos('C', 'Params')

    # if blockSize % 2 == 0:
    #     blockSize = blockSize + 1
    # if median_blur % 2 == 0:
    #     median_blur = median_blur + 1
    if apertureSize % 2 == 0:
        apertureSize = apertureSize + 1
        if apertureSize > 7:
            apertureSize = 7

    if rho == -1:
        rho = 10
    if threshold == -1:
        threshold = 50
    if min_line_length == -1:
        min_line_length = 50
    if max_line_gap == -1:
        max_line_gap = 20

    lower_black = (10, 10, 10)
    upper_black = (75, 75, 75)
    temp = denoise(temp)
    mask = cv.inRange(temp, lower_black, upper_black)
    res = cv.bitwise_and(temp, temp, mask=mask)

    gray = cv.split(res)[2]
    gray = cv.threshold(gray, 25, 255, cv.THRESH_BINARY)[1]
    #gray = cv2.dilate(gray, (5, 5), iterations=1)

    # edges = cv2.Canny(mask, threshold1, threshold2, apertureSize=apertureSize)

    # lines = cv2.HoughLinesP(edges, rho=rho, theta=np.pi / 180, threshold=threshold,
    #                         minLineLength=min_line_length, maxLineGap=max_line_gap)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                               param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    if circles is not None:
        print(len(circles[0]))
        for i in circles[0, :]:
            # draw the outer circle
            #cv2.circle(temp, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(temp, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow("Test", temp)
