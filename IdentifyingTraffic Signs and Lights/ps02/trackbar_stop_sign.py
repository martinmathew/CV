import cv2
import numpy as np
import ps2


# Function documentation can be found in https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=createtrackbar

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

#img = cv2.imread('input_images/scene_some_signs_noisy.png')
img = cv2.imread('input_images/challenge_images/img-5-a-2.png')

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

cv2.namedWindow(window)
# cv2.createTrackbar(k, window, 9, 100, nothing)

cv2.createTrackbar(rho, window, 1, 100, nothing)
cv2.createTrackbar(threshold, window, 59, 100, nothing)
cv2.createTrackbar(min_line_length, window, 20, 100, nothing)
cv2.createTrackbar(max_line_gap, window, 20, 100, nothing)
cv2.createTrackbar(threshold1, window, 50, 200, nothing)
cv2.createTrackbar(threshold2, window, 150, 200, nothing)
cv2.createTrackbar(apertureSize, window, 3, 7, nothing)

# cv2.createTrackbar(dk, window, 4, 20, nothing)


# cv2.createTrackbar(dp, window, 1, 100, nothing)
# cv2.createTrackbar(minDist, window, 20, 100, nothing)
# cv2.createTrackbar(param1, window, 50, 100, nothing)
# cv2.createTrackbar(param2, window, 30, 100, nothing)
# cv2.createTrackbar(minRadius, window, 0, 100, nothing)
# cv2.createTrackbar(maxRadius, window, 0, 100, nothing)

# cv2.createTrackbar(blockSize, window, 11, 100, nothing)
# cv2.createTrackbar(C, window, 2, 100, nothing)

while 1:
    temp = img.copy()

    k1 = cv2.waitKey(1) & 0xFF

    if k1 == 27:
        break

    k = cv2.getTrackbarPos('k', 'Params')
    if k % 2 == 0:
        k = k + 1

    rho = cv2.getTrackbarPos('rho', 'Params')

    # theta = cv2.getTrackbarPos('theta', 'Params')

    threshold = cv2.getTrackbarPos('threshold', 'Params')

    min_line_length = cv2.getTrackbarPos('min_line_length', 'Params')

    max_line_gap = cv2.getTrackbarPos('max_line_gap', 'Params')

    threshold1 = cv2.getTrackbarPos('threshold1', 'Params')

    threshold2 = cv2.getTrackbarPos('threshold2', 'Params')

    apertureSize = cv2.getTrackbarPos('apertureSize', 'Params')

    # dk = cv2.getTrackbarPos('dilation kernel', 'Params')

    # dp = cv2.getTrackbarPos('dp', 'Params')
    # minDist = cv2.getTrackbarPos('minDist', 'Params')
    # param1 = cv2.getTrackbarPos('param1', 'Params')
    # param2 = cv2.getTrackbarPos('param2', 'Params')
    # minRadius = cv2.getTrackbarPos('minRadius', 'Params')
    # maxRadius = cv2.getTrackbarPos('maxRadius', 'Params')
    #
    # blockSize = cv2.getTrackbarPos('blockSize', 'Params')
    # C = cv2.getTrackbarPos('C', 'Params')

    # if blockSize%2 == 0:
    #     blockSize = blockSize +1
    # if apertureSize%2 == 0:
    #     apertureSize = apertureSize+1
    #     if apertureSize > 7:
    #         apertureSize = 7
    # temp = ps2.denoise(temp)
    # lower_red = (0, 0, 253)
    # upper_red = (10, 10, 255)
    #
    # lower_white = (245, 245, 245)
    # upper_white = (255, 255, 255)
    # red_mask = cv2.inRange(temp, lower_red, upper_red)
    # white_mask = cv2.inRange(temp, lower_white, upper_white)
    # mask = red_mask + white_mask
    #
    # res = cv2.bitwise_and(temp, temp, mask=mask)
    # res = cv2.split(res)[2]
    # # gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # gray = cv2.threshold(res, 127, 255, cv2.THRESH_BINARY)[1]
    # kernel = np.ones((5, 5), np.uint8)
    # gray = cv2.erode(gray, kernel, iterations=1)
    # gray[gray == 0] = 255
    # gray[gray != 255] = 0

    # kernel = np.ones((dk, dk), np.uint8)

    # dilation = cv2.dilate(gray, kernel, iterations=1)

    # blur = cv2.GaussianBlur(gray, (k, k), 0)
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.BORDER_REPLICATE, cv2.THRESH_BINARY, blockSize=blockSize, C=C)
    #
    # thresh = cv2.bitwise_not(thresh)
    # Apply edge detection method on the image
    # edges = cv2.Canny(thresh, threshold1, threshold2, apertureSize=apertureSize)

    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
    #                           param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    # blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply edge detection method on the image
    # temp = ps2.denoise(temp)
    # hsv_img = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)

    light_orange = (0, 0, 70)
    dark_orange = (25, 20, 100)

    mask = cv2.inRange(temp, light_orange, dark_orange)

    res = cv2.bitwise_and(temp, temp, mask=mask)

    gray = cv2.split(res)[2]
    #
    ret, gray = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    gray = cv2.dilate(gray, (55, 55), iterations=1)
    # gray = cv2.medianBlur(gray, 7)

    if rho == -1:
        rho = 10
    if threshold == -1:
        threshold = 50
    if min_line_length == -1:
        min_line_length = 50
    if max_line_gap == -1:
        max_line_gap = 20

    edges = cv2.Canny(gray, threshold1=threshold1, threshold2=threshold2, apertureSize=apertureSize)
    lines = cv2.HoughLinesP(edges, rho=rho, theta=np.pi / 180, threshold=threshold,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)


    if lines is None or len(lines) < 8:
        print(len(lines))
        continue
    res = []
    linek = []
    for i in range(len(lines)):
        line1 = lines[i][0]
        for j in range(i + 1, len(lines)):
            line2 = lines[j][0]
            for k in range(j + 1, len(lines)):
                line3 = lines[k][0]
                for m in range(k + 1, len(lines)):
                    line4 = lines[m][0]
                    for n in range(m + 1, len(lines)):
                        line5 = lines[n][0]
                        for o in range(n + 1, len(lines)):
                            line6 = lines[o][0]
                            for p in range(o + 1, len(lines)):
                                line7 = lines[p][0]
                                for q in range(p + 1, len(lines)):
                                    line8 = lines[q][0]
                                    # str = str(linelength(line1))+":"str(linelength(line2))
                                    # res.append(str)
                                    linek.append(line1)
                                    linek.append(line2)
                                    linek.append(line3)
                                    if ps2.isOctagon(line1, line2, line3, line4, line5, line6, line7, line8):
                                        midx, midy = (line1[0] + line2[0] + line3[0] + line4[0] + line5[0] + line6[0] +
                                                      line7[0] + line8[0] + line1[2] + line2[2] + line3[2] + line4[2] +
                                                      line5[2] + line6[2] + line7[2] + line8[2]) / 16, (
                                                             line1[1] + line2[1] + line3[1] + line4[1] + line5[1] +
                                                             line6[1] + line7[1] + line8[1] + line1[3] + line2[3] +
                                                             line3[3] + line4[3] + line5[3] + line6[3] + line7[3] +
                                                             line8[3]) / 16
                                        cv2.circle(temp, (int(midx), int(midy)), 1, (0, 255, 0), 3)
                                        break



    #
    # # circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
    # #                            param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    if lines is not None:
        print(len(lines))
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(temp, (x1, y1), (x2, y2), (255, 0, 0), 5)
    # if circles is not None:
    #     for i in circles[0, :]:
    #         # draw the outer circle
    #         cv2.circle(temp, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #         # draw the center of the circle
    #         cv2.circle(temp, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow("Test", temp)
