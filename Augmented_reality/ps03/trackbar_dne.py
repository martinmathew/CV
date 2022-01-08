import cv2
import numpy as np



# Function documentation can be found in https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=createtrackbar

def nothing(x):
    pass


def createImg():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (100, 100), 30, (255, 255, 255), 50)
    img = cv2.medianBlur(img, 5)
    return img


#img = cv2.imread('input_images/test_images/stop_249_149_background.png')
#img = cv2.imread('input_images/scene_stp_1.png')
#img = cv2.imread('input_images/scene_constr_1.png')
#img = cv2.imread('input_images/test_images/stop_249_149_blank.png')
# img = cv2.imread('input_images/test_images/no_entry_145_145_background.png')
#
# img = cv2.imread('input_images/test_images/no_entry_145_145_blank.png')
#img = cv2.imread('input_images/scene_all_signs.png')

#img = cv2.imread('input_images/scene_some_signs_noisy.png')
#img = cv2.imread('input_images/scene_all_signs_noisy.png')
#img = cv2.imread('input_images/real_images/no_entry/5833993298_4b09eec863_o.jpg')
#img = cv2.imread('input_images/real_images/no_entry/3328387196_6cb541db37_o.jpg')
img = cv2.imread('input_images/sim_noisy_scene_1.jpg')

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
dp="dp"
minDist="minDist"
param1="param1"
param2="param2"
minRadius="minRadius"
maxRadius="maxRadius"
blockSize="blockSize"
C="C"
median_val = "median_val"
ke = "ke"

cv2.namedWindow(window)
# cv2.createTrackbar(k, window, 9, 100, nothing)
# cv2.createTrackbar(rho, window, 1, 100, nothing)
# cv2.createTrackbar(threshold, window, 20, 100, nothing)
# cv2.createTrackbar(min_line_length, window, 20, 100, nothing)
# cv2.createTrackbar(max_line_gap, window, 20, 100, nothing)
# cv2.createTrackbar(threshold1, window, 50, 200, nothing)
# cv2.createTrackbar(threshold2, window, 150, 200, nothing)
# cv2.createTrackbar(apertureSize, window, 3, 7, nothing)
# cv2.createTrackbar(dk, window, 4, 20, nothing)


cv2.createTrackbar(dp, window, 1, 100, nothing)
cv2.createTrackbar(minDist, window, 20, 200, nothing)
cv2.createTrackbar(param1, window, 50, 100, nothing)
cv2.createTrackbar(param2, window, 30, 100, nothing)
cv2.createTrackbar(minRadius, window, 0, 200, nothing)
cv2.createTrackbar(maxRadius, window, 0, 100, nothing)
cv2.createTrackbar(blockSize, window, 11, 100, nothing)
cv2.createTrackbar(C, window, 2, 100, nothing)
cv2.createTrackbar(ke, window, 7, 100, nothing)

while 1:
    temp = img.copy()

    k1 = cv2.waitKey(1) & 0xFF

    if k1 == 27:
        break

    k = cv2.getTrackbarPos('k', 'Params')
    if k%2 == 0:
        k = k+1

    #rho = cv2.getTrackbarPos('rho', 'Params')

    # theta = cv2.getTrackbarPos('theta', 'Params')

    #threshold = cv2.getTrackbarPos('threshold', 'Params')

    #min_line_length = cv2.getTrackbarPos('min_line_length', 'Params')

    #max_line_gap = cv2.getTrackbarPos('max_line_gap', 'Params')

    #threshold1 = cv2.getTrackbarPos('threshold1', 'Params')

    #threshold2 = cv2.getTrackbarPos('threshold2', 'Params')

    #apertureSize = cv2.getTrackbarPos('apertureSize', 'Params')

    #dk = cv2.getTrackbarPos('dilation kernel', 'Params')


    dp = cv2.getTrackbarPos('dp', 'Params')
    minDist = cv2.getTrackbarPos('minDist', 'Params')
    param1 = cv2.getTrackbarPos('param1', 'Params')
    param2 = cv2.getTrackbarPos('param2', 'Params')
    minRadius = cv2.getTrackbarPos('minRadius', 'Params')
    maxRadius = cv2.getTrackbarPos('maxRadius', 'Params')
    ke = cv2.getTrackbarPos('ke', 'Params')

    # blockSize = cv2.getTrackbarPos('blockSize', 'Params')
    # C = cv2.getTrackbarPos('C', 'Params')
    #
    #
    # if blockSize%2 == 0:
    #     blockSize = blockSize +1
    # # if apertureSize%2 == 0:
    # #     apertureSize = apertureSize+1
    # #     if apertureSize > 7:
    # #         apertureSize = 7
    # temp = cv2.medianBlur(temp, 7)
    # gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    #gray = cv.cvtColor(img_in, cv.COLOR_BGR2GRAY)
    # gray[gray == 0] = 255
    # gray[gray != 255] = 0



    #kernel = np.ones((dk, dk), np.uint8)

    #dilation = cv2.dilate(gray, kernel, iterations=1)

    #blur = cv2.GaussianBlur(gray, (k, k), 0)
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.BORDER_REPLICATE, cv2.THRESH_BINARY, blockSize=blockSize, C=C)
    #
    # thresh = cv2.bitwise_not(thresh)
    # Apply edge detection method on the image
    # edges = cv2.Canny(thresh, threshold1, threshold2, apertureSize=apertureSize)

    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
    #                           param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    # temp = ps2.denoise(temp)

    #temp = cv2.cvtColor(temp, cv2.COLOR_BGR@)
    if ke%2 == 0:
        ke = ke + 1
    # lower_red = (160, 160, 160)
    # upper_red = (255, 255, 255)
    # mask = cv2.inRange(temp, lower_red, upper_red)
    # res = cv2.bitwise_and(temp, temp, mask=mask)

    # res = cv2.bitwise_and(temp, temp, mask=mask)

    # gray = cv2.split(res)[2]
    # kernel = np.ones((ke, ke), np.uint8)
    # img_dilation = cv2.erode(gray, kernel, iterations=1)
    gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                              param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    # blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply edge detection method on the image
    #edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    if rho == -1:
        rho = 10
    if threshold == -1:
        threshold = 50
    if min_line_length == -1:
        min_line_length = 50
    if max_line_gap == -1:
        max_line_gap = 20

    # lines = cv2.HoughLinesP(edges, rho=rho, theta=np.pi / 180, threshold=threshold,
    #                         minLineLength=min_line_length, maxLineGap=max_line_gap)
    # if lines is not None:
    #     print(len(lines))
    #     for line in lines:
    #         for x1, y1, x2, y2 in line:
    #             cv2.line(temp, (x1, y1), (x2, y2), (255, 0, 0), 5)
    if circles is not None:
        for i in circles[0, :]:
        # draw the outer circle
            cv2.circle(temp, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
            cv2.circle(temp, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow("Test", temp)
