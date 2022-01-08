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

img = cv2.imread('input_images/scene_some_signs_noisy.png')

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
h = "h"
templateWindowSize = "templateWindowSize"
searchWindowSize = "searchWindowSize"

cv2.namedWindow(window)


cv2.createTrackbar(k, window, 3, 100, nothing)
cv2.createTrackbar(h, window, 10, 100, nothing)
cv2.createTrackbar(templateWindowSize, window, 7, 100, nothing)
cv2.createTrackbar(searchWindowSize, window, 21, 100, nothing)
count = 0
while 1:
    temp = img.copy()

    k1 = cv2.waitKey(1) & 0xFF

    if k1 == 27:
        break

    k = cv2.getTrackbarPos('k', 'Params')
    if k % 2 == 0:
        k = k + 1

    h = cv2.getTrackbarPos('h', 'Params')
    templateWindowSize = cv2.getTrackbarPos('templateWindowSize', 'Params')
    searchWindowSize = cv2.getTrackbarPos('searchWindowSize', 'Params')

    if templateWindowSize % 2 == 0:
        templateWindowSize = templateWindowSize + 1
    if searchWindowSize % 2 == 0:
        searchWindowSize = searchWindowSize + 1
    #dst = np.copy(temp)*0
    #temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)

    temp = cv2.fastNlMeansDenoisingColored(temp, None, h=h, hColor=h, templateWindowSize=templateWindowSize, searchWindowSize = searchWindowSize)
    temp = cv2.medianBlur(temp, k)
    # dict = ps2.traffic_sign_detection(dst)

    # for i in list(dict.values()):
    #     cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

    print("processed {}".format(count))
    count = count + 1



    cv2.imshow("Test", temp)
