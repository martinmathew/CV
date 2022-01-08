import cv2
import numpy as np
import cv2 as cv
import math
from scipy import ndimage, misc
import ps3


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


def nothing(x):
    pass


def denoise(img_in):
    dst = cv.fastNlMeansDenoisingColored(img_in, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    # temp = cv.medianBlur(dst, 3)
    return dst


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


def convert(tup):
    di = {}
    for a, b in tup:
        di[a] = b
    return di


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
temp_img = cv2.imread("input_images/template_proc.jpg")
# img = cv2.imread('input_images/sim_noisy_scene_2.jpg')


img = cv2.imread('input_images/ps3-2-e_base.jpg')
# template_img = cv2.imread('input_images/template.jpg')
# img = cv2.imread('input_images/sim_clear_scene.jpg')

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

gray_scale_threshold = "Gray Scale Threshold"

#cv2.namedWindow(window)
# cv2.createTrackbar(k, window, 9, 100, nothing)

# cv2.createTrackbar(rho, window, 2, 100, nothing)
#cv2.createTrackbar(threshold, window, 51, 100, nothing)
# cv2.createTrackbar(min_line_length, window, 15, 100, nothing)
# cv2.createTrackbar(max_line_gap, window, 53, 100, nothing)
# cv2.createTrackbar(threshold1, window, 15, 200, nothing)
# cv2.createTrackbar(threshold2, window, 45, 200, nothing)
# cv2.createTrackbar(apertureSize, window, 3, 7, nothing)
#
# cv2.createTrackbar(dk, window, 3, 200, nothing)
# cv2.createTrackbar(dp, window, 1, 100, nothing)
# cv2.createTrackbar(dp, window, 1, 100, nothing)
# cv2.createTrackbar(minDist, window, 20, 100, nothing)
# cv2.createTrackbar(param1, window, 50, 100, nothing)
# cv2.createTrackbar(param2, window, 30, 100, nothing)
# cv2.createTrackbar(minRadius, window, 0, 100, nothing)
# cv2.createTrackbar(maxRadius, window, 0, 100, nothing)

# cv2.createTrackbar(blockSize, window, 11, 100, nothing)
# cv2.createTrackbar(C, window, 2, 100, nothing)

# cv2.createTrackbar(blk, window, 2, 100, nothing)

#cv2.createTrackbar(gray_scale_threshold, window, 186, 255, nothing)

while 1:
    inp_img_files = {'ps3-2-a_base.jpg': [(119, 166), (115, 537), (919, 175), (918, 540)],
                     'ps3-2-b_base.jpg': [(202, 242), (208, 529), (884, 203), (885, 564)],
                     'ps3-2-c_base.jpg': [(365, 89), (371, 677), (641, 93), (633, 673)],
                     'ps3-2-d_base.jpg': [(205, 181), (156, 498), (923, 282), (869, 622)],
                     'ps3-2-e_base.jpg': [(53, 205), (101, 573), (943, 198), (902, 577)]}
    for threshold in range(1, 100):
        for gray_scale_threshold in range(181, 255):
            print("Trying Threshold - {}, Gray Scale - {}".format(threshold, gray_scale_threshold))
            results = {}
            for img_file in inp_img_files.keys():
                results[img_file] = False
                img = cv2.imread('input_images/{}'.format(img_file))
                temp = img.copy()
                temp1 = img.copy()
                template_img1 = temp_img.copy()
                template = temp_img.copy()

                temp = denoise(temp)
                gray = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
                _, gray = cv.threshold(gray, gray_scale_threshold, 255, cv.THRESH_BINARY_INV)

                gray_template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
                _, gray_template = cv.threshold(gray_template, 44, 255, cv.THRESH_BINARY_INV)

                for angle in range(0, 360, 15):
                    # print("Angle - {}".format(angle))
                    rotated_img = ndimage.rotate(gray_template, angle, reshape=False)
                    # cv2.imshow("Test11", rotated_img)
                    res = cv2.matchTemplate(gray, rotated_img, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= threshold / 100)

                    if loc is not None and len(loc[::-1][0]) > 4:
                        # print(len(loc[::-1][0]))
                        x = []
                        y = []
                        for pt in zip(*loc[::-1]):
                            # print("{} - {}".format(pt[0], pt[1]))
                            x.append(int(pt[0] + int(template.shape[0] / 2)))
                            y.append(int(pt[1] + int(template.shape[1]) / 2))

                        pts = [x, y]
                        dict = cluster(pts, 5)
                        if len(dict) != 4:
                            # print("Four Centers not found")
                            continue
                        dict = sorted(dict.items(), key=lambda x: len(x[1]), reverse=True)
                        dict = convert(dict[:4])
                        markers = getCenters(dict, pts)
                        markers = ps3.sort_by_leftright(markers)
                        for i in range(len(markers)):
                            if distance(markers[i], inp_img_files[img_file][i]) > 5:
                                print("Distance GReater That Threshold for - {}".format(markers[i]))
                                break
                        print("Found Solution for - {}".format(img_file))
                        results[img_file] = True
            if all(list(results.values())):
                print("Found Threshold - {}, Gray Scale - {}".format(threshold, gray_scale_threshold))

    # temp = img.copy()
    # temp1 = img.copy()
    # template_img1 = temp_img.copy()
    # template = temp_img.copy()
    # k1 = cv2.waitKey(1) & 0xFF
    #
    # if k1 == 27:
    #     break

    # k = cv2.getTrackbarPos('k', 'Params')
    # if k % 2 == 0:
    #     k = k + 1

    # rho = cv2.getTrackbarPos('rho', 'Params')

    # theta = cv2.getTrackbarPos('theta', 'Params')

    # threshold = cv2.getTrackbarPos('threshold', 'Params')
    #
    # min_line_length = cv2.getTrackbarPos('min_line_length', 'Params')
    #
    # max_line_gap = cv2.getTrackbarPos('max_line_gap', 'Params')
    #
    # threshold1 = cv2.getTrackbarPos('threshold1', 'Params')
    #
    # threshold2 = cv2.getTrackbarPos('threshold2', 'Params')
    #
    # apertureSize = cv2.getTrackbarPos('apertureSize', 'Params')
    #
    # dk = cv2.getTrackbarPos('dilation kernel', 'Params')
    #
    # blk = cv2.getTrackbarPos('Block Size', 'Params')

    # dp = cv2.getTrackbarPos('dp', 'Params')
    # minDist = cv2.getTrackbarPos('minDist', 'Params')
    # param1 = cv2.getTrackbarPos('param1', 'Params')
    # param2 = cv2.getTrackbarPos('param2', 'Params')
    # minRadius = cv2.getTrackbarPos('minRadius', 'Params')
    # maxRadius = cv2.getTrackbarPos('maxRadius', 'Params')

    # gray_scale_threshold = cv2.getTrackbarPos('Gray Scale Threshold', 'Params')
    #
    # blockSize = cv2.getTrackbarPos('blockSize', 'Params')
    # C = cv2.getTrackbarPos('C', 'Params')

    # if blockSize%2 == 0:
    #     blockSize = blockSize +1
    # if apertureSize % 2 == 0:
    #     apertureSize = apertureSize + 1
    #     if apertureSize > 7:
    #         apertureSize = 7

    # kernel = np.ones((0, 0), np.uint8)

    # kernel1 = np.ones((dk, dk), np.uint8)
    # den = denoise(temp)

    # temp = denoise(temp)
    # gray = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
    # _, gray = cv.threshold(gray, gray_scale_threshold, 255, cv.THRESH_BINARY_INV)
    #
    # gray_template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    # _, gray_template = cv.threshold(gray_template, 44, 255, cv.THRESH_BINARY_INV)
    #
    # # cv2.imshow("Test1", gray)
    # cv2.imshow("Test2", gray_template)
    # for angle in range(0, 360, 15):
    #     print("Angle - {}".format(angle))
    #     rotated_img = ndimage.rotate(gray_template, angle,  reshape=False)
    #     # cv2.imshow("Test11", rotated_img)
    #     res = cv2.matchTemplate(gray, rotated_img, cv2.TM_CCOEFF_NORMED)
    #     loc = np.where(res >= threshold/100)
    #
    #     if loc is not None and len(loc[::-1][0]) > 4:
    #         # print(len(loc[::-1][0]))
    #         x = []
    #         y = []
    #         for pt in zip(*loc[::-1]):
    #             # print("{} - {}".format(pt[0], pt[1]))
    #             x.append(pt[0])
    #             y.append(pt[1])
    #
    #         pts = [x, y]
    #         dict = cluster(pts, 5)
    #         if len(dict) < 4:
    #             print("Four Centers not found")
    #             continue
    #         dict = sorted(dict.items(), key=lambda x: len(x[1]), reverse=True)
    #         dict = convert(dict[:4])
    #         markers = getCenters(dict, pts)
    #         print("Marker - {}".format(len(markers)))
    #         for pt in markers:
    #             cv2.circle(temp1, (int(pt[0] + int(template.shape[0]/2)), int(pt[1] + int(template.shape[1])/2)), 2, (0, 0, 255), 1)
    #         cv2.imshow("Test", temp1)
    #         break
    #     else:
    #         cv2.imshow("Test", temp1)

    # template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    # _, template = cv.threshold(template, 127, 255, cv.THRESH_BINARY)
    # _, gray = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)

    #
    # gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

    # if rho == -1:
    #     rho = 10
    # if threshold == -1:
    #     threshold = 50
    # if min_line_length == -1:
    #     min_line_length = 50
    # if max_line_gap == -1:
    #     max_line_gap = 20
    #
    # circles = cv2.HoughCircles(template, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
    #                            param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    #
    # dis = 0
    # if circles is not None:
    #     print(len(circles))
    #     for i in circles[0, :]:
    #         # draw the outer circle
    #         cv2.circle(template_img1, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #         # draw the center of the circle
    #         cv2.circle(template_img1, (i[0], i[1]), 2, (0, 0, 255), 3)

    # img_45 = ndimage.rotate(template, 45, reshape=False)
    # res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    # threshold = threshold/100
    # loc = np.where(res >= threshold)
    # print(len(loc[::-1][0]))
    # for pt in zip(*loc[::-1]):
    #     #cv2.rectangle(temp1, pt, (pt[0] + 33 , pt[1] + 33), (0, 0, 255), 2)
    #     cv2.circle(temp1, (int(pt[0] + int(template.shape[0]/2)), int(pt[1] + int(template.shape[1])/2)), 2, (0, 255, 0), 1)

    # cv2.imshow("Test", template_img1)
    # cv2.imshow("Test1", img_45)
