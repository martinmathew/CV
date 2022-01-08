"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2 as cv
import math
import numpy as np


def find_color(rgb):
    if rgb[1] == 255 and rgb[2] == 255:
        return "yellow"
    elif rgb[1] == 128 and rgb[2] == 255:
        return "orange"
    elif rgb[2] == 255:
        return "red"
    elif rgb[1] == 255:
        return "green"


def isOn(rgb):
    return rgb[0] == 255 or rgb[1] == 255 or rgb[2] == 255


def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.

    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.

    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.

    It is recommended you use Hough tools to find these circles in
    the image.

    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.

    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.

    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """

    img = cv.medianBlur(img_in, 5)
    cimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(cimg, cv.HOUGH_GRADIENT, 1, 20,
                              param1=75, param2=10, minRadius=radii_range.start, maxRadius=radii_range.stop)
    if circles is not None:
        circles = circles[0].astype(int)
        i = circles[1]
        # # for i in circles:
        #     # draw the outer circle
        # cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        #     # draw the center of the circle
        # cv.circle(img, (i[0], i[1]), 1, (0, 0, 255), 3)
        # print(str(i[0]) + ":" + str(i[1]))
        # print(img[i[1], i[0]])
        # cv.imshow('detected circles', img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        # circles = circles[0].astype(int)

        sorted(circles, key=lambda circle: circle[0] ** 2 + circle[1] ** 2)
        on = None;
        if isOn(img[circles[0][1], circles[0][0]]):
            on = "red"
        elif isOn(img[circles[1][1], circles[1][0]]):
            on = "yellow"
        else:
            on = "green"

        return (circles[1][0], circles[1][1]), on


def yield_sign_detection(img_in):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.
    """

    gray = cv.cvtColor(img_in, cv.COLOR_BGR2GRAY)

    kernel = np.ones((4, 4), np.uint8)
    dilation = cv.dilate(gray, kernel, iterations=1)
    blur = cv.GaussianBlur(dilation, (5, 5), 0)

    # Apply edge detection method on the image
    edges = cv.Canny(blur, 50, 150, apertureSize=3)

    # This returns an array of r and theta values

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 10  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img_in) * 0  # creating a blank to draw lines on

    lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                           min_line_length, max_line_gap)
    if len(lines) < 3:
        return None
    for i in range(len(lines)):
        line1 = lines[i][0]
        for j in range(i + 1, len(lines)):
            line2 = lines[j][0]
            for k in range(j + 1, len(lines)):
                line3 = lines[k][0]
                if isTriangle(line1, line2, line3):
                    return (line1[0] + line2[0] + line3[0] + line1[2] + line2[2] + line3[2]) / 6, (
                            line1[1] + line2[1] + line3[1] + line1[3] + line2[3] + line3[3]) / 6


def linelength(line):
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def find_angle(line1, line2):
    degree1 = math.degrees(math.atan((line1[1] - line1[3]) / (line1[0] - line1[2])))

    degree2 = math.degrees(math.atan((line2[1] - line2[3]) / (line2[0] - line2[2])))

    diff = abs(degree1 - degree2)

    if diff > 90:
        return 180 - diff
    return diff


def do_lines_touch(line1, line2):
    dist1 = math.sqrt((line1[0] - line2[0]) ** 2 + (line1[1] - line2[1]) ** 2)
    dist2 = math.sqrt((line1[2] - line2[2]) ** 2 + (line1[3] - line2[3]) ** 2)

    dist3 = math.sqrt((line1[0] - line2[2]) ** 2 + (line1[1] - line2[3]) ** 2)
    dist4 = math.sqrt((line1[2] - line2[0]) ** 2 + (line1[3] - line2[1]) ** 2)

    return (dist1 <= 20) ^ (dist2 <= 20) ^ (dist3 <= 20) ^ (dist4 <= 20)


def isTriangle(line1, line2, line3):
    len1 = linelength(line1)
    len2 = linelength(line2)
    len3 = linelength(line3)
    degree1 = find_angle(line1, line2)
    degree2 = find_angle(line2, line3)
    degree3 = find_angle(line3, line1)
    diff = 180 - (degree1 + degree2 + degree3)

    return (len1 + len2 > len3) and (len2 + len3 > len1) and (len3 + len1 > len2) and abs(diff) < 5 and do_lines_touch(
        line1, line2) and do_lines_touch(line2, line3) and do_lines_touch(line3, line1)


def isOctagon(line1, line2, line3, line4, line5, line6, line7, line8):
    len1 = linelength(line1)
    len2 = linelength(line2)
    len3 = linelength(line3)
    len4 = linelength(line4)
    len5 = linelength(line5)
    len6 = linelength(line6)
    len7 = linelength(line7)
    len8 = linelength(line8)

    if not (abs(len1 - len2) < 25 and abs(len1 - len3) < 25 and abs(len1 - len4) < 25 and abs(len1 - len5) < 25 and abs(
            len1 - len6) < 25 and abs(len1 - len7) < 25 and abs(len1 - len8) < 25):
        return False

    # llist = [line1, line2, line3, line4, line5, line6, line7, line8]
    # for i in range(len(llist)):
    #     count = 0
    #     for k in range(len(llist)):
    #         if i == k:
    #             continue
    #         if do_lines_touch(llist[i], llist[k]):
    #             count = count + 1
    #     if count != 2:
    #         return False
    return True


def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """
    median_val = 9
    median = cv.medianBlur(img_in, median_val)
    gray = cv.cvtColor(median, cv.COLOR_BGR2GRAY)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 22  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 19  # minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments
    dk = 5
    k = 7

    kernel = np.ones((dk, dk), np.uint8)

    # dilation = cv.dilate(gray, kernel, iterations=1)

    # blur = cv.GaussianBlur(dilation, (k, k), 0)

    # Apply edge detection method on the image
    edges = cv.Canny(gray, 50, 150, apertureSize=3)

    # This returns an array of r and theta values

    line_image = np.copy(img_in) * 0  # creating a blank to draw lines on

    lines = cv.HoughLinesP(edges, rho=rho, theta=np.pi / 180, threshold=threshold,
                           minLineLength=min_line_length, maxLineGap=max_line_gap)

    if lines is None or len(lines) < 8:
        return len(lines), len(lines)
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
                                    if isOctagon(line1, line2, line3, line4, line5, line6, line7, line8):
                                        midx, midy = (line1[0] + line2[0] + line3[0] + line4[0] + line5[0] + line6[0] +
                                                      line7[0] + line8[0] + line1[2] + line2[2] + line3[2] + line4[2] +
                                                      line5[2] + line6[2] + line7[2] + line8[2]) / 16, (
                                                             line1[1] + line2[1] + line3[1] + line4[1] + line5[1] +
                                                             line6[1] + line7[1] + line8[1] + line1[3] + line2[3] +
                                                             line3[3] + line4[3] + line5[3] + line6[3] + line7[3] +
                                                             line8[3]) / 16
                                        return int(midx), int(midy)
    if len(linek) > 2:
        return int(linelength(linek[0]) - linelength(linek[1])), int(linelength(linek[0]) - linelength(linek[2]))
    return 1, 2

    # for line in lines:
    #     for x1, y1, x2, y2 in line:
    #         cv.line(img_in, (x1, y1), (x2, y2), (255, 0, 0), 5)
    #
    # cv.imshow("Test", median)
    # cv.waitKey(0)


def isSquare(line1, line2, line3, line4):
    len1 = linelength(line1)
    len2 = linelength(line2)
    len3 = linelength(line3)
    len4 = linelength(line4)
    if not (abs(len1 - len2) < 3 and abs(len2 - len3) < 3 and abs(len3 - len4) < 3 and abs(len4 - len1) < 3):
        return False

    degree1 = find_angle(line1, line2)
    degree2 = find_angle(line1, line3)
    degree3 = find_angle(line1, line4)
    if abs(180 - (degree1 + degree2 + degree3)) > 9:
        return False

    if degree1 < 5:
        return do_lines_touch(line2, line3) and do_lines_touch(line2, line4) and not do_lines_touch(line1, line2)

    if degree2 < 5:
        return do_lines_touch(line3, line4) and do_lines_touch(line3, line2) and not do_lines_touch(line1, line3)

    if degree3 < 5:
        return do_lines_touch(line2, line4) and do_lines_touch(line3, line4) and not do_lines_touch(line1, line4)


def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    gray = cv.cvtColor(img_in, cv.COLOR_BGR2GRAY)

    # kernel = np.ones((2, 2), np.uint8)
    # dilation = cv.dilate(gray, kernel, iterations=1)
    # blur = cv.GaussianBlur(gray, (3, 3), 0)

    # Apply edge detection method on the image
    edges = cv.Canny(gray, 50, 150, apertureSize=3)

    # This returns an array of r and theta values

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 45  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 45  # minimum number of pixels making up a line
    max_line_gap = 11  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img_in) * 0  # creating a blank to draw lines on

    lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                           min_line_length, max_line_gap)
    if len(lines) < 4:
        return None

    for i in range(len(lines)):
        line1 = lines[i][0]
        for j in range(i + 1, len(lines)):
            line2 = lines[j][0]
            for k in range(j + 1, len(lines)):
                line3 = lines[k][0]
                for m in range(k + 1, len(lines)):
                    line4 = lines[m][0]
                    if isSquare(line1, line2, line3, line4):
                        midx, midy = (line1[0] + line2[0] + line3[0] + line4[0] + line1[2] + line2[2] + line3[2] +
                                      line4[2]) / 8, (
                                             line1[1] + line2[1] + line3[1] + line4[1] + line1[3] + line2[3] + line3[
                                         3] + line4[3]) / 8
                        if find_color(img_in[int(midy), int(midx)]) == "yellow":
                            return midx, midy

    # for line in lines:
    #     for x1, y1, x2, y2 in line:
    #         cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    #
    # cv.imshow("Test", line_image)
    # cv.waitKey(0)


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the

    provided image.


    Args:

        img_in (numpy.array): image containing a traffic light.


    Returns:

        (x,y) tuple of the coordinates of the center of the sign.

    """

    dk = 1

    # gray = img_in[img_in != 0] = 255

    hsv_img = cv.cvtColor(img_in, cv.COLOR_BGR2HSV)

    light_orange = (1, 190, 200)
    dark_orange = (25, 255, 255)
    mask = cv.inRange(hsv_img, light_orange, dark_orange)

    res = cv.bitwise_and(img_in, img_in, mask=mask)

    gray = cv.split(res)[2]
    #
    # gray[gray == 0] = 255
    #
    # gray[gray != 255] = 0
    #
    # kernel = np.ones((dk, dk), np.uint8)
    #
    # dilation = cv.dilate(gray, kernel, iterations=1)
    #
    # # blur = cv.GaussianBlur(gray, (3, 3), 0)
    #
    # # Apply edge detection method on the image
    #
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    #
    # # This returns an array of r and theta values
    #
    rho = 1  # distance resolution in pixels of the Hough grid

    theta = np.pi / 180  # angular resolution in radians of the Hough grid

    threshold = 28  # minimum number of votes (intersections in Hough grid cell)

    min_line_length = 35  # minimum number of pixels making up a line

    max_line_gap = 9  # maximum gap in pixels between connectable line segments

    line_image = np.copy(img_in) * 0  # creating a blank to draw lines on

    lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]),

                           min_line_length, max_line_gap)

    if len(lines) < 4:
        return None

    for i in range(len(lines)):

        line1 = lines[i][0]

        for j in range(i + 1, len(lines)):

            line2 = lines[j][0]

            for k in range(j + 1, len(lines)):

                line3 = lines[k][0]

                for m in range(k + 1, len(lines)):

                    line4 = lines[m][0]

                    if isSquare(line1, line2, line3, line4):

                        midx, midy = (line1[0] + line2[0] + line3[0] + line4[0] + line1[2] + line2[2] + line3[2] +

                                      line4[2]) / 8, (

                                             line1[1] + line2[1] + line3[1] + line4[1] + line1[3] + line2[3] + line3[

                                         3] + line4[3]) / 8

                        if find_color(img_in[int(midy), int(midx)]) == "orange":
                            return midx, midy
    return None

    # for line in lines:
    #
    #     for x1, y1, x2, y2 in line:
    #         cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    #
    # cv.imshow("Test", line_image)
    #
    # cv.waitKey(0)


def do_not_enter_sign_detection(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """
    dp = 1
    minDist = 20
    param1 = 50
    param2 = 31
    minRadius = 0
    maxRadius = 0
    blockSize = 7
    C = 4
    gray = cv.cvtColor(img_in, cv.COLOR_BGR2GRAY)
    thresh = cv.adaptiveThreshold(gray, 255, cv.BORDER_REPLICATE, cv.THRESH_BINARY, blockSize=blockSize, C=C)
    thresh = cv.bitwise_not(thresh)

    circles = cv.HoughCircles(thresh, cv.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                              param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    if circles is not None:
        for i in circles[0, :]:
            if img_in[int(i[1]), int(i[0])][0] == 255 and img_in[int(i[1]), int(i[0])][1] == 255 and \
                    img_in[int(i[1]), int(i[0])][2] == 255:
                return (i[0], i[1])
        # # draw the outer circle
        #     cv2.circle(temp, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # # draw the center of the circle
        #     cv2.circle(temp, (i[0], i[1]), 2, (0, 0, 255), 3)


def traffic_light_detection_mixed(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.

    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.

    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.

    It is recommended you use Hough tools to find these circles in
    the image.

    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.

    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.

    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """
    lower_black = (49, 49, 49)
    upper_black = (52, 52, 52)
    mask = cv.inRange(img_in, lower_black, upper_black)

    edges = cv.Canny(mask, 50, 150, apertureSize=3)
    circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, dp=2, minDist=20,
                              param1=50, param2=30, minRadius=10, maxRadius=9)
    if circles is None:
        return circles
    mid_traffic_light = findcircles_nearby(circles[0], 50, 5)

    return mid_traffic_light


def findcircles_nearby(circles, dist, radius):
    for i in range(len(circles)):
        circle1 = circles[i]
        list = []
        for j in range(len(circles)):
            if i == j:
                continue
            circle2 = circles[j]
            if linelength((circle1[0], circle1[1], circle2[0], circle2[1])) < dist and abs(
                    circle1[2] - circle2[2]) < radius:
                list.append(circle2)
        if len(list) == 2:
            return circle1
    return None


def traffic_sign_detection(img_in):
    """Finds all traffic signs in a synthetic image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    dict = {}
    radii_range = range(10, 30, 1)
    cord = traffic_light_detection_mixed(img_in, radii_range)
    if cord is not None:
        dict['traffic_light'] = cord
    dne_cord = detect_dne(img_in)
    if dne_cord is not None:
        dict['no_entry'] = dne_cord
    stop_cord = detect_stop(img_in)
    if stop_cord is not None:
        dict['stop'] = stop_cord
    yield_cord = detect_yield(img_in)
    if yield_cord is not None:
        dict['yield'] = yield_cord
    const_cord = construction_sign_detection(img_in)
    if const_cord is not None:
        dict['construction'] = const_cord
    warn_cord = warning_sign_detection(img_in)
    if warn_cord is not None:
        dict['warning'] = warn_cord
    return dict


def detect_yield(img_in):
    temp = cv.medianBlur(img_in, 9)
    lower_red = (0, 0, 253)
    upper_red = (0, 0, 255)

    lower_white = (254, 254, 254)
    upper_white = (255, 255, 255)
    red_mask = cv.inRange(temp, lower_red, upper_red)
    white_mask = cv.inRange(temp, lower_white, upper_white)
    mask = red_mask + white_mask

    res = cv.bitwise_and(temp, temp, mask=mask)

    gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    gray = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)[1]

    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40  # minimum number of pixels making up a line
    max_line_gap = 3  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img_in) * 0  # creating a blank to draw lines on
    threshold1 = 50
    threshold2 = 150
    apertureSize = 3

    edges = cv.Canny(gray, threshold1=threshold1, threshold2=threshold2, apertureSize=apertureSize)
    lines = cv.HoughLinesP(edges, rho=rho, theta=np.pi / 180, threshold=threshold,
                           minLineLength=min_line_length, maxLineGap=max_line_gap)

    if lines is None:
        return None
    if len(lines) < 3:
        return None
    for i in range(len(lines)):
        line1 = lines[i][0]
        for j in range(i + 1, len(lines)):
            line2 = lines[j][0]
            for k in range(j + 1, len(lines)):
                line3 = lines[k][0]
                if isTriangle(line1, line2, line3):
                    return (line1[0] + line2[0] + line3[0] + line1[2] + line2[2] + line3[2]) / 6, (
                            line1[1] + line2[1] + line3[1] + line1[3] + line2[3] + line3[3]) / 6

    return None


def detect_stop(img_in):
    lower_red = (0, 0, 204)
    upper_red = (0, 0, 207)
    mask = cv.inRange(img_in, lower_red, upper_red)
    edges = cv.Canny(mask, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=25,
                           minLineLength=20, maxLineGap=20)

    if lines is None:
        return 0, 0

    if lines is None or len(lines) < 8:
        return len(lines), len(lines)
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
                                    if isOctagon(line1, line2, line3, line4, line5, line6, line7, line8):
                                        midx, midy = (line1[0] + line2[0] + line3[0] + line4[0] + line5[0] + line6[0] +
                                                      line7[0] + line8[0] + line1[2] + line2[2] + line3[2] + line4[2] +
                                                      line5[2] + line6[2] + line7[2] + line8[2]) / 16, (
                                                             line1[1] + line2[1] + line3[1] + line4[1] + line5[1] +
                                                             line6[1] + line7[1] + line8[1] + line1[3] + line2[3] +
                                                             line3[3] + line4[3] + line5[3] + line6[3] + line7[3] +
                                                             line8[3]) / 16
                                        return int(midx), int(midy)
    if len(linek) > 2:
        return int(linelength(linek[0]) - linelength(linek[1])), int(linelength(linek[0]) - linelength(linek[2]))
    return 1, 2


def detect_dne(img):
    lower_red = (0, 0, 240)
    upper_red = (0, 0, 255)
    mask = cv.inRange(img, lower_red, upper_red)
    circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT, dp=2, minDist=27,
                              param1=50, param2=41, minRadius=11, maxRadius=41)
    if circles is None:
        return None
    for circle in circles[0]:
        x1, y1, _ = circle
        b, g, r = img[int(y1), int(x1)]
        if b == 255 and g == 255 and r == 255:
            return (x1, y1)


def detect_yield_noise(temp):
    lower_red = (0, 0, 253)
    upper_red = (10, 10, 255)

    lower_white = (245, 245, 245)
    upper_white = (255, 255, 255)
    red_mask = cv.inRange(temp, lower_red, upper_red)
    white_mask = cv.inRange(temp, lower_white, upper_white)
    mask = red_mask + white_mask

    res = cv.bitwise_and(temp, temp, mask=mask)
    res = cv.split(res)[2]
    # gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    gray = cv.threshold(res, 127, 255, cv.THRESH_BINARY)[1]
    kernel = np.ones((5, 5), np.uint8)
    gray = cv.erode(gray, kernel, iterations=1)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 43  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 12  # minimum number of pixels making up a line
    max_line_gap = 82  # maximum gap in pixels between connectable line segments

    threshold1 = 50
    threshold2 = 150
    apertureSize = 3

    edges = cv.Canny(gray, threshold1=threshold1, threshold2=threshold2, apertureSize=apertureSize)
    lines = cv.HoughLinesP(edges, rho=rho, theta=np.pi / 180, threshold=threshold,
                           minLineLength=min_line_length, maxLineGap=max_line_gap)

    if lines is None:
        return None
    if len(lines) < 3:
        return None
    for i in range(len(lines)):
        line1 = lines[i][0]
        for j in range(i + 1, len(lines)):
            line2 = lines[j][0]
            for k in range(j + 1, len(lines)):
                line3 = lines[k][0]
                if isTriangle(line1, line2, line3):
                    return (line1[0] + line2[0] + line3[0] + line1[2] + line2[2] + line3[2]) / 6, (
                            line1[1] + line2[1] + line3[1] + line1[3] + line2[3] + line3[3]) / 6

    return None


def detect_dne_noise(img):
    lower_red = (0, 0, 240)
    upper_red = (10, 10, 255)
    mask = cv.inRange(img, lower_red, upper_red)
    res = cv.bitwise_and(img, img, mask=mask)
    res = cv.split(res)[2]
    circles = cv.HoughCircles(res, cv.HOUGH_GRADIENT, dp=2, minDist=89,
                              param1=50, param2=42, minRadius=3, maxRadius=35)
    if circles is None:
        return None
    for circle in circles[0]:
        x1, y1, _ = circle
        b, g, r = img[int(y1), int(x1)]
        if b >= 250 and g >= 250 and r >= 250:
            return (x1, y1)


def denoise(img_in):
    temp = cv.medianBlur(img_in, 5)
    dst = cv.fastNlMeansDenoisingColored(temp, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    return dst


def traffic_light_detection_noisy(temp):
    """Finds the coordinates of a traffic light image given a radii
    range.

    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.

    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.

    It is recommended you use Hough tools to find these circles in
    the image.

    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.

    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.

    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """
    lower_black = (10, 10, 10)
    upper_black = (75, 75, 75)
    mask = cv.inRange(temp, lower_black, upper_black)
    res = cv.bitwise_and(temp, temp, mask=mask)

    gray = cv.split(res)[2]
    gray = cv.threshold(gray, 25, 255, cv.THRESH_BINARY)[1]
    # gray = cv2.dilate(gray, (5, 5), iterations=1)

    # edges = cv2.Canny(mask, threshold1, threshold2, apertureSize=apertureSize)

    # lines = cv2.HoughLinesP(edges, rho=rho, theta=np.pi / 180, threshold=threshold,
    #                         minLineLength=min_line_length, maxLineGap=max_line_gap)

    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=2, minDist=20,
                              param1=50, param2=30, minRadius=2, maxRadius=17)
    if circles is None:
        return circles
    mid_traffic_light = findcircles_nearby(circles[0], 50, 5)

    return mid_traffic_light


def detect_stop_noise(temp):
    lower_red = (0, 0, 160)
    upper_red = (50, 50, 225)

    # lower_white = (230, 230, 230)
    # upper_white = (255, 255, 255)

    red_mask = cv.inRange(temp, lower_red, upper_red)
    # white_mask = cv2.inRange(temp, lower_white, upper_white)
    mask = red_mask
    res = cv.bitwise_and(temp, temp, mask=mask)
    gray = cv.split(res)[2]
    # gray = cv2.dilate(gray, (15, 15), iterations=1)
    edges = cv.Canny(res, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=29,
                           minLineLength=20, maxLineGap=20)

    if lines is None:
        return 0, 0

    if lines is None or len(lines) < 8:
        return len(lines), len(lines)
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
                                    if isOctagon(line1, line2, line3, line4, line5, line6, line7, line8):
                                        midx, midy = (line1[0] + line2[0] + line3[0] + line4[0] + line5[0] + line6[0] +
                                                      line7[0] + line8[0] + line1[2] + line2[2] + line3[2] + line4[2] +
                                                      line5[2] + line6[2] + line7[2] + line8[2]) / 16, (
                                                             line1[1] + line2[1] + line3[1] + line4[1] + line5[1] +
                                                             line6[1] + line7[1] + line8[1] + line1[3] + line2[3] +
                                                             line3[3] + line4[3] + line5[3] + line6[3] + line7[3] +
                                                             line8[3]) / 16
                                        return int(midx), int(midy)
    if len(linek) > 2:
        return int(linelength(linek[0]) - linelength(linek[1])), int(linelength(linek[0]) - linelength(linek[2]))
    return 1, 2


def construction_sign_detection_noisy(temp):
    """Finds the centroid coordinates of a construction sign in the

    provided image.


    Args:

        img_in (numpy.array): image containing a traffic light.


    Returns:

        (x,y) tuple of the coordinates of the center of the sign.

    """

    dk = 1

    # gray = img_in[img_in != 0] = 255

    hsv_img = cv.cvtColor(temp, cv.COLOR_BGR2HSV)

    light_orange = (1, 120, 200)
    dark_orange = (25, 255, 255)

    mask = cv.inRange(hsv_img, light_orange, dark_orange)

    res = cv.bitwise_and(hsv_img, hsv_img, mask=mask)

    gray = cv.split(res)[2]
    #
    # gray[gray == 0] = 255
    #
    # gray[gray != 255] = 0
    #
    # kernel = np.ones((dk, dk), np.uint8)
    #
    # dilation = cv.dilate(gray, kernel, iterations=1)
    #
    # # blur = cv.GaussianBlur(gray, (3, 3), 0)
    #
    # # Apply edge detection method on the image
    #
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    #
    # # This returns an array of r and theta values
    #
    rho = 1  # distance resolution in pixels of the Hough grid

    theta = np.pi / 180  # angular resolution in radians of the Hough grid

    threshold = 59  # minimum number of votes (intersections in Hough grid cell)

    min_line_length = 45  # minimum number of pixels making up a line

    max_line_gap = 20  # maximum gap in pixels between connectable line segments

    lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]),

                           min_line_length, max_line_gap)

    if len(lines) < 4:
        return None

    for i in range(len(lines)):
        line1 = lines[i][0]
        for j in range(i + 1, len(lines)):
            line2 = lines[j][0]
            for k in range(j + 1, len(lines)):
                line3 = lines[k][0]
                for m in range(k + 1, len(lines)):
                    line4 = lines[m][0]
                    if isSquare(line1, line2, line3, line4):
                        midx, midy = (line1[0] + line2[0] + line3[0] + line4[0] + line1[2] + line2[2] + line3[2] +
                                      line4[2]) / 8, (
                                             line1[1] + line2[1] + line3[1] + line4[1] + line1[3] + line2[3] + line3[
                                         3] + line4[3]) / 8

                        return midx, midy
    return None

    # for line in lines:
    #
    #     for x1, y1, x2, y2 in line:
    #         cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    #
    # cv.imshow("Test", line_image)
    #
    # cv.waitKey(0)

def warning_sign_detection_noisy(img_in):


    light_orange = (0, 200, 200)
    dark_orange = (10, 255, 255)

    mask = cv.inRange(img_in, light_orange, dark_orange)

    res = cv.bitwise_and(img_in, img_in, mask=mask)

    gray = cv.split(res)[2]
    #
    # gray[gray == 0] = 255
    #
    # gray[gray != 255] = 0
    #
    # kernel = np.ones((dk, dk), np.uint8)
    #
    # dilation = cv.dilate(gray, kernel, iterations=1)
    #
    # # blur = cv.GaussianBlur(gray, (3, 3), 0)
    #
    # # Apply edge detection method on the image
    #
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    #
    # # This returns an array of r and theta values
    #
    rho = 2  # distance resolution in pixels of the Hough grid

    theta = np.pi / 180  # angular resolution in radians of the Hough grid

    threshold = 29  # minimum number of votes (intersections in Hough grid cell)

    min_line_length = 44  # minimum number of pixels making up a line

    max_line_gap = 23  # maximum gap in pixels between connectable line segments

    lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]),

                           min_line_length, max_line_gap)

    if len(lines) < 4:
        return None

    for i in range(len(lines)):
        line1 = lines[i][0]
        for j in range(i + 1, len(lines)):
            line2 = lines[j][0]
            for k in range(j + 1, len(lines)):
                line3 = lines[k][0]
                for m in range(k + 1, len(lines)):
                    line4 = lines[m][0]
                    if isSquare(line1, line2, line3, line4):
                        midx, midy = (line1[0] + line2[0] + line3[0] + line4[0] + line1[2] + line2[2] + line3[2] +
                                      line4[2]) / 8, (
                                             line1[1] + line2[1] + line3[1] + line4[1] + line1[3] + line2[3] + line3[
                                         3] + line4[3]) / 8

                        return midx, midy
    return None


def traffic_sign_detection_noisy(img_in):
    """Finds all traffic signs in a synthetic noisy image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    img_in = denoise(img_in)
    dict = {}
    traffic_cord = traffic_light_detection_noisy(img_in)
    if traffic_cord is not None:
        dict['traffic_light'] = traffic_cord
    dne_cord = detect_dne_noise(img_in)
    if dne_cord is not None:
        dict['no_entry'] = dne_cord
    yield_cord = detect_yield_noise(img_in)
    if yield_cord is not None:
        dict['yield'] = yield_cord
    stop_cord = detect_stop_noise(img_in)
    if stop_cord is not None:
        dict['stop'] = stop_cord

    constr_cord = construction_sign_detection_noisy(img_in)
    if constr_cord is not None:
        dict['construction'] = constr_cord
    warning_sign_cord = warning_sign_detection_noisy(img_in)
    if warning_sign_cord is not None:
        dict['warning'] = warning_sign_cord
    return dict

def detect_dne_challenge(temp):

    lower_red = (160, 160, 160)
    upper_red = (255, 255, 255)
    mask = cv.inRange(temp, lower_red, upper_red)
    res = cv.bitwise_and(temp, temp, mask=mask)


    gray = cv.split(res)[2]
    kernel = np.ones((7, 7), np.uint8)
    img_dilation = cv.erode(gray, kernel, iterations=1)

    circles = cv.HoughCircles(img_dilation, cv.HOUGH_GRADIENT, dp=1, minDist=186,
                              param1=50, param2=30, minRadius=171, maxRadius=0)

    if circles is None:
        return None
    for circle in circles[0]:
        x1, y1, _ = circle
        b, g, r = temp[int(y1), int(x1)]
        if b >= 200 and g >= 200 and r >= 200:
            return (x1, y1)


def detect_stop_challenge(temp):
    light_orange = (0, 0, 70)
    dark_orange = (25, 20, 100)

    mask = cv.inRange(temp, light_orange, dark_orange)

    res = cv.bitwise_and(temp, temp, mask=mask)

    gray = cv.split(res)[2]
    #
    ret, gray = cv.threshold(gray, 50, 255, cv.THRESH_BINARY)
    gray = cv.dilate(gray, (55, 55), iterations=1)

    edges = cv.Canny(gray, threshold1=50, threshold2=150, apertureSize=3)
    lines = cv.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=59,
                            minLineLength=92, maxLineGap=60)


    if lines is None or len(lines) < 8:
        return None

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
                                    if isOctagon(line1, line2, line3, line4, line5, line6, line7, line8):
                                        midx, midy = (line1[0] + line2[0] + line3[0] + line4[0] + line5[0] + line6[0] +
                                                      line7[0] + line8[0] + line1[2] + line2[2] + line3[2] + line4[2] +
                                                      line5[2] + line6[2] + line7[2] + line8[2]) / 16, (
                                                             line1[1] + line2[1] + line3[1] + line4[1] + line5[1] +
                                                             line6[1] + line7[1] + line8[1] + line1[3] + line2[3] +
                                                             line3[3] + line4[3] + line5[3] + line6[3] + line7[3] +
                                                             line8[3]) / 16
                                        return (int(midx), int(midy))





def traffic_sign_detection_challenge(img_in):
    """Finds traffic signs in an real image

    See point 5 in the instructions for details.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    dict = {}
    dne_cord = detect_dne_challenge(img_in)
    if dne_cord is not None:
        dict['no_entry'] = dne_cord
        return dict
    stop_challenge = detect_stop_challenge(img_in)
    if stop_challenge is not None:
        dict['stop'] = stop_challenge
        return dict
    return dict








################################ CHANGE BELOW FOR MORE CUSTOMIZATION #######################
""" The functions below are used for each individual part of the report section.

Feel free to change the return statements but ensure that the return type remains the same 
for the autograder. 

"""


# Part 2 outputs
def ps2_2_a_1(img_in):
    return do_not_enter_sign_detection(img_in)


def ps2_2_a_2(img_in):
    return stop_sign_detection(img_in)


def ps2_2_a_3(img_in):
    return construction_sign_detection(img_in)


def ps2_2_a_4(img_in):
    return warning_sign_detection(img_in)


def ps2_2_a_5(img_in):
    return yield_sign_detection(img_in)


# Part 3 outputs
def ps2_3_a_1(img_in):
    return traffic_sign_detection(img_in)


def ps2_3_a_2(img_in):
    return traffic_sign_detection(img_in)


# Part 4 outputs
def ps2_4_a_1(img_in):
    return traffic_sign_detection_noisy(img_in)


def ps2_4_a_2(img_in):
    return traffic_sign_detection_noisy(img_in)


# Part 5 outputs
def ps2_5_a(img_in):
    return traffic_sign_detection_challenge(img_in)
