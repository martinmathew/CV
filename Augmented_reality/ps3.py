"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import cv2 as cv
import numpy as np
import math
from scipy import ndimage
import os
import cv2.aruco as aru


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
    dst = cv.fastNlMeansDenoisingColored(img_in, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    # temp = cv.medianBlur(dst, 3)
    return dst


def euclidean_distance(a, b):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """

    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def get_corners_list(image):
    """Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    return [(0, 0), (0, image.shape[0] - 1), (image.shape[1] - 1, 0), (image.shape[1] - 1, image.shape[0] - 1)]


def find_markers(temp, template=None):
    """Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    # kernel1 = np.ones((3, 3), np.uint8)
    # marker = []
    # temp = denoise(temp)
    # gray = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
    # _, gray = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
    # temp = cv2.erode(gray, kernel1, iterations=1)
    # temp = np.float32(temp)
    # dst = cv.cornerHarris(temp, 2, 3, 0.04)
    # res = np.where(dst == dst.max())
    #
    # cluster_dict = cluster(res, 30)
    # center = getCenters(cluster_dict, res)
    # if center is None:
    #     return marker
    # for i in center:
    #     # draw the outer circle
    #     # cv2.circle(temp, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #     # draw the center of the circle
    #     marker.append((i[1], i[0]))

    marker = []
    temp = denoise(temp)
    gray = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    _, template = cv.threshold(template, 127, 255, cv.THRESH_BINARY)
    _, gray = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        # cv2.rectangle(temp1, pt, (pt[0] + 33 , pt[1] + 33), (0, 0, 255), 2)
        marker.append((int(pt[0] + int(template.shape[1] / 2)), int(pt[1] + int(template.shape[0]) / 2)))

    return sort_by_leftright(marker)


def convert(tup):
    di = {}
    for a, b in tup:
        di[a] = b
    return di


def rotated_templates():
    temp_img = cv2.imread("input_images/template_proc.jpg")
    rotated_img = []
    gray_template = cv.cvtColor(temp_img, cv.COLOR_BGR2GRAY)
    _, gray_template = cv.threshold(gray_template, 200, 255, cv.THRESH_BINARY_INV)
    for angle in range(0, 180, 90):
        # print("Angle - {}".format(angle))
        rotated_img.append(ndimage.rotate(gray_template, angle, reshape=False))
    return rotated_img


def finer_rotated_templates():
    temp_img = cv2.imread("input_images/template_proc.jpg")
    rotated_img = []
    gray_template = cv.cvtColor(temp_img, cv.COLOR_BGR2GRAY)
    _, gray_template = cv.threshold(gray_template, 200, 255, cv.THRESH_BINARY_INV)
    for angle in range(0, 180, 45):
        # print("Angle - {}".format(angle))
        rotated_img.append(ndimage.rotate(gray_template, angle, reshape=False))
    return rotated_img


def finer_rotated_templates_half():
    temp_img = cv2.imread("input_images/template_proc.jpg")
    rotated_img = []
    gray_template = cv.cvtColor(temp_img, cv.COLOR_BGR2GRAY)
    _, gray_template = cv.threshold(gray_template, 200, 255, cv.THRESH_BINARY_INV)
    for angle in range(0, 180, 45):
        # print("Angle - {}".format(angle))
        rotated_img.append(ndimage.rotate(gray_template, angle, reshape=False))

    height, width = gray_template.shape

    # Cut the image in half
    width_cutoff = width // 2
    s1 = gray_template[:, :width_cutoff]
    s2 = gray_template[:, width_cutoff:]

    height_cutoff = height // 2
    s3 = gray_template[:height_cutoff, :]
    s4 = gray_template[height_cutoff:, :]

    for angle in range(0, 180, 45):
        # print("Angle - {}".format(angle))
        rotated_img.append(ndimage.rotate(s1, angle, reshape=False))
        rotated_img.append(ndimage.rotate(s2, angle, reshape=False))
        rotated_img.append(ndimage.rotate(s3, angle, reshape=False))
        rotated_img.append(ndimage.rotate(s4, angle, reshape=False))

    return rotated_img


def find_markers1(temp, rotated_templates):
    temp_img = cv2.imread("input_images/template_proc.jpg")

    temp = denoise(temp)
    gray = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
    _, gray = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)
    # cv.imshow("win",gray)
    # cv.waitKey(0)
    # gray_template = cv.cvtColor(temp_img, cv.COLOR_BGR2GRAY)
    # _, gray_template = cv.threshold(gray_template, 200, 255, cv.THRESH_BINARY_INV)
    results = {}
    for rotated_img in rotated_templates:
        # print("Angle - {}".format(angle))
        # rotated_img = ndimage.rotate(gray_template, angle, reshape=False)
        # cv2.imshow("Test11", rotated_img)
        res = cv2.matchTemplate(gray, rotated_img, cv2.TM_CCOEFF_NORMED)

        res_flatten = res.flatten()
        size = 1000  # int(len(res_flatten)/32)
        loc = (np.vstack(np.unravel_index(np.argpartition(res_flatten, -size)[-size:], res.shape))[0],
               np.vstack(np.unravel_index(np.argpartition(res_flatten, -size)[-size:], res.shape))[1])
        max_tenth = np.sort(res.flatten())[::-size][0]
        # loc = np.where(res >= res.max()*0.75)

        if loc is not None and len(loc[::-1][0]) > 4:
            # print(len(loc[::-1][0]))
            x = []
            y = []
            for pt in zip(*loc[::-1]):
                # print("{} - {}".format(pt[0], pt[1]))
                x.append(int(pt[0] + int(temp_img.shape[1] / 2)))
                y.append(int(pt[1] + int(temp_img.shape[0]) / 2))

            pts = [x, y]
            dict = cluster(pts, 10)
            dict = sorted(dict.items(), key=lambda x: len(x[1]), reverse=True)
            dict = convert(dict[:4])
            markers = getCenters(dict, pts)
            results[max_tenth] = markers
    key = sorted(results, reverse=True)[0]
    return results[key]


def find_markers_quarters(temp, rotated_templates):
    temp_img = cv2.imread("input_images/template_proc.jpg")

    temp = denoise(temp)
    gray = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
    _, gray = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)
    # cv.imshow("win",gray)
    # cv.waitKey(0)
    x = gray.shape[1]
    y = gray.shape[0]
    quarters = [(gray[0:int(y / 2), 0: int(x / 2)], 0, 0), (gray[int(y / 2):y, 0: int(x / 2)], 0, int(y / 2)),
                (gray[0:int(y / 2), int(x / 2):x], int(x / 2), 0),
                (gray[int(y / 2): y, int(x / 2):x], int(x / 2), int(y / 2))]
    markers = []
    for quarter, x, y in quarters:
        x_cord, y_cord = process_perquarter(quarter, rotated_templates)
        markers.append((x_cord + x, y_cord + y))
    return markers


def process_perquarter(quarter, rotated_images):
    results = {}
    for rotated_image in rotated_images:
        res = cv2.matchTemplate(quarter, rotated_image, cv2.TM_CCOEFF_NORMED)
        res_flatten = res.flatten()
        size = 50  # int(len(res_flatten)/32)
        loc = (np.vstack(np.unravel_index(np.argpartition(res_flatten, -size)[-size:], res.shape))[0],
               np.vstack(np.unravel_index(np.argpartition(res_flatten, -size)[-size:], res.shape))[1])
        max_tenth = np.sort(res.flatten())[::-size][0]

        if loc is not None and len(loc[::-1][0]) > 1:
            # print(len(loc[::-1][0]))
            x = []
            y = []
            for pt in zip(*loc[::-1]):
                # print("{} - {}".format(pt[0], pt[1]))
                x.append(int(pt[0] + int(rotated_image.shape[1] / 2)))
                y.append(int(pt[1] + int(rotated_image.shape[0] / 2)))

            pts = [x, y]
            dict = cluster(pts, 10)
            dict = sorted(dict.items(), key=lambda x: len(x[1]), reverse=True)
            dict = convert(dict[:1])
            markers = getCenters(dict, pts)
            results[max_tenth] = markers
    key = sorted(results, reverse=True)[0]
    return results[key][0]


def sort_by_leftright(marker):
    sortby_y = sorted(marker, key=lambda point: point[1])[:2]
    sortby_y_bottom = sorted(marker, key=lambda point: -1 * point[1])[:2]
    sortby_x_behind = sorted(marker, key=lambda point: -1 * point[0])[:2]

    top_left = sorted(sortby_y, key=lambda pt: pt[0])[0]
    bottom_left = sorted(sortby_y_bottom, key=lambda pt: pt[0])[0]
    top_right = sorted(sortby_x_behind, key=lambda pt: pt[1])[0]
    bottom_right = sorted(sortby_x_behind, key=lambda pt: pt[1])[1]
    return [top_left, bottom_left, top_right, bottom_right]


def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """

    markers = sort_by_leftright(markers)
    cv.line(image, markers[0], markers[1], (0, 0, 255), thickness)
    cv.line(image, markers[1], markers[3], (0, 0, 255), thickness)
    cv.line(image, markers[3], markers[2], (0, 0, 255), thickness)
    cv.line(image, markers[2], markers[0], (0, 0, 255), thickness)
    return image


def project_imageA_onto_imageB(advert, scene, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """
    advert_corners = get_corners_list(advert)
    for x in range(advert_corners[2][0]):
        for y in range(advert_corners[1][1]):
            try:
                dp = np.dot(homography, np.array([[x, y, 1]]).T)
                newx = int(dp[0][0] / dp[2][0])
                newy = int(dp[1][0] / dp[2][0])
                # # if markers is not None:
                # if newx >= scene.shape[1] or newy >= scene.shape[0]:
                #     print("X - {}, y- {}".format(newx, newy))
                scene[newy][newx] = advert[y][x]
            except:
                print("Index Out of Bound Exception")

    return scene


def project_imageA_onto_imageB_beyond_5(advert, scene, homography, num_of_trial):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """
    advert_corners = get_corners_list(advert)
    for x in range(advert_corners[2][0]):
        for y in range(advert_corners[1][1]):
            try:
                dp = np.dot(homography, np.array([[x, y, 1]]).T)
                newx = int(dp[0][0] / dp[2][0])
                newy = int(dp[1][0] / dp[2][0])
                # if markers is not None:
                scene[newy][newx] = advert[y][x]
            except:
                print("Index Out of Bound Exception - {}".format(num_of_trial))
                if num_of_trial == 1:
                    return None
                # save_image('error-{}.jpg'.format(np.random.random(100000000)),scene)

    return scene


def get_grid(x, y):
    coords = np.indices((x, y)).reshape(2, -1)
    return np.vstack((coords, np.ones(coords.shape[1])))


def project_imageA_onto_imageB_beyond_inverse_warping(advert, scene, homography, num_of_trial):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """

    advert_corners = get_corners_list(advert)
    width = advert_corners[3][0]
    height = advert_corners[3][1]
    coords = get_grid(width, height).astype(np.int)

    x2, y2 = coords[0], coords[1]
    warp_coords = (homography@coords).astype(np.int)

    x1, y1 = warp_coords[0, :], warp_coords[1, :]

    # Get pixels within image boundaries
    indices = np.where((x1 >= 0) & (x1 < width) &
                       (y1 >= 0) & (y1 < height))

    xpix1, ypix1 = x2[indices], y2[indices]
    xpix2, ypix2 = x1[indices], y1[indices]
    scene[ypix1, xpix1] = advert[ypix2, xpix2]
    # for y1,y2 in list(zip(ypix1, ypix2)):
    #     for x1, x2 in list(zip(xpix1, xpix2)):
    #         scene[y1][x1] = advert[y2][x2]

    # for x in range(advert_corners[2][0]):
    #     for y in range(advert_corners[1][1]):
    #         try:
    #             dp = np.dot(homography, np.array([[x, y, 1]]).T)
    #             newx = int(dp[0][0] / dp[2][0])
    #             newy = int(dp[1][0] / dp[2][0])
    #             # if markers is not None:
    #             scene[newy][newx] = advert[y][x]
    #         except:
    #             print("Index Out of Bound Exception - {}".format(num_of_trial))
    #             if num_of_trial == 1:
    #                 return None
    #             # save_image('error-{}.jpg'.format(np.random.random(100000000)),scene)

    return scene


def save_image(filename, image):
    """Convenient wrapper for writing images to the output directory."""
    cv2.imwrite(os.path.join('./', filename), image)


def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """

    A = np.array([[src_points[0][0], src_points[0][1], 1, 0, 0, 0, -1 * src_points[0][0] * dst_points[0][0],
                   -1 * src_points[0][1] * dst_points[0][0]],
                  [0, 0, 0, src_points[0][0], src_points[0][1], 1, -1 * src_points[0][0] * dst_points[0][1],
                   -1 * src_points[0][1] * dst_points[0][1]],
                  [src_points[1][0], src_points[1][1], 1, 0, 0, 0, -1 * src_points[1][0] * dst_points[1][0],
                   -1 * src_points[1][1] * dst_points[1][0]],
                  [0, 0, 0, src_points[1][0], src_points[1][1], 1, -1 * src_points[1][0] * dst_points[1][1],
                   -1 * src_points[1][1] * dst_points[1][1]],
                  [src_points[2][0], src_points[2][1], 1, 0, 0, 0, -1 * src_points[2][0] * dst_points[2][0],
                   -1 * src_points[2][1] * dst_points[2][0]],
                  [0, 0, 0, src_points[2][0], src_points[2][1], 1, -1 * src_points[2][0] * dst_points[2][1],
                   -1 * src_points[2][1] * dst_points[2][1]],
                  [src_points[3][0], src_points[3][1], 1, 0, 0, 0, -1 * src_points[3][0] * dst_points[3][0],
                   -1 * src_points[3][1] * dst_points[3][0]],
                  [0, 0, 0, src_points[3][0], src_points[3][1], 1, -1 * src_points[3][0] * dst_points[3][1],
                   -1 * src_points[3][1] * dst_points[3][1]]
                  ])

    b = np.array([[dst_points[0][0]],
                  [dst_points[0][1]],
                  [dst_points[1][0]],
                  [dst_points[1][1]],
                  [dst_points[2][0]],
                  [dst_points[2][1]],
                  [dst_points[3][0]],
                  [dst_points[3][1]]
                  ])

    tmp = np.dot(np.linalg.inv(A), b)
    H = np.array([[tmp[0][0], tmp[1][0], tmp[2][0]],
                  [tmp[3][0], tmp[4][0], tmp[5][0]],
                  [tmp[6][0], tmp[7][0], 1]
                  ])

    return H

    raise NotImplementedError


def find_four_point_transform_1(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """

    A = np.array([[-1 * src_points[0][0], -1 * src_points[0][1], -1, 0, 0, 0, src_points[0][0] * dst_points[0][0],
                   src_points[0][1] * dst_points[0][0], dst_points[0][0]],
                  [0, 0, 0, -1 * src_points[0][0], -1 * src_points[0][1], -1, src_points[0][0] * dst_points[0][1],
                   src_points[0][1] * dst_points[0][1], dst_points[0][1]],

                  [-1 * src_points[1][0], -1 * src_points[1][1], -1, 0, 0, 0, src_points[1][0] * dst_points[1][0],
                   src_points[1][1] * dst_points[1][0], dst_points[1][0]],
                  [0, 0, 0, -1 * src_points[1][0], -1 * src_points[1][1], -1, src_points[1][0] * dst_points[1][1],
                   src_points[1][1] * dst_points[1][1], dst_points[1][1]],

                  [-1 * src_points[2][0], -1 * src_points[2][1], -1, 0, 0, 0, src_points[2][0] * dst_points[2][0],
                   src_points[2][1] * dst_points[2][0], dst_points[2][0]],
                  [0, 0, 0, -1 * src_points[2][0], -1 * src_points[2][1], -1, src_points[2][0] * dst_points[2][1],
                   src_points[2][1] * dst_points[2][1], dst_points[2][1]],

                  [-1 * src_points[3][0], -1 * src_points[3][1], -1, 0, 0, 0, src_points[3][0] * dst_points[3][0],
                   src_points[3][1] * dst_points[3][0], dst_points[3][0]],
                  [0, 0, 0, -1 * src_points[3][0], -1 * src_points[3][1], -1, src_points[3][0] * dst_points[3][1],
                   src_points[3][1] * dst_points[3][1], dst_points[3][1]]
                  ])

    [_, _, Vtran] = np.linalg.svd(A)

    Vtran = Vtran[-1, :] / Vtran[-1, -1]
    H = Vtran.reshape(3, 3)

    return H

    raise NotImplementedError


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break
    video.release()
    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    yield None
    raise NotImplementedError


def find_aruco_markers(image, aruco_dict=cv2.aruco.DICT_5X5_50):
    """Finds all ArUco markers and their ID in a given image.

    Hint: you are free to use cv2.aruco module

    Args:
        image (numpy.array): image array.
        aruco_dict (integer): pre-defined ArUco marker dictionary enum.

        For aruco_dict, use cv2.aruco.DICT_5X5_50 for this assignment.
        To find the IDs of markers, use an appropriate function in cv2.aruco module.

    Returns:
        numpy.array: corner coordinate of detected ArUco marker
            in (X, 4, 2) dimension when X is number of detected markers
            and (4, 2) is each corner's x,y coordinate in the order of
            top-left, bottom-left, top-right, and bottom-right.
        List: list of detected ArUco marker IDs.
    """
    image = denoise(image)
    aruco_dictionary = aru.Dictionary_get(aruco_dict)
    aruco_parameters = aru.DetectorParameters_create()
    (corners, ids, rejected) = aru.detectMarkers(image, aruco_dictionary, parameters=aruco_parameters)
    return np.array(corners).reshape(4, 4, 2), ids


def find_aruco_center(markers, ids):
    """Draw a bounding box of each marker in image. Also, put a marker ID
        on the top-left of each marker.

    Args:
        image (numpy.array): image array.
        markers (numpy.array): corner coordinate of detected ArUco marker
            in (X, 4, 2) dimension when X is number of detected markers
            and (4, 2) is each corner's x,y coordinate in the order of
            top-left, bottom-left, top-right, and bottom-right.
        ids (list): list of detected ArUco marker IDs.

    Returns:
        List: list of centers of ArUco markers. Each element needs to be
            (x, y) coordinate tuple.
    """
    avg = np.array([np.average(x, axis=0) for x in markers]).astype(int)
    return list(map(tuple, avg))

    raise NotImplementedError
