"""Problem Set 4: Motion Detection"""

import numpy as np
import cv2

# Utility function
import sys


def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """

    raise NotImplementedError


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """

    raise NotImplementedError


def optic_flow_lk(img1, img2, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """

    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    if k_type == 'gaussian':
        img1 = cv2.GaussianBlur(img1, (k_size, k_size), 0)
        img2 = cv2.GaussianBlur(img2, (k_size, k_size), 0)
    elif k_type == 'uniform':
        img1 = cv2.blur(img1, (k_size, k_size), 0)
        img2 = cv2.blur(img2, (k_size, k_size), 0)
        print("Uniform ............")

    # # img1 = cv2.GaussianBlur(img1, (5, 5), 0)
    # # img2 = cv2.GaussianBlur(img2, (5, 5), 0)
    # img1 = img1.astype('double') / 255
    # img2 = img2.astype('double') / 255

    Ix = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3, scale=1 / 8)
    Iy = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3, scale=1 / 8)
    It = img2 - img1
    u = np.zeros(img1.shape)
    v = np.zeros(img1.shape)
    w = 45
    tau = 10 / 1000
    for x in range(0, img1.shape[0]):
        for y in range(0, img1.shape[1]):
            Ixf = Ix[x:x + w, y:y + w].flatten()
            Iyf = Iy[x:x + w, y:y + w].flatten()
            Itf = It[x:x + w, y:y + w].flatten()

            A = np.vstack((Ixf, Iyf)).T
            res = np.matmul(np.linalg.pinv(np.matmul(A.T, A)), -np.matmul(A.T, Itf))

            # if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
            u[x, y] = res[0]
            v[x, y] = res[1]

    return u, v

    raise NotImplementedError


def is_invertible(a):
     a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def optic_flow_lk1(img1, img2, k_size, k_type, windows, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """

    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    if k_type == 'gaussian':
        img1 = cv2.GaussianBlur(img1, (k_size, k_size), 0)
        img2 = cv2.GaussianBlur(img2, (k_size, k_size), 0)
    elif k_type == 'uniform':
        img1 = cv2.blur(img1, (k_size, k_size), 0)
        img2 = cv2.blur(img2, (k_size, k_size), 0)
        print("Uniform ............")

    # # img1 = cv2.GaussianBlur(img1, (5, 5), 0)
    # # img2 = cv2.GaussianBlur(img2, (5, 5), 0)
    # img1 = img1.astype('double') / 255
    # img2 = img2.astype('double') / 255

    Ix = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3, scale=1 / 8)
    Iy = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3, scale=1 / 8)
    It = img2 - img1
    u = np.zeros(img1.shape)
    v = np.zeros(img1.shape)
    w = windows
    tau = 10 / 1000
    for x in range(0, img1.shape[0]):
        for y in range(0, img1.shape[1]):
            Ixf = Ix[x:x + w, y:y + w].flatten()
            Iyf = Iy[x:x + w, y:y + w].flatten()
            Itf = It[x:x + w, y:y + w].flatten()

            A = np.vstack((Ixf, Iyf)).T
            AtA = np.matmul(A.T, A)
            if is_invertible(AtA):
                res = np.matmul(np.linalg.pinv(AtA), -np.matmul(A.T, Itf))
                u[x, y] = res[0]
                v[x, y] = res[1]


    return u, v

    raise NotImplementedError


def getKernelReduce(a):
    row = np.array([1, 4, 6, 4, 1]) / 16
    return row


def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """
    kernel = getKernelReduce(0.3 - 0.6 * 0.38)
    filtered = cv2.sepFilter2D(image, -1, kernel, kernel.T)
    res = filtered[::2, ::2]
    # if res.shape[0] % 2 == 1:
    #     res = np.delete(res, res.shape[0] -1 , axis= 0)
    # if res.shape[1] % 2 == 1:
    #     res = np.delete(res, res.shape[1] -1, axis=1)
    return res


def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """
    res = [image.copy()]
    for i in range(1, levels):
        res.append(reduce_image(res[i - 1].copy()))
    return res


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """
    ht = img_list[0].shape[0]
    width = 0
    for img in img_list:
        width = width + img.shape[1]

    output = np.ones([ht, width]) * 255
    width = 0
    for img in img_list:
        output[0:img.shape[0], width:width + img.shape[1]] = normalize_and_scale(img)
        width = width + img.shape[1]
    return output
    raise NotImplementedError


def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """
    expand_kernel = np.array([1, 4, 6, 4, 1]) / 8
    expanded_image = np.zeros([image.shape[0] * 2, image.shape[1] * 2])
    expanded_image[::2, ::2] = image
    filtered = cv2.sepFilter2D(expanded_image, -1, expand_kernel, expand_kernel.T)
    return filtered


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """
    l_pyr = [g_pyr[len(g_pyr) - 1]]
    for i in range(len(g_pyr) - 2, -1, -1):
        exp = expand_image(g_pyr[i + 1])
        if g_pyr[i].shape[0] % 2 == 1:
            exp = exp[:-1, :]
        if g_pyr[i].shape[1] % 2 == 1:
            exp = exp[:, :-1]
        l_pyr.insert(0, (g_pyr[i] - exp))

    return l_pyr


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """
    A = np.zeros((image.shape))
    M, N = A.shape
    X, Y = np.meshgrid(range(N), range(M))
    map_x = (X + U).astype(np.float32)
    map_y = (Y + V).astype(np.float32)
    dst = cv2.remap(image, map_x, map_y, interpolation, borderMode=border_mode)
    return dst


def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """

    gpyr1 = gaussian_pyramid(img_a, levels)
    gpyr2 = gaussian_pyramid(img_b, levels)

    uk1 = np.zeros(gpyr2[levels-1].shape)
    vk1 = np.zeros(gpyr2[levels-1].shape)

    for i in range(levels - 1, -1, -1):
        h, w = gpyr1[i].shape
        uk1 = (expand_image(uk1)*2)[:h, :w]
        vk1 = (expand_image(vk1)*2)[:h, :w]
        warp_img = warp(gpyr2[i], uk1, vk1, interpolation, border_mode)
        ugi, vgi = optic_flow_lk1(gpyr1[i], warp_img, k_size, k_type, 45, sigma)
        uk1 = uk1 + ugi
        vk1 = vk1 + vgi

    return uk1, vk1
