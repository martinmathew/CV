import math
import numpy as np
import cv2 as cv


# # Implement the functions below.


def extract_red(image):
    """ Returns the red channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the red channel.
    """
    return image[:, :, 2]


def extract_green(image):
    """ Returns the green channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the green channel.
    """
    return image[:, :, 1]


def extract_blue(image):
    """ Returns the blue channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the blue channel.
    """
    return image[:, :, 0]


def swap_green_blue(image):
    """ Returns an image with the green and blue channels of the input image swapped. It is highly
    recommended to make a copy of the input image in order to avoid modifying the original array.
    You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 3D array with the green and blue channels swapped.
    """
    temp_image = np.copy(image)
    green_channel = np.copy(temp_image[:, :, 1])
    temp_image[:, :, 1] = temp_image[:, :, 0]
    temp_image[:, :, 0] = green_channel
    return temp_image


def copy_paste_middle(src, dst, shape):
    srcleftRP = (src.shape[0] / 2 - shape[0] / 2, src.shape[1] / 2 - shape[1] / 2)
    srcrightBT = (src.shape[0] / 2 + shape[0] / 2, src.shape[1] / 2 + shape[1] / 2)

    destleftRP = (dst.shape[0] / 2 - shape[0] / 2, dst.shape[1] / 2 - shape[1] / 2)
    destrightBT = (dst.shape[0] / 2 + shape[0] / 2, dst.shape[1] / 2 + shape[1] / 2)

    temp_image_dst = np.copy(dst)

    temp_image_dst[int(destleftRP[0]):int(destrightBT[0]), int(destleftRP[1]):int(destrightBT[1])] = src[int(
        srcleftRP[0]):int(srcrightBT[0]), int(srcleftRP[1]):int(srcrightBT[1])]
    return temp_image_dst
    """ Copies the middle region of size shape from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:
    temp_image = np.copy(image)

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

        Note: Where 'middle' is ambiguous because of any difference in the oddness
        or evenness of the size of the copied region and the image size, the function
        rounds downwards.  E.g. in copying a shape = (1,1) from a src image of size (2,2)
        into an dst image of size (3,3), the function copies the range [0:1,0:1] of
        the src into the range [1:2,1:2] of the dst.

    Args:
        src (numpy.array): 2D array where the rectangular shape will be copied from.
        dst (numpy.array): 2D array where the rectangular shape will be copied to.
        shape (tuple): Tuple containing the height (int) and width (int) of the section to be
                       copied.

    Returns:
        numpy.array: Output monochrome image (2D array)
    """
    raise NotImplementedError


def isWithinCircle(center, radius, test_point):
    return ((test_point[0] - center[0]) ** 2 + (test_point[1] - center[1]) ** 2) <= radius ** 2




    """ Copies the middle circle region of radius "radius" from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:
    temp_image = np.copy(image)

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

    Args:
        src (numpy.array): 2D array where the circular shape will be copied from.
        dst (numpy.array): 2D array where the circular shape will be copied to.
        radius (scalar): scalar value of the radius.

    Returns:
        numpy.array: Output monochrome image (2D array)
    """
    raise NotImplementedError

def copy_paste_middle_circle1(src, dst, radius):
    x, y = int((src.shape[0] - 1) / 2), int((src.shape[1] - 1) / 2)
    radius = int(radius)

    srcRSt = int((src.shape[0]-1) / 2) - radius

    srcCSt = int((src.shape[1]-1) / 2) - radius

    destRSt = int((dst.shape[0]-1) / 2) - radius

    destCSt = int((dst.shape[1]-1) / 2) - radius
    dst_img = np.copy(dst)
    for row in range(radius * 2+1):
        for col in range(radius * 2+1):
            if isWithinCircle((x, y), radius, (srcRSt + row, srcCSt + col)):
                dst_img[destRSt + row:destRSt + row+1, destCSt + col:destCSt + col+1] = src[srcRSt + row:srcRSt + row+1, srcCSt + col:srcCSt + col+1]
    dst_img = dst_img.astype('int')
    return dst_img


    """ Copies the middle circle region of radius "radius" from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:
    temp_image = np.copy(image)

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

    Args:
        src (numpy.array): 2D array where the circular shape will be copied from.
        dst (numpy.array): 2D array where the circular shape will be copied to.
        radius (scalar): scalar value of the radius.

    Returns:
        numpy.array: Output monochrome image (2D array)
    """
    raise NotImplementedError


def copy_paste_middle_circle(src, dst, radius):
    x, y = int((src.shape[0]) / 2), int((src.shape[1]) / 2)
    radius = int(radius)

    srcRSt = x - radius

    srcCSt = y - radius

    destRSt = int((dst.shape[0]) / 2) - radius

    destCSt = int((dst.shape[1]) / 2) - radius

    dst_img = np.copy(dst)
    count = 0
    for row in range(radius * 2+1):
        for col in range(radius * 2+1):
            if isWithinCircle((x, y), radius, (srcRSt + row, srcCSt + col)):
                count = count + 1
                dst_img[destRSt + row:destRSt + row+1, destCSt + col:destCSt + col+1] = src[srcRSt + row:srcRSt + row+1, srcCSt + col:srcCSt + col+1]
    dst_img = dst_img.astype('int')
    print(count)
    return dst_img




def image_stats(image):
    return (float(image.min()), float(image.max()), image.mean(), image.std())
    """ Returns the tuple (min,max,mean,stddev) of statistics for the input monochrome image.
    In order to become more familiar with Numpy, you should look for pre-defined functions
    that do these operations i.e. numpy.min.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.

    Returns:
        tuple: Four-element tuple containing:
               min (float): Input array minimum value.
               max (float): Input array maximum value.
               mean (float): Input array mean / average value.
               stddev (float): Input array standard deviation.
    """
    raise NotImplementedError


def center_and_normalize(image, scale):
    return ((image - np.mean(image)) / np.std(image)) * scale + np.mean(image)
    """ Returns an image with the same mean as the original but with values scaled about the
    mean so as to have a standard deviation of "scale".

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        scale (int or float): scale factor.

    Returns:
        numpy.array: Output 2D image.
    """
    raise NotImplementedError


def shift_image_left(image, shift):
    transition_matrix = np.float32([[1, 0, shift*-1], [0, 1, 0]])
    temp_image = np.copy(image).astype('float32')
    shifted = cv.warpAffine(temp_image, transition_matrix, (image.shape[1] - shift, image.shape[0]))

    new_shifted = cv.copyMakeBorder(shifted, 0, 0,  0, shift, cv.BORDER_REPLICATE)
    return new_shifted
    """ Outputs the input monochrome image shifted shift pixels to the left.

    The returned image has the same shape as the original with
    the BORDER_REPLICATE rule to fill-in missing values.  See

    http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/copyMakeBorder/copyMakeBorder.html?highlight=copy

    for further explanation.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        shift (int): Displacement value representing the number of pixels to shift the input image.
            This parameter may be 0 representing zero displacement.

    Returns:
        numpy.array: Output shifted 2D image.
    """
    raise NotImplementedError


def difference_image(img1, img2):
    img1 = np.copy(img1).astype('double')
    img2 = np.copy(img2).astype('double')
    normalizedImg = np.zeros(img1.shape)
    normalizedImg = cv.normalize(img1 - img2, normalizedImg, 0, 255, cv.NORM_MINMAX)
    return normalizedImg
    """ Returns the difference between the two input images (img1 - img2). The resulting array must be normalized
    and scaled to fit [0, 255].

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        img1 (numpy.array): Input 2D image.
        img2 (numpy.array): Input 2D image.

    Returns:
        numpy.array: Output 2D image containing the result of subtracting img2 from img1.
    """
    raise NotImplementedError


def add_noise(image, channel, sigma):
    temp_image = np.copy(image).astype('double')
    rand_mat = np.random.normal(0.0, sigma, (temp_image.shape[0], temp_image.shape[1]))
    rand_mat.mean()
    temp_image[:, :, channel] = temp_image[:, :, channel] + rand_mat
    return temp_image
    """ Returns a copy of the input color image with Gaussian noise added to
    channel (0-2). The Gaussian noise mean must be zero. The parameter sigma
    controls the standard deviation of the noise.

    The returned array values must not be clipped or normalized and scaled. This means that
    there could be values that are not in [0, 255].

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): input RGB (BGR in OpenCV) image.
        channel (int): Channel index value.
        sigma (float): Gaussian noise standard deviation.

    Returns:
        numpy.array: Output 3D array containing the result of adding Gaussian noise to the
            specified channel.
    """
    raise NotImplementedError


def build_hybrid_image(image1, image2, cutoff_frequency):
    """ 
    Takes two images and creates a hybrid image given a cutoff frequency.
    Args:
        image1: numpy nd-array of dim (m, n, c)
        image2: numpy nd-array of dim (m, n, c)
        cutoff_frequency: scalar
    
    Returns:
        hybrid_image: numpy nd-array of dim (m, n, c)


how much high frequency to remove from the first image and how much low frequency to leave in the
second image. This is called the "cutoff-frequency". In the starter code, the cutoff frequency is
controlled by changing the standard deviation of the Gausian filter used in constructing the hybrid
images.
    Credits:
        Assignment developed based on a similar project by James Hays. 
    """
    image1 = np.copy(image1).astype('double')
    image2 = np.copy(image2).astype('double')

    filter = cv.getGaussianKernel(ksize=cutoff_frequency * 4 + 1,
                                   sigma=cutoff_frequency)
    filter = np.dot(filter, filter.T)

    low_frequencies = cv.filter2D(image1, -1, filter)

    high_frequencies = image2 - cv.filter2D(image2, -1, filter)

    return low_frequencies + high_frequencies


def vis_hybrid_image(hybrid_image):
    """ 
    Tools to visualize the hybrid image at different scale.

    Credits:
        Assignment developed based on a similar project by James Hays. 
    """

    scales = 5
    scale_factor = 0.5
    padding = 5
    original_height = hybrid_image.shape[0]
    num_colors = 1 if hybrid_image.ndim == 2 else 3

    output = np.copy(hybrid_image)
    cur_image = np.copy(hybrid_image)
    for scale in range(2, scales + 1):
        # add padding
        output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                            dtype=np.float32)))

        # downsample image
        cur_image = cv.resize(cur_image, (0, 0), fx=scale_factor, fy=scale_factor)

        # pad the top to append to the output
        pad = np.ones((original_height - cur_image.shape[0], cur_image.shape[1],
                       num_colors), dtype=np.float32)
        tmp = np.vstack((pad, cur_image))
        output = np.hstack((output, tmp))

    return output
