import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
def scaling(img,val):
    return img * val


def blend(a, b, alpha):
    img = a*alpha + b*(1.0 - alpha)
    return img.astype('uint8')


def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output




def find_template_1D(t, s):
    # TODO: Locate template t in signal s and return index. Use scipy.signal.correlate2d
    pass




def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.



    # cropped = img[110:310, 10:760]
    # cv.imshow("Cropped" , cropped)
    # ht, width, channel = cropped.shape
    # print(width)
    # green_channel = img[:, :, 1]
    # cv.imshow("Green Channel", green_channel);
    # print(green_channel[150, :])
    # plt.plot(green_channel[150,:])
    # plt.show()

    img1 = cv.cvtColor(cv.imread('img/image.jpg'), cv.COLOR_BGR2GRAY)
    cropped = img1[110:310, 10:760]
    cv.imshow("Cropped", cropped)
    #img1 = cv.imread('img/buil.jpg')
    #img2 = cv.cvtColor(cv.imread('img/fruit.jpeg'), cv.COLOR_BGR2GRAY)
    #gray = cv.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(img1.shape)
    #print(img2.shape)

    #summed = img1 + img2
    #cv.imshow("Summed", summed)


    avg = (img1 + img2)/2
    print(img1.shape)
    sub = img1 - img2
    print(sub)
    diff = cv.absdiff(img1,img2)
    sub = sub.astype('uint8')
    diff = diff.astype('uint8')
    ht, wt = img1.shape
    rand_mat = np.random.rand(ht, wt)*10
    img1 = img1 + rand_mat
    img1 = img1.astype('uint8')
    sp_img = sp_noise(img1, 0.05)
    blur = cv.GaussianBlur(img1, (11, 11), 7, borderType=cv.BORDER_WRAP )
    blur = cv.medianBlur(sp_img,3)
    res = cv.matchTemplate(img1, cropped, cv.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    print(res)


    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum

    # top_left = max_loc
    # bottom_right = (top_left[0] + (760 - 10 + 1), top_left[1] + (310 - 110 + 1))
    #
    # cv.rectangle(img1, top_left, bottom_right, 255, 2)
    #
    # plt.subplot(121), plt.imshow(res, cmap='gray')
    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(img1, cmap='gray')
    # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    #
    #
    # plt.show()

    #cv.imshow("Window", res.astype('uint8'))







    cv.waitKey(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
