import numpy as np
import cv2
def print_hi(p, f):
    A = np.array([f, 0, 0, 0], [0, f, 0, 0], [0, 0, 1, 0])
    return np.array([(p[0]/p[2])*f, (p[1]/p[2])*f])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img = np.ones((4, 4))
    assert (np.allclose(cv2.multiply(img, 256), np.multiply(img, 256)))
    print("No Error")