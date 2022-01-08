import cv2
import numpy as np
import experiment as ex
from scipy import signal
import unittest
import os
import ps5 as ps5
import traceback
from ps5_utils import run_kalman_filter, run_particle_filter
# Function documentation can be found in https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=createtrackbar
input_dir = "input_images"
output_dir = "./"
def nothing(x):
    pass




# img = cv2.imread('input_images/test_images/stop_249_149_background.png')
# img = cv2.imread('input_images/scene_stp_1.png')
# img = cv2.imread('input_images/scene_constr_1.png')
# img = cv2.imread('input_images/test_images/stop_249_149_blank.png')
# img = cv2.imread('input_images/test_images/no_entry_145_145_background.png')
#
# img = cv2.imread('input_images/test_images/no_entry_145_145_blank.png')
# img = cv2.imread('input_images/scene_all_signs.png')

# img = cv2.imread('input_images/scene_some_signs_noisy.png')
# img = cv2.imread('input_images/scene_all_signs_noisy.png')
# img = cv2.imread('input_images/real_images/no_entry/5833993298_4b09eec863_o.jpg')
# img = cv2.imread('input_images/real_images/no_entry/3328387196_6cb541db37_o.jpg')

# img1 = cv2.imread('input_images/Urban2/urban01.png')
# img2 = cv2.imread('input_images/Urban2/urban02.png')
#
# img1 = cv2.imread('input_images/TestSeq/Shift0.png.png')
# img2 = cv2.imread('input_images/TestSeq/ShiftR10.png')

# img1 = cv2.imread('input_images/test_images/test_lk3.png')
# img2 = cv2.imread('input_images/test_images/test_lk4.png')
input_dir = "input_images"
#
# img1 = cv2.imread(os.path.join(input_dir, 'MiniCooper', 'mc02.png'), 0) / 255.
# img2 = cv2.imread(os.path.join(input_dir, 'MiniCooper', 'mc03.png'), 0) / 255.

# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# img1 = cv2.GaussianBlur(img1, (5, 5), 0)
# img2 = cv2.GaussianBlur(img2, (5, 5), 0)
# img1 = img1.astype('double') / 255
# img2 = img2.astype('double') / 255

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
ke = "ke"
k_size = "k_size"
kk_size = "kk_size"
windows = "windows"
tau = "tau"
un_gau = "un_gau"
level = "level"
sublevel ="sublevel"

num_particles="num_particles"
sigma_mse="sigma_mse"
sigma_dyn="sigma_dyn"

is_blur = "is_blur"
template_kernel = "template_kernel"
frame_kernel = "frame_kernel"
alpha="alpha"
vx_sigma="vx_sigma"
vy_sigma="vy_sigma"
threshold="threshold"
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


# cv2.createTrackbar(dp, window, 1, 100, nothing)
# cv2.createTrackbar(minDist, window, 20, 200, nothing)
# cv2.createTrackbar(param1, window, 50, 100, nothing)
# cv2.createTrackbar(param2, window, 30, 100, nothing)
# cv2.createTrackbar(minRadius, window, 0, 200, nothing)
# cv2.createTrackbar(maxRadius, window, 0, 100, nothing)
# cv2.createTrackbar(blockSize, window, 11, 100, nothing)
# cv2.createTrackbar(C, window, 2, 100, nothing)
# cv2.createTrackbar(ke, window, 7, 100, nothing)
# cv2.createTrackbar(k_size, window, 48, 2400, nothing)
# cv2.createTrackbar(windows, window, 45, 100, nothing)
# cv2.createTrackbar(tau, window, 100, 10000, nothing)
# cv2.createTrackbar(un_gau, window, 2, 2, nothing)
# cv2.createTrackbar(level, window, 5, 50, nothing)
# cv2.createTrackbar(kk_size, window, 48, 2400, nothing)









cv2.createTrackbar(num_particles, window, 300, 2000, nothing)
cv2.createTrackbar(sigma_mse, window, 16, 200, nothing)
cv2.createTrackbar(sigma_dyn, window, 5, 200, nothing)

cv2.createTrackbar(template_kernel, window, 0, 200, nothing)
cv2.createTrackbar(frame_kernel, window, 0, 200, nothing)

cv2.createTrackbar(alpha, window, 0, 100, nothing)
cv2.createTrackbar(vx_sigma, window, 1, 10000, nothing)
cv2.createTrackbar(vy_sigma, window, 1, 10000, nothing)
cv2.createTrackbar(threshold, window, 26, 100, nothing)

# cv2.createTrackbar(sublevel, window, 4, 50, nothing)
# template_rect = [{'x': 290, 'y': 224, 'w': 50, 'h': 80},{'x': 59, 'y': 155, 'w': 90, 'h': 171},{'x': 0, 'y': 212, 'w': 32, 'h': 78}]

template_rect = {'x': 210, 'y': 37, 'w': 93, 'h': 285}

save_frames = {
    29: os.path.join(output_dir, 'ps5-5-a-1.png'),
    56: os.path.join(output_dir, 'ps5-5-a-2.png'),
    71: os.path.join(output_dir, 'ps5-5-a-3.png')
}
while 1:
    #
    # temp1 = img1.copy()
    # temp2 = img2.copy()

    k1 = cv2.waitKey(1) & 0xFF

    if k1 == 27:
        break

    k = cv2.getTrackbarPos('k', 'Params')
    if k % 2 == 0:
        k = k + 1

    # k_size = cv2.getTrackbarPos('k_size', 'Params')
    # windows = cv2.getTrackbarPos('windows', 'Params')
    # tau = cv2.getTrackbarPos('tau', 'Params')
    # un_gau = cv2.getTrackbarPos('un_gau', 'Params')
    # level = cv2.getTrackbarPos('level', 'Params')
    #
    # kk_size = cv2.getTrackbarPos('kk_size', 'Params')
    # kk_size = cv2.getTrackbarPos('kk_size', 'Params')

    num_particles = cv2.getTrackbarPos('num_particles', 'Params')
    sigma_mse = cv2.getTrackbarPos('sigma_mse', 'Params')
    sigma_dyn = cv2.getTrackbarPos('sigma_dyn', 'Params')
    template_kernel = cv2.getTrackbarPos('template_kernel', 'Params')
    frame_kernel = cv2.getTrackbarPos('frame_kernel', 'Params')
    alpha = cv2.getTrackbarPos('alpha', 'Params')

    vx_sigma = cv2.getTrackbarPos('vx_sigma', 'Params')
    vy_sigma = cv2.getTrackbarPos('vy_sigma', 'Params')
    vy_sigma = cv2.getTrackbarPos('threshold', 'Params')

    threshold = cv2.getTrackbarPos('threshold', 'Params')


    if template_kernel%2 == 0:
        template_kernel = template_kernel + 1
    if frame_kernel%2 == 0:
        frame_kernel = frame_kernel + 1

    # suite = unittest.TestSuite()
    # suite.addTest(ps5.PS5_PF_Tests("test_PF_base_2_trial" , num_particles= num_particles, sigma_mse=sigma_mse, sigma_dyn=sigma_dyn))
    try:
        out = run_particle_filter(
            ps5.MDParticleFilterVelocityAndScale,# particle filter model class
            os.path.join(input_dir, "pedestrians"),
            template_rect,
            save_frames,
            num_particles=num_particles,
            sigma_exp=sigma_mse,
            sigma_dyn=sigma_dyn,
            is_blur = True,
            template_kernel = template_kernel,
            frame_kernel = frame_kernel,
            alpha = alpha,
            vy_sigma = vy_sigma/1000.0,
            vx_sigma = vx_sigma/1000.0,
            threshold = threshold,
            template_coords=template_rect)
        # print(unittest.TextTestRunner().run(suite))
    except Exception as e: traceback.print_exc()
    # ps5.test_PF_base_2_trial(num_particles, sigma_mse, sigma_dyn)

    # sigma = 1  # TODO: Select a sigma value if you are using a gaussian kernel
    # interpolation = cv2.INTER_CUBIC  # You may try different values
    # border_mode = cv2.BORDER_REFLECT101  # You may try different values
    # # u40, v40 = ps4.hierarchical_lk(temp1, temp2, level, k_size, type,
    # #                                sigma, interpolation, border_mode)
    # temp1 = cv2.GaussianBlur(temp1, (k_size, k_size), 0)
    # temp2 = cv2.GaussianBlur(temp2, (k_size, k_size), 0)
    # u, v = ps4.optic_flow_lk(temp1, temp2, k_size, 'uniform')
    # u_v = ex.quiver(u, v, scale= 1, stride=10)



    # interpolation = cv2.INTER_CUBIC  # You may try different values
    # border_mode = cv2.BORDER_REFLECT101  # You may try different values
    # if kk_size%2 == 0:
    #     kk_size = kk_size + 1
    # tempp1 = cv2.GaussianBlur(temp1, (kk_size, kk_size), 0)
    # tempp2 = cv2.GaussianBlur(temp2, (kk_size, kk_size), 0)
    # u, v = ps4.hierarchical_lk(tempp1, tempp2, level, k_size, type,
    #                                sigma, interpolation, border_mode)
    #
    #
    #
    # # M, N = shift_0.shape
    # # X, Y = np.meshgrid(range(N), range(M))
    # itrps = [0.2, 0.4,]
    # res1 = temp1.copy()
    # for itrp in itrps:
    #     wi = ps4.warp(temp1, -itrp*u, -itrp*v,interpolation,border_mode)
    #     res1 = np.concatenate((res1,wi), axis= 1)
    #
    # itrps = [0.6, 0.8]
    # res2 = None
    # for itrp in itrps:
    #     wi = ps4.warp(temp1, -itrp * u, -itrp * v, interpolation, border_mode)
    #     if res2 is not None:
    #         res2 = np.concatenate((res2, wi), axis=1)
    #     else:
    #         res2 = wi
    # res2 = np.concatenate((res2, temp2), axis=1)
    # res = np.concatenate((res1, res2), axis=0)
    # cv2.imwrite(os.path.join(output_dir, "ps4-5-b-2.png"),
    #             ps4.normalize_and_scale(res))





    # Ix = cv2.Sobel(temp1, cv2.CV_64F, 1, 0, ksize=3, scale=1 / 8)
    # Iy = cv2.Sobel(temp1, cv2.CV_64F, 0, 1, ksize=3, scale=1 / 8)
    # It = temp2 - temp1
    # u = np.zeros(temp1.shape)
    # v = np.zeros(temp1.shape)
    # w = windows
    # tau = tau / 1000
    # for x in range(0, temp1.shape[0] - w):
    #     for y in range(0, temp1.shape[1] - w):
    #         Ixf = Ix[x - 1:x + w, y - 1:y + w].flatten()
    #         Iyf = Iy[x - 1:x + w, y - 1:y + w].flatten()
    #         Itf = It[x - 1:x + w, y - 1:y + w].flatten()
    #
    #         A = np.vstack((Ixf, Iyf)).T
    #         res = np.matmul(np.linalg.pinv(np.matmul(A.T, A)), -np.matmul(A.T, Itf))
    #         # print("X-{}, Y- {} Tau - {}".format(x,y, np.min(abs(np.linalg.eigvals(np.matmul(A.T, A))))))
    #         # if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
    #         u[x, y] = res[0]
    #         v[x, y] = res[1]
    # # u = (u - np.min(u))/np.ptp(u)
    # # v = (v - np.min(v))/np.ptp(v)
    # u_v = ex.quiver(u, v, scale=6, stride=10)
    # print("Rendering .............")

    # res = cv2.imread('ps4-5-b-2.png', 0)
    # cv2.imshow("Test", res)
    # cv2.imshow("Difference", temp1 - warp_img)
