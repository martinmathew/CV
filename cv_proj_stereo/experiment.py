import cv2
import os
import stereo_project as sp

INPUT_DIR = "input_img"
OUTPUT_DIR = "src/"


def save_image(filename, image):
    """Convenient wrapper for writing images to the output directory."""
    cv2.imwrite(cv2.os.path.join('./', filename), image)


def test_simple_ssd():


    left_img = cv2.imread(os.path.join(INPUT_DIR, 'adirondack/im0.png'))
    right_img = cv2.imread(os.path.join(INPUT_DIR, 'adirondack/im1.png'))
    depth_img = sp.calculate_ssd(left_img, right_img, 5)
    save_image("adirondack_ssd.png", depth_img)

    left_img = cv2.imread(os.path.join(INPUT_DIR, 'motorcycle/im0.png'))
    right_img = cv2.imread(os.path.join(INPUT_DIR, 'motorcycle/im1.png'))
    depth_img = sp.calculate_ssd(left_img, right_img, 5)
    save_image("motorcycle_ssd.png", depth_img)

    # left_img = cv2.imread(os.path.join(INPUT_DIR, 'jadeplant/im0.png'))
    # right_img = cv2.imread(os.path.join(INPUT_DIR, 'jadeplant/im1.png'))
    # depth_img  = sp.calculate_ssd(left_img, right_img, 8)
    # save_image("jadeplant_ssd.png", depth_img)


def test_energy_minimzation():
    # left_image = cv2.imread(os.path.join(INPUT_DIR, "inp/im0.jpg"))
    # right_image = cv2.imread(os.path.join(INPUT_DIR, "inp/im1.jpg"))

    left_image = cv2.imread(os.path.join(INPUT_DIR, 'motorcycle/im0.png'))
    right_image = cv2.imread(os.path.join(INPUT_DIR, 'motorcycle/im1.png'))
    disp = sp.graph_cut_energy_minimization(left_image, right_image)
    save_image("motorcycle_gc.png", disp)

    left_img = cv2.imread(os.path.join(INPUT_DIR, 'adirondack/im0.png'))
    right_img = cv2.imread(os.path.join(INPUT_DIR, 'adirondack/im1.png'))
    depth_img = sp.graph_cut_energy_minimization(left_img, right_img)
    save_image("adirondack_gc.png", depth_img)

    left_img = cv2.imread(os.path.join(INPUT_DIR, 'bowl/view0.png'))
    right_img = cv2.imread(os.path.join(INPUT_DIR, 'bowl/view1.png'))
    depth_img = sp.graph_cut_energy_minimization(left_img, right_img)
    save_image("bowl_gc.png", depth_img)


test_simple_ssd()
test_energy_minimzation()
