import cv2
import ps5
import os
import numpy as np

# I/O directories
input_dir = "input_images"
output_dir = "output"

NOISE_1 = {'x': 2.5, 'y': 2.5}
NOISE_2 = {'x': 7.5, 'y': 7.5}


# Helper code
def run_particle_filter(filter_class, imgs_dir, template_rect,
                        save_frames={}, **kwargs):
    """Runs a particle filter on a given video and template.

    Create an object of type pf_class, passing in initial video frame,
    template (extracted from first frame using template_rect), and any
    keyword arguments.

    Do not modify this function except for the debugging flag.

    Args:
        filter_class (object): particle filter class to instantiate
                           (e.g. ParticleFilter).
        imgs_dir (str): path to input images.
        template_rect (dict): template bounds (x, y, w, h), as float
                              or int.
        save_frames (dict): frames to save
                            {<frame number>|'template': <filename>}.
        **kwargs: arbitrary keyword arguments passed on to particle
                  filter class.

    Returns:
        None.
    """

    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()

    # Initialize objects
    template = None
    pf = None
    frame_num = 0

    # Loop over video (till last frame or Ctrl+C is presssed)
    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))

        # Extract template and initialize (one-time only)
        if frame_num > -1 and template is None:
            template = frame[int(template_rect['y']):
                             int(template_rect['y'] + template_rect['h']),
                             int(template_rect['x']):
                             int(template_rect['x'] + template_rect['w'])]
            cv2.imshow("Template", template)


            if 'template' in save_frames:
                cv2.imwrite(save_frames['template'], template)

            pf = filter_class(frame, template, **kwargs)

        # Process frame
        # out_frame = frame.copy()
        # pf.render(out_frame)
        # cv2.imshow('Tracking', out_frame)
        # cv2.waitKey(1)
        if frame_num > -1:
            pf.process(frame)

        if frame_num > -1:  # For debugging, it displays every frame
            out_frame = frame.copy()
            pf.render(out_frame)
            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(10)

        # Render and save output, if indicated
        if frame_num > -1 and frame_num in save_frames:
            frame_out = frame.copy()
            pf.render(frame_out)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 1 == 0:
            print('Working on frame {}'.format(frame_num))
    return 0




# Helper code
def run_particle_filter_multiple(filter_class, imgs_dir, template_rect,
                        save_frames={}, **kwargs):
    """Runs a particle filter on a given video and template.

    Create an object of type pf_class, passing in initial video frame,
    template (extracted from first frame using template_rect), and any
    keyword arguments.

    Do not modify this function except for the debugging flag.

    Args:
        filter_class (object): particle filter class to instantiate
                           (e.g. ParticleFilter).
        imgs_dir (str): path to input images.
        template_rect (dict): template bounds (x, y, w, h), as float
                              or int.
        save_frames (dict): frames to save
                            {<frame number>|'template': <filename>}.
        **kwargs: arbitrary keyword arguments passed on to particle
                  filter class.

    Returns:
        None.
    """

    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()

    # Initialize objects
    template2 = None

    pf = None
    frame_num = 0

    template1 = None
    template3 = None

    # Loop over video (till last frame or Ctrl+C is presssed)
    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))

        # Extract template and initialize (one-time only)
        if template1 is None:
            template1 = frame[int(template_rect[0]['y']):
                             int(template_rect[0]['y'] + template_rect[0]['h']),
                             int(template_rect[0]['x']):
                             int(template_rect[0]['x'] + template_rect[0]['w'])]
            # cv2.imshow("Template1", template1)

        # if template2 is None:
        #     template2 = frame[int(template_rect[1]['y']):
        #                         int(template_rect[1]['y'] + template_rect[1]['h']),
        #                 int(template_rect[1]['x']):
        #                 int(template_rect[1]['x'] + template_rect[1]['w'])]
            # cv2.imshow("Template2" , template2)

        # if template3 is None and frame_num > 25:
        #     template3 = frame[int(template_rect[2]['y']):
        #                         int(template_rect[2]['y'] + template_rect[2]['h']),
        #                 int(template_rect[2]['x']):
        #                 int(template_rect[2]['x'] + template_rect[2]['w'])]
            # cv2.imshow("Template3", template3)




        if 'template' in save_frames:
            cv2.imwrite(save_frames['template'], template1)
        kwargs1 = kwargs.copy()
        kwargs1['template_coords'] = template_rect[0]
        pf1 = filter_class(frame, template1, **kwargs1)

        kwargs2 = kwargs.copy()
        kwargs2['template_coords'] = template_rect[1]
        # pf2 = filter_class(frame, template2, **kwargs2)
        if frame_num > 25:
            kwargs3 = kwargs.copy()
            kwargs3['template_coords'] = template_rect[2]
            # pf3 = filter_class(frame, template3, **kwargs3)

        # Process frame
        # out_frame = frame.copy()
        # pf.render(out_frame)
        # cv2.imshow('Tracking', out_frame)
        # cv2.waitKey(1)
        pf1.process(frame)
        # pf2.process(frame)
        # if frame_num > 25:
            # pf3.process(frame)

        if True:  # For debugging, it displays every frame
            out_frame = frame.copy()
            pf1.render(out_frame)
            # pf2.render(out_frame)
            # if frame_num > 25:
                # pf3.render(out_frame)
            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num in save_frames:
            frame_out = frame.copy()
            pf1.render(frame_out)
            # pf2.render(frame_out)
            # if frame_num > 25:
                # pf3.render(frame_out)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 20 == 0:
            print('Working on frame {}'.format(frame_num))
    return 0



def run_kalman_filter(filter_class,
                      imgs_dir,
                      noise,
                      sensor,
                      save_frames={},
                      template_loc=None,
                      Q=0.1 * np.eye(4),
                      R=0.1 * np.eye(2)):
    kf = filter_class(template_loc['x'], template_loc['y'], Q, R)

    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.png')]
    imgs_list.sort()

    frame_num = 0

    if sensor == "hog":
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    elif sensor == "matching":
        frame = cv2.imread(os.path.join(imgs_dir, imgs_list[0]))
        template = frame[template_loc['y']:
                         template_loc['y'] + template_loc['h'],
                         template_loc['x']:
                         template_loc['x'] + template_loc['w']]

    else:
        raise ValueError("Unknown sensor name. Choose between 'hog' or "
                         "'matching'")

    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))

        # Sensor
        if sensor == "hog":
            (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
                                                    padding=(8, 8), scale=1.05)

            if len(weights) > 0:
                max_w_id = np.argmax(weights)
                z_x, z_y, z_w, z_h = rects[max_w_id]

                z_x += z_w // 2
                z_y += z_h // 2

                z_x += np.random.normal(0, noise['x'])
                z_y += np.random.normal(0, noise['y'])

        elif sensor == "matching":
            corr_map = cv2.matchTemplate(frame, template, cv2.TM_SQDIFF)
            z_y, z_x = np.unravel_index(np.argmin(corr_map), corr_map.shape)

            z_w = template_loc['w']
            z_h = template_loc['h']

            z_x += z_w // 2 + np.random.normal(0, noise['x'])
            z_y += z_h // 2 + np.random.normal(0, noise['y'])

        x, y = kf.process(z_x, z_y)

        if False:  # For debugging, it displays every frame
            out_frame = frame.copy()
            cv2.circle(out_frame, (int(z_x), int(z_y)), 20, (0, 0, 255), 2)
            cv2.circle(out_frame, (int(x), int(y)), 10, (255, 0, 0), 2)
            cv2.rectangle(out_frame, (int(z_x) - z_w // 2, int(z_y) - z_h // 2),
                          (int(z_x) + z_w // 2, int(z_y) + z_h // 2),
                          (0, 0, 255), 2)

            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num in save_frames:
            frame_out = frame.copy()
            cv2.circle(frame_out, (int(x), int(y)), 10, (255, 0, 0), 2)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 20 == 0:
            print('Working on frame {}'.format(frame_num))
    return 0

