"""Problem Set 7: Particle Filter Tracking."""

import cv2
import numpy as np
import math
import os
import sys
from ps5_utils import run_kalman_filter, run_particle_filter

np.random.seed(42)  # DO NOT CHANGE THIS SEED VALUE

# I/O directories
input_dir = "input"
output_dir = "output"

# TODO: Remove unnecessary classes

MIN_FLOAT = sys.float_info.min


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

def fix_size(frame_cutout, template):
    hf, wf = frame_cutout.shape

    ht, wt = template.shape

    h = min(hf, ht)
    w = min(wt, wf)

    template1 = template[: h, :w]
    frame_cutout1 = frame_cutout[:h, :w]
    return (frame_cutout1, template1)


class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        self.deltaT = 1.0
        self.center = np.array([init_x, init_y, 0., 0.])  # state
        self.covariance = np.diag([55.0] * 4)  # P
        self.transition = np.diag([1.0] * 4)
        self.transition[:2, 2:] = np.diag([self.deltaT] * 2)  # Dt
        self.measurementmt = np.matrix(np.diag([1.0] * 4)[:2])  # Mt or H
        self.Q = Q
        self.R = R

    def predict(self):
        self.center = self.transition * self.center
        self.covariance = self.transition * self.covariance * self.transition.T + self.Q

    def correct(self, meas_x, meas_y):
        measurement_at_time_t = np.array([[meas_x], [meas_y]])
        Kt = self.covariance * self.measurementmt.T * np.linalg.inv(
            self.measurementmt * self.covariance * self.measurementmt.T + self.R)
        self.center = self.center + Kt * (measurement_at_time_t - self.measurementmt * self.center)
        self.covariance = (np.diag([1.0] * 4) - Kt * self.measurementmt) * self.covariance

    def process(self, measurement_x, measurement_y):
        self.predict()
        self.correct(measurement_x, measurement_y)
        return self.center[0, 0], self.center[1, 0]


class ParticleFilter(object):
    """A particle filter tracker.
    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.
        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.
        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        self.isblur = kwargs.get('is_blur', False)
        self.template_kernel = int(kwargs.get('template_kernel', 0))
        self.frame_kernel = kwargs.get('frame_kernel', 0)
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #

        self.template = cv2.cvtColor(template.copy(), cv2.COLOR_BGR2GRAY).astype(float)
        if self.isblur:
            print("Blur Enabled")
            self.template = cv2.GaussianBlur(self.template, (self.template_kernel, self.template_kernel), 0)
        self.frame = None
        self.particles = None  # Initialize during processing
        self.weights = np.random.random(self.num_particles)
        self.weights = self.weights / np.sum(self.weights)

        self.x, self.y = (
            self.template_rect['x'] + self.template_rect['w'] / 2,
            self.template_rect['y'] + self.template_rect['h'] / 2)

        x = np.random.normal(self.x, self.sigma_dyn, self.num_particles)
        y = np.random.normal(self.y, self.sigma_dyn, self.num_particles)

        self.particles = np.vstack((x, y)).T

    def get_particles(self):
        """Returns the current particles state.
        This method is used by the autograder. Do not modify this function.
        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.
        This method is used by the autograder. Do not modify this function.
        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.
        Returns:
            float: similarity value.
        """
        # Use mse of template vs frame
        hf, wf = frame_cutout.shape

        ht, wt = template.shape

        h = min(hf, ht)
        w = min(wt, wf)

        template = template[: h, :w]
        frame_cutout = frame_cutout[:h, :w]

        sum1 = np.sum((template - frame_cutout) ** 2)

        # for x in (0, frame_cutout.shape[0] - template.shape[0]):

        #     for y in (0, frame_cutout.shape[1] - template.shape[1]):

        #         sum1 = sum1 + np.sum(template - frame_cutout[x:x+template.shape[0], y:y+template.shape[1]])**2

        sum1 = sum1 / (template.shape[0] * template.shape[1])

        return np.exp(-sum1 / (2.0 * (self.sigma_exp ** 2.0)))

    def resample_particles(self):
        """Returns a new set of particles
        This method does not alter self.particles.
        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.
        See np.random.choice or np.random.multinomial.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles[
            np.random.choice(a=self.num_particles, size=self.num_particles, p=self.weights, replace=True)]

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.
        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.
        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.
        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].
        Returns:
            None.
        """

        frm = None
        # if self.isblur:
        #     frm = cv2.medianBlur(frame.copy(), self.frame_kernel)
        #     frm = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY).astype(float)
        # else:
        frm = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY).astype(np.float64)
        x = np.random.normal(self.x, self.sigma_dyn, self.num_particles)
        y = np.random.normal(self.y, self.sigma_dyn, self.num_particles)

        self.particles = np.vstack((x, y)).T

        for i in range(len(self.particles)):
            particle = self.particles[i]
            cut_out = self.get_frame_cutout(frm, (particle[0], particle[1]))
            self.weights[i] = self.get_error_metric(self.template, cut_out)

        self.weights /= np.sum(self.weights)

        self.particles = self.resample_particles()

        x_weighted_mean = 0.0
        y_weighted_mean = 0.0
        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        self.x = x_weighted_mean
        self.y = y_weighted_mean

        # print("X - {}, {} Y - {}, {}".format(x_weighted_mean, self.x, y_weighted_mean, self.y))

    def get_frame_cutout(self, frame, center):

        frm_x, frm_y = frame.shape[0], frame.shape[1]
        center_x, center_y = center[0], center[1]
        template_depth, template_width = self.template.shape[0], self.template.shape[1]

        fx_from, fx_to = center_x - (template_width / 2), center_x + (template_width / 2)
        fy_from, fy_to = center_y - (template_depth / 2), center_y + (template_depth / 2)

        if fx_from <= 0:
            fx_from, fx_to = 1, 1 + template_width

        if fy_from <= 0:
            fy_from, fy_to = 1, 1 + template_depth

        if fx_to > frm_y - 1:
            fx_from, fx_to = fx_from - (fx_to - frm_y) - 1, frm_y - 1

        if fy_to > frm_x - 1:
            fy_from, fy_to = fy_from - (fy_to - frm_x) - 1, frm_x - 1

        return frame[int(fy_from):int(fy_to), int(fx_from):int(fx_to)]

    def render(self, frame_in):
        """Visualizes current particle filter state.
        This method may not be called for all frames, so don't do any model
        updates here!
        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.
        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:
        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.
        This function should work for all particle filters in this problem set.
        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """
        '''
        x_weighted_mean = 0
        y_weighted_mean = 0
        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]
        # Complete the rest of the code as instructed.
        raise NotImplementedError
        '''

        x, y = self.x, self.y
        h, w = self.template.shape[0], self.template.shape[1]

        cv2.rectangle(frame_in, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                      (255, 255, 255), 2)

        for pt_x, pt_y in self.particles.astype((int)):
            cv2.circle(frame_in, (pt_x, pt_y), 1, (0, 255, 0), -1)

        center = np.array([self.x, self.y])
        dist = np.sqrt(((self.particles[:, :-1] - center) ** 2).sum(axis=1))
        radius = np.average(dist, axis=0, weights=self.weights).astype(np.int)
        cv2.circle(frame_in, (int(x), int(y)), radius, (255, 255, 255), 1)


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha', 0.2)  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        self.first_template = template

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        ParticleFilter.process(self, frame)
        frm = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY).astype(np.float64)
        best = self.get_frame_cutout(frm, (self.x, self.y))
        best, template = fix_size(best, self.template)
        self.template = best * self.alpha + (1.0 - self.alpha) * template


class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        self.alpha = kwargs.get('alpha', 100) / 100.0

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        frm = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY).astype(np.float64)
        x = np.random.normal(self.x, self.sigma_dyn, self.num_particles)
        y = np.random.normal(self.y, self.sigma_dyn, self.num_particles)
        scale = np.random.randint(90, 101, size=self.num_particles) / 100.00

        self.particles = np.vstack((np.vstack((x, y)), scale)).T
        templates = {}
        scale_list = []
        similarity_ls = []
        for i in range(len(self.particles)):
            particle_n_scale = self.particles[i]
            scale = particle_n_scale[2]
            resized_template = cv2.resize(self.template.copy(), (0, 0), fx=scale, fy=scale)
            scale_list.append(scale)
            cut_out = self.get_frame_cutout(frm, (particle_n_scale[0], particle_n_scale[1]))
            resized_cut_out = cv2.resize(cut_out, (resized_template.shape[1], resized_template.shape[0]))

            if resized_template.shape != resized_cut_out.shape:
                resized_frame_cut = cv2.resize(src=resized_frame_cut, dsize=resized_template.shape[::-1]).astype(
                    (float))

            sim = self.get_error_metric(resized_template, resized_cut_out)
            self.weights[i] = sim
            similarity_ls.append(sim)
            templates[i] = resized_template

        idx = np.argmax(similarity_ls)
        print("IDX - {}".format(scale_list[idx]))

        if scale_list[idx] > 0.96:
            self.template = templates[idx]

        self.weights /= np.sum(self.weights)

        # self.particles = self.resample_particles()
        #
        # x_weighted_mean = 0.0
        # y_weighted_mean = 0.0
        # for i in range(self.num_particles):
        #     x_weighted_mean += self.particles[i, 0] * self.weights[i]
        #     y_weighted_mean += self.particles[i, 1] * self.weights[i]
        #
        # state = np.average(self.particles, 0, self.weights).astype(np.int)
        # self.x = state[0]
        # self.y = state[1]

    def render(self, frame_in):
        """Visualizes current particle filter state.
        This method may not be called for all frames, so don't do any model
        updates here!
        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.
        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:
        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.
        This function should work for all particle filters in this problem set.
        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """
        '''
        x_weighted_mean = 0
        y_weighted_mean = 0
        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]
        # Complete the rest of the code as instructed.
        raise NotImplementedError
        '''

        x, y = self.x, self.y
        h, w = self.template.shape[0], self.template.shape[1]

        cv2.rectangle(frame_in, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                      (255, 255, 255), 2)

        for pt_x, pt_y, _ in self.particles.astype((int)):
            cv2.circle(frame_in, (pt_x, pt_y), 1, (0, 255, 0), -1)

        center = np.array([self.x, self.y])
        dist = np.sqrt(((self.particles[:, :-1] - center) ** 2).sum(axis=1))
        radius = np.average(dist, axis=0, weights=self.weights).astype(np.int)
        print(radius)
        cv2.circle(frame_in, (int(x), int(y)), radius, (255, 255, 255), 1)


class MDParticleFilterVelocityAndScale(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):

        """Initializes MD particle filter object.


        The documentation for this class is the same as the ParticleFilter

        above. By calling super(...) all the elements used in ParticleFilter

        will be inherited so you don't have to declare them again.

        """

        self.num_particles = kwargs.get('num_particles')  # required by the autograder

        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder

        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder

        self.template_rect = kwargs.get('template_coords')  # required by the autograder

        self.isblur = kwargs.get('is_blur', False)

        self.template_kernel = int(kwargs.get('template_kernel', 0))

        self.frame_kernel = kwargs.get('frame_kernel', 0)

        # If you want to add more parameters, make sure you set a default value so that

        # your test doesn't fail the autograder because of an unknown or None value.

        #

        self.template = cv2.cvtColor(template.copy(), cv2.COLOR_BGR2GRAY)

        if self.isblur:
            print("Blur Enabled")

            self.template = cv2.GaussianBlur(self.template, (self.template_kernel, self.template_kernel), 0)

        self.frame = None

        self.particles = None  # Initialize during processing

        self.weights = np.random.random(self.num_particles)

        self.weights = self.weights / np.sum(self.weights)

        self.x, self.y = (

            self.template_rect['x'] + self.template_rect['w'] / 2,

            self.template_rect['y'] + self.template_rect['h'] / 2)

        self.vx = 0.0

        self.vy = 0.0

        self.scale = 1.000

        x = np.random.normal(self.x, self.sigma_dyn, self.num_particles)

        y = np.random.normal(self.y, self.sigma_dyn, self.num_particles)

        self.particles = np.vstack((x, y)).T

        self.alpha = kwargs.get('alpha', 100) / 100.0

    def process(self, frame):

        """Processes a video frame (image) and updates the filter's state.


        This process is also inherited from ParticleFilter. Depending on your

        implementation, you may comment out this function and use helper

        methods that implement the "More Dynamics" procedure.


        Args:

            frame (numpy.array): color BGR uint8 image of current video frame,

                                 values in [0, 255].


        Returns:

            None.

        """

        frm = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY).astype(np.float64)

        x = np.random.normal(self.x + self.vx, self.sigma_dyn, self.num_particles)

        y = np.random.normal(self.y + self.vy, self.sigma_dyn, self.num_particles)

        scale = np.random.normal(0.99, 0.1, self.num_particles)

        vx = x - self.x + np.random.normal(0, 1.0, self.num_particles)

        vy = y - self.y + np.random.normal(0, 1.0, self.num_particles)

        velocity = np.vstack((vx, vy))

        self.particles = np.vstack((np.vstack((np.vstack((x, y)), scale)), velocity)).T

        templates = {}

        scale_list = []

        similarity_ls = []

        for i in range(len(self.particles)):

            particle_n_scale = self.particles[i]

            scale = particle_n_scale[2]

            x = int(self.template.shape[0] * scale)
            y = int(self.template.shape[1] * scale)
            resized_template = cv2.resize(self.template.copy(), (0, 0), fx=scale, fy = scale)

            scale_list.append(scale)

            cut_out = self.get_frame_cutout(frm, (particle_n_scale[0], particle_n_scale[1]))

            resized_cut_out = cv2.resize(cut_out, (resized_template.shape[1], resized_template.shape[0]))

            if resized_template.shape != resized_cut_out.shape:
                resized_frame_cut = cv2.resize(src=resized_frame_cut, dsize=resized_template.shape[::-1]).astype(

                    (float))

            sim = self.get_error_metric(resized_template.astype(float), resized_cut_out)

            self.weights[i] = sim

            similarity_ls.append(sim)

            templates[i] = resized_template

        idx = np.argmax(similarity_ls)
        position = ((int)(5), (int)(25))
        cv2.putText(
            frame,  # numpy array on which text is written
            "Similarity-{}".format(similarity_ls[idx]),  # text
            position,  # position at which writing has to start
            cv2.FONT_ITALIC,  # font family
            0.5,  # font size
            (0, 0, 0),  # font color
            1)
        print("Particle-{}-{}-{}".format(self.particles[idx][0], self.particles[idx][1], self.particles[idx][2]))
        print("Similarity - {}".format(similarity_ls[idx]))
        if similarity_ls[idx] > 6.003402835201956e-06:

            print("******************Similarity - {}".format(similarity_ls[idx]))

            self.template = templates[idx]

            self.weights /= np.sum(self.weights)

            self.particles = self.resample_particles()

            cv2.imshow("Template ", self.template)

            x_weighted_mean = 0.0

            y_weighted_mean = 0.0

            vx_weighted_mean = 0.0

            vy_weighted_mean = 0.0

            scale = 0.0

            for i in range(self.num_particles):
                x_weighted_mean += self.particles[i, 0] * self.weights[i]

                y_weighted_mean += self.particles[i, 1] * self.weights[i]

                vx_weighted_mean += self.particles[i, 3] * self.weights[i]

                vy_weighted_mean += self.particles[i, 4] * self.weights[i]

                scale += self.particles[i, 2] * self.weights[i]

            self.x = x_weighted_mean

            self.y = y_weighted_mean

            self.vx = vx_weighted_mean

            self.vy = vx_weighted_mean

            self.scale = scale

        # print("Velocity -x - {} y - {}".format(self.vx, self.vy))

    def render(self, frame_in):

        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model

        updates here!

        These steps will calculate the weighted mean. The resulting values

        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay

        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be

          plotted by drawing a colored dot point on the image. Remember that

          this should be the center of the window, not the corner.

        - Draw the rectangle of the tracking window associated with the

          Bayesian estimate for the current location which is simply the

          weighted mean of the (x, y) of the particles.

        - Finally we need to get some sense of the standard deviation or

          spread of the distribution. First, find the distance of every

          particle to the weighted mean. Next, take the weighted sum of these

          distances and plot a circle centered at the weighted mean with this

          radius.

        This function should work for all particle filters in this problem set.

        Args:

            frame_in (numpy.array): copy of frame to render the state of the

                                    particle filter.

        """

        '''

        x_weighted_mean = 0

        y_weighted_mean = 0

        for i in range(self.num_particles):

            x_weighted_mean += self.particles[i, 0] * self.weights[i]

            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        # Complete the rest of the code as instructed.

        raise NotImplementedError

        '''

        x, y = self.x, self.y

        h, w = self.template.shape[0], self.template.shape[1]

        cv2.rectangle(frame_in, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),

                      (255, 255, 255), 2)

        for pt_x, pt_y, _, _, _ in self.particles.astype((int)):
            cv2.circle(frame_in, (pt_x, pt_y), 1, (0, 255, 0), -1)

        center = np.array([self.x, self.y])

        dist = np.sqrt(((self.particles[:, :-3] - center) ** 2).sum(axis=1))

        radius = np.average(dist, axis=0, weights=self.weights).astype(np.int)

        cv2.circle(frame_in, (int(x), int(y)), radius, (0, 0, 0), 1)


class MDParticleFilterVelocityAndConstantScale(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        self.isblur = kwargs.get('is_blur', False)
        self.template_kernel = int(kwargs.get('template_kernel', 0))
        self.frame_kernel = kwargs.get('frame_kernel', 0)
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #

        self.template = cv2.cvtColor(template.copy(), cv2.COLOR_BGR2GRAY)
        if self.isblur:
            print("Blur Enabled")
            self.template = cv2.GaussianBlur(self.template, (self.template_kernel, self.template_kernel), 0)
        self.frame = None
        self.particles = None  # Initialize during processing
        self.weights = np.random.random(self.num_particles)
        self.weights = self.weights / np.sum(self.weights)

        self.x, self.y = (
            self.template_rect['x'] + self.template_rect['w'] / 2,
            self.template_rect['y'] + self.template_rect['h'] / 2)
        self.vx = 0.0
        self.vy = 0.0
        self.scale = 0.99
        x = np.random.normal(self.x, self.sigma_dyn, self.num_particles)
        y = np.random.normal(self.y, self.sigma_dyn, self.num_particles)

        self.particles = np.vstack((x, y)).T
        self.alpha = kwargs.get('alpha', 100) / 100.0

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        frm = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY).astype(np.float64)
        x = np.random.normal(self.x + self.vx, self.sigma_dyn, self.num_particles)
        y = np.random.normal(self.y + self.vy, self.sigma_dyn, self.num_particles)
        scale = np.random.randint(1000, 1001, size=self.num_particles) / 1000.00
        vx = x - self.x + np.random.normal(0, 1.0, self.num_particles)
        vy = y - self.y + np.random.normal(0, 1.0, self.num_particles)
        velocity = np.vstack((vx, vy))

        self.particles = np.vstack((np.vstack((np.vstack((x, y)), scale)), velocity)).T
        templates = {}
        scale_list = []
        similarity_ls = []
        for i in range(len(self.particles)):
            particle_n_scale = self.particles[i]
            scale = particle_n_scale[2]
            x = int(self.template.shape[0] * scale)
            y = int(self.template.shape[1] * scale)
            resized_template = cv2.resize(self.template.copy(), (x, y))
            scale_list.append(scale)
            cut_out = self.get_frame_cutout(frm, (particle_n_scale[0], particle_n_scale[1]))
            resized_cut_out = cv2.resize(cut_out, (resized_template.shape[1], resized_template.shape[0]))

            if resized_template.shape != resized_cut_out.shape:
                resized_frame_cut = cv2.resize(src=resized_frame_cut, dsize=resized_template.shape[::-1]).astype(
                    (float))

            sim = self.get_error_metric(resized_template.astype(float), resized_cut_out)
            self.weights[i] = sim
            similarity_ls.append(sim)
            templates[i] = resized_template

        idx = np.argmax(similarity_ls)

        if similarity_ls[idx] > 0.00009:
            print("Similarity - {}".format(similarity_ls[idx]))
            self.template = templates[idx]
            self.weights /= np.sum(self.weights)
            self.particles = self.resample_particles()
            cv2.imshow("Template ", self.template)
            x_weighted_mean = 0.0
            y_weighted_mean = 0.0
            vx_weighted_mean = 0.0
            vy_weighted_mean = 0.0
            for i in range(self.num_particles):
                x_weighted_mean += self.particles[i, 0] * self.weights[i]
                y_weighted_mean += self.particles[i, 1] * self.weights[i]
                vx_weighted_mean += self.particles[i, 3] * self.weights[i]
                vy_weighted_mean += self.particles[i, 4] * self.weights[i]

            self.x = x_weighted_mean
            self.y = y_weighted_mean
            self.vx = vx_weighted_mean
            self.vy = vx_weighted_mean

        # print("Velocity -x - {} y - {}".format(self.vx, self.vy))

    def render(self, frame_in):
        """Visualizes current particle filter state.
        This method may not be called for all frames, so don't do any model
        updates here!
        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.
        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:
        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.
        This function should work for all particle filters in this problem set.
        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """
        '''
        x_weighted_mean = 0
        y_weighted_mean = 0
        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]
        # Complete the rest of the code as instructed.
        raise NotImplementedError
        '''

        x, y = self.x, self.y
        h, w = self.template.shape[0], self.template.shape[1]

        cv2.rectangle(frame_in, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                      (255, 255, 255), 2)

        for pt_x, pt_y, _, _, _ in self.particles.astype((int)):
            cv2.circle(frame_in, (pt_x, pt_y), 1, (0, 255, 0), -1)

        center = np.array([self.x, self.y])
        dist = np.sqrt(((self.particles[:, :-3] - center) ** 2).sum(axis=1))
        radius = np.average(dist, axis=0, weights=self.weights).astype(np.int)
        cv2.circle(frame_in, (int(x), int(y)), radius, (238, 130, 238), 1)


class MDParticleFilterVelocity(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        self.isblur = kwargs.get('is_blur', False)
        self.template_kernel = int(kwargs.get('template_kernel', 0))
        self.frame_kernel = kwargs.get('frame_kernel', 0)
        self.threshold = kwargs.get('threshold', 1.0) * 1.0
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #

        self.template = cv2.cvtColor(template.copy(), cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Template", self.template)
        # cv2.waitKey(0)
        # if self.isblur:
        #     print("Blur Enabled")
        #     self.template = cv2.GaussianBlur(self.template, (self.template_kernel, self.template_kernel), 0)
        self.frame = None
        self.particles = None  # Initialize during processing
        self.weights = np.random.random(self.num_particles)
        self.weights = self.weights / np.sum(self.weights)
        self.vx_sigma = kwargs.get("vx_sigma")
        self.vy_sigma = kwargs.get("vy_sigma")
        print("Vx Sigma - {} Vy Sigma".format(self.vx_sigma, self.vy_sigma))
        self.x, self.y = (
            self.template_rect['x'] + self.template_rect['w'] / 2,
            self.template_rect['y'] + self.template_rect['h'] / 2)
        self.vx = 0.0
        self.vy = 0.0
        self.scale = 0.99
        x = np.random.normal(self.x, self.sigma_dyn, self.num_particles)
        y = np.random.normal(self.y, self.sigma_dyn, self.num_particles)

        self.particles = np.vstack((x, y)).T
        self.alpha = kwargs.get('alpha', 20) / 100.0

    def resample_particles_temp(self, weights):
        """Returns a new set of particles
        This method does not alter self.particles.
        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.
        See np.random.choice or np.random.multinomial.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles[
            np.random.choice(a=self.num_particles, size=self.num_particles, p=weights, replace=True)]

    def std(self, particles, weights):

        x_weighted_mean = 0.0
        y_weighted_mean = 0.0
        vx_weighted_mean = 0.0
        vy_weighted_mean = 0.0
        for i in range(self.num_particles):
            x_weighted_mean += particles[i, 0] * weights[i]
            y_weighted_mean += particles[i, 1] * weights[i]
            vx_weighted_mean += particles[i, 2] * weights[i]
            vy_weighted_mean += particles[i, 3] * weights[i]

        new_center = np.array([x_weighted_mean, y_weighted_mean])
        current_center = np.array([self.x, self.y])
        dist = np.sqrt(((current_center - new_center) ** 2).sum(axis=0))
        # radius = np.average(dist, axis=0, weights=weights).astype(np.int)
        return dist

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        frm = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY).astype(np.float64)
        x = np.random.normal(self.x + self.vx, self.sigma_dyn, self.num_particles)
        y = np.random.normal(self.y + self.vy, self.sigma_dyn, self.num_particles)

        vx = np.random.normal((x - self.x), self.vx_sigma, self.num_particles)
        vy = np.random.normal((y - self.y), self.vy_sigma, self.num_particles)
        velocity = np.vstack((vx, vy))

        self.particles = np.vstack((np.vstack((x, y)), velocity)).T
        temp_weight = self.weights.copy()

        for i in range(len(self.particles)):
            particle_n_scale = self.particles[i]
            cut_out = self.get_frame_cutout(frm, (particle_n_scale[0], particle_n_scale[1]))

            sim = self.get_error_metric(self.template.astype(np.float64), cut_out)
            temp_weight[i] = sim

        idx = np.argmax(temp_weight)
        temp_weight /= np.sum(temp_weight)

        mean_dev = (np.max(temp_weight) - np.mean(temp_weight)) * 100.0
        print("Mean Dev - {}".format(mean_dev))
        print("STD - {}".format(np.std(temp_weight)))
        temp_particle = self.resample_particles_temp(temp_weight)
        radius = self.std(temp_particle, temp_weight)
        print("Radius - {}".format(radius))
        standard_deviation = np.std(temp_particle)
        if mean_dev < self.threshold:
            # print("radius - {}".format(radius))
            self.weights = temp_weight.copy()
            self.weights /= np.sum(self.weights)
            self.particles = self.resample_particles()

            # cv2.imshow("Template ", self.template)
            # cv2.waitKey(0)
            x_weighted_mean = 0.0
            y_weighted_mean = 0.0
            vx_weighted_mean = 0.0
            vy_weighted_mean = 0.0
            for i in range(self.num_particles):
                vx_weighted_mean += (self.particles[i, 2]) * self.weights[i]
                vy_weighted_mean += (self.particles[i, 3]) * self.weights[i]

                x_weighted_mean += (self.particles[i, 0]) * self.weights[i]
                y_weighted_mean += (self.particles[i, 1]) * self.weights[i]

            self.x = x_weighted_mean
            self.y = y_weighted_mean
            self.vx = vx_weighted_mean
            self.vy = vy_weighted_mean

        print("Cordinates -x  {} y  {}".format(self.x, self.y))
        print("Velocity - vx  {} vy  {}".format(self.vx, self.vy))

    def render(self, frame_in):
        """Visualizes current particle filter state.
        This method may not be called for all frames, so don't do any model
        updates here!
        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.
        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:
        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.
        This function should work for all particle filters in this problem set.
        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """
        '''
        x_weighted_mean = 0
        y_weighted_mean = 0
        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]
        # Complete the rest of the code as instructed.
        raise NotImplementedError
        '''

        x, y = self.x, self.y
        h, w = self.template.shape[0], self.template.shape[1]

        cv2.rectangle(frame_in, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                      (255, 255, 255), 2)

        for pt_x, pt_y, _, _ in self.particles.astype((int)):
            cv2.circle(frame_in, (pt_x, pt_y), 1, (0, 255, 0), -1)

        center = np.array([self.x, self.y])
        dist = np.sqrt(((self.particles[:, :-2] - center) ** 2).sum(axis=1))
        radius = np.average(dist, axis=0, weights=self.weights).astype(np.int)
        cv2.circle(frame_in, (int(x), int(y)), radius, (238, 130, 238), 2)


def run_particle_filter1(imgs_dir, template_rect,
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
    template0 = None
    pf1 = None

    template1 = None
    pf2 = None
    template2 = None
    frame_num = 0

    # Loop over video (till last frame or Ctrl+C is presssed)
    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))

        # Extract template and initialize (one-time only)
        if template0 is None:
            template0 = frame[int(template_rect[0]['y']):
                              int(template_rect[0]['y'] + template_rect[0]['h']),
                        int(template_rect[0]['x']):
                        int(template_rect[0]['x'] + template_rect[0]['w'])]
            # cv2.imshow("Template", template)
            # cv2.waitKey(0)

            if 'template' in save_frames:
                cv2.imwrite(save_frames['template'], template0)
            kwargs0 = kwargs.copy()
            kwargs0['template_coords'] = template_rect[0]
            kwargs0['num_particles'] = 400
            kwargs0['sigma_exp'] = 16
            kwargs0['sigma_dyn'] = 3
            kwargs0['vy_sigma'] = 1 / 1000.0
            kwargs0['vx_sigma'] = 1 / 1000.0
            kwargs0['threshold'] = 1
            kwargs0['alpha'] = 0
            pf1 = MDParticleFilterVelocity(frame, template0, **kwargs0)

        # Extract template and initialize (one-time only)
        if template1 is None:
            template1 = frame[int(template_rect[1]['y']):
                              int(template_rect[1]['y'] + template_rect[1]['h']),
                        int(template_rect[1]['x']):
                        int(template_rect[1]['x'] + template_rect[1]['w'])]
            # cv2.imshow("Template", template1)
            # cv2.waitKey(0)

            if 'template' in save_frames:
                cv2.imwrite(save_frames['template'], template1)
            kwargs1 = kwargs.copy()
            kwargs1['template_coords'] = template_rect[1]
            kwargs1['num_particles'] = 400
            kwargs1['sigma_exp'] = 16
            kwargs1['sigma_dyn'] = 6
            kwargs1['vy_sigma'] = 1 / 1000.0
            kwargs1['vx_sigma'] = 1 / 1000.0
            kwargs1['threshold'] = 7
            kwargs1['alpha'] = 0
            pf2 = MDParticleFilterVelocity(frame, template1, **kwargs1)

        if frame_num > 26 and template2 is None:
            template2 = frame[int(template_rect[2]['y']):
                              int(template_rect[2]['y'] + template_rect[2]['h']),
                        int(template_rect[2]['x']):
                        int(template_rect[2]['x'] + template_rect[2]['w'])]
            # cv2.imshow("Template", template2)
            # cv2.waitKey(0)

            if 'template' in save_frames:
                cv2.imwrite(save_frames['template'], template2)
            kwargs2 = kwargs.copy()
            kwargs2['template_coords'] = template_rect[2]
            kwargs2['num_particles'] = 400
            kwargs2['sigma_exp'] = 16
            kwargs2['sigma_dyn'] = 8
            kwargs2['vy_sigma'] = 1 / 1000.0
            kwargs2['vx_sigma'] = 1 / 1000.0
            kwargs2['threshold'] = 26
            kwargs2['alpha'] = 0
            pf3 = MDParticleFilterVelocity(frame, template2, **kwargs2)

        # Process frame
        # out_frame = frame.copy()
        # pf.render(out_frame)
        # cv2.imshow('Tracking', out_frame)
        # cv2.waitKey(1)
        pf1.process(frame)
        pf2.process(frame)
        if frame_num > 26:
            pf3.process(frame)

        if True:  # For debugging, it displays every frame
            out_frame = frame.copy()
            pf1.render(out_frame)
            pf2.render(out_frame)
            if frame_num > 26:
                pf3.render(out_frame)
            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(10)

        # Render and save output, if indicated
        if frame_num in save_frames:
            frame_out = frame.copy()
            pf1.render(frame_out)
            pf2.render(frame_out)
            if frame_num > 26:
                pf3.render(frame_out)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 1 == 0:
            print('Working on frame {}'.format(frame_num))
    return 0


def part_1b(obj_class, template_loc, save_frames, input_folder):
    Q = 0.1 * np.eye(4)  # Process noise array
    R = 0.1 * np.eye(2)  # Measurement noise array
    NOISE_2 = {'x': 7.5, 'y': 7.5}
    out = run_kalman_filter(obj_class, input_folder, NOISE_2, "matching",
                            save_frames, template_loc, Q, R)
    return out


def part_1c(obj_class, template_loc, save_frames, input_folder):
    Q = 0.1 * np.eye(4)  # Process noise array
    R = 0.1 * np.eye(2)  # Measurement noise array
    NOISE_1 = {'x': 2.5, 'y': 2.5}
    out = run_kalman_filter(obj_class, input_folder, NOISE_1, "hog",
                            save_frames, template_loc, Q, R)
    return out


def part_2a(obj_class, template_loc, save_frames, input_folder):
    num_particles = 200  # Define the number of particles
    sigma_mse = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 5  # Define the value of sigma for the particles movement (dynamics)
    num_particles = 377
    sigma_mse = 5.0
    sigma_dyn = 8.0
    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        template_loc,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        template_coords=template_loc)  # Add more if you need to
    return out


def part_2b(obj_class, template_loc, save_frames, input_folder):
    num_particles = 200  # Define the number of particles
    sigma_mse = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 5  # Define the value of sigma for the particles movement (dynamics)
    template_kernel = 5

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        template_loc,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        is_blur=False,
        template_kernel=5,
        frame_kernel=5,
        template_coords=template_loc)  # Add more if you need to
    return out


def part_3(obj_class, template_rect, save_frames, input_folder):
    num_particles = 338  # Define the number of particles
    sigma_mse = 7  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 31  # Define the value of sigma for the particles movement (dynamics)
    alpha = 2  # Set a value for alpha

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        # input video
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        alpha=alpha,
        template_coords=template_rect)  # Add more if you need to
    return out


def part_4(obj_class, template_rect, save_frames, input_folder):
    num_particles = 400  # Define the number of particles
    sigma_md = 16  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 5  # Define the value of sigma for the particles movement (dynamics)

    out = run_particle_filter(
        obj_class,
        input_folder,
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_md,
        sigma_dyn=sigma_dyn,
        template_coords=template_rect)  # Add more if you need to
    return out


def part_5(obj_class, template_rect, save_frames, input_folder):
    num_particles = 400  # Define the number of particles
    sigma_md = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 5  # Define the value of sigma for the particles movement (dynamics)

    out = run_particle_filter1(
        input_folder,
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_md,
        sigma_dyn=sigma_dyn,
        template_coords=template_rect)  # Add more if you need to
    return out
