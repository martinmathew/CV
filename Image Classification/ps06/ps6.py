"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import numpy as np
import cv2
import os

from helper_classes import WeakClassifier, VJ_Classifier

SBJ = "subject"
sbjlen = len(SBJ)


def get_num_from_text(str1):
    idx2 = str1.index('.')
    return int(str1[7:idx2:1])


# assignment code
def load_images(folder, size=(32, 32)):
    """Load images to workspace.

    Args:
        folder (String): path to folder with images.
        size   (tuple): new image sizes

    Returns:
        tuple: two-element tuple containing:
            X (numpy.array): data matrix of flatten images
                             (row:observations, col:features) (float).
            y (numpy.array): 1D array of labels (int).
    """

    images_files = [f for f in os.listdir(folder) if f.endswith(".png")]
    output = np.empty((0, size[0] * size[1]), np.float64)
    idxs = np.empty((len(images_files)), int)
    for i, img in enumerate(images_files):
        frame = cv2.imread(os.path.join(folder, img))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        flat = frame.flatten()
        output = np.append(output, flat.reshape(1, flat.shape[0]), axis=0)
        id = get_num_from_text(img)
        idxs[i] = id

    return output, idxs.reshape(len(images_files))


def split_dataset(X, y, p):
    """Split dataset into training and test sets.

    Let M be the number of images in X, select N random images that will
    compose the training data (see np.random.permutation). The images that
    were not selected (M - N) will be part of the test data. Record the labels
    accordingly.

    Args:
        X (numpy.array): 2D dataset.
        y (numpy.array): 1D array of labels (int).
        p (float): Decimal value that determines the percentage of the data
                   that will be the training data.

    Returns:
        tuple: Four-element tuple containing:
            Xtrain (numpy.array): Training data 2D array.
            ytrain (numpy.array): Training data labels.
            Xtest (numpy.array): Test data test 2D array.
            ytest (numpy.array): Test data labels.
    """
    clubbed_stack = np.column_stack((X, y))
    np.random.shuffle(clubbed_stack)
    training, test = clubbed_stack[:int(len(X) * p)], clubbed_stack[int(len(X) * p):]
    return training[:, :-1], training[:, -1], test[:, :-1], test[:, -1]


def get_mean_face(x):
    """Return the mean face.

    Calculate the mean for each column.

    Args:
        x (numpy.array): array of flattened images.

    Returns:
        numpy.array: Mean face.
    """
    # arr = np.mean(x, axis=0)
    # return np.reshape(arr , (-1, 32))
    return np.mean(x, axis=0)


def pca(X, k):
    """PCA Reduction method.

    Return the top k eigenvectors and eigenvalues using the covariance array
    obtained from X.


    Args:
        X (numpy.array): 2D data array of flatten images (row:observations,
                         col:features) (float).
        k (int): new dimension space

    Returns:
        tuple: two-element tuple containing
            eigenvectors (numpy.array): 2D array with the top k eigenvectors.
            eigenvalues (numpy.array): array with the top k eigenvalues.
    """

    mean_face = get_mean_face(X)
    u = np.dot((X - mean_face).T, (X - mean_face))
    eigen_value, eigen_vector = np.linalg.eigh(u)
    eigen_value = eigen_value[::-1][:k]
    eigen_vector = eigen_vector.T[::-1][:k].T
    return eigen_vector, eigen_value


class Boosting:
    """Boosting classifier.

    Args:
        X (numpy.array): Data array of flattened images
                         (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (observations, ).
        num_iterations (int): number of iterations
                              (ie number of weak classifiers).

    Attributes:
        Xtrain (numpy.array): Array of flattened images (float32).
        ytrain (numpy.array): Labels array (float32).
        num_iterations (int): Number of iterations for the boosting loop.
        weakClassifiers (list): List of weak classifiers appended in each
                               iteration.
        alphas (list): List of alpha values, one for each classifier.
        num_obs (int): Number of observations.
        weights (numpy.array): Array of normalized weights, one for each
                               observation.
        eps (float): Error threshold value to indicate whether to update
                     the current weights or stop training.
    """

    def __init__(self, X, y, num_iterations):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.num_iterations = num_iterations

        self.alphas = [0.0] * num_iterations
        self.num_obs = X.shape[0]
        self.weights = np.array([1.0 / self.num_obs] * self.num_obs)  # uniform weights
        self.weakClassifiers = [None] * num_iterations
        self.eps = 0.0001

    def train(self):
        """Implement the for loop shown in the problem set instructions."""
        # for i in range(0, self.num_iterations):
        #     mod = WeakClassifier(X=self.Xtrain, y=self.ytrain, weights=self.weights)
        #     mod.train()
        #     mod_j = mod.predict(np.transpose(self.Xtrain))
        #     erridx = self.ytrain != mod_j
        #     err_sum = np.sum(self.weights[erridx])/np.sum(self.weights)
        #     alpha = 0.5 * np.log((1. - err_sum)/err_sum)
        #     self.weakClassifiers = np.append(self.weakClassifiers, mod)
        #     self.alphas = np.append(self.alphas, alpha)
        #     if err_sum > self.eps:
        #         self.weights[erridx] = self.weights[erridx] * np.exp(-alpha * mod_j[erridx] * self.ytrain[erridx])
        #     else:
        #         break
        # epsilon = 9999.0

        for i in range(self.num_iterations):
            wk_clf = WeakClassifier(self.Xtrain, self.ytrain, self.weights)
            wk_clf.train()
            wk_results = np.array([wk_clf.predict(x) for x in self.Xtrain])
            idxes_not_eql = np.where(wk_results != self.ytrain)
            epsilon = np.sum(self.weights[idxes_not_eql]) / np.sum(self.weights)
            print(epsilon)
            alpha = 0.5 * np.log((1.0 - epsilon) / epsilon)
            self.alphas[i] = alpha
            if epsilon > self.eps:
                self.weights[idxes_not_eql] = self.weights[idxes_not_eql] * np.exp(
                    -alpha * wk_results[idxes_not_eql] * self.ytrain[idxes_not_eql])
                self.weakClassifiers[i] = wk_clf
            else:
                return

    def evaluate(self):
        """Return the number of correct and incorrect predictions.

        Use the training data (self.Xtrain) to obtain predictions. Compare
        them with the training labels (self.ytrain) and return how many
        where correct and incorrect.

        Returns:
            tuple: two-element tuple containing:
                correct (int): Number of correct predictions.
                incorrect (int): Number of incorrect predictions.
        """
        prediction = [self.predict(x) for x in self.Xtrain]
        good = np.sum(prediction == self.ytrain)
        return good, len(self.Xtrain) - good

    def predict(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.aphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """
        predicts = [cls.predict(X.T) for cls, alpha in zip(self.weakClassifiers, self.alphas)]
        alphaxpredict = [alpha * predict for predict, alpha in zip(predicts, self.alphas)]

        return np.sign(np.sum(alphaxpredict, axis=0))


class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (2, 2)}.
        position (tuple): (row, col) position of the feature's top left corner.
        size (tuple): Feature's (height, width)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size

    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        x, y = self.position
        ht, wd = self.size
        feature = np.zeros(shape)
        feature[x:x + int(ht / 2), y:y + wd] = 255
        feature[x + int(ht / 2):x + ht, y:y + wd] = 126
        return feature

    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        x, y = self.position
        ht, wd = self.size
        feature = np.zeros(shape)
        feature[x:x + ht, y:y + int(wd / 2)] = 255
        feature[x:x + ht, y + int(wd / 2):y + wd] = 126
        return feature

    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        x, y = self.position
        ht, wd = self.size
        feature = np.zeros(shape)
        feature[x:x + int(ht / 3), y:y + wd] = 255
        feature[x + int(ht / 3):x + int(2 * ht / 3), y:y + wd] = 126
        feature[x + int(2 * ht / 3): x + ht, y:y + wd] = 255
        return feature

    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        x, y = self.position
        ht, wd = self.size
        feature = np.zeros(shape)
        feature[x:x + ht, y:y + int(wd / 3)] = 255
        feature[x:x + ht, y + int(wd / 3):y + int(2 * wd / 3)] = 126
        feature[x:x + ht, y + int(2 * wd / 3):y + wd] = 255
        return feature

    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        x, y = self.position
        ht, wd = self.size
        feature = np.zeros(shape)
        feature[x:x + int(ht / 2), y:y + int(wd / 2)] = 126
        feature[x:x + int(ht / 2), y + int(wd / 2):y + wd] = 255
        feature[x + int(ht / 2):x + ht, y:y + int(wd / 2)] = 255
        feature[x + int(ht / 2):x + ht, y + int(wd / 2):y + wd] = 126
        return feature

    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (row, col) and (height, width) format.

        Args:
            shape (tuple): Array numpy-style shape (rows, cols).
                           Defaults to (24, 24).

        Returns:
            numpy.array: Array containing a Haar feature (float or uint8).
        """

        if self.feat_type == (2, 1):  # two_horizontal
            X = self._create_two_horizontal_feature(shape)

        if self.feat_type == (1, 2):  # two_vertical
            X = self._create_two_vertical_feature(shape)

        if self.feat_type == (3, 1):  # three_horizontal
            X = self._create_three_horizontal_feature(shape)

        if self.feat_type == (1, 3):  # three_vertical
            X = self._create_three_vertical_feature(shape)

        if self.feat_type == (2, 2):  # four_square
            X = self._create_four_square_feature(shape)

        if filename is None:
            cv2.imwrite("output/{}_feature.png".format(self.feat_type), X)

        else:
            cv2.imwrite("output/{}.png".format(filename), X)

        return X

    def evaluate(self, ii):
        """Evaluates a feature's score on a given integral image.

        Calculate the score of a feature defined by the self.feat_type.
        Using the integral image and the sum / subtraction of rectangles to
        obtain a feature's value. Add the feature's white area value and
        subtract the gray area.

        For example, on a feature of type (2, 1):
        score = sum of pixels in the white area - sum of pixels in the gray area

        Keep in mind you will need to use the rectangle sum / subtraction
        method and not numpy.sum(). This will make this process faster and
        will be useful in the ViolaJones algorithm.

        Args:
            ii (numpy.array): Integral Image.

        Returns:
            float: Score value.
        """

        x, y = self.position
        ht, wd = self.size
        if self.feat_type == (2, 1):
            white1 = ii[x + int(ht / 2) - 1, y + wd - 1] - ii[x - 1, y + wd - 1] - ii[x + int(ht / 2) - 1, y - 1] + ii[
                x - 1, y - 1]
            black = ii[x + int(ht) - 1, y + wd - 1] - ii[x + int(ht / 2) - 1, y + wd - 1] - ii[x + int(ht) - 1, y - 1] + \
                    ii[x + int(ht / 2) - 1, y - 1]
            return white1 - black

        if self.feat_type == (1, 2):
            white1 = ii[x + int(ht) - 1, y + int(wd / 2) - 1] - ii[x - 1, y + int(wd / 2) - 1] - ii[
                x + int(ht) - 1, y - 1] + ii[x - 1, y - 1]

            black = ii[x + int(ht) - 1, y + int(wd) - 1] - ii[x - 1, y + int(wd) - 1] - ii[
                x + int(ht) - 1, y + int(wd / 2) - 1] + ii[x - 1, y + int(wd / 2) - 1]

            return white1 - black

        if self.feat_type == (3, 1):
            white0 = ii[x + int(ht / 3) - 1, y + wd - 1] - ii[x - 1, y + wd - 1] - ii[x + int(ht / 3) - 1, y - 1] + ii[
                x - 1, y - 1]

            black = ii[x + int(ht / 3) + int(ht / 3) - 1, y + wd - 1] - ii[x + int(ht / 3) - 1, y + wd - 1] - ii[
                x + int(ht / 3) + int(ht / 3) - 1, y - 1] + ii[x + int(ht / 3) - 1, y - 1]

            white1 = ii[x + ht - 1, y + wd - 1] - ii[x + int(ht / 3) + int(ht / 3) - 1, y + wd - 1] - ii[
                x + ht - 1, y - 1] + ii[x + int(ht / 3) + int(ht / 3) - 1, y - 1]

            return white0 - black + white1

        if self.feat_type == (1, 3):
            white0 = ii[x + int(ht) - 1, y + int(wd / 3) - 1] - ii[x - 1, y + int(wd / 3) - 1] - ii[
                x + int(ht) - 1, y - 1] + ii[x - 1, y - 1]

            black = ii[x + int(ht) - 1, y + int(wd / 3) + int(wd / 3) - 1] - ii[
                x - 1, y + int(wd / 3) + int(wd / 3) - 1] - ii[x + int(ht) - 1, y + int(wd / 3) - 1] + ii[
                        x - 1, y + int(wd / 3) - 1]

            white1 = ii[x + ht - 1, y + int(wd) - 1] - ii[x - 1, y + int(wd) - 1] - ii[
                x + ht - 1, y + int(wd / 3) + int(wd / 3) - 1] + ii[x - 1, y + int(wd / 3) + int(wd / 3) - 1]

            return white0 - black + white1

        if self.feat_type == (2, 2):
            white1 = ii[x + int(ht / 2) - 1, y + wd - 1] - ii[x - 1, y + wd - 1] - ii[
                x + int(ht / 2) - 1, y + int(wd / 2) - 1] + ii[x - 1, y + int(wd / 2) - 1]

            white2 = ii[x + ht - 1, y + int(wd / 2) - 1] - ii[x + int(ht / 2) - 1, y + int(wd / 2) - 1] - ii[
                x + ht - 1, y - 1] + ii[x + int(ht / 2) - 1, y - 1]
            black = ii[x + int(ht / 2) - 1, y + int(wd / 2) - 1] - ii[x - 1, y + int(wd / 2) - 1] - ii[
                x + int(ht / 2) - 1, y - 1] + ii[x - 1, y - 1]

            black1 = ii[x + ht - 1, y + wd - 1] - ii[x + int(ht / 2) - 1, y + wd - 1] - ii[
                x + ht - 1, y + int(wd / 2) - 1] + ii[x + int(ht / 2) - 1, y + int(wd / 2) - 1]
            return white1 + white2 - black1 - black


def convert_images_to_integral_images(images):
    """Convert a list of grayscale images to integral images.

    Args:
        images (list): List of grayscale images (uint8 or float).

    Returns:
        (list): List of integral images.
    """
    integral_images = []
    for image in images:
        integral_image = np.zeros(shape=image.shape).astype(np.float64)
        for row in range(len(image)):
            for col in range(len(image[row])):
                sum = 0.0

                if row == 0 and col == 0:
                    sum = sum + image[row][col]
                elif row == 0:
                    sum = integral_image[row][col - 1] + image[row][col]
                elif col == 0:
                    sum = integral_image[row - 1][col] + image[row][col]
                else:
                    sum = integral_image[row][col - 1] + integral_image[row - 1][col] + image[row][col] - \
                          integral_image[row - 1][col - 1]

                integral_image[row][col] = sum
        integral_images.append(integral_image)
    return integral_images


class ViolaJones:
    """Viola Jones face detection method

    Args:
        pos (list): List of positive images.
        neg (list): List of negative images.
        integral_images (list): List of integral images.

    Attributes:
        haarFeatures (list): List of haarFeature objects.
        integralImages (list): List of integral images.
        classifiers (list): List of weak classifiers (VJ_Classifier).
        alphas (list): Alpha values, one for each weak classifier.
        posImages (list): List of positive images.
        negImages (list): List of negative images.
        labels (numpy.array): Positive and negative labels.
    """

    def __init__(self, pos, neg, integral_images):
        self.haarFeatures = []
        self.integralImages = integral_images
        self.classifiers = []
        self.alphas = []
        self.posImages = pos
        self.negImages = neg
        self.labels = np.hstack((np.ones(len(pos)), -1 * np.ones(len(neg))))

    def createHaarFeatures(self):
        # Let's take detector resolution of 24x24 like in the paper
        FeatureTypes = {"two_horizontal": (2, 1),
                        "two_vertical": (1, 2),
                        "three_horizontal": (3, 1),
                        "three_vertical": (1, 3),
                        "four_square": (2, 2)}

        haarFeatures = []
        for _, feat_type in FeatureTypes.items():
            for sizei in range(feat_type[0], 24 + 1, feat_type[0]):
                for sizej in range(feat_type[1], 24 + 1, feat_type[1]):
                    for posi in range(0, 24 - sizei + 1, 4):
                        for posj in range(0, 24 - sizej + 1, 4):
                            haarFeatures.append(
                                HaarFeature(feat_type, [posi, posj],
                                            [sizei - 1, sizej - 1]))
        self.haarFeatures = haarFeatures

    def train(self, num_classifiers):

        # Use this scores array to train a weak classifier using VJ_Classifier
        # in the for loop below.
        scores = np.zeros((len(self.integralImages), len(self.haarFeatures)))
        print(" -- compute all scores --")
        for i, im in enumerate(self.integralImages):
            scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        weights_pos = np.ones(len(self.posImages), dtype='float') * 1.0 / (
                2 * len(self.posImages))
        weights_neg = np.ones(len(self.negImages), dtype='float') * 1.0 / (
                2 * len(self.negImages))
        weights = np.hstack((weights_pos, weights_neg))

        print(" -- select classifiers --")
        for i in range(num_classifiers):
            # TODO: Complete the Viola Jones algorithm
            weights = weights / np.sum(weights)
            VJ = VJ_Classifier(scores, self.labels, weights)
            VJ.train()
            errors = [0 if VJ.predict(score) == label else 1 for score, label in zip(scores, self.labels)]
            self.classifiers.append(VJ)
            Bt = VJ.error / (1.0 - VJ.error)
            weights = [wt * np.power(Bt, 1.0 - error) for wt, error in zip(weights, errors)]
            alpat = np.log(1.0 / Bt)
            self.alphas.append(alpat)

    def predict(self, images):
        """Return predictions for a given list of images.

        Args:
            images (list of element of type numpy.array): list of images (observations).

        Returns:
            list: Predictions, one for each element in images.
        """

        ii = convert_images_to_integral_images(images)

        scores = np.zeros((len(ii), len(self.haarFeatures)))

        # Populate the score location for each classifier 'clf' in
        # self.classifiers.

        # Obtain the Haar feature id from clf.feature

        # Use this id to select the respective feature object from
        # self.haarFeatures

        # Add the score value to score[x, feature id] calling the feature's
        # evaluate function. 'x' is each image in 'ii'

        prediction_alphas = np.zeros((len(ii), len(self.classifiers)))

        for i in range(len(self.classifiers)):
            feature_id = self.classifiers[i].feature
            for j in range(len(ii)):
                scores[j, feature_id] = self.haarFeatures[feature_id].evaluate(ii[j])

        for i in range(len(self.classifiers)):
            prediction_alphas[:, i] = [self.classifiers[i].predict(scores[j]) * self.alphas[i] for j in range(len(ii))]

        result = [1 if np.sum(x) >= sum(self.alphas)/2.0 else -1 for x in prediction_alphas]

        return result

    def faceDetection(self, image, filename):
        """Scans for faces in a given image.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """
        image_gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Wind", image_gray_scale)
        # cv2.waitKey(0)
        size = 24
        sub_images = {}
        for i in range(image_gray_scale.shape[0]-size):
            for j in range(image_gray_scale.shape[1] - size):
                sub_images[(i, j)] = image_gray_scale[i:i+size, j:j+size]

        results = []
        for key in sub_images:
            # cv2.imshow("win", sub_images[key])
            # cv2.waitKey(5)
            res = self.predict([sub_images[key]])
            print(res)
            if res[0] == 1:
                results.append(key)
        print("Results - {}", results[int(len(results)/2)])
        x,y = results[int(len(results) / 2)]
        output_image = cv2.rectangle(image, (y,x), (y+size, x+size), (0,255,0), 2)
        cv2.imwrite("output/{}.png".format(filename), output_image)

