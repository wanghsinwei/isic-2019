from Augmentor.Operations import *
from math import floor, ceil
import random

class CropPercentageRange(Operation):
    """
    This class is used to crop images by a percentage of their area.
    """
    def __init__(self, probability, min_percentage_area, max_percentage_area, centre):
        """
        As well as the always required :attr:`probability` parameter, the
        constructor requires a :attr:`percentage_area` to control the area
        of the image to crop in terms of its percentage of the original image,
        and a :attr:`centre` parameter toggle whether a random area or the
        centre of the images should be cropped.
        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param min_percentage_area: The lower bound random percentage area of the original image to crop.
        :param min_percentage_area: The upper bound random percentage area of the original image to crop.
        :param centre: Whether to crop from the centre of the image or
         crop a random location within the image.
        :type probability: Float
        :type min_percentage_area: Float
        :type max_percentage_area: Float
        :type centre: Boolean
        """
        Operation.__init__(self, probability)
        self.min_percentage_area = min_percentage_area
        self.max_percentage_area = max_percentage_area
        self.centre = centre

    def perform_operation(self, images):
        """
        Crop the passed :attr:`images` by percentage area, returning the crop as an
        image.
        :param images: The image(s) to crop an area from.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

        r_percentage_area = round(random.uniform(self.min_percentage_area, self.max_percentage_area), 2)

        # The images must be of identical size, which is checked by Pipeline.ground_truth().
        w, h = images[0].size

        w_new = int(floor(w * r_percentage_area))  # TODO: Floor might return 0, so we need to check this.
        h_new = int(floor(h * r_percentage_area))

        left_shift = random.randint(0, int((w - w_new)))
        down_shift = random.randint(0, int((h - h_new)))

        def do(image):
            if self.centre:
                return image.crop(((w/2)-(w_new/2), (h/2)-(h_new/2), (w/2)+(w_new/2), (h/2)+(h_new/2)))
            else:
                return image.crop((left_shift, down_shift, w_new + left_shift, h_new + down_shift))

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images