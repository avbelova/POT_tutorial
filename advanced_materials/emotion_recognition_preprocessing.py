from PIL import Image
import numpy as np

# base class for all preprocessors is Preprocessor
from ..preprocessor import Preprocessor
# helpers for configuration parsing
from ..config import NumberField


class EmotionRecognitionResize(Preprocessor):
    # name of preprocessor for configuration
    __provider__ = 'emotion_recognition_preprocessing'

    # definition of important configuration parameters
    # for image resizing we need to know target size
    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'size': NumberField(
                value_type=int, optional=False, min_value=1, description="Destination sizes for both dimensions."
            ),
        })

        return parameters

    def configure(self):
        # getting parameters from config
        self.size = self.get_value_from_config('size')

    def process(self, image, annotation_meta=None):
        """
        Preprocessor realization function, which will be called for each image in the input dataset
        :param image: DataRepresentation entry which include read image and related metadata for it.
        :param annotation_meta: Dictionary with info from  annotation.
                                optional, used in case when we need to use or update some info about image
        :return: DataRepresentation with updated image
        """
        # image dasta stored inside DataRepresentation in data field
        data = image.data
        # internally we work with numpy arrays, so we need to convert it to pillow image object for making resize
        resized_data = Image.fromarray(data).resize((self.size, self.size), Image.ANTIALIAS)
        # return back data to numpy array
        data = np.array(resized_data)
        # expand dims for gray scale image
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=-1)
        image.data = data
        # return updated DataRepresentation
        return image
