from PIL import Image
import numpy as np
# classes which represent configuration parameters in the code
from ..config import PathField, BoolField
# data type which was generated during annotation conversion
from ..representation import ClassificationAnnotation
from ..utils import read_csv

from .format_converter import BaseFormatConverter, ConverterReturn


class FERPlusFormatConverter(BaseFormatConverter):
    """
    FER+ dataset converter. All annotation converters should be derived from BaseFormatConverter class.
    Annotation data for conversion can be found here https://www.kaggle.com/deadskull7/fer2013
    """

    # register name for this converter
    # this name will be used for converter class look up
    __provider__ = 'fer_plus'
    # specify a hint about generated data type
    annotation_types = (ClassificationAnnotation, )
    label_to_model_label = {0: 4, 1:5, 2:6, 3:1, 4:3, 5:2, 6:0, 7:-1}

    @classmethod
    def parameters(cls):
        """
        describe config parsing template for this converter
        :return: dictionary, where config fields used as keys and helpers for config parsing as values.
        """
        # get basic parameters from parent class
        configuration_parameters = super().parameters()
        # update them with new
        configuration_parameters.update({
            'annotation_file': PathField(description="Path to csv file which contain dataset."),
            'convert_images': BoolField(
                optional=True,
                default=True,
                description="Allows to convert images from pickle file to user specified directory."
            ),
            'converted_images_dir': PathField(
                optional=True, is_directory=True, check_exists=False, description="Path to converted images location."
            )
        })

        return configuration_parameters

    def configure(self):
        """
        This method is responsible for obtaining the necessary parameters
        for converting from the command line or config.
        """
        self.csv_file = self.get_value_from_config('annotation_file')
        self.converted_images_dir = self.get_value_from_config('converted_images_dir')
        self.convert_images = self.get_value_from_config('convert_images')
        if self.convert_images and not self.converted_images_dir:
            self.converted_images_dir = self.csv_file.parent / 'converted_images'
        if not self.converted_images_dir.exists():
            self.converted_images_dir.mkdir(parents=True)

        if self.convert_images and Image is None:
            raise ValueError(
                "conversion mnist images requires Pillow installation, please install it before usage"
            )

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        """
        This method is executed automatically when convert.py is started.
        All arguments are automatically got from command line arguments or config file in method configure

        Returns:
            annotations: list of annotation representation objects.
            meta: dictionary with additional dataset level metadata.
            content errors: service field for errors handling
        """
        annotations = []
        # read original dataset annotation
        annotation_table = read_csv(self.csv_file)
        # process object by object
        for index, annotation in enumerate(annotation_table):
            # ignore data not from testing subset
            if annotation['Usage'] != 'PublicTest':
                continue
            # identifier is unique name of data in the dataset. For images usually file name used
            identifier = '{}.png'.format(index)
            # getting label
            label = int(annotation['emotion'])
            remaped_label = self.label_to_model_label[label]
            # since our annotation contains pixels intensity inside the table,
            # we need to get images from it for more convenient usage.
            # convert images once, we can turn off this flag in the config and use pregenerated images
            if self.convert_images:
                pixels_array = [int(y) for y in annotation['pixels'].split()]
                pixels = np.array(pixels_array).reshape(48, 48)
                image = Image.fromarray(pixels.astype(np.uint8))
                image = image.convert("L")
                image.save(str(self.converted_images_dir / identifier))
            # create new instance of annotation representation
            # different representations can have different set of parameters required for metric calculation
            # for ClassificationAnnotation, identifier and label used.
            
            annotations.append(ClassificationAnnotation(identifier, remaped_label))

        # metadata contains specific info about dataset which can help during evaluation
        # (e.g. mapping of labels, has background label in the dataset or not)
        # for some task where addition info is not required it can be stayed empty of None
        meta = self.get_meta()

        # finally, this method should return the named tuple with fields annotations, meta and content errors
        return ConverterReturn(annotations, meta, None)


    def get_meta(self):
        # use original lables from dataset
        emotion_table = {'neutral': 0, 'happiness': 1, 'surprise': 2, 'sadness': 3, 'anger': 4, 'disgust': 5, 'fear': 6,
                         'contempt': 7}
        # inside Accuracy Checker we use class_id as label, so label map should be represented as class_id: class_name
        label_map = {v: k  for v, k in emotion_table.items()}

        # dataset meta should be represented like dictionary, label_map key used for storing label mapping
        return {'label_map': label_map}
