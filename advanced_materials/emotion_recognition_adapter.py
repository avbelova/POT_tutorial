import numpy as np
# base class for all adapters
from ..adapters import Adapter
# output format
from ..representation import ClassificationPrediction


class EmotionRecognitionAdapter(Adapter):
    """
    Class for converting output of emotion recognition model to ClassificationPrediction representation
    """
    # new adapter name in the config
    __provider__ = 'emotion_recognition'
    # like other components adapter can have parameters for configuration, but in our case they are not used
    # we can stay default implementation of these parameters

    def process(self, raw, identifiers=None, frame_meta=None):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
            frame_meta: list of meta information about each frame
        Returns:
            list of ClassificationPrediction objects
        """
        def softmax(x):
            """Compute softmax values (probabilities from 0 to 1) for each possible label."""
            x = x.reshape(-1)
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)
        # in some cases output can be return as list of dictionaries, while dictionary expected.
        # We need handle it inside extract prediction
        prediction = self._extract_predictions(raw, frame_meta)[self.output_blob]

        # define container to store batch of predictions as independent entities
        result = []
        # go throw batch dementio and extract results for specific image
        for identifier, output in zip(identifiers, prediction):
            # depends from task output representation can be different and has own parameters
            # for classification, identifier and class probabilities are required
            single_prediction = ClassificationPrediction(identifier, softmax(output).flatten())
            result.append(single_prediction)

        # return list of prediction representations
        return result
