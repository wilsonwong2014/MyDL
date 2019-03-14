"""Template for special prize model prediction evaluation."""


class Model:
    """Represent a model."""

    def __init__(self, model_path):
        """Set up model.

        Args:
            model_path (str): Path to model.

        """
        self.path = model_path

    def predict(self, image_data):
        """Predict labels of image_data using model.

        Args:
            image_data (numpy array): Image data array.

        Returns:
            The predicted set of labels.

        """
        labels = set()
        return labels
