import numpy as np
from django.test import TestCase
from prediction.utils import predict_car_category, transform

class ClassificationModelTest(TestCase):

    def setUp(self):
        # Set up valid input with expected number of features (4)
        self.valid_input = [2018, 1, 1500, 100]  # (1, 4)
        self.transformed = transform(self.valid_input)

    def test_model_accepts_expected_input(self):
        """Test that the model accepts properly transformed input with intercept."""
        # Add intercept (shape should be (1, 5))
        intercept = np.ones((self.transformed.shape[0], 1))
        input_with_intercept = np.concatenate((intercept, self.transformed), axis=1)

        try:
            prediction = predict_car_category(input_with_intercept)
            self.assertIsInstance(prediction, int)
        except Exception as e:
            self.fail(f"Model failed to accept valid input: {e}")

    def test_model_output_shape(self):
        """Test that the model output is a single class label (not a list, array, etc.)"""
        intercept = np.ones((self.transformed.shape[0], 1))
        input_with_intercept = np.concatenate((intercept, self.transformed), axis=1)

        prediction = predict_car_category(input_with_intercept)
        self.assertTrue(isinstance(prediction, int), "Output should be a single integer class label")
