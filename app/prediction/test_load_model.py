from django.test import TestCase
from prediction.utils import load_latest_mlflow_model  # Adjust import as needed

class LoadModelTest(TestCase):

    def test_model_loads(self):
        model = load_latest_mlflow_model()
        self.assertIsNotNone(model)
        from mlflow.pyfunc import PyFuncModel
        self.assertIsInstance(model, PyFuncModel)