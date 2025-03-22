import json
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .utils import predict_car_price, predict_car_price_poly, transform, predict_car_category

# Median values for default fields
MEDIAN_VALUES = {
    "year": 2015,
    "engine": 1248,
    "max_power": 82.4
}

@csrf_exempt
def predict_view(request):
    if request.method == 'POST':
        try:
            # Parse JSON data
            body = json.loads(request.body)
            features = body.get('features', [])
            use_poly = body.get('poly', False)
            
            year = features[0] if features[0] else MEDIAN_VALUES["year"]
            transmission = features[1]  # Transmission is always required
            engine = features[2] if features[2] else MEDIAN_VALUES["engine"]
            max_power = features[3] if features[3] else MEDIAN_VALUES["max_power"]

            # Validate the inputs
            if not (1994 <= int(year) <= 2020):
                return JsonResponse({"error": "Year must be between 1994 and 2020."}, status=400)
            if not (600 <= int(engine) <= 3700):
                return JsonResponse({"error": "Engine size must be between 600 and 3700 CC."}, status=400)
            if not (30 <= float(max_power) <= 400):
                return JsonResponse({"error": "Max power must be between 30 and 400 HP."}, status=400)

            # Prepare prediction
            filled_features = [year, transmission, engine, max_power]
            
            # Transform features
            transformed_features = transform(filled_features)

            if use_poly:
                predicted_price = predict_car_price_poly(transformed_features)
            else:
                predicted_price = predict_car_price(transformed_features)

            return JsonResponse({'predicted_price': predicted_price}, status=200)
        
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON payload.'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'message': 'Send POST request with features[] data.'}, status=405)

@csrf_exempt
def predict_classification_view(request):
    """Handles classification-based car category predictions (MLflow Model)."""
    if request.method == 'POST':
        try:
            body = json.loads(request.body)
            features = body.get('features', [])

            if not features or len(features) != 4:
                return JsonResponse({'error': 'Exactly 4 features are required.'}, status=400)

            # Extract & validate input features
            year = features[0] if features[0] else MEDIAN_VALUES["year"]
            transmission = features[1]  # Required field
            engine = features[2] if features[2] else MEDIAN_VALUES["engine"]
            max_power = features[3] if features[3] else MEDIAN_VALUES["max_power"]

            if not (1994 <= int(year) <= 2020):
                return JsonResponse({"error": "Year must be between 1994 and 2020."}, status=400)
            if not (600 <= int(engine) <= 3700):
                return JsonResponse({"error": "Engine size must be between 600 and 3700 CC."}, status=400)
            if not (30 <= float(max_power) <= 400):
                return JsonResponse({"error": "Max power must be between 30 and 400 HP."}, status=400)

            # ✅ Ensure the input shape is (1, 4)
            filled_features = [year, transmission, engine, max_power]  # Shape (1, 4)

            # ✅ Ensure correct shape before passing to transform()
            transformed_features = transform(filled_features)  # Shape (1, 4)

            # ✅ Add intercept column (bias term) correctly
            intercept = np.ones((transformed_features.shape[0], 1))  # Shape (1, 1)
            transformed_features = np.concatenate((intercept, transformed_features), axis=1)  # Shape (1, 5)

            # Get predicted category
            predicted_category = predict_car_category(transformed_features)

            return JsonResponse({'predicted_category': predicted_category}, status=200)
        
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON payload.'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

    return JsonResponse({'message': 'Send POST request with features[] data.'}, status=405)

def home_view(request):
    return render(request, 'prediction.html')