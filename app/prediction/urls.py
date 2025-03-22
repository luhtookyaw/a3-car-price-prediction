from django.urls import path
from . import views

urlpatterns = [
  path('', views.home_view, name="home"),
  path('predict/', views.predict_view, name='predict'),
  path('predict_classification/', views.predict_classification_view, name='predict_classification'),  # Classification API
]
