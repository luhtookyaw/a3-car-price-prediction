"""
URL configuration for car_price_app project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.shortcuts import render
from django.urls import path, include

# Define a simple view for the root URL
def home(request):
    return render(request, "index.html")

# Define a new view for the polynomial regression prediction page
def poly_prediction_view(request):
    return render(request, "poly_prediction.html")

def classfication_view(request):
    return render(request, "classification.html")

urlpatterns = [
    path('admin/', admin.site.urls),
    path('prediction/', include('prediction.urls')),
    path('poly_prediction/', poly_prediction_view, name='poly_prediction'),
    path('classification/', classfication_view, name='classifcation'),
    path('', home)
]
