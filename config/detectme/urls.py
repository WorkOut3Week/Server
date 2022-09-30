# from django.urls import url, include
from django.urls import path, include

from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path("detectme", views.detectme, name="detectme"),
]
