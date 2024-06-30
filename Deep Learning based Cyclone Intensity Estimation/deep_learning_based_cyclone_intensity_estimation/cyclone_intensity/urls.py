from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("get_data_from_user", views.get_data_from_user, name="get_data_from_user"),
    path('predict_intensity', views.predict_intensity, name="predict_intensity"),
]