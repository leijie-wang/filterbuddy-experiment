from django.urls import path
from wordfilter import views


urlpatterns = [
    path('main', views.main),
]
