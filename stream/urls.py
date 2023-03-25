from django.urls import path
from . import detect_faces, football_movement

urlpatterns = [
    path('face', detect_faces.video_feed, name='video_feed'),
    path('ball', football_movement.video_feed, name='video_feed'),
]
