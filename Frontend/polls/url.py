from django.urls import path
from . import views 
urlpatterns = [
    path("", views.features,name='index'),
    path("features/", views.features,name='features'),
    path("features/<int:video_id>/", views.features_with_Id,name='features_full'),
    path('upload/', views.upload_video, name='upload_video'),
    path('video_list/upload/', views.upload_video, name='upload_video'),
    path('video_list/', views.video_list, name='video_list'),
    path('delete_video/<int:video_id>/', views.delete_video, name='delete_video'),
    path('choose_video/<int:video_id>/', views.choose_video, name='choose_video'),
]
