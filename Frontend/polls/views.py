from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import VideoUploadForm
from .models import Video
import os
from django.conf import settings
# Create your views here.
def index(request):
    return render(request, "polls/index.html")

def features(request):
    return render(request, "polls/features.html")


def upload_video(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            # return redirect('video_list')
    else:
        form = VideoUploadForm()
    return render(request, 'upload_video.html', {'form': form})
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

def video_list(request):
    videos = Video.objects.all()
    for video in videos:
        print(video.id)
        print(video.title)
        print(video.file.url)
        print(video.upload_date)
    return render(request, 'video_list.html', {'videos': videos})

@csrf_exempt
def features_with_Id(request, video_id):
    try:
        video_file = Video.objects.get(pk=video_id)
        # video_file = video.file.url
        # with open(video_file.path, 'rb') as video_data:
        #     response = HttpResponse(video_data.read(), content_type='video/mp4')
        #     response['Content-Disposition'] = f'inline; filename="{video_file.name}"'
        #     return response
        return render(request, "polls/features.html", {'video_file':video_file})
    except Video.DoesNotExist:
        return HttpResponse("Video không tồn tại", status=404)
    


@csrf_exempt  # Đây là decorator để tắt kiểm tra CSRF token, hãy sử dụng nó cẩn thận và chỉ cho những view có cần
def delete_video(request, video_id):
    response = {"success": False}
    try:
        video = Video.objects.get(id=video_id)
        url_video = '..'+str(video.file.url) 
        print(url_video)
        file_path = os.path.join(settings.MEDIA_ROOT, url_video)
        # Xóa tệp tin video từ thư mục media
        if os.path.exists(file_path):
            os.remove(file_path)
        video.delete()
        response["success"] = True
    except Video.DoesNotExist:
        pass

    return JsonResponse(response)

@csrf_exempt  # Đây là decorator để tắt kiểm tra CSRF token, hãy sử dụng nó cẩn thận và chỉ cho những view có cần
def choose_video(request, video_id):
    response = {"success": False}
    try:
        video = Video.objects.get(id=video_id)
        url_video = '..'+str(video.file.url) 
        print(url_video)
        file_path = os.path.join(settings.MEDIA_ROOT, url_video)
        response["success"] = True
        return render(request, "polls/features.html")
    except Video.DoesNotExist:
        pass
    return JsonResponse(response)