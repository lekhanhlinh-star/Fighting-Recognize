from django.db import models

# Create your models here.
class Video(models.Model):
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='videos/')
    upload_date = models.DateTimeField(auto_now_add=True)