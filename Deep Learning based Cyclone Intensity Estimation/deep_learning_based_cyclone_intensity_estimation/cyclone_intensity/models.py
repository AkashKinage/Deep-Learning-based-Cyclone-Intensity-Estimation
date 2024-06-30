from django.db import models
from django.utils import timezone
import os
import uuid

# def upload_to_input_image(instance, filename):
#     ext = filename.split('.')[-1]
#     filename = f'input_image.{ext}'
#     return os.path.join('uploads/', filename)

class UploadedData(models.Model):
    upload_datetime = models.DateTimeField(default=timezone.now)
    # coordinates = models.CharField(max_length=255)  # Assuming coordinates as a string
    # intensity = models.CharField(max_length=255)
    # image = models.ImageField(upload_to=upload_to_input_image, default='uploads/')  # 'uploads/' is the subdirectory within 'static/'
    image = models.ImageField(upload_to='uploads/')
    img_date = models.DateField()
    img_time = models.TimeField()

    def __str__(self):
        return f"{self.img_date}"
    
    def save(self, *args, **kwargs):
        # Set the image name to 'input_image'
        self.image.name = 'input_image.jpg'  # Change the extension accordingly

        # Check if an image with the name 'input_image' already exists
        existing_image_path = os.path.join('uploads/', 'input_image.jpg')  # Change the extension accordingly
        if os.path.isfile(existing_image_path):
            # Delete the existing image
            os.remove(existing_image_path)

        super(UploadedData, self).save(*args, **kwargs)