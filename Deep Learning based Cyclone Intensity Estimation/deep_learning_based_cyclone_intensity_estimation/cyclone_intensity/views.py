from django.shortcuts import render, redirect
from .models import UploadedData
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import os 
from django.conf import settings

def index(request):
    return render(request, "index.html")

def get_data_from_user(request):
    if request.method == 'POST':
        img_date = request.POST.get("image_date")
        img_time = request.POST.get("image_time")
        # latitude = request.POST.get("latitude")
        # longitude = request.POST.get("longitude")

        # coordinates = "(" + str(latitude) + "," + str(longitude) + ")"

        # get image from html form
        imagefile = request.FILES['imagefile']

        uploaded_data = UploadedData(
            # coordinates = coordinates,
            image = imagefile,
            img_date = img_date,
            img_time = img_time
        )

        uploaded_data.save()

        return redirect(predict_intensity)
    return render(request, "index.html")

def standardize_data1(test_images):
    test_images[test_images < 0] = 0
    st_dev = np.std(test_images)
    mean = np.mean(test_images)
    test_images = np.divide(np.subtract(test_images, mean), st_dev)
    return test_images

def category_of(wind_speed):
    if wind_speed <= 33:
        return 'Tropical Depression'
    elif wind_speed >= 34 and wind_speed <= 73:
        return 'Tropical Storm'
    elif wind_speed >= 74 and wind_speed <= 95:
        return 'Category 1: Tropical Cyclone'
    elif wind_speed >= 96 and wind_speed <= 110:
        return 'Category 2: Tropical Cyclone'
    elif wind_speed >= 111 and wind_speed <= 130:
        return 'Category 3: Severe Tropical Cyclone'
    elif wind_speed >= 131 and wind_speed <= 155:
        return 'Category 4: Severe Tropical Cyclone'
    elif wind_speed > 155:
        return 'Category 5: Severe Tropical Cyclone'

def predict_intensity(request):
    model = load_model('Combined_Model.h5')
    # Assuming the 'uploads' folder is in the same level as manage.py
    # base_directory = os.path.dirname(os.path.abspath(__file__))
    uploads_path = os.path.join(settings.BASE_DIR, 'uploads')
    
    # Construct the path for the 'input_image' dynamically  
    input_image_path = os.path.join(uploads_path, 'input_image.jpg')
    # print('Path: '+input_image_path)
    # print(type(input_image_path))
    input_image = image.load_img(input_image_path, color_mode='grayscale', target_size=(50, 50))
    input_image = image.img_to_array(input_image)
    # print(input_image.shape)
    input_image = np.expand_dims(input_image, axis=0)
    # print(input_image.shape)
    input_image = standardize_data1(input_image)

    prediction = model.predict(input_image)
    prediction = prediction[0][0]

    classification = category_of(int(prediction))

    params = {'prediction':prediction, 'classification': classification}

    return render(request, 'index.html', params)


