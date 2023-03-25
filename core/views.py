from django.shortcuts import render, redirect
from .detect import blue_detection
from .form import ImageUploadForm
from .models import Image
import os

# Create your views here.

def detect(request):
    success = False
    data = None
    image = None
    form = ImageUploadForm()

    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)

        if form.is_valid():
            image = form.cleaned_data["item"]
            new_image = Image.objects.create(item=image)
            new_image.save()
            print(image)
            print(new_image)
            print("image was saved successfully")
            
            if Image.objects.filter(item=image).exists():
                image_instance = Image.objects.get(item=image)
                data = blue_detection(image_instance)

                
    context = {
        "data": data,
        "form": form,
        'image': image
    }

    return render(request, "detect.html", context)