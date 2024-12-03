from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth import authenticate, login as auth_login
import cv2
import numpy as np
import joblib
import pandas as pd


def home(request):
    return render(request, 'home.html')


def login(request):
    if request.method == "POST":
        un = request.POST['username']
        pw = request.POST['password']
        user = authenticate(request, username=un, password=pw)
        if user is not None:
            auth_login(request, user)
            return redirect('/profile')
        else:
            msg = 'Invalid Username/Password'
            form = AuthenticationForm(request.POST)
            return render(request, 'login.html', {'form': form, 'msg': msg})
    else:
        form = AuthenticationForm()
        return render(request, 'login.html', {'form': form})


def signup(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            un = form.cleaned_data.get('username')
            pw = form.cleaned_data.get('password1')
            authenticate(username=un, password=pw)
            return redirect('/login')
    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})


# Load the model
model = joblib.load('logistic_regression_model.joblib')

# Read the disease information from the Excel file
try:
    df = pd.read_excel('disease_info.xlsx')
    df['Disease Name'] = df['Disease'].str.lower()
    disease_info_dict = pd.Series(df['Information'].values, index=df['Disease Name']).to_dict()
except FileNotFoundError:
    disease_info_dict = {}

def profile(request):
    if request.method == "POST":
        if request.FILES.get('uploadImage'):
            # Handle file upload
            uploaded_image = request.FILES['uploadImage']
            fs = FileSystemStorage()
            filename = fs.save(uploaded_image.name, uploaded_image)
            img_url = fs.url(filename)

            # Get the path of the uploaded image
            img_path = fs.path(filename)

            # Start implementing the image processing
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (64, 64))  # Resize the image to the model's input size
            img = img.flatten()  # Flatten the image
            img = np.expand_dims(img, axis=0)  # Expand the dimensions for the model

            # Predict the disease using the model
            predict = model.predict(img)[0]  # Get the prediction (most likely a string or integer)

            # Define the list of possible skin diseases (if the model outputs integers)
            skin_disease_names = [
                'Cellulitis', 'Impetigo', 'Athlete Foot', 'Nail Fungus', 'Ringworm',
                'Cutaneous Larva Migrans', 'Chickenpox', 'Shingles'
            ]

            # If the model returns an integer, map it to the disease name
            if isinstance(predict, int):
                result1 = skin_disease_names[predict]  # Map integer to disease name
            else:
                result1 = predict  # Otherwise, use the string output directly

            # Remove "VI-", "FU-", "BA-", or "PA-" prefixes if they exist
            if result1.startswith("VI-"):
                result1 = result1[3:]
            elif result1.startswith("FU-"):
                result1 = result1[3:]
            elif result1.startswith("BA-"):
                result1 = result1[3:]
            elif result1.startswith("PA-"):
                result1 = result1[3:]

            result1 = result1.replace("-", " ").title()

            result2 = "Diagnosed with " + result1

            # Print the dictionary and predicted result for debugging
            print("Disease Info Dictionary:", disease_info_dict)
            print("Predicted Result:", result1)

            # Fetch additional disease information from the dictionary
            disease_info = disease_info_dict.get(result1.lower(), "No additional information available.")

            # Return the result to the profile page with the image, diagnosis, and disease info
            return render(request, 'profile.html', {
                'img': img_url,
                'obj1': result1,
                'obj2': result2,
                'disease_info': disease_info  # Pass the disease info to the template
            })

    # If no image is uploaded, simply render the page
    return render(request, 'profile.html')
