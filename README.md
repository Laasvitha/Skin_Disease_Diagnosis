# Skin Disease Diagnosis Project

## Overview

This project aims to classify skin diseases based on images using machine learning techniques. The backend is built using **Django**, and the model for image classification is based on **Logistic Regression**, though alternative approaches like **Convolutional Neural Networks (CNNs)** can be explored for better accuracy. The project involves preprocessing images, training a machine learning model, and integrating the model into the Django application to classify skin diseases in uploaded images.

## Project Structure

The project is organized into the following files:

- **Django Setup:**
  - `urls.py`: Defines the URLs and routes for the project.
  - `views.py`: Contains the views for rendering pages like home, login, signup, and profile.
  
- **Machine Learning Setup:**
  - `model.py`: Contains functions to load and preprocess images, train the Logistic Regression model, evaluate it, and save the trained model.

- **Model Deployment:**
  - The machine learning model (Logistic Regression) is trained on images from the `train_set` and `test_set` directories and saved using `joblib`.

## Requirements

### Python Libraries

- Django
- OpenCV (`cv2`)
- NumPy
- scikit-learn
- joblib

You can install the necessary libraries using:

```bash
pip install django opencv-python numpy scikit-learn joblib
```

## Setup Instructions

### Step 1: Clone the Repository

Clone the repository to your local machine:

```bash
git clone <repository-url>
cd <project-directory>
```

### Step 2: Django Setup

1. Ensure that you have Django installed in your environment.

2. Apply migrations:

   ```bash
   python manage.py migrate
   ```

3. Run the development server:

   ```bash
   python manage.py runserver
   ```

4. Access the app at `http://127.0.0.1:8000/`.

### Step 3: Machine Learning Model

#### Data Preparation

The project uses a dataset of skin disease images, you can use your own dataset:

The images are read from the directories and resized to 64x64 pixels.

#### Model Training

1. The images are loaded and resized using OpenCV.
2. The images are then flattened into 1D arrays and normalized to a range between 0 and 1.
3. **StandardScaler** is used to scale the pixel values before training.
4. A **Logistic Regression** model is trained on the data.
5. The trained model is saved using **joblib** to `logistic_regression_model.joblib`.

To retrain the model or experiment with different algorithms, you can modify the code in `model.py`.

### Step 4: Machine Learning Model Evaluation

After training, the model is evaluated using a classification report and accuracy score. You can print more detailed metrics, such as confusion matrices, to further assess the model's performance.

### Step 5: Model Integration with Django

The trained model can be integrated into the Django views to classify uploaded skin images. After receiving an image via the profile page, the model will predict the skin disease type and display the result.

## File Breakdown

### `urls.py`

Defines the routes for different views in the application:

```python
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('login/', views.login, name='login'),
    path('signup/', views.signup, name='signup'),
    path('profile/', views.profile, name='profile'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

### `model.py`

This file contains all the logic for loading images, preprocessing them, training the Logistic Regression model, and saving it.

Key components:
- `load_images_and_labels(data_dir)`: Loads and preprocesses images from the specified directory.
- `train_model()`: Trains a Logistic Regression model on the images and saves it using `joblib`.

### `views.py`

Handles the views for rendering pages:
- **Home**: Renders the homepage of the app.
- **Login**: User login page.
- **Signup**: User signup page.
- **Profile**: Displays the uploaded image and diagnosis result.

### `settings.py` (Snippet for Static Files)

Ensure that your `MEDIA_URL` and `MEDIA_ROOT` are configured to handle user-uploaded images:

```python
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
```
## Conclusion

This project demonstrates the integration of machine learning with Django for image classification tasks. By following the setup instructions, you can get the project up and running on your local machine and further explore improvements to both the model and the web application.

---
