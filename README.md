# FaceFun Express 🎭

## Overview  
**FaceFun Express** is an interactive project that combines **Facial Expression Recognition (FER)** technology with fun filters and masks, making user experiences more engaging. The system detects emotions in real-time using **machine learning** and **computer vision**, and applies creative filters based on the user's facial expressions.

## Features  
- Real-time facial expression detection (e.g., Happy, Sad, Angry, Surprised).
- Dynamic application of filters and masks based on detected emotions.
- Built with **Convolutional Neural Networks (CNNs)** for accurate emotion recognition.
- Supports **facial landmark detection** for precise mask/overlay placement.
- Handles occlusions and lighting variations using **data augmentation**.
- Easy integration for social media, gaming, and other interactive platforms.

## Technologies Used  
- **Python**  
- **OpenCV** – for facial detection and real-time video processing.  
- **TensorFlow/PyTorch** – for deep learning model training.  
- **Convolutional Neural Networks (CNNs)** – for emotion recognition.
- **Dataset** - https://www.kaggle.com/datasets/msambare/fer2013

## Installation  

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/FaceFunExpress.git
   cd FaceFunExpress
   ```

2. **Install dependencies**  
   Use `pip` to install required libraries:

3. **Run the project**  
   ```bash
   python main.py
   ```
   ```bash
   python fer.py
   ```

4. **Apply filters**  
   Once the app is running, it will detect your face and apply filters based on the recognized emotion in real-time.

## Project Structure  
```plaintext
├── main.py                # Train the models using this file over dataset
├── app.py                 # Main application file
└── README.md              # Project documentation (this file)
```

## Usage

- Run the project, and the system will use your webcam to detect your face.
- Based on your facial expression, a filter or mask will be applied in real-time.

## How It Works

1. **Facial Detection:** The system uses **OpenCV** to detect faces in a live video stream.
2. **Facial Expression Recognition:** A **CNN** model is used to predict the emotion based on the detected face.
3. **Applying Filters/Masks:** Filters are applied on the face using facial landmarks, aligning them perfectly with facial features.

## Future Improvements  
- Adding more filters and emotion categories.
- Improving accuracy for challenging environments (e.g., low light or partial occlusions).
- Optimizing performance for mobile platforms.

## Contact  
For any questions or suggestions, feel free to reach out:  
**Jeet Soni** – jeetsoni1005@gmail.com
