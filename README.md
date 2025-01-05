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
- **Windows GUI** using **Tkinter** and a **web interface** using **Flask**.

## Technologies Used  
- **Python**  
- **OpenCV** – for facial detection and real-time video processing.  
- **TensorFlow/PyTorch** – for deep learning model training.  
- **Convolutional Neural Networks (CNNs)** – for emotion recognition.  
- **Tkinter** – for the Windows GUI application.  
- **Flask** – for the web interface.
- **Dataset** – [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)

## Installation  

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Jeet-Soni-1005/FaceFunExpress.git
   cd FaceFunExpress
   ```

2. **Install dependencies**  
   Use `pip` to install required libraries:

3. **Train the model**  
   Run the following command to train the model using the FER-2013 dataset and save it:
   ```bash
   python main.py
   ```

4. **Run the application**  
   For the Windows GUI application:
   ```bash
   python Facefun_app_tkinter.py
   ```
   For the web interface:
   ```bash
   python Facefun_webapp_flask.py
   ```

## Project Structure  
```plaintext
├── main.py                # Train the model using the dataset
├── Facefun_app_tkinter.py # Windows GUI application file
├── Facefun_webapp_flask.py# Web interface application file
├── README.md              # Project documentation (this file)
```

## Usage

1. **Train the Model**: Ensure you run `main.py` first to train the model and save it.
2. **Choose an Interface**:
   - For a Windows GUI application, run `Facefun_app_tkinter.py`.
   - For a web interface, run `Facefun_webapp_flask.py` and access the app in your browser.
3. The system will use your webcam to detect your face.
4. Based on your facial expression, a filter or mask will be applied in real-time.

## How It Works

1. **Facial Detection:** The system uses **OpenCV** to detect faces in a live video stream.
2. **Facial Expression Recognition:** A **CNN** model is used to predict the emotion based on the detected face.
3. **Applying Filters/Masks:** Filters are applied on the face using facial landmarks, aligning them perfectly with facial features.

## App UI using TKinter
- **Selection Window**:
![Selection window]


## Future Improvements  
- Adding more filters and emotion categories.
- Improving accuracy for challenging environments (e.g., low light or partial occlusions).
- Optimizing performance for mobile platforms.

## Contact  
For any questions or suggestions, feel free to reach out:  
**Jeet Soni** – jeetsoni1005@gmail.com

