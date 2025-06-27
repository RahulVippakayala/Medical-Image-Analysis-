import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Text, Scrollbar
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image as PILImage, ImageTk  # For displaying images in Tkinter
import cv2  # Import OpenCV for displaying the image

# Global variables for the dataset path, model, and class labels
global model, train_dir, class_labels, history

# Function to upload dataset (Single folder containing two classes: "Yes" and "No")
def load_dataset():
    global train_dir, class_labels
    folder_path = filedialog.askdirectory(title="Select Dataset Folder")
    
    if folder_path:
        train_dir = folder_path  # The entire folder with subfolders for each class
        
        # Check if subfolders (classes) are inside the selected folder
        if os.path.isdir(train_dir):
            subfolders = os.listdir(train_dir)
            if len(subfolders) == 2:  # Check if there are exactly two classes
                class_labels = subfolders  # Store class labels (folder names)
                text.delete('1.0', tk.END)
                text.insert(tk.END, f"Dataset loaded from: {train_dir}\n")
                text.insert(tk.END, f"Classes found: {', '.join(subfolders)}\n")
            else:
                messagebox.showerror("Error", "Please ensure the dataset has exactly two classes.")
        else:
            messagebox.showerror("Error", "Please select a valid dataset folder with subfolders for each class.")
    else:
        messagebox.showerror("Error", "Please select a valid dataset folder.")

# Image Preprocessing (Resizing and normalizing)
def preprocess_data():
    global train_dir
    if not train_dir:
        messagebox.showerror("Error", "Please upload a dataset first.")
        return

    # Image Preprocessing
    image_size = (150, 150)  # Resize images to 150x150

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Using 20% data for validation

    # Training data generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=32,
        class_mode='binary',  # Binary classification (Yes or No)
        subset='training'  # Use 80% of data for training
    )

    # Validation data generator
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=32,
        class_mode='binary',  # Binary classification (Yes or No)
        subset='validation'  # Use 20% of data for validation
    )

    text.delete(1.0, tk.END)
    text.insert(tk.END, "Data Preprocessing Completed.\n")
    text.insert(tk.END, f"Training set: {train_generator.samples} images\n")
    text.insert(tk.END, f"Validation set: {validation_generator.samples} images\n")

# CNN-based model for binary classification (Yes or No)
def create_gnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # 1 output unit (binary classification)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to train the selected model (CNN)
def train_model(model_type, epochs_gnn=8):
    global model, train_dir, history
    if not train_dir:
        messagebox.showerror("Error", "Preprocessing data first is required.")
        return

    # Assuming image size is (150, 150, 3)
    if model_type == 'GNN':
        model = create_gnn_model((150, 150, 3))
        epochs = epochs_gnn

    # Get data generators for training and validation
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    # Train the model
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

    # Show training results
    text.delete(1.0, tk.END)
    text.insert(tk.END, f"Model Training Completed.\n")
    

def upload_and_predict():
    global model, class_labels
    if model is None:
        messagebox.showerror("Error", "Model is not trained yet. Please train the model first.")
        return
    
    # Select image to upload
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    
    if not file_path:
        return

    # Preprocess image
    img = image.load_img(file_path, target_size=(150, 150))  # Resize image to match model input
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize image

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = "PNEUMONIA" if prediction[0] > 0.5 else "NORMAL"  # Convert prediction to Yes or No based on threshold (0.5)

    # Display prediction result in Tkinter text box
    text.delete(1.0, tk.END)
    text.insert(tk.END, f"Predicted Class: {predicted_class}\n")

    # Provide recommendation based on prediction
    if predicted_class == "PNEUMONIA":
        text.insert(tk.END, "Recommendation: Continue monitoring and proceed with necessary actions.\n")
    else:
        text.insert(tk.END, "Recommendation: No immediate action needed. Continue regular check-ups.\n")

    # Load the original image for Tkinter display
    img_original = PILImage.open(file_path)
    img_original = img_original.resize((150, 150))  # Resize the image to match the display size
    img_original_tk = ImageTk.PhotoImage(img_original)

    # Display the original image in Tkinter
    original_image_label.config(image=img_original_tk)
    original_image_label.image = img_original_tk  # Keep a reference to the image to prevent garbage collection

    # OpenCV for displaying the image in a separate window
    img_cv2 = cv2.imread(file_path)  # Read image using OpenCV
    resized_img = cv2.resize(img_cv2, (400, 400))
    cv2.putText(resized_img, f"Prediction: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)  # Add prediction text
    cv2.imshow("Prediction Image", resized_img)  # Show image with prediction in OpenCV

    # Wait for a key press to close the OpenCV window
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # Close the OpenCV window

# Initialize Tkinter Application
root = tk.Tk()
root.title("Generative Adversial Networks for Creating Synthetic Data to Improve Indian Medical Image Analysis")
root.geometry("1300x800")
root.configure(bg='LightSkyBlue')  # Main window background color set to gray

# Title label with a green background color
title_label = tk.Label(root, text="Generative Adversial Networks for Creating Synthetic Data to Improve Indian Medical Image Analysis", font=("Helvetica", 18, "bold"), bg='greenyellow', height=2, width=100)
title_label.pack(pady=6, fill=tk.X)

# Textbox to show output messages with a scrollbar
text_frame = tk.Frame(root, bg='gray')
text_frame.pack(pady=20)

scrollbar = Scrollbar(text_frame, orient=tk.VERTICAL)
text = Text(text_frame, height=15, width=80, wrap=tk.WORD, font=("Arial", 12, 'bold'), yscrollcommand=scrollbar.set)
scrollbar.config(command=text.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
text.pack(side=tk.LEFT)

# Create a frame for buttons to arrange them in two rows
button_frame = tk.Frame(root, bg='LightSkyBlue')  # Background of button frame to match main window
button_frame.pack(pady=10)

# Buttons with updated styles
button_style = {'font': ("Arial", 13, 'bold'), 'height':1, 'width':15, 'bg': 'light gray'}

load_dataset_button = tk.Button(button_frame, text="Load Dataset", command=load_dataset, **button_style)
load_dataset_button.grid(row=0, column=0, padx=10, pady=5)

preprocess_data_button = tk.Button(button_frame, text="Preprocess Data", command=preprocess_data, **button_style)
preprocess_data_button.grid(row=0, column=1, padx=10, pady=5)

train_model_button = tk.Button(button_frame, text="Train GNN Model", command=lambda: train_model('GNN'), **button_style)
train_model_button.grid(row=0, column=2, padx=10, pady=5)

upload_and_predict_button = tk.Button(button_frame, text="Upload & Predict", command=upload_and_predict, **button_style)
upload_and_predict_button.grid(row=0, column=4, padx=10, pady=5)

# Label to display the uploaded image
original_image_label = tk.Label(root)
original_image_label.pack(pady=10)

# Run the Tkinter main loop
root.mainloop()
