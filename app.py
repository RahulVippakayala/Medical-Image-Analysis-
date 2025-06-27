import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Text, Scrollbar
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Input, Conv2DTranspose, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image as PILImage, ImageTk  # For displaying images in Tkinter
import cv2  # Import OpenCV for displaying the image

# Global variables for the dataset path, model, and class labels
global model, train_dir, class_labels, history, gan, generator, discriminator

latent_dim = 100  # Latent space dimension for the GAN

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
def create_cnn_model(input_shape):
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
def train_model(model_type, epochs_cnn=4):
    global model, train_dir, history
    if not train_dir:
        messagebox.showerror("Error", "Preprocessing data first is required.")
        return

    # Assuming image size is (150, 150, 3)
    if model_type == 'CNN':
        model = create_cnn_model((150, 150, 3))
        epochs = epochs_cnn

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

# Generator Model
def build_generator(latent_dim):
    model = Sequential()

    # Fully connected layer
    model.add(Dense(128 * 7 * 7, activation='relu', input_dim=latent_dim))
    model.add(Reshape((7, 7, 128)))  # Reshape to 7x7x128

    # Upsample layers
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # Output layer (single channel image)
    model.add(Conv2DTranspose(1, kernel_size=3, strides=1, padding='same', activation='tanh'))  # Grayscale image

    return model

# Discriminator Model
def build_discriminator(image_shape):
    model = Sequential()

    # Downsample layers
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=image_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))  # Binary classification (real or fake)

    return model

# GAN Model (Combines generator and discriminator)
def build_gan(generator, discriminator):
    discriminator.trainable = False  # Only train the generator in the GAN

    gan_input = Input(shape=(latent_dim,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)

    model = Model(gan_input, gan_output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    return model

# Function to train the GAN
def train_gan(generator, discriminator, gan, epochs=1, batch_size=128, latent_dim=100, train_data=None):
    global text
    # Rescale to [-1, 1]
    real_images = train_data
    #real_images = (real_images.astype(np.float32) - 127.5) / 127.5
    real_images = np.float32(real_images) / 255

    # Create labels
    real_labels = np.ones((batch_size, 1))  # Real labels
    fake_labels = np.zeros((batch_size, 1))  # Fake labels

    # Training loop
    for epoch in range(epochs):
        # Train discriminator on real images
        idx = np.random.randint(0, real_images.shape[0], batch_size)
        real_batch = real_images[idx]
        d_loss_real = discriminator.train_on_batch(real_batch, real_labels)

        # Generate fake images using the generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_batch = generator.predict(noise)
        d_loss_fake = discriminator.train_on_batch(fake_batch, fake_labels)

        # Calculate the total discriminator loss
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the generator via the GAN model (attempt to fool the discriminator)
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real_labels)  # We want to fool the discriminator with fake labels

        # Print the progress
        text.delete(1.0, tk.END)
        text.insert(tk.END, f"Epoch {epoch}/{epochs} [D loss: {d_loss}] [G loss: {g_loss}]\n")
        root.update()  # Update the Tkinter window

# Function to upload an image and predict its class
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
    cv2.putText(img_cv2, f"Prediction: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)  # Add prediction text
    cv2.imshow("Prediction Image", img_cv2)  # Show image with prediction in OpenCV

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

train_model_button = tk.Button(button_frame, text="Train CNN Model", command=lambda: train_model('CNN'), **button_style)
train_model_button.grid(row=0, column=2, padx=10, pady=5)

train_gan_button = tk.Button(button_frame, text="Train GAN Model", command=lambda: train_gan(build_generator, build_discriminator, build_gan, epochs=1, batch_size=128, latent_dim=100, train_data=None), **button_style)
train_gan_button.grid(row=0, column=3, padx=10, pady=5)

upload_and_predict_button = tk.Button(button_frame, text="Upload & Predict", command=upload_and_predict, **button_style)
upload_and_predict_button.grid(row=0, column=4, padx=10, pady=5)

# Label to display the uploaded image
original_image_label = tk.Label(root)
original_image_label.pack(pady=10)

# Run the Tkinter main loop
root.mainloop()
