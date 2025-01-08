import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
import warnings
from tkinter import Tk, filedialog

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Constants
IMAGE_SIZE = (224, 224)  # Image size for ResNet50
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0001
DATASET_DIR = 'C:/Users/ahmad/Downloads/finetune-20241230T201420Z-001/finetune/randomized_dataset'  # Updated dataset path
MODEL_SAVE_PATH = 'model.h5'

# Load the dataset
def load_dataset(dataset_dir):
    class_names = []
    images = []
    labels = []

    for root, dirs, files in os.walk(dataset_dir):
        for dir_name in dirs:
            class_names.append(dir_name)
            dir_path = os.path.join(root, dir_name)
            for img_name in os.listdir(dir_path):
                img_path = os.path.join(dir_path, img_name)
                if os.path.isfile(img_path):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            img = load_img(img_path, target_size=IMAGE_SIZE)
                        img_array = img_to_array(img) / 255.0
                        images.append(img_array)
                        labels.append(class_names.index(dir_name))
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")

    images = np.array(images)
    labels = np.array(labels)
    return images, labels, class_names

# Define Data Augmentation
data_augmentation = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Load dataset
train_images, train_labels, class_names = load_dataset(DATASET_DIR)
print(f"Classes: {class_names}")

# Check if dataset is empty
if len(train_images) == 0:
    raise ValueError("No images found in the dataset. Please check the directory paths and ensure there are images.")

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Check if model exists
if os.path.exists(MODEL_SAVE_PATH):
    # Load the model
    model = load_model(MODEL_SAVE_PATH)
    print(f"Loaded pre-trained model from {MODEL_SAVE_PATH}")
else:
    # Define ResNet50 Model with custom top layers
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    base_model.trainable = False  # Freeze the base model

    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(len(class_names), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    # Train the model
    history = model.fit(
        data_augmentation.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping]
    )

    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    # Save the model in Keras format
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

# Function to predict on a new image
def predict_image(image_path, model, class_names):
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    print(f"Predicted Class: {predicted_class}")

    plt.imshow(img)
    plt.title(f"Prediction: {predicted_class}")
    plt.show()

# Use Tkinter to open a file dialog for selecting an image
root = Tk()
root.withdraw()  # Hide the root window
img_path = filedialog.askopenfilename(title="Select an image file")

# Predict the class of the selected image
if img_path:
    predict_image(img_path, model, class_names)
else:
    print("No image selected.")

# Plot training and validation accuracy/loss
if 'history' in locals():
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()
else:
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy of loaded model: {val_accuracy * 100:.2f}%")