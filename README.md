
# Medical Organ Classification Project

This project focuses on developing an AI model to classify the primary organs in medical images. The targeted organ categories include **Heart**, **Brain**, **Breast**, and **Limbs**. The model leverages transfer learning using ResNet50, combined with custom classification layers, to accurately distinguish between these organ types.

## Objective

The primary goal of this project is to create a robust AI system capable of automatically classifying medical images into their respective organ categories. This can assist healthcare professionals in streamlining diagnostics and improving clinical workflows.

---

## Project Structure

```
.
|-- dataset/
|   |-- Train/           # Training data (70% of the dataset)
|   |   |-- Brain/
|   |   |-- Heart/
|   |   |-- Breast/
|   |   |-- Limbs/
|   |-- Test/            # Testing data (20% of the dataset)
|   |   |-- Brain/
|   |   |-- Heart/
|   |   |-- Breast/
|   |   |-- Limbs/
|   |-- Validate/        # Validation data (10% of the dataset)
|       |-- Brain/
|       |-- Heart/
|       |-- Breast/
|       |-- Limbs/
|       |-- src/
|       |-- main.py          # Contains the entire pipeline: data loading, model training, evaluation, and prediction
|       |-- README.md            # Project documentation
|       |-- requirements.txt     # Dependencies for the project
|       |-- LICENSE              # License information
```

---

## Dataset Description

The dataset consists of labeled images for the following four categories:
- **Heart**
- **Brain**
- **Breast**
- **Limbs**

### Dataset Organization
- **Train**: 70% of the dataset, used for training the AI model.
- **Validation**: 10% of the dataset, used to fine-tune hyperparameters.
- **Test**: 20% of the dataset, used to evaluate the model's final performance.

Each organ category is stored in its respective subfolder within the `Train`, `Test`, and `Validate` directories.

---

## AI Model Architecture

The AI model uses a transfer learning approach based on **ResNet50**, which has been pre-trained on the ImageNet dataset. Key modifications include:
- Removing the fully connected layers of ResNet50.
- Adding custom dense layers for classification.
- Using **Global Average Pooling (GAP)** to reduce the feature maps.
- Outputting four logits corresponding to the four organ classes.

### Model Summary
- **Base Model**: ResNet50 (pre-trained, frozen during initial training).
- **Custom Layers**: Global Average Pooling, Dense Layers with ReLU activation, and Softmax for output.
- **Loss Function**: Sparse Categorical Crossentropy.
- **Optimizer**: Adam (Learning Rate = 0.0001).
- **Metrics**: Accuracy.

### Training Details
- **Input Image Size**: 224 x 224 pixels.
- **Batch Size**: 32.
- **Epochs**: 15 (with early stopping based on validation loss).
- **Data Augmentation**: Rotation, width/height shift, shear, zoom, and horizontal flip.

---

## Implementation

### Prerequisites
- Python 3.10 or later.
- TensorFlow 2.x.
- Ensure the dataset is correctly structured and accessible.

### Setup
1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-repo/medical-organ-classification.git
   cd medical-organ-classification
   ```

2. **Install Dependencies**
   Install the required Python packages using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**
   Ensure the dataset is properly organized in the `dataset/` directory as described above.

### Training the Model
Run the `main.py` script to train the model on the dataset:
```bash
python src/main.py
```

### Testing the Model
Evaluate the model's performance on the test dataset using:
```bash
python src/main.py
```

### Predicting a New Image
Use the `main.py` script to classify a single image:
```bash
python src/main.py
```

---

## Results

The model achieves high accuracy on both the validation and test datasets, demonstrating its ability to generalize well. Example metrics include:
- **Training Accuracy**: ~90%
- **Validation Accuracy**: ~79%
- **Test Accuracy**: ~85%

### Example Predictions
Sample predictions on unseen images show the model’s ability to correctly classify organ categories even under challenging conditions such as variations in lighting, orientation, and resolution.

![Brain Example](Screenshots/Brain.jpg)


---

## Visualizations

### Training and Validation Metrics
The training process includes visualizations of accuracy and loss curves to monitor model performance.

- **Accuracy Curve**
  Shows the model's accuracy over epochs for both training and validation sets.
- **Loss Curve**
  Visualizes the reduction in training and validation loss over time.

### Confusion Matrix
Displays the model’s performance for each class on the test dataset, highlighting areas of improvement.

---

## Future Work

- **Dataset Expansion**: Include additional organ types such as **Breast**, **Lungs**, or **Kidney**.
- **Model Optimization**: Experiment with other architectures like EfficientNet or Vision Transformers.
- **Explainability**: Use Grad-CAM or similar techniques to visualize regions influencing predictions.
- **Deployment**: Create a web or mobile application for real-time predictions.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
