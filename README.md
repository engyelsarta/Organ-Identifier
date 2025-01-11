
# Organ Classification AI Model

## Overview
This repository contains the implementation of an AI model designed to classify organ images into four categories: **Brain**, **Chest**, **Limbs**, and **Breast**. The model is trained, validated, and tested on a dataset of organ images and can predict the organ category for a new image provided by the user. The project aims to assist in organ classification tasks, which can be applied in fields like medical imaging and education.

---

## Table of Contents
- [Project Structure](#project-structure)
- [Dataset Description](#dataset-description)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Project Structure
The project is organized as follows:

```
.
|-- dataset/
|   |-- Train/
|   |   |-- Brain/
|   |   |-- Chest/
|   |   |-- Limbs/
|   |   |-- Breast/
|   |-- Test/
|   |   |-- Brain/
|   |   |-- Chest/
|   |   |-- Limbs/
|   |   |-- Breast/
|   |-- Validate/
|       |-- Brain/
|       |-- Chest/
|       |-- Limbs/
|       |-- Breast/
|-- src/
|   |-- model.py  # Contains the model architecture
|   |-- train.py  # Script to train the model
|   |-- test.py   # Script to test the model
|   |-- utils.py  # Utility functions for preprocessing and evaluation
|-- README.md
|-- requirements.txt
|-- LICENSE
```

---

## Dataset Description
The dataset consists of 1,000 labeled images for each of the four organ categories:
- **Brain**
- **Chest**
- **Limbs**
- **Breast**

The images are divided into three subsets:
- **Train**: 70% of the dataset, used for training the model.
- **Validation**: 10% of the dataset, used to fine-tune hyperparameters.
- **Test**: 20% of the dataset, used to evaluate the model's performance.

Each organ category is stored in its respective subfolder within the Train, Test, and Validate directories.

---

## Model Architecture
The model is built using **TensorFlow/Keras** and is based on a Convolutional Neural Network (CNN) architecture. The key layers include:
- Convolutional layers for feature extraction
- MaxPooling layers for dimensionality reduction
- Fully connected layers for classification
- Softmax activation for the output layer

The model uses **categorical cross-entropy** as the loss function and **Adam optimizer** for training.

---

## Installation
### Prerequisites
- Python 3.8+
- Virtual environment (optional but recommended)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/organ-classification-ai.git
   cd organ-classification-ai
   ```
2. Create and activate a virtual environment (optional):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
### Training the Model
To train the model on the dataset:
```bash
python src/train.py
```

### Testing the Model
To evaluate the model's performance on the test set:
```bash
python src/test.py
```

### Predicting a New Image
To classify a new image:
1. Place the image in the `src/` directory.

2. Run the following command:
   ```bash
   python src/predict.py --image your_image.jpg
   ```
3. The model will output the predicted organ category.

---

## Results
### Performance Metrics
- **Training Accuracy**: ~98%
- **Validation Accuracy**: ~95%
- **Test Accuracy**: ~94%

### Confusion Matrix
The confusion matrix shows the performance of the model in distinguishing between the four organ categories. Refer to the `results/` folder for detailed visualizations and metrics.

---

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature description"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


