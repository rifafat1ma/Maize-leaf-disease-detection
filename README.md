#  Maize Leaf Disease Detection

A deep learning project for detecting diseases in maize leaves using **MobileNetV2 Transfer Learning** and **PyTorch**.

The model classifies maize leaf images into four categories:

- Blight
- Common Rust
- Gray Leaf Spot
- Healthy


##  Dataset

The dataset contains **4,188 maize leaf images** divided into four classes:

| Class | Number of Images |
|---|---:|
| Blight | 1,146 |
| Common Rust | 1,306 |
| Gray Leaf Spot | 574 |
| Healthy | 1,162 |

The dataset was split into:

- **70% Training** — 2,931 images
- **15% Validation** — 628 images
- **15% Testing** — 629 images


##  Data Preprocessing

Before training, the images were:

- Resized to **224 × 224 pixels**
- Converted to PyTorch tensors
- Normalized using ImageNet mean and standard deviation

The dataset was also checked for incorrect folder organization and labeling.


##  Data Augmentation

To improve the model's ability to work with new images, the training data uses:

- Random horizontal flipping
- Random rotation
- Random resized cropping
- Brightness and contrast adjustment


##  Model

This project uses **MobileNetV2 with Transfer Learning**.

The pretrained MobileNetV2 feature extraction layers are frozen, and the final classifier is modified to predict the four maize leaf classes.


##  Results

**Best Validation Accuracy: 91.24%**

**Test Accuracy: 91.10%**

A confusion matrix was also generated to analyze the performance of the model across the four classes.


##  Technologies Used

- Python
- PyTorch
- Torchvision
- MobileNetV2
- Streamlit
- Scikit-learn
- Matplotlib
- Pillow


##  Streamlit Application

A simple Streamlit interface is included where users can upload a maize leaf image and receive:

- Predicted disease
- Confidence score
- Probability for each class

Run the application using:

```bash
streamlit run app.py