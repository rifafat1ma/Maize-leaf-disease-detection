import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

from tqdm import tqdm

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# ==========================================
# SETTINGS
# ==========================================

DATASET_PATH = "dataset"

IMAGE_SIZE = 224
BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# ==========================================
# TEST IMAGE TRANSFORM
# ==========================================

test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# LOAD DATASET
# ==========================================

base_dataset = datasets.ImageFolder(
    DATASET_PATH
)

class_names = base_dataset.classes

print("Classes:", class_names)
print("Total images:", len(base_dataset))


# ==========================================
# RECREATE SAME TRAIN / VALIDATION / TEST
# SPLIT USED DURING TRAINING
# ==========================================

train_size = int(
    0.70 * len(base_dataset)
)

val_size = int(
    0.15 * len(base_dataset)
)

test_size = (
    len(base_dataset)
    - train_size
    - val_size
)


generator = torch.Generator().manual_seed(42)


train_subset, val_subset, test_subset = random_split(

    base_dataset,

    [
        train_size,
        val_size,
        test_size
    ],

    generator=generator
)


print("Training images:", len(train_subset))
print("Validation images:", len(val_subset))
print("Test images:", len(test_subset))


# ==========================================
# CREATE TEST DATASET
# ==========================================

test_dataset_full = datasets.ImageFolder(

    DATASET_PATH,

    transform=test_transform
)


test_dataset = torch.utils.data.Subset(

    test_dataset_full,

    test_subset.indices
)


# ==========================================
# TEST DATA LOADER
# ==========================================

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0
)


print("Test DataLoader created successfully!")


# ==========================================
# CREATE MOBILENETV2 MODEL
# ==========================================

model = models.mobilenet_v2(
    weights=None
)


# Change final layer for our 4 classes

model.classifier[1] = nn.Linear(

    model.last_channel,

    len(class_names)
)


# ==========================================
# LOAD OUR TRAINED MODEL
# ==========================================

model.load_state_dict(

    torch.load(

        "model/best_maize_model.pth",

        map_location=DEVICE
    )
)


model = model.to(DEVICE)

model.eval()


print("Best trained model loaded successfully!")


# ==========================================
# TEST MODEL
# ==========================================

test_correct = 0
test_total = 0


# Store predictions for confusion matrix

all_labels = []

all_predictions = []


with torch.no_grad():

    for images, labels in tqdm(

        test_loader,

        desc="Testing"
    ):

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)


        # Make prediction

        outputs = model(images)


        # Get predicted class

        _, predicted = torch.max(
            outputs,
            1
        )


        # Count total images

        test_total += labels.size(0)


        # Count correct predictions

        test_correct += (

            predicted == labels

        ).sum().item()


        # Save true labels

        all_labels.extend(

            labels.cpu().numpy()

        )


        # Save predictions

        all_predictions.extend(

            predicted.cpu().numpy()

        )


# ==========================================
# TEST ACCURACY
# ==========================================

test_accuracy = (

    100
    * test_correct
    / test_total
)


print("\n==============================")

print(
    f"Test Accuracy: "
    f"{test_accuracy:.2f}%"
)

print("==============================")


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:\n")


report = classification_report(

    all_labels,

    all_predictions,

    target_names=class_names,

    digits=4
)


print(report)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(

    all_labels,

    all_predictions
)


print("\nConfusion Matrix:\n")

print(cm)


# ==========================================
# DISPLAY CONFUSION MATRIX
# ==========================================

display = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=class_names
)


display.plot(

    xticks_rotation=45
)


plt.title(
    "Maize Leaf Disease Detection - Confusion Matrix"
)


plt.tight_layout()


# ==========================================
# SAVE CONFUSION MATRIX
# ==========================================

plt.savefig(

    "results/confusion_matrix.png",

    dpi=300,

    bbox_inches="tight"
)


print(
    "\nConfusion matrix saved to:"
)

print(
    "results/confusion_matrix.png"
)


# Show graph

plt.show()