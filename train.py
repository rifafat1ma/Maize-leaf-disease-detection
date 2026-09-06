import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

from tqdm import tqdm


# -----------------------------
# BASIC SETTINGS
# -----------------------------

DATASET_PATH = "dataset"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# -----------------------------
# DATA AUGMENTATION
# -----------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.8, 1.0)
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# VALIDATION / TEST TRANSFORM
# -----------------------------

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# LOAD BASE DATASET
# -----------------------------

base_dataset = datasets.ImageFolder(
    DATASET_PATH
)

class_names = base_dataset.classes

print("Classes:", class_names)
print("Total images:", len(base_dataset))


# -----------------------------
# SPLIT DATASET
# -----------------------------

train_size = int(0.70 * len(base_dataset))
val_size = int(0.15 * len(base_dataset))
test_size = len(base_dataset) - train_size - val_size

generator = torch.Generator().manual_seed(42)

train_subset, val_subset, test_subset = random_split(
    base_dataset,
    [train_size, val_size, test_size],
    generator=generator
)

print("Training images:", len(train_subset))
print("Validation images:", len(val_subset))
print("Test images:", len(test_subset))


# -----------------------------
# CREATE DATASETS
# -----------------------------

train_dataset_full = datasets.ImageFolder(
    DATASET_PATH,
    transform=train_transform
)

val_dataset_full = datasets.ImageFolder(
    DATASET_PATH,
    transform=test_transform
)

test_dataset_full = datasets.ImageFolder(
    DATASET_PATH,
    transform=test_transform
)


train_dataset = torch.utils.data.Subset(
    train_dataset_full,
    train_subset.indices
)

val_dataset = torch.utils.data.Subset(
    val_dataset_full,
    val_subset.indices
)

test_dataset = torch.utils.data.Subset(
    test_dataset_full,
    test_subset.indices
)


# -----------------------------
# DATA LOADERS
# -----------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Data loaders created successfully!")


# -----------------------------
# LOAD MOBILENETV2
# -----------------------------

model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.DEFAULT
)

# Freeze pretrained layers
for param in model.features.parameters():
    param.requires_grad = False


# Change output layer to 4 classes
model.classifier[1] = nn.Linear(
    model.last_channel,
    len(class_names)
)

model = model.to(DEVICE)

print("MobileNetV2 loaded successfully!")


# -----------------------------
# LOSS AND OPTIMIZER
# -----------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)


# -----------------------------
# TRAIN MODEL
# -----------------------------

best_val_accuracy = 0.0

for epoch in range(EPOCHS):

    # =============================
    # TRAINING
    # =============================

    model.train()

    training_correct = 0
    training_total = 0

    for images, labels in tqdm(
        train_loader,
        desc=f"Training Epoch {epoch + 1}/{EPOCHS}"
    ):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        _, predicted = torch.max(
            outputs,
            1
        )

        training_total += labels.size(0)

        training_correct += (
            predicted == labels
        ).sum().item()


    train_accuracy = (
        100 * training_correct / training_total
    )


    # =============================
    # VALIDATION
    # =============================

    model.eval()

    validation_correct = 0
    validation_total = 0

    with torch.no_grad():

        for images, labels in tqdm(
            val_loader,
            desc=f"Validation Epoch {epoch + 1}/{EPOCHS}"
        ):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            validation_total += labels.size(0)

            validation_correct += (
                predicted == labels
            ).sum().item()


    val_accuracy = (
        100 * validation_correct / validation_total
    )


    # =============================
    # DISPLAY RESULTS
    # =============================

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Accuracy: {train_accuracy:.2f}% | "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )


    # =============================
    # SAVE BEST MODEL
    # =============================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "model/best_maize_model.pth"
        )

        print(
            f"Best model saved! "
            f"Validation Accuracy: {val_accuracy:.2f}%"
        )


print("\nTraining finished!")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)