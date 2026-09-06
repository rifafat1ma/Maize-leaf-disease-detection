import torch
import torch.nn as nn

from torchvision import transforms, models
from PIL import Image

# -----------------------------
# SETTINGS
# -----------------------------

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

class_names = [
    "Blight",
    "Common_Rust",
    "Gray_Leaf_Spot",
    "Healthy"
]

# -----------------------------
# IMAGE TRANSFORM
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# CREATE MODEL STRUCTURE
# -----------------------------

model = models.mobilenet_v2(
    weights=None
)

model.classifier[1] = nn.Linear(
    model.last_channel,
    len(class_names)
)

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------

model.load_state_dict(
    torch.load(
        "model/best_maize_model.pth",
        map_location=DEVICE
    )
)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully!")

# -----------------------------
# GET IMAGE PATH
# -----------------------------

image_path = input(
    "\nEnter image path: "
)

# -----------------------------
# OPEN IMAGE
# -----------------------------

image = Image.open(
    image_path
).convert("RGB")

# Apply same preprocessing
image_tensor = transform(
    image
)

# Add batch dimension
image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(DEVICE)

# -----------------------------
# MAKE PREDICTION
# -----------------------------

with torch.no_grad():

    outputs = model(
        image_tensor
    )

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

# -----------------------------
# GET RESULT
# -----------------------------

confidence, predicted = torch.max(
    probabilities,
    1
)

predicted_class = class_names[
    predicted.item()
]

confidence_percentage = (
    confidence.item() * 100
)

# -----------------------------
# DISPLAY RESULT
# -----------------------------

print("\n==============================")
print("Prediction Result")
print("==============================")

print(
    "Predicted Disease:",
    predicted_class
)

print(
    f"Confidence: "
    f"{confidence_percentage:.2f}%"
)

print("\nAll probabilities:\n")

for class_name, probability in zip(
    class_names,
    probabilities[0]
):

    print(
        f"{class_name}: "
        f"{probability.item() * 100:.2f}%"
    )