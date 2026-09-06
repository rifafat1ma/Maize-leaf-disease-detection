import streamlit as st
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

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# LOAD MODEL
# -----------------------------

@st.cache_resource
def load_model():

    model = models.mobilenet_v2(
        weights=None
    )

    model.classifier[1] = nn.Linear(
        model.last_channel,
        len(class_names)
    )

    model.load_state_dict(
        torch.load(
            "model/best_maize_model.pth",
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)

    model.eval()

    return model


model = load_model()


# -----------------------------
# STREAMLIT INTERFACE
# -----------------------------

st.title(
    "🌽 Maize Leaf Disease Detection"
)

st.write(
    "Upload a maize leaf image to predict its condition."
)


uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )


    # -----------------------------
    # PREPROCESS IMAGE
    # -----------------------------

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)


    # -----------------------------
    # PREDICT
    # -----------------------------

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )


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

    st.subheader(
        "Prediction Result"
    )

    st.success(
        f"Predicted Disease: {predicted_class}"
    )

    st.write(
        f"Confidence: {confidence_percentage:.2f}%"
    )


    st.subheader(
        "All Class Probabilities"
    )


    for class_name, probability in zip(
        class_names,
        probabilities[0]
    ):

        st.write(
            f"{class_name}: "
            f"{probability.item() * 100:.2f}%"
        )