import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import tensorflow as tf

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Intel Image Classifier",
    page_icon="🌍",
    layout="centered"
)

# ── Class names ────────────────────────────────────────────
CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
CLASS_ICONS = {
    'buildings': '🏢',
    'forest':    '🌲',
    'glacier':   '🧊',
    'mountain':  '⛰️',
    'sea':       '🌊',
    'street':    '🛣️'
}
IMG_SIZE = 224

# ── PyTorch Model Definition ───────────────────────────────
class IntelCNN_PyTorch(nn.Module):
    def __init__(self, num_classes=6):
        super(IntelCNN_PyTorch, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

# ── Load Models ────────────────────────────────────────────
@st.cache_resource
def load_models():
    device = torch.device("cpu")

    # Load PyTorch model
    pytorch_model = IntelCNN_PyTorch(num_classes=6).to(device)
    pytorch_model.load_state_dict(
        torch.load("maimouna_model.pth", map_location=device)
    )
    pytorch_model.eval()

    # Load TensorFlow model
    tf_model = tf.keras.models.load_model("maimouna_model.keras")

    return pytorch_model, tf_model

# ── Preprocessing ──────────────────────────────────────────
def preprocess_pytorch(image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def preprocess_tensorflow(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ── UI ─────────────────────────────────────────────────────
st.title("🌍 Intel Image Classifier")
st.markdown("Classify natural scenes using Deep Learning — **Maimouna Ndoye**")
st.divider()

# Model selector
model_choice = st.selectbox(
    "🤖 Select Model",
    ["PyTorch CNN", "TensorFlow CNN"]
)

# Image upload
uploaded_file = st.file_uploader(
    "🖼️ Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("🔍 Classify Image", type="primary"):
        with st.spinner("Classifying..."):

            pytorch_model, tf_model = load_models()

            if model_choice == "PyTorch CNN":
                tensor = preprocess_pytorch(image)
                with torch.no_grad():
                    outputs = pytorch_model(tensor)
                    probs   = torch.softmax(outputs, dim=1)[0]
                    idx     = probs.argmax().item()
                confidence = round(probs[idx].item() * 100, 2)

            else:
                arr   = preprocess_tensorflow(image)
                probs = tf_model.predict(arr, verbose=0)[0]
                idx   = int(np.argmax(probs))
                confidence = round(float(probs[idx]) * 100, 2)

            predicted_class = CLASSES[idx]
            icon = CLASS_ICONS[predicted_class]

        # Show result
        st.divider()
        st.success("✅ Classification complete!")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Class", f"{icon} {predicted_class.upper()}")
        with col2:
            st.metric("Confidence", f"{confidence}%")

        st.progress(confidence / 100)
        st.caption(f"Model used: {model_choice}")