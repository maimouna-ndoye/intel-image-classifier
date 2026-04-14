import os
import io
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image

# ── PyTorch ────────────────────────────────────────────────
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# ── TensorFlow ─────────────────────────────────────────────
import tensorflow as tf

app = Flask(__name__)

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
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════
# PyTorch Model Definition
# ══════════════════════════════════════════════════════════
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
        self.gap        = nn.AdaptiveAvgPool2d(1)
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

# ══════════════════════════════════════════════════════════
# Load Models
# ══════════════════════════════════════════════════════════
print("Loading PyTorch model...")
pytorch_model = IntelCNN_PyTorch(num_classes=6).to(device)
pytorch_model.load_state_dict(torch.load("maimouna_model.pth", map_location=device))
pytorch_model.eval()
print("✅ PyTorch model loaded")

print("Loading TensorFlow model...")
tf_model = tf.keras.models.load_model("maimouna_model.keras")
print("✅ TensorFlow model loaded")

# ══════════════════════════════════════════════════════════
# Preprocessing
# ══════════════════════════════════════════════════════════
def preprocess_pytorch(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

def preprocess_tensorflow(image: Image.Image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0    # [224, 224, 3]
    return np.expand_dims(arr, axis=0)                # [1, 224, 224, 3]

# ══════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file        = request.files["image"]
    image       = Image.open(io.BytesIO(file.read())).convert("RGB")
    model_choice = request.form.get("model", "pytorch")

    if model_choice == "pytorch":
        tensor  = preprocess_pytorch(image)
        with torch.no_grad():
            outputs = pytorch_model(tensor)
            probs   = torch.softmax(outputs, dim=1)[0]
            idx     = probs.argmax().item()
        confidence = round(probs[idx].item() * 100, 2)

    elif model_choice == "tensorflow":
        arr     = preprocess_tensorflow(image)
        probs   = tf_model.predict(arr, verbose=0)[0]
        idx     = int(np.argmax(probs))
        confidence = round(float(probs[idx]) * 100, 2)

    else:
        return jsonify({"error": "Unknown model"}), 400

    predicted_class = CLASSES[idx]
    icon            = CLASS_ICONS[predicted_class]

    return jsonify({
        "class":      predicted_class,
        "icon":       icon,
        "confidence": confidence,
        "model":      model_choice
    })

# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8080)