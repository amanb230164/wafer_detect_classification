import torch
import torch.nn as nn
import numpy as np
import pickle
from PIL import Image


class WaferCNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 8 * 8,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# Load Label Encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Load Model
model = WaferCNN(
    len(le.classes_)
)

model.load_state_dict(
    torch.load(
        "wafer_best_model.pth",
        map_location="cpu"
    )
)

model.eval()


def predict_wafer(image):

    image = image.convert("RGB")

    image = image.resize((64, 64))

    image = np.array(image)

    image = (
        torch.tensor(
            image,
            dtype=torch.float32
        )
        .permute(2, 0, 1)
        .unsqueeze(0)
        / 255.0
    )

    with torch.no_grad():

        output = model(image)

        pred = output.argmax(
            dim=1
        ).item()

    prediction = le.inverse_transform(
        [pred]
    )[0]

    return prediction