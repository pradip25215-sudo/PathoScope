
from pathlib import Path
import zipfile
import tempfile
import shutil
import re

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

import onnxruntime as ort
import gradio as gr


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

SEG_MODEL = MODEL_DIR / "deeplabv3plus_resnet34.onnx"
MIL_MODEL = MODEL_DIR / "combined_attention_mil.pth"
EFFNET_WEIGHTS = MODEL_DIR / "efficientnet_b0_rwightman-7f5810bc.pth"

DEVICE = torch.device("cpu")


class AttentionMIL(nn.Module):
    def __init__(self, input_dim=1284, hidden_dim=256, num_classes=6):
        super().__init__()

        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, return_attention=False):
        h = self.feature_projection(x)

        attention_scores = self.attention(h)

        attention_weights = torch.softmax(
            attention_scores,
            dim=0
        )

        pooled = torch.sum(
            attention_weights * h,
            dim=0
        )

        logits = self.classifier(pooled)

        if return_attention:
            return logits, attention_weights.squeeze(-1)

        return logits


print("Loading ONNX segmentation model...")

seg_session = ort.InferenceSession(
    str(SEG_MODEL),
    providers=["CPUExecutionProvider"]
)

seg_input_name = seg_session.get_inputs()[0].name

print("Segmentation model loaded.")


print("Loading EfficientNet-B0...")

efficientnet = models.efficientnet_b0(
    weights=None
)

state_dict = torch.load(
    EFFNET_WEIGHTS,
    map_location="cpu",
    weights_only=True
)

efficientnet.load_state_dict(state_dict)

efficientnet.classifier = nn.Identity()

efficientnet.eval()
efficientnet.to(DEVICE)

effnet_weights = EfficientNet_B0_Weights.DEFAULT
effnet_transform = effnet_weights.transforms()

print("EfficientNet-B0 loaded.")


print("Loading Attention-MIL...")

mil_model = AttentionMIL(
    input_dim=1284,
    hidden_dim=256,
    num_classes=6
)

checkpoint = torch.load(
    MIL_MODEL,
    map_location="cpu",
    weights_only=False
)

mil_model.load_state_dict(
    checkpoint["model_state_dict"]
)

mil_model.eval()
mil_model.to(DEVICE)

print("Attention-MIL loaded.")


def prepare_segmentation_image(image):
    image = image.convert("RGB")
    image = image.resize((256, 256))

    array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    array = np.transpose(
        array,
        (2, 0, 1)
    )

    array = np.expand_dims(
        array,
        axis=0
    )

    return array.astype(np.float32)


def run_segmentation(image):
    input_array = prepare_segmentation_image(image)

    output = seg_session.run(
        None,
        {seg_input_name: input_array}
    )[0]

    logits = output[0, 0]

    probability = 1.0 / (
        1.0 + np.exp(-logits)
    )

    return probability


def create_heatmap(probability):
    heat = np.clip(
        probability * 255.0,
        0,
        255
    ).astype(np.uint8)

    heat_rgb = np.zeros(
        (heat.shape[0], heat.shape[1], 3),
        dtype=np.uint8
    )

    heat_rgb[:, :, 0] = heat
    heat_rgb[:, :, 1] = (
        heat * 0.35
    ).astype(np.uint8)

    return Image.fromarray(heat_rgb)


def create_overlay(image, probability):
    base = image.convert("RGB").resize(
        (256, 256)
    )

    heatmap = create_heatmap(
        probability
    )

    base_array = np.asarray(
        base,
        dtype=np.float32
    )

    heat_array = np.asarray(
        heatmap,
        dtype=np.float32
    )

    overlay = (
        0.65 * base_array +
        0.35 * heat_array
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(overlay)


def analyze_tile(image):
    if image is None:
        return (
            None,
            None,
            None,
            "Please upload a tile image."
        )

    probability = run_segmentation(
        image
    )

    heatmap = create_heatmap(
        probability
    )

    overlay = create_overlay(
        image,
        probability
    )

    report = {
        "mean_tumor_probability":
            float(probability.mean()),

        "maximum_pixel_tumor_probability":
            float(probability.max()),

        "tumor_area_probability_ge_0.10":
            float((probability >= 0.10).mean()),

        "tumor_area_probability_ge_0.50":
            float((probability >= 0.50).mean())
    }

    return (
        heatmap,
        overlay,
        report,
        "Tile segmentation completed."
    )


def load_zip_tiles(zip_path):
    if zip_path is None:
        raise ValueError(
            "Please upload a ZIP containing 64 normalized tiles."
        )

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="pathoscope_"
        )
    )

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as z:
        z.extractall(temp_dir)

    image_files = sorted(
        [
            p for p in temp_dir.rglob("*")
            if p.suffix.lower() in [
                ".jpg",
                ".jpeg",
                ".png"
            ]
        ]
    )

    if len(image_files) != 64:
        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

        raise ValueError(
            f"Expected exactly 64 tile images, found {len(image_files)}."
        )

    return temp_dir, image_files


def extract_slide_features(image_files):
    cnn_features = []
    segmentation_features = []

    for image_path in image_files:

        image = Image.open(
            image_path
        ).convert("RGB")

        eff_input = effnet_transform(
            image
        ).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            feature = efficientnet(
                eff_input
            )

        cnn_features.append(
            feature.squeeze(0)
            .cpu()
            .numpy()
        )

        probability = run_segmentation(
            image
        )

        segmentation_features.append(
            [
                float(probability.mean()),
                float(probability.max()),
                float((probability >= 0.10).mean()),
                float((probability >= 0.50).mean())
            ]
        )

    cnn_features = np.asarray(
        cnn_features,
        dtype=np.float32
    )

    segmentation_features = np.asarray(
        segmentation_features,
        dtype=np.float32
    )

    return np.concatenate(
        [
            cnn_features,
            segmentation_features
        ],
        axis=1
    )


def parse_coordinates(filename):
    match = re.search(
        r"_x(\d+)_y(\d+)",
        filename
    )

    if match:
        return (
            int(match.group(1)),
            int(match.group(2))
        )

    return (
        None,
        None
    )


def predict_slide(zip_path):
    temp_dir = None

    try:
        temp_dir, image_files = load_zip_tiles(
            zip_path
        )

        features = extract_slide_features(
            image_files
        )

        feature_tensor = torch.from_numpy(
            features
        ).float().to(DEVICE)

        with torch.inference_mode():
            logits, attention = mil_model(
                feature_tensor,
                return_attention=True
            )

            probabilities = torch.softmax(
                logits,
                dim=0
            ).cpu().numpy()

            predicted_isup = int(
                np.argmax(probabilities)
            )

            attention = attention.cpu().numpy()

        top_indices = np.argsort(
            attention
        )[::-1][:10]

        top_tiles = []

        for index in top_indices:
            filename = image_files[index].name

            x, y = parse_coordinates(
                filename
            )

            top_tiles.append(
                {
                    "tile": filename,
                    "x": x,
                    "y": y,
                    "attention": float(
                        attention[index]
                    )
                }
            )

        return {
            "predicted_ISUP_grade":
                predicted_isup,

            "probability_ISUP_0":
                float(probabilities[0]),

            "probability_ISUP_1":
                float(probabilities[1]),

            "probability_ISUP_2":
                float(probabilities[2]),

            "probability_ISUP_3":
                float(probabilities[3]),

            "probability_ISUP_4":
                float(probabilities[4]),

            "probability_ISUP_5":
                float(probabilities[5]),

            "attention_sum":
                float(attention.sum()),

            "top_10_attention_tiles":
                top_tiles
        }

    finally:
        if temp_dir is not None:
            shutil.rmtree(
                temp_dir,
                ignore_errors=True
            )


with gr.Blocks(
    title="PathoScope"
) as demo:

    gr.Markdown(
        """
        # PathoScope

        **AI-Driven Digital Pathology Pipeline for Prostate Cancer Detection & ISUP Grading**

        Upload a normalized histopathology tile for tumor segmentation,
        or upload a ZIP containing exactly 64 normalized tiles for
        slide-level ISUP grading.
        """
    )

    with gr.Tab("Tumor Segmentation"):

        tile_input = gr.Image(
            type="pil",
            label="Upload normalized tile"
        )

        tile_button = gr.Button(
            "Run Tumor Segmentation"
        )

        with gr.Row():

            heatmap_output = gr.Image(
                label="Tumor Probability Map"
            )

            overlay_output = gr.Image(
                label="Tumor Overlay"
            )

        tile_report = gr.JSON(
            label="Segmentation Results"
        )

        tile_status = gr.Textbox(
            label="Status"
        )

        tile_button.click(
            fn=analyze_tile,
            inputs=tile_input,
            outputs=[
                heatmap_output,
                overlay_output,
                tile_report,
                tile_status
            ]
        )

    with gr.Tab("Slide-Level ISUP Grading"):

        zip_input = gr.File(
            label="Upload ZIP containing exactly 64 normalized tiles",
            file_types=[".zip"],
            type="filepath"
        )

        slide_button = gr.Button(
            "Run ISUP Grading"
        )

        slide_result = gr.JSON(
            label="Slide-Level Prediction"
        )

        slide_button.click(
            fn=predict_slide,
            inputs=zip_input,
            outputs=slide_result
        )


if __name__ == "__main__":
    demo.launch()
