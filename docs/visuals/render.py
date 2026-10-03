"""Regenerate the README schematics: python docs/visuals/render.py.

Requires Pillow (already a project dependency). These diagrams describe the
implemented software; they are not camera screenshots or measured results.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
NAVY, TEAL, INK = "#142C45", "#087F83", "#21384F"
MUTED, LINE, PALE, WHITE = "#53697B", "#CDDCE4", "#EFF7F8", "#FFFFFF"


def font(size, bold=False):
    for name in [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


class Diagram:
    def __init__(self, title, subtitle, height=1110):
        self.image = Image.new("RGB", (1600, height), WHITE)
        self.draw = ImageDraw.Draw(self.image)
        self.height = height
        self.draw.rounded_rectangle((1, 1, 1598, height - 2), 24, outline=LINE, width=2)
        self.text((60, 40), "FACE MASK DETECTION · SOFTWARE DESIGN", 24, TEAL, True)
        self.text((60, 86), title, 46, NAVY, True)
        self.text((60, 154), subtitle, 29, MUTED)

    def text(self, pos, value, size=29, color=INK, bold=False):
        self.draw.text(pos, value, fill=color, font=font(size, bold))

    def card(self, x, y, w, h, title, lines, accent=False, label=None):
        self.draw.rounded_rectangle(
            (x, y, x + w, y + h), 17, fill=PALE if accent else WHITE, outline=LINE, width=2
        )
        self.draw.rounded_rectangle((x, y, x + 7, y + h), 3, fill=TEAL)
        if label:
            self.text((x + 26, y + 20), label, 22, TEAL, True)
        offset = 56 if label else 23
        self.text((x + 26, y + offset), title, 32, NAVY, True)
        for i, line in enumerate(lines):
            self.text((x + 26, y + offset + 53 + i * 39), line, 28, MUTED)

    def arrow(self, pts, color=TEAL):
        self.draw.line(pts, fill=color, width=4, joint="curve")
        (px, py), (x, y) = pts[-2:]
        if x > px:
            triangle = [(x, y), (x - 13, y - 8), (x - 13, y + 8)]
        elif x < px:
            triangle = [(x, y), (x + 13, y - 8), (x + 13, y + 8)]
        elif y > py:
            triangle = [(x, y), (x - 8, y - 13), (x + 8, y - 13)]
        else:
            triangle = [(x, y), (x - 8, y + 13), (x + 8, y + 13)]
        self.draw.polygon(triangle, fill=color)

    def footer(self, text):
        self.draw.line((60, self.height - 86, 1540, self.height - 86), fill=LINE, width=2)
        self.text((60, self.height - 63), text, 25, MUTED)

    def save(self, name):
        self.image.save(ROOT / name, optimize=True)


def inputs():
    d = Diagram(
        "Two input routes. One RGB model contract.",
        "Training crops, saved images and camera crops must agree on what each channel means.",
        1220,
    )
    d.card(
        60,
        241,
        560,
        207,
        "Cropped image file",
        ["Decode with Pillow → RGB", "Bilinear resize to recorded dimensions"],
        label="HEADLESS INFERENCE / TRAINING DATA",
    )
    d.card(
        60,
        505,
        560,
        246,
        "OpenCV camera frame",
        [
            "Haar face detector → clipped crop",
            "BGR → RGB, then bilinear resize",
            "Crop pixels before drawing overlays",
        ],
        label="OPTIONAL CAMERA ADAPTER",
    )
    d.card(
        764,
        322,
        776,
        197,
        "Matching input arrays",
        ["RGB · float32 · pixels in [0, 255]", "Shape: height × width × 3"],
        accent=True,
    )
    d.arrow([(620, 344), (684, 344), (684, 420), (764, 420)])
    d.arrow([(620, 622), (684, 622), (684, 420)])
    d.card(
        764,
        590,
        776,
        197,
        "CNN + checked model metadata",
        [
            "Normalization is inside the saved model",
            "Metadata supplies shape, hash and ordered classes",
        ],
        accent=True,
    )
    d.arrow([(1152, 519), (1152, 590)])
    d.card(
        764,
        858,
        776,
        218,
        "Prediction with semantic labels",
        [
            "Categorical: ordered probabilities → argmax",
            "Binary: p(class index 1) → threshold",
            "Class names follow the recorded metadata order",
        ],
    )
    d.arrow([(1152, 787), (1152, 858)])
    d.text((60, 822), "Headless input is already a face crop.", 28, NAVY, True)
    d.text((60, 868), "The camera adapter finds the crop first.", 27, MUTED)
    d.text((60, 914), "Face detection and classification differ.", 27, MUTED)
    d.footer(
        "Implementation schematic · No face photos, webcam output or model-performance claims are shown."
    )
    d.save("input-contract.png")


def architecture():
    d = Diagram(
        "A compact convolutional classifier",
        "Shared RGB feature extractor; the class count determines the saved output contract.",
        1180,
    )
    d.card(
        60, 232, 410, 210, "RGB image", ["128 × 128 × 3", "float32 pixels: [0, 255]"], label="INPUT"
    )
    d.card(
        560,
        232,
        430,
        210,
        "Rescaling",
        ["Multiply pixels by 1 / 255", "Saved inside the model"],
        accent=True,
        label="NORMALIZATION",
    )
    d.card(
        1080,
        232,
        460,
        210,
        "Feature extraction",
        ["Four Conv2D + pool blocks", "ReLU activations"],
        label="LEARNED REPRESENTATION",
    )
    d.arrow([(470, 324), (560, 324)])
    d.arrow([(990, 324), (1080, 324)])
    d.draw.rounded_rectangle((60, 486, 1540, 747), 18, fill=PALE)
    d.text((88, 513), "Inside the four convolution blocks", 32, NAVY, True)
    for i, filters in enumerate([32, 64, 128, 128]):
        x = 90 + i * 368
        d.card(x, 578, 314, 132, f"{filters} filters · 3 × 3", ["ReLU → max pooling"])
        if i < 3:
            d.arrow([(x + 314, 642), (x + 368, 642)])
    d.arrow([(1310, 442), (1310, 486)])
    d.card(
        60, 817, 420, 205, "Global average pool", ["Spatial maps → feature vector", "Dropout: 0.5"]
    )
    d.card(
        570,
        817,
        420,
        205,
        "Dense: 128",
        ["ReLU activation", "Combine learned features"],
        accent=True,
    )
    d.card(
        1080,
        817,
        460,
        205,
        "Output head",
        [
            "3 classes: softmax vector",
            "2 classes: sigmoid probability",
            "Metadata records class order",
        ],
    )
    d.arrow([(280, 747), (280, 817)])
    d.arrow([(480, 900), (570, 900)])
    d.arrow([(990, 900), (1080, 900)])
    d.footer(
        "Architecture schematic from src/mask_detection/model.py · Blocks are conceptual, not to scale."
    )
    d.save("cnn-architecture.png")


if __name__ == "__main__":
    inputs()
    architecture()
