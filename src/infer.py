import argparse
import torch
from PIL import Image
from torchvision import transforms

from model import build_model


def main():
    parser = argparse.ArgumentParser(description="MURA model inference")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pt)")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    model = build_model()
    checkpoint = torch.load(args.weights, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img = Image.open(args.image).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)

    print("Model output:", output)


if __name__ == "__main__":
    main()
