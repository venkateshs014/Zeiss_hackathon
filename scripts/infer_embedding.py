"""Export one frozen retinal encoder embedding from a fundus image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.checkpoints import load_encoder_checkpoint
from src.models.encoders import build_encoder
from src.utils.device import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a 2048-dimensional frozen ResNet-50 embedding")
    parser.add_argument("--image", required=True, help="Input fundus image")
    parser.add_argument("--checkpoint", required=True, help="Encoder-only .pt export")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    device = resolve_device(args.device)
    encoder, embedding_dim = build_encoder("resnet50", pretrained=False)
    checkpoint = load_encoder_checkpoint(encoder, args.checkpoint, device)
    encoder.to(device).eval()
    transform = transforms.Compose([
        transforms.Resize(args.image_size + 32),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = encoder(tensor).squeeze(0).cpu().tolist()
    if len(embedding) != embedding_dim:
        raise RuntimeError(f"Unexpected embedding dimension {len(embedding)}; expected {embedding_dim}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"image": str(image_path), "checkpoint": str(args.checkpoint), "encoder": "torchvision.resnet50", "embedding_dim": embedding_dim, "embedding": embedding}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
