import argparse
import os
import torch
import clip
from PIL import Image


def main():
    parser = argparse.ArgumentParser(
        description="Run CLIP on an image with text prompts"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="images",
        help="directory containing images to analyze",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B/32",
        help="name of the CLIP model to use",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["table", "chart", "logo", "architecture"],
        help="text prompts to compare with the images",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.model, device=device)

    text = clip.tokenize(args.prompts).to(device)

    for img_file in sorted(os.listdir(args.image_dir)):
        img_path = os.path.join(args.image_dir, img_file)
        try:
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Skipping {img_file}: {e}")
            continue

        with torch.no_grad():
            logits_per_image, _ = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        print(f"Results for {img_file}:")
        for label, prob in zip(args.prompts, probs[0]):
            print(f"  {label}: {prob:.4f}")
        print()


if __name__ == "__main__":
    main()

