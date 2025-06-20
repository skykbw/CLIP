import argparse
import torch
import clip
from PIL import Image


def main():
    parser = argparse.ArgumentParser(
        description="Run CLIP on an image with text prompts"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="CLIP.png",
        help="path to the image to analyze",
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
        default=["a diagram", "a dog", "a cat"],
        help="text prompts to compare with the image",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.model, device=device)

    image = preprocess(Image.open(args.image)).unsqueeze(0).to(device)
    text = clip.tokenize(args.prompts).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    for label, prob in zip(args.prompts, probs[0]):
        print(f"{label}: {prob:.4f}")


if __name__ == "__main__":
    main()

