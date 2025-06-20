import os
import torch
import clip
from PIL import Image
import pandas as pd
import streamlit as st


def load_model(name: str, device: str):
    """Load CLIP model with caching."""
    @st.cache_resource
    def _load(name, device):
        return clip.load(name, device=device)

    return _load(name, device)


def classify_image(model, preprocess, image_path: str, prompts, device: str):
    """Return probability for each prompt given an image."""
    text = clip.tokenize(prompts).to(device)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(image, text)
        probs = logits.softmax(dim=-1).cpu().numpy()[0]
    return dict(zip(prompts, probs))


def main():
    st.title("CLIP Image Classification")

    image_dir = "images"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = st.sidebar.selectbox("Model", clip.available_models(), index=0)
    prompts_str = st.sidebar.text_input(
        "Comma separated prompts", "table, chart, logo, architecture"
    )
    prompts = [p.strip() for p in prompts_str.split(",") if p.strip()]

    model, preprocess = load_model(model_name, device)

    images = sorted(
        f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.write("Images")
        selected_img = st.selectbox("Select image", images)
        st.table(pd.DataFrame({"Image": images}))

    with col2:
        if selected_img:
            img_path = os.path.join(image_dir, selected_img)
            st.image(img_path, caption=selected_img)
            probs = classify_image(model, preprocess, img_path, prompts, device)
            st.write("Classification")
            for label, prob in probs.items():
                st.write(f"{label}: {prob:.4f}")


if __name__ == "__main__":
    main()
