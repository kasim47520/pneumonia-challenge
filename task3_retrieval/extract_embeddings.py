"""
Task 3: Extract embeddings from all test images using BioMedCLIP / PMC-CLIP.

Usage:
    python extract_embeddings.py --output_dir outputs/task3
"""

import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.dataset import get_raw_dataset


def load_embedding_model(model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                         device="cpu"):
    """
    Load BioMedCLIP for embedding extraction.
    BioMedCLIP is a biomedical vision-language model trained on PubMed figure-caption pairs.
    It captures medically relevant visual semantics.
    """
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(model_name)
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(device).eval()
        print(f"Loaded BioMedCLIP from open_clip")
        return model, preprocess, tokenizer, "biomedclip"
    except Exception as e:
        print(f"BioMedCLIP failed ({e}), falling back to OpenAI CLIP ViT-B/32")
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            model = model.to(device).eval()
            print("Loaded CLIP ViT-B/32 (OpenAI)")
            return model, preprocess, tokenizer, "clip"
        except Exception as e2:
            print(f"CLIP also failed ({e2}), using CNN features from timm")
            return None, None, None, "timm"


@torch.no_grad()
def extract_clip_embeddings(model, preprocess, dataset, device, batch_size=64):
    """Extract visual embeddings using CLIP-style model."""
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from PIL import Image

    all_embeddings = []
    all_labels = []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Extracting embeddings"):
        batch_imgs, batch_lbls = [], []
        for j in range(i, min(i + batch_size, len(dataset))):
            img_tensor, label = dataset[j]
            # Convert tensor to PIL for preprocess
            img_np = img_tensor.permute(1, 2, 0).numpy()
            img_np = (img_np * 0.5 + 0.5).clip(0, 1)
            if img_np.shape[2] == 1:
                img_np = img_np.squeeze(-1)
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray((img_np * 255).astype(np.uint8))
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            batch_imgs.append(preprocess(pil_img))
            batch_lbls.append(label.item())

        batch_tensor = torch.stack(batch_imgs).to(device)
        embeddings = model.encode_image(batch_tensor)
        # L2 normalize
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        all_embeddings.append(embeddings.cpu().float().numpy())
        all_labels.extend(batch_lbls)

    return np.vstack(all_embeddings), np.array(all_labels)


@torch.no_grad()
def extract_timm_embeddings(dataset, device, batch_size=64):
    """Fallback: use EfficientNet features from timm."""
    import timm
    from torch.utils.data import DataLoader

    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
    model = model.to(device).eval()

    all_embeddings = []
    all_labels = []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Extracting timm embeddings"):
        batch_imgs, batch_lbls = [], []
        for j in range(i, min(i + batch_size, len(dataset))):
            img_tensor, label = dataset[j]
            batch_imgs.append(img_tensor)
            batch_lbls.append(label.item())

        batch_tensor = torch.stack(batch_imgs).to(device)
        embeddings = model(batch_tensor)
        embeddings = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-8)
        all_embeddings.append(embeddings.cpu().float().numpy())
        all_labels.extend(batch_lbls)

    return np.vstack(all_embeddings), np.array(all_labels)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset (test split for the index)
    print("Loading PneumoniaMNIST test set...")
    dataset = get_raw_dataset(split="test", image_size=224)
    print(f"Dataset size: {len(dataset)}")

    # Load embedding model
    model, preprocess, tokenizer, model_type = load_embedding_model(
        args.model_name, str(device)
    )

    # Extract embeddings
    if model is not None:
        embeddings, labels = extract_clip_embeddings(model, preprocess, dataset, device, args.batch_size)
    else:
        embeddings, labels = extract_timm_embeddings(dataset, device, args.batch_size)
        model_type = "timm_efficientnet"

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels: {np.unique(labels, return_counts=True)}")

    # Save
    np.save(os.path.join(args.output_dir, "embeddings.npy"), embeddings)
    np.save(os.path.join(args.output_dir, "labels.npy"), labels)

    # Save model info
    import json
    info = {"model_type": model_type, "model_name": args.model_name,
            "embedding_dim": embeddings.shape[1], "num_samples": len(labels)}
    with open(os.path.join(args.output_dir, "embedding_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nSaved embeddings and labels to {args.output_dir}")
    print(f"Model: {model_type} | Embedding dim: {embeddings.shape[1]}")

    # Also save tokenizer reference for text search
    if tokenizer is not None:
        print("Tokenizer available — text-to-image search is supported!")
    else:
        print("No tokenizer — only image-to-image search is available.")

    return model, tokenizer, model_type


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", default="outputs/task3")
    args = parser.parse_args()
    main(args)
