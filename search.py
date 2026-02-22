"""
Task 3: Build FAISS vector index and perform image-to-image / text-to-image retrieval.

Usage:
    # Build index
    python search.py build --embeddings_dir outputs/task3

    # Image-to-image search
    python search.py search --query_idx 42 --k 5 --embeddings_dir outputs/task3

    # Text-to-image search  
    python search.py text_search --query "bilateral lung consolidation pneumonia" --k 5 \
        --embeddings_dir outputs/task3

    # Evaluate Precision@k
    python search.py evaluate --k 5 --embeddings_dir outputs/task3
"""

import argparse
import os
import json
import numpy as np
import torch
import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.dataset import get_raw_dataset

CLASS_NAMES = ["Normal", "Pneumonia"]


# ─── Index Management ────────────────────────────────────────────────────────

def build_index(embeddings: np.ndarray, output_dir: str, index_type: str = "flat"):
    """
    Build and save a FAISS index.
    Uses IndexFlatIP (Inner Product) for cosine similarity on L2-normalized vectors.
    """
    d = embeddings.shape[1]
    print(f"Building FAISS index: dim={d}, n={len(embeddings)}, type={index_type}")

    if index_type == "flat":
        index = faiss.IndexFlatIP(d)
    elif index_type == "ivf":
        quantizer = faiss.IndexFlatIP(d)
        nlist = min(100, len(embeddings) // 10)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings.astype(np.float32))
        index.nprobe = 10
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    index.add(embeddings.astype(np.float32))
    path = os.path.join(output_dir, "faiss_index.bin")
    faiss.write_index(index, path)
    print(f"Index saved: {path} | Total vectors: {index.ntotal}")
    return index


def load_index(output_dir: str):
    path = os.path.join(output_dir, "faiss_index.bin")
    index = faiss.read_index(path)
    print(f"Loaded index: {index.ntotal} vectors, dim={index.d}")
    return index


# ─── Image-to-Image Search ────────────────────────────────────────────────────

def image_to_image_search(index, query_embedding: np.ndarray, k: int = 5):
    """
    Retrieve top-k similar images using cosine similarity (inner product on L2-normed vectors).
    """
    q = query_embedding.reshape(1, -1).astype(np.float32)
    scores, indices = index.search(q, k + 1)  # +1 to exclude self
    # Exclude the query itself (score=1.0 if it's in the index)
    results = [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])
               if idx != -1]
    return results[1:k+1] if len(results) > k else results[:k]


# ─── Text-to-Image Search ─────────────────────────────────────────────────────

def text_to_image_search(index, query_text: str, model_type: str, k: int = 5, device: str = "cpu"):
    """
    Encode query text with the same CLIP model and retrieve relevant images.
    """
    try:
        import open_clip
        if model_type == "biomedclip":
            model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        else:
            model_name = "ViT-B-32"

        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=("openai" if model_type == "clip" else None)
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(device).eval()

        with torch.no_grad():
            tokens = tokenizer([query_text]).to(device)
            text_embedding = model.encode_text(tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            text_embedding = text_embedding.cpu().float().numpy()

        scores, indices = index.search(text_embedding, k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx != -1]

    except Exception as e:
        print(f"Text search failed: {e}")
        return []


# ─── Evaluation: Precision@k ──────────────────────────────────────────────────

def evaluate_precision_at_k(index, embeddings: np.ndarray, labels: np.ndarray,
                             k_values=(1, 5, 10), n_queries: int = 200):
    """
    Compute Precision@k: fraction of top-k retrieved items sharing the query label.
    Uses a random sample of queries from the index.
    """
    np.random.seed(42)
    query_indices = np.random.choice(len(embeddings), n_queries, replace=False)
    results = {k: [] for k in k_values}

    for q_idx in tqdm(query_indices, desc="Evaluating P@k"):
        q_emb = embeddings[q_idx].reshape(1, -1).astype(np.float32)
        max_k = max(k_values)
        scores, retrieved = index.search(q_emb, max_k + 1)
        retrieved = [int(i) for i in retrieved[0] if i != -1 and i != q_idx]

        q_label = labels[q_idx]
        for k in k_values:
            top_k = retrieved[:k]
            if len(top_k) == 0:
                results[k].append(0.0)
                continue
            precision = np.mean([labels[i] == q_label for i in top_k])
            results[k].append(precision)

    summary = {f"P@{k}": float(np.mean(results[k])) for k in k_values}
    return summary


# ─── Visualization ────────────────────────────────────────────────────────────

def visualize_retrieval(query_idx, retrieved, embeddings, labels, dataset,
                        output_dir, title="Image-to-Image Retrieval"):
    """Plot query image alongside retrieved results."""
    def get_img(idx):
        img_t, _ = dataset[idx]
        img = img_t.permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)[:, :, 0]
        return img

    k = len(retrieved)
    fig, axes = plt.subplots(1, k + 1, figsize=(3 * (k + 1), 3.5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # Query
    axes[0].imshow(get_img(query_idx), cmap="gray")
    axes[0].set_title(f"QUERY\n{CLASS_NAMES[labels[query_idx]]}", fontweight="bold", color="blue")
    axes[0].axis("off")
    for ax in axes[0].spines.values():
        ax.set_visible(True); ax.set_color("blue"); ax.set_linewidth(3)

    # Retrieved
    for j, (ret_idx, score) in enumerate(retrieved):
        match = labels[ret_idx] == labels[query_idx]
        axes[j + 1].imshow(get_img(ret_idx), cmap="gray")
        axes[j + 1].set_title(
            f"#{j+1}: {CLASS_NAMES[labels[ret_idx]]}\nSim: {score:.3f}",
            color="green" if match else "red", fontsize=8
        )
        axes[j + 1].axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, f"retrieval_query{query_idx}.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def visualize_precision_at_k(precision_dict, output_dir):
    k_vals = [int(k.split("@")[1]) for k in precision_dict]
    p_vals = list(precision_dict.values())

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([str(k) for k in k_vals], p_vals, color="#2196F3", width=0.5, edgecolor="white")
    ax.set_xlabel("k"); ax.set_ylabel("Precision@k")
    ax.set_title("Retrieval Precision@k — Image-to-Image Search")
    ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, p_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.3f}", ha="center", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "precision_at_k.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─── Main CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CBIR System for PneumoniaMNIST")
    subparsers = parser.add_subparsers(dest="command")

    # Build
    bp = subparsers.add_parser("build")
    bp.add_argument("--embeddings_dir", default="outputs/task3")
    bp.add_argument("--index_type", default="flat", choices=["flat", "ivf"])

    # Search
    sp = subparsers.add_parser("search")
    sp.add_argument("--query_idx", type=int, default=0)
    sp.add_argument("--k", type=int, default=5)
    sp.add_argument("--embeddings_dir", default="outputs/task3")

    # Text search
    tp = subparsers.add_parser("text_search")
    tp.add_argument("--query", type=str, default="bilateral pneumonia consolidation")
    tp.add_argument("--k", type=int, default=5)
    tp.add_argument("--embeddings_dir", default="outputs/task3")

    # Evaluate
    ep = subparsers.add_parser("evaluate")
    ep.add_argument("--k_values", nargs="+", type=int, default=[1, 5, 10])
    ep.add_argument("--n_queries", type=int, default=200)
    ep.add_argument("--embeddings_dir", default="outputs/task3")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    emb_dir = args.embeddings_dir
    embeddings = np.load(os.path.join(emb_dir, "embeddings.npy"))
    labels = np.load(os.path.join(emb_dir, "labels.npy"))
    with open(os.path.join(emb_dir, "embedding_info.json")) as f:
        info = json.load(f)
    model_type = info["model_type"]

    if args.command == "build":
        build_index(embeddings, emb_dir, args.index_type)

    elif args.command == "search":
        index = load_index(emb_dir)
        dataset = get_raw_dataset("test", image_size=224)
        retrieved = image_to_image_search(index, embeddings[args.query_idx], args.k)
        print(f"\nQuery idx={args.query_idx} | Label={CLASS_NAMES[labels[args.query_idx]]}")
        for i, (idx, score) in enumerate(retrieved):
            match = "✓" if labels[idx] == labels[args.query_idx] else "✗"
            print(f"  #{i+1}: idx={idx} | Label={CLASS_NAMES[labels[idx]]} | Score={score:.4f} {match}")
        visualize_retrieval(args.query_idx, retrieved, embeddings, labels, dataset, emb_dir)

    elif args.command == "text_search":
        index = load_index(emb_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dataset = get_raw_dataset("test", image_size=224)
        retrieved = text_to_image_search(index, args.query, model_type, args.k, device)
        print(f"\nText query: '{args.query}'")
        for i, (idx, score) in enumerate(retrieved):
            print(f"  #{i+1}: idx={idx} | Label={CLASS_NAMES[labels[idx]]} | Score={score:.4f}")
        if retrieved:
            visualize_retrieval(retrieved[0][0], retrieved[1:], embeddings, labels, dataset, emb_dir,
                                title=f"Text Search: '{args.query}'")

    elif args.command == "evaluate":
        index = load_index(emb_dir)
        precision = evaluate_precision_at_k(index, embeddings, labels,
                                             args.k_values, args.n_queries)
        print("\n" + "="*40)
        print("PRECISION@K RESULTS")
        print("="*40)
        for key, val in precision.items():
            print(f"  {key}: {val:.4f}")

        with open(os.path.join(emb_dir, "precision_at_k.json"), "w") as f:
            json.dump(precision, f, indent=2)
        visualize_precision_at_k(precision, emb_dir)


if __name__ == "__main__":
    main()
