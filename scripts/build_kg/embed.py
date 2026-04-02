"""Step 5: Embed hyperedge descriptions + entity names with MedEmbed-large, build FAISS indices."""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def build_index(
    hg_path: str = "data/medical_hg.json",
    index_dir: str = "data/",
    model_name: str = "abhinand/MedEmbed-large-v0.1",
    batch_size: int = 256,
    device: str | None = None,
):
    """Build FAISS indices for hyperedge descriptions and entity names."""
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    print(f"Embedding device: {model.device}")

    print(f"Loading hypergraph from {hg_path}")
    with open(hg_path) as f:
        data = json.load(f)

    Path(index_dir).mkdir(parents=True, exist_ok=True)

    # --- Hyperedge index (primary retrieval target) ---
    hedge_ids = [h['id'] for h in data['hyperedges']]
    hedge_texts = [h['description'] for h in data['hyperedges']]
    print(f"Encoding {len(hedge_texts):,} hyperedge descriptions...")

    emb = model.encode(
        hedge_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    dim = emb.shape[1]

    idx = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
    idx.add(emb)
    faiss.write_index(idx, f"{index_dir}/index_hyperedge.bin")
    np.save(f"{index_dir}/hedge_ids.npy", np.array(hedge_ids))
    print(f"  Hyperedge index: {idx.ntotal:,} vectors, dim={dim}")

    # --- Entity index (expand retrieval via entity match) ---
    ent_names = list(data['entities'].keys())
    print(f"Encoding {len(ent_names):,} entity names...")

    emb_ent = model.encode(
        ent_names,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    idx_ent = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
    idx_ent.add(emb_ent)
    faiss.write_index(idx_ent, f"{index_dir}/index_entity.bin")
    np.save(f"{index_dir}/entity_names.npy", np.array(ent_names))
    print(f"  Entity index: {idx_ent.ntotal:,} vectors, dim={dim}")

    print("Done building indices.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build FAISS indices')
    parser.add_argument('--hg-path', default='data/medical_hg.json')
    parser.add_argument('--index-dir', default='data/')
    parser.add_argument('--model', default='abhinand/MedEmbed-large-v0.1')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    build_index(
        args.hg_path,
        args.index_dir,
        args.model,
        args.batch_size,
        args.device,
    )
