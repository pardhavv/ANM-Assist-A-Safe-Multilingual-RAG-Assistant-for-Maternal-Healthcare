"""
Fine-tune the bi-encoder embedding model on domain (query, passage) pairs.

What's actually being trained here, and why this is the right thing to train:
    The LLM (Gemini, via API) is not fine-tuned in this project — that's neither
    necessary nor available through the standard API. What IS trainable, and
    what most directly improves RAG quality, is the embedding model that
    powers retrieval. A base multilingual embedding model is trained on
    generic web-scale similarity, not on "does this ANM's phrasing of a
    symptom match this specific guideline passage" — a domain gap this
    fine-tune targets directly.

Method: MultipleNegativesRankingLoss (in-batch negatives).
    For each (query, positive_passage) pair in a batch, every OTHER passage
    in that same batch is treated as a negative example, automatically. This
    needs no explicit negative mining/labeling — just positive pairs — which
    is exactly the kind of data you can hand-author quickly (as in
    training_data/embedding_train_pairs.json) rather than needing labeled
    hard negatives. It's the standard choice for small-scale bi-encoder
    fine-tuning on domain data.

Usage:
    python -m app.training.finetune_embeddings
    # writes the fine-tuned model to models/finetuned-embedder/
"""
import json
import os

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

from app.config.settings import EMBEDDING_MODEL, BASE_DIR

TRAIN_PAIRS_PATH = os.path.join(BASE_DIR, "training_data", "embedding_train_pairs.json")
OUTPUT_MODEL_DIR = os.path.join(BASE_DIR, "models", "finetuned-embedder")


def load_training_examples(path: str = TRAIN_PAIRS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    return [InputExample(texts=[p["query"], p["passage"]]) for p in pairs]


def finetune(
    base_model_name: str = EMBEDDING_MODEL,
    epochs: int = 8,
    batch_size: int = 8,
    output_dir: str = OUTPUT_MODEL_DIR,
):
    print(f"Loading base embedding model: {base_model_name}")
    model = SentenceTransformer(base_model_name)

    examples = load_training_examples()
    print(f"Loaded {len(examples)} training pairs from {TRAIN_PAIRS_PATH}")

    # Small dataset -> small batch size, since MultipleNegativesRankingLoss
    # needs batch_size >= 2 and treats every other item in the batch as a
    # negative; too large a batch relative to dataset size means too few
    # gradient steps per epoch.
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_dataloader) * epochs * 0.1)
    print(f"Fine-tuning for {epochs} epochs, batch_size={batch_size}, warmup_steps={warmup_steps}")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        output_path=output_dir,
    )
    print(f"Fine-tuned model saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    finetune()
