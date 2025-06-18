"""Build HuggingFace Dataset and save to data/hf_dataset.

This script is used by the DVC `build_dataset` stage.
"""
from pathlib import Path
import sys, os

# Ensure project root is on sys.path for `import src`
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.data.preprocess import Preprocessor
from src.data.tokenizer import HFTokenizer
from src.data.dataset_builder import DatasetBuilder


def main():
    builder = DatasetBuilder(
        jsonl_path=Path("dataset/weolbu_posts.jsonl"),
        tokenizer=HFTokenizer(max_length=128),
        preprocessor=Preprocessor(),
    )
    builder.build(save_dir="data/hf_dataset")


if __name__ == "__main__":
    main()
