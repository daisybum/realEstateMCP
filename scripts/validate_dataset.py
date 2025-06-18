"""Validate tokenised dataset data/hf_dataset and write report.json.

Used by DVC `validate` stage.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from datasets import DatasetDict

from src.data.quality import DataQualityValidator
from src.data.tokenizer import HFTokenizer


def main():
    ds_dir = Path("data/hf_dataset")
    out_path = Path("report.json")

    ds = DatasetDict.load_from_disk(str(ds_dir))
    validator = DataQualityValidator(ds, HFTokenizer(max_length=128))
    validator.run()
    validator.save(out_path)


if __name__ == "__main__":
    main()
