"""Quick smoke-test script to verify the EXAONE-Deep-2.4B model loads.

Run:
    python scripts/test_load_model.py --max-length 32 --prompt "Hello, world!"

The script loads the model in half precision (FP16) and generates a short
completion to ensure everything is wired correctly. It avoids heavy sampling
by default to remain memory-friendly.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
import sys

# Add project root to path so `src` can be imported regardless of cwd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402  pylint: disable=wrong-import-position
from transformers import TextStreamer  # noqa: E402

from src.model.base_model import BaseModelLoader, LoaderConfig  # noqa: E402

logging.basicConfig(level=logging.INFO)


def main() -> None:  # noqa: D401 (simple)
    parser = argparse.ArgumentParser(description="EXAONE model smoke test")
    parser.add_argument("--prompt", type=str, default="Hello, I am an AI model.")
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()

    loader = BaseModelLoader(LoaderConfig())
    start = time.time()
    model, tokenizer = loader.load_model()
    logging.info("Model loaded in %.2f s", time.time() - start)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        streamer = TextStreamer(tokenizer)
        _ = model.generate(
            **inputs,
            max_new_tokens=args.max_length,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            streamer=streamer,
        )


if __name__ == "__main__":
    main()
