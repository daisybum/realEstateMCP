"""Model loading utilities for Task-Master AI.

Currently exposes `BaseModelLoader` for loading the EXAONE-Deep-2.4B model with
memory-efficient settings (FP16, device-map auto, gradient checkpointing, LoRA
prep, etc.).
"""
