#!/usr/bin/env python
"""
Train BiLSTM model only
"""
import sys
from src.ml_utils import train_and_save_models

if __name__ == "__main__":
    print("Training BiLSTM model...")
    train_and_save_models()
    print("✅ BiLSTM model training complete!")
    sys.exit(0)
