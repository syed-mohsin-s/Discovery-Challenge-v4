#!/usr/bin/env python3
"""
download_model.py
=================
Downloads and caches the text embedding model locally to comply with the 
zero-network sandbox rules during Stage 3 code reproduction.
"""
import os
from sentence_transformers import SentenceTransformer

def main():
    model_name = "all-MiniLM-L6-v2"
    cache_dir = "./model_cache/all-MiniLM-L6-v2"
    
    print(f"Initializing download for {model_name}...")
    os.makedirs("./model_cache", exist_ok=True)
    
    # Download and save natively to target path
    model = SentenceTransformer(model_name)
    model.save(cache_dir)
    print(f"[SUCCESS] Model successfully cached to: {cache_dir}")
    print("You can now safely run rank.py in offline mode.")

if __name__ == "__main__":
    main()