import pandas as pd
import subprocess
import os

# Step 1: Pseudo label news_headline_clean.csv ke file sementara
# Ganti nama output sementara
import sys
sys.path.insert(0, 'src')
from pseudo_labeling import PseudoLabeler

labeler = PseudoLabeler(model_type="indobert")
df_news = labeler.label_from_csv(
    "data/raw/news_headline_clean.csv",
    title_col="title",
    save=False  # jangan simpan dulu
)
print(f"News headline pseudo labeled: {len(df_news)}")

# Step 2: Load yang sudah ada (scraped)
df_scraped = pd.read_csv("data/annotated/csv/pseudo_labeled.csv")
print(f"Scraped pseudo labeled: {len(df_scraped)}")

# Step 3: Merge keduanya
merged = pd.concat([df_scraped, df_news], ignore_index=True)
before = len(merged)
merged.drop_duplicates(subset=["title"], inplace=True)
merged.reset_index(drop=True, inplace=True)

print(f"\nTotal gabungan : {len(merged)} (removed {before - len(merged)} duplikat)")
print(f"Clickbait      : {merged['label'].sum()}")
print(f"Non-clickbait  : {(merged['label']==0).sum()}")

# Step 4: Simpan
merged.to_csv("data/annotated/csv/pseudo_labeled.csv", index=False)
print("Saved -> data/annotated/csv/pseudo_labeled.csv")