import pandas as pd

# Daftar source yang valid
VALID_SOURCES = {
    "tempo", "kompas", "detik", "tribunnews", "okezone",
    "kumparan", "cnnindonesia", "cnbcindonesia", "suara",
    "liputan6", "republika", "sindonews", "antara", "bisnis"
}

rows = []
with open("clickbait-nlp/data/raw/news_headline.csv", "r", encoding="utf-8-sig") as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split(",")
        if len(parts) >= 3:
            source = parts[1].strip('"').strip().lower()
            title  = parts[2].strip('"').strip()
            # Filter: source harus valid, title minimal 10 karakter
            if source in VALID_SOURCES and len(title) > 10:
                rows.append({"source": source, "title": title})

df = pd.DataFrame(rows)
df.drop_duplicates(subset=["title"], inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"Total bersih: {len(df)} baris")
print(df["source"].value_counts())
print(df.head(10))

# Simpan
df.to_csv("data/raw/news_headline_clean.csv", index=False)
print("\nDisimpan ke data/raw/news_headline_clean.csv")