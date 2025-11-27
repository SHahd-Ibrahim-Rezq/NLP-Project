import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import os

# -------------------------------
# 1. Load processed dataset
# -------------------------------
df = pd.read_csv(r"data/cleaned/clean_dataset.csv")

# For NER/POS tasks, 'label' column must exist
if 'label' not in df.columns:
    raise ValueError("Dataset must have 'label' column with POS or NER tags for each word!")

# -------------------------------
# 2. Clean and tokenize sentences
# -------------------------------
def prepare_tokens(text):
    text = str(text).strip()
    # Remove brackets [] if present
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    # Split sentence into words
    tokens = text.strip().split()
    return tokens

# Apply tokenization
df['tokens'] = df['processed_sentence'].apply(prepare_tokens)
# Convert label strings to lists if necessary
df['labels'] = df['label'].apply(lambda x: eval(str(x)) if isinstance(x, str) else x)

# Prepare lists of sentences and labels
sentences = df['tokens'].tolist()
labels = df['labels'].tolist()

# -------------------------------
# 3. Train Word2Vec
# -------------------------------
embedding_size = 100
window_size = 5
min_count = 1
workers = 4

w2v_model = Word2Vec(
    sentences,
    vector_size=embedding_size,
    window=window_size,
    min_count=min_count,
    workers=workers,
    sg=1  # skip-gram
)

# Create directory and save the model
os.makedirs("data/embeddings", exist_ok=True)
w2v_model.save("data/embeddings/word2vec.model")
print("✔ Word2Vec model saved: data/embeddings/word2vec.model")

# -------------------------------
# 4. Prepare embeddings for each word
# -------------------------------
X = []
y = []

for sent, lab in zip(sentences, labels):
    for word, tag in zip(sent, lab):
        if word in w2v_model.wv:
            X.append(w2v_model.wv[word])
            y.append(tag)

X = np.array(X)
y = np.array(y)

# -------------------------------
# 5. Save embeddings for training
# -------------------------------
np.save("data/embeddings/X_word2vec.npy", X)
np.save("data/embeddings/y_word2vec.npy", y)
print(f"✔ Saved embeddings: X_word2vec.npy ({X.shape}), y_word2vec.npy ({y.shape})")
