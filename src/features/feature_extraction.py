import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ----------------------------
# 1. Load processed data
# ----------------------------
df = pd.read_csv('data/cleaned/clean_dataset.csv')   
texts = df['processed_sentence'].astype(str)

# ----------------------------
# 2. CountVectorizer (Bag of Words)
# ----------------------------
count_vectorizer = CountVectorizer()
count_features = count_vectorizer.fit_transform(texts)

df_count = pd.DataFrame(
    count_features.toarray(),
    columns=count_vectorizer.get_feature_names_out()
)
df_count.to_csv('data/features_output/features_count.csv', index=False)

print("✔ Saved: features_count.csv")

# ----------------------------
# 3. TF-IDF Features
# ----------------------------
tfidf_vectorizer = TfidfVectorizer()
tfidf_features = tfidf_vectorizer.fit_transform(texts)

df_tfidf = pd.DataFrame(
    tfidf_features.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)
df_tfidf.to_csv('data/features_output/features_tfidf.csv', index=False)

print("✔ Saved: features_tfidf.csv")

# ----------------------------
# 4. Binary Features (presence/absence)
# ----------------------------
binary_vectorizer = CountVectorizer(binary=True)
binary_features = binary_vectorizer.fit_transform(texts)

df_binary = pd.DataFrame(
    binary_features.toarray(),
    columns=binary_vectorizer.get_feature_names_out()
)
df_binary.to_csv('data/features_output/features_binary.csv', index=False)

print("✔ Saved: features_binary.csv")

print("🎉 Feature Extraction Completed Successfully!")
