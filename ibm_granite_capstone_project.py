# ============================================================
# 📌 Capstone Project: Multi-Class Text Classification and Summarization
#     on 20 Newsgroups Dataset Using IBM Granite (Replicate API)
# ============================================================

# === 1. Setup Environment ===
!pip install datasets scikit-learn transformers torch wordcloud matplotlib seaborn kaggle replicate langchain-community

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import os

# === 2. Import Dataset dari Kaggle ===
from google.colab import files
files.upload()  # upload kaggle.json

!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d crawford/20-newsgroups -p /content/
!unzip -q /content/20-newsgroups.zip -d /content/20news/

# === 3. Load Data ===
data = []
labels = []
base_path = "/content/20news"

for file in os.listdir(base_path):
    if file.endswith(".txt") and file != "list.csv":
        category = file.replace(".txt", "")
        with open(os.path.join(base_path, file), errors="ignore") as f:
            texts = f.readlines()
            for text in texts:
                if text.strip():
                    data.append(text.strip())
                    labels.append(category)

df = pd.DataFrame({"text": data, "label": labels})
print(df.head())
print("Jumlah kategori:", df['label'].nunique())

# Distribusi kategori
plt.figure(figsize=(10,6))
sns.countplot(y=df['label'], order=df['label'].value_counts().index)
plt.title("Distribusi Artikel per Kategori")
plt.show()

# === 4. Preprocessing & Split Data ===
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# TF-IDF
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# === 5. Baseline Model: Logistic Regression ===
model_lr = LogisticRegression(max_iter=2000)
model_lr.fit(X_train_tfidf, y_train)
y_pred = model_lr.predict(X_test_tfidf)

print("=== Logistic Regression Report ===")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model_lr.classes_)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=False, cmap="Blues",
            xticklabels=model_lr.classes_,
            yticklabels=model_lr.classes_)
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# === 6. IBM Granite via Replicate API ===
from langchain_community.llms import Replicate
from google.colab import userdata

# Ambil API token dari Colab Secrets (set dulu di Runtime → Secrets)
api_token = userdata.get('api_token')
os.environ["REPLICATE_API_TOKEN"] = api_token

# Pilih model Granite
model = "ibm-granite/granite-3.3-8b-instruct"
llm = Replicate(model=model, replicate_api_token=api_token)

# Fungsi Classification
def granite_classify(text, categories):
    prompt = f"Classify the following text into one of these categories: {', '.join(categories)}.\n\nText:\n{text}"
    return llm.invoke(prompt)

# Fungsi Summarization
def granite_summarize(text):
    prompt = f"Summarize this text in 3-4 sentences:\n\n{text}"
    return llm.invoke(prompt)

# === 7. Contoh Prediksi Granite ===
sample_text = df['text'].iloc[0][:500]
categories = df['label'].unique().tolist()[:5]  # coba 5 kategori dulu

print("=== Contoh Prediksi IBM Granite ===")
print(granite_classify(sample_text, categories))

print("\n=== Contoh Summarization ===")
print(granite_summarize(sample_text))

# === 8. Wordcloud per Kategori ===
category = "talk.politics.misc"
text_cat = " ".join(df[df["label"]==category]["text"].values)

wc = WordCloud(width=800, height=400, background_color="white", max_words=200).generate(text_cat)
plt.figure(figsize=(10,6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title(f"Wordcloud for {category}")
plt.show()

