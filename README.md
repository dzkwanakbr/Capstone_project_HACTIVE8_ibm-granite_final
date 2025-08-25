# Capstone_project_HACTIVE8_ibm-granite
Melakukan perbandingan antara logistic regression dan juga ibm granite instruct (via replicate api)

# Capstone Project – Text Classification & Summarization (20 Newsgroups)

# 📌 Overview
Proyek ini bertujuan untuk membangun sistem klasifikasi teks dan ringkasan otomatis menggunakan dataset 20 Newsgroups.
Pendekatan yang digunakan:
Baseline Machine Learning → Logistic Regression + TF-IDF.
Large Language Model (LLM) → IBM Granite Instruct melalui Replicate API.
Proyek ini tidak hanya fokus pada akurasi klasifikasi, tetapi juga pada interpretabilitas model dan kemampuan summarization untuk mempermudah pemahaman teks panjang.

# 📂 Dataset
Dataset: 20 Newsgroups (Kaggle) (https://www.kaggle.com/datasets/crawford/20-newsgroups)
Jumlah kategori: 20
Jumlah total artikel: 273,060

5 Contoh kategori:
1.alt.atheism
2.comp.graphics
3.sci.space
4.talk.politics.mideast
5. rec.sport.hockey

# 🔎 Insight & Findings

📊 Logistic Regression (Baseline)
Akurasi: 44%
Macro F1: 0.43
Performa terbaik pada kategori dengan kosakata spesifik → misalnya talk.politics.mideast dan comp.os.ms-windows.misc.
Kategori mirip (misalnya antar topik agama & hardware komputer) sering tertukar.
Distribusi data tidak seimbang → kategori politik jauh lebih dominan dibanding kategori lain.

📌 Contoh Visualisasi:
Confusion Matrix → menunjukkan error terbesar antar kategori mirip.
Distribusi Artikel → kategori talk.politics.mideast paling dominan.
Wordcloud → kata dominan sesuai konteks kategori.


🤖 IBM Granite (Replicate API)
Zero-Shot Classification → mampu memberi alasan jika data tidak cukup, bukan sekadar menebak.
Summarization → menghasilkan ringkasan singkat, jelas, dan sesuai konteks.

# 🧠 AI Support Explanation
Proyek ini memanfaatkan dua pendekatan utama:
Baseline ML (Logistic Regression + TF-IDF)
Cepat & efisien.
Dapat digunakan sebagai benchmark awal.
IBM Granite Instruct (via Replicate API)
Model LLM berbasis instruksi (instruction-tuned).
Bisa melakukan klasifikasi tanpa training tambahan (zero-shot learning).
Mampu menghasilkan ringkasan teks panjang secara otomatis (abstractive summarization).
Memberikan interpretasi lebih baik (dapat menolak prediksi bila data tidak cukup).

✅ Kesimpulan
Logistic Regression → baseline dengan akurasi 44%.
IBM Granite → lebih unggul dalam reasoning & summarization.
Kombinasi keduanya memberikan keseimbangan antara kecepatan & interpretabilitas.
