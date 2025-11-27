# NLP Project

## **1. Project Overview**

This project applies standard NLP preprocessing steps to a linguistic dataset and then computes sentence probabilities using an **N-Gram (Bigram) Language Model** with the **Markov assumption**.

The project satisfies the requirements of the first evaluation:

1. **Data Selection & Preprocessing**
2. **N-Gram Sentence Probability Calculation**

---

## **2. Dataset Description (UD_English-EWT)**

The dataset used is **Universal Dependencies English Web Treebank (EWT)** in **CONLL-U format**.

### **Why I Chose This Dataset**

* A standard benchmark used in NLP research.
* Already segmented into **sentences**, ready for processing.
* Contains **lemmas**, POS tags, morphological features → ideal for preprocessing.
* Works perfectly with Python libraries like `conllu`.
* High quality, well-structured, and widely used.

---

## **3. Understanding the CONLL-U Format**

Each sentence is represented by a block of rows. Each row has 10 columns:

| Column     | Meaning                    | Example       |
| ---------- | -------------------------- | ------------- |
| **ID**     | Token index                | 1             |
| **FORM**   | Word as it appears in text | killed        |
| **LEMMA**  | Base form                  | kill          |
| **UPOS**   | Universal POS tag          | VERB          |
| **XPOS**   | Language-specific POS      | VBD           |
| **FEATS**  | Morphological features     | Tense=Past    |
| **HEAD**   | Head token index           | 7             |
| **DEPREL** | Dependency relation        | obj           |
| **DEPS**   | Enhanced dependencies      | _             |
| **MISC**   | Other info                 | SpaceAfter=No |

This structure makes it easy to extract (FORM, LEMMA) pairs.

---

## **4. Preprocessing Steps**

### ✔ Lower casing

Convert all tokens to lowercase.

### ✔ Lemmatization

Use the `lemma` column from the dataset.

### ✔ Remove punctuation

Using regex to detect punctuation-only tokens.

### ✔ Remove numbers

Remove tokens that are integers or floats.

### ✔ Remove stopwords

Using:

```python
from nltk.corpus import stopwords
```

### ✔ Remove empty tokens

Ensure only clean tokens remain.

**Example:**

**Before:**
`American forces killed Shaikh Abdullah al-Ani, near the Syrian border.`

**After:**
`american force kill shaikh abdullah ani syrian border`

---

## **5. N-Gram Language Model**

We compute probabilities using **Bigram Model** with **Add-1 Smoothing**.

### **Sentence Probability (Bigram Model)**

$$
P(w_1, w_2, \ldots, w_n)
= P(w_1) \times \prod_{i=2}^{n} P(w_i \mid w_{i-1})
$$

### **Bigram Conditional Probability (Add-1 Smoothing)**

$$
P(w_i \mid w_{i-1})
= \frac{\text{Count}(w_{i-1},, w_i) + 1}{\text{Count}(w_{i-1}) + V}
$$

Where:

* **Count(wᵢ₋₁, wᵢ)** = frequency of the bigram
* **Count(wᵢ₋₁)** = frequency of the previous word
* **V** = vocabulary size
* Add-1 smoothing prevents zero probabilities

---

## **6. Output Format (Before & After Preprocessing)**

The script prints each sentence like this:

```
=====================================================
Sentence #1 (Original):
American forces killed Shaikh Abdullah al-Ani, near the Syrian border.

After Preprocessing:
american force kill shaikh abdullah ani syrian border

Bigram Probability:
2.481e-07
=====================================================
```

This format is used for all 10 sentences.

---

## **7. Project Structure**

```
NLP_PROJECT/
│
├── .vscode/                  
│
├── data/                     
│   ├── features/             
│   │   ├── features_binary.csv    
│   │   ├── features_count.csv     
│   │   └── features_tfidf.csv     
│   │
│   ├── preprocessing_output/            
│   │   └── sentence_probs.csv
│   │
│   ├── cleaned/            
│   │   └── clean_dataset.csv     
│   │
│   └── raw/                  
│       └── en_ewt-ud-train.conllu  
│
├── src/                      
│   ├── features/             
│   │   └── feature_extraction.py
│   │   └── word2vec_preparation.py
│   │
│   └── preprocessing/        
│       └── preprocess_and_markov.py  
│                   
│
└── README.md                 
```

---

## **8. Feature Extraction**

After preprocessing, the **feature extraction module** converts cleaned text into numerical vectors suitable for machine learning and NLP tasks.

### Objectives

* Generate multiple vectorized representations of the text
* Compare how different feature types capture linguistic characteristics
* Save feature matrices in reusable CSV files
* Prepare data for modeling, clustering, or statistical analysis

### Extracted Feature Types

1. **Count Vectorization** – word frequency counts (`features_count.csv`)
2. **Binary Vectorization** – token presence/absence (`features_binary.csv`)
3. **TF-IDF** – weighted importance across corpus (`features_tfidf.csv`)

**Implementation Script:** `src/features/feature_extraction.py`

> All matrices share the same vocabulary to ensure consistency.

**Why These Features?**

| Feature Type | Strength                   | Best Use Case              |
| ------------ | -------------------------- | -------------------------- |
| Count        | Simple & interpretable     | Baseline models            |
| Binary       | Removes frequency bias     | When word presence matters |
| TF-IDF       | Highlights important words | Most ML & NLP tasks        |

---

## **9. Word2Vec Feature Extraction**

This section explains how word-level features are generated using the Word2Vec Skip-Gram model to support POS and NER classification tasks.

### 1. Loading the Processed Dataset
The cleaned dataset is loaded from the `data/cleaned` directory.  
It must contain:
- `processed_sentence`: the preprocessed and tokenized text  
- `label`: the POS or NER tag associated with each word  

### 2. Tokenization and Label Preparation
Each processed sentence is split into tokens.  
Label sequences are also converted into lists to ensure perfect alignment between tokens and their corresponding tags.  
The outcome of this step is:
- A list of tokenized sentences  
- A list of aligned label sequences  

### 3. Training the Word2Vec Model
A Word2Vec Skip-Gram model is trained on the entire collection of tokenized sentences.  
The model configuration includes:
- Vector size: 100  
- Window size: 5  
- Minimum word count: 1  
- Workers: 4  

The trained embedding model is saved as:  
`data/embeddings/word2vec.model`

### 4. Creating Embedding Features
Each word in every sentence is converted into its Word2Vec embedding vector, and each vector is paired with its correct POS/NER tag.  
This step produces two feature arrays:
- `X`: the matrix of word embeddings  
- `y`: the corresponding labels for each word  

### 5. Saving the Final Feature Files
The generated feature matrices are saved as:
- `data/embeddings/X_word2vec.npy`
- `data/embeddings/y_word2vec.npy`

### These files are later used as inputs for machine-learning models such as SVM, Logistic Regression, and various neural network classifiers.
---


## **10. How to Run**

### Install dependencies

```bash
pip install nltk conllu tqdm
```

### Run preprocessing and Markov probability script

```bash
python src/preprocessing/preprocess_and_markov.py
```

### Run feature extraction

```bash
python src/features/feature_extraction.py
```

---

## **11. FAQs & Important Notes**

* **Why this dataset?**
  High-quality, linguistically annotated dataset ideal for NLP preprocessing and modeling.

* **Why lemmatization instead of stemming?**
  Lemmatization preserves meaningful base forms (kill, run, eat).

* **What is the Markov assumption?**
  Each word depends only on the previous word (bigram).

* **Why remove stopwords?**
  They add noise and do not contribute to sentence meaning.

* **What is feature extraction?**
  Converts tokens into structured representations for ML: binary, count-based, and TF-IDF features.

---

## **12. Conclusion**

This project demonstrates:

* Full NLP preprocessing pipeline
* Clean handling of CONLL-U datasets
* Construction of **unigram & bigram** models
* Probability calculation for **10 sentences**
* Extraction of binary, count-based, and TF-IDF features
* Clean formatted output

The project meets all required evaluation criteria.

---
