
# Assignment Report: Cross-Lingual Word Embedding Alignment

## Understanding the Assignment
The task was to align word embeddings from two different languages (English and Hindi) into a shared space. This allows comparison between words across languages. We were required to use a supervised method (Procrustes alignment), and optionally explore an unsupervised method (as in the MUSE paper).

## Prior Knowledge I Used
- I had previous experience working with word embeddings like Word2Vec and FastText.
- I was familiar with cosine similarity, matrix multiplication, and basic linear algebra.
- I had used Python libraries such as NumPy, Scikit-learn, and PyTorch for ML tasks.

## What I Learned New
- I learned about the Procrustes alignment method using SVD.
- I explored the MUSE system and its training process based on adversarial learning.
- I learned the limitations of unsupervised alignment when languages use different scripts.
- I practiced reproducible research methods by avoiding hardcoded paths and documenting steps.

## My Approach to the Assignment
1. Downloaded FastText embeddings for English and Hindi.
2. Filtered the top 100,000 words and saved them in `.vec` format.
3. Cleaned the bilingual dictionary from the MUSE repo.
4. Implemented Procrustes alignment:
   - Extracted word vectors for training pairs
   - Normalized them
   - Used SVD to compute the optimal mapping matrix
   - Transformed Hindi embeddings into the English space
5. Evaluated alignment using cosine similarity on test pairs.
6. Attempted the unsupervised MUSE method (did not work due to script mismatch).

## Results
| Method       | Precision@1 | Precision@5 |
|--------------|-------------|-------------|
| Supervised   | 0.2577      | 0.508       |
| Unsupervised | 0.0000      | 0.000       |

The supervised method worked well. The unsupervised method failed for this language pair.

## Difficult Parts
- Cleaning the bilingual dictionary and matching it with FastText vocab.
- Making sure the `.vec` format was correct (with headers).
- Running MUSE with the right configuration (required manual fixes).

## Summary
This was a good hands-on task to understand cross-lingual alignment. I was able to successfully implement the supervised method and evaluate it. The unsupervised method helped me understand challenges in aligning different scripts. Overall, I practiced working with embeddings, matrix operations, and evaluation of alignment quality.

---

## How to Run the Code (Notebook)

### 1. Clone the Repository
Clone the MUSE repository (for optional unsupervised training):
```bash
git clone https://github.com/facebookresearch/MUSE.git
```

### 2. Install Dependencies
Make sure you're using **Python 3.9 or 3.10**. Then install required packages:
```bash
pip install -r requirements.txt
```

### 3. Download FastText Models
Run the following code in your notebook:
```python
import fasttext.util
fasttext.util.download_model('en', if_exists='ignore')
fasttext.util.download_model('hi', if_exists='ignore')
```
This will download pretrained English and Hindi word vectors (`.bin` format).

### 4. Run the Notebook
Open the notebook in Jupyter or Google Colab and run each section step by step.
- The notebook is written without functions, using loops and simple code.
- It covers vocabulary filtering, Procrustes alignment, and evaluation.

### 5. Folder and Directory Structure
Organize your working directory like this:
```
project_root/
├── notebook.ipynb
├── requirements.txt
├── data/
│   ├── vec/
│   │   ├── wiki.en.vec
│   │   └── wiki.hi.vec
│   └── dictionaries/
│       └── en-hi.txt
├── checkpoints/  (optional for MUSE output)
└── MUSE/          (if using MUSE unsupervised code)
```

### 6. (Optional) Run Unsupervised Alignment
Navigate to the cloned MUSE folder and run:
```bash
python unsupervised.py   --src_lang en   --tgt_lang hi   --src_emb data/vec/wiki.en.vec   --tgt_emb data/vec/wiki.hi.vec   --export pth   --exp_path checkpoints/   --exp_name unsup_en_hi   --cuda False
```
This is optional and does not yield useful results for Hindi script.

---
