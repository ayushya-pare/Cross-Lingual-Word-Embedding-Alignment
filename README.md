
# Cross-Lingual Word Embedding Alignment (English–Hindi)

The task was to align word embeddings from two different languages (English and Hindi) into a shared space. This allows comparison between words across languages. We were required to use a supervised method (Procrustes alignment), and optionally explore an unsupervised method (as in the MUSE paper).

## Prior Knowledge

- experience working with word embeddings like Word2Vec and FastText.

- familiarity with cosine similarity, matrix multiplication, and basic linear algebra.

- used Python libraries such as NumPy, Scikit-learn, and tensorflow for ML tasks.


## New Learnings
- Procrustes alignment method using SVD.

- explored the MUSE system and its training process based on adversarial learning.

- learnt the limitations of unsupervised alignment when languages use different scripts.


## Methodology 
- Downloaded FastText embeddings for English and Hindi.

- Filtered the top 100,000 words and saved them in .vec format.

- Cleaned the bilingual dictionary from the MUSE repo.

- Implemented Procrustes alignment:

  - Extracted word vectors for training pairs

  - Normalized them

  - Used SVD to compute the optimal mapping matrix

  - Transformed Hindi embeddings into the English space

- Evaluated alignment using cosine similarity on test pairs.

- Attempted the unsupervised MUSE method (did not work).

## Results
| Method       | Precision@1 | Precision@5 |
|--------------|-------------|-------------|
| Supervised   | 0.2577      | 0.508       |
| Unsupervised | 0.0000      | 0.000       |

The supervised method worked well. The unsupervised method failed for this language pair.

The unsupervised alignment using the MUSE framework was attempted as extra credit. However, it did not work successfully in this case. Precision@1 was 0.0000. This is likely due to the mismatch between English (Latin script) and Hindi (Devanagari script), and the format MUSE expects.

## Summary
I was able to successfully implement the supervised method and evaluate it. The unsupervised method helped me understand challenges in aligning different scripts. Overall, I practiced working with embeddings, matrix operations, and evaluation of alignment quality.

The code for the unsupervised method is included only for reference.

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

### 5. Folder and Directory Structure
Organize your working directory like this:
```
project_root/
├── requirements.txt
├── README.md
├── data/
│   ├── fasttext_pretrained_vectors/
│   │   ├── en_top100k.vec
│   │   └── hi_top100k.vec
│   |── muse_bilingual_dictionary/
│   |    |── en-hi.txt
│   |    └── valid_pairs.txt
|   |── test_pairs.txt
|   |── train_pairs.txt
├── Documentation/
│   ├── documentation_and_results.pdf
├── notebooks/
│   ├── cross_lingual_words_embeddings_final.ipynb

```

### 6. (Optional) Run Unsupervised Alignment
Navigate to the cloned MUSE folder and run:
```bash
python unsupervised.py   --src_lang en   --tgt_lang hi   --src_emb data/vec/wiki.en.vec   --tgt_emb data/vec/wiki.hi.vec   --export pth   --exp_path checkpoints/   --exp_name unsup_en_hi   --cuda False
```
This is optional and does not yield useful results for Hindi script.


### 7. Translation Test Example (After Alignment)

After completing alignment, you can test some word translations like this:

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Example: Translate English words using aligned embeddings

english_test_words = ["water", "house", "school", "computer", "love"]

for word in english_test_words:
    if word not in en_vectors:
        print(f"{word} not found in English vocab.")
        continue

    en_vec = en_vectors[word]
    en_vec = en_vec / np.linalg.norm(en_vec)

    sim_scores = cosine_similarity(en_vec.reshape(1, -1), aligned_matrix)[0]
    top_indices = np.argsort(sim_scores)[::-1][:5]
    top_preds = [aligned_words[i] for i in top_indices]

    print(f"{word} → {', '.join(top_preds)}")
```

This will print the top 5 closest Hindi words from the aligned embedding space.
---
