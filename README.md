# UG Workshop - Supervised Learning with CRFSuite
This project uses Conditional Random Fields (CRF) to perform morphological segmentation of Georgian verbs. It identifies morpheme boundaries by labeling each character as either the start of a morpheme (**S**), inside a morpheme (**I**), or a null/non-functional character (**N**).

## Project Structure

- `train.py`: The main entry point to train the CRF model using features defined in `features.py`.
- `evaluate.py`: Generates performance reports, confusion matrices, and inspects feature weights.
- `predict.py`: An interactive script to test the trained model on custom Georgian words.
- `features.py`: The "Sandbox" where linguistic features (prefixes, suffixes, and rules) are defined.
- `lib/`: Contains core logic for I/O (`io.py`), string processing (`string_utils.py`), and sequence encoding (`sequence.py`).
- `data/`: Contains `words.csv` and `forms.csv` datasets.

## Quick Start

### Option 1: Google Colab (Recommended for Workshop)
You can run this entire pipeline in the cloud without local setup. Open and copy the Google Collab notebook, then follow the instructions inside.

[Click to open](https://colab.research.google.com/drive/1GtPEtvycglagAaa1g4X1IvLl1oSBC5tS?usp=sharing)


### Option 2: Local Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/joshm21/ug-workshop2/
   cd georgian-segmenter
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

   # Interpreting the CRF Performance Report

# How to Read the Evaluate.py Results

## 1. Classification Metrics (The Table)
This section measures how well the model predicts each specific label (**I**, **N**, **S**).

*   **Precision:** Of all items the model *predicted* as a specific label, how many were actually correct? (Low precision = many False Positives).
*   **Recall:** Of all items that *actually* belonged to a label, how many did the model find? (Low recall = many False Negatives).
*   **F1-Score:** The harmonic mean of Precision and Recall. It's the best "all-around" metric, especially if you have imbalanced classes.
*   **Support:** The actual count of occurrences of that label in your test dataset. (e.g., Label 'I' appears 16,596 times).

### Summary Rows:
*   **Accuracy:** The percentage of total guesses that were correct (99.6%).
*   **Macro Avg:** The arithmetic mean of scores across all classes. It treats 'N' and 'I' as equally important, regardless of their support size.
*   **Weighted Avg:** The mean of scores weighted by the number of instances (Support). Since 'I' is the most common label, its high performance dominates this average.

---

## 2. Confusion Matrix
This shows exactly where the model is getting "confused."

*   **Rows:** Represent the **Actual** labels.
*   **Columns:** Represent the **Predicted** labels.
*   **Diagonal (Top-left to Bottom-right):** These are the correct predictions.
    *   *Example:* The model correctly predicted **S** as **S** 4,989 times.
*   **Off-Diagonal:** These are errors.
    *   *Example:* Look at Row 'I', Column 'S' (**22**). This means 22 times a token was actually an 'I', but the model mistakenly labeled it as an 'S'.

---

## 3. Most Influential Features
This explains *why* the model is making its decisions. CRF is a "white-box" model, meaning we can see the logic.

*   **Weight:** 
    *   **Positive (+) Weights:** This feature strongly suggests this label. (e.g., `trigram_next:დებ` with +6.14 heavily pushes the model to choose **S**).
    *   **Negative (-) Weights:** This feature makes the label *less* likely. (e.g., if the character is `ვ`, it is almost certainly **not** an **I** because of the -7.22 weight).
*   **Label:** The label being impacted.
*   **Feature:** The specific pattern found in the text (bigrams, trigrams, characters, etc.).

---

## Summary Verdict
The model is performing **exceptionally well** (0.99 F1-score). It handles the most frequent class ('I') almost perfectly. The lowest performance is on class 'N', but even that is at 0.98, which is very high.
