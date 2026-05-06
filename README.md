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
You can run this entire pipeline in the cloud without local setup:
[Click to open](https://colab.research.google.com/drive/1GtPEtvycglagAaa1g4X1IvLl1oSBC5tS?usp=sharing)

1. Run the **Bootstrap** cell to install dependencies.
2. Edit `features.py` directly in the notebook cell.
3. Run the **Train** cell to see your model learn.

### Option 2: Local Setup
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd georgian-segmenter

## 🎓 How to Read Your Model Report

#### 1. The Metrics
* **Precision:** "When the model says a boundary starts (S), how often is it right?"
* **Recall:** "Out of all the actual boundaries in the data, how many did the model find?"
* **F1-Score:** The harmonic mean of Precision and Recall. It is your "Overall Grade."
* **Support:** The number of times that specific label appeared in the test set.

#### 2. The Confusion Matrix
This table shows you exactly which labels are being confused for one another.
* **Rows:** The **Actual** labels (Ground Truth).
* **Columns:** The **Predicted** labels.
* *Interpretation:* Look at the diagonal line from top-left to bottom-right. You want the highest numbers to be there. Off-diagonal numbers show you exactly where the model is "confused."

#### 3. Feature Weights
This is the "Explainable AI" section. 
* **Positive Weights:** These features strongly **encourage** the model to choose that label.
* **Negative Weights:** These features **discourage** the model from choosing that label.
* *Tip:* If you see a feature with a high weight that doesn't make linguistic sense, you might have a "leakage" issue or a data cleaning problem!
