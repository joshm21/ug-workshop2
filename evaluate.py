import pandas as pd
from sklearn_crfsuite import metrics
from sklearn.metrics import confusion_matrix
from lib.io import load_artifacts, get_latest_run_dir


def run_evaluation(model_dir: str):
    """
    Evaluates the trained CRF model using the held-out test data.
    Displays:
    1. Classification Report (P/R/F1)
    2. Confusion Matrix (Label mix-ups)
    3. Top Feature Weights (Explainability)
    """

    # 1. Load data using our IO helper
    try:
        crf, X_test, y_test = load_artifacts(model_dir)
    except FileNotFoundError as e:
        print(f"Evaluation Error: {e}")
        return

    # 2. Generate Predictions for the unseen input data (X_test)
    # The CRF takes a list of feature dictionaries and returns a list of labels
    y_pred = crf.predict(X_test)
    labels = list(crf.classes_)

    # 3. Standard Classification Report
    # flat_classification_report treats the sequence as a flat list of tokens
    print("\n" + "="*45)
    print("        CRF PERFORMANCE REPORT")
    print("="*45)
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=labels, digits=3))

    # 4. Detailed Confusion Matrix
    # Shows which labels (I, S, N) are being confused for one another
    print("\n[1] Confusion Matrix:")
    y_true_flat = [label for seq in y_test for label in seq]
    y_pred_flat = [label for seq in y_pred for label in seq]

    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    print(df_cm.to_string())

    # 5. Top Feature Weights
    # This is the 'Explainability' section to see what
    # the model actually learned about Georgian morphology.
    print(f"\n[2] Top 15 Most Influential Features:")

    # State features map (attribute, label) -> weight
    features = crf.state_features_
    # Sort by absolute weight (magnitude) to see the strongest signals
    sorted_features = sorted(
        features.items(), key=lambda x: abs(x[1]), reverse=True)

    print(f"{'Weight':>10} | {'Label':<6} | {'Feature'}")
    print("-" * 45)
    for (attr, label), weight in sorted_features[:15]:
        print(f"{weight:10.4f} | {label:<6} | {attr}")


if __name__ == "__main__":
    try:
        latest = get_latest_run_dir()
        run_evaluation(latest)
    except Exception as e:
        print(e)
