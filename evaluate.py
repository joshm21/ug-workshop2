print('starting...')
from lib.io import load_artifacts, get_latest_run_dir, save_evaluation
from sklearn.metrics import confusion_matrix
from sklearn_crfsuite import metrics
import pandas as pd


def run_evaluation(model_dir=None):
    """
    Evaluates the trained CRF model using the held-out test data.
    Displays:
    1. Classification Report (P/R/F1)
    2. Confusion Matrix (Label mix-ups)
    3. Top Feature Weights (Explainability)
    """

    if model_dir is None:
        try:
            model_dir = get_latest_run_dir()
        except FileNotFoundError as e:
            print(e)
            return

    print(f"--- Evaluating Model: {model_dir} ---")

    # 1. Load data using our IO helper
    try:
        crf, X_test, y_test = load_artifacts(model_dir)
    except Exception as e:
        print(f"Evaluation Error: {e}")
        return

    # 2. Generate Predictions for the unseen input data (X_test)
    # The CRF takes a list of feature dictionaries and returns a list of labels
    y_pred = crf.predict(X_test)
    labels = list(crf.classes_)

    # --- Start building the report string ---
    output = []

    # 3. Standard Classification Report
    # flat_classification_report treats the sequence as a flat list of tokens
    output.append("\n" + "="*45)
    output.append("        CRF PERFORMANCE REPORT")
    output.append("="*45)
    output.append(metrics.flat_classification_report(
        y_test, y_pred, labels=labels, digits=3))

    # 4. Detailed Confusion Matrix
    # Shows which labels (I, S, N) are being confused for one another
    output.append("\n[1] Confusion Matrix:")
    y_true_flat = [label for seq in y_test for label in seq]
    y_pred_flat = [label for seq in y_pred for label in seq]

    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    output.append(df_cm.to_string())

    # 5. Top Feature Weights
    # This is the 'Explainability' section to see what
    # the model actually learned about Georgian morphology.
    num_to_include = 15
    output.append(
        f"\n[2] Top {str(num_to_include)} Most Influential Features:")

    # State features map (attribute, label) -> weight
    features = crf.state_features_
    # Sort by absolute weight (magnitude) to see the strongest signals
    sorted_features = sorted(
        features.items(), key=lambda x: abs(x[1]), reverse=True)

    output.append(f"{'Weight':>10} | {'Label':<6} | {'Feature'}")
    output.append("-" * 45)
    for (attr, label), weight in sorted_features[:num_to_include]:
        output.append(f"{weight:10.4f} | {label:<6} | {attr}")

    # Combine all lines into a single string
    full_report = "\n".join(output)

    # Print to console
    print(full_report)

    # Save to the model directory using the new IO helper
    save_evaluation(model_dir, full_report)


if __name__ == "__main__":
    # Specify a path here to test a specific model, or leave as None for latest
    target_model = None
    run_evaluation(target_model)
