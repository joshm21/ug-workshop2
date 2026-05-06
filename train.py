import datetime
import sklearn_crfsuite
from sklearn.model_selection import train_test_split

from lib.io import load_georgian_data, save_artifacts
from lib.sequence import encode_sequence
from features import char2features


def run_training():
    print("--- Training Pipeline Started ---")
    df = load_georgian_data('data')
    if df is None:
        print("No data found, exiting")
        return

    # X is our features (inputs)
    # y is our labels (ouputs)
    X, y = [], []
    for _, row in df.iterrows():
        chars, labels = encode_sequence(row)
        if not chars:
            continue

        # Build feature sequence for the word using features.py
        word_features = [
            char2features(chars, i, row)
            for i in range(len(chars))
        ]
        X.append(word_features)
        y.append(labels)

    # Split data into training and testing groups
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    crf = sklearn_crfsuite.CRF(
        # We use the 'lbfgs' algorithm, which is standard for gradient-based optimization.
        algorithm='lbfgs',

        # L1 Regularization (Sparsity):
        # Encourages the model to set unimportant feature weights to zero.
        # Higher values create a "simpler" model that ignores noisy features.
        c1=0.1,

        # L2 Regularization (Smoothing):
        # Prevents any single feature from having an overwhelming weight.
        # Higher values help prevent "overfitting" to specific words in the training set.
        c2=0.1,

        # Discards features that appear very rarely.
        # Useful for preventing the model from "memorizing" typos or outliers.
        min_freq=0,

        # Allows the model to learn transitions that never appeared in the data.
        # (e.g., Learning that an 'S' label is almost never followed by another 'S').
        all_possible_transitions=True,

        # If True, the model creates weights for
        # (feature, label) combinations even if they never appeared together.
        all_possible_states=True,

        # The maximum number of times the optimizer will try to improve the model.
        # 100 is usually enough for this size of morphological data.
        max_iterations=100,

        # The "Stopping Threshold."
        # A larger number (e.g., 1e-3) makes training faster but less precise.
        epsilon=1e-5,

        # Set to True so can see the 'Loss' decrease in the console.
        verbose=True
    )

    # Train the model to fit the training data group
    # Note: the test data group never gets sent to the model
    # Note: evaluate.py will show how well this model fits the unseen test data
    crf.fit(X_train, y_train)

    # Save with a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"models/run_{timestamp}"
    save_artifacts(output_dir, crf, X_test, y_test)
    print(f"\n✅ Model trained and saved to: {output_dir}")


if __name__ == "__main__":
    run_training()
