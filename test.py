print('starting...')
from lib.io import load_artifacts, get_latest_run_dir
from lib.sequence import decode
from features import word_to_features


def run_manual_test(model_dir=None, test_cases=None):
    if model_dir is None:
        model_dir = get_latest_run_dir()

    if test_cases is None:
        print("No test cases provided.")
        return

    print(f"--- Running {len(test_cases)} Tests with Model: {model_dir} ---\n")

    # 1. Load the model artifacts
    crf, _, _ = load_artifacts(model_dir)

    for case in test_cases:
        word = case["word"]
        meta = case["meta"]

        # 2. Featurize and Predict directly
        features = word_to_features(word, meta)
        # The CRF expects a list of sequences
        predicted_labels = crf.predict([features])[0]

        # 3. Display result using decode
        result = decode(list(word), predicted_labels)
        print(f"Input:  {word}")
        print(f"Result: {result}")
        print("-" * 20)


if __name__ == "__main__":
    # Specify model path or leave as None for latest
    target_model = None

    # Define a list of test cases
    my_tests = [
        {
            "word": "გავაკეთებთ",
            "meta": {
                "base_word": "გაკეთება",
                "screeve": "fut",
                "person": "1p",
                "word_type": 1,
                "has_swap": "n"
            }
        },
        # You can easily add more dictionaries here later
    ]

    run_manual_test(target_model, my_tests)
