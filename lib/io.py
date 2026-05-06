import os

import pandas as pd
import joblib


def load_georgian_data(data_dir: str = 'data') -> pd.DataFrame:
    """
    Loads words.csv and forms.csv and merges them on word_id.
    """
    words_path = os.path.join(data_dir, 'words.csv')
    forms_path = os.path.join(data_dir, 'forms.csv')

    try:
        words_df = pd.read_csv(words_path)
        forms_df = pd.read_csv(forms_path)
        df = forms_df.merge(words_df, on='word_id')

        # Explicit Validation
        df['word_id'] = df['word_id'].astype(str)
        df['word_type'] = pd.to_numeric(
            df['word_type'], errors='coerce').fillna(0).astype(int)
        df['has_swap'] = df['has_swap'].fillna('n')
        print(f"Successfully loaded {len(df)} conjugated forms.")
        return df
    except FileNotFoundError as e:
        print(f"Error: Could not find data files in {data_dir}. {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def save_artifacts(directory, model, X_test, y_test):
    """Tucks away the complexity of saving binary files."""
    os.makedirs(directory, exist_ok=True)
    joblib.dump(model, os.path.join(directory, 'model.pkl'))
    joblib.dump((X_test, y_test), os.path.join(directory, 'test_data.pkl'))


def load_artifacts(model_dir: str):
    """
    Loads the model and test data from a specific run directory.
    This ensures we are evaluating the correct version of the model.
    """
    model_path = os.path.join(model_dir, 'model.pkl')
    test_data_path = os.path.join(model_dir, 'test_data.pkl')

    if not os.path.exists(model_path) or not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Missing artifacts in {model_dir}")

    crf = joblib.load(model_path)
    X_test, y_test = joblib.load(test_data_path)
    return crf, X_test, y_test


def get_latest_run_dir(model_root: str = 'models') -> str:
    """
    Automatically finds the most recently created model folder.
    """
    if not os.path.exists(model_root) or not os.listdir(model_root):
        raise FileNotFoundError(
            f"No models found in {model_root}. Did you train first?")

    # Get all subdirectories and sort by creation time
    all_runs = [os.path.join(model_root, d) for d in os.listdir(model_root)
                if os.path.isdir(os.path.join(model_root, d))]

    if not all_runs:
        raise FileNotFoundError(
            f"No run directories found inside {model_root}.")

    return max(all_runs, key=os.path.getmtime)
