from pathlib import Path
import pandas as pd


def load_dataset(file_path):

    # Step 1: Convert the given path into a Path object
    path = Path(file_path)

    # Step 2: Check whether the file exists
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    # Step 3: Check whether the given path is actually a file
    if not path.is_file():
        raise ValueError(f"The given path is not a file: {file_path}")

    # Step 4: Validate the file extension
    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported.")

    # Step 5: Load the dataset
    try:
        df = pd.read_csv(path)

        # Step 6: Check whether the dataset is empty
        if df.empty:
            raise ValueError("The dataset contains no records.")

        return df

    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty or contains no readable data.")

    except pd.errors.ParserError:
        raise ValueError("Unable to parse the CSV file. Check its format.")

    except UnicodeDecodeError:
        raise ValueError("Unable to decode the CSV file. Check its encoding.")