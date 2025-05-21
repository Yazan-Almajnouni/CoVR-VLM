import pandas as pd

def filter_train(train_path: str, val_path: str, out_path: str) -> None:
    """
    Reads train and validation CSVs, filters out rows in train where the
    'pth1' value exists in val, and writes the result to out_path.
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Get the set of pth1 values in the validation set
    val_pth1 = set(val_df["pth1"].unique())

    # Filter train rows whose pth1 is not in val_pth1
    filtered = train_df[train_df["pth1"].isin(val_pth1)]

    # Write the result
    filtered.to_csv(out_path, index=False)
    print(f"Wrote filtered data to {out_path}. "
          f"Kept {len(filtered)} of {len(train_df)} rows.")

train_csv = "annotations/webvid8m-covr_test.csv" 
val_csv = "annotations/validation_set.csv"

out_csv = "annotations/val_set.csv"

filter_train(train_csv, val_csv, out_csv)

