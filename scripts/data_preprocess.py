import pandas as pd
from sklearn import preprocessing


def main():
    pass


# & Feature Encoding
def preprocess():
    # Load the dataset
    df = pd.read_csv("../data/processed/cleaned_sleep_dataset.csv")

    # Get all the catogorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Encode categorical features using Label Encoding
    label_encoder = preprocessing.LabelEncoder()
    for col in cat_cols:
        df[col] = label_encoder.fit_transform(df[col])

    # Save the preprocessed dataset
    df.to_csv("../data/processed/preprocessed_sleep_dataset.csv", index=False)

    return df


if __name__ == "__main__":
    main()
