import pandas as pd


def main():
    pass


def clean(path="../data/sleep_dataset.csv", out_path="../data/processed/"):
    # Load the dataset
    df = pd.read_csv("../data/sleep_dataset.csv")

    # Drop unnecessary columns
    df.drop("Person ID", axis=1, inplace=True)

    # Replacing the null values with 'None' in the 'Sleep Disorder' column
    df["Sleep Disorder"].fillna("None", inplace=True)

    # Splitting the 'Blood Pressure' column into 'Systolic' and 'Diastolic'
    df[["systolic_bp", "diastolic_bp"]] = df["Blood Pressure"].str.split(
        "/", expand=True
    )

    # Dropping the original 'Blood Pressure' column
    df.drop("Blood Pressure", axis=1, inplace=True)

    # Converting the 'systolic_bp' and 'diastolic_bp' columns to integers
    df["systolic_bp"] = df["systolic_bp"].astype(int)
    df["diastolic_bp"] = df["diastolic_bp"].astype(int)

    # Replacing 'Normal weight' with 'Normal in "BMI Category" column
    df["BMI Category"].replace("Normal weight", "Normal", inplace=True)

    # Saving the cleaned dataset to a new CSV file
    df.to_csv(out_path + "cleaned_sleep_dataset.csv", index=False)

    return df


if __name__ == "__main__":
    main()
