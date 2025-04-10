import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    model()


def model(
    path="../data/processed/preprocessed_sleep_dataset.csv", model_path="../models/"
):
    # Load the dataset
    df = pd.read_csv(path)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=["Sleep Disorder"]),
        df["Sleep Disorder"],
        test_size=0.3,
        random_state=42,
    )

    ## Initialize the Decision Tree Classifier model
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)

    # Evaluate Decision Tree Classifier  model - Training and Testing Accuracy - Comment later
    # print("Training Accuracy: ", dtree.score(X_train, y_train))
    # print("Testing Accuracy: ", dtree.score(X_test, y_test))

    ## Initialize the Random Forest Classifier model


if __name__ == "__main__":
    main()
