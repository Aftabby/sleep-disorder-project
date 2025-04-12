from data_cleaning import clean
from data_preprocess import preprocess
from model_building import model
import pandas as pd
import pickle


def main():
    pipeline()


# & Run all the scripts and save/output necessary files
def pipeline(app_path="../app/static/data/"):
    cleaned_df = clean()
    preprocess()
    dtree_model, rfc_model, compare_df = model()

    # Save clean data in app folder for flask
    cleaned_df.to_csv(app_path + "cleaned_sleep_dataset.csv", index=False)

    # Save the processed data with predictions in app folder for flask (Only the actual, and above two columns)
    compare_df.to_csv(app_path + "compare_df.csv", index=False)

    # Save the models to both the project and flask app models folder
    pckl = {"dtree_model": dtree_model, "rfc_model": rfc_model}

    pickle.dump(
        pckl, open("../app/models/trained_models.pickle", "wb")
    )  # Saving in Flask app folder
    pickle.dump(
        pckl, open("../models/trained_models.pickle", "wb")
    )  # Saving in project folder


if __name__ == "__main__":
    main()
