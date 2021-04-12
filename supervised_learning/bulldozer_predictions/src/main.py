from pathlib import Path

from supervised_learning.bulldozer_predictions.src.analysis.dataset_analysis import DatasetAnalysis
from supervised_learning.bulldozer_predictions.src.features.feature_engineering import FeatureEngineering
from supervised_learning.bulldozer_predictions.src.features.preprocessing import Preprocessing
from supervised_learning.bulldozer_predictions.src.model_creation.model_functions import get_training_dataset, get_validation_dataset
from supervised_learning.bulldozer_predictions.src.model_creation.model_functions import print_scores

def main():
    # Read in and analyze the dataset
    data_path = Path().absolute().parent / "data"
    if data_path.exists():
        data = DatasetAnalysis(description_path=data_path / "data_description.csv",
                               dataset_path=data_path / "TrainandValid.csv")
        data.display_analysisgraph()
        df = data.dataset_df

        # Preprocess dataset by:
        # 1. Removing specific columns
        # 2. Correcting tire size (String to Int)
        # 3. Fill in Missing values in Machine Current Meter with median
        preprocess = Preprocessing(df=df)
        preprocess.all_preprocessing()
        df = preprocess.df

        # Feature Engineering
        # 1. Get day, month, and year as separate columns - then removed original sale date column
        # 2. Specific columns were turned into ordinal categories and the order was adjusted manually*
        # 3. Remaining categorical columns were transformed using sklearns one hot encoding
        # Note*: Simply transforming columns to ordinal values wasn't sufficient since the order was not maintained.
        # For example small, medium, large -> 2,1,3, therefore I changed it to 1,2,3.
        feat_eng = FeatureEngineering(df=df)
        feat_eng.all_feature_engineering()
        df_train_org = feat_eng.new_df

        # Split training data into a training and validation set
        X_train, y_train = get_training_dataset(df=df_train_org)
        X_valid, y_valid = get_validation_dataset(df=df_train_org)

        # Train model using training set (X_train, y_train)
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_jobs=-1, random_state=42)
        model.fit(X_train, y_train.SalePrice.values)  # y_train also contains SalesID

        # Display the Mean Absolute Error and Root Mean Squared Error score
        print_scores(model, X_train, y_train, X_valid, y_valid)
    else:
        print("Not a valid path for", data_path)


if __name__ == '__main__':
    main()

