from sklearn.metrics import mean_squared_log_error, mean_absolute_error

import numpy as np

def get_training_dataset(df):
    df_train = df[df.sale_year != 2012]

    X_train = df_train.drop(["SalesID", "SalePrice"], axis=1)
    y_train = df_train[["SalesID", "SalePrice"]]

    return X_train, y_train


def get_validation_dataset(df):
    df_val = df[df.sale_year == 2012]

    X_valid = df_val.drop(["SalesID", "SalePrice"], axis=1)
    y_valid = df_val[["SalesID", "SalePrice"]]

    return X_valid, y_valid

def root_mean_squared_log_error(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def print_scores(model, X_train, y_train, X_valid, y_valid):
    train_predictions = model.predict(X_train)
    validation_predictions = model.predict(X_valid)
    y_train_true = y_train.SalePrice.values
    y_valid_true = y_valid.SalePrice.values
    print("Training MAE", mean_absolute_error(y_train_true, train_predictions))
    print("Valid MAE", mean_absolute_error(y_valid_true, validation_predictions))
    print("Training RMSLE", root_mean_squared_log_error(y_train_true, train_predictions))
    print("Valid RMSLE", root_mean_squared_log_error(y_valid_true, validation_predictions))
