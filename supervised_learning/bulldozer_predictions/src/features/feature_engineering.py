import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from supervised_learning.bulldozer_predictions.src.features.category_order import category_order, create_ordinal_dict
from supervised_learning.bulldozer_predictions.src.constants.constants import ORDINAL_COLUMNS, OHE_COLUMNS

pd.options.mode.chained_assignment = None  # default='warn'

class FeatureEngineering:
    def __init__(self, df):
        self.original_df = df
        self.df = df.copy()
        self.ordinal_values = create_ordinal_dict()
        self.enc = OneHotEncoder(handle_unknown='ignore')
        self.enc_code_names = None
        self.new_df = None

    def create_sale_day_month_year_columns(self, drop_original_date=True):
        self.df["sale_year"] = self.df.saledate.dt.year
        self.df["sale_month"] = self.df.saledate.dt.month
        self.df["sale_day"] = self.df.saledate.dt.day
        self.df.drop(["saledate"], axis=1, inplace=True)

    def convert_columns_to_ordinal_values(self):
        for column in ORDINAL_COLUMNS:
            values = self.df[column]
            self.df[column] = self.__ordinal_converter_helper(column, values)

    def __ordinal_converter_helper(self, column, values):
        new_values = list()
        for value in values:
            if value in self.ordinal_values[column]:
                new_values.append(self.ordinal_values[column][value])
            else:
                new_values.append(self.ordinal_values[column]["unknown"])
        return new_values

    def convert_columns_to_onehotencoding(self):
        X_objects = self.df.select_dtypes('object')
        X_objects.drop(["Tire_Size"], axis=1, inplace=True)
        column_names = X_objects.columns.values
        self.enc.fit(X_objects)
        codes = self.enc.transform(X_objects).toarray()
        self.enc_code_names = self.enc.get_feature_names()
        self.__get_new_encoded_dataframe(column_names, codes)

    def __get_new_encoded_dataframe(self, column_names, codes):
        column_names_for_frame = list()
        index = 0
        for i in range(len(column_names)):
            for j in range(index, len(self.enc_code_names)):
                if int(self.enc_code_names[j][1]) == i:
                    column_names_for_frame.append(str(column_names[i] + self.enc_code_names[j][2:]))
                else:
                    index = j
                    break
        df_train = self.df.copy()
        df_train.drop(column_names, axis=1, inplace=True)
        ohe = pd.DataFrame(data=codes, columns=column_names_for_frame)
        self.new_df = pd.concat([df_train, ohe], axis=1)

    def print_columns_converted_to_ordinal(self):
        for column in ORDINAL_COLUMNS:
            print(column)

    def print_columns_converted_to_onehotencoding(self):
        for column in ORDINAL_COLUMNS:
            print(column)

    def all_feature_engineering(self):
        self.create_sale_day_month_year_columns()
        self.convert_columns_to_ordinal_values()
        self.convert_columns_to_onehotencoding()

    def transform_new_data(self, data):
        self.__sale_day_month_year_transform(data)
        self.__ordinal_values_transformer(data)
        return self.__onehotencoding_values_transformer(data)

    def __sale_day_month_year_transform(self, data, drop_original_date=True):
        data["sale_year"] = int(data.saledate.dt.year)
        data["sale_month"] = data.saledate.dt.month
        data["sale_day"] = data.saledate.dt.day
        data.drop(["saledate"], axis=1, inplace=True)

    def __ordinal_values_transformer(self, data):
        for column in ORDINAL_COLUMNS:
            values = data[column]
            data[column] = self.__ordinal_converter_helper(column, values)

    def __onehotencoding_values_transformer(self, data):
        X_objects = data.select_dtypes('object')
        X_objects.drop(["Tire_Size"], axis=1, inplace=True)
        column_names = X_objects.columns.values
        codes = self.enc.transform(X_objects).toarray()
        enc_code_names = self.enc.get_feature_names()
        return self.__get_new_encoded_transformer(data, enc_code_names, column_names, codes)

    def __get_new_encoded_transformer(self, data, enc_code_names, column_names, codes):
        column_names_for_frame = list()
        index = 0
        for i in range(len(column_names)):
            for j in range(index, len(enc_code_names)):
                if int(enc_code_names[j][1]) == i:
                    column_names_for_frame.append(str(column_names[i] + enc_code_names[j][2:]))
                else:
                    index = j
                    break
        df_train = data.copy()
        df_train.drop(column_names, axis=1, inplace=True)
        ohe = pd.DataFrame(data=codes, columns=column_names_for_frame)
        return pd.concat([df_train, ohe], axis=1)
