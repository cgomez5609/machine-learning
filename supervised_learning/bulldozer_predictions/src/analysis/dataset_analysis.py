import pandas as pd
import matplotlib.pyplot as plt

class DatasetAnalysis:
    def __init__(self, description_path, dataset_path):
        if description_path.exists() and dataset_path.exists():
            self.desc_df = pd.read_csv(description_path)
            self.original_dataset = pd.read_csv(dataset_path, low_memory=False, parse_dates=["saledate"])
            self.dataset_df = self.original_dataset.copy()
            self.valid_paths = True
            self.column_names = self.dataset_df.columns
            self.desc_dict = self.create_feature_description_dict()
            self.valid_paths = True
        else:
            print("not a valid path")
            self.valid_paths = False

    def create_feature_description_dict(self):
        if self.valid_paths:
            desc_dict = dict()
            # column names are Variable and Description
            for i in range(len(self.desc_df)):
                desc_dict[self.desc_df.Variable.values[i]] = self.desc_df.Description.values[i]
            return desc_dict

    def print_description_for_feature(self, feature_name):
        if self.valid_paths:
            if feature_name in self.desc_dict:
                print(self.desc_dict[feature_name])
            else:
                print("Feature name not found")

    def get_sample_from_dataset(self, num_of_datapoints):
        if self.valid_paths:
            return self.dataset_df.sample(n=num_of_datapoints)

    def display_analysisgraph(self):
        fig, ax = plt.subplots()
        ax.scatter(self.dataset_df["saledate"][:1000], self.dataset_df["SalePrice"][:1000])
        ax.set_xlabel("Sale Date")
        ax.set_ylabel("Sale Price")
        plt.show()



