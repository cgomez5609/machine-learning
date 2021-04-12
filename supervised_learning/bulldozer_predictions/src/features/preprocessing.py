import re

from supervised_learning.bulldozer_predictions.src.constants.constants import COLUMNS_TO_REMOVE


class Preprocessing:
    def __init__(self, df):
        self.df = df

    def remove_columns(self):
        self.df.drop(columns=COLUMNS_TO_REMOVE, axis=1, inplace=True)

    def print_columns_that_were_removed(self):
        print("The following columns were removed")
        for column in COLUMNS_TO_REMOVE:
            print(column)

    # Remove punctuation and letters from Tire Size, then convert to integer
    def correct_tire_size(self):
        tire_size = self.df.Tire_Size.values
        for i in range(len(tire_size)):
            tire_size[i] = str(tire_size[i])
            tire_size[i] = re.sub("[^0-9]", "", tire_size[i])
            if tire_size[i] == '':
                tire_size[i] = 0
            else:
                tire_size[i] = int(tire_size[i])
        avg = int(sum(tire_size) / len(tire_size))
        for i in range(len(tire_size)):
            if tire_size[i] == 0:
                tire_size[i] = avg
        self.df.Tire_Size = tire_size

    def correct_machine_hours(self):
        self.df["MachineHoursCurrentMeter"].fillna(self.df['MachineHoursCurrentMeter'].median(), inplace=True)

    def all_preprocessing(self):
        self.remove_columns()
        self.correct_tire_size()
        self.correct_machine_hours()