from supervised_learning.bulldozer_predictions.src.constants.constants import ORDINAL_COLUMNS


def category_order(df, column):
    if column == "UsageBand":
        df[column].cat.reorder_categories(['Low', 'Medium', 'High'], inplace=True)
    elif column == "ProductSize":
        df[column].cat.reorder_categories(['Mini', 'Small', 'Compact', 'Medium',
                                           'Large / Medium', 'Large'], inplace=True)
    elif column == "Ride_Control":
        df[column].cat.reorder_categories(['None or Unspecified', 'No', 'Yes'], inplace=True)
    elif column == "Grouser_Type":
        df[column].cat.reorder_categories(['Single', 'Double', 'Triple'], inplace=True)
    elif column == "Differential_Type":
        df[column].cat.reorder_categories(['No Spin', 'Standard', 'Limited Slip', 'Locking'], inplace=True)
    elif column == "Steering_Controls":
        df[column].cat.reorder_categories(['No', 'Wheel', 'Conventional', 'Four Wheel Standard', 'Command Control'], inplace=True)


def create_ordinal_dict():
    ordinal_dict = dict()
    for column in ORDINAL_COLUMNS:
        if column == "UsageBand":
            ordinal_dict[column] = dict()
            order_number = 0
            for cat in ['unknown', 'Low', 'Medium', 'High']:
                ordinal_dict[column][cat] = order_number
                order_number += 1
        elif column == "ProductSize":
            ordinal_dict[column] = dict()
            order_number = 0
            for cat in ['unknown','Mini', 'Small', 'Compact', 'Medium', 'Large / Medium', 'Large']:
                ordinal_dict[column][cat] = order_number
                order_number += 1
        elif column == "Ride_Control":
            ordinal_dict[column] = dict()
            order_number = 0
            for cat in ['unknown', 'No', 'Yes']:
                ordinal_dict[column][cat] = order_number
                order_number += 1
        elif column == "Grouser_Type":
            ordinal_dict[column] = dict()
            order_number = 0
            for cat in ['unknown', 'Single', 'Double', 'Triple']:
                ordinal_dict[column][cat] = order_number
                order_number += 1
        elif column == "Differential_Type":
            ordinal_dict[column] = dict()
            order_number = 0
            for cat in ['unknown', 'No Spin', 'Standard', 'Limited Slip', 'Locking']:
                ordinal_dict[column][cat] = order_number
                order_number += 1
        elif column == "Steering_Controls":
            ordinal_dict[column] = dict()
            order_number = 0
            for cat in ['unknown', 'No', 'Wheel', 'Conventional', 'Four Wheel Standard', 'Command Control']:
                ordinal_dict[column][cat] = order_number
                order_number += 1
    return ordinal_dict
