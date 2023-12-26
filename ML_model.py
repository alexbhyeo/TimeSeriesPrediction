import pandas as pd

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def prepare_data(energy_data):

    # loop through the data and get last 48-data for training data
    # if data % 49 == 0

    input_data = []

    reform_data = []
    iterator = 0
    for index, row in energy_data.iterrows():
        iterator = iterator + 1
        if iterator % 49 == 0:
            reform_data.append({
                "label" : row["hourly_energy"],
                "input" : input_data
            })
            input_data = []
        else:
            input_data.append(row["hourly_energy"])


    # process to this format [feat_48, feat_47... label]
    trainable_data = []
    print("reform_data : ", reform_data)
    for index, content in enumerate(reform_data):
        if len(content["input"]) < 48:
            continue
        #print(content)
        #print(len(content["input"]))
        trainable_data.append({
            "feature_48" : content["input"][0],
            "feature_47": content["input"][1],
            "feature_46": content["input"][2],
            "feature_45": content["input"][3],
            "feature_44": content["input"][4],
            "feature_43": content["input"][5],
            "feature_42": content["input"][6],
            "feature_41": content["input"][7],
            "feature_40": content["input"][8],
            "feature_39": content["input"][9],
            "feature_38": content["input"][10],
            "feature_37": content["input"][11],
            "feature_36": content["input"][12],
            "feature_35": content["input"][13],
            "feature_34": content["input"][14],
            "feature_33": content["input"][15],
            "feature_32": content["input"][16],
            "feature_31": content["input"][17],
            "feature_30": content["input"][18],
            "feature_29": content["input"][19],
            "feature_28": content["input"][20],
            "feature_27": content["input"][21],
            "feature_26": content["input"][22],
            "feature_25": content["input"][23],
            "feature_24": content["input"][24],
            "feature_23": content["input"][25],
            "feature_22": content["input"][26],
            "feature_21": content["input"][27],
            "feature_20": content["input"][28],
            "feature_19": content["input"][29],
            "feature_18": content["input"][30],
            "feature_17": content["input"][31],
            "feature_16": content["input"][32],
            "feature_15": content["input"][33],
            "feature_14": content["input"][34],
            "feature_13": content["input"][35],
            "feature_12": content["input"][36],
            "feature_11": content["input"][37],
            "feature_10": content["input"][38],
            "feature_09": content["input"][39],
            "feature_08": content["input"][40],
            "feature_07": content["input"][41],
            "feature_06": content["input"][42],
            "feature_05": content["input"][43],
            "feature_04": content["input"][44],
            "feature_03": content["input"][45],
            "feature_02": content["input"][46],
            "feature_01": content["input"][47],
            "label" : content["label"]
        })
        #break
    
    trainable_data_df = pd.DataFrame(trainable_data)

    #print(trainable_data_df.head())
    return trainable_data_df



hourly_data = pd.read_csv("processed_energy.csv")
print(hourly_data.head())

training_data_list = []
for index in range(0, 48):
    hourly_data.drop(index=hourly_data.index[0], axis=0, inplace=True)
    training_data_list.append(prepare_data(hourly_data))

full_training_data = pd.concat(training_data_list)
print(len(training_data_list[0]))
print(len(full_training_data))
print(full_training_data.head())


def train_ml_model(training_content):
    # input [feature_48, feature_47..., feature_1]
    # output [this hour energy]

    model_features = ["feature_48", "feature_47", "feature_46", "feature_45", "feature_44", "feature_43", "feature_42", "feature_41", "feature_40", "feature_39", "feature_38", "feature_37", "feature_36", "feature_35", "feature_34", "feature_33", "feature_32", "feature_31", "feature_30", "feature_29", "feature_28", "feature_27", "feature_26", "feature_25", "feature_24", "feature_23", "feature_22", "feature_21", "feature_20", "feature_19", "feature_18", "feature_17", "feature_16", "feature_15", "feature_14", "feature_13", "feature_12", "feature_11", "feature_10", "feature_09", "feature_08", "feature_07", "feature_06", "feature_05", "feature_04", "feature_03", "feature_02", "feature_01"]
    label_feature = "label"
    training_features = training_content[model_features]
    target_feature = training_content[label_feature]

    # Normalize the inputs and labels
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_normalized = scaler_X.fit_transform(training_features)
    y_normalized = scaler_y.fit_transform(target_feature.values.reshape(-1, 1)).flatten()

    # Split the normalized data into training and testing sets
    X_train, X_valid, y_train, y_valid = train_test_split(X_normalized, y_normalized, test_size=0.25)
    xg_model = xgb.XGBRegressor(objective='reg:squarederror', max_depth=10, n_estimators=200, silent=False, nthread=50, subsample=0.5, colsample_bytree=0.5, seed=0)
    xg_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], early_stopping_rounds=10)

    # Evaluation
    y_pred = xg_model.predict(X_valid)
    # Inverse transform back to the original scale
    y_pred_original_scale = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Evaluate the model using Mean Squared Error
    mse = mean_squared_error(scaler_y.inverse_transform(y_valid.reshape(-1, 1)).flatten(), y_pred_original_scale)
    print(f"Mean Squared Error on Original Scale: {mse:.2f}")

    return

train_ml_model(full_training_data)
