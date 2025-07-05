import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from tqdm import tqdm
import argparse

train_data = pd.read_csv("train.csv")
y_train = train_data["Target"]
X_train = train_data.drop("Target", axis=1)

def get_missing_values(X:pd.DataFrame)->pd.DataFrame:
    missing_values = X.isnull().sum()
    missing_values = missing_values/len(train_data)*100
    return missing_values
missing_values = get_missing_values(X_train)

def columns_with_high_nulls(df_train: pd.DataFrame, threshold: float = 0.9):
    null_percentage_train = df_train.isnull().mean()
    cols_to_drop = null_percentage_train[null_percentage_train > threshold].index
    print(f"Columns to be dropped from the data: {list(cols_to_drop)}")
    return cols_to_drop

columns_to_be_dropped = columns_with_high_nulls(X_train)

def fill_with_mode(columns:list, train_df:pd.DataFrame, df:pd.DataFrame)->pd.DataFrame:
    for column in columns:
        mode_value = train_df[column].mode()[0]
        df[column] = df[column].fillna(mode_value)
    return df

def fill_with_mean(columns:list, train_df:pd.DataFrame, df:pd.DataFrame)->pd.DataFrame:
    for column in columns:
        mean_value = train_df[column].mean()
        df[column] = df[column].fillna(mean_value)
    return df

def fill_with_median(columns:list, train_df:pd.DataFrame, df:pd.DataFrame)->pd.DataFrame:
    for column in columns:
        median_value = train_df[column].median()
        df[column] = df[column].fillna(median_value)
    return df

mean_columns = ["CultivatedAreaSqft1", "FieldSizeSqft", "TaxAgrarianValue", "TaxLandValue", "TotalCultivatedAreaSqft",
                "TotalTaxAssessed", "TotalValue", "CropSpeciesVariety", "FarmingUnitCount", "HarvestProcessingType", "LandUsageType",
                "SoilFertilityType", "WaterAccessPointsCalc"]
mode_columns = ["CropSpeciesVariety", "FarmingUnitCount", "HarvestProcessingType", "LandUsageType",
                "SoilFertilityType", "WaterAccessPointsCalc"]
median_columns = ["MainIrrigationSystemCount", "StorageAndFacilityCount", "WaterAccessPoints"]

def apply_preprocessing(train_df:pd.DataFrame, df:pd.DataFrame, columns_drop:list, columns_to_fill:dict)->pd.DataFrame:
    df = df.drop(columns=columns_drop)
    for x in columns_to_fill:
        if(x=="mean"):
            df = fill_with_mean(columns_to_fill["mean"], train_df, df)
        elif (x=="median"):
            df = fill_with_median(columns_to_fill["median"], train_df, df)
        else:
            df = fill_with_mode(columns_to_fill["mode"], train_df, df)
    if df.isnull().values.any():
        df = df.apply(lambda col: col.fillna(train_df[col.name].mean()))
    return df

columns_drop = columns_to_be_dropped.tolist()
columns_to_fill = {"mean":mean_columns, "median":median_columns, "mode":mode_columns}

labels = y_train.unique()
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)
y_train = encoder.transform(y_train)
X_train = X_train.drop(columns=["UID"])

X_train = apply_preprocessing(X_train, X_train, columns_drop, columns_to_fill)

device = "cuda" if torch.cuda.is_available() else "cpu"

model= TabNetClassifier(optimizer_fn=torch.optim.Adam,
                       scheduler_params={"step_size":10,
                                         "gamma":0.9},
                       scheduler_fn=torch.optim.lr_scheduler.StepLR,
                       device_name = device
                      )

model.fit(
    X_train.values,y_train,
    eval_set=[(X_train.values, y_train), (X_train.values, y_train)],
    eval_name=['train', 'test'],
    eval_metric=['balanced_accuracy'],
    max_epochs=200, patience=60,
    batch_size=1024, virtual_batch_size=16,
    num_workers=1,
    weights=1,
    drop_last=False
)

def make_predictions(test_fname, predictions_fname):
    test_data = pd.read_csv(test_fname)
    ids = test_data['UID']
    X_test = test_data.drop(columns=['UID'])
    X_test = apply_preprocessing(X_train, X_test, columns_to_be_dropped, columns_to_fill)

    predictions = model.predict(X_test.values)
    predictions = encoder.inverse_transform(predictions)
    submission = pd.DataFrame({
        'UID': ids,
        'Target': predictions
    })
    submission.to_csv(predictions_fname, index=False)
    print(f"Predictions saved to {predictions_fname}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=str, help='file path of train.csv')
    parser.add_argument("--test-file", type=str, help='file path of test.csv')
    parser.add_argument("--predictions-file", type=str, help='save path of predictions')
    args = parser.parse_args()
    make_predictions(args.test_file, args.predictions_file)
