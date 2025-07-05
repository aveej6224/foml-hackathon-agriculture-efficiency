
# Estimating Agrarian Land Efficiency – VNB FoML Hackathon 2024

This repository contains the complete solution to the VNB FoML 2024 Hackathon conducted as part of the Foundations of Machine Learning course at IIT Hyderabad under the instruction of Prof. Vineeth N Balasubramanian. The objective of the hackathon is to build a robust classifier that determines whether a given agricultural field is using its resources efficiently, relative to its potential.

## Problem Overview

Agricultural fields are to be classified into one of three categories:

- **Low Performing**
- **Moderately Performing**
- **High Performing**

These categories indicate how efficiently the field is performing relative to its resources and constraints. For example, a field may produce reasonable yield but still be considered low performing due to factors like market access or land utilization.

The main challenge lies in identifying this relative performance using features like soil quality, infrastructure, accessibility, and resource availability.

## Dataset Description

The dataset provided for the hackathon includes the following files:

- `train.csv`: Contains 112,569 training examples with features and a `Target` column indicating one of the three performance labels.
- `test.csv`: Contains 15,921 instances without labels, requiring prediction.
- `sample_submission.csv`: A template to format the predicted outputs for Kaggle submission.
- `descriptions.txt`: Describes all features present in the dataset.

In addition to the public test set used for leaderboard evaluation, a **private test set** (32,324 samples) is used for final scoring based on submitted notebooks.

## Project Structure

```
.
├── cs24mtech14011_foml24_hackathon.py         # End-to-end pipeline: preprocessing, training, prediction
├── cs24mtech14011_foml24_hackathon_report.pdf # Detailed report describing methodology and findings
├── train.csv                                   # Training data
├── test.csv                                    # Test data 
└── README.md                                   # Project documentation
```

## Methodology

### Preprocessing

- Missing value imputation
- Label encoding and one-hot encoding for categorical features
- Feature normalization using `StandardScaler`

### Model Selection

- Explored various models: Logistic Regression, Random Forest, SVM, XGBoost
- Used cross-validation and grid search for hyperparameter tuning
- Final model selected based on best macro F1-score on validation set

### Prediction Interface

To facilitate private leaderboard evaluation, the script includes the following function as required:

```python
def make_predictions(test_fname, predictions_fname):
    # Reads test data from test_fname
    # Writes predictions to predictions_fname in required format
    pass
```

The function reads test data from a file and writes predictions for each UID in the required CSV format.

## Evaluation Metric

Both public and private leaderboard performance is evaluated using the **macro F1-score**, which averages F1-scores across all three classes without weighting by class frequency.

Example:

For ground truth = [low, medium, medium, high, low]  
And predictions = [low, low, medium, high, high]  
The macro F1-score = average(F1_low, F1_medium, F1_high) = 0.61

This metric ensures equal importance to all classes, making it suitable for imbalanced datasets.

## Submission Guidelines

### Public Leaderboard (Kaggle)

Submit a `.csv` file in this format:

```
UID,Target
2200,low
50,medium
600,low
...
```

Ensure that each UID from `test.csv` has a corresponding prediction and the file has a header.

### Private Leaderboard (Google Classroom)

Each student must submit the following files:

- `cs24mtech14011_foml24_hackathon.py`
- `cs24mtech14011_foml24_hackathon_report.pdf`

Both files must be zipped as:

```
cs24mtech14011_foml24_hackathon.zip
```

Upload the `.zip` file to Google Classroom. Note that while the notebook/script is the same for teammates, the report must highlight individual contributions.

## How to Reproduce

Clone the repository:

```
git clone https://github.com/<your-username>/foml-hackathon-2024.git
cd foml-hackathon-2024
```

Run the main Python script:

```
python cs24mtech14011_foml24_hackathon.py
```

To generate predictions:

```python
make_predictions("test.csv", "predictions.csv")
```

## Author

- **Name:** Aviraj Antala
- **Roll Number:** CS24MTECH14011
- **Institute:** IIT Hyderabad
- **Course:** Foundations of Machine Learning (FoML)

## Useful Links

- Hackathon page: https://www.kaggle.com/competitions/vnb-foml-2024-hackathon
