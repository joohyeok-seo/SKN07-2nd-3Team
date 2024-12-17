import pandas as pd

# preprossing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler , LabelEncoder , OrdinalEncoder
from sklearn.pipeline import Pipeline , FeatureUnion
from sklearn_features.transformers import DataFrameSelector
from datasist.structdata import detect_outliers
from sklearn.model_selection import cross_val_score, cross_val_predict
from imblearn.over_sampling import SMOTE



def preprocess(df):
    """

    Args:
        df (pandas.Dataframe): input data.
    """
    label_encoders = {}
    categorical_columns = ['International plan', 'Voice mail plan', 'Churn']

    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    numerical_columns = [
        'Account length', 'Number vmail messages', 'Total day minutes', 'Total day calls',
        'Total day charge', 'Total eve minutes', 'Total eve calls', 'Total eve charge',
        'Total night minutes', 'Total night calls', 'Total night charge',
        'Total intl minutes', 'Total intl calls', 'Total intl charge',
        'Customer service calls'
    ]

    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    # test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])

    columns_to_drop = ['State', 'Area code']
    df = df.drop(columns=columns_to_drop, axis=1)
    # test_data = test_data.drop(columns=columns_to_drop, axis=1)

    x = df.drop('Churn', axis=1)
    y = df['Churn']

    return x, y



if __name__ == "__main__":
    ...