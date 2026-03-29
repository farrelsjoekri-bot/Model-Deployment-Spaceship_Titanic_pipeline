import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def build_preprocessor(x_train):
    cat_feat = x_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    num_feat = x_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    numeric_preprocess = Pipeline([
            ('num_imputer', SimpleImputer(strategy='median'))
        ])

    categorical_preprocess = Pipeline([
        ('cat_imputer', SimpleImputer(strategy='most_frequent')),
        ('cat_encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ('numPreprocess', numeric_preprocess, num_feat),
            ('catPreprocess', categorical_preprocess, cat_feat)
        ],
        remainder='drop'
    )
    return preprocess