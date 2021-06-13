import os
import subprocess
import sys

import fire
import pickle
import numpy as np
import pandas as pd

import hypertune

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import balanced_accuracy_score
from category_encoders.hashing import HashingEncoder


def train_evaluate(job_dir, training_dataset_path, 
                   validation_dataset_path, min_samples_leaf, max_depth, 
                   max_features, hptune):
    
    #Grab data
    df_train = pd.read_csv(training_dataset_path).dropna()
    df_validation = pd.read_csv(validation_dataset_path).dropna()

    if not hptune:
        df_train = pd.concat([df_train, df_validation])

    #Transform data
    numeric_feature_indexes = slice(0, 8)
    hash_features_indexes = slice(9,10)
    categorical_feature_indexes = slice(11, 12)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_feature_indexes),
            ('hash', HashingEncoder(), hash_features_indexes),
            ('cat', OneHotEncoder(), categorical_feature_indexes) 
    ])


    #Create Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(tol=1e-3))
    ])

    #Cast Data
    num_features_type_map = {feature: 'float64' for feature 
                             in df_train.columns[numeric_feature_indexes]}
    df_train = df_train.astype(num_features_type_map)
    df_validation = df_validation.astype(num_features_type_map) 

    #make training dataset & train
    print('Starting training: min_samples_leaf={}, max_depth={}, max_features={}'.format(min_samples_leaf, max_depth, max_features))
    X_train = df_train.drop('sold', axis=1)
    y_train = df_train['sold']

    pipeline.set_params(classifier__min_samples_leaf=min_samples_leaf, 
        classifier__max_depth=max_depth, 
        classifier__max_features=max_features
    )
    pipeline.fit(X_train, y_train)

    if hptune:
        X_validation = df_validation.drop('sold', axis=1)
        y_validation = df_validation['sold']
        y_val_pred=pipeline.predict(X_validation)
        accuracy = balanced_accuracy_score(y_validation, y_val_pred)
        print('Model accuracy: {}'.format(accuracy))
        # Log it with hypertune
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
          hyperparameter_metric_tag='accuracy',
          metric_value=accuracy
        )
    else:
        # Save the model
        model_filename = 'model.pkl'
        with open(model_filename, 'wb') as model_file:
            pickle.dump(pipeline, model_file)
        gcs_model_path = "{}/{}".format(job_dir, model_filename)
        subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path],
                          stderr=sys.stdout)
        print("Saved model in: {}".format(gcs_model_path)) 
    
if __name__ == "__main__":
    fire.Fire(train_evaluate)