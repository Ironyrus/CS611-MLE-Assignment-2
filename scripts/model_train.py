import os
import glob
import argparse
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def model_train(model_train_date_str, model_train_date_end):
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # set up config
    #start collecting data from model_train_date_str onwards
    # model_train_date_str = "2023-01-01"
    train_test_period_months = 6
    #oot period is after train_test_period is done, and is used as unseen data for final evaluation
    # oot_period_months = 2
    train_test_ratio = 0.8

    config = {}
    #After bronze, silver, gold is in, train a model [monthly basis]
    config["model_train_date_str"] = model_train_date_str
    config["model_train_end_date_str"] = model_train_date_end
    config["train_test_period_months"] = train_test_period_months
    # config["oot_period_months"] =  oot_period_months
    config["model_train_date"] =  datetime.strptime(model_train_date_str, "%Y-%m-%d")
    # config["oot_end_date"] =  config['model_train_date'] - timedelta(days = 1)
    # config["oot_start_date"] =  config['model_train_date'] - relativedelta(months = oot_period_months)
    # config["train_test_end_date"] =  config["oot_start_date"] - timedelta(days = 1)
    # config["train_test_start_date"] =  config["oot_start_date"] - relativedelta(months = train_test_period_months)
    config["train_test_end_date"] =  datetime.strptime(model_train_date_end, "%Y-%m-%d")
    config["train_test_start_date"] =  datetime.strptime(model_train_date_str, "%Y-%m-%d")
    config["train_test_ratio"] = train_test_ratio 

    pprint.pprint(config)

    # connect to Gold label store
    folder_path = "datamart/gold/label_store/"
    files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
    #Spark DF created
    label_store_sdf = spark.read.option("header", "true").parquet(*files_list)
    # print("row_count:",label_store_sdf.count())
    # label_store_sdf.show()

    # extract label store
    #Get dates after train_test_start_date and before oot ends
    # labels_sdf = label_store_sdf.filter((col("snapshot_date") >= config["train_test_start_date"]) & (col("snapshot_date") <= config["oot_end_date"]))
    labels_sdf = label_store_sdf.filter((col("snapshot_date") >= config["train_test_start_date"]) & (col("snapshot_date") <= config["train_test_end_date"]))
    # Get features
    feature_location = "data/feature_clickstream.csv"

    # Load CSV into DataFrame - connect to feature store
    features_store_sdf = spark.read.csv(feature_location, header=True, inferSchema=True)
    # extract label store
    features_sdf = features_store_sdf.filter((col("snapshot_date") >= config["train_test_start_date"]) & (col("snapshot_date") <= config["train_test_end_date"]))
    # prepare data for modeling
    data_pdf = labels_sdf.join(features_sdf, on=["Customer_ID", "snapshot_date"], how="left").toPandas()
    
    ######################
    ### START TRAINING ###
    ######################

    #train_test_end_date is on a loop
    month = int(model_train_date_end.split("-")[1])
    #Train only if Mar, July, Nov
    if month in [3, 7, 11]:
        
        # split data into train - test - oot
        # oot_pdf = data_pdf[(data_pdf['snapshot_date'] >= config["oot_start_date"].date()) & (data_pdf['snapshot_date'] <= config["oot_end_date"].date())]
        train_test_pdf = data_pdf[(data_pdf['snapshot_date'] >= config["train_test_start_date"].date()) & (data_pdf['snapshot_date'] <= config["train_test_end_date"].date())]
    
        feature_cols = [fe_col for fe_col in data_pdf.columns if fe_col.startswith('fe_')]
    
        # X_oot = oot_pdf[feature_cols]
        # y_oot = oot_pdf["label"]
        X_train, X_test, y_train, y_test = train_test_split(
            train_test_pdf[feature_cols], train_test_pdf["label"], 
            test_size= 1 - config["train_test_ratio"],
            random_state=88,     # Ensures reproducibility
            shuffle=True,        # Shuffle the data before splitting
            stratify=train_test_pdf["label"]           # Stratify based on the label column
        )
    
        #Preprocess data
        # set up standard scalar preprocessing
        scaler = StandardScaler()
    
        transformer_stdscaler = scaler.fit(X_train) # Q which should we use? train? test? oot? all?
    
        # transform data
        X_train_processed = transformer_stdscaler.transform(X_train)
        X_test_processed = transformer_stdscaler.transform(X_test)
        # X_oot_processed = transformer_stdscaler.transform(X_oot)
    
        # Train model
        # Define the XGBoost classifier
        xgb_clf = xgb.XGBClassifier(eval_metric='logloss', random_state=88)
    
        # Define the hyperparameter space to search
        param_dist = {
            'n_estimators': [25, 50],
            'max_depth': [2, 3],  # lower max_depth to simplify the model
            'learning_rate': [0.01, 0.1],
            'subsample': [0.6, 0.8],
            'colsample_bytree': [0.6, 0.8],
            'gamma': [0, 0.1],
            'min_child_weight': [1, 3, 5],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 1.5, 2]
        }
    
        # Create a scorer based on AUC score
        auc_scorer = make_scorer(roc_auc_score)
    
        # Set up the random search with cross-validation
        random_search = RandomizedSearchCV(
            estimator=xgb_clf,
            param_distributions=param_dist,
            scoring=auc_scorer,
            n_iter=100,  # Number of iterations for random search
            cv=3,       # Number of folds in cross-validation
            verbose=1,
            random_state=42,
            n_jobs=-1   # Use all available cores
        )
    
        # Perform the random search
        random_search.fit(X_train_processed, y_train)
    
        # Output the best parameters and best score
        print("Best parameters found (Train): ", random_search.best_params_)
        print("Best AUC score (Train): ", random_search.best_score_)
    
        # Evaluate the model on the train set
        best_model = random_search.best_estimator_
        y_pred_proba = best_model.predict_proba(X_train_processed)[:, 1]
        train_auc_score = roc_auc_score(y_train, y_pred_proba)
        print("Train AUC score (Train): ", train_auc_score)
    
        # Evaluate the model on the test set
        best_model = random_search.best_estimator_
        y_pred_proba = best_model.predict_proba(X_test_processed)[:, 1]
        test_auc_score = roc_auc_score(y_test, y_pred_proba)
        print("Test AUC score: ", test_auc_score)
    
        print("TRAIN GINI score: ", round(2*train_auc_score-1,3))
        print("Test GINI score: ", round(2*test_auc_score-1,3))
    
        #Prepare model artefact to save
        model_artefact = {}
    
        model_artefact['model'] = best_model
        model_artefact['model_version'] = "credit_model_"+config["model_train_end_date_str"].replace('-','_')
        model_artefact['preprocessing_transformers'] = {}
        model_artefact['preprocessing_transformers']['stdscaler'] = transformer_stdscaler
        model_artefact['data_dates'] = config
        model_artefact['data_stats'] = {}
        model_artefact['data_stats']['X_train'] = X_train.shape[0]
        model_artefact['data_stats']['X_test'] = X_test.shape[0]
        model_artefact['data_stats']['y_train'] = round(y_train.mean(),2)
        model_artefact['data_stats']['y_test'] = round(y_test.mean(),2)
        model_artefact['results'] = {}
        model_artefact['results']['auc_train'] = train_auc_score
        model_artefact['results']['auc_test'] = test_auc_score
        model_artefact['results']['gini_train'] = round(2*train_auc_score-1,3)
        model_artefact['results']['gini_test'] = round(2*test_auc_score-1,3)
        model_artefact['hp_params'] = random_search.best_params_
    
        pprint.pprint(model_artefact)
    
        #save artefact to model bank
        # create model_bank dir
        model_bank_directory = "model_bank/"
    
        if not os.path.exists(model_bank_directory):
            os.makedirs(model_bank_directory)
    
        # Full path to the file
        file_path = os.path.join(model_bank_directory, model_artefact['model_version'] + '.pkl')
    
        # Write the model to a pickle file
        with open(file_path, 'wb') as file:
            pickle.dump(model_artefact, file)
    
        print(f"Model saved to {file_path}")

        # Save train-test validation results to Gold table
        new_metrics_df = spark.createDataFrame([(model_train_date_end, model_artefact['model_version'], float(model_artefact['results']['auc_train']), float(model_artefact['results']['auc_test']), float(model_artefact['results']['gini_train']), float(model_artefact['results']['gini_test']))], ["snapshot_date", "model_version", "auc_train", "auc_test", "gini_train", "gini_test"])

        model_metrics = "./datamart/gold/model_metrics_val"
        if not os.path.exists(model_metrics):
            os.makedirs(model_metrics)
        partition_name = model_train_date_end.replace('-','_') + '.parquet'
        filepath = os.path.join(model_metrics, partition_name)
        new_metrics_df.write.mode('overwrite').parquet(filepath)

    #############################################
    ### Make inference a month after training ###
    #############################################
    
    #train_test_end_date is on a loop
    month = int(model_train_date_end.split("-")[1])
    last_month = month - 1
    #Make inference only if Apr, Aug, Dec
    if last_month in [3, 7, 11]:

        #Load last month's model to infer on this month's unseen data
        month = config["train_test_end_date"] - relativedelta(months = 1)
        model_version = "credit_model_" + month.strftime("%Y-%m-%d").replace('-','_')
        print("Model Version ", model_version) 

        # initialize model_bank dir
        model_bank_directory = "model_bank/"

        # Full path to the model file
        file_path = os.path.join(model_bank_directory, model_version + '.pkl')
        
        # Load the model to get a scaler
        with open(file_path, 'rb') as file:
            loaded_model_artefact = pickle.load(file)
            print("Model loaded successfully! ", loaded_model_artefact['model_version'])

        config["oot_date"] = config["train_test_end_date"]
        # OOT is one month after training
        oot_pdf = data_pdf[(data_pdf['snapshot_date'] >= config["oot_date"].date()) & (data_pdf['snapshot_date'] <= config["oot_date"].date())]
        feature_cols = [fe_col for fe_col in data_pdf.columns if fe_col.startswith('fe_')]
    
        X_oot = oot_pdf[feature_cols]
        y_oot = oot_pdf["label"]
            
        transformer_stdscaler = loaded_model_artefact['preprocessing_transformers']['stdscaler']
        # Use the same scaler to transform OOT data
        X_oot_processed = transformer_stdscaler.transform(X_oot)
    
        y_pred_proba = loaded_model_artefact['model'].predict_proba(X_oot_processed)[:, 1]
        oot_auc_score = roc_auc_score(y_oot, y_pred_proba)
        oot_gini_score = round(2*oot_auc_score-1,3)
        print("OOT AUC score and date: ", config["train_test_end_date"], " ", oot_auc_score)            
        print("trainteststart", config["train_test_start_date"], config["train_test_end_date"])

        # Save inference results to Gold table
        new_metrics_df = spark.createDataFrame([(model_train_date_end, loaded_model_artefact['model_version'], float(oot_auc_score), float(oot_gini_score))], ["snapshot_date", "model_version", "auc_oot", "gini_oot"])

        model_metrics = "./datamart/gold/model_metrics_oot"
        if not os.path.exists(model_metrics):
            os.makedirs(model_metrics)
        partition_name = model_train_date_end.replace('-','_') + '.parquet'
        filepath = os.path.join(model_metrics, partition_name)
        new_metrics_df.write.mode('overwrite').parquet(filepath)

    spark.stop()

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="train and infer model")
    parser.add_argument("--train_start_date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--train_end_date", type=str, required=True, help="YYYY-MM-DD")

    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    model_train(args.train_start_date, args.train_end_date)