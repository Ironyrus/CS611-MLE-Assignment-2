import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from tqdm import tqdm

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, MapType, NumericType, ArrayType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer, VectorAssembler, StandardScaler

def one_hot_encoder(df, category_col):
    """
    Utility function for one hot encoding
    """
    # Get label encoding
    indexer = StringIndexer(inputCol=category_col, outputCol=f"{category_col}_index", handleInvalid="keep")
    indexer_model = indexer.fit(df)
    df = indexer_model.transform(df)

    # Transform into one hot encoding
    encoder = OneHotEncoder(inputCol=f"{category_col}_index", outputCol=f"{category_col}_ohe", dropLast=False)
    df = encoder.fit(df).transform(df)
    vector_to_array_udf = F.udf(lambda v: v.toArray().tolist(), ArrayType(FloatType()))
    df = df.withColumn(f"{category_col}_array", vector_to_array_udf(f"{category_col}_ohe"))

    # Split into columns
    categories = [cat.lower() for cat in indexer_model.labels]

    for i, cat in enumerate(categories):
        df = df.withColumn(f"{category_col}_{cat}", df[f"{category_col}_array"][i])
        df = df.withColumn(f"{category_col}_{cat}", col(f"{category_col}_{cat}").cast(IntegerType()))

    # Optional: drop intermediate columns
    df = df.drop(category_col, f"{category_col}_index", f"{category_col}_ohe", f"{category_col}_array")
    return df
    
def read_silver_table(table_name, silver_db, spark):
    folder_path = os.path.join(silver_db, table_name)
    files_list = [os.path.join(folder_path, os.path.basename(f)) for f in glob.glob(os.path.join(folder_path, '*'))]
    df = spark.read.option("header", "true").parquet(*files_list)
    return df

###################
### Label Store ###
###################
def build_label_store(mob, dpd, df):
    # get customer at mob
    # df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "customer_id", "label", "label_def", "snapshot_date")

    return df

#####################
### Feature Store ###
#####################
def build_feature_store(df_attributes, df_financials, df_clickstream, df_lms, df_label):

    # Join attributes table and financials table
    df_joined = df_attributes.join(df_financials, on=["customer_id", "snapshot_date"], how="inner")
    # drop identifiers and duplicated columns
    df_joined = df_joined.drop("name", "ssn", "type_of_loan", "credit_history_age", "type_of_loan") 
    # filter by user IDs that have labels
    df_joined = df_joined.join(df_label.select("customer_id"), on="customer_id", how="left_semi") 
    print("1. Joined dataframes")

    # Turn categorical variables into one hot encoded columns
    df_joined = one_hot_encoder(df_joined, "occupation")
    df_joined = one_hot_encoder(df_joined, "payment_of_min_amount")
    df_joined = one_hot_encoder(df_joined, "credit_mix")
    print("2. Performed one-hot encoding")

    # Aggregate mean clickstream data for each user
    # Filter clickstream data
    # df_lms_mob0 = df_lms.filter(col("mob") == 0) # creates new df where mob is 0, drops rows otherwise
    df_lms_mob0 = df_lms
    df_lms_mob0 = df_lms_mob0.withColumnRenamed("snapshot_date", "mob_date") # rename snapshot_date to mob_date
    df_lms_mob0 = df_lms_mob0.select("customer_id", "mob_date") # keep only customer_id and mob_date cols 
    df_clickstream_filtered = df_clickstream.join(df_lms_mob0, on="customer_id", how="inner") # filter by user IDs that have labels
    df_clickstream_filtered = df_clickstream_filtered.filter(col("snapshot_date") <= col("mob_date"))

    # Do mean aggregation
    agg_exprs = [F.avg(f'fe_{i}').alias(f"avg_fe_{i}") for i in range(1, 21)]
    df_clickstream_filtered = df_clickstream_filtered.groupBy("customer_id").agg(*agg_exprs)
    print("3. Processed clickstream data")

    # Join clickstream data with attributes and financials
    df_joined = df_joined.join(df_clickstream_filtered, on=["customer_id"], how="left")
    print("4. Joined clickstream data with the rest of the features")
    return df_joined
    
def process_labels_gold_table(date_str, silver_db, spark, mob, dpd):    
        # Read silver tables
        df_attributes  = read_silver_table('attributes', silver_db, spark)
        df_clickstream = read_silver_table('clickstream', silver_db, spark)
        df_financials = read_silver_table('financials', silver_db, spark)
        df_lms = read_silver_table('lms', silver_db, spark)
    
        # create Gold datalake
        gold_label_store_directory = "datamart/gold/label_store/"
        gold_features_store_directory = "datamart/gold/feature_store/"

        if not os.path.exists(gold_label_store_directory):
            os.makedirs(gold_label_store_directory)
        if not os.path.exists(gold_features_store_directory):
            os.makedirs(gold_features_store_directory)

        # Build labels
        print("Building label store")
        df_label = build_label_store(mob, dpd, df_lms)

        # Build features
        print("Building features")
        df_features = build_feature_store(df_attributes, df_financials, df_clickstream, df_lms, df_label)

        # Saving features
        partition_name = date_str.replace('-','_') + '.parquet'
        feature_filepath = os.path.join('datamart/gold/', 'feature_store', partition_name)
        df_features = df_features.filter(col('snapshot_date')==date_str)
        df_features.write.mode('overwrite').parquet(feature_filepath)

        # Saving labels
        partition_name = date_str.replace('-','_') + '.parquet'
        label_filepath = os.path.join('datamart/gold/', 'label_store', partition_name)
        df_label = df_label.filter(col('snapshot_date')==date_str)
        df_label.write.mode('overwrite').parquet(label_filepath)

        return df_features, df_label