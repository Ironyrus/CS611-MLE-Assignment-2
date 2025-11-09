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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_bronze_table(table_name, snapshot_date_str, bronze_lms_directory, spark):
    
    if not os.path.exists(bronze_lms_directory):
        os.makedirs(bronze_lms_directory)
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to source back end - IRL connect to back end source system
    if table_name == 'lms':
        csv_file_path = "data/lms_loan_daily.csv"
    elif table_name == 'clickstream':
        csv_file_path = "data/feature_clickstream.csv"
    elif table_name == 'attributes':
        csv_file_path = "data/features_attributes.csv"
    else:
        csv_file_path = "data/features_financials.csv"

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)

    # save bronze table to datamart - IRL connect to database to write
    partition_name = table_name + "_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)

    return df