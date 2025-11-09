import argparse
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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from utils.data_processing_bronze_table import process_bronze_table
from utils.data_processing_silver_table import process_silver_table
# from scripts.utils.data_processing_gold_table import process_labels_gold_table

# to call this script: python silver_label_store.py --snapshotdate "2023-01-01"

def main(snapshotdate):
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # initialize bronze datalake
    bronze_db = "datamart/bronze/"
    
    # create silver datalake
    silver_db = "datamart/silver/"

    silver_lms_directory = "./datamart/silver/lms/"
    silver_clickstream_directory = "./datamart/silver/clickstream/"
    silver_attributes_directory = "./datamart/silver/attributes/"
    silver_financials_directory = "./datamart/silver/financials/"

    # run data processing
    process_silver_table(snapshotdate, 'lms', bronze_db, silver_lms_directory, spark)
    process_silver_table(snapshotdate, 'clickstream', bronze_db, silver_clickstream_directory, spark)
    process_silver_table(snapshotdate, 'attributes', bronze_db, silver_attributes_directory, spark)
    process_silver_table(snapshotdate, 'financials', bronze_db, silver_financials_directory, spark)
    
    # end spark session
    spark.stop()
    

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate)