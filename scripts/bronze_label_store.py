import argparse
import os
import pyspark

from utils.data_processing_bronze_table import process_bronze_table

# to call this script: python bronze_label_store.py --snapshotdate "2023-01-01"

def main(snapshotdate):    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # load arguments
    date_str = snapshotdate
    
    # create bronze datalake
    bronze_lms_directory = "./datamart/bronze/lms/"
    bronze_clickstream_directory = "./datamart/bronze/clickstream/"
    bronze_attributes_directory = "./datamart/bronze/attributes/"
    bronze_financials_directory = "./datamart/bronze/financials/"

    # run data processing
    process_bronze_table('lms', date_str, bronze_lms_directory, spark)
    process_bronze_table('clickstream', date_str, bronze_clickstream_directory, spark)
    process_bronze_table('attributes', date_str, bronze_attributes_directory, spark)
    process_bronze_table('financials', date_str, bronze_financials_directory, spark)

    # end spark session
    spark.stop()
    
if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate)