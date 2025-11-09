#!/usr/bin/env python
import argparse
import matplotlib.pyplot as plt
import os
import glob
import matplotlib.pyplot as plt
import pyspark
from pyspark.sql.functions import col, to_date

def read_table(gold_db, spark):
    """
    Helper function to read all partitions of a silver table
    """
    folder_path = os.path.join(gold_db)
    files_list = [os.path.join(folder_path, os.path.basename(f)) for f in glob.glob(os.path.join(folder_path, '*'))]
    df = spark.read.option("header", "true").parquet(*files_list)
    return df

def generate_plot(snapshotdate, spark, val_or_oot):
        
        if val_or_oot == "val":
            metrics_df = "./datamart/gold/model_metrics_val"

            df = read_table(metrics_df, spark)
            df = df.withColumn("snapshot_date", to_date(col("snapshot_date"), "yyyy-MM-dd"))
            df = df.orderBy(col("snapshot_date").asc())

            # Plot
            plt.figure(figsize=(12,6))
            plt.plot(df.toPandas()['snapshot_date'], df.toPandas()['auc_train'], label='auc_train', color='red', marker='o', linestyle='-', linewidth=2)
            plt.plot(df.toPandas()['snapshot_date'], df.toPandas()['auc_test'], label='auc_test', color='red', marker='o', linestyle='dotted', linewidth=2)
            plt.plot(df.toPandas()['snapshot_date'], df.toPandas()['gini_train'], label='gini_train', color='green', marker='o', linestyle='-', linewidth=2)
            plt.plot(df.toPandas()['snapshot_date'], df.toPandas()['gini_test'], label='gini_test', color='green', marker='o', linestyle='dotted', linewidth=2)
            plt.title("Train and Test Metrics Over Time", fontsize=14)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Score", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()

        else:
            metrics_df = "./datamart/gold/model_metrics_oot"
            df = read_table(metrics_df, spark)
            df = df.withColumn("snapshot_date", to_date(col("snapshot_date"), "yyyy-MM-dd"))
            df = df.orderBy(col("snapshot_date").asc())

            # Plot
            plt.figure(figsize=(12,6))
            plt.plot(df.toPandas()['snapshot_date'], df.toPandas()['auc_oot'], label='auc_oot', color='red', marker='o', linestyle='-', linewidth=2)
            plt.plot(df.toPandas()['snapshot_date'], df.toPandas()['gini_oot'], label='gini_oot', color='green', marker='o', linestyle='-', linewidth=2)
            plt.title("OOT Metrics Over Time", fontsize=14)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Score", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()

        # Save to shared Airflow-accessible folder
        output_path = "./images/model_performance_trend_" + val_or_oot + snapshotdate + ".png"
        if not os.path.exists( "./images/"):
            os.makedirs( "./images/")
        plt.savefig(output_path)
        print(f"Saved model performance trend chart to {output_path}")

def visualize(snapshotdate):
    # if snapshotdate in ["2023-04-01", "2023-08-01","2023-12-01", "2024-04-01", "2024-08-01","2024-12-01"]:
    if snapshotdate in ["2024-12-01"]:
        # Need to start a new Spark session when we want to read parquet, because we closed it in gold_label_store.py
        spark = pyspark.sql.SparkSession.builder \
            .appName("dev") \
            .master("local[*]") \
            .getOrCreate()

        # Set log level to ERROR to hide warnings
        spark.sparkContext.setLogLevel("ERROR")
        
        # Generate plot for val and oot
        generate_plot(snapshotdate, spark, "val")
        generate_plot(snapshotdate, spark, "oot")

        spark.stop()

if __name__ == "__main__":    
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="visualize")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    visualize(args.snapshotdate)