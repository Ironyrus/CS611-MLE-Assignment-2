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
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, MapType


from collections import Counter

##################
### Attributes ###
##################
def process_df_attributes(df):

    # 0 or 1 [- or +], 0 or more digits, 0 or 1 [.], 1 or more digits
    numeric_regex = r'([-+]?\d*\.?\d+)'
    # Extract numeric part from string in 'Age' column
    df = df.withColumn("age", F.regexp_extract(col("age"), numeric_regex, 1))

    # Define column data types
    columns = {
        'customer_id': StringType(),
        'name': StringType(),
        'age': IntegerType(),
        'ssn': StringType(),
        'occupation': StringType(),
        'snapshot_date': DateType()
    }

    # Cast columns to the proper data type
    for column, new_type in columns.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # Enforce valid age constraints, 0 to 120
    df = df.withColumn(
        "age",
        F.when((col("age") >= 0) & (col("age") <= 120), col("age"))  # keep valid
        .otherwise(55)  # input 55 if age is invalid
    )

    # Enforce valid SSN
    # ^ means start of String, 3 digits, -, 2 digits, -, 4 digits, $ end of String, (...) shows group
    df = df.withColumn(
        "ssn",
        F.regexp_extract(col("ssn"), r'^(\d{3}-\d{2}-\d{4})$', 1)
    )
    df = df.withColumn(
        "ssn",
        F.when(col("ssn") == "", '000-00-0000').otherwise(col("ssn"))
    )

    # Null empty occupation
    df = df.withColumn(
        "occupation",
        F.when(col("occupation") == "_______", 'Unemployed').otherwise(col("occupation"))
    )
    return df

###################
### Clickstream ###
###################
def process_df_clickstream(df):
    # Define column data types
    # First line creates a dict of fe_1 : IntegerType()...
    # ** unpacks the dict
    columns = {
        **{f'fe_{i}': IntegerType() for i in range(1, 21)},
        'customer_id': StringType(),
        'snapshot_date': DateType()
    }

    # key, value
    for column, new_type in columns.items():
        df = df.withColumn(column, col(column).cast(new_type))
    return df

##################
### Financials ###
##################
# Replacing String in Credit_History_Age to number of months
def str_to_mths(col_value):
    col_value_list = col_value.split(" ")
    year_in_mths = int(col_value_list[0]) * 12
    mths = year_in_mths + int(col_value_list[3])
    return int(mths)

def process_df_financials(df, spark):

    numeric_regex = r'([-+]?\d*\.?\d+)'
    
    columns = {
        'annual_income': FloatType(),
        'monthly_inhand_salary': FloatType(),
        'num_bank_accounts': IntegerType(),
        'num_credit_card': IntegerType(),
        'interest_rate': IntegerType(),
        'num_of_loan': IntegerType(),
        'delay_from_due_date': IntegerType(),
        'num_of_delayed_payment': IntegerType(),
        'changed_credit_limit': FloatType(),
        'num_credit_inquiries': FloatType(),
        'outstanding_debt': FloatType(),
        'credit_utilization_ratio': FloatType(),
        'total_emi_per_month': FloatType(),
        'amount_invested_monthly': FloatType(),
        'monthly_balance': FloatType()
    }

    # Cast columns to the proper data type
    for col_name, new_type in columns.items():
        df = df.withColumn(col_name, F.regexp_extract(col(col_name), numeric_regex, 1))
        df = df.withColumn(col_name, col(col_name).cast(new_type))

    str_to_mths_udf = F.udf(str_to_mths, IntegerType())
    # Apply UDF to column
    df = df.withColumn("Credit_History_Age_Mths", str_to_mths_udf(col("Credit_History_Age")))    
    
    # Remove erroneous negative values
    for column_name in ['num_of_loan', 'delay_from_due_date', 'num_of_delayed_payment']:
        df = df.withColumn(
            column_name,
            F.when(col(column_name) >= 0, col(column_name))  # keep valid
            .otherwise(None)  # redact invalid
    ) 

    return df


###########
### LMS ###
###########

def process_df_lms(df):
    column_type_map = {
        "loan_id": StringType(),
        "customer_id": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    # Cast columns to proper data type
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))
        
    #MOB = Month on Book
    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    return df

############################
### Process Silver Table ###
############################
def process_silver_table(snapshot_date_str, table_name, bronze_db, silver_db, spark):

    if not os.path.exists(silver_db):
        os.makedirs(silver_db)
    
    # connect to bronze table
    partition_name = table_name + "_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = os.path.join(bronze_db, table_name, partition_name)
    df = spark.read.csv(filepath, header=True, inferSchema=True)

    df = df.toDF(*[col_name.lower() for col_name in df.columns])
    
    if table_name == "attributes":
        df = process_df_attributes(df)
    elif table_name == 'clickstream':
        df = process_df_clickstream(df)
    elif table_name == "financials":
        df = process_df_financials(df, spark)   
    elif table_name == "lms":
        df = process_df_lms(df)
    else:
        raise ValueError("Table does not exist!")

    # Save silver table
    partition_name = table_name + "_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = os.path.join(silver_db, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    return df