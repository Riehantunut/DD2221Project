
# Import SparkSession
from pyspark.sql import SparkSession
from pyspark.sql.functions import lag
from pyspark.sql.window import Window
import os

# Get path to directory (we all have unique paths to the repo)
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Create SparkSession 
spark = SparkSession.builder \
      .master("local[1]") \
      .appName("dataProject") \
      .getOrCreate() 

def add_lag(df, n):
    # Appends n columns with the sale data shifted by n down
    w = Window().partitionBy("family","store_nbr").orderBy("store_nbr","date")
    
    for i in range(n):
        col_name = "sales_lag_" + str(i+1)
        df = df.withColumn(col_name,lag("sales", i+1).over(w))
    
    return df

# Read in data (this method felt easy to understand, but is verbose)
train_df = spark.read.csv(os.path.join(__location__) + "/store-sales-time-series-forecasting/train.csv",inferSchema=True, header = True)
holidays_events_df = spark.read.csv(os.path.join(__location__) + "/store-sales-time-series-forecasting/holidays_events.csv",inferSchema=True, header = True)
oil_df = spark.read.csv(os.path.join(__location__) + "/store-sales-time-series-forecasting/oil.csv",inferSchema=True, header = True)
stores_df = spark.read.csv(os.path.join(__location__) + "/store-sales-time-series-forecasting/stores.csv",inferSchema=True, header = True)
test_df = spark.read.csv(os.path.join(__location__) + "/store-sales-time-series-forecasting/test.csv",inferSchema=True, header = True)
transactions_df = spark.read.csv(os.path.join(__location__) + "/store-sales-time-series-forecasting/transactions.csv",inferSchema=True, header = True)

# Add 3 columns with sales of previous days, fill null values with 0
train_df = add_lag(train_df, 3).na.fill(0)

# Show result
train_df.filter("store_nbr = 5 and family = 'EGGS'")\
        .show()

# If you want to see how it looks like
# train_df.show()
# transactions_df.show()
# oil_df.show()