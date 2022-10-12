
# Import SparkSession
from pyspark.sql import SparkSession
import os

# Get path to directory (we all have unique paths to the repo)
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Create SparkSession 
spark = SparkSession.builder \
      .master("local[1]") \
      .appName("dataProject") \
      .getOrCreate() 


# Read in data (this method felt easy to understand, but is verbose)
train_df = spark.read.csv(os.path.join(__location__) + "/store-sales-time-series-forecasting/train.csv")
holidays_events_df = spark.read.csv(os.path.join(__location__) + "/store-sales-time-series-forecasting/holidays_events.csv")
oil_df = spark.read.csv(os.path.join(__location__) + "/store-sales-time-series-forecasting/oil.csv")
stores_df = spark.read.csv(os.path.join(__location__) + "/store-sales-time-series-forecasting/stores.csv")
test_df = spark.read.csv(os.path.join(__location__) + "/store-sales-time-series-forecasting/test.csv")
transactions_df = spark.read.csv(os.path.join(__location__) + "/store-sales-time-series-forecasting/transactions.csv")



# If you want to see how it looks like
# train_df.show()
# transactions_df.show()
# oil_df.show()