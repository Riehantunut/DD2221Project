
# Import SparkSession
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, VectorIndexer
from pyspark.ml.regression import RandomForestRegressor


import matplotlib.pyplot as plt
import os

def add_lag(df, n, col):
    # Appends n columns with the sale data shifted by n down
    w = Window().partitionBy("family","store_nbr").orderBy("store_nbr","date")
    
    for i in range(n):
        col_name = col + "_lag_" + str(i+1)
        df = df.withColumn(col_name, F.lag(col, i+1).over(w))
    
    return df

def add_day_month(df):
    df = df.withColumn("Year", F.year(df.date))\
           .withColumn("Month", F.month(df.date))\
           .withColumn("DayOfWeek", F.dayofweek(df.date))
    return df

def main():
    # Parameters
    days_lagged = 7
    columns_lagged = ['sales']
    columns_to_features = ['Year','Month','DayOfWeek','family_encoded','store_nbr','onpromotion']
    column_label = 'sales'

    # Get path to directory (we all have unique paths to the repo)
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    # Create SparkSession 
    spark = SparkSession.builder \
          .master("local[1]") \
          .appName("dataProject") \
          .getOrCreate()
        
    # Read in data (this method felt easy to understand, but is verbose)
    train_df = spark.read.csv(os.path.join(__location__) + "/store-sales-time-series-forecasting/train.csv",inferSchema=True, header = True)
    holidays_events_df = spark.read.csv(os.path.join(__location__) + "/store-sales-time-series-forecasting/holidays_events.csv",inferSchema=True, header = True)
    oil_df = spark.read.csv(os.path.join(__location__) + "/store-sales-time-series-forecasting/oil.csv",inferSchema=True, header = True)
    stores_df = spark.read.csv(os.path.join(__location__) + "/store-sales-time-series-forecasting/stores.csv",inferSchema=True, header = True)
    test_df = spark.read.csv(os.path.join(__location__) + "/store-sales-time-series-forecasting/test.csv",inferSchema=True, header = True)
    transactions_df = spark.read.csv(os.path.join(__location__) + "/store-sales-time-series-forecasting/transactions.csv",inferSchema=True, header = True)

    # Add data of previous days to the df
    for col in columns_lagged:
        train_df = add_lag(train_df, days_lagged, col).na.fill(0)
        for i in range(days_lagged):
            columns_to_features.append(col + "_lag_" + str(i+1))
    
    # Add day of the week, month and year to the dataframe
    train_df = add_day_month(train_df)
    
    # Create Pipeline
    # Index product family -> One Hot Encode Index -> Assemble Vector -> RF regressor
    stringInd = StringIndexer(inputCol='family', outputCol='family_index')
    
    OneHotEnc = OneHotEncoder(inputCol=stringInd.getOutputCol(), outputCol='family_encoded')
    
    vectorAssembler = VectorAssembler(inputCols = columns_to_features, outputCol = 'features')
    
    rf = RandomForestRegressor(featuresCol=vectorAssembler.getOutputCol(), labelCol=column_label,
                               maxDepth = 5, maxBins = 32, maxMemoryInMB = 256, numTrees= 20,
                               featureSubsetStrategy = 'auto')
    
    pipeline = Pipeline(stages=[stringInd, OneHotEnc, vectorAssembler, rf])
    
    # Fit Model
    model = pipeline.fit(train_df)
    
    # Make Prediction (not on the training data ideally)
    predictions = model.transform(train_df)

    # Displaying some stuff
    predictions.select('id','sales', 'prediction').show()
    
    # Calculate RMS, print some diagnostics
    evaluator = RegressionEvaluator(labelCol="sales", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
    rfModel = model.stages[1]
    print(rfModel)
    
    # Get lists of first elements to plot
    tmp1 = train_df.select('sales').take(5000)
    tmp2 = predictions.select('prediction').take(5000)

    # Plotting stuff
    fig, ax = plt.subplots()

    plt.plot(tmp1)
    plt.plot(tmp2)

    ax.set_ylabel('Sales')
    ax.set_xlabel('id')
    ax.set_title('Sales per id')

    plt.show()

if(__name__=='__main__'):
    main()
