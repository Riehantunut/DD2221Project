
# Import SparkSession
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, VectorIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit


import matplotlib.pyplot as plt
import os


def add_lag(df, n, col):
    # Appends n columns with the sale data shifted by n down
    # Not really used
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

def merge_oil_price_to_train(training_frame, oil_frame):
    # Add column with oil price to training dataframe. Oil price of the weekend is the Friday's price
    w = Window().partitionBy("family","store_nbr").orderBy("store_nbr","date")

    output = training_frame.join(oil_frame,['date'],how='left')

    output = output.withColumn(
        "dcoilwtico",
        F.when(
            F.col("dcoilwtico").isNull(),
            F.last("dcoilwtico", ignorenulls=True).over(w)
        ).otherwise(F.col("dcoilwtico"))
    )

    return output

def get_full_model(columns_to_OHE, columns_to_features, column_label, use_tvs, max_depth = 5, N_trees = 20):

    encoded_column_names = [x + "_encoded" for x in columns_to_OHE]
    
    # Create Pipeline
    # Index product family -> One Hot Encode Index -> Assemble Vector -> RF regressor
    stringInd = StringIndexer(inputCol='family', outputCol='family_index')
    
    OneHotEnc = OneHotEncoder(inputCols=columns_to_OHE, outputCols=encoded_column_names)
    
    vectorAssembler = VectorAssembler(inputCols = columns_to_features, outputCol = 'features')
    
    rf = RandomForestRegressor(featuresCol=vectorAssembler.getOutputCol(), labelCol=column_label,
                               maxDepth = max_depth, maxBins = 32, maxMemoryInMB = 256, numTrees= N_trees,
                               featureSubsetStrategy = 'auto')
    
    if use_tvs:
        paramGrid = ParamGridBuilder()\
        .addGrid(rf.maxDepth, [10, 20]) \
        .addGrid(rf.numTrees, [20, 50])\
        .addGrid(rf.maxBins, [32, 48])\
        .build()
        
        pipeline = Pipeline(stages=[stringInd, OneHotEnc, vectorAssembler, rf])
        
        tvs = TrainValidationSplit(estimator=pipeline,
                               estimatorParamMaps=paramGrid,
                               evaluator=RegressionEvaluator(),
                               trainRatio=0.8)
        return tvs
    
    return Pipeline(stages=[stringInd, OneHotEnc, vectorAssembler, rf])

def get_single_family_model(columns_to_OHE, columns_to_features, column_label, use_tvs, max_depth = 5, N_trees = 20):  
    # Create Pipeline
    # One Hot Encode Index -> Assemble Vector -> RF regressor
    encoded_column_names = [x + "_encoded" for x in columns_to_OHE]
    
    OneHotEnc = OneHotEncoder(inputCols=columns_to_OHE, outputCols=encoded_column_names)
    
    vectorAssembler = VectorAssembler(inputCols = columns_to_features, outputCol = 'features')
    
    rf = RandomForestRegressor(featuresCol=vectorAssembler.getOutputCol(), labelCol=column_label,
                               maxDepth = max_depth, maxBins = 32, maxMemoryInMB = 256, numTrees= N_trees,
                               featureSubsetStrategy = 'auto')
    
    if use_tvs:
        paramGrid = ParamGridBuilder()\
        .addGrid(rf.maxDepth, [10, 20]) \
        .addGrid(rf.numTrees, [20, 50])\
        .addGrid(rf.maxBins, [32, 48])\
        .build()
        
        pipeline = Pipeline(stages=[OneHotEnc, vectorAssembler, rf])
        
        tvs = TrainValidationSplit(estimator=pipeline,
                               estimatorParamMaps=paramGrid,
                               evaluator=RegressionEvaluator(),
                               trainRatio=0.8)
        return tvs
    
    return Pipeline(stages=[OneHotEnc, vectorAssembler, rf])

def get_single_family_single_store_model(columns_to_features, column_label, use_tvs, max_depth = 5, N_trees = 20):  
    # Create Pipeline
    # Assemble Vector -> RF regressor
    vectorAssembler = VectorAssembler(inputCols = columns_to_features, outputCol = 'features')
    
    rf = RandomForestRegressor(featuresCol=vectorAssembler.getOutputCol(), labelCol=column_label,
                               maxDepth = max_depth, maxBins = 32, maxMemoryInMB = 256, numTrees= N_trees,
                               featureSubsetStrategy = 'auto')
    
    if use_tvs:
        paramGrid = ParamGridBuilder()\
        .addGrid(rf.maxDepth, [10, 20]) \
        .addGrid(rf.numTrees, [20, 50])\
        .addGrid(rf.maxBins, [32, 48])\
        .build()
        
        pipeline = Pipeline(stages=[vectorAssembler, rf])
        
        tvs = TrainValidationSplit(estimator=pipeline,
                               estimatorParamMaps=paramGrid,
                               evaluator=RegressionEvaluator(),
                               trainRatio=0.8)
        return tvs
    return Pipeline(stages=[vectorAssembler, rf])
    

def main():
    #################################
    # Parameters for data processing
    single_family = False
    single_store = False
    grid_search_parameters = False # Using 'True' is dangerously slow for some reason, can't be the code I wrote, can it?
    
    # RandomForest Parameters, used when not using grid search
    n_trees = 50
    max_depth = 5
    
    # What single family/store to use when training
    store_number = 1
    family_name = "GROCERY I"
    
    # What store number and family to use when predicting
    store_number_pred = 1
    family_name_pred = "GROCERY I"
    
    # Not used anymore
    days_lagged = 0
    columns_lagged = ['sales']
    #################################
    
    if single_family and single_store:
        columns_to_OHE = []
        columns_to_features = ['Year','Month', 'DayOfWeek', 'onpromotion', 'dcoilwtico']
    elif single_family and not single_store:
        columns_to_OHE = ['store_nbr']
        columns_to_features = ['Year','Month', 'DayOfWeek', 'store_nbr_encoded', 'onpromotion', 'dcoilwtico']
    elif not single_family and not single_store:
        columns_to_OHE = ['family_index','store_nbr']
        columns_to_features = ['Year','Month', 'DayOfWeek', 'store_nbr_encoded', 'family_index_encoded', 'onpromotion', 'dcoilwtico']

    column_label = 'label'

    # Get path to directory (we all have unique paths to the repo)
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    # Create SparkSession 
    spark = SparkSession.builder \
          .master("local[1]") \
          .appName("dataProject") \
          .getOrCreate()
        
    # Load all relevant data
    train_df = spark.read.csv(os.path.join(__location__) + "/store-sales-time-series-forecasting/train.csv",inferSchema=True, header = True)
    oil_df = spark.read.csv(os.path.join(__location__) + "/store-sales-time-series-forecasting/oil.csv",inferSchema=True, header = True)
    test_df = spark.read.csv(os.path.join(__location__) + "/store-sales-time-series-forecasting/test.csv",inferSchema=True, header = True)
    
    
    # Add data of previous days to the df
    for col in columns_lagged:
        train_df = add_lag(train_df, days_lagged, col).na.fill(0)
        for i in range(days_lagged):
            columns_to_features.append(col + "_lag_" + str(i+1))
    
    # Add day of the week, month and year to the dataframe
    train_df = add_day_month(train_df)
    test_df  = add_day_month(test_df)
    
    # Join with oil prices
    train_df = merge_oil_price_to_train(train_df, oil_df)
    train_df = train_df.filter(train_df.dcoilwtico.isNotNull())
    
    test_df  = merge_oil_price_to_train(test_df, oil_df)
    test_df = test_df.filter(test_df.dcoilwtico.isNotNull())

    if single_family:
        train_df = train_df.filter(train_df.family == family_name)
        test_df = test_df.filter(test_df.family == family_name)
    
    if single_store:
        train_df = train_df.filter(train_df.store_nbr == store_number)
        test_df = test_df.filter(test_df.store_nbr == store_number)
    
    # TrainValidationSplit thing wants label as label 
    train_df = train_df.withColumnRenamed("sales", column_label)    
    
    if single_family and single_store:
        pipeline = get_single_family_single_store_model(columns_to_features, column_label, grid_search_parameters, max_depth, n_trees)
    elif single_family and not single_store:
        pipeline = get_single_family_model(columns_to_OHE, columns_to_features, column_label, grid_search_parameters, max_depth, n_trees)
    elif not single_family and not single_store:
        pipeline = get_full_model(columns_to_OHE, columns_to_features, column_label, grid_search_parameters, max_depth, n_trees)
    
    
    # Fit Model
    model = pipeline.fit(train_df)
    
    if grid_search_parameters:
        model = model.bestModel
    
    # Filter data to predict
    train_df = train_df.filter(train_df.store_nbr == store_number_pred)
    test_df = test_df.filter(test_df.store_nbr == store_number_pred)

    train_df = train_df.filter(train_df.family == family_name_pred)
    test_df = test_df.filter(test_df.family == family_name_pred)
    
    # Make Predictions
    predictions_train = model.transform(train_df)
    predictions_test = model.transform(test_df)

    # Displaying some stuff
    # predictions_train.select(column_label, 'prediction').show()
    # predictions_test.select(column_label, 'prediction').show()
    
    # Calculate RMS, print some diagnostics
    evaluator = RegressionEvaluator(labelCol=column_label, predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions_train)
    print("Root Mean Squared Error (RMSE) on train data = %g" % rmse)
    
    # Plotting stuff goes here
    dates_train = train_df.select(train_df.date).sort(train_df.date.asc()).collect()
    dates_test = test_df.select(test_df.date).sort(test_df.date.asc()).collect()
    actual_sales = train_df.select(column_label).sort(predictions_train.date.asc()).collect()
    predicted_sales_train = predictions_train.select('prediction').sort(predictions_train.date.asc()).collect()
    predicted_sales_test = predictions_test.select('prediction').sort(predictions_test.date.asc()).collect()

    # Plotting stuff
    fig, ax = plt.subplots()

    plt.plot(dates_train, actual_sales, c='b')
    plt.plot(dates_train, predicted_sales_train, c='orange')
    plt.plot(dates_test, predicted_sales_test, c='r')

    ax.set_ylabel('Sales')
    ax.set_xlabel('date')
    ax.set_title('Sales per date, Category: ' + family_name_pred + ', Store: ' + str(store_number_pred))

    plt.show()


if(__name__=='__main__'):
    main()
