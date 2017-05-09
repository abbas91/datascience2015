>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>                          >
>         PySpark          >
>                          > 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>


"""
Apache Spark comes with an interactive shell for python as it does for Scala. 
The shell for python is known as “PySpark”. To use PySpark you will have to have python installed on your machine. 
As we know that each Linux machine comes preinstalled with python so you need not worry about python installation. 
To get started in a standalone mode you can download the pre-built version of spark from its official home page listed in 
the pre-requisites section of the PySpark tutorial. Decompress the downloaded file. On decompressing the spark downloadable, 
you will see the following structure:
"""

Spark 2.0 architecture

Spark Session - {SparkConf, SparkContext, SQLContext, HiveContext, StreamingContext}
                # Entry point for reading data, working with meta data, configuring the session, managing the cluster resourse

#####################
# Execution Process #
#####################


#####################   Distribute collection of JVM object
#        RDD        #   Fuctional Operators (map,filter,etc)
#####################
          |
          |
#####################   Distribute collection of row objects
#    Data Frames    #   Expression-based operations and UDFs
#####################   Logical plans and optimizer --- Fast/Efficient internal presentation ----------- Good for Python
          |
          |
#####################   internally rows, externally JVM objects
#      Data Set     #   Almost the "Best of both world - type safe + fast"
#####################   But slower than DF Not as good for interactive analysis, especially Python ----- Good for Scala/Java


######################   
# Catalyst Optimizer #   
######################


#####################
#  Project Tungsten #
#####################


########## Local Files ##############

bin    
"Holds all the binaries"

conf
"Holds all the necessary configuration files to run any spark application"

ec2
"Holds the scripts to launch a cluster on amazon cloud space with multiple ec2 instances"

lib 
" Holds the prebuilt libraries which make up the spark APIS"

licenses 

python

python API

README.md 

"Holds important instructions to get started with spark"

"Holds important startup scripts that are required to setup distributed cluster"

CHANGES.txt 

"Holds all the changes information for each version of  apache spark"

data

"Holds data that is used in the examples"

examples 

"Has examples which are a good place to learn the usage of spark functions."

LICENSE  NOTICE   

"Important information"

R

"Holds API of R language"

RELEASE   

"Holds make info of the downloaded version"





"
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Resilient Distributed Datasets [RDD] 
# Creating RDD


# Transformation


# Action

---------------------------------------------------- Quick Code Notes ---------------

spark "sparkSession"
spark.sparkContext

# loading data to RDD
dataRDD = spark.sparkContext.textFile('wasb:///example/data/fruits.txt')

data = np.array([1,1,1,2,3,4,5,5,5,6,6,7,8,9,9,10,10,10]) # array
data = [1,2,3,4,5,6] # list
RDD = spark.sparkContext.parallelize(data)

# ------------------------ Transformations 

data2Transf = data2RDD.map(lambda x: x * 100)
"[0,1,2,3,4] -> [0, 100, 200, 300, 400]"
data3Transf = data3RDD.flatMap(lambda x: x * 5)
"[[0,1],[2,3,4]] -> [0, 100, 200, 300, 400]" 

data5 = [1,2,3,4,5]
data5_1 = [6,7,8,9,10]
data5RDD = spark.sparkContext.parallelize(data5)
data5_1RDD = spark.sparkContext.parallelize(data5_1)
data5_allRDD = data5RDD.union(data5_1RDD)
"[1,2,3,4,5,6,7,8,9,10]"


data4 = np.array([1,1,1,2,3,4,5,5,5,6,6,7,8,9,9,10,10,10])
data4RDD = spark.sparkContext.parallelize(data4)
data4Transf = data4RDD.distinct()
"[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"


data5 = [1,2,3,4,5,6,7] 
data5_1 = [4,5,6,7,8,9,10]
data5RDD = spark.sparkContext.parallelize(data5)
data5_1RDD = spark.sparkContext.parallelize(data5_1)
data5_allRDD = data5RDD.intersection(data5_1RDD)
"[4,5,6,7]"


data5 = [1,2,3] 
data5_1 = [4,5,6]
data5RDD = spark.sparkContext.parallelize(data5)
data5_1RDD = spark.sparkContext.parallelize(data5_1)
data5_allRDD = data5RDD.cartesian(data5_1RDD)
"[(1, 4), (1, 5), (1, 6), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6)]"


def MultiplyTen(num):
    return num * 10
data7 = [1,7,7,9,1,2,4,4,4,1,5,5,1]
data7RDD = spark.sparkContext.parallelize(data7)
data7RDD.foreach(MultiplyTen) # apply function to each element and return nothing
data7RDD.collect()
"[1, 7, 7, 9, 1, 2, 4, 4, 4, 1, 5, 5, 1]"


# ----------------------------------- Actions 

data6 = [0,1,2,3,4,5,6,7,8,9]
data6RDD = spark.sparkContext.parallelize(data6)
sampleRDD = data6RDD.sample(True, 2.0, 123) # Replacement = T, sample 200%, seed = 123
"[0, 0, 0, 0, 0, 1, 1, 3, 3, 3, 4, 4, 4, 6, 6, 6, 9]"
sampleRDD = data6RDD.takeSample(True, 50) # Replacement = T, sample 50 cases
"[1, 6, 8, 9, 7, 6, 8, 2, 5, 4, 0, 1, 5, 1, 1, 4, 8, 8, 0, 8, 7, 2, 3, 5, 7, 1, 4, 9, 3, 6, 4, 1, 0, 0, 4, 3, 6, 3, 8, 3, 3, 2, 2, 5, 2, 1, 0, 0, 4, 9]"


data7 = [1,1,2,4,1,5,1]
data7RDD = spark.sparkContext.parallelize(data7)
data7RDD.reduce(lambda x,y: x + y)
"15"


data7 = [1,7,7,9,1,2,4,4,4,1,5,5,1]
data7RDD = spark.sparkContext.parallelize(data7)
data7RDD.countByValue() # Frequency table
"defaultdict(<type 'int'>, {1: 4, 2: 1, 4: 3, 5: 2, 7: 2, 9: 1})"



dataPair = spark.sparkContext.parallelize({('a',1),('b',2),('a',2),('b',7)})
dataPair.reduceByKey(lambda x,y: x + y).collect()
dataPair.groupByKey() ""



# cache with different choice
from pyspark import StorageLevel

data5 = [1,2,3,4,5] 
data5_1 = [6,7,8,9,10]
data5RDD = spark.sparkContext.parallelize(data5)
data5_1RDD = spark.sparkContext.parallelize(data5_1)
data5_allRDD = data5RDD.union(data5_1RDD)
data5_allRDD.persist(StorageLevel.MEMORY_ONLY_SER)
data5_allRDD.unpersist()









"
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Data Frame [DF] 
" One of the main reason why Python is initially slower within Spark is due to the communication layer between Python sub-processinges and the JVM "
" For Python users, DataFrame has a Python wrapper around Scala DataFrames that avoids the Python sub-process.JVM communication overhead "
# ---- Create DataFrame
from pyspark.sql.types import *
DF = spark.read.json(JSON_File) # From outside file
# - OR
Data = sc.parallelize([xxxxx]) # Frome internal data
Schema = StructType([
	StructField("id", LongType(), True),
	StructField("name", StringType(), True)
	]) # name, datatype, nullable
DF = spark.createDataFrame(Data, Schema)
# Create temparoal view from DF
DF.createOrReplaceTempView("DF_table")
DF_table.printSchema()
# - OR
FilePath = "/xxx/xxx/xxx.csv"
DF = spark.read.csv(FilePath, header = 'true', inferSchema='true', sep='\t')
DF.createOrReplaceTempView("DF_table")
DF.cache() # query runs faster



# --- Querying with Data Frame API
DF_table.count() # number of rows
DF_table.collect()
DF_table.show(n)
DF_table.take(n)
DF_table.select("id", "age").filter("age = 22").show()
DF_table.select(Data_table.id, Data_table.age).filter(Data_table == 22).show()
DF_table.select("name", "eyeColor").filter("eyeColor like 'b%'").show()
# In SQL
spark.sql("select count(1) from DF_table").show()
spark.sql("select id, age from DF_table where age = 22").show()
spark.sql("select name, eyeColor from DF_table where eyeColor like 'b%'").show()
spark.sql("""
	select a.city,
	       f.origin,
	       sum(f.delay) as Delays
	from DF_table1 a 
	join DF_table2 f
	on a.ID = f.ID
	where a.State = 'WA'
	group by a.city, f.origin
	order by sum(f.delay) desc """).show()


# Manipulating the Data Before modeling
DF = spark.createdataFrame([
	(1, 144.5, 5.9, 33, 'M'),
	(2, 121.5, 3.9, 67, 'F'),
	(3, 194.5, 2.1, 29, 'M'),
	(......................)], ['id','weight','height','age','gender'])




-> "Duplication"

DF.count()
DF.distinct().count # deduplicate exactly same rows
DF.dropDuplicates() # drop those duplicated rows

DF.select([c for c in DF.columns if c != 'id']).distinct().count() # check duplication ignoring column 'id'
DF = DF.dropDuplicates(subset=[c for c in DF.columns if c != 'id']) # drop duplicated rows ignoring column 'id'

import pyspark.sql.functions as fu
DF.agg(
	fn.count('id').alias('count'),
	fn.countDistinct('id').alias('distinct')
	).show() # Calculate total distinct number of each column

DF.withColumn('new_id',fn.monotonically_increasing_id()).show() # Add a new IDs for each row




-> "Missing Value"

DF.rdd.map(
	lambda row: (row['id'], sum([c == None for c in row]))
	).collect() # Find out missing observation per row
DF.where('id == 3').show() # Spot check 

DF.agg(*[
	(1 - (fn.count(c) / fn.count('*'))).alias(c + '_missing')
	for c in DF.columns
	]).show() # find % missing in each column

DF = DF.select([c for c in DF.columns if c != 'income']) # Drop columns

DF.dropna(thresh=3).show() # drop rows with a missing observation threshold

imputation_dir = DF.agg(*[fn.mean(c).alias(c) for c in DF.columns if c != 'gender']).toPandas().to_dict('records')[0] # generate a dict to store impute values
imputation_dir['gender'] = 'missing'
DF.fillna(means).show() # impute data DF with Dictionaery ('colnames': 'impute value')




-> "Outlier"
cols = ['weight', 'hieght', 'age']
bounds = {}

for col in cols:
	quantiles = DF.approxQuantile(col, [0.25, 0.75], 0.05) # colnames, 1st - 3rd Qt, acceptible error level

	IQR = quantiles[1] - quantiles[0]

	bounds[col] = [
	    quantiles[0] - 1.5 * IQR
        quantiles[1] + 1.5 * IQR
	]
" {'age': [9.0, 51.0], " # colname, lower and uper bounds
"  ..................  "
"  ..........., 71.0]} "

outliers = DF.select(*['id'] + [
	(
		(DF[c] < bounds[c][0]) | # less lower or higher than upper 
		(DF[c] > bounds[c][1])
	).alias(c + '_o') for c in cols

	])
outliers.show() # columns of T,F - outliers?

DF = DF.join(outliers, on = 'id')
DF.filter('weight_o').select('id','weight').show() # filter out outliers on 'weight'




-> "Describe the Data"
import pyspark.sql.types as typ # express all data type - IntegerType(), floatType()

DF.groupby('gender').count().show()
# OR
numeric = ['var1','var2','var3']
desc = DF.describe(numeric)
desc.show()
# OR
DF.agg({'var1': 'skewness'}).show() # a list of aggregation function like 'skewness'
" avg, count, countDistinct, fisrt, kurtosis, max, mean, min, shewness, stddev, stddev_pop, stddev_samp, sum, sumDistinct, var_pop, var_samp, variance"

DF.corr('var1','var2') # correlation


-> "Visualization - matplotlib & bokeh"
"[1] - aggregate data in workers, return aggregated list of bins and counts in each bin of histgram to the driver "
"[2] - Return all data points to the driver and allow the ploting libraries to do the jobs "
"[3] - Sample your data and then return them to the driver and allow the ploting libraries to do the jobs "

%matplotlib inline # make sure plot appear within the notebook
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import bokeh.charts as chrt
from bokeh.io import output_notebook
output_notebook() # make sure plot appear within the notebook

# Histgram - aggregate in workers
hists = DF.select('var1').rdd.flatMap(
	lambda row: row
	).histogram(20)
# plot dictionary
data = {
	'bins': hists[0][:-1],
	'freq': hists[1]
}
# plotting
plt.bar(data['bins'], data['freq'], width=2000)
plt.title('Histogram of \'balance\'')












"
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> MLlib 
" MLlib mainly operates on RDD"

--> "loading and transforming data"

import pyspark.sql.types as typ
labels = [
    ('var1', typ.StringType()),
    ('var2', typ.IntegerType()),
    ('var3', typ.StringType()),
    ('var4', typ.IntegerType()),
    ('var5', typ.StringType()),
    ('var6', typ.IntegerType()),
    ('var7', typ.StringType())
]

schema = typ.StructType([
	typ.StructField(e[0],e[1],False) for e in labels
	])

DF = spark.read.csv('births_train.csv.gz',
	                 header=True,
	                 schema=schema)



# select features and recode features

select_features = [

    'var1',
    'var2',
    'var4',
    'var7'
]

DF_new = DF.select(select_features) # create a new DF

import pyspark.sql.functions as func # canot use Python function directly on DF, it needs to be converted to UDF

# recode 99 to 0, else to self
def correct_cig(feat):
	return func.when(func.col(feat) != 99, func.col(feat)).otherwise(0) # func - when value != 99, self, otherwise = 0

DF_transf1 = DF_new \
     .withColumn('var4', correct_cig('var4')) \
     .withColumn('var7', correct_cig('var7'))  # Takes the names of column and the transformation function

# recode 'Y' to 1, else to 0
recode_dictionary = {

	'YNU': {
	    'Y': 1,
	    'N': 0,
	    'U': 0
	}
}

def recode(col, key):
	return recode_dictionary[key][col] # Python fun that looks up key and col to rerturn the assigned values in dict

rec_integer = func.udf(recode, typ.IntegerType()) # UDF defining - function names, output data type

cols = [(col.name, col.dataType) for col in DF_new.schema] # we create a list of tuple holds col names and data type
YNU_cols = []
for i, s in enumerate(cols): # we loop through all these cols 
	if s[1] == typ.StringType(): # and for all string columns
		dis = DF.select(s[0]).distinct().rdd.map(lambda row: row[0]).collect() # calculate distinct values 
	    if 'Y' in dis: # if 'Y' in the values
	    	YNU_cols.append(s[0]) # we append the col names to this list

# Illustrate how DF can transform the features in bulk while selecting features.
DF.select(['var1',
	      rec_integer('var1', func.lit('YNU')) \
	      .alias('var1_recode')])
# create a list of the above transformations and apply to DF
transf_fun_DF = [

    rec_integer(x, func.lit('YNU')),alias(x)
    if x in YNU_cols
    else x 
    for x in DF_transf1
]
# apply to DF
DF_transf1 =  DF_transf1.select(transf_fun_DF)






--> "Getting to know the data - MLlib takes RDD"
import pyspark.mllib.stat as st
import numpy as np

# Numeric vars - basic stats
numeric_cols = ['var3','var6','var7']
numeric_rdd = DF_transf1.select(numeric_cols).rdd.map(lambda row: [e for e in row])
mllib_stats = st.Statistics.colStats(numeric_rdd)

for col, m, v in zip(numeric_cols, mllib_stats.mean(), mllib_stats.variance()): # count(), max(), mean(), min(), normL1(), normL2(), numNonzeros(), variance()
	print('{0}: \t{1:.sf} \t {2:.2f}'.format(col, m, np.sqrt(v)))
" var3    28.30   6.08  "
" var6    ............  "
" var7    ............  "

# categorical vars - level frequency
categorical_cols = [e for e in DF_transf.columns if e not in numeric_cols] # if not in numeric vars
categorical_rdd = DF_transf.select(categorical_cols).rdd.map(lambda row: [e for e in row])

for i, col in enumerate(categorical_cols):
	agg = categorical_rdd.groupBy(lambda row: row[i]).map(lambda row: (row[0], len(row[1])))
	print(col, sorted(agg.collect()), key=lambda el: el[1], reverse=True)
" var4    [(1, 23434), (0, 222424)]           "
" var5    [(1, 5345), (4, 345345), (3, 75645)]"
" var2    ..................................  "

# Colinearity - numeric
corrs = st.Statistics.corr(numeric_rdd)
for i, el in enumerate(corrs > 0.5):
	correlated = [(numeric_cols[j], corrs[i][j]) for j, e in enumerate(el) if e == 1.0 and j != i]
	if len(correlated) > 0:
		for e in correlated:
			print('{0}-to-{1}: {2:.2f}'.format(numeric_cols[i], e[0], e[1]))
" Coefficient matrix of col pairs that > 0.5" # Only select feature has low correlations
# Filter data with outcome
Features_to_keep = ['var4','var1','var5','var2']
DF_transf1 = DF_transf1.select([e for e in Features_to_keep]) # Filter original DF


# Chi-square test - categorical
import pyspark.mllib.linalg as ln
for cat in categorical_cols[1:]:
	agg = DF_transf1.groupBy('var5').pivot(cat).count()
	agg_rdd = agg.rdd.map(lambda row: (row[1:])).flatMap(lambda row: [0 if e == None else e for e in row]).collect()
	row_length = len(agg.collect()[0]) - 1
	agg = ln.Matrices.dense(row_length, 2, agg_rdd) # put into matrix - num of rows, num of cols, data

	test = st.Statistics.chiSqTest(agg)
	print(cat, round(test.pValue,4))
" var4    0.0  "
" var5    0.0  "
" var2    .... " # 0.0 p-value shows significant diff which is good to prediction








--> "Creating Final DataSet"
" LabeledPoint -- is a Mllib structure that is used to train the machine learning models (LabeledPoint: [label]-[feature]) "
" [label] -> target variable "
" [feature] -> array, list, pyspark.mllib.linalg.SparseVector, pyspark.mllib.linalg.DenseVector, scipy.sparse"
# Create an RDD of LabeledPoints
import pyspark.mllib.linalg as ln
import pyspark.mllib.feature as ft
import pyspark.mllib.regression as reg
hashing = ft.HashingTF(7) # Create hashing model - That categorical variable - var4 has 7 levels
DF_transf1_hashed = DF_transf1 \
      .rdd \
      .map(lambda row: [list(hashing.transform(row[1]).toArray()) if col == 'var4' else row[i] for i, col in enumerate(Features_to_keep)]) \
      .map(lambda row: [[e] if type(e) == int else e for e in row]) \
      .map(lambda row: [item for sublist in row for item in sublist]) \
      .map(lambda row: reg.LabeledPoint(row[0], ln.Vectors.dense(row[1:])))


--> "Spliting into training / Testing sets"
DF_train, DF_test = DF_transf1_hashed.randomSplit([0.6, 0.4]) # 60/40 split



--> "Predicting "
# Logistic Regression
" MLlib provides Logistic regression with LBFGS (Limited-memory-Broyden-Fletcher-Goldfarb-Shanno)"
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
# Training
LR_Model = LogisticRegressionWithLBFGS.train(DF_train, iterations=10) # Train model, iteration specified - not takes too long to run
# Prediction
LR_result = (
	DF_test.map(lambda row: row.label) \
	.zip(LR_Model \
		.predict(DF_test \
			.map(lambda row: row.features)))
	).map(lambda row: (row[0], row[1] * 1.0)) # Create a vector ([real_label] [predicted_label])
# Evaluation model
import pyspark.mllib.evaluation as ev
LR_evaluation = ev.BinaryClassificationMetrics(LR_result)
print('Area under PR: {0:.2f}'.format(LR_evaluation.areaUnderPR))
print('Area under ROC: {0:.2f}'.format(LR_evaluation.areaUnderROC))
LR_evaluation.unpersist() # Dont have to cache
" Area under PR: 0.85 "
" Area under ROC: 0.63 "




# Slecting predictable features using Chi-sq
import pyspark.mllib.feature as ft
selector = ft.ChiSqSelector(4).fit(DF_train) # select top 4 features

topFeatures_train = (
	DF_train.map(lambda row: row.label) \
	.zip(selector \
		.transform(DF_train \
			.map(lambda row: row.features)))
	).map(lambda row: reg.LabeledPoint(row[0], row[1]))

topFeatures_test = (
	DF_test.map(lambda row: row.label) \
	.zip(selector \
		.transform(DF_test \
			.map(lambda row: row.features)))
	).map(lambda row: reg.LabeledPoint(row[0], row[1]))






# Random Forest
from pyspark.mllib.tree import RandomForest
RF_model = RandomForest.trainClassifier(data=topFeatures_train,
	                                    numClasses=2,
	                                    categoricalFeaturesInfo={},
	                                    numTrees=6,
	                                    featureSubsetStrategy='all',
	                                    seed=666)
# Evaluation model
RF_result = (
	topFeatures_test.map(lambda row: row.label) \
	.zip(RF_model \
		.predict(topFeatures_test \
			.map(lambda row: row.features)))
	)

RF_evaluation = ev.BinaryClassificationMetrics(RF_result)
print('Area under PR: {0:.2f}'.format(LR_evaluation.areaUnderPR))
print('Area under ROC: {0:.2f}'.format(LR_evaluation.areaUnderROC))
LR_evaluation.unpersist() # Dont have to cache
" Area under PR: 0.86 "
" Area under ROC: 0.63 "













"
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ML
" ML operates on DataFrame "

["Transformer"] ["Estimator"] ["Pipeline"]

--> "Transformer"
" Transforms your data by appending a new column to your DataFrame"
" some method has E letter next to it, it is still in experment, testing may fail"

import spark.ml.feature as ft
" ft.Transformer(inputCol=xxx, outputCol=xxxx).transform(DF) "

-1- Binarizer

-2- Bucketizer

-3- ChiSqSelector

-4- CountVectorizer

-5- DCT

-6- ElementwiseProduct

-7- HashingTF

-8- IDF

-9- IndexToString

-10- MaxAbsScaler

-11- MinMaxScaler

-12- NGram

-13- Normalizer 

-14- OneHotEncoder

-15- PCA

-16- PolynomialExpansion

-17- QuantileDiscretizer

-18- RegexTokenizer

-19- RFormula

-20- SQLTransformer

-21- StandardScaler

-22- StopWordsRemover

-23- StringIndexer

-24- Tokenizer

-25- VectorAssembler

-26- VectorIndexer

-27- VectorSlicer

-28- word2Vec







--> "Estimator"
" Models to select "
import pyspark.ml.classification 
import pyspark.ml.regression 
import pyspark.ml.clustering 

-1- classification

>>> LogisticRegression

>>> DecisionTreeClassifier

>>> GBTClassifier

>>> RandomForestClassifier

>>> NaiveBayes

>>> MultilayerPerceptronClassifier

>>> OneVsRest





-2- Regression

>>> AFTSurvivalRegression

>>> DecisionTreeRegressor

>>> GBTRegressor

>>> GeneralizedLinearRegression

>>> IsotonicRegression

>>> IsotonicRegression

>>> LinearRegression

>>> RandomForestRegressor






-3- Clustering

>>> BisectingKMeans

>>> KMeans

>>> GaussianMixture

>>> LDA






-4- Parameter Tuning 

>>>

>>>

>>>








--> "Pipeline"
" End to end transformation - estimation process with distinct stages that ingests some raw data "
" Can be purly transformer "
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[

	transformer1,
	transformer2,
	transformer3,
	Estimator

	])




================================ "Full Pipeline Example" =======================================
# -------------------- Load the data
import pyspark.sql.types as typ
labels = [

    ('var1', typ.IntegerType()),
    ('var2', typ.IntegerType()),
    ('var3', typ.StringType()),
    ('var4', typ.IntegerType()),
    ('var5', typ.IntegerType()),
    ('var6', typ.StringType()),
    ('var7', typ.IntegerType()),
    ('var8', typ.StringType()),
    ('var9', typ.IntegerType()),

]

schema = typ.StructType([

	typ.StructField(e[0], e[1], False) for e in labels

	])

DF = spark.read.csv('xxxx.csv.gz',
	                header=True,
	                schema=schema)







# ---------------------- Create Transformers
DF = DF.withColumn('var3_Int', DF['var3'].cast(typ.IntegerType())) # cast a string column to integer type

import pyspark.ml.feature as ft

encoder = ft.OneHotEncoder( # Use encoder to transform 'var3'
	inputCol='var3_Int',
	outputCol='var3_Vec'
	)

featureCreator = ft.VectorAssembler( # Connect to previous transformer and transform again
	inputCols=[col[0] for col in labels[2:]] + [encoder.getOutputCol()], # Multiple input cols in a list
	outputCol='features'
	)






# ---------------------- Create Estimator
import pyspark.ml.classification as cl

logistic = cl.LogisticRegression(
	maxIter=10,
	regParam=0.01,
	labelCol='var9')




# --------------------- Create Pipeline
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[

	encoder,
	featureCreator,
	logistic

	])


# --------------------- Fitting a model
DF_train, DF_test = DF.randomSplit([0.7, 0.3], seed=666)
"OR"
DF_train, DF_valid, DF_test = DF.randomSplit([0.7, 0.2, 0.1], seed=666)

model = pipeline.fit(DF_train) # call action 
test_model = model.transform(DF_test)
test_model(1)
" [Row(var1=x, var2=x, ...., rawPrediction=DenseVector([xxx,xxx]), Probability=DenseVector([0.7, 0.3]), prediction=0.0)] "



# ---------------------- Evaluation on Model
import pyspark.ml.evaluation as ev

evaluator = ev.BinaryClassificationEvaluator(
	rawPredictionCol='probability', # 'rawPrediction' | 'probability'
	labelCol='var9')

print(evaluator.evaluate(test_model, {evaluator.metricName: 'areaUnderROC'}))
print(evaluator.evaluate(test_model, {evaluator.metricName: 'areaUnderPR'}))
" 0.74 "
" 0.71 "




# ------------------------ Saving the Model for latter use
# Save Pipeline
pipelinePath = './DF_onehotEncoder_Logistic_Pipeline'
pipeline.write().overwrite().save(pipelinePath)
" If load ... "
loadedPipeline = Pipeline.load(pipelinePath)
loadedPipeline.fit(DF_train).transform(DF_test).take(1)
" [Row(var1=x, var2=x, ...., rawPrediction=DenseVector([xxx,xxx]), Probability=DenseVector([0.7, 0.3]), prediction=0.0)] "


# Save Pipeline-Model
from pyspark.ml import PipelineModel
modelPath = './DF_onehotEncoder_Logistic_PipelineModel'
model.write().overwrite().save(modelPath)
" if load ... "
loadedPipelineModel = PipelineModel.load(modelPath)
test_reloadedModel = loadedPipelineModel.transform(DF_test) # Only need DF_test since we already saved the model with train result
test_reloadedModel.take(1)
" [Row(var1=x, var2=x, ...., rawPrediction=DenseVector([xxx,xxx]), Probability=DenseVector([0.7, 0.3]), prediction=0.0)] "







===================================== "Parameter Hyper - Tuning / Grid search" =========================================
import pyspark.ml.tuning as tune
import pyspark.ml.classification as cl
import pyspark.ml.evaluation as ev

# estimator
logistic = cl.LogisticRegression(
	labelCol = 'var9')

# grid
grid = tune.ParamGridBuilder() \
    .addGrid(logistic.maxIter, [2, 10, 50]) \
    .addGrid(logistic.regParam, [0.01, 0.05, 0.3]) \
    .build()

# evaluator create
evaluator = ev.BinaryClassificationEvaluator(
	rawPredictionCol='probability',
	labelCol='var9')

# Cross-validation
cv = tune.CrossValidator(

	estimator=logistic,
	estimatorParamMaps=grid,
	evaluator=evaluator

	)

# Transform data with pipe
pipeline = Pipeline(stages=[encoder, featureCreator])
data_transformer = Pipeline.fit(DF_train)


# Launch model tune
cvModel = cv.fit(data_transformer.transform(DF_train))
" It will return the best model estimated ... "

# Check result with Test set
data_train = data_transformer.transform(DF_test)
results = cvModel.transform(data_train)


print(evaluator.evaluate(results, {evaluator.metricName: 'areaUnderROC'}))
print(evaluator.evaluate(results, {evaluator.metricName: 'areaUnderPR'}))

" test result "
" 0.74 "
" 0.71 "






===================================== "Train - Validation Spliting" =========================================
# Add a selector to pipeline
selector = ft.ChiSqSelector(
	numTopFeatures=5,
	featuresCol=featureCreator.getOutputCol(),
	outputCol='selectedFeatures',
	labelCol='var9'
	)

logistic = cl.LogisticRegression(
	labelCol='var9',
	featuresCol='selectedFeatures')

# Transform data with pipe
pipeline = Pipeline(stages=[encoder, featuresCreator, selector])
data_transformer = pipeline.fit(DF_train)

# Train-validation
tvs = tune.TrainValidationSplit(

	estimator=logistic,
	estimatorParamMaps=grid,
	evaluator=evaluator

	)

# Launch model tune
tvsModel = tvs.fit(
	data_transformer \
	    .transform(DF_train)
	    )

# Check result with Test set
data_train = data_transformer.transform(DF_test)
results = tvsModel.transform(DF_train)

print(evaluator.evaluate(results, {evaluator.metricName: 'areaUnderROC'}))
print(evaluator.evaluate(results, {evaluator.metricName: 'areaUnderPR'}))

" test result "
" 0.73 "
" 0.70 "





===================================== "Feature Engineering in ML" =========================================

--> "NLP related feature extractors"
text_data = spark.createDataFrame([

	['''asdfasdfasdfasdfasdfasdfasdfasd
	    sdfasdfasdfasdfa asdfasf sdfasd
	    asdf asdfwgerg qwesdf sdfqer.'''],

	    ......

    ['''asdfasdfasdfasdfasdfasdfasdfasd
	    sdfasdfasdfasdfa asdfasf sdfasd
	    asdf asdfwgerg qwesdf sdfqer.''']

	], ['input'])

tokenizer = ft.RegexTokenizer(
	inputCol='input',
	outputCol='input_arr',
	pattern='\s+|[,.\"]' # tokenize space sep, remove commas, full stops, backslashes and quotation marks
	)

" [Row(input_arr=['xxx', 'asdasd', ....... , 'vwdca'])] "

stopwords = ft.StopWordsRemover(
	inputCol=tokenizer.getOutputCol(),
	outputCol='input_stop'
	)

# Build a N-gram model
ngram = ft.NGram(
	n=2,
	inputCol=stopwords.getOutputCol(),
	coutputCol="nGrams"
	)

# Pipeline
pipeline = Pipeline(stages=[tokenizer, stopwords, ngram])
data_ngram = pipeline \
     .fit(text_data) \
     .transform(text_data)

data_ngram.select('nGrams').take(1)
" we have got our Ngram vectors - 2 words "






--> "Discretizing continuous variables"
# create fake data
import numpy as np
x = np.arange(0, 100)
x = x / 100.0 * np.pi * 4
y = x * np.sin(x / 1.764) + 20.1234

schema = typ.StructType([
	typ.StructField('continuos_var',
		typ.DoubleType(),
		False
		)
	])

data = spark.createDataFrame([[float(e),] for e in y], schema=schema)


discretizer = ft.QuantileDiscretizer(
	numBuckets=5,
	inputCol='continuous_var',
	outputCol='discretized')

data_discretized = discretized.fit(data).transform(data)






--> "Standardizing continuous variables"
































"
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> GraphFrames
" GraphFrame leverages the distribution and expression capabilities of the DataFrame API to both simplify your queries and "
" leverages the performance optimization of the Apache Spark SQL engine. "

" [Vertices] --- edges --- [Vertices] "

--> " Installing GraphFrame "
# Spark CLI -- use GraphFrame 0.3, Spark 2.0, Scala 2.11
> $SPARK_HOME/bin/spark-shell --packages graphframes:graphframes:0.3.0-spark2.0-s_2.11



--> " Preparing DataSet "
# Load data
path1 = "/ssdfgsd/ndfgdfg/data1.csv"
path2 = "/shgdfg/w4tvaedf/rjuyj/data2.txt"

DF1 = spark.read.csv(path1, header='true', inferSchema='true', sep='\t')
DF1.createOrReplaceTempView("DF1_t")
DF2 = spark.read.csv(path2, header='true', inferSchema='true')
DF2.createOrReplaceTempView("DF2_t")
DF2_t.cache() # later join ETL, better performance

# Transform to create 'vertice dataset' and 'edge dataset'
DF1_transf = spark.sql("select ..... DF1_t....DF2_t...")
DF1_transf.cache()
DF2_transf = spark.sql("select .. AS src, .. AS dst..... DF1_t....DF2_t...")
DF2_transf.cache()

DF1_transf.show(10)
DF2_transf.show(10)




--> " Building the GraphFrame "

" We are going to build our Graph - "
" [Vertices] - a DF column named 'id' to have all the unique names of vertices "
" [edges] - a DF column named 'src' gives a vertice name (from), a column named 'dst' gives a vertice name (to), many other meta columns give information of the edge "

" Note, ensure you have already installed the GraphFrames spark-package "
from pyspark.sql.functions import *
from graphframes import *

# Create vertices and edges
DF_vertices = DF1_transf.withColumnRenamed("original_id", "id").distinct() 
DF_edges = DF2_transf.select("original_id", "src", "dst", "meta1", "meta2")
DF_vertices.cache()
DF_edges.cache()

display(DF_vertices) # check

DF_Graph = GraphFrame(DF_vertices, DF_edges)






--> " Excuting queries "
# How many vertices / edges
DF_Graph.vertices.count()
DF_Graph.edges.count()

# Max value of 'meta1' in group of vertices
DF_Graph.edges.groupBy().max("meta1")

# Filtering
DF_Graph.edges.filter("meta1 <= 0").count() # count rows that 'meta1' less or equal to 0
DF_Graph.edges.filter("meta1 > 0").count()
DF_Graph.edges.filter("src = 'SEA' and meta1 > 0").groupBy("src", "dst").avg("meta1").sort(desc("avg(meta1)")).show(5) # filter -> groupby takes AVG -> sort -> show top 5


# GroupBy + aggregate
import pyspark.sql.functions as func
TopTiers = DF_Graph.edges.groupBy("src", "dst").agg(func.count("meta1").alias("Tiers"))
TopTiers.orderBy(TopTiers.Tiers.desc()).limit(20)


# Graph - degree
" Number of edges around a vertice = degree "
DF_Graph.degrees.sort(desc("degree")).limit(20) # most 20 connected vertices
DF_Graph.inDegrees.sort(desc("inDegree")).limit(20) # coming edges
DF_Graph.outDegrees.sort(desc("outDegree")).limit(20) # going edges

inDeg = DF_Graph.inDegrees
outDeg = DF_Graph.outDegrees
degreeRatio = inDeg.join(outDeg, inDeg.id == outDeg.id) \
                    .drop(outDeg.id) \
                       .selectExpr("id", "double(inDegree)/double(outDegree) as degreeRatio") \
                          .cache()

DF1_transf_ratio = degreeRatio.join(DF1_transf, degreeRatio.id == DF1_transf.original_id) \
                                  .selectExpr("id", "city", "degreeRatio") \
                                      .filter("degreeRatio between 0.9 and 1.1")




# Understanding Motifs
" To easily understand complex relationship -> use motifs to find patterns "
"(b) - represent middle vertice == 'SFO' "
"(a) - represent original vertice "
"(c) - represent destination vertice "
"[ab] - represnt edges from a -> b "
"[bc] - represnt edges from b -> c " 
motifs = DF_Graph.find("(a)-[ab]->(b); (b)-[bc]->(c)") \
                     .filter("(b.id = 'SFO') and (ab.meta1 > 500 or bc.meta1 > 500) and bc.meta2 > ab.meta2 and bc.meta2 < ab.meta2 + 1000")



# Using 'PageRank' Algorithm 
ranks = DF_Graph.pageRank(resetProbability=0.15, maxIter=5) # resetProbability - prob of setting a random vertex, maxIter - a set number of iteration
ranks.vertices.pagerank.desc().limit(20)


# Find shortest path - Breadth-first search (BFS)
filter_path = DF_Graph.bfs(
	formExpr = "id = 'SEA'",
	toExpr = "id = 'SFO'",
	maxPathLength = 1)

filter_path.show(5)

" from                   e0                      to               "
" {'id':'SEA', ......}  {'..edges infore...'}    {'id':'SFO ....} "
" ............................................................... "























"
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TensorFrames
" TensorFlow - is a Google open source software library for numerical computation using data flow graphs which is an open source machine learning library focusing on Deep learning. "
" It also built on C++ with Python interface. "
" TensorFlow performs numerical computation using data flow graphs: "
"           - [Vertice] - mathmatical operation                     "
"           - [edge] - multidimensional arrays                      "

" Tensor -- multidimensional arrays  " " Flow -- mathmatical operation -> pass through " " ===> 'TensorFlow' "


--> " Install TensorFlow "
$ pip install tensorflow
$ pip install tensorflow-gpu



--> " Use TensorFlow offline with Python "
# Import TensorFlow
import tensorflow as tf

-1- "Using constant matrix "
# Setup the matrix
#   c1: 1X3 matrix
#   c2: 3X1 matrix
c1 = tf.constant([3., 2., 1.])
c2 = tf.constant([[-1.], [2.], [1.]])

" c1 "
" [3.  2.  1.]"

" c2 "
" [-1.] "
" [2. ] "
" [1. ] "

# m3: matrix multiplication (m1 X m3)
mp = tf.matmul(c1, c2)

# launch the default Graph
s = tf.Session()

# run: Execute the ops in graph
r = s.run(mp)
print(r)

" [[2.]] "

# Close the session
s.close()



-2- "Using placeholder "
# Setup placeholder for your model
#   t1: placeholder tensor
#   t2: placeholder tensor
t1 = tf.placeholder(tf.float32)
t2 = tf.placeholder(tf.float32)

# m3: matrix multiplication (m1 X m3)
tp = tf.matmul(t1, t2)

# Define input matrices
m1 = [[3., 2., 1.]]
m2 = [[-1.], [2.], [1.]]

# Execute the graph within a session
with tf.Session() as s:
	print(s.run([tp], feed_dict={t1:m1, t2:m2}))

" [array([[2.]], dtype=float32)] "







--> " TensorFrame - use TensorFlow in Spark with DataFrame "

" DataFrame --> | TensorFrame (Tensors) --> Operation --> TendorFrame (Tensor) | --> DataFrame "
" TensorFrame bridge between Spark DataFrame and TensorFlow "

# Install tensorframe
$SPARK_HOME/bin/pyspark --packages tjhunter:tensorframes:0.2.2-s_2.10

# Import TensorFlow, TensorFrames, and Row
import tensorflow as tf
import tensorframes as tfs
from pyspark.sql import Row

-1- " Add a constant to a column using Tensorflow "
# Create RDD of floats and convert into DataFrame 'df' 
rdd = [Row(x=float(x)) for x in range(10)]
df = sqlContext.createDataFrame(rdd)

df.show()

" | X | "
" |0.0| "
" |1.0| "
" |2.0| "
" |.. | "

# Run TensorFlow program executes:
#   The 'op' performs the addition (i.e. 'x' + '3')
#   Place the data back into a dataFrame 
with tf.Graph().as_default() as g:

#   The placeholder that corresponds to column 'x'
#   The shape of the placeholder is automatically
#   inferred from the DataFrame 
    x = tfs.block(df, "x")

    # The output that adds 3 to x
    z = tf.add(x, 3, name='z')

    # The resulting 'df2' dataFrame
    df2 = tfs.map_blocks(z, df)	

df2.show()

" | Z || X | "
" |3.0||0.0| "
" |4.0||1.0| "
" |5.0||2.0| "
" |.. ||.. | "



-2- " Blockwise reducing operations "
# Build a DataFrame of vectors
data = [Row(y=[float(y), float(-y)]) for y in range(10)]
df = sqlContext.createDataFrame(data)
df.show()

" |      y     | "
" | [ 0.0, 0.0]| "
" | [-1.0, 1.0]| "
" | [-2.0, 2.0]| "
" | [.......  ]| "

# We need to analyze the DataFrame to determine its shape.
tfs.print_schema(df)
" |-- y: array (nullable = true) double[?,?} " # ? means TensorFlow doesn't know its shape

df2 = tfs.analyze(df)
tfs.print_schema(df2)
" |-- y: array (nullable = true) double[?,2} " # knows has contains vectors of size 2


# Lets make a copy of the y column
df3 = df2.select(df2.y, df2.y.alias("z"))

# execute the Tensor Graph
with tf.Graph().as_default() as g:

	# The placeholders
	# Note the special name that end with '_input':
	  y_input = tfs.block(df3, 'y', tf_name="y_input")
	  z_input = tfs.block(df3, 'z', tf_name="z_input")

	# performa elementwise sum and minimum
	  y = tf.reduce_sum(y_input, [0], name='y') # First sum row, the sum column
	  z = tf.reduce_min(z_input, [0], name='z') # First min row, the min column

# The resulting DataFrame
(data_sum, data_min) = tfs.reduce_blocks([y, z], df3)

data_sum.show()
" [45., -45.] "

data_min.show()
" [0.0, -9.0] "



















"
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Structured Streaming Data [based on Spark Streaming]
" At its core, Spark Streaming is a scalable, fault-tolernt streaming system that takes RDD batch paradigm and speed it up. "

" Why do we need Streaming? "
" -- Streaming ETL "
" -- Triggers      "
" -- Data Enrichmwnt "
" -- Complex session and continue learning "

 
-1- "[Simple Streaming] Application using DStream "

# --------------- word_count.py ------------------ #
# Create a local SparkContext and Streaming Context
from pyspark import SparkContext
from pyspark.streaming import StreamingContext

# Create sc with two working threads
sc = SparkContext("local[2]", "NetworkWordCount")

# Create local StreamingContextwith batch interval of 1 second
ssc = StreamingContext(sc, 1)

# Create DStream that connect to localhost: 9999
lines = ssc.socketTextStream("localhost", 9999)


# Split lines into words
words = lines.flatMap(lambda line: line.split(" "))

# Count each word in each batch
pairs = words.map(lambda word: (word, 1))
wordCounts = pairs.reduceByKey(lambda x, y: x + y)

# Print the first tem elements of each RDD in this DStream
wordCounts.pprint()

# Start the computation
ssc.start()

# Wait for the computation to terminate
ssc.awaitTermination()
# ------------------------------------------------ #

# sending words to local port 9999
$> nc -lk 9999
["green green bule bule"]
["green blue" ]


" Time: 2017-01-14 13:30:32 "
" (u'blue', 2)"
" (u'green', 2)"

" Time: 2017-01-14 13:30:33 "
" (u'blue', 1)"
" (u'green', 1)"


** "The word counts doesn't aggregated and cumulated "

" Aggregated numbers over a specific time window -- [calculating a stateful aggregation] "

-1-2- "[Stateful Streaming] Application using DStream "
# --------------- word_count.py ------------------ #
# Create a local SparkContext and Streaming Contexts
from pyspark import SparkContext
from pyspark.streaming import StreamingContext

# Create sc with two working threads
sc = SparkContext("local[2]", "StatefulNetworkWordCount")

# Create local StreamingContext with batch interval of 1 sec
ssc = StreamingContext(sc, 1)

# Create checkpoint for local StreamingContext
ssc.checkpoint("checkpoint")

# Define updateFunc: sum of the (key, value) pairs
def updateFunc(new_values, last_sum):
	return sum(new_values) + (last_sum or 0)

# Create DStream that connects to localhost: 9999
lines = ssc.socketTextStream("localhost", 9999)

# Calculate running counts
running_counts = lines.flatMap(lambda line: line.split(" ")) \
                      .map(lambda word: (word, 1)) \
                      .updateStateByKey(updateFunc)

# Print the first ten elements of each RDD generated in this Stateful DStream to the console
running_counts.pprint()

# Start the computation
ssc.start()

# Wait for the computation to terminate
ssc.awaitTermination()
# ------------------------------------------------ #










-2- " [Structure Streaming] "
" Simplifying streaming by introducing the concept of 'Structrued Streaming' which bridges the concepts of streaming with DataSet and DataFrame "

" Example: reads data stream from S3 and saves it to a MySQL database: "



# ------------------------------------ one time write
logs = spark.read.json('s3://logs')

logs.groupBy(logs.UserId).agg(sum(logs.Duration)) \
          .write.jdbc('jdbc:mysql//...')



# ------------------------------------ Continuous aggregation
logs = spark.readStream.json('s3://logs').load()

sq = logs.groupBy(logs.UserId).agg(sum(logs.Duration)) \
          .writeStream.format('json').start() # Create sq var is to allow you to check the status of job and terminate it

# Will return true if the 'sq' stream is active
sq.isActive

# Will terminate the 'sq' stream
sq.stop()




# ------------------------------------- WordCount Example
# Import the necessary classes and create a local SparkSession
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split

# We donot need to establish a Streaming Context as this is already included within the SparkSessions
spark = SparkSession \
    .builder \
    .appName("StructuredNetworkWordCount")
    .getOrCreate()

# Create DataFrame representing the stream or input lines from connection to localhost 9999
lines = spark \
   .readStream\
   .format('socket') \
   .option('host', 'localhost') \
   .option('port', 9999) \
   .load()

# Split the lines into words
words = lines.select(
	explode( # Spark SQL func - explode
		split(lines.value, ' ')
		).alias('word')
	)

# Generate running word count
wordCounts = words.groupBy('word').count() # DataFrame funcs

# Start running the query that prints the running counts to the console
query = wordCounts \
    .writeStream \
    .outputMode('complete') \
    .format('console') \
    .start()

# Await Spark Streaming termination
query.awaitTermination()
# ----------------------------------------------------- #



$> nc -lk 9999
["green green bule bule"]
["green blue" ]




" Batch: 0 "
" (u'blue', 2)"
" (u'green', 2)"

" Batch: 1 "
" Time: 2017-01-14 13:30:33 "
" (u'blue', 3)"
" (u'green', 3)"

" ......... "












"
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Packaging Spark Application


-1- " spark-submit "
" The entry point for submitting jobs to Spark -- spark-submit "

$> spark-submit ["options"] <python file> ["app arguments"]


 - ["app argument"] # The arguments you pass to the Python app

 - ['options'] # The parameters pass to spark-submit

 --master: " local, spark://host:port:, mesos://host:post:, yarn "

 --deploy-mode: " client, cluster "

 --name:

 --py-files:

--files:

--conf:

--properties-files:

--driver-memory:

--executor-memory:

--help:

--version:

--supervise:

--kill:

--status:

# Yarn

--queue:

--num-executors:




-2- " Create SparkSession "
" When spark-submit, you need to prepare SparkSession yourself and configure it to your application "

from pyspark.sql import SparkSession

spark = SparkSession \
        .builder \
        .appName('CalculatingGeoDistance') \
        .getOrCreate()

print('Session created')
# Builder -- internal class
# appName -- give your app a name
# getOrCreate -- create or reused an already existed session





-3- " Modularizing Code "
" You can modularize your methods and then reused them at a later point "

** " Example - build a modular that calculates distance on our dataset "

# File Strictre
" AdditionalCode "
" |-- setup.py "
" |-- utilities "
"     |-- __init__.py "
"     |--base.py "
"     |--converters "
"     |  |-- __init__.py "
"     |  |-- distance.py "
"     |-- geoCalc.py "
"                    "
" 2 directories, 6 files "







** " setup.py "
# -------------------------------------------------------- #
# Used to package up our app
from setuptools import setup

setup(
	name='PySparkUtilities',
	version='0.1dev',
	packages=['utilities', 'utilities/converters'],
	license='''
	    Creative Commons
	    Attribution-Noncommerical-Share Alike license''',
	long_description='''
	    An example of how to package code for PySpark'''
	    )
# -------------------------------------------------------- #







** " __init__.py "
# -------------------------------------------------------- #
# It effectively exposes the 'geoCalc.py' and 'converters'
from .geoCalc import geoCalc
__all__ = ['geoCalc', 'converters']
# -------------------------------------------------------- #







** " base.py "
# -------------------------------------------------------- #
from abc import ABCMeta, abstractmethod

class BaseConverter(metaclass=ABCMeta):
	@staticmethod
	@abstractmethod
	def convert(f, t):
		raise NotImplementedError
# -------------------------------------------------------- #




** " distance.py " # convert between different measurement

** " geoCalc.py " # there is a class called "calculateDistance()" calculating the direct distance between any two points


>>> Building an Python egg 
# so that you can use app as pass to --py-files
# In AdditionalCode folder
python setup.py bdist_egg
" pySparkUtilities-0.1.dev0-py3.5.egg "




*** " calculatingGeoDistance.py "
# -------------------------------------------------------- #
import utilities.geoCalc as geo
from utilities.converters import metricImperial

getDistance = func.udf(
	lambda lat1, long1, lat2, long2:
	   geo.calculateDistance(
	   	(lat1, long1),
	   	(lat2, long2)
	   	)
	   )

convertMiles = func.udf(lambda m:
	metricImperial.convert(str(m) + ' mile', 'km'))

uber = uber.withColumn(
	'miles',
	    getDistance(
	    	func.col('pickup_latitude'),
	    	func.col('pickup_longitude'),
	    	func.col('dropoff_latitude'),
	    	func.col('dropoff_longitude')
	    	)
	    )

uber = uber.withColumn(
	'kilometers',
	convertMiles(func.col('miles')))
# -------------------------------------------------------- #





*** " lanuch_spark_submit.sh "
# -------------------------------------------------------- #
#!/bin/bash

unset PYSPARK_DRIVER_PYTHON
spark-submit $*
export PYSPARK_DRIVER_PYTHON=jupyter
# -------------------------------------------------------- #





>>> Submitting  a jobs
$ ./lanuch_spark_submit.sh \
  --master local[4] \
  --py-file AdditionalCode/dist/PySparkUtilities-0.1.dev0-py3.5.egg \
  calculatingGeoDistance.py



" After launch --> http://localhost:4040 --> UI "

































"
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Polyglot Persistence with Blaze
" Polyglot Persistence -- Store different data with different techniques for different purposes but use them as a whole with persisitence "
" Blaze -- create data abstract overlap all different techniques so that we can manipulate data from different sources in one API "

