# Spark ----------- Scala, Python, R

# download
http://spark.apache.org/downloads.html

# Install on local
tar -xf spark-1.3.0-bin-hadoop2.4.tgz
cd spark-1.3.0-bin-hadoop2.4

bin/pyspark # lauch PySpark shell
bin/spark-shell # lauch Scala shell

# Or launch in Ipython Notebook
IPYTHON=1 ./bin/pyspark
IPYTHON_OPTS="notebook" ./bin/pyspark # lauch Pyspark with notebook

# access Spark UI
http://[ipaddress]:4040 # see all sorts of information about your task and cluster



"""
Spark is writen in Scala and runs on JVM, to run Spark, all you need is an installation of Java 6 or newer.
If you wish to use Python API, you will also need an Python interpreter (version 2.6 or newer) / Not yet work with Python 3

Spark can runs on - local mode, Mesos, YARN, or other standalone scheduler

Spark, unlike other shell which only allows you interact with disk, memory on single machine, it allows you interact with data across 
multiple machine's disk and memory / It takes care of the automatic distribution process for you

Cause calculation in memory is faster, ability to load into memory for all machines makes Spark the fastest
"""

" Data Scientist = use the interactive shell to prototype model and play data to discover patterns and prediction, use SQL for exploration "

" Engineer = use Spark to parallel application across clusters, and hindes the complexity of distributed system programming, network communication, and fault tolerance."





# [0] ----------------------------------------------------- Spark Stack

"1"
#########################
       Spark Core       #
#########################
" Spark core conrtains basic functionalities of Spark - task scheduling, memory management, fault recovery, "
" interacting with storage systems and more. Also, it is the home to the API that defines RDD - Spark's main "
" programming abstraction. Spark core provides many APIs for building and manipulating these collections. "



"2"
#########################
        Spark SQL       #
#########################
" It is Spark's package for working with structure data. It allows query data by SQL as well as HQL, and it "
" supports many sources of data, including HIVE tables, Parquet, and JSON. Beyound providing SQL interface, "
" Spark SQL  allows intermix SQL with RDD manipulation in Python, R, Scala all within single application.   "



"3"
#########################
     Spark Streaming    #
#########################
" It is Spark's component that handles live stream data like log files by web server, or queues of message "
" containing status updates posted by users of web server. It provides an API to manipulate data streams that "
" closely matches the Spark Core's RDD API which makes it easy to learn, and move between applications that "
" manipulates data stored in memory, on disk, or arriving in real-time. "



"4"
#########################
      Spark MLlib       #
#########################
" It contains common machine learning functionality (algorithms + evaluation + workflows)"



"5"
#########################
      Spark GraphX      #
#########################
" It is a library for manipulating graphs (ex. social networks) and performing graph-parallel computations. "
" Like Spark SQL, Streaming, GraphX extends the Spark RDD API, allowing us to create a directed graph. "



"6"
#########################
  Spark Cluster Manager #
#########################
" Spark is designed to efficiently scale up from one to many thousands of compute nodes. To achieve this, Spark can run "
" over a variety of Cluster Managers - Hadoop YARN or Mesos clusters. If just installing Spark on a empty set of machines "
" (without existing cluster manager) the Standalone Scheduler (pre-built cluster manager in Spark) will help and gets started."






# [1] ------------------------------------------------------ Core Spark Concept

" Deal with failure - Spark deals with failed or slow machines by re-executing "
" failed or slow tasks. Send the tasks to other nodes. Even when no fail but just slow."
" code may be called multiple times. "


---------------- Run interactive Shell ------------------

# driver program - launch operation on a cluster
> # interactive shell (Driver program)
"""
Driver program - contains your application's main function and 
defines distributed datasets on a cluster, then applies operations to them
 """

# Driver program access Spark through a "SparkContext" object - which represents connection to a computation cluster 
lines = sc.textFile("README.md") # 
sc # SparkContext object - You can use it to build RDDs which can be applied with operations by Driver program
lines # RDDs
lines.count() # Driver program runs operations on RDDs 

"""
Driver program usually manage a number of nodes called executors, one operation task will be divided for different
nodes to execute / You can just write code in single driver and it will be automatically execute in paralelle
"""

 -------------- Run standalone application ------------- 

"""
Spark can be linked to a standalone application in Java, Python and Scala. The main difference between using standalone
and using interactive is that you need to initialize your own 'SparkContext'. Others are the same.
"""

# @1 link Spark to application
# In Java, Scala using Build Tool
groupId = org.apache.spark
artifactId = spark-core_2.10
version = 1.3.0 # latest spark version 
" Add the dependency to you application to create binary - run binary "

# In PySpark
bin/spark-submit script.py # run py script directly use "bin/spark-submit"


# @2 initializing a SparkContext
"""
Once you link Spark to your application, you need to import Spark packages into your program and create a SparkContext - 
you do so by creating a SparkConf object to configure your application, and then build a SparkContext for it 
"""
# Pyspark
from pyspark import SparkConf, SparkContext
conf = SparkConf().setMaster("local").setAppName("My App") # Create conf object - a cluster URL / your app name
sc = SparkContext(conf = conf) # build a SparkContext for it

# Scala
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

val conf = new SparkConf().setMaster("local").setAppName("My App") # Create conf object - a cluster URL / your app name
val sc = new SparkContext(conf) # build a SparkContext for it

# can use the sc to build RDDs
lines = sc.sc.textFile("README.md") 

# done
stop() # shutdown Spark
System.exit(0) # simply exit the application
sys.exit()




------------------------ Example of creating a application -----------------------

# ---------- Scala Script --------- #
# // Create a Scala Spark Context.
val conf = new SparkConf().setAppName("MyApp1") # Create conf object - a cluster URL / your app name
val sc = new SparkContext(conf) # build a SparkContext for it
# // Load our input data.
val input = sc.textFile(inputFile)
# // Split up into words.
val words = input.flatMap(line => line.split(" "))
# // transform into pairs and count.
val counts = words.map(word => (word,1)).reduceByKey{case (x,y) => x + y}
# // Save the word count back out to a text file, causing evaluation.
counts.saveAsTextFile(outputFile)
# --------------------------------- #


# ---------- BUILD File ------------ #
name = "learning-spark-mini-example"
... spark-core, 1.3.0
# ---------------------------------- #


# Scala build and run
$SPARK_HOME/bin/spark-submit --class com.oreilly.learningsparkexample.mini.scala.MyApp1 ./target/...(as above) ./README.md ./myapp1



























# [2] ------------------------------------------------------ Programming with RDDs
"""
Spark's core abstraction for working with data -- resilient distributed dataset (RDD) 
RDD - a distributed collection of elements.
In Spark - all works => creating a new RDD, 
                        transforming on exisiting RDDs, (Operations)
                        calling actions on RDDs to compute result (Operations)

Spark automatically paralelle the storage / operations on the dataset



RDD is simply an immutable distributed collection of objects => each RDD is splited into multiple 
partitions, which will be computed in a different nodes of the cluster. RDD can contains any type of 
Python, Scala, Java objects including user defined classes.
"""


# Two ways to create RDDs: 
# [1] Loading an external dataset
lines = sc.textFile("README.md") # into RDD (Distributed in cluster)

# [2] Distributing an internal collection of objects (a list, set) in a driver program
Lines = sc.parallelize(["pandas", "i like pandas"]) # into RDD (Distributed in cluster)




# Once created, two types of operations: (transformations) / (actions)
" Transformation: construct a new RDD from a previous one " " Default: (Lazy fashion) Will not actually compute until use 'Action' on RDD "
                                                            " Default: Only stream RDD (Not presist data) one time while call Action "
                                                            " Default: You can use '.persist' to reuse RDD in multiple Actions "
pythonlines = lines.filter(lambda line: "Python" in lines) # create a new filtered RDD from previous one

" Actions: compute a result based on a RDD, and either return it to the driver program or save it to an external storage system, ex HDFS "
pythonLines.first() # compute a result of the first 5 

" (Transformation - Action) workflow "
"""
1. Create some input RDDs from external data.
2. Transform them to define new RDDs using transformation like filter().
3. Ask Spark to '.presist()' any intermediate RDDs that will need to be reused.
4. Launch actions such as count() and first() to kick off a parallel computation, which is then optimized and executed by Spark.
"""
# Example:
lines = sc.textFile("README.md") # 1.
pythonlines = lines.filter(lambda line: "Python" in lines) # 2.
pythonlines.persist # 3.
pythonlines.count() # -4.
pythonlines.first() # -4.





# Creating RDD 
" Again, two ways - loading an external dataset / parallelizing a collection in your driver program "

# 1. parallelizing
lines = sc.parallelize(["pandas", "i like pandas"]) # Python

val lines = sc.parallelize(List("pandas", "i like pandas")) # Scala

" Useful when learning Spark. Not very common in real, since you have pre-save data in one node. "

# 2. Loading text file
lines = sc.textFile("/path/...../README.md") # Python
val lines = sc.textFile("/path/..../README.md") # Scala




# RDD Operations (Transformation | Action)

# - Transformation (Not compute)
" Operations returns new RDD from a old RDD with Lazy Evaluation "

New_RDD = RDD.map()
New_RDD = RDD.filter(lambda x: "error" in x)

Newest_RDD = New_RDD.union(RDD) # Transformation object can be reused "Gets the numer of lines in both New_RDD and RDD"

" The relationship: RDD -> New_RDD -> Newest_RDD which is called 'lineage graph' is tracked so that can recover lost data if "
" part of presistent RDD is lost "



# - Action (Start compute) * When you want to see
" Actually return a final value or write data to an external storage system "

New_RDD.count()
Newest_RDD.first()

for line in Newest_RDD.take(10): # get 10 elements to local
    print line

RDD.collect() # return whole set * sure it fits single machine
RDD.saveAsSequenceFile() # write the data to a distributed system: HDFS or S3

" Keep in mind, whenever call a action Data will computed from scratch, make sure 'presist' data "






# Lazy Evaluation
" It means Spark will not execute until sees a Action "
" User be more freely to chain/test their operations "

New_RDD = RDD.map() # Don't think it as aspecific data but a transform instruction (Not execute yet)

" You can force Spark to execute using '.count()' as a way to test "





# Pass Functions to Spark

" Most of Spark's transformations, and some of its actions, depends on passing in function that are used "
" by Spark to compute data. Each of the core langauages has a slightly different mechanism for passing functions to Spark."

# ---- Python

1. "lambda" # Shorter function
word = RDD.filter(lambda x: "error" in x)

2. "Top-level function"
def contain(s):
  return "error" in s

3. "Locally defined function"
class wordf7unction(object):
  def __init__(self, query):
    self.query = query
  def isMatch(self,s):
    return self.query in s
  def getMatchesNoReference(self,RDD):
    # Safe: extract only the field we need into a local variable
    query = self.query # extract into a local variable (within function) -> will only send what you need
    return RDD.filter(lambda x: query in x)
  def getMatchesFunctionReference(self,RDD):
    # Problem: references all of "self" in "self.isMatch"
    return RDD.filter(self.isMatch) # directly use 'self.isMatch' which belongs to 'wordf7unction', Spark sends whole object to this function

# ---- Scala 

" To be comtinous ... "











# ----------------------------------------------------------------------------------------------------------------------------------------- Common Transformation and Actions

" Additional operations are available on RDDs containing certain types of data - for example, statistical functions on RDDs of numbers, and the "
" key/value operations such as aggregationg data by key on RDDs of key/value pairs."

1. " Basic RDD - perform on all regradless of the data "

1-1. "element-wise transformation"
.map() # Takes function and apply it to each element in RDD, one input = one output (1,2,3) => ([1-1,1-2],[2-1,2-2],[3-1,3-2])
nums = sc.parallelize([1,2,3,4])
squared = nums.map(lambda x: x * x).collect()
for num in squared:
  print "%i " % (num)

.flatMap() # Takes function and apply it to each element in RDD, one input = multiple output (1,2,3) => (1-1,1-2,2-1,2-2,3-1,3-2) *flat
lines = sc.parallelize(["helloe world", "hi"])
words = lines.flayMap(lambda line: line.split(" "))
words.first() # return "hello"


.filter() # Filter RDD and keep those meets the requirement
New_RDD = RDD.filter(lambda x: "error" in x)


1-2. "pseudo set operations" # require operated RDD euqals to the same type
.distinct() # remove duplicated elements to obtain unique elements
New_RDD = RDD.distinct() # {1,1,2,2,3,3,4} => {1,2,3,4} *Expensive computation

.union() # Combine by rows and sort by rows of two RDDs
New_RDD = RDD1.union(RDD2) 

.intersection() # Returns elements only in both RDD1 and RDD2 / Also remove all duplicated elements *Expensive computation
New_RDD = RDD1.intersection(RDD2)

.subtract() # returns elements only in RDD1 not in RDD2 *Expensive computation
New_RDD = RDD1.subtract(RDD2)

.cartesian() # Returns all possible pairs of combination between elements in RDD1 and RDD2 *Very Expensive computation
New_RDD = RDD1.cartesian(RDD2) 

.sample() # Sample an RDD, with or without replacement
New_RDD = RDD1.sample(false,0.5,123) # sample(withreplacement, fraction, [seed])


1-3. "Action"
.reduce(func) # Takes two elements in RDD and then aggregates | require return type of RDD = same type
SUM = RDD1.reduce(lambda x,y: x + y) # RDD{1,2,3,3} => {9}

.fold(zero)(func) # same as reduce() but with the provided zero value
SUM = RDD1.fold(0)((lambda x,y: x + y)) # RDD{1,2,3,3} => {9}

.aggregate(zeroValue)(seq0p, comb0p) "


.collect() # Which return the entire RDD's content {1,2,3,3}
RDD.collect()

.count()
RDD.count() # count the number of elements 4

.countByValue() # count frequency by each unique elements {(1,1),(2,1),(3,2)}
RDD.countByValue()

.take(num) # return num elements from RDD {1,2,3,3} => {1,2} *Not always the order you expected
RDD.take(2)

.top(num) # return num of top elements | will follow the order {1,2,3,3} => {3,3}
RDD.top(2)

.takeOrdered(num)(ordering) # Take the first num elements in a ordered RDD 
RDD.takeOrdered(5, key = lambda x: x[0]) # sort by key asc
RDD.takeOrdered(5, key = lambda x: -x[0]) # sort by key desc
RDD.takeOrdered(5, key = lambda x: x[1]) # sort by values asc
RDD.takeOrdered(5, key = lambda x: -x[1]) # sort by values desc

.takeSample(withreplacement, num, [seed]) # sample the RDD by 1 element
RDD.takeSample(false,1)

.foreach(func) # Apply function to each element of the RDD | Unit side effect not return new RDD
RDD.foreach(func) # {1,2,3,3} => nothing



1-4. "Persistence (Caching)"
" As 'lazy evaluation' you might want to presistend the RDD which you want to use multiple times "
" If we presist a RDD, the node computed it will store the partitions. If certain parttition is lost "
" Spark will recompute that partition - or replicate data on multiple nodes so that failures not slow us down "

-- Multiple persistent storage level --

"       Level           Space_used         CPU_time          In_memory       On_disk         Comments  "

" MEMORY_ONLY           HIGH               Low               Y               N                         "
" MEMORY_ONLY_SER       Low                High              Y               N                         "
" MEMORY_AND_DISK       HIGH               Medium            some            some            Spills to disk if there is too much data to fit in memory"
" MEMORY_AND_DISK_SER   Low                High              some            some            Spills to disk if there is too much data to fit in memory. Stores serialized representation in memory"
" DISK_ONLY             Low                High              N               Y                         "



# Ex. In Scala
import org.apache.spark.storage.StorageLevel
val result = input.map(x => x * x)
result.persist(StorageLevel.DISK_ONLY) # persist itself not evaluate immediately, need to call action
result.unpersist() # remove it from cachy














# ----------------------------------------------------------------------------------------------------------------------------------------- Working with Key/Value Pairs
" RDD of Key/Value pairs - commonly used to perform aggregations, and often we will do some initial ETL to get "
" our data into Key/Value pairs format. For operations - counting up reviews for each product; grouping together data with the same keys"
" Or grouping together two different RDDs"

" Paired RDD - " # Useful building blocks
" {('a',4),('r',5),('t',10)}" # tupe of two elements in a dictionary

-- Creating Pair RDDs --
# Turn a regular RDD to pair RDD
RDD = SparkContent.parallelize(inmemory.content)
pairs = RDD.map(lambda x: (x.split(" ")[0], x)) # first letter as key, full words as value


-- Transformations on Pair RDDs --
# on one RDD - 
RDD = {(1,2),(3,4),(3,6)}
.filter() # filter Pair RDD by keyValue[0]:key, keyValue[1]:value
RDD.filter(lambda keyValue: keyValue[1] >2) # {(3,4),(3,6)}

.foldBykey() # quite similiar with reduceBykey() but use (zero) value of the same type of data in our RDD and combination function.
RDD.foldBykey(0)((lambda x,y: x + y)) # {(1,2),(3,10)}

.reduceBykey() # Combine values with the same keys
RDD.reduceBykey(lambda x,y: x + y) # {(1,2),(3,10)}

.groupBykey() # Group values with the same keys
RDD.groupBykey() # {(1,[2]),(3,[4,6])}

.combineBykey(createCombiner,mergeValue,mergeCombiners,partitioner) #
RDD.combineBykey() "
RDD.combineBykey((lambda x: (x,1)),
               (lambda x, y: (x[0] + y, x[1] + 1)),
               (lambda x, y: (x[0] + y[0], x[1] + y[1])))

.mapValues(func) # Apply a function to each value of a pair RDD without changing the key
RDD.mapValues(lambda x: x + 1) # {(1,3),(3,5),(3,7)}

.flatMapValues(func) # Apply a function that returns an iterator to each value of a pair RDD, and for each element returned, produce a key/value entry with old key. Often used for tokenization
RDD.flatMapValues(lambda x: x to 5) " # {(1,3),(1,4),(1,5),(3,4),(3,5)} 

.keys() # return an RDD of just the keys
RDD.keys() # {1,3,3}

.values() # return just the values
RDD.values() # {2,4,6}

.sortBykey() # return s RDD sorted by keys
RDD.sortBykey() # {(1,2),(3,4),(3,6)}
RDD.sortBykey(ascending=True, numPartitions=None, keyfunc=lambda x: str(x)) # customized - sort it by converting integers as strings




# On two RDDS
RDD1 = {(1,2),(3,4),(3,6)}; RDD2 = {(3,9)}
.subtractBykey() # remove elements with a key present in the other RDD
RDD1.subtractBykey(RDD2) # {(1,2)} *except

.join() # Perform an inner join between two RDDs
RDD1.join(RDD2) # {(3,(4,9)),(3,(6,9))} *inner join

.rightOuterJoin() # Perform a join between two RDDs where the key must be Present in the RDD2
RDD1.rightOuterJoin(RDD2) " # {(3,(some(4),9), (3,(some(6),9))} 

.leftOuterJoin() # Perform a join between two RDDs where the key must be Present in the RDD1
RDD1.leftOutJoin(RDD2) " # {(1,(2,None)), (3,(4,some(9))), (3,(6,some(9)))}

.cogroup() # group data from both RDDs sharing the same key
RDD1.cogroup(RDD2) # {(1,([2],{})), (3,([4,6],[9]))}




-- Actions on Pair RDDs --
RDD = {(1,2),(3,4),(3,6)}
.countBykey() # count the number of elements under each key
RDD.countBykey() # {(1,1),(3,2)}

.collectAsMap() # collect the RDD result as a map to provide easier lookup
RDD.collectAsMap() 

.lookup(key) # return all values associated with provided key
RDD.lookup(3) # [4,6]





-- Data Partitioning (Advanced) --
# Tunning level of parallelism in an operation
data = [("a",3),("b",4),("a",1)]
sc.parallelize(data).reduceBykey(lambda x, y: x + y) # default parallelism
sc.parallelize(data).reduceBykey(lambda x, y: x + y, 10) # custom parallelism - specify number of partitions
" Most operation function takes a second value which specify the number of custom partitions "

# Tunning level of parallelism in RDD itself
.repartition() # cutomize partition in RDD (Very experiensive calculation)
.coalesce() # optimized version only used when decrease current partition numbers

# Advanced partitioning
" How to control dataset partitioning across nodes to minize network traffic and improve performance. "
" If an RDD is scaned only once, there is no need to partitioning it in advanced. If a dataset is used multiple times "
" in a key-oriented operations such as 'join'."

" It ensure a set of keys will appear together on some node - faster process."

.partitionBy() # It is a transformation (Return a new RDD)
userData = sc.sequenceFile[UserID, UserInfo]("hdfs://...")
             .partitionBy(HashPartitioner(100)) # Create 100 partitions - as many as your cores
             .persist()
" userData has been used multiple times later. So better..."
" More details regarding partitioner() in Scala and Java, P64-66"




















# ----------------------------------------------------------------------------------------------------------------------------------------- Loading and Saving your Data
" File " | " File System " | " Structured Data Source " | " Database & key/value store "


1. " File "
# Text Files (Unstructure) - 
--load--
input = sc.textFile("file:///home/holden/repo/.../test.md") # each line will become an element -> one RDD (Also support *.txt)
input = sc.wholeTextFiles("file:///home/holden/repo") # a pair RDD with keys as each file name, value as content
--save--
result.saveAsTextFile(outputFile)



# JSON (Semi-structure) - 
" Load Json as txt, then map it with Json parser "
--load--
import json
input = sc.textFile("file:///home/holden/repo/.../test.js")
data = input.map(lambda x: json.load(x)) # works if one JSON record per row;
--save--
result.map(lambda x: json.dumps(x)).saveTextFile(outputFile)



# CSV (Structure) - 
" Load CSV as txt, then map it with CSV parser "
--load-- 
# load by line
import csv
import StringIO

def loadRecord(line):
  """Parse a CSV line"""
  input = StringIO.StringIO(line)
  reader = csv.DictReader(input, fieldnames=["C1", "C2"])
  return reader.next()
input = sc.textFile("file:///home/holden/repo/.../test.csv").map(loadRecord)

# load as whole
import csv
import StringIO

def loadRecords(fileNameContents):
  """Load all records in a given file """
  input = StringIO.StringIO(fileNameContents[1]) 
  reader = csv.DictReader(input, fieldnames=["C1", "C2"])
  return reader
fullFileData = sc.wholeTextFiles("file:///home/holden/repo/.../test.csv").map(loadRecords)

--save--
def writeRecords(records):
  """ Write out CSV lines """
  output = StringIO.StringIO()
  writer = csv.DictWriter(output, fieldnames=["C1", "C2"])
  for record in records:
    writer.writerow(record)
  return [output.getvalue()]
pandaLovers.mapPartitions(writeRecords).saveAsTextFile(outputFile)



# Sequence File (Structure) - 
" a popular Hadoop format composed of flat files with key/value pairs. It is a common input/output "
" format for MapReduce as well. -- Better use Scala/Java ..."
--load--
# Python
data = sc.sequenceFile(inFile, "org.apache.hadoop.io.Text", "org.apache.hadoop.io.IntWritable")
# Scala
val data = sc.sequenceFile(inFile, classOf[text], classOf[Intwritable]).map{case (x,y) => (x.toString, y.get())}
--save--
# Scala
val data = sc.parallelize(List(("panda", 3), ("kay", 6)))
data.saveAsSequenceFile(outputFile)



# Object Files (Structure)- 
" Simple wrapper around SequenceFiles that allows us to save our RDDs containing just values."
--load--

--save--


# Hadoop input/output format - 
" In addition to the format Spark has wrapper for, we can also interact with any Hadoop-supported formats."
--load--
# Scala
val input = sc.newAPIHadoopFile(inputFile, classOf[lzoJasonInputformat],
  classOf[longWritable], classOf[MapWritable], conf) # each MapWritable in "input" represents a JSON object

--save--
# Java
result.saveAsHadoopFile(filename, text.class, IntWritable.class, SequenceFileOutputFormat.class);



# Compressed File - 
" textFile(), SequenceFile() handle some of compression, better use -  Hadoop input/output format "
" gzip, lzo, bzip2, zlib, Snappy "
--load--

--save--



2. " File System "
# Local / Regular File System - 
" Spark supports loading files from the local filesystem, it requires that the files are avaiable at "
" the same path on all nodes in your cluster. If your data already in file system like NFS, AFS, and "
" MapR's NFS layer and it is mounted at the same path on each node."

input = sc.textFile("file:///home/holden/repo/.../test.md") # directly load from path

"Or you can load data to a driver and then parallelize it across worker nodes - But can be slow "
" So better use - HDFS, NFS, S3 file system instead "



# Amazon S3 - 
" S3 is even fater if your compute nodes are in EC2 but can be bad if you go over public internet "
AWS_ACCESS_KEY_ID # environment variable
AWS_SECRET_ACCESS_KEY # environment varibale

input = sc.textFile("s3n://bucket1/holden/repo/.../test.md")



# HDFS - 
" Spark and HDFS can be collocated on the same machines, take advantage of this data locallity "
" to avoid overhead. "

input = sc.textFile("hdfs://master:port/path/.../test.md")



3. " Structure Data with Spark SQL "
" Spark's perferred way to work with strature or semi-structure data - have  "
" Use SQL query to select data fields from these data sources "

# Apache HIVE - 
cp hive-site.xml ./conf/ # copy xxx on Spark core xxx

from pyspark.sql import HiveContext

hiveCtx = HiveContext(sc)
rows = hiveCtx.sql("SELECT name, age FROM users")
firstRow = rows.first()
print fisrtRow.name


# JSON - 
" If you have JSON data with consistent schema. Spark SQL can infer their schema and load this data as rows as well "
" To load JSON data, first create a HiveContext as when using HIVE. Use the HiveContext.jsonFile method to get an RDD of row "
" objects for the whole file. In stead of using whole Row object, you can also register this RDD as a table and select specific fields from it."

tweets = hiveCtx.jsonFile("tweets.json")
tweets.registerTemTable("tweets")
results = hiveCtx.sql("SELECT user.name, text FROM tweets") 




4. " Database & key/value store " 
" Spark can access several popular databases using either Hadoop connectiors or custom Spark connectors. "
# Java Database Connectivity - 
" Supports load data from any relational database that supports Java Database (JDBC) - MySQL, Postgres, etc. "
" To access, we construct an 'org.apache.spark.rdd.jdbcRDD' and provide it with our SparkContext and other parameters"
# ---- Scala
def createConnection() = {
  Class.forName("com.mysql.jdbc.Driver").newInstance();
  DriverManager.getConnection("jdbc:mysql://localhost/test?user=holden");
}
""" 
Provide a function to establish a connection on our database which allow each node has their own connection to data.
"""
def extractValues(r: ResultSet) = {
  (r.getInt(1), r.getString(2))
}
"""
Create function for data extraction - convert output.
"""
val data = new jdbcRDD(sc,
  createConnection, "SELECT * FROM panda WHERE ? <= id AND id <= ?", 
  lowerBound = 1, upperBound = 3, numPartitions = 2, mapRow = extractValues)
println(data.collect().toList)
"""
Provide a query and bounds.
"""


# Cassandra - 
" Spark Cassandra connector "
" P- 94 "

# HBase - 
" P- 96"

# Elasticsearch - 
" p= 96"





















# ----------------------------------------------------------------------------------------------------------------------------------------- Advanced Spark Programming 
" accumulators (shared variable) " | " broadcast variable (shared variable) " | " pre-partition operations " | " piping to external program: R " | " numeric RDD operations "

" shared variable - Good for batch operations for tasks with high set-up costs, like querying a database. "

1.accumulators # aggregate information --------------- Used for debugging / Test purpose
" AGGREGATING VALUES FROM WORKER NODES BACK TO DRIVER PROGRAM "
" create by calling 'SparkContext.accumulator(initial value) - return typw is 'org.apache.spark.Accumulator[T] object (write-only which means worker node can't read value from it only driver program) where T is initial value "
" Worker code in Spark closures can add the accumulator by += "
" Driver program can call 'value' on object to access value "
" Accumulator in Action - Only call once; Accumulator in transformation - call multiple times when Spark detect fail or slow "
" Custom Accumulator - initial value can also be Double, long, float, Int; aggregation method can also be Max - P104"
file = sc.textFile(inputFile)
# Create Accumulator[Int] initialized to 0 * can create multiple Accumulators for different purposes
blankLines = sc.Accumulator(0)

def extractCallSigns(line):
  global blankLines # Make the global variable accessible
  if (line == ""):
    blankLines += 1
  return line.split(" ")

callSigns = file.flatMap(extractCallSigns)
callSigns.saveAsTextFile(outputDir + "/callsigns") # Call 'avtion' to activate flow
print "Blank Lines: %d" % blankLines.value




2.broadcast variable # efficiently distribute large values
" ALLOWS PROGRAM TO EFFICIENTLY SEND A LARGE, READ-ONLY VALUE TO ALL THE WORKER NODES FOR USED IN ONE OR MORE SPARK OPERATIONS "
" For example, if your application needs a large, read-only lookup table to all nodes, or large feature vector in a machine learning algorithm "
" 'broadcast a object to make it read-Only"

# Lookup the locations of the call signs on the
# RDD contractCounts. We load a list of call sign
# prefixes to country code to support this lookup 
signPrefixes = sc.broadcast(loadCallSignTable()); " alternative - signPrefixes = loadCallSignTable() "

def processSignCount(sign_count, signPrefixes):
  country = lookupCountry(sign_count[0], signPrefixes.value)
  count = sign_count[1]
  return (country, count)

countryContactCounts = (contactCounts
                      .map(processSignCount)
                      .reduceByKey((lambda x, y: X + y)))

countryContactCounts.saveAsTextFile(outputDir + "/countries.txt")





3.pre-partition operations 
" Avoid doing setup work for each data item, like opening a database connection, or creating a random number "
" generator. Spark has pre-partition versions of map and foreach to help reduce the cost of these operations by letting you "
" run code only once for each partition of an RDD "
" P-108 "
.mapPartitions()
.mapPartitionsWithIndex()
.foreachPartition()

def processCallSign(sign):
  .........
  return (x)

def fetchCallSigns(input):
  """ fetch call signs """
    return input.mapPartitions(lambda callSigns : processCallSign(callSigns)) # map partition on the call

contactsContactList = fetchCallSigns(validSigns)






4.piping to external program: R
" P - 110 "
# --- finddistance.R
#!/usr/bin/env Rscript
library("Imap")
f <- file("stdin")
open(f)
while(length(line <- readLines(f, n=1)) > 0) {
  " process line"
  contents <- Map(as.numeric, strsplit(line, ","))
  mydist <- gdist(contents[[1]][1], contents[[1]][2],
                contents[[1]][3], contents[[1]][4],
                units="m", a=6378137.0, b=6356752.3142, verbose = FALSE)
  write(mydist, stdout())
}
# --- R

# --- Python
# Compute the distance of each call using an external R program
distScrpt = "./src/R/finddistance.R"
distScriptName = "finddistance.R"
sc.addFile(distScript)
def hasDistInfo(call):
  """ Verify that a call has the fields required to compute the distance """
  requiredFields = ["mulat", "mylong", "contactlat", "contactlong"]
  return all(map(lambda f: call[f], requiredFields))
def formatCall(call):
  """ Format a call so that it can be parsed by our R program """
  return "{0},{1},{2},{3}".format(
    call["mylat"], call["mylong"],
    call["contactlat"], call["contactlong"])
pipeInputs = contactsContactList.values().flatMap(
  lambda calls: map(formatCall, filter(hasDistInfo, calls)))
distances = pipeInputs.pipe(SparkFiles.get(distScriptName))
print distances.collect()






5.numeric RDD operations 
" Descriptive Satistics operations on RDDs "

RDD.count() # Number of elements in the RDD

RDD.mean() # Average of the elements

RDD.sum() # Total

RDD.max() # Maximum value

RDD.min() # Minimum value

RDD.variance() # Variance of the elements

RDD.sampleVariance() # variance of the elements, computed for a sample

RDD.stdev() # standard deviation

RDD.sampleStdev() # Sample standard deviation

























# ----------------------------------------------------------------------------------------------------------------------------------------- Running on a Cluster

" You can use local mode to prototype and then add instances to use the cluster mode. You will also be able to use the same API. User can prototype on smaller machine "
" and then run on larger cluster using the same code. "

-1- "Spark Run Time Architecture "

              ##############################
              #                            # * iNTERACTE WITH CLUSTER MANAGERS 
              #        Spark Driver        # * CONVERT A USER PROGRAM INTO TASKS
              #                            # * SCHEDULING TASKS ON EXECUTORS
              ############################## [1] DIFFERENT JAVA PROCESS THAN EXECUTORS (SAME JAVA PROCESS - LOCAL MODE)
                                          |
                                          |
                                          |
                                          |
              ############################## [1] Mesos
              #                            # [2] YARN 
              #       Cluster Manager      # [3] Standalone
              #                            #  * CREATE AND DISTRIBUTE THE JOBS GAVE BY DRIVER 
              ##############################  * PLUGGABLE COMPONENT - INITIALLY LAUNCH DRIVER AND EXECUTORS
                                          |                 # CONTROLS RESOURCE YOUR APPLICATION GETS
                                          |
                                          |
                                          |
        |================================================================|
        |                                 |                              |
        |                                 |                              |
####################            ####################           #################### * RUN INDIVIDUAL TASK ON A GAVEN JOB
#  Cluster Worker  #            #  Cluster Worker  #           #  Cluster Worker  # * RETURN THE FINISHED RESULT TO DRIVER
#    (Executor)    #            #    (Executor)    #           #    (Executor)    # * PROVIDES IN-MEMORY STORAGE OF THE RDD CREATED BY DRIVER (BLOCK MANAGER)
####################            ####################           #################### [1] DIFFERENT JAVA PROCESS THAN DRIVER OR OTHER EXECUTORS (SAME JAVA PROCESS - LOCAL MODE)


-2-
#######################
# Launching a Program # 
#######################
spark-submit ...
" No matter which cluster manager you are using, Spark provides a single script you can use to submit your program to it call 'spark-submit'. "
" Through various option, spark-submit can connect to different cluster manager and control how many resources your application gets. "

# template
$bin/spark-submit [options] <app jar | python file> [app options]


$bin/spark-submit my_script.py # submit locally
$bin/spark-submit --master spark://host:7077 --executor-memory 10g my_script.py # submit cluster mode
# detail about the options - P121-122
" Two categories - scheduling information [amount of resource,etc] | runtime dependencies [libraries, etc]"




-3-
###################################
# Packaging code and dependencies # 
###################################
" To make sure all your dependencies are present at the runtime of your Apark cluster "

# Python
" Pyspark uses the existing Python installation on worker machines, you can install dependency libraries directly "
" on worker machines, using standard Python package manager - pip"
pip install <package name>

# Scala
" use 'build.sbt' file " # use -jar to pass build file - sbt
" P - 126 "

** " Dependency Conflicts " --
" A user application and a Spark itself both depend on the same library "
" - Solution: modify application to depend on the same version of the thrid-party library that Spark does "




-4-
###################################################
# Scheduling within and between Spark Application # 
###################################################

" Situation that multiple users submitting jobs to a cluster (Usual situation) "

" Spark relies on Cluster Manager for this "

#########################
# Spark Cluster Manager #
#########################

--- Standalone --> " Quick start and new development \ only use Spark "
" - P 130 "
" copy 'Spark' file to home "
" SSH-Key Gen and copy to all workers "
" sbin/start-all.sh " / " sbin/stop-all.sh "
# --- Other system 
" bin/spark-class org.apache.spark.deploy.master.Master " 
" bin/spark-class org.apache.spark.deploy.worker.Worker spark://masternode:7077 "

# --- submit application
spark-submit --master spark://masternode:7077 your_app
# --- Web UI
http://masternode:8080 # check if cluster is running
# --- launch interactive shell
spark-shell --master spark://masternode:7070
pyspark --master spark://masternode:7077 

# two deploy modes
spark-submit app # runs only on the machine you execute the app --- client mode
spark-submit --deploy-mode cluster app # launch within a standalone cluster as another process on one of the worker node --- cluster mode

# configuring resource usage
--executor-memory # executor memory
--total-executor-cores # maximum number of total cores

# High availability
" Accept application even when individual nodes go down "
" - Use ZooKeeper - "




-- Hadoop YARN --> " Want to run Spark alongside other applications, or use richer resource scheduling capabilities."
" P - 133 "


-- Apache Mesos --> " Even better when multiple users are using interactive shell on Spark "
" P - 134 "

-- Amazon EC2
" P - 135 "




















# ----------------------------------------------------------------------------------------------------------------------------------------- Tuning and Debugging Spark
" How to configure a Spark Application and how to tune and debug production Spark workloads "


###############################
#  Application Configuration  #
###############################

--1. " Configure with 'SparkConf' class "
# Create a application using a SparkConf in Python
conf = new SparkConf() # this instance contains key/value pairs of configuration that user would like to override
conf.set("spark.app.name", "My Spark App") # call 'set' to add configuration values
conf.set("spark.master", "local[4]")
conf.set("spark.ui.port", "36000") # override the default port

# Create a SparkContext with this configuration
sc = SparkContext(conf)

--2. " Configure with 'spark-submit' tool "
# Setting configuration values at runtime using flags
$ bin/spark-submit \
  --class com.example.MyApp \
  --master local[4] \
  --name "My Spark App" \
  --conf spark.ui.port=36000 \
  myapp.jar

# Setting configuration values using a file
$ bin/spark-submit \
  --class com.example.MyApp \
  --properties-file my-config.conf \ 
  myapp.jar

cat my-config.conf
### Contents ###
spark.master local[4]
spark.app.name "my Spark App"
spark.ui.port 36000

" If both 'spark-submit' and set() are used, set is first "
" You can exame configuration in the web UI "

" Common configuration list - P 144 - 145 "

** "Almost all config made in 'SparkConf' except - setting the local storage directories for Spark to use for data shuffle "
export SPARK_LOCAL_DIRS # inside the conf/spark-env.sh to a comma-separated list of storage locations.



#######################################################
#  Components of ExecutionL: Jobs, Tasks, and Stages  #
#######################################################

--Example.

# ------- input.txt -------- #
INFO This is a message 
INFO This is another

INFO Here are more
WARN this is a warning

ERROR something bad
WARN More details
INFO back to normal
# ------- input.txt -------- #
#             |
#             |
#             |
#             |
#             |
#             |
# ------- Scala Spark File -------- #
// Read input file
val input = sc.textFile("input")
// Split into words and remove empty lines
val tokenized = input.
    |  filter(line => line.size > 0).
    |  map(line => line.split(" "))
// Extract the first word from each line (the log level) and do a count
val counts = tokenized.
    |  map(words => (words(0), 1)).
    |  reduceBykey{ (a,b) => a + b }
# ------- Scala Spark File -------- #
#             |
#             |
#             |
#             |
#             |
#             |
" Using '.toDebugString' "
# ------- Check back-end log tasks --------- #                                                                                                                          (Data Partitions for different machines)
input.toDebugString                                       #                                                                                                                               |
res85: String =                                           #                                                                                                                               |
(2) input.text MappedRDD[292] at textFile at <console>:13 # --- same------------ Stage 1                                                      |                             |           Task ....        |
 |  input.text HadoopRDD[291] at textFile at <console>:13 # --- same------------ Stage 1                                                      |  Job 1: A set of stages     |           Task ....        |
#                                                                                                                                                                           |           Task ....        |
counts.toDebugString                                 #                                                                                                                      |           Task ....        |
res84: String =                                      #                                                                                                                      |           Task ....        |           * Action called
(2) ShuffledRDD[296] at reducedBykey at <console>:17 # ------------------------- Stage 2 (Indention) * Only 2 stages since only 2 indentions  |                             |------->   Task ....        |---------> Shuffle back to
 +-(2) MappedRDD[295] at map at <console>:17         # ------------------------- Stage 1 (Indention)                                          |                             |           Task ....        |                  |
    |  FilteredRDD[294] at filter at <console>:15    # ------------------------- Stage 1                                                      |                             |           Task ....        |                  |
    |  MappedRDD[293] at map at <console>:15         # ------------------------- Stage 1                                                      |  Job 2: A set of stages     |           Task ....        |                  |
    |  input.text MappedRDD[292] at textFile at <console>:13 # --- same--------- Stage 1                                                      |                             |           Task ....        |            [Driver Program]
    |  input.text HadoopRDD[291] at textFile at <console>:13 # --- same--------- Stage 1                                                      |                             |           Task ....        |           {External Storage}

" All operations creates a directed acyclic graph (DAG) of RDD objects that will be used later once a action is called "
" Keeps pointers to one or more parents along with meta data about what type of relationship is. "
" Execute from bottom up " | " You can find it more in the Web UI "
#             |
#             |
#             |
#             |
#             |
#             |
# ------- Call a action -------- #
counts.collect()
res86: Array[(String, Int)] = Array((ERROR,1), (INFO,4), (WARN,2))
" When an Action called, Spark Scheduler creates a physical execution plan for the DAG above, every partition of the RDD must be materialized "
" tasks are created and dispatched to an internal scheduler" | " Stages in physical plan depend on each other - executed in sepcific order considering dependencies "
" and then transfer to the Driver program or saved to outside storage "



** # Caching RDD will reduce the stages required
counts.cache()
counts.collect() # First time, still requires 2 stages
counts.collect() # Second time, only 1 stage required 





######################
#  Find Information  #
######################

" Spark records keeps all detail information regarding clusters and applications which have been stored in two places: "
" One is in Web UI and another is in log file produced by Driver program "

--1. " Spark Web UI "
# This is available on the machine where the Driver program is running, with default port: 4040
" caveat: If use YARN cluster where application driver runs inside the cluster, you shoudld access UI through the YARN ResourceManager, "
" which proxies requests directly to the driver "
[1] "Jobs: Progress and metrics of stages, tasks, and more" # --- P 151
[2] "Storage: Information for RDDs that are persisted" # --- P 153
[3] "Executors: A list of executors present in the application" # P 153
[4] "Environment: Debugging Spark's configurations" # P 153



--2. " Driver and Executor Logs " 
# Logs contains more details of events such as internal warning or detailed execution of the code
" The exact location of Spark's logfiles depends on the deployment mode: "
[1] "Standalone Mode: directly display in master's web UI and also find it in the 'work/' directory of each worker machine "
[2] "Mesos Mode: directly display in Mesos's web UI and also find it in the 'work/' directory of each Mesos slave node " 
[3] "YARN Mode: Use YARN's log collection tool -- 'yarn logs -applicationId <app ID>' * Work only if the applicationhas fully finished."
    "For viewing a running application, use resourceManager UI to the node page, browse to particular node, from there a particular container." # - P 154


** "Configure log format" # - P 154
log4j
conf/logfj.properties.template # copy example to this file to modify
# Once done editing, add file to Spark
spark-submit .... --files log4j.properties





###################################
#  Key Performance Consideration  #
###################################
" Common performance issue you might encountered and tips to solve "

--1. "Level of Parallelism"
" A single partition contains a subset of total data --> Creates signle task for a partition "
" Usually a single task requires a single core " | " Like 1000 cores (1000 partitions allowed) available but only run 30 tasks "
" Too little parallelism --> Spark might leave resource ideal " | 
" Too much parallelism --> Creates significant overhead associated with each partition (small application with heavy partition)" |
# Two ways to tune the level of parallelism
[1] " During operations that shuffles data ---> Give a degree of parallelism to the produced RDD as a parameter"
[2] " Existing RDD to be redistributed to more or few partition ---> RDD.repartition(#) | RDD.coalesce(#) [scale down] "
# Example - 
# Wildcard input that may match thousands of files
input = sc.textFile("s3n://log-files/2014/*.log")
input.getNumPartitions()
35154
# A filter that excludes almost all data
lines = input.filter(lambda line: line.startswith("2014-10-17"))
lines.getNumPartitions()
35154
# We coalesce the lines RDD before caching
lines = lines.coalesce(5).cache()
lines.getNumPartitions()
5
# Subsequent analysis can operate on the coalesce RDD ...
lines.count()





--2. "Serialization Format"
" When Spark is transfering data over network or spilling to disk, it needs to serialize objects into binary format. "
" This comes into play in shuffle operation where large amount of data are transferred. "
# Default                                                # Supports (Option 2)
" Use Java's built-in serializer "                       " Kryo - a thrid party serialization library that improves - faster / compater "
# To use Kryo serializer
val conf = new SparkConf()
conf.set("spark.serializer". "org.apache.spark.serializer.kryoSerializer")
// Be strict about class registration
conf.set("spark.kryo.registrationRequired", "true")
conf.registerKryoClasses(Array(classOf[MyClass], classOf[MyOtherClass]))
# Whether use which one, you encountered error - "NotSerializableException" if your code refers to a class that does not extend Java's Serializable interface.
" Solution - JVM supports special option to help debug - find the class problematic "
spark-submit .... --driver-java-options --executor-java-options
" Once find the class - modify it to implement serializable "







--3. "Memory Management"
" Spark use memeory in different ways, you can customize it to optimize your application "
" Inside each executor - purpose of usage in memory: "

[1] " RDD storage - deafult 60% memory allocated "
persist()
cache()
" Store partition in memory buffer - if limit exceeds, old partition will be drop. When need will recompute again "
spark.storage.memoryFraction # Set limitation


[2] " Shuffle and aggregation buffers - deafult 20% memory allocated "
" When create shuffle operations, Spark creates buffer for store Shuffle output data "
spark.shuffle.memoryFraction # Set limitation


[2] " User Code - deafult 20%(Or rest) memory allocated "
" Spark executes arbitrary user code, so user functions themselves can require substantial memory - application allocates large array/objects"
" Spark usually allocate whatever the rest memory to 'User Code' usage "

** " Tip1 - Adjuct the limitation thresholds based on your application - ex. if your code is allocating large objects = increase user code (by decrease rest two)"
   " Tip2 - If RDD storage is big, When 'persist()' or 'cache()' use parameter - MEMORY_AND_DISK rather than MEMORY_ONLY *default could save you time, since if "
   "        exceed limitation and drop, recompute takes more time than read from DISK"
   " Tip3 - If cache() large amount of data like 'gigabytes', use MEMORY_AND_DISK_SER or MEMORY_ONLY_SER since even though serialization slow down process but it "
   "        saves lots time in 'garbage collection' which scales linear with # of objects but serialization save many records as one seralized buffer"
   "        ** check 'garbage collection - GC time column in UI for each task ** "










--4. "Hardware Provisioning"
" Very significant impact on completion time of Spark job "
" Spark allows linear scaling: adding twice the resource will generate twice the current performance = will benefits from having more memory and cores"

" Parameters: "

[1] " Amount memory per executor" | 
set spark.executor.memory | spark-submit .... --executor-memory 12387128

[2] " Amount of cores per executor" | 
" YARN " - set spark.executor.cores | spark-submit .... --executor-cores 40 
" Mesos " & " Standalone " - "will use as many cores as are offered by scheduler" - set spark.cores.max 1000 # can control total cores

[3] " Total amount of executors" | 
" YARN " | spark-submit .... ... --num-executors 40
" Mesos " & " Standalone " - "will use as many executors as are offered by scheduler"

[4] " number of local disks" | # Used combined in RDD storage & Shuffle - High number increase performance (Less recompuation if memory exceeds)

" YARN " - "Provides its own in YARN configure "
" Standalone " - set SPARK_LOCAL_DIRS "environment variable in" spark-env.sh "When deploying cluster will implement it "
" Mesos " - ser spark.local.dir 1000 

** " Tips1 - If want to cache large amount of data, subset data in a small cluster, check memory need for cache and then extrapolate to estimate total memory needed "
   " Caveat - Not 'The more the better' - very large heap size (one executor size) can cause 'garbage collection time' extramly large"
   "        - Solution 1 - YARN & Mesos can packing multiple smaller executors within one physical host - mechine | Manual in 'Standalone' - P 159 "
   "        - Solution 2 - Seralize object as mentioned in previous section "

























# ----------------------------------------------------------------------------------------------------------------------------------------- Spark SQL
" Spark's interface for working with Structured and semistructured data. Structured data: data that has a schema "
[1] " It provides a DataFrame abstraction in Python, Java, and Scala that simplifies working with structured data. DataFrame similar to relational database "
[2] " It can read/write data in a variety of structured formats - JSON, Hive, Tables, Parquet "
[3] " It lets you query data using SQL, both data in Spark program and external tools that connects to Spark SQL through the standard database connection JDBC/ODBC "


" DataFRame - extendion of the RDD model - contains an RDD of 'Row' objects, each representing a record (Store data in a more efficient way) - can be created from external data source or exisiting RDD "




############################
#  Linking with Spark SQL  #
############################
" If download Spark in binary form, Hive support already installed "
" If build from source - "
sbt/sbt -Phive assembly # P 163 

1. " Spark SQL built without Hive "
SQLContext



2. " Spark SQL built with Hive "
" Allows us access to Hive tables, UDFs, SerDes (Serialization and deserialization formats), and the Hive query language (HiveQL). No need for exisiting Hive installation"
HiveContext

" To connect Spark SQL to an exisiting Hive installation - "
" Copy 'hive-site.xml' to Spark's configuration directory ($SPARK_HOME/conf) "
" If you don't have an exisiting Hive, Spark SQL will still run and will create its own Hive metastore in your program's wd called 'metastore_db'. "
" If you attemps to create table using ' CREATE TABLE ....' it will be placed in 'user/hive/warehouse' dir on your default system "




####################################
#  Using SPARK SQL in Application  #
####################################

" Quickly load data and query it in SQL and same time program the code in Java, Python, and Scala "

# --- Initialize Spark SQL
# Import Spark SQL
from pyspark.sql import HiveContext, Row
# Or if you can't include the hive requirements
from pyspark.sql import SQLContext, Row


# --- Create the object
sc = SparkContext(...)
hiveCtx = HiveContext(sc) # we have a HiveContext, we are ready to load data and query it

" Basic Query Example - "
# --- Load data and query it
input = hiveCtx.jsonFile(inputFile)
# Register the input dataframe
input.registerTempTable("tweet")
# Select tweets based on the retweetConut
topTweets = hiveCtx.sql(""" SELECT text, retweetcount FROM tweets ORDER BY retweetCount LIMIT 10""")


-- " DataFrame " -- # Both loading data or executing queries return "DataFrame"
" DataFrame -- a RDD composed of Row objects with additional schema information of the types in each column "
" Row object -- wrappers around arrays of basic type (integers, strings) "
# DF provides a way to access RDD
DataFrame.rdd() # or register it to query

" Basic DataFrame Operations -- "
show(); # Show the content of DataFrame
df.show()

select(); # Select the specified fields / functions
df.select("name", df("age")+1)

filter(): # Select only the rows meeting the criteria
df.filter(df("age") > 19)

groupBy(); # Group together on a column, need to followed by an aggregation like min(), max(), mean()
df.groupBy(df("name")).min()

" Data Type that can be stored by DataFrame "
" - P 168 "


-- " Converting between DF and RDDs " -- 
# In Scala, Java
" P - 169 "
# In Python
topTweetText = topTweets.rdd().map(lambda row: row.column_name)




# --- Caching
" Since we know the type of the object, caching is more efficient in SPARK SQL "
" When we expect to run multiple tasks or queries against the same data - 'Cache' = Only exisits for the life of the driver program "
" To make sure we cache using memory efficient representation rather than full object - "
hiveCtx.cacheTable("tablename")
" Or - command line client on the JDNC server "
CACHE TABLE [tablename] 
UNCACHE TABLE [tablename]





#############################
#  Loading and Saving Data  #
#############################

" Supports a number of structured data source out-of-box to allow you get 'Row' objects from them easily "
" Sources: Hive table, JSON, Parquet files " | " If define only subset in query - SQL only scan subset of data (Different from Spark.Context.hadoopFile which scan everything) "

" Spark SQL also has a DataSource API which allows others to integrate with Spark SQL - API: Avro, Apache HBase, Elasticsearch, Cassandra, etc " # www.spark-packages.org
" Copy 'hive-site.xml' to Spark's configuration directory ($SPARK_HOME/conf) "

-- Hive --
hiveCtx = hiveContext(sc)
rows = hiveCtx.sql("Select key, value FROM mytable")
keys = rows.map(lambda row: row[0])

# Write back to hive table
saveAsTable("tablename")


-- Data Source / Parquet --
# Parquet - a popular column-oriented storage format 
rows = hiveCtx.load(parquetFile, "parquet") # If use Avro - change to "com.databricks.spark.avro"
names = rows.map(lambda row: row.name)
print "Everyone"
print names.collect()
# Parquet query
tbl = rows.registerTempTable("popele")
pandaFriends = hiveCtx.sql("Select .... ")
print "Pandas Friend"
print PandasFriends.map(lambda row: row.name).collect()
# Save Parquet file
pandasFriends.save("hdfs://...", "parquet") # If use Avro - change to "com.databricks.spark.avro"



-- JSON --
" Spark can infer the schema from your JSON file "
input = hiveCtx.jsonFile(inputFile);

printSchema()
# check schema of Json file

SELECT key1[0].key2[2].key3 FROM table 
# access different level in JSON file


-- From RDD --
" Create DataFrame from RDD "
RDD = sc.parallelize([Row(name="holden", favour="coffee")])
df = hiveCtx.inferSchema(RDD)
df.registerTempTable("df_peope")



-- JDBC / ODBC Server --
# JDBC Driver
" Spark SQL also privodes JDBC connectivity - JDBC server runs on a standalone driver program which can be shared by multiple users "
" Requires Spark built with Hive support "
# Launch JDBC server
./sbin/start-thriftserver.sh --master sparkMaster # In Spark dir / default listen on localhost:10000
# Change configures - P 177
(HIVE_SERVER2_THRIFT_PORT)
(HIVE_SERVER2_THRIFT_BIND_HOST)

" Connecting to the JDBC server - use 'Beeline' "
$ ./bin/beeline -u jdbc:hive2://localhost:10000
....
....
> show tables;

# ODBC Driver
" Produced by Simba (www.simba.com) and can be download from various Spark vendors "
" OBDC commonly used by BI tools like tableau - check your BI tools and see how it can be connect to Spark SQL - most those have Hive connector can connect Spark SQL "

" Beeline " - " Within the Beeline command, you can use HQL to query tables - P 178 "

[Long lived Tables and Queries]
" One Advantage of Spark SQL JDBC server - share cached tables between multiple programs. Since JDBC Thrift server is a single driver program. "
" To do so, just need to" .registerTempTable() "then run" .cache()



-- User Defined Function (UDF) --
" It allows you to register functions in Python, Java, Scala and use it within SQL. "
" It supports its own Spark UDF interface as well as Apache Hive UDFs "

"--- Spark SQL UDFs" 
# Import the IntegerType we are returning
from pyspark.sql.types import IntegerType
# make a UDF to tell us how long some text is
hiveCtx.registerFunction("strLenPython", lambda x: len(x), IntegerType())
lengthDataFrame = hiveCtx.sql("SELECT strLenPython('text') FROM tweet LIMIT 10")


"# --- Hive UDFs"
" Exisiting Hive UDFs already included " # Hive programming details
" Must use hiveCtx " hiveCtx.sql("CREATE TEMPORARY FUNCTION name AS class.function")



###########################
#  Spark SQL performance  #
###########################

" Allows Spark more efficiently processing data - types, high level query"
" Allows easy conditional aggregations " | " Minimize the data read / read only subset "

# Performance Tunning Parameters
" In Beeline to set those parameters - "
$ ./bin/beeline -u jdbc:hive2://localhost:10000
beeline> set spark.sql.codegen=true;
SET spark.sql.codegen=true

"Option"                                            "Default"              "Usage"
-------------------------------------------------------------------------------------------------------------------------------
spark.sql.codegen                                    false                  "When true, Spark SQL will compile each query "
                                                                            "to Java bytecode on the fly. This can improve " 
                                                                            "performance for large queries, but codegen can "
                                                                            "slow down very short query"
spark.sql.inMemoryColumnarStorage.compressed         true                   "Compress the in-memory columnar storage automatically. "
                                                                            "previously defaulted to false "
spark.sql.inMemoryColumnarStorage.batchSize          1000                   "The batch size for columnar caching. Larger values may "
                                                                            "cause out-of-memory problems"
spark.sql.parquet.compression.codec                  snappy                 "Which compression codec to use. Possible options include"
                                                                            " 'uncompressed, snappy, gzip, and lzo."
--------------------------------------------------------------------------------------------------------------------------------























# ----------------------------------------------------------------------------------------------------------------------------------------- Spark Streaming

" Much like Spark is built on RDDs, Spark Streaming is built on an abstration called 'DStream' - discretized stream. A DStream is a sequence of data arriving over time. "
" It takes data from different source - Flume, KafKa, HDFS "
" Once a DStream built -> Two types of operations = [transformation] - which yeids a new DStream \ [output] - which write data to an external system. "

-- "Simple Example" --

" Recieve a stream of newline delimited lines of text from a server running at port: 7777, filter only the lines that contain the word 'error' and print them "

# -- Maven build file -- #
groupId = org.apache.spark
artifactId = spark-streaming_2.10
version = 1.3.0
# ---------------------- #


# ----- Scala imports ----- #
# ------- Scala Script ----------- #

import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.StreamingContext._
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.streaming.Duration
import org.apache.spark.streaming.Seconds


// "Create a StreamingContext with a 1-second batch size from a SparkConf"
val ssc = new StreamingContext(conf, Seconds(1))
// "Create a DStream using data received after connecting to port 7777 on the local machine "
val lines = ssc.socketTextStream("localhost", 7777)
// "Filter our DStream for lines with 'error'"
val errorLines = lines.filter(_.contains("error"))
// "print out the lines with errors "
errorLines.print()
# -------------------------------- #

" This set up only process when recieved data "

// "Start our streaming context and wait for it to 'finish' "
ssc.start()
// "Wait for the job to finish"
ssc.awaitTermination()

** "Streaming Context can be start only once and must be started after we set up all operations we want "


# ----------- Run the App ------------------- #
$ spark-submit --class com.oreilly.learningsparkexamples.scala.StreamingInput ASSEMBLY_JAR local[4]
$ nc localhost 7777 # Lets type your input lines to send to server
<type in>





-- "Architecture and Abstraction" ---------------------------------------------------------------------------------------------------------------------------------------

#### Data Sources ##########           ############################ Spark Streaming Workflow #######################################################################

#                                             <By Time Intervals>                       #####################
############      Data                                                       
# SOURCE 1 #---------------|            |        |        |        |                         Executor 2                      ____________________________________________________________________
############               |            |        |        |        |                                                        |                                                                   |
#                          |            |        |        |        |                    #####################               | Checkpointing - every 5-10 batches of data                        |
#                          |    AGG     |        |        |        |                                                   |----|                 save to a reliable file system like HDFS          |
############      Data     |  streams   |        |        |        |                                                   |    |                 Speed up recomputation, start at latest checkpoint|
# SOURCE 2 #---------------|---------->>|        |        |        |                                                   |    |___________________________________________________________________|              
############               |    |      0|_______1|_______2|_______3|_______4...>                                       |                   
#                          |    |            |        |        |                                                       |                   
#                          |    |            |        |        |                                                       |                  
############      Data     |    |          RDD1     RDD2     RDD3  ...   RDD4 ...    <---------------------------------|
# SOURCE 3 #---------------|    |            |        |        |
############                    |        [Spark]   [Spark]   [Spark] ..  [Spark] ... -------------- (Micro batch Jobs runs when one interval ends, periodically)
#                               |            |        |        |                                       
#                               |          RDD(1)   RDD(2)   RDD(3)       RDD(4) ... 
#                               |            |        |        |
#                               |            |        |        |
#                               |          *Sent    *Sent    *Sent --------------------------------->>>> External system *
#                               |           AGG  +   AGG   +  AGG  --------------------------------->>>> Aggregate Result across times *
#                               |
############################    |       #############################################################################################################################
#                               |
#                              \/
#                       ################    ###########################
#
#                          Executor 2        cached - Data1 Data2 ...    ------------------- Recompute if jobs in Executor 1 failed
#
#                       ################    ###########################                      * However, it can be slow to recompute if cumulated data is big
#
#

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------



-- "Transformation" --

--1. "Transformation - Stateless --> Process each batch without depending on the data of "
                                    "its previous batches. Included common RDD transformations - map(), filter(), reduceBykey() "
                                    ** # Ex. DStream.reduceBykey() - unique keys in each batch NOT across time

# ---- Stateless DStream transformations (More..)
.map()          -> ds.map(x => x + 1) # Apply a function to each element in the DStream and return a DStream of result
.flatMap()      -> ds.flatMap(x => x.split(" ")) # Apply a function to each element in the DStream and return a DStream of the contents of the iterators returned
.filter()       -> ds.filter(x => x != 1) # Return a DStream consisting of only elements that pass the condition passed ti filter
.repartition()  -> ds.repartition(10) # change the number of partitions of the DStream
.reduceBykey()  -> ds.reduceBykey((x,y) => x + y) # Combine values with the same key in each batch
.groupBykey()   -> ds.groupBykey() # Group the values with the same key in batch

" Example - use map() and reduceBykey() to count log events by IP address in each batch "
// "Assumes ApacheAccessLog is a utility class for parsing entries from Apache logs "
val accessLogDStream = logData.map(line => ApacheAccessLog.parseFromLogLine(line))
val ipDStream = accessLogsDStream.map(entry => (entry.getIpAddress(),1))
val ipCountsDStream = ipDStream.reduceBykey((x,y) => x + y)

" Example - can also still combine data from multiple DStreams, again only within each batch." ds.cogroup() - ds.join() - ds.leftOuterJoin()
"           We had data keyed by IP address, and we join the request count against the bytes transformed "
val ipBytesDStream = accessLogsDStream.map(entry => (entry.getIpAddress(), entry.getContentSize()))
val ipBytesSumDStream = ipBytesDStream.reduceBykey((x,y) => x + y)
val ipBytesRequestCountDStream = ipCountsDStream.join(ipBytesSumDStream) 

" Example - we can also merge the contents of two different "
sd.union() - StreamingContext.union() # multiple streams
val outlierStream = accessLogsDStream.transform { rdd => extractOutliers(rdd) } # Use regular Spark code function 'extractOutliers()' on each RDD of a DStream







--2. "Transformation - Stateful --> Process each batch using data or intermediate results from "
                                   "previous batches to compute the results of the current batch. Included transformations on "
                                   "sliding windows and on tracking state across time. "
                                   ** # Ex. DStream.reduceBykey() - unique keys across times (Multiple batches)
                                   ** "Requires 'checkpointing' been enabled in StreamingContext "
                                   ssc.checkpoint("hdfs://...") # enable by passing a directory to it
                                   ssc.checkpoint("/temp") # also local path works if developing on local

" Window Transformation "
" - All window transformations need two parameters - 1. window durations: controls how many previous batches of data considered - 'windowDuration/batchInterval' (interval = 10 sec, windowDuration = 30 sec = 3 batches) " 
"                                                    2. sliding durations: which default to the batch interval, controls how freuqently the new DStream computes results (interval = 10 sec, slideInterval = 20 sec = 2 batches to compute) "

# ---- Stateful DStream transformations (More..)
.updateStateBykey()       -> ds.updateStateBykey() # Used to track state across events for each key
.window()                 -> ds.window(Seconds(30), Seconds(10)) # Returns a new DStream with the data for the requested window - each RDD in the DStream resulting from window() will contain data from multiple batches, which we can process with count(), transform(), etc.
.reduceByWindow()         -> ds.reduceByWindow(
                                   {(x,y) => x + y}, // " Adding elements in the new batches entering the window "
                                   {(x,y) => x - y}, // " Removing elemnts from the oldest batches exiting the window "
                                   Seconds(30), // "Window duration"
                                   Seconds(10)) // "Slide duration"
.reduceBykeyAndWindow()   -> ds.reduceByWindow(
                                   {(x,y) => x + y}, // " Adding elements in the new batches entering the window "
                                   {(x,y) => x - y}, // " Removing elemnts from the oldest batches exiting the window "
                                   Seconds(30), // "Window duration"
                                   Seconds(10)) // "Slide duration"
.countByWindow()          -> ds.countByWindow(Seconds(30), Seconds(10))
.countByValueAndWindow()  -> ds.countByValueAndWindow(Seconds(30), Seconds(10))






" Example - A windowed stream with a window duration of 3 batches and a slide duration of 2 batches; every two time steps, we compute a result over the previous 3 batches "

val accessLogsWindow = accessLogsDStream.window(Seconds(30), Seconds(20)) // "window duration of 3 batches, slide duration of 2 batches (every 20 seconds) "
val windowCounts = accessLogsWindow.count() // "Build function on top of window()"

### Network Input ###           #### Windowed Stream ####
#   Interval: 10 sec                 Window: 30 sec
#                                    slide: 20 sec
#
#
#########                          
#   1   #\                        
######### \
#          \ 
#           ---------------\
#########                   \      #########
#   2   #--------------------------#   1   #
#########\                         #########
#         \
#          \
#########   \                      
#   3   #----\                    
#########     \                    
#              \____________ 
#                           \
#########                    \     #########
#   4   #--------------------------#   2   #
######### \                        #########
#          \
#           \
#########    \                
#   5   #-----\              
#########      \______________                   
#                             \
#                              \
#########                       \  #########
#   6   #------------------------\-#   3   #
#########                          #########



" Example - .reduceByWindow() / reduceByKeyAndWindow() allow reduction on each more efficient "

val ipDSTream = accessLogsDStream.map(logEntry => (logEntry.getIpAddress(),i))
val ipCountDStream = ipDStream.reduceByWindow(
  {(x,y) => x + y}, // " Adding elements in the new batches entering the window "
  {(x,y) => x - y}, // " Removing elemnts from the oldest batches exiting the window "
  Seconds(30), // "Window duration"
  Seconds(10)) // "Slide duration"


### Network Input ###           #### Windowed Stream ####
#   Interval: 10 sec                 Window: 30 sec
#                                    slide: 10 sec
#
#
#########                          
#  1,1  #\------------------------------------------------------                        
######### \                                                    |
#          \                                                   |
#           \                                                  | 
#########    \                                                 |
#  4,2  #-----\-------------------------------------           |
#########\     \                                   |           |
#         \     \+ 2                               |           |
#          \+ 6  \                                 |           |
#########   \     |                                |           |
#   9   #----\    |                                |           |
######### +9  \   |                                |           |
#              \__|_________                       |           |
#                           \+ 17                  |           | 
#########                    \     #########       |           | 
#   3   #--------------------------#   20  #       |           |
#########                + 3       #########       |           |  
#                                      |           |           |
#                                      |+ 20       |           |- 2 * This Batch out of the 3 previous batches window
#########                          #########       |           |
#  3,1  #--------------------------#   22  #--------------------
#########                + 4       #########       |       
#                                      |           | 
#                                      |+ 22       |- 6 * This Batch out of the 3 previous batches window
#########                          #########       |
#   1   #--------------------------#   17  #--------
#########                + 1       #########



" Example - .countByWindow() and .countByValueAndWindow() "

val ipDStream = accessLogsDStream.map{entry => entry.getIpAddress()}
val ipAddressRequestCount = ipDStream.countByValueAndWindow(Seconds(30), Seconds(10))
val requestCount = accessLogsDStream.countByWindow(Seconds(30), Seconds(10))




" Example - .updateStateBykey() - To maintain state across the batches in the DStream, ex. track sessions as users vist a site "

def updateRunningSum(values: Seq[long], state: Option[long]) = {
  Some(state.getOrElse(0L) + values.size)
}
" update( events: a list of events arrived in the current batch,"
"         oldState: an optional state object, stored within an Option; it might bemissing if no previous state"
"       )"

" Keep running count of the number of log messages with each HTTP response code "
val responseCodeDStream = accessLogsDSTREAM.map(log => (log.getResponseCode(), 1L))
val responseCodeCountDStream = responseCodeDStream.updateStateByKey(updateRunningSum _)








-- "Output Operations" --
" Output operations specify what  needs to be done with the final transformed data in stream (ex. pushing it to a external database or print it to screen) "


# ---- Output operations
.print()                 -> ds.print() # first 10 elements
.saveAsTextFiles()       -> ds.saveAsTextFiles("outputDir", "txt") # Save each batch to txt file
.saveAsHadoopFiles()     -> ds.saveAsHadoopFiles[SequenceFileOutputFormat[Text, LongWritable]]("outputDir", "txt") # allow save file in Hadoop outputFormat
.foreachRDD()            -> ds.foreachRDD{ "...operations on RDD..."}


" Example - .print() **Usually used in debugging the output operations - returns the first 10 elements from each batch"
DStream.print()



" Example - .saveAsTextFiles() **Export ds and save each batch to save as subdirectory in the given directory with time and suffix in the filename."
ipAddressRequestCount.saveAsTextFiles("outputDir", "txt") 



" Example - .saveAsHadoopFiles() **takes a Hadoop Outputformat - ex. Sequence File "
val writableIpAddressRequestCount = ipAddressRequestCount.map {
  (ip, count) => (new Text(ip), new LongWritable(count)) 
}
writableIpAddressRequestCount.saveAsHadoopFiles[SequenceFileOutputFormat[Text, LongWritable]]("outputDir", "txt")



" Example - .foreachRDD() ** Generic output operation let us run computations on the RDDs on the DStream. Like transform() which gives access to each RDD "
"                            Usecase example - write data to an external database "
ipAddressRequestCount.foreachRDD { rdd =>

  rdd.foreachPartition { partition => 
    // "open connection to storage system (e.g. a database connection)"
    partition.foreach { item => 
      // "Use connection to push item to system"
      }
    // "Close connection"
  }
}








-- "Input Source" --
" Some 'core source' are built into the Spark Streaming Maven artifact while others are available through additional artifacts "
" If designing a new application, suggested to start with HDFS and Kafka "



# Example of different input recievers
// "Create a StreamingContext with a 1-second batch size from a SparkConf"
val ssc = new StreamingContext(conf, Seconds(1))

1."sockets - local machine port 7777 "
// "Create a DStream using data received after connecting to port 7777 on the local machine "
val lines = ssc.socketTextStream("localhost", 7777)


2. "Stream of files - reading from any Hadoop-compatible filesystem (Popular option due to wide supports) - "
// "rquirements: consistent data format, directory names, files have to be created atomically (ex. by moving the file into the directory Spark is monitoring so that Spark can detetcts 'mv file') "
val logData = ssc.textFileStream("logDir")
// "Streaming SequenceFiles written to a directory"
val logData = ssc.fileStream[LongWriteable, IntWritable, 
                  SequenceFileInputFormat[LongWriteable, IntWritable]](inputDirectory).map {
                  case (x, y) => (x.get(), y.get())
                  }


3. "Akka actor stream P - 202"
import org.apache.spark.streaming.receiver.ActorHelper
ssc.actorStream()


4. "Apache Kafka P - 203-204 (e.g. Apache kafka subscribing to Panda's topic "
import org.apache.spark.streaming.kafka._
...
// "Create a map of topics to number of reciever threads to use"
val topics = List(("pandas",1), ("logs",1)).toMap
val topicLines = kafkaUtils.createStream(ssc, zkQuorum, group, topics)
topicLines.print()



5. "Apache Flume - P - 204"
" Push-based receiver - P 205"
"Pull-based receiver - P 206"




6. "Custom input sources - you can also implement your own receiver "
"Spark's documentation in Streaming Custom Receivers guide "


** "We can combine DS from different source using operations like '.union()' so we can combine multiple inputs "
** "Each receiver runs as a long running task within executors and occupies CPU cores allocated to application - which "
** "means in order to run receivers you should have as many cores as receivers plus those enough to run computations "
** "(e.g. - 10 receivers need at least 11 cores to run) "






-- "Configurations for running 24 / 7" --
" Strong fault tolerance - 'exactly once semantic' - just like only processed once "
" What you need to do: "

1. "Checkpointing"
*"Limiting the state that must be recomputed on failure" | *"Providing fault tolerance for the driver"
ssc.scheckpointing("hdfs://...")


2. "Driver Fault Tolerance"
" The faulre of driver node requires a special way of creating ssc."
def createStreamingContext() = {
  ...
  val sc = new SparkContext(conf)
  // "Create a StreamingContext with a 1 second batch size "
  val ssc = new StreamingContext(sc, Seconds(1))
  ssc.checkpointing(checkpointDir)
}
...
val ssc = StreamingContext.getOrCreate(checkpointDir, createStreamingContext)

" In order to restart when driver crush, you need to use tool like 'monit' and restart it - specify your environment "
# e.g. in standalone mode using '--supervise' flag to monitor 
./bin/spark-submit --deploy-mode cluster --supervise --master spark://... App.jar 

** "Want Spark cluster manager: standalone to be fault-tolerance - Using ZooKeeper P - 210"


3. "Worker Fault Tolerance"
" Automatically, input data RDD has been replicated to other nodes in case worker crushed "


4. "Receiver Fault Tolerance"
" Spark restart it on other node if receiver failed - If it lost data? -> It depends on the nature of the source"
** "Choose reliable source: HDFS, directly consumming Kafka pr pull-based Flume (Source can resend data, data is already replicated, update the source with recieved data)"
** "Should enable 'write ahead logging: it ensures that your batch jobs and streaming jobs will see the same data and produce the same result"
spark.streaming.receiver.writeAheadLog.enable "to" TURE


5. "Processing Guarantees"
" Within Spark operation, recomputation gets the same result as it was only computed once "
" However, if the operation is output operation, it may send the data multiple times to external system "
" depends on outside system to detect and deduplicates data"
** "saveAs...File() in Spark automatically deduplicate and check data "


-- "Streaming UI" --
" Streaming tab on the normal Spark UI: 4040 "
" Statistics for batch processing and receivers. - fails, how many records each receiver process, time takes to finish, delay, etc"



-- "Performance Consideration" --
" On top of general optimization tips - few tips for streaming process "

--1. "Batch and window size"
" Usually 500 milliseconds is a good choice for batch size. You can also test by start with a big batch size like 10 seconds and work it down, "
" check whether the processing time changes, if not, can continue decrease batch size, if time increased, you may already reached limit of your application"
" For window, consider increase window interval for expensive computations. "

--2. "Level of Parallelism"
" A common way to reduce the processing time of batches is to increase the parallelism."
" [1] - Increase the number of receivers: You may want multiple receivers to create multiple DStream, then apply '.union()' to combine when they are done."
" [2] - Explicitly repartitioning received data: If no more receivers can be added, you can further partitioning the received data - like 'ds.repartition()'"
" [3] - Increase parallelism in aggregation: For operation like 'reduceByKey()' you can specify the parallelism as a second parameter."

--3. "Garbage Collection and Memory Usage"
" Java GC may cause problem: "
" enable Concurrent Mark-Sweep garbage collector whihc does consume more resources overall but introduce less pauses"
$ spark-submit --conf spark.executor.extraJavaOptions=-XX:+UseConcMarkSweepGC App.jar
" Or can cache RDDs in seralized form( in stead of as native objects) also reduces GC pressure. "





















# ----------------------------------------------------------------------------------------------------------------------------------------- Machine Learning with MLlib
" MLlib is Spark's library of machine learning functions which designed to run in parallel on cluster - functions run on RDD "
" It mainly contains algorithms that are good for parallel computation "
" Good to run algorithm on a large dataset. If many small datasets on which you want to train different model, better to use single node learning library - scikit learn and call it with parallelize across node use 'map()' "
"                                           If tune different parameters on small dataset, use Spark's 'parallelize()' over your list of parameters to train different models on different nodes. "

** "MLlib Requirements - 'gfortran' runtime library P - 216 | 'NumPy' in Python P - 216 " 

" Example - Email Spam classifier: "
from pyspark.mllid.regression import LabeledPoint
from pyspark.mllid.feature import HashingTF
from pyspark.mllid.classification import LogisticRegressionWithSGD

spam = sc.textFile("spam.txt") # txt file with each line represents a email message
normal = sc.textFile("normal.txt")

# Create a HashingTF instance to map email text to vectors of10,000 features.
tf = HashingTF(numFeatures = 10000)
# each email is split into words, and each words is mapped to one feature.
spamFeatures = spam.map(lambda email: tf.transform(email.split(" ")))
normalFeatures = normal.map(lambda email: tf.transform(email.split(" ")))

# Create LabeledPoint datasets for positive (spam) and negative (normal) example.
positiveExamples = spamFeatures.map(lambda features: LabeledPoint(1, features))
negativeExamples = normalFeatures.map(lambda features: LabeledPoint(1, features))

trainingData = positiveExamples.union(negativeExamples)
trainingData.cache() # cache since Logistic Regression is an iterative algorithm

# Run Logistic Regression using the RDD algorithm.
model = LogisticRegressionWithSGD.train(trainingData)

# Test on a positive example and negative example. We apply the same 'hashingTF' to get vectors amd then apply the model.
posTest = tf.transform("O M G GET cheap stuff by sending money to ...".split(" "))
negTest = tf.transform("Hi Dad, I am starting to learn ...".split(" "))
print "Prediction for positive test example: %g" % model.predict(posTest)
print "Prediction for negative test example: %g" % model.predict(negTest)




-- "Data Types" --
" You will firstly need to create RDD on external data and then transform to the following objects..."
import org.apache.spark.mllib # Java/Scala
import pyspark.mllid # Python

1. "Vector P - 221"
" A mathematical vector - dense vector: every entry stored"
"                         parse vector: only non-zero entry stored"
# In Java/Scala
import org.apache.spark.mllid.linalg.Vectors 

# In Python
from numpy import array
from pyspark.mllid.linalg import Vectors 
# --- Dense Vector
# Create the dense vector <1.0, 2.0, 3.0>
denseVec1 = array([1.0, 2.0, 3.0]) # Numpy array can be passed directly to MLlib
denseVec2 = Vectors.dense([1.0, 2.0, 3.0]) # .. or you can use the Vectors class
# --- Sparse Vector
# Create the sparse vector <1.0, 0.0, 2.0, 0.0>
sparseVec1 = Vectors.sparse(4, {0:1.0, 2:2.0}) # 4 slots, position 0 = 1.0, position 2 = 2.0, rest = 0
sparseVec2 = Vectors.sparse(4, [0,2],[1.0,2.0])



2. "LabeledPoint"
" A labeled data point for supervised learning algorithms includes a "
" feature vector and a label (which is a floating-point value)"

import org.apache.spark.mllib.regression 
import pyspark.mllid.regression



3. "Rating"
" A rating of a product by a user - product recommendation"

import org.apache.spark.mllib.recommendation
import pyspark.mllid.recommendation



4. "Various 'Model' classes"
" Each model is the result of a training algorithm, and typically has a predict() method"

model = LogisticRegressionWithSGD.train(trainingData)






-- "Algorithms" --
" Cover key algorithms available in MLlib "

[1] "--- Feature Extraction"

1."TF-IDF"
" is simply a way to generate feature vectors from text documents - it calculate the term frequency which is the number of times occur in the document TF"
" and the inverse document frequency IDF" | "HashingTF (good for expensive) or IDF"
# Example
from pyspark.mllid.feature import HashingTF
sentence = "hello hello world"
words = sentence.split() # Split sentence into a list of terms
tf = HashingTF(10000) # Create vectors of size S = 10,000
tf.transform(words)
"SparseVector(10000, {3065: 1.0, 6861: 2.0})"

rdd = sc.wholeTextFiles("data").map(lambda (name, text): text.split())
tf = HashingTF()
tfVectors = tf.transform(rdd).cache() # Transforms an entire RDD, cache since will used twice
# Compute the IDF, then the TF-IDF vectors
idf = IDF()
idfModel = idf.fit(tfVectors)
tfidfVectors = idfModel.transform(tfVectors)

2."Scaling"
" All features have a mean of 0 and sd of 1"
from pyspark.mllib.feature import StandardScaler
vector = [Vector.dense([-2.0, 5.0, 1.0]), Vector.dense([2.0, 0.0, 1.0])]
dataset = sc.parallelize(vectors)
scaler = StandardScaler(withMean=True, withStd=True)
model = scaler.fit(dataset)
result = model.transform(dataset)
" Result: {[-0.7071, 0.7071, 0.0],[0.7071, -0.7071, 0.0]}"

3."Normalization"
" Normalizing vector to length 1"
Normalizer().transform(rdd)

4."Word2Vec"
" is a featurization algorithm for text based on neural networks that can be used to feed data into many down stream algoritms"
from pyspark.mllib.feature import Word2Vec
" P - 225"





[2] "--- Statistics"

0."Describtive Statistics"
RDD.mean(); RDD.stdev(); RDD.sum()

1."Statistical summary of an RDD of vectors"
from pyspark.mllib.stat import Statistics
Statistics.colStats(RDD)

2."Correlation matrix between columns in an RDD of vectors"
Statistics.corr(RDD, [method]) # method - pearson / spearman

3."Correlation between two RDDs of floating-point values"
Statistics.corr(RDD1, RDD2, [method]) # method - pearson / spearman

4."Pearson's independence test for every feature with label on an RDD of 'LabeledPoint' object"
Statistics.chiSqTest(RDD) # return an array shows p-value, test statistics, and DF of each feature







[3] "--- Classification / Regression - P 227"
" Both use 'LabeledPoint' class "

1."Linear Regression"
" Most common model - supports L1, L2 regularization - LASSO, RIDGE regression"
from pyspark,mllib.regression import LabeledPoint
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.regression import LassoWithSGD
from pyspark.mllib.regression import RidgeRegressionWithSGD
points = "[Create RDD of LabeledPoint]"
model = LinearRegressionWithSGD.train(points, iteration=200, intercept=True)
print "weight: %s, intercept: %s" % (model.weights, model.intercept)


2."Logistic Regression"
" Binary classification - takes LabeledPoints 0, 1 - returns a 'LogisticRegressionModel' class to predict new points"
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.classification import LogisticRegressionWithSGD
points = "[Create RDD of LabeledPoint]"
model = LogisticRegressionWithSGD.train(points, iteration=200, intercept=True)


3."Support Vector Machine"
" Binary classification - takes LabeledPoints 0, 1 - returns a 'SVMModel' class to predict new points"
from pyspark.mllib.classification import SVMWithSGD


4."Naive bayes"
" Multi-classes classification algorithm - commonly used in text classification with TF-IDF (Multinomial Naive Bayes)"
from pyspark.mllib.classification import NaiveBayes
" it supports 'lambda' used for soomthing" | "You can call it on an RDD of 'LabeledPoints' where labels are between 0 and C-1 for C class"
" Two parameters - 'theta' = matrix of class probabilities for each feature (of size C X D for C classes and D features)"
"                  'pi' = C-dimensional vector of class priors "


5."Decision Tree and Random forest"
" Can used for both classification and regression - returns 'WeightedEnsembleModel' object contains several trees"
from pyspark.mllib.tree import DecisionTree
points = "[Create RDD of LabeledPoint]"
model = DecisionTree.trainClassifier(points,2,['gini or entropy'], 5, 32, {1: 2, 2: 3}) # data, number of predicted class, impurity, maximum depth of tree, number of bins to split data into, categorical vars info - var1 - 2 classes, var2 - 3 classes 
model = DecisionTree.trainRegressor(points, 'variance', 5, 32, {1: 2, 2: 3}) # data, impurity, maximum depth of tree, number of bins to split data into, categorical vars info - var1 - 2 classes, var2 - 3 classes 
model.predict(new_points)
model.toDebugString() # print decision tree

from pyspark.mllib.tree import RandomForest
points = "[Create RDD of LabeledPoint]"
model = RandomForest.trainClassifier(points,100,['auto','all','sqrt', 'log2', 'onethird'], 123) # data, number of trees, subset of features, seed
model = RandomForest.trainRegressor(points,100,['auto','all','sqrt', 'log2', 'onethird'], 123) # data, number of trees, subset of features, seed
model.predict(new_points)
model.toDebugString() # print all decision trees







[4] "--- Clustering"

1."K-Mean"
" K-Mean | K-Mean ++"
from pyspark.mllib.clustering import KMeans
points = "[vector]"
model = KMeans.train(points) 
model.predict(new_points) # return the cluster centers 





[5] "---Collaborative Filtering and Recommendation"

1."Alternating Least Square"
" A recommender system using user's ratings and interactions to recommend new ones "
from pyspark.mllib.recommendation import ALS
" P - 233 "



[6] "---Dimensionality Reduction"

1."Principal component analysis"
from pyspark.mllib.feature import PCA as PCAmllib

rdd = sc.parallelize([
    Vectors.dense([1, 2, 0]),
    Vectors.dense([2, 0, 1]),
    Vectors.dense([0, 1, 0])])

model = PCAmllib(2).fit(rdd)
transformed = model.transform(rdd)



2."Singular value decomposition"
" P - 235"



[7] "---Model evaluation"

"P - 236"





-- "Tips and Performance Considerations" --

"Preparing Features"
" Scale feature, featurize correctly, label class correctly"


"Configuring Algorithms"



"Caching RDDs to Reuse"
" Important to 'cache' input dataset "


"Recognizing Sparsity"



"Level of Parallelism"



** "Pipeline API"







