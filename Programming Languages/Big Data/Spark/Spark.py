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
 
 
 
 
 
4. " Database & key/value store " 
# Java Database Connectivity - 
 
 
 
# Cassandra - 
 
 
 
# HBase - 
 
 
 
# Elasticsearch - 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
# ----------------------------------------------------------------------------------------------------------------------------------------- Advanced Spark Programming
 
 
 
































