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






















# [1] ------------------------------------------------------ Core Spark Concept

# - Run interactive Shell - #

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

# - Run standalone application - #

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




# -- Example of creating a application -- #

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
                        transforming exisiting RDDs, 
                        calling operations on RDDs to compute result

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













































