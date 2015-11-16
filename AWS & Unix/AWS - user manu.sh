## AWS

# Content
# CTRL f -> "Setup instance" >>> lanugh instances EC2
# CTRL f -> "method 1" >>> Single local machine - R-studio launch
# CTRL f -> "method 2" >>> Hadoop cluster
# CTRL f -> "method 3" >>> Build API connection + Connect to RDS + automate update



## <<<< Setup instance >>>>> ##
## [1]
## ---- Login Credential ---- ##
Username: whereislem@hotmail.com
Password: fgh67743185

## [2]
## EC2 - request instances - ubuntu (Usually use the AMI)
## link for Rstudio AMI
http://www.louisaslett.com/RStudio_AMI/

## [3]
## Security group config
SSH               TCP      22                   anywhere
All ICMP          ICMP     0 - 65535(or na)     anywhere
Custom TCP Rule   TCP      9001                 anywhere # can ignore if only one
Custom TCP Rule   TCP      9000                 anywhere # can ignore if only one
Custom TCP Rule   TCP      50000 - 50100        anywhere # can ignore if only one
HTTP              TCP      80                   anywhere #for Rstudio port

## [4]
Create key pair: "EC2UbuntuR"

## Launch! Wait till 2/2 checked

## [5]
## Download Putty / Puttygen
http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/putty.html #Putty
http://www.chiark.greenend.org.uk/~sgtatham/putty/download.html #Puttygen
Puttygen
# open, choose "SSH-2 RSA" and "2048"
# load file - select the current saved key pair (all file *.*), open
# save private key - rename "EC2UbuntuR" - yes
Putty
## Host Name - "ubuntu@public DNS"
## Go "SSH" then "Auth" - browse to add the new key pair generated
## Go back "Session" - open! - SSH launched for that instance!


## - Keep in note!
Public DNS #change every time
Hadoop1: ec2-54-85-248-63.compute-1.amazonaws.com
Hadoop2: ec2-52-23-167-94.compute-1.amazonaws.com
Hadoop3: ec2-54-208-72-51.compute-1.amazonaws.com

Private IP
172.31.57.179 Hadoop1
172.31.57.180 Hadoop2
172.31.57.181 Hadoop3

RSA Key
Hadoop1: ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC5wkoxoa7iAbDo5oGjrlbP8+9vwlqz74EE67pCa0qXBWBqc4MdJgtXzEvlBfSqpvwiHoMKhipkLqtUVxSGPzMf8NjqjofKYiIjeWpSxvVtAYL69d22cpM66b5zK2dCw0ZhdSOThr+pUG0MOMgpbfQ+UvN+76MJ5wwl0q7RZAk+TDJGpcawh9f9a45SxrwLJ0MWeOtV952DNoNxJLhfQMD9NeiTCDVC4s5F1S7lUjtg/V/9MzV6yMo5riAFsgFlvKtpaYfhoxTNPvu/y69NbcT1fuWEuURyDANoUNwnvfmXzAovWOmjGJC13RzOIL2uFgNMIEwYjqsA174lKugHpyfn ubuntu@ip-172-31-57-179
Hadoop2: ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCT4xDDjJd3oqY0BOJf+9fs8wgmfxiPnQABP5B4VItyjU6g1qxjz1tKwN2gi/whJwPixZlseLFrNH3FIZg93krYos9h6d4rShSyDzrIPKzo5JOPrFNE2OYNyPYDb+qiudugbFCXbW9gd2j3SE+BT/vVeBFnutJEof1tUXNmDqSyY2/D8RXJKljeMt96PdjJqL5Wwuj6am9v+ZoC3LVvUcX/vcd+AwQ4M5NPI3q8YIYvZsHaNldSCQIHJud1hixAdyr9TK3yWceyxv9jywtA7nPasuFF8H/e8xGTkSUTxl1oI/fW2CksgkL4A6nW5NuAMCOhYPX8OiZijH+8PLe8GmVD ubuntu@ip-172-31-57-180
Hadoop3: ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC27hloummu71NaqNXBjWmsMEQbtW4p/xAG9eIleCa7xuUcPZM6mWmTYaMdAkjP+m6hjrFlw6CRjbG/NPor1iNK0NE27j61uB7UEedr3blQdG41EuXCr90NK2eIW88jI7PcI6M8MRioPMRBPKCe8antIFeilOyYfaRwJxrGhYwikZAVIgjqbXzZ70AAr7NiS1l6qOCeX+IAQ3Ocwgi+J0pDYTIAyccx1GhKtO0Qh1PQL73Qd6o8UWPHyE8xI78mjdLP/d2/gi5EsaMZ1xqoZi/56qEvjLhsdBoBIkCnCrTOQ5LZqxCwPjvvSbguaADHmDYbwk7AtR39HpXr+LsOr4H7 ubuntu@ip-172-31-57-181
## - end - ##







## <<<< Single local machine - R-studio launch >>>> ## >>>>> Method 1
# Regardless Putty, directly copy "Public DNS" of an instance to launch Rstudio Server
login: rstudio
password: rstudio
# need to reset
Happy analyzing!





## <<<< Hadoop cluster >>>> ## >>>>>> Method 2
# [1] (All instances)
Login with Putty!

# [2]
# Generate your Server Key (All instances)
ssh-keygen -t rsa 
# hit enter 3 times, you will created ssh-rsa key in the file - ".ssh/id_rsa.pub"
# Access file to get the key
cat .ssh/id_rsa.pub # copy paste key into container above!

# [3]
# Configure "authorized_keys" file on each of the three (All instances)
vi ~/.ssh/authorized_keys
# then, "i" in insert mode, "ESC > o" to the last line, copy paste all rsa keys with no space
# save by "ESC", ":wq"

# [4]
# Configure "hosts" file on each of the three instances (All instances)
sudo vi /etc/hosts
# then, "i" in insert mode, "ESC > o" to the last line, copy paste all IP name with no space
# save by "ESC", ":wq"

# [5]
# By now, you have told each instance where the other instances can be reached and provided
# them security keys in order to access each other. Here is the moment of truth: we will issue
# ping commands from each instance to the other two
ping hadoop1
ping hadoop2
ping hadoop3
# exit -> "Ctrl + c"

# [6]
# Now that the servers are configured and speaking with one another, we can install Java, and
# eventually Hadoop. Each instance will need Java, so this will need to be repeated on each of the
# three nodes
sudo add-apt-repository ppa:webupd8team/java # choose enter
sudo apt-get update # update to latest version
sudo apt-get install oracle-java7-installer # (choose yes,  select “ok” once, navigate from "no" to "yes" once, user cursor to select second “ok” and enter again)
sudo apt-get install oracle-java7-set-default 
exit # change setting need to exit
# logout and log back, test whether it installed
echo $JAVA_HOME
# you should see - "/usr/lib/jvm/java-7-oracle"

# [7]
# Installing and Configuring Hadoop (Finally) (Only instance hadoop1)
wget http://www.webhostingreviewjam.com/mirror/apache/hadoop/common/hadoop-2.6.0/hadoop-2.6.0.tar.gz #(change this link if update)
tar -xzvf hadoop-2.6.0.tar.gz #(replace 2.6.0 with lastest version if update)
cd hadoop-2.6.0/etc/hadoop # change 2.6.0 if necessary

# [8]
# This will configure the Java path so that Hadoop knows where you installed Java
vi hadoop-env.sh # Edit the "hadoop-env.sh" file by typing
# Look for the line # The java implementation to use, "i" to edit
export JAVA_HOME=/usr/lib/jvm/java-7-oracle # Set the line below
# Save the file with :wq

# [9]
# Configure the Core Site File
vi core-site.xml
# Insert the following between <configuration> and </configuration>
  <property>
    <name>fs.default.name</name>
        <value>hdfs://hadoop1:9000</value>
  </property>

  <property>
    <name>hadoop.tmp.dir</name>
    <value>/home/ubuntu/hadoop/tmp</value>
  </property>
# :wq
# Create a new tmp folder for hadoop
cd ~
mkdir hadoop
cd hadoop
mkdir tmp
# Configure the Redundancy - edit "hdfs-site.xml" file
vi ~/hadoop-2.6.0/etc/hadoop/hdfs-site.xml
# Insert the following between <configuration> and </configuration>
<property>
    <name>dfs.replication</name>
    <value>2</value> 
</property>
# :wq
# Edit the "slaves" file
vi ~/hadoop-2.6.0/etc/hadoop/slaves
# Delete "localhost" and replace it with:
hadoop2
hadoop3
# :wq
# copy these configurations to the two slave nodes
scp -r ~/hadoop-2.6.0 hadoop2:/home/ubuntu # yes
scp -r ~/hadoop-2.6.0 hadoop3:/home/ubuntu # yes
# close the firewall for each instance (All instances)
sudo ufw disable

# [10]
# Formatting the master node (only hadoop1)
~/hadoop-2.6.0/bin/hadoop namenode -format
# This should produce the output:
# Storage directory /home/ubuntu/hadoop/tmp/dfs/name has been successfully
# formatted.
# find the folder
cd ~/hadoop/tmp/
# Starting Hadoop!
~/hadoop-2.6.0/sbin/start-all.sh
# check runing with:
jps
# should see: hadoop1
3459 ResourceManager
3114 NameNode
3716 Jps
3325 SecondaryNameNode
# run on hadoop2, hadoop3
jps
# should see:
3023 DataNode
3242 Jps
3158 NodeManager
# finally stop hadoop:
~/hadoop-2.6.0/sbin/stop-all.sh

# [Optional] [Not working <<<>>> exit]
# Modifying the path variable (All instances)
sudo vi /etc/environment
#change to:
PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/u sr/local/games:/home/ubuntu/hadoop-2.6.0/bin:/home/ubuntu/hadoop-2.6.0/sbin"
#;wq
source /etc/environment
# try short cut
start-all.sh
  

# [11] configurate Rhadoop (All instances)
sudo apt-get install liblzma-dev # To install the dependencies for Java, run the following · code in the shell
sudo vi ~/.bashrc # Adding environment variables
# Add those lines:
export LD_LIBRARY_PATH=$JAVA_HOME/lib/amd64:$JAVA_HOME/jre/lib/amd64/server
export HADOOP_HOME=/home/ubuntu/hadoop-2.6.0
export HADOOP_CMD=$HADOOP_HOME/bin/hadoop
export HADOOP_STREAMING=$HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-2.6.0.jar
# Then:
source ~/.bashrc
# Configure the path of Java in R
sudo R CMD javareconf JAVA_HOME=$JAVA_HOME

# [12] Download R packages (All instances)
R
install.packages(c('rJava','codetools','Rcpp','reshape2','iterators','itertools','digest','RJSONIO','functional','bitops','caTools'))
library(rJava) # Most important package 'rJave'

# install rhdfs (Only Hadoop1)
wget https://raw.githubusercontent.com/RevolutionAnalytics/rhdfs/master/build/rhdfs_1.0.8.tar.gz
R CMD INSTALL rhdfs_1.0.8.tar.gz
R # enter R
library(rhdfs)
hdfs.init() 
hdfs.ls('/') # equivalent to the command hadoop fs -ls /
# ** Note that each time before you want to do some operations on HDFS in R, it's necessary to
# run hdfs.init()

# install rmr2 (...)
wget https://raw.githubusercontent.com/RevolutionAnalytics/rmr2/master/build/rmr2_3.2.0.tar.gz
R CMD INSTALL rmr2_3.2.0.tar.gz
R # enter R
library(rhdfs)
hdfs.init()
library(rmr2)
from.dfs(to.dfs(1:10)) # test see whether can read and write into hadoop

# install plyrmr (...)
R # enter R
install.packages(c('Hmisc','dplyr','R.methodsS3')) # dependencies
q() # quit R
# Install plyrmr(Exit R, run in the shell):
wget https://raw.githubusercontent.com/RevolutionAnalytics/plyrmr/master/build/plyrmr_0.4.0.tar.gz
R CMD INSTALL plyrmr_0.4.0.tar.gz
# check
R # enter R
library(plyrmr)
require(plyrmr)
dat = bind.cols(mtcars, carb.per.cyl = carb/cyl)
head(dat) # see data processed

# Done! rhdfs, rmr2, plyrmr installed!





## <<<< Build API connection + Connect to RDS + automate update >>>> ## >>>>>> Method 3
# Request instance with AMI on EC2
edit security group to allow "mysql"
# Login shell - check (Python, R, mysql server)

# Create file sturcture - dir>mysql, dir>r, dir>python, dir>output.file, dir>output.log, dir>auth
$ls -l; $mkdir dir; $> file; $cd /path/
# Create r / python script for API
$sudo Rscript xxxx.r > ~/path/xxx.txt # execute r file and save "process" to a log file
$python xxxxx.py # execute python file
# execute a shell command until it success (until loop)
n=0
until [ $n -ge 20 ] # try 20 times
do
   sudo Rscript API_script_loop.r > ~/data/data.log/log.txt && break 
   n=$[$n+1]
   sleep 5 # each time gap 5 seconds
done

# Set up mysql database
# Create RDS instances (Through UI - read document) - records:
mysql-copilot-dash-instance1.cmzj9dljhdnd.us-east-1.rds.amazonaws.com # hostname - end point
database; table # the names you created in the process
username; password # the ones you created in the process
port:3306
use the same security group EC2 instance used; add "all ICMP" - all IP
# security group confi
HTTP            TCP     80           0.0.0.0/0
ALL TCP         TCP     0 - 65535    0.0.0.0/0
SSH             TCP     22           0.0.0.0/0
MYSQL/Aurora    TCP     3306         0.0.0.0/0
ALL ICMP        ALL     N/A          0.0.0.0/0 # can all IPs, or specific IPs
# Launch! Wait a while let it fully set-up
# Back to shh - download mysql server and set up
$ sudo apt-get install mysql-server
# install
$ sudo netstat -tap | grep mysql
# check running?
$ sudo service mysql restart
# if not, restart it
$ vi /etc/mysql/my.cnf
# configuring: log file, port, etc, bind-address(IP)
$ sudo service mysql restart
# restart after conf
$ sudo dpkg-reconfigure mysql-server-5.5
# change mysql root password
$ mysql -h localhost -V
# check the mysql version on the local
# Login with specified credentials
$ mysql -h mysql-copilot-dash-instance1.cmzj9dljhdnd.us-east-1.rds.amazonaws.com -P 3306 -u username -p database # host, port, username, database
password # Typr in password
# >>> mysql

# Alternatively create a login auth file
$> mysql_login.txt
$vi mysql_login.txt
# add
[client]
host="hostname2"
user="username2"
password="password2"
database="database2"
# Save :wq
# login
$ mysql --defaults-file="~/path/mysql_login.txt" --local-infile=1 # enable upload local file
# when lged in mysql command
SHOW DATABASES;
SHOW TABLES;

# Create scahma
CREATE TABLE copilot_dash_log
(
Seat_ID int,
day date,
advertiser_id int,
advertiser_name varchar(255),
insertion_order_id int,
insertion_order_name varchar(255),
line_item_id int,
line_item_name varchar(255),
campaign_id int,
campaign_name varchar(255),
imps int,
clicks int,
total_convs int,
convs_rate float(6,6),
revenue float(6,6),
cost float(6,6),
profit float(6,6),
cpm float(6,6)
)

# Load local file into the created table
LOAD DATA LOCAL INFILE '~/data/data.output/1837.processed.csv' 
INTO TABLE copilot_dash_log
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS; # ignore column names

# Alternatively do everything in shell command
# Create a sql script file
$> sql_script.sql
$vi sql_script.sql # loading script above; :wq
# auto-login -> execute the script in file
$mysql --defaults-file="~/path/mysql_login.txt" --local-infile=1 < ~/mysql_server_login/sql_script.sql

# Last - combine all function into one file.sh -> execute it or execute it when instance start
$sudo vi /etc/bashrc
# Add into the file
. ~/.bashrc
#:wq
# Create file.sh / edit premission
$> file.sh
$chmod 755 ~/file.sh
# add scripts need to be execute
$vi ~/file.sh
$bash ~/file.sh # one command for all functionalities
# add this file to root to execute when log in
$vi ~/.bashrc
































