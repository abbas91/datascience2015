################################################
#                                              #
#                                              #
#                    HIVE                      #
#                                              #
#                                              # 
################################################ 



# [1] Getting Start with Hive CLI
$cd $HIVE_HOME
$bin/hive # Open!
"or"
$hive # Open! (Add "$HIVE_HOME/bin" to enviroment's PATH)

# Start hive promot 
hive> 
hive> SELECT * FROM x;
"""
OK
Time taken: 3.543 seconds
"""
hive> exit;

# Configure top level directory storage for tables (Separate for each user)
hive> set hive.metastore.warehouse.dir=/user/myname/hive/warehouse;
"or"
$vi $HOME/.hiverc # add the above "set ..." to it to let it run very start

# Command Line Interface (CLI)
$hive --help --service cli
$hive -d, --define <key=value>     Variable substitution apply to hive
      -e "Select ..." --           SQL from CLI
      -f file.hql     --           SQL from file
      -H, --help                   Print help information
      -h hostname     --           connecting to hive server on remote host
      --hiveconf <property=value>  Use value for given property
      --hivevar  <key=value>       variable substitution apply to hive
      -i file         --           Initialization SQL file
      -p <port>       --           connecting to hive server on port number
      -S, --silent                 slient mode in interactive shell (No "ok", no "time")
      -v, --verbose                verbose mode (echo executed SQL to console)                

# variable & property <1>
"""
Namespace           Access           Description
hivevar             R/W              user defined custom variables
hiveconf            R/W              Hive-specific configuration properties
system              R/W              Java defined configuration properties
env                 R only           enviroment variable defined by shell enviroment. ex.bash
"""
$hive
# set var
hive> set system:var=myvar1;
hive> set hivevar:var=myvar2;
hive> set hiveconf:var=myvar3;
$var=mayvar3 # env, no W
# display var
hive> set system:var;
"system:var=myvar"
# use var in Hive
hive> SELECT * From mytable where var = ${system:var};
hive> SELECT * From mytable where var = ${env:var};

# Execute hive command in shell
$hive -e "SELECT ..."; # Return to console
$hive -S -e "SELECT ..." > /xxx/xxx/myquery # -S keep silent
$hive -S -e "set" | grep xxxx # handle property name you can't quit remember

# execute hive from a file
$hive -f /path/xxx/myquery.hql # in shell
hive> source /path/xxx/myquery.hql # in hive

# the .hiverc file
$hive -i hiverc # create the start file
$HOME/.hiverc

# autocomplete
type "tab" key at prompt -> give options for ex. SEL -> SELECT
# shell extension
hive> ! /bin/echo "whats up." # add "!" to execute shell in hive
"whats up."
*** no | pipe, no regex ***
# hadoop dfs from hive
hive> dfs -ls / ; <=> $hadoop dfs -ls












