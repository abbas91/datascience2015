/********************************************/
/*                                          */
/*                                          */
/*                   PIG                    */
/*                                          */
/*                                          */
/********************************************/




/* [1] ------------------- Getting Start with Pig on grunt */
$pig "or" $pig -x mapreduce /* mapreduce mode */
$pig -x local /* local mode */
grunt> /* start pig interactive */
grunt> quit "or" Ctrl-D  /* exit */

/* Command-Line options */
$pig -e pig.command /* execute a single command */ ***???
     -h             /* list all avaiable command-line options */
     -h <property>  /* list the property pig will use if it is set by the user already */
     -P             /* specify a property pig will read */
     -version       /* print current version */
     -D <key=value> /* pass property to pig on command line & put it in front of every other command arguments*/ 
/* Pig result return code */
"""
Value    Meaning          
0        Success       
1        Retriable failure
2        Failure
3        Partial failure
4        Illegal arguments passed to pig
5        IOException thrown
6        PigException thrown
7        ParseException thrown 
8        Throwable thrown
"""




/* [2] ------------------- Grunt Pig interactive shell */
$bin/pig /* start pig! */
grunt> 
/* HDFS Commands in grunt */
"All HDFS CLI avaiable in grunt, acted as HDFS shell -> Pig version >=0.5"
grunt> fs -ls /* example */
/* Unix shell commands in grunt */
"All but no pipe, redirect. need to use absolute path"
grunt> sh ls /* example */
/* Controling Pig from grunt */
kill <jobid> /* kill Mapreduce job associated with jobid */
exec [[-param param_name = param_value]] [[-param_file filename]] script /* Execute the pig script, not imported into grunt */
run [[-param param_name = param_value]] [[-param_file filename]] script /* Execute the pig script in current grunt shell */





/* [3] ------------------ Pig data model / data type */
/* data type */
"scalar type" /* java.lang class */
int / 42 / java.lang.Integer / 4 byte
long / 500000000L / java.lang.long / 8 byte
float / 4.14f 'or' 6.022e23f /*exponent format*/ /java.lang.Float / 4 byte
double / 2.71828 'or' 6.626e-34 /*exponent format*/ / java.lang.Double / 8 byte
chararray / 'string' 'or' \t\u / java.lang.String / 2 byte per character
bytearray / any&default / blob 'or' array of bytes


'complex type' /* group those scalar items or other complex items */
map / ['var1'#'string', 'var2'#405]
"""
"""
tuple / ('string', 405)
"""
"""
bag / {('string1', 405), ('string2', 404), ('string3', 403)}
"""
"""

'Schemas' /* define data types */
table = load 'file_name' as (var1:chararray, var2:float, var3:int); /*define schemas using 'load' function */
table = load 'file_name' as (var1, var2, var3); /* Not define - default to 'bytearray' */
"example"
int > 











