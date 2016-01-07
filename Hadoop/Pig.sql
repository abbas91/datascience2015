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











