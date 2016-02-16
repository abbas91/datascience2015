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
A map is a chararray to data elemnt mapping, 
where that element can be any Pig type, including complex type.
It is called a Key and used as an index to find the element, referred to as the value.
"""
tuple / ('string', 405)
"""
A tuple is a fixed-length, ordered collection of Pig data elements. It has been divided into fields,
with each filed containing one data element (They don't have to be the same type). It has equal meaning to
'row' in SQL, each field being a SQL column.
"""
bag / {('string1', 405), ('string2', 404), ('string3', 403)}
"""
A bag is an unordered collection of tuples. So, it is not possible to reference tuples in a bag by
position. 
"""

'Schemas' /* define data types */
table = load 'file_name' as (var1:chararray, var2:float, var3:int); /*define schemas using 'load' function */
table = load 'file_name' as (var1, var2, var3); /* Not define - default to 'bytearray' */
"example"
int > as (a:int)
long > as (a:long)
float > as (a:float)
double > as (a:double)
chararray > as (a:chararray)
bytearray > as (a:bytearray)
map > as (a:map[], 
	      b:map[int])
tuple > as (a:tuple(), 
	        b:tuple(x:int, y:int))
bag > (a:bag{}, 
	   b:bag{t:(x:int, y:int)})
"""
If not define -> all 'btyearray' data type
Pig will guess and redefine in the later data processing
If schema data merge with data without scehma -> all lost scehma (Contagious)
"""
var1 * 1000 /* arithmatic operators */ -> as floating points casting to doubles
     > /* both in strings and numbers */ -> NO way to guess

/* Casting */
"Undefined"
table2 = foreach table1 generate map1#'var1' - map1#'var2'
"Defined - casting" /* define type by () casting */
table2 = foreach table1 generate (int)map1#'var1' - (int)map1#'var2'
***"Rules - Page.31 table top"
"""
During Pig data type guessing - always widen types when automatic casting
int vs long -> long
int, long, vs float -> float
int, long, float vs double -> double
Null vs everythong -> Null (Viral)
"""




/* [4] ------------------ Introduction to Pig Latin */
-- Case Sensitivities --
load = LOAD ; A <> a 
""" Command, keywords not sensitive ; 
Relationship, table sensitive to case """

-- Comments --
""" -- xxxx -- (Single line comments) 
    /* xxxxxx */ (Multiple lines comments) """

-- Input & Output --
load
table = load '/data/example/file' using PigStorage(); -- (Default) can specify actual path in HDFS of that file; Can also do the full path like
                                                      -- 'hdfs://nn.acme.com/data/example/file' to read the file from the HDFS NameNode:nn.acme.com
                                                      -- If no path specified, run in Home Dir of your HDFS
                                                      -- Default using 'PigStorage' (load tab-delimited file from HDFS)
                                  using PigStorage(,); -- Change to ',' separater
table = load '/data/example/file' using HBaseStorage(); -- If data in storage systems, ex. Hbase
table = load '/data/example/file' as (var1, var2, var3, var4); -- Specify Schema to the data loaded (above)
table = load '/data/example/' -- Will load ALL files in that directory or sub-directory
table = load '/data/example/?' -- Any single character
             '/data/example/*' -- Zero or more characters
             '/data/example/[abc]' -- match single character from the list
             '/data/example/[a-z]' -- match single character from the range
             '/data/example/[^abc]' -- NOT match single character from the list 
             '/data/example/[^a-z]' -- NOT match single character from the range
             '/data/example/\c' -- escaple the letter meaning
             '/data/example/{ad,cd}' -- match a string from the list of strings     

-- Store --
Store
store table into '/data/example/file' using PigStorage(); -- Write the result out to HDFS or Storage system / can specify actual path like in 'load'
                                      using PigStorage(,);
                                      using HBaseStorage();

-- Dump -- 
dump -- Display the output to screen for ac hoc debugging
dump table; -- complex type: []map, ()tuples, {}bags, each fields separated by ','


-- Relational Operations --
"""They help you transform your data by sorting, grouping, joining, projecting, and filtering"""
""" More advanced operators are in Page 57 """
-- foreach --
""" it takes a set of expressions and applies them to every record in the data pipeline """
foreach
table = load '/data/example/file' as (var1, var2, var3, var4);
table2 = foreach table generate var1, var3; -- only extract 'var1' and 'var3' ; 'Null' is viral ex. 1 + null = null
table2 = foreach table generate var2 - var4; -- + - * / % (By name)
                                $2 - $4; -- (By position)
                                *; -- All fields
                                ..var3; -- var1, var2, var3
                                var2..var3; -- var2, var3
                                var2..; -- var2, var3, var4
-- Condition operator --
2 == 2 ? 1 : 4 -- if 2 equals 2, then return 1; if not, returns 4
null == 2 ? 1 : 4 -- returns 'Null' viral
2 == 2 ? 1 : 'xxx' -- type error; both need to be same type




























