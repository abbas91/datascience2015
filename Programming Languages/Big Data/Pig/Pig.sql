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
B = foreach A generate (a == 'yes' ? 'ok' : 'No') as b;

-- Projection operators -- (Extract value from complex type)
A = load 'file' as (bat:map[]); 
B = foreach A generate bat#'string'; -- map projection operator '#'

A = load 'file' as (t:tuple(x:int, y:int));
B = foreach A generate t.x, t.$2; -- tuple projection operator '.'

A = load 'file' as (b:bag{t:tuple(x:int, y:int)});
B = foreach A generate b.x; 
                       b.(x,y); -- Different from above two, bag need to create a new bag to extract
                                -- + - * / not work on bag objects

-- UDF (User Defined Function) -- ***
"""UDF can be invoke in 'foreach' called 'evaluation function'
   takes one record at a time and produce one output """

A = load 'file' as (name:chararry, age:int, gpa:float);
B = foreach A generate myudf.FUN(name); -- define a fun for further comstomization

"UDF can be define in Java, Python and Javascript" -- Apendix A


-- name fields --
foreach """will infer the names of the new tuple by the old data input, will try to keep the same
           if no operation applied. If any operation applied, there will be no name if not assigned"""
describe A; -- tell data type
"A: {name: chararray,chararray}"


-- Filter -- 
""" evaluate the data, only those qualify will be passed download. It generate a booleaan value T/F """
A = load 'file' as (name:chararry, age:int, gpa:float);
B = filter A by age == scalar; maps; tuple 
                    != scalar; maps; tuple
                    >  scalar;
                    <  scalar;
                    >= scalar;
                    <= scalar;
             by name matches 'regex';
                not name matches 'regex';
--combine
a and b or not c => (a and b) or (not c)-- highest -> 'not', 'and', 'or'   
1st and 2nd : "if ist failed, 2nd not eval"
1st or 2nd : "if ist pass, 2nd not eval"
-- 'A == NULL' => NULL; NULL will be ignored by regex as well
a is null ; a is not null -- to test null
-- UDF in filter
B = filter my_udf.fun(); -- will generate boolean value to filter data too



-- Group --
""" No like group by in SQL, not direct link to aggregation, it only collect all records from the same value
    and then group them into a bag{} 'A key - named groups and a bag of collected records'. 
    Then you can pass this into aggregation functions if you want."""
A = load 'file' as (name:chararry, var:charrary, age:int, gpa:float);
B = group A by name; -- group name will be 'A'
cnt = foreach B generate group, COUNT(A); -- passs to aggregated function
store B into 'file_group' -- Save it for later

B = group A by name;
describe B;
"B: {group: bytearray, 
 A: {name:chararry, var:charrary, age:int, gpa:float}}" -- Single, key field named 'group'
B = group A by (name, var);
describe B;
"B: {group: (name:chararry, var:charrary), 
 A: {name:chararry, var:charrary, age:int, gpa:float}}" -- multiple, key field named 'group'
B = group A all;
describe B;
"B: {all: bytearray, 
 A: {name:chararry, var:charrary, age:int, gpa:float}}"
--***All NULL will be in the same 'NULL' group when grouping--



-- Order By --
""" use 'order' to sort your data by certain vars """ -- only scalar, complex type sorting produce error / Null is smallest when sorting
A = load 'file' as (name:chararry, var:charrary, age:int, gpa:float);
B = order A by age;
            by age, gpa;
            by age desc, gpa; -- define sort 
            by age, gpa asc; -- define sort



-- Distinct --
""" same like distinct in SQL, but works only on all data """
A = load 'file' as (name:chararry, var:charrary, age:int, gpa:float);
B = foreach generate name, var;
dis_B = distinct B; -- reduce duplicated records



-- Join --
""" Join data file by primary keys (default - inner join) not match, null droped """
A = load 'file' as (name:chararry, var:charrary, age:int, gpa:float);
B = load 'file' as (name:chararry, var:charrary, hieght:int, weight:float);
J = join A by name, B by name; -- single key (inner join)
    join A by (name, var), B by (name, var); -- multiple keys (inner join)
    join A by name left outer, B by name; -- left join
    join A by name right outer, B by name; -- right join
    join A by name full outer, B by name; -- full join
    join A by name, B by name, C by name; -- Multiple join (Only for inner join)

A1 = load 'file' as (name:chararry, var:charrary, age:int, gpa:float);
A2 = load 'file' as (name:chararry, var:charrary, age:int, gpa:float);
B = join A1 by name, A2 by name; -- self join is supported (Need to load data twice A1, A2)

-- join will preserve the name from data source --
describe J;
"jnd: {A::name: chararrary, A::var: chararrary, A::....
       B::name: chararrary, B::var: chararrary, B::....} " -- when use vars, if same name needs to define like 'A::name' or 'B::var'
-- Outter join (left right full) 



-- Limit -- (Still read all)
A = load 'file';
A_limilt = limit A 10; -- Only return 10 rows (Not same each time)
A_limit = order limit A 10; -- Only return 10 rows (Same each time)


-- Sample --
A = load 'file';
some = sample A 0.2; -- sample 20% of the data



-- Parallel --
# Parallel your reduce jobs 
group, order, distinct, join, limit, cogroup, cross -- reduce operator
A = group B by var1 parallel 10; -- set individually for each
A_sort = order A by var2 desc parallel 2;
# Set as script wise
set default parallel 10;
'...all operation use 10 reducers'





-- UDF - User Defined Function
" let user define their function, mostly in java, then in Python"
" Pig uses Jython to execute Python / No Python 3 features"
" Also, there are built-in UDFs in Pig"
" Piggybank is a collection of UFS release along with Pig, need to register"
-- Register UDF ex. want to use 'reverse UDF' in Piggybank
register 'your/path/file/piggybank.jar'; -- You need you specify the path of the jar in your local system where you have extracted the pig.
define Reverse org.apache.pig.piggybank.evaluation.string.Reverse(); -- define the path to the actual fun
define Reverse org.apache.pig.piggybank.evaluation.string.Reverse('F'); -- if argument needs to be, can add it
A = load 'file' as (a, b, c);
B = foreach A generate Reverse(b); 
-- Register Python UDF
-- the Python script must in your current directory
register 'test.py' using jython as bballudfs;
A = load 'file' as (a, b, c);
B = foreach A generate bballudfs.test(b); -- careful of Python dependents in the nodes

-- Calling Static Java Functions --
p54 ???





-- Advanced Pig Latin --

flatten -- sometimes you have a tuple or a bag and you want to remove that level of nesting

A = load 'file' as (name:bag{t:(x:int, y:int)});
B = foreach A generate flatten(name) as name;

"""
name1, x1, y1
name1, x2, y2
"""
# n * m rows

foreach -- nested foreach

A = load 'file' as (x:chararray, y:int);
B = group A by x;
C = foreach B {
    f = A.y; # create inner operations to manipulate vars more
    g = distinct f;
    generate group, COUNT(g); # last line always be "generate" Or use UDF() here
}


join -- multiple implementation

# join small to large data
A = load 'file_big' as (x:int, c:int);
B = load 'file_small' as (x:int, b:int);
C = join A by x, B by x using 'replicated'; -- fragment replicated join: save small data into each instance memory
                                            -- only the most left data gets replicated

# join skewed data (Some nodes has more data to process)
A = load 'file' as (x:chararray);
B = load 'file2' as (x:chararray, c:int);
C = join A by x, B by x using 'skewed'; -- most left has most guanularity

# join sorted data (first sort all inputs by join keys, then merge together) 
A = load 'file_1' as (x:int, c:int);
B = load 'file_2' as (x:int, b:int);
C = join A by x, B by x using 'merge';


cogroup -- instead of collecting records of one input based on a key,
        -- it collects records of n inputs based on a key
A = load 'file1' as (id:chararray, var1:int);
B = load 'file2' as (id:chararray, var2:int);
C = cogroup A by id, B by id;
describe C;
" C: {group: int, A: {id:chararray, var1:int}, 
                  B: {id:chararray, var2:int}}"


union -- Put two data sources by concatenating them (No match key needed)

A = load 'file1' as (id:chararray, var:int);
B = load 'file2' as (id:chararray, var:int);
C = union A, B;
describe C;
" C: {id:chararray, var:int} "

# if scehma totally different
A = load 'file1' as (id:chararray, var:int, x:int);
B = load 'file2' as (id:chararray, var:int, y:float, z:int);
C = union onschema A, B;
describe C;
" C: {id:chararray, var:int, x:int, y:float, z:int} "



cross -- take every record in A and combine with every record in B, A:n, B:m -> n X m output

A = load 'file1' as (id:chararray, var:int);
B = load 'file2' as (id:chararray, var:int);
C = cross A, B;




-- Integrate other script to pig --

stream -- Pass the result pig object to another script





































