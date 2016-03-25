# Python - data analysis - Basic






# ------------------------------ Python Basic Class 1
# package:
import Numpy as Np # Numpy, short for Numerical Python, is the foundational package for scientific computing in Python

import SciPy as Sp # SciPy is a collection of packages addressing a number of different standard problem domains in scientific computing.

import Pandas as Pd # Pandas provides rich data structures and functions designed to make working with structured data fast, easy, and expressive. The DataFrame object in pandas is just like the data.frame object in R

import Matplotlib as Mp # Matplotlib is the most popular Python library for producing plots and other 2D data visualizations

# -- Using R in Python (Hybrid Programming with R and Python)
# Download R in local -- 
# Run following in command to install rpy2
sudo easy_install rpy2 # mac #
sudo apt-get install pip # ubuntu #
sudo pip install rpy2 # linux #
# Execute command in python
%load_ext rpy2.ipython
%%R # start R session
x = rnorm(10)
y = 1:10
summary(lm(y~x))



# -- Keywords (Command)
import keyword as kw
and, or, ...



# -- Magic functions
"""line-oriented: Line magics are prefixed with 
the % character and work much like OS command-line calls: 
they get as an argument the rest of the line, where arguments are passed without parentheses or quotes."""

"""cell-oriented: Cell magics are prefixed with a double %%, 
and they are functions that get as an argument not only the rest of the line, 
but also the lines below it in a separate argument."""

%ls # magic function:list all the files and folders
%%! # a cell of shell codes
%lsmagic # The %lsmagic magic is used to list all available magics.


# print
print "hello, %s!" %'world'
print "hello, %d!" %2015
print "hello, %f!" %2015


# basic objects in python
2,5,100 # int, used to represent integers
2.75, 3.14 # float, used to represent real numbers
True, False # bool, used to represent Boolean values
print "5 / 2.0 = %s" %(5 / 2.0)


# built-in 'type' returns type
print type(2.0) # <type 'float'>
type(int(2.0)) # change type


# use Python as calculator
+ - * / % **
print "5 ** 2 = %d" %(5 ** 2)


# Comparison operators
print 1 > 1
print 1 < 1
print 1 >= 1
print 1 <= 1
print 1 == 1
print 1 != 1

# Operators on bools
print True and True
print True or  False
print not True



# Strings
type('1324')
str(1234) # convert int to string
str(1234.0) # convert float to string
str(True) # convert bool to string


# Operators on strings
print 3 * 'a'          # repeat 'a' 3 times
print 'a' + 'b' + 'c'  # join 'a', 'b' and 'c'
print 'a' + str(123)   # convert and join
print len('abcd')      # length 



# Indexing and Slicing
print '123'[0] # the first character of string '123'
print '123'[1] # the second character of string '123'
print '123'[2] # the third character of string '123'
print '123'[0:2] # the first two characters of string '123'
print '123'[1:3] # the second and third characters of string '123'


# Variables
radius1 = 1
radius2 = 3
print "The area of of radius1 is %f" %(pi * radius1 ** 2)
print "The area of of radius2 is %f" %(pi * radius2 ** 2)



# Functions
def polynomial(x, a, b=0, c=0, d=0):
    result = a + b * x**1 + c * x**2 + d * x**3
    return result



# Conditionals
# - type1
if <condition>:
    do()
else:
    do_another()

# - type2
if <condition>:
    do()
elif <condition>:
    do_other()
else:
    do_another()

# - example:
# - nested condition
if x % 2 == 0: 
    if x % 3 == 0 :
        print "x can be divided by 2 and 3"
    else:
        print "x can be divided by 2, but can not by 3."
else:
    if x % 3 == 0 :
        print "x can be divided by 3, but can not by 2."
    else:
        print "x can not be divided by 2 or 3."




# - simplified 
div2 = x % 2 == 0
div3 = x % 3 == 0
if div2 and div3: 
    print "x can be divided by 2 and 3"
elif div2 and not div3: 
    print "x can be divided by 2, but can not by 3."
elif not div2 and div3: 
    print "x can be divided by 3, but can not by 2."
else:
    print "x can not be divided by 2 or 3."




# Recursion Function
def fib(n):
    """Calculates nth Fibonacci number"""
    if n == 0:
        return 1
    elif n == 1:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)




# Loop
for, while 

# -- while
while <condition>:
    do()
    update_condition()

for x in <sequence>:
    do()



# -- example:
# while
x = 1
while x <= 10:
    if x % 2 == 0 and x % 3 != 0:
        print x
    x += 1

# for
for x in range(11):
    if x % 2 == 0 and x % 3 != 0:
        print x





# For list comprehension
# comprehension
result = [i + 1 for i in fibonacci_numbers if i % 2 == 0]
# normal for loop
result = []
for i in fibonacci_numbers:
    if i % 2 == 0:
        result.append(i + 1)
print result







# binarySearch Fun
def binarySearchSquareRoot(x, eps = 1e-8):
    start = 0
    end = x
    mid = (start + end) / 2.0
    while abs(mid ** 2 - x) >= eps:
        if mid**2 > x:
            end = mid
        else:
            start = mid
        mid = (start + end) / 2.0
    return mid





# Exceptions
"""Another flow control tool is the exception. 
They are used to indicate an unusual condition that 
needs to be handled higher up on the stack."""

"""Exceptions raised by statements in body of try are handled by 
the except statement and execution continues 
with the body of the except statement."""

try:
    1 / 0
except:
    raise  Exception("Can not divide 0!") # return error message / keep running


try:
    1 / 0
except:
    print 'Can not divide 0!'

print 'hi, I keep running!' # print some message / keep running




# Auto-correct leverage 'exception'
"""
SyntaxError: can’t parse program
NameError: name not found
TypeError: operand doesn’t have correct type
ZeroDivisionError: integer division or modulo by zero.
"""

def divide(x, y):
    try:
        result =  x / y
    except ZeroDivisionError:
        result = None
    except TypeError:
        result = divide(float(x), float(y))
    return result


print divide(3, 0)
print divide(3, 1)
print divide(3, '1')


# loop in exceptions
"""
try:
else: executed when execution of associated try body completes with no exceptions.
finally: always run.
"""
try:
    do()
except:
    do_something()
else:
    do_something()
finally:
    do_something()


# example
def divide2(x, y):
    try:
        result =  x / y
    except ZeroDivisionError:
        result = None
    except TypeError:
        result = divide2(float(x), float(y))
    else:
        print "result is", result
    finally:
        print "done!"
        return result



# Data Structures
# - List: Lists are useful for storing sequential data sets, but we can also use them as the foundation for other data structures.
fibonacci_numbers = [1, 1, 2, 3, 5, 8, 13, 21]
strings = ['a', 'b', 'c', 'd']
num_str_bool = ['a', 1, 'b', 2, 'c', 3, True, False]
[fibonacci_numbers, strings, num_str_bool] # list of lists
# Indexing and Slicing (List)
print fibonacci_numbers[0] # the first element
print fibonacci_numbers[3] # the fourth element
print fibonacci_numbers[0:3] #slicing
print fibonacci_numbers[3:] #slicing
# Searching & Appending
fibonacci_numbers.index(8) #searching, the index whose values is 8
fibonacci_numbers.append(34) # add one more element



# Data Structures
# -- tuple
"""" Tuples are similar to lists in many ways, 
except they can't be modified once you've created them """

fib_numbers = (1, 1, 2, 3, 5, 8, 13, 21)
fib_numbers[1] # indexing 
fib_numbers[1:5] # slicing
fib_numbers.index(8) # searching

# -- set
""" A set is an unordered sequence of unique values. Members of a set must be hashable. """
set1 = {1, 2, 3}
set2 = {2, 3, 1}
set1 == set2
integers = {0, 1, 1, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9, 9, 8, 19}
integers
# {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19} Note that the duplicated elements are automatically removed.
# Operation on sets
evens = {0, 2, 4, 6, 8, 10, 12}
integers = {0, 1, 1, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9, 9, 8, 19}
evens.difference(integers) # The elements that appears in evens and not in integers.
evens.intersection(integers) # The elements that appears both in evens and integers.
evens.union(integers) # The elements that appears in evens or integers.




# -- dictionary
""" A dictionary is a set of keys with associated values. 
The key must be hashable, but the value can be any object. 
You might know this as a hash map or key-value pairs. """

someone = {'sex': 'male', 'height': 6.1, 'age': 30}
someone['age'] # index

print someone.keys()
print someone.values()
# ['age', 'height', 'sex']
# [30, 180, 'male']

someone.items()
# [('age', 30), ('height', 180), ('sex', 'male')]

# Un-ordered as tuple
someone1 = {'sex': 'male', 'age': 30}
someone2 = {'age': 30, 'sex': 'male'}
someone1 == someone2 # True





# ------------------------------ Python Programming Class 2
# - built-in objects
int
float
str
list
set
tuples
dictionary

# check object
"""
Each object has:
A type (a particular object is said to be an instance of a type)
Attributes(primitive)
Methods(a set of procedures for interaction with the object)
"""
li = [1, 2, 3, 4]
type(li) # type
isinstance(li, list) # is li a instance of list?
"Object attributes"
print li.__doc__
print li.__len__() # length of li, the same as len(li)
print li.count(1) # number of 1 appears
# - other procedures
"The “.” operator is used to access an attribute of an object"
li.append(...), li.reverse(...), li.pop(...), li.sort(...)
li.insert(...), li.remove(...), li.index(...)




# [Class]
""" Classes are user types that let you combine several fields and 
functions that operate on that data. It lets you group all of it together and 
treat it as a single abstract thing.
In Python, the class statement is used to define a new type. """

class People(object):
    '''
    definitions
    '''

"""
Classes can inherit attributes from other classes, 
in this case People inherits from the object class.
People is said to be a subclass of object, object is a superclass of People.
The subclass can overwrite inherited attributes of the superclass.
"""

# Create a class
class People(object):
    def __init__(self, Name, Id, Age):
        self.Name = Name
        self.Id = Id
        self.Age = Age

"""
Method is a procedure that “belongs” to this class.

When calling a method of an object(for example, the People object), 
Python always passes the object(People) as the first argument. By convention, 
we use self as the name of the first argument of methods.

The “.” operator is used to access an attribute of an object. 
So the __init__ method above is defining three attributes for the new People object: Name, Id and Age.

When an object is created, the __init__ method is called.
"""


# Create an instance for that class
Jack = People('Jack', 1, 20)
Lucy = People('Lucy', 2, 22)

print "My name is %s, I'm %s years old. My Id number is %s." %(Jack.Name, Jack.Age, Jack.Id)
print "My name is %s, I'm %s years old. My Id number is %s." %(Lucy.Name, Lucy.Age, Lucy.Id)
"""
My name is Jack, I'm 20 years old. My Id number is 1.
My name is Lucy, I'm 22 years old. My Id number is 2.
"""

# Representation of an object
print Jack
""" <__main__.People object at 0x7fb038426d50> """ # Python uses an uninformative print presentation for an object.
type(Jack)
""" __main__.People """
isinstance(Jack, People)
True




class People(object):
    def __init__(self, name, Id, age):
        self.Name = name
        self.Id = Id
        self.Age = age
        
    def __str__(self): # This __str__ method will be called when the object needs a string to print.
        return "My name is %s, I'm %s years old. My Id number is %s." %(self.Name, self.Age, self.Id)

    def addAge(self): # a function of adding 1 to the original when it is called
        self.Age = self.Age + 1
    
    def alterName(self, newName): # a function of changing name
        self.Name = newName

print Jack
"""
My name is Jack, I'm 20 years old. My Id number is 1.
"""

Jack.addAge() # one year later, Jack is 21
Jack.alterName('John') # change his name to John



# Inheritance
"""
In this case, create a men class inherits from the object People.
men is said to be a subclass of People, People is a superclass of men.
"""
# For men
class men(People):
    def __init__(self, Name, Id, Age):
        People.__init__(self, Name, Id, Age)
        self.sex = 'M'
    
    def isAdult(self):
        return self.Age >= 18
    
    def __str__(self):
        if self.isAdult():
            status = 'man'
        else:
            status = 'boy'
        return "My name is %s, I'm a %s years old %s." %(self.Name, self.Age, status)


# For women
class women(People):
    def __init__(self, Name, Id, Age):
        People.__init__(self, Name, Id, Age)
        self.sex = 'F'
    
    def isAdult(self):
        return self.Age >= 18
    
    def __str__(self):
        if self.isAdult():
            status = 'woman'
        else:
            status = 'girl'
        return "My name is %s, I'm a %s years old %s." %(self.Name, self.Age, status)

# test
Aron = men('Aron', 3, 15)

print type(Aron)
print isinstance(Aron, men)
print isinstance(Aron, People) # also People (All for People Still works for men)

"""
<class '__main__.men'>
True
True
"""








# [File]
"""
By using the function open, we create a file instance.
"""
f = open('script/foo.txt', 'w') # Mode can be: 'r' - reading(default) / 'w' - writing /
                                #              'a' - appending / 'r+' - open the file for both reading & writing

print type(f)
print isinstance(f, file)
"""
<type 'file'>
True
"""

# Write files
f.write('this is some text.\nThis is another line.\n')
f.close() # Do not forget to close the file after using. Otherwise, you can not open the file with the other applications.

# Reading files
f = open('script/foo.txt', 'r') 
f.read() # print all content in that file
f.close()

# Reading part of the file
f = open('script/foo.txt', 'r') 
f.read(5) # the first 5 bytes 
f.read() # the remaining contents
f.close()


# Other way of using readlines()
"""
Another way to read files is using the method readlines, which return a list in which each element refers to one line.
"""
f = open('script/foo.txt', 'r')
print f.readlines()
f.close()


# Iterate a file object (By each element)
f = open('script/foo.txt', 'r')
for i in f:
    print i
f.close()


f = open('script/foo.txt', 'r')
content = []
num = 0
for i in f:
    num += 1
    if num % 2 == 0:
        content.append(i)
        print num, ':', i
f.close()
print content

"""
2 : This is another line.

4 : Fourth line.

['This is another line.\n', 'Fourth line.\n']
"""


# Better way to deal with files
"""
A better way to open a file is to use with statement, it will automatically close the file after using.
"""
with open('script/foo.txt', 'r') as f: #open the file for reading
    data = f.read()
    
data # the file is closed now

# More operations about file
help(file) # Find more ****



# Sort a dictionary
"""
In this case, we create a new class myfile which is a subclass of the file class, 
and add some more methods, such as wordCountSort method used to sort the frequencies of the words.

Before I show you all the codes, it's necessary to know how to sort a dictionary.

Since dictionary is unordered, you need to convert it to a list by using .items method.
Then the function sorted can be used to sort a list.
"""

d = {'a': 5, 'b': 3, 'c':2}
d = d.items() # convert to a list
d
"[('a', 5), ('c', 2), ('b', 3)]" # a list wrap with tuples
sorted(d, key = lambda x: x[1])
"[('c', 2), ('b', 3), ('a', 5)]" 

# The argument key in sorted need a function which is callable.
# Here lambda x: x[1] is a anonymous function, which is similar to the function defined by the keyword def.
# This whole command means sort the list by the second element.


# Lambda Functions
square1 = lambda x: x**2 # anonymous function

def square1(x):
    return x ** 2 # function

print square1(2)


# Class & File ------------------------------------------ Create a new class
import string
class myfile(file):
    def __init__(self, name, mode):
        file.__init__(self, name, mode)
        
    def __str__(self):
        return "Opening file %s" %self.name
    
    def wordCount(self, punctuation='\n', ignoreCase = True):
        '''
        punctuation: punctuations to remove
    
        returns: a dict contains each word and it's corresponding frequency
        '''
        ## read contents and convert to lower
        try:
            raw_string = self.read()
            if ignoreCase:
                raw_string = raw_string.lower()
        except:
            raise Exception("Can't read file %s"%self.name)
            
        ## repalce all the punctuations with space
        for i in string.punctuation:
            raw_string = raw_string.replace(i, ' ')
        
        if punctuation != None:
            for i in punctuation:
                raw_string = raw_string.replace(i, ' ')
        
        ## split by space, count each word
        raw_list = raw_string.split(' ')
        result = {}
        for word in raw_list:
            if word in result.keys():
                result[word] += 1
            else:
                result[word] = 1
    
        # remove null character
        # len('') is 0
        result = {key:value for (key, value) in result.items() if len(key) != 0}
        return result
    
    def wordCountSort(self, descend = True, punctuation='\n', ignoreCase = True):
        '''
        return the sorted word frequency
        '''
        result = self.wordCount(punctuation, ignoreCase = ignoreCase)
        result = sorted(result.items(), key = lambda x: x[1] , reverse = descend)
        return result
    
    def mostCommonWord(self, num=5, punctuation='\n', descend = True, ignoreCase = True):
        '''
        return the most common words
        '''
        result = self.wordCountSort(punctuation=punctuation, ignoreCase = ignoreCase, descend = descend)
        if num > len(result):
            Warning('There are only %s words'%len(result))
            return result
        else:
            return result[:num]
# -----------------------------------------------------------------------------
# test
with myfile('data/abalone.data', 'r') as f: 
    print f, '...'
    print f.mostCommonWord(num = 5) 
    print f.mostCommonWord(num = 5, ignoreCase=False)
    print f.mostCommonWord(num = 5, ignoreCase=False, descend=False)
"""
Opening file abalone.data ...
[('0', 27723), ('1', 1634), ('m', 1528), ('i', 1342), ('f', 1307)]
[('0', 27723), ('1', 1634), ('M', 1528), ('I', 1342), ('F', 1307)]
[('7845', 1), ('6795', 1), ('9255', 1), ('94', 1), ('8835', 1)]
"""







# [Script]
"""
Run Python scripts

You may have heard that python is a script language, 
which means we can run a script as a shell commands without going into the python shell.
"""

# Here is simple example, write a line "print '1 + 1 = %s' %(1+1)" into the file "script1.py":
$echo "print '1 + 1 = %s' %(1+1)" > script/script1.py 

$python script/script1.py
" 1 + 1 = 2 "

"""
Run Python scripts

In practice, we usually need to interact with scripts.

Take in some inputs.
Run the script.
Return some outputs.

raw_input is a function for getting input from users.
"""

age = raw_input('how old are you?\n')
# After running this command, it will wait for user to type some inputs. 
# Then the string typed by user will be assigned to the variable age.

# write scripts in a file
script2 = open('script/script2.py', 'w')
script2.write('name = raw_input("What is your name?\\n")\n')
script2.write('age = raw_input("How old are you?\\n")\n')
script2.write('print "You are %s, %s years old."%(name, age)\n')
script2.close()

$python script/script2.py


"""
Run Python scripts

We can not run script2.py directly in this notebook, because it does not support the interactive input.
Another way to run a script with some inputs is to add more arguments for the script.
Actually, in the command python script2.py, the script2.py part is called an argument.
"""

with open('script/script3.py', 'w') as f:
    f.write('''from sys import argv
               script, name, age = argv
               print "The script is called:", script
               print "My name is %s." %name
               print "Im %s years old." %age''')

$python script/script3.py jack 18
"""
The script is called: script/script3.py
My name is jack.
Im 18 years old.
script: script/script3.py
name: jack
age: 18
"""


"""
Run Python scripts
In this case, I write a script named wordCount.py, which is used to count the word frequencies.
This script needs three arguments:

script: this wordCount.py itself
target: the target file to be counted
number: the number of most common words to return
"""

# wordCount.py --------------------------------------
" We already write a class myfile which has a method of mostCommonWord. "
# Part 1: define the arguments.
import string
from sys import argv
script, target, number = argv

# Part 2: define a new class myfile.
'''
The myfile class we defined previously, would not show you again.
'''

# Part 3: count the target file and return the result.
f = myfile(target, 'r')
print '%12s:%8s' %('Word', 'Frequency')
print '-----------------------'
for i, j in f.mostCommonWord(int(number)):
	print '%12s:%8s' %(i, j)


# Script
# After writing a script, it's always good to add more information.
# - Interpreter
"""
In Unix, an executable file can indicate what interpreter to use by having a #! at the start of the first line, 
followed by the interpreter. It's python in this case.
If you have several versions of Python installed, /usr/bin/env ensure the interpreter used is the first one 
on your environment's path.
"""
#!/usr/bin/env python

# - encoding
"""
Replace the encoding name with encoding format, 'utf-8' is recommended.
This way of specifying the encoding of a Python file comes from PEP 0263 - Defining Python Source Code Encodings.
"""
# -*- coding: <encoding name> -*-

# - author
"""
If you upload your script on the internet, it's encouraged to add you name and contact information.
"""
# Author: NYC data science <http://nycdatascience.com/>







# [Handling and Processing Strings]
print "string", 'string'
print "string in 'string' of string"
print 'string in "string" of string'
print """ string
          string """
print ''' string
          string '''        

:
print 'Hi, I do not want a \n new line.' # start a new line
print r'Hi, I do not want a \n new line.' # ignore and print '\n'
print 'Hi, I do not want a \\n new line.' # ignore and print '\n'

# Basic String Manipulations
len('abcd') # 4
'abcd'[0] # the first element
'abcd'[0:2] # the first and second element

s = 'abcdefg'
print s[-1] # -1 means the first character from the right side
print s[-2] # -2 means the second character from the right side

"For the string 'abcdefg', the index can be 0 to 6 or -1 to -7, otherwise you will get a IndexError."

# case coversion
'ABcd'.lower() # convert to lower case 
'ABcd'.upper() # convert to upper case 
'ABcd'.swapcase() # swap case(lower -> upper, upper -> lower) 
'acd acd'.title() # first element between space Go up
'a b c d'.split(' ') # split by ' ' -> ['a', 'b', 'c', 'd']
'a b c d'.replace(' ', '>') # replace ' ' with '>'
'a b c d'.count(' ') # count the number of ' ' appears
' '.join(['a', 'b', 'c']) # The join method is used to join all the elements in a list with a separator.


# Advanced String Manipulations: Regular Expressions

# Metacharacters
. ^ $ * + ? { } [ ] \ | ( )

" The library re is used to implement regular expressions in python. "
import re

raw_string = 'Hi, how are you today?'
print re.search('Hi', raw_string) 
print re.search('Hello', raw_string)
# <_sre.SRE_Match object at 0x7fb9506ae308>
# None
s = re.search('Hi', raw_string)
print s.start() # the starting position of of the matched string
print s.end()   # the ending position index of the matched string
print s.span()  # a tuple containing the (start, end) positions of the matched string
# 0
# 2
# (0,2)
print s.group() # the matched string
print raw_string[s.start():s.end()] # same
# Hi
# Hi

. # refers to any single characters. For example, a. matches any two characters start with 'a': aa, ab, an, a1, a#, etc. 
print re.search('a.', 'aa') != None
print re.search('a.', 'ab') != None
print re.search('a.', 'a1') != None
print re.search('a.', 'a#') != None
# True
# True
# True
# True

? # matches a character either once or zero times.
print re.search('ba?b', 'bb') != None    # match
print re.search('ba?b', 'bab') != None   # match
print re.search('ba?b', 'baab') != None  # does not match
# True
# True
# False

+ # matches a character at least once.
print re.search('ba+b', 'bb') != None    # does not match
print re.search('ba+b', 'bab') != None   # match
print re.search('ba+b', 'baab') != None  # match
print re.search('ba+b', 'baaaab') != None  # match
print re.search('ba+b', 'baaaaaab') != None  # match
# False
# True
# True
# True
# True


* # matches a character arbitrary times.
print re.search('ba*b', 'bb') != None    # match
print re.search('ba*b', 'bab') != None   # match
print re.search('ba*b', 'baaaaaab') != None  # match
# True
# True
# True


{m,n} # matches a character at least m times and at most n times.
print re.search('ba{1,3}b', 'bab') != None    # match
print re.search('ba{1,3}b', 'baab') != None   # match
print re.search('ba{1,3}b', 'baaab') != None  # match
print re.search('ba{1,3}b', 'bb') != None     # does not match
print re.search('ba{1,3}b', 'baaaab') != None # does not match
# True
# True
# True
# False
# False


^ # refers to the beginning of a text, while $ refers to the ending of a text.
print re.search('^a', 'abc') != None    # match
print re.search('^a', 'abcde') != None  # match
print re.search('^a', ' abcde') != None # does not match
# True
# True
# False


a$ # matches all the text ends with character a.
print re.search('a$', 'aba') != None    # match
print re.search('a$', 'abcba') != None  # match
print re.search('a$', ' aba ') != None  # does not match
# True
# True
# False




[] # [] is used for specifying a set of characters that you wish to match. 
   # For example, [123abc] will match any of the characters 1, 2, 3, a, b, or c ; this is the same as [1-3a-c], 
   # which uses a range to express the same set of characters. Further more [a-z] matches all the lower letters, 
   # while [0-9] matches all the numbers.
print re.search('[123abc]', 'defg')  != None   # does not match
print re.search('[123abc]', '1defg') != None   # match
print re.search('[1-3a-c]', '2defg') != None   # match
print re.search('[123abc]', 'adefg') != None   # match
print re.search('[1-3a-c]', 'bdefg') != None   # match
# False
# True
# True
# True
# True



() # is very similar to the mathematical meaning, they group together the expressions contained inside them, 
   # and you can repeat the contents of a group with a repeating qualifier.
print re.search('(abc){2,3}', 'abc')  != None         # does not match
print re.search('(abc){2,3}', 'abcabc')  != None      # match
print re.search('(abc){2,3}', 'abcabcabc')  != None   # match
# False
# True
# True


| # is a logical operator. For examples, a|b matches a and b, which is similar to [ab].
print re.search('[ab]', 'a') != None   # match
print re.search('[ab]', 'b') != None   # match
print re.search('[ab]', 'c') != None   # does not match
print re.search('abc|123', 'a') != None   # does not match
print re.search('abc|123', '1') != None   # does not match
print re.search('abc|123', '123') != None # match
print re.search('abc|123', 'abc') != None # match


\
# If you want to match exactly ?, it is necessary to add a backslash \?.
print re.search('\?', 'Hi, how are you today?') != None # match


# Useful Functions
"""
re.split(pattern, string): Split the string into a list by the pattern.
re.sub(pattern, replace, string): Replace the substrings in the string that matches the pattern with the argument replace.
re.findall(pattern, string): Find all substrings where the pattern matches, and returns them as a list.
"""

# - split
s = '''The re module was added in Python 1.5, 
and provides Perl-style regular expression patterns. 
Earlier versions of Python came with the regex module, 
which provided Emacs-style patterns. 
The regex module was removed completely in Python 2.5.'''
s2 = s
for i in [',', '.', '-', '\n']:
    s2 = s2.replace(i, ' ')
s2.split(' ')

# - split
re.split('[\n ,\.-]+', s)


# - sub
s3 = s
s3 = re.sub('[\n,.-]', ' ', s3)
print s3
re.split(' +', s3) 
# since there are empty characters in the result,
# \ we split it by one or more blank space


# - findall
re.findall('[a-zA-Z]+', s) # if you want number too, run re.findall('[a-zA-Z0-9]+', s) 


# Special sequence in regular expression
\d: Matches any decimal digit; this is equivalent to the class [0-9].
\D: Matches any non-digit character; this is equivalent to the class [^0-9].
\w: Matches any alphanumeric character; this is equivalent to the class [a-zA-Z0-9_].
\W: Matches any non-alphanumeric character; this is equivalent to the class [^a-zA-Z0-9_].
\s: Matches any whitespace character; this is equivalent to the class [ \t\n\r\f\v].
\S: Matches any non-whitespace character; this is equivalent to the class [^ \t\n\r\f\v].
\t: tab,
\v: vertical tab.
\r: Carraige return. Move to the leading end (left) of the current line.
\n: Line Feed. Move to next line, staying in the same column. Prior to Unix, usually used only after CR or LF.
\f: Form feed. Feed paper to a pre-established position on the form, usually top of the page.



"""
Operations

Typically, regular expression patterns consist of a combination of using various operators.
Among the various types of operators, their main use reduces to four basic operations for creating regular expressions:

Concatenation: concatenating a set of characters together. For example 'abcd', which just matches the single string 'abcd'.

Logical OR: denoted by the vertical bar | or []. For example 'ab|cd', which matches two strings 'ab' and 'cd', while '[abcd] matches 'a', 'b', 'c' or 'd'.

Replication: define a pattern that matches under multiple possibilities. ?,+,*,{m,n}.

Grouping: denoted with a expression inside parentheses ( ).
"""

# Is it email?
users = ['somename9', 'some_name', 'contact', 'some.name', 'some.name', 'some_name']
for i in users:
    if re.search('^[a-z0-9]+[_\.]?[a-z0-9]+', i) != None:
        print "Match!"
    else:
        print "Does not match!"
# Is it a website?
domain = ['gmail.com', 'yahoo.com', 'supstat.com.cn', 'an-email.com', 'an.email.com', '163.com']
for i in domain:
    if re.search('[a-z0-9]+([-\.]?[a-z]){1,3}$', i) != None:
        print "Match!"
    else:
        print "Does not match!"




















# Python for data analysis

"""
Numpy: Numpy, short for Numerical Python, is the foundational package for scientific computing in Python.

Matplotlib: Matplotlib is the most popular Python library for producing plots and other 2D data visualizations.

SciPy: SciPy is a collection of packages addressing a number of different standard problem domains in scientific computing.

scipy.stats : standard continuous and discrete probability distributions, various statistical tests, and more descriptive statistics
"""

# [NumPy]
"""
NumPy is a matrix type for python, and a large number of functions to operate on these matrices.

It’s a library that makes doing calculations easy and faster to execute, because the calculations are done in C rather than python.

There are two mainly data types in NumPy: the array and the matrix.

The operations on arrays and matrices are slightly different. But both types allow you to remove loops.
"""

# - array
from numpy import array # Import array from numpy, add up two arrays directly without a for loop in regular Python.
a1 = array([1, 1, 1])
a2 = array([1, 2, 3])
a1 + a2
"array([2, 3, 4])"

"""
array in Numpy
All the basic operations(+, -, *, /, etc.) can be done element-wisely in array without loop.
Multiply each element by a constant, which also require a loop in regular python.
"""
a1 * 2
"array([2, 2, 2])"
a1 * a2
'array([1, 2, 3])'

a1 = array((1, 2, 3)) # In the array of Numpy, you can access the elements like it was a list:
print a1[0]
print a1[1:]
'1'
'[2 3]'



"""
array in Numpy
Numpy also support multidimensional arrays, and access the elements like lists or a matrix.
"""
am = array([[1, 2, 3], [4, 5, 6]])
"""
array([[1, 2, 3],
       [4, 5, 6]])
"""
am[0] # acess the first row
"array([1, 2, 3])"
am[0][1] # access the elements like lists
"2"
am[0, 1] # same with am[0][1]. access the elements like a matrix
"2"




"""
Array Creation
Given list objects in regular python, it's easy to convert them to array object:
"""

a1 = [1, 2, 3]
a2 = [[1, 2, 3], [4, 5 ,6]]
a1 = array(a1)
print type(a1);
a1
"<type 'numpy.ndarray'>"

# It also support mix of tuples and lists.
array([[0, 0], [1, 1], (2, 2)]) # list, list, tuple
"""
array([[0, 0],
       [1, 1],
       [2, 2]])
"""


"""
Array Creation
Numpy also provides some built-in functions to create special regular arrays.
"""

import numpy as np
np.arange(10) # start from 0 in default / np.arange: create arrays with regularly incrementing values, which is similar to range
"array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"


np.arange(9.5) 
# a float result in a array consist of float
# the same as np.arange(10, dtype=np.float)
"array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])"

np.arange(10, dtype=np.float)
"array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])"

np.arange(2, 10) # start from 2
"array([2, 3, 4, 5, 6, 7, 8, 9])"

np.arange(1, 10, 0.3) # start from 1 and increment by 0.5, end with a number less than 10.
"""
array([ 1. ,  1.3,  1.6,  1.9,  2.2,  2.5,  2.8,  3.1,  3.4,  3.7,  4. ,
        4.3,  4.6,  4.9,  5.2,  5.5,  5.8,  6.1,  6.4,  6.7,  7. ,  7.3,
        7.6,  7.9,  8.2,  8.5,  8.8,  9.1,  9.4,  9.7])
"""




"""
Array Creation
np.linspace: create arrays with a specified number of elements, and spaced equally between the specified beginning and end values.
"""

np.linspace(10, 20) 
# start from 10, end with 20(included!), length is 50 in default
"""
array([ 10.        ,  10.20408163,  10.40816327,  10.6122449 ,
        10.81632653,  11.02040816,  11.2244898 ,  11.42857143,
        11.63265306,  11.83673469,  12.04081633,  12.24489796,
        12.44897959,  12.65306122,  12.85714286,  13.06122449,
        13.26530612,  13.46938776,  13.67346939,  13.87755102,
        14.08163265,  14.28571429,  14.48979592,  14.69387755,
        14.89795918,  15.10204082,  15.30612245,  15.51020408,
        15.71428571,  15.91836735,  16.12244898,  16.32653061,
        16.53061224,  16.73469388,  16.93877551,  17.14285714,
        17.34693878,  17.55102041,  17.75510204,  17.95918367,
        18.16326531,  18.36734694,  18.57142857,  18.7755102 ,
        18.97959184,  19.18367347,  19.3877551 ,  19.59183673,
        19.79591837,  20.        ])
"""






"""
Array Creation
np.zeros np.ones create an array filled with 0 values with the specified shape.
"""

np.zeros([2, 3]) # 2 rows, 3 columns
"""
array([[ 0.,  0.,  0.],
       [ 0.,  0.,  0.]])
"""
np.ones([2, 3]) # 2 rows, 3 columns
"""
array([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.]])
"""
np.eye(3) # np.eye create an identity array with the specified shape.
"""
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]])
"""







"""
matrix in Numpy
Now let's turn to matrix in Numpy. Similar to array, it's necessary to import matrix or mat from numpy.
"""
from numpy import mat, matrix
m1 = mat([1, 2 ,3]) # use 'mat' only after import mat, otherwise 'matrix'
m1
"matrix([[1, 2, 3]])"

m2 = mat([[1, 2, 3], [4, 5, 6]])
m2
"""
matrix([[1, 2, 3],
        [4, 5, 6]])
"""

# index - The operation of access the elements of a matrix is similar to array but not the same:
m2 = mat([[1, 2, 3], [4, 5, 6]])
m2[1] # second row, still a matrix
"matrix([[4, 5, 6]])"
m2[0, 1] # element in the 1th row and 2th column
"2"
m2[0][1] XXXXXXX # Not works






"""
Operation in Matrices¶
It's ok to do addition, subtraction and division between matrices, 
just like arrays, the result is add or minus by each individuals:
"""
m1 = mat([[1, 2, 3], [4, 5, 6]])
m2 = mat([[1, 2, 3], [4, 5, 6]])
print m1 + m2
print m1 - m2
print m1 / m2
"""
[[ 2  4  6]
 [ 8 10 12]]
[[0 0 0]
 [0 0 0]]
[[1 1 1]
 [1 1 1]]
"""

" Multiplication between matrices do not work in the same way as array, it's not element-wised: "

m2 = m1.T # The .T method is used to transpose a matrix, it can also be applied on a array object.

m1 * m2 # can be between matrix and arrary and returns an arrary (first #row = second #column)

from numpy import multiply
m1 = mat([[1, 2], [3, 4]])
m2 = mat([[5, 6], [7, 8]])
multiply(m1, m2)

"""
matrix([[ 5, 12],
        [21, 32]])
"""

" In addition, you can also multiple a square matrix by itself: "
m1 = mat([[1, 2], [3, 4]])
print m1 ** 2 # same as m1 * m1
print m1 ** 3 # same as m1 * m1 * m1


# To get the dimensions of an array or a matrix, import the function shape in Numpy:
from numpy import shape
m1 = mat([[1, 2, 3], [4, 5, 6]])
a1 = array([[1, 2, 3], [4, 5, 6]])
print shape(m1)
print shape(a1)
print a1.shape
print m1.shape
"""
(2,3)
(2,3)
(2,3)
(2,3)
"""

# The reshape method can be used to reshape array and matrix.
a1.reshape([6, 1])
"""
array([[1],
       [2],
       [3],
       [4],
       [5],
       [6]])
"""
m1.reshape([3, 2])
"""
matrix([[1, 2],
        [3, 4],
        [5, 6]])
"""


# Inverse a matrix
" To get the inverse of a square matrix, use the .I method "
m = mat([[1, .5], [.5, 1]])
m.I
"""
matrix([[ 1.33333333, -0.66666667],
        [-0.66666667,  1.33333333]])
"""

" Test whether the result of the inverse of  mm  multiply  mm  is a identity matrix: "
m * m.I
"""




































































































