#################################
#                               #
#                               #
#   Python for Data Analysis    #
#                               #
#                               #
#################################

# ------------------------------ Python Data Analysis Basic 
# package:
import [built_in] # Basic Python data sctructure and operations, classes

import Numpy as Np # Numpy, short for Numerical Python, is the foundational package for scientific computing in Python

import SciPy as Sp # SciPy is a collection of packages addressing a number of different standard problem domains in scientific computing.

import Pandas as Pd # Pandas provides rich data structures and functions designed to make working with structured data fast, easy, and expressive. The DataFrame object in pandas is just like the data.frame object in R





-----------------------------------------------------------------------------------------------------------------------------
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>                               >  
>                               >
>    Python Built_in Basic      >
>                               >
>                               >
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Python - data analysis - Basic

# ------------------------------ Install Python
# [1] Set up 
# anaconda python
Google: anaconda python
-> Download "anaconda"
-> 2.7 version
-> choose installer
# IPython notebook
anaconda promte
-> conda update conda # update conda
-> conda update ipython ipython-notebook ipython-qtconsole # update ipython
-> ipython notebook # open notebook
# install packages in python
anaconda promte
-> conda install *packages # will take care all dependencies




# ------------------------------ I/O programming in Python
# print sub
print "%s is %d is %r" % ('mark', 12, "\nmark")
" mark is 12 is \nmark" 

print "I" + "S"
"IS"



# User input
tire_degree = int(raw_input("\aHow tired you are? "))*100
Do_you = raw_input("\aYou want to save and quit? ")
Again = raw_input("\aAre you sure? ")
print "You are %s times tired. \n%s is whether you want to quit. \nWhen I ask you sure? You said \"%s\"." % (tire_degree, Do_you, Again)

promt = ">> "
print "Do you feel bad today?"
feel = raw_input(promt)





# From command line - argv
command line: python python.py 1st 2nd ...
from sys import argv

scrpit, a, b, c = argv

print "scrpit is:", scrpit
print "a is:", a
print "b is:", b
print "c is:", c





# Define function in file.py then import it in python to use
def add(a, b, c):
    print "adding %d, %d, and %d together:" % (a, b, c)
    return a + b + c
def super_add(a, b, c, d):
    print "adding %d, %d, and %d together:" % (a, b, c)
    print "Then, divided by %d." % d
    return (a + b + c) / d
$python
>>> import filename
>>> filename.add(10, 20, 30)
>>> filename.super_add(10, 20, 30, 5)
or
>>> from filename import *
>>> add(10, 20, 30)
>>> super_add(10, 20, 30, 5)



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


# -- set working directory
import os
os.getcwd() # disply current wd
os.chdir('/user/xxxx/xxx') # change current wd to




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
print True or (False and True)


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

# -- Flow control
break

pass

continue

# -- enumerate // give index next to value
my_list = ['apple', 'banana', 'grapes', 'pear']
for c, value in enumerate(my_list,1):

    print(c, value)



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
        
    f
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

sexCount = sorted(sexCount.items(), key=lambda x: x[1]) # call fun to create a var tmp

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








-----------------------------------------------------------------------------------------------------------------------------
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>                               >  
>                               >
>    Numpy - Numeric Python     >
>                               >
>                               >
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

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

np.arange(1, 10, 0.3) # start from 1 and increment by 0.3, end with a number less than 10.
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



" concatenate arraries "
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.concatenate([a, b])





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



# mean
" To calculate the mean of the numbers in matrix or array, use the .mean() method: "
m = mat([3, 2, 1])
a = array(m) # convert a matrix to array
print a.mean()
print m.mean()
2.0
2.0

from numpy import mean
print mean(a)
print mean(m)
2.0
2.0


# Sort
" .sort() method can be used to sort a matrix or array: "
m = mat([3, 2, 1])
m.sort()
m
"matrix([[1, 2, 3]])" # Note that use this method will loss the original order, make a copy if you want to keep your original data.


# Random sampling in Numpy
rand(d0, d1, ..., dn) # Random values in a given shape from [0, 1] uniformly.
randn(d0, d1, ..., dn) # Return a sample (or samples) from the “standard normal” distribution.
randint(low[, high, size]) # Return random integers from low (inclusive) to high (exclusive).
random_integers(low, high, size) # Return random integers between low and high, inclusive.
random_sample([size]) # Return random floats in the half-open interval [0.0, 1.0).
random([size]) # Return random floats in the half-open interval [0.0, 1.0).
ranf([size]) # Return random floats in the half-open interval [0.0, 1.0).
sample([size]) # Return random floats in the half-open interval [0.0, 1.0).
choice(a, size, replace, p]) # Generates a random sample from a given 1-D array

from numpy import random
random.rand(2, 3)
"""
array([[ 0.70001978,  0.69059499,  0.64586369],
       [ 0.50370903,  0.87010881,  0.74388861]]) 
"""
random.randn(2, 3)
"""
array([[ 1.07956698,  0.42241258, -0.93731842],
       [ 0.21060919, -0.13326986, -0.61693296]])
"""
random.randint(5, size=10) # Generate 10 random integers from 0(include) to 5(exclude):
"array([4, 3, 1, 3, 0, 1, 2, 0, 2, 4])"
random.randint(low=1, high=6, size=(5, 5)) 
# Return random integers from low (inclusive) to high (exclusive)
"""
array([[4, 2, 1, 1, 1],
       [4, 3, 1, 5, 1],
       [3, 2, 2, 3, 3],
       [1, 1, 1, 3, 2],
       [3, 5, 4, 5, 4]])
"""
random.random_integers(low=1, high=5, size=(5, 5)) 
# Return random integers from low (inclusive) to high (inclusive)
"""

array([[2, 5, 5, 1, 1],
       [2, 2, 2, 4, 1],
       [4, 1, 5, 5, 2],
       [1, 3, 4, 4, 1],
       [5, 3, 2, 5, 5]])
"""
"""
The only difference between function randint and random_integers is the high argument 
is excluded in randint while it's included in random_integers.
"""

# The functions random_sample, random, ranf, sample are similar to rand, 
# but different from the way to define the dimensions.
random.rand(3, 3) # 3-3 array
"""
array([[ 0.59456034,  0.28652971,  0.47488913],
       [ 0.89846765,  0.82264671,  0.75137963],
       [ 0.60963399,  0.00620382,  0.78680171]])
"""
random.ranf([3, 3]) 
# 3-3 array, note that the dimension should be included in a list or array
"""
array([[ 0.61330923,  0.7010375 ,  0.73574806],
       [ 0.18704789,  0.60682643,  0.45903498],
       [ 0.81561344,  0.94776034,  0.95131061]])
"""
random.ranf(2, 3, 3) # otherwise trigger a error
!error



# choice 
"Sampling from a data set"
all_set = [1, 2, 3, 5, 8, 13, 21, 34, 55]
random.choice(all_set, size=10)
"array([55, 13,  2, 21,  1,  8, 34,  2,  3,  8])"
# In default, function choice sample from the given set with replacement.
# We can also set the replacement with false, 
# but the size can not be larger than the size of the set:
random.choice(all_set, size=10, replace=False)
!error

" example - Birthday problem "
def generateBirthday(num):
    '''
    num: number of people
    '''
    birthdays = range(366)
    randomChoose = np.random.choice(birthdays, size=num, replace=True)
    return len(set(randomChoose)) != num 

def sameBirthdayProb(num, times = 1e3):
    '''
    num: number of people
    times: number of simulation times
    '''
    result = []
    for i in np.arange(times):
        result.append(generateBirthday(num))
    return np.mean(result)

sameBirthdayProb(20)
0.39900000

# More Function
"Function    Description"
abs, fabs   # Compute the absolute value element-wise for integer, floating point, or complex values. Use fabs as a faster alternative for non-complex-valued data.
sqrt    # Compute the square root of each element.
square  # Compute the square of each element.
exp  # Compute the exponent e x of each element
log, log10, log2, log1p # Natural logarithm (base e), log base 10, log base 2, and log(1 + x), respectively
sign    # Compute the sign of each element: 1 (positive), 0 (zero), or -1 (negative)
ceil    # Compute the ceiling of each element, i.e. the smallest integer greater than or equal to each element
floor   # Compute the floor of each element, i.e. the largest integer less than or equal to each element
rint    # Round elements to the nearest integer, preserving the dtype
modf    # Return fractional and integral parts of array as separate array
isnan   # Return boolean array indicating whether each value is NaN (Not a Number)
isfinite, isinf # Return boolean array indicating whether each element is finite (non- inf , non- NaN ) or infinite, respectively
cos, cosh, sin, sinh, tan, tanh # Regular and hyperbolic trigonometric functions
arccos, arccosh, arcsin, arcsinh, arctan, arctanh  # Inverse trigonometric functions




# Descriptive statistics
"""
Descriptive statistics are what you are probably most familiar with - 
some examples are the mean and variance of a sample.
These are measures of central tendency(mean), spread(variance) and shape(skewness, kurtosis) 
respectively.
"""


import numpy as np
print np.mean([1, 2]) # mean
print np.mean([1, 2, 3, 4, 5])

print np.var([1, 2 ,3 ,4, 5]) # variance
print np.var([1, 2])

# Skewness - Skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable about its mean.
x = [1, 2, 3, 4, 5]
def skew(x):
    x = np.array(x)
    return np.mean(((x - np.mean(x)) / np.std(x)) ** 3)
skew(x)


# Kurtosis - is a measure of the "peakedness" of the data. In a similar way to the concept of skewness, kurtosis is a descriptor of 
# the shape of a probability distribution.
def kurtosis(x):
    x = np.array(x)
    return np.mean((x - np.mean(x))**4) / np.var(x)**2
kurtosis([1, 2, 3 ,4 ,5])


# Distributions
"""
In addition to looking at summary values, we can also look at the data as a whole and see how it is shaped, or distributed.

One of the easiest ways is through a histogram. We choose a number of buckets and store the count of each member that belongs to that bucket.

What data strucutes could we use for representing histograms?
The mode is the most frequent value. How would we find that with your histogram data structure?
"""

# - Read Same dataset into Python on afrequency

class Abalone(object):
    def __init__(self, sex, length, diameter, height, whole_weight, shucked_weight, viscera_weight, shell_weight, rings):
        self.sex = sex
        self.length = float(length)
        self.diameter = float(diameter)
        self.height = float(height)
        self.whole_weight = float(whole_weight)
        self.shucked_weight = float(shucked_weight)
        self.viscera_weight = float(viscera_weight)
        self.rings = int(rings)
    
    def data(self):
        return [self.sex, self.length, self.diameter, self.height, self.whole_weight, \
                self.shucked_weight, self.viscera_weight, self.rings]

with open('data/abalone.data') as f:
    abalone_file = f.readlines()
    abalones = [Abalone(*row.split(',')) for row in abalone_file]


for i in range(5): # Read the file top 5
    print abalones[i].data() 

"""
['M', 0.455, 0.365, 0.095, 0.514, 0.2245, 0.101, 15]
['M', 0.35, 0.265, 0.09, 0.2255, 0.0995, 0.0485, 7]
['F', 0.53, 0.42, 0.135, 0.677, 0.2565, 0.1415, 9]
['M', 0.44, 0.365, 0.125, 0.516, 0.2155, 0.114, 10]
['I', 0.33, 0.255, 0.08, 0.205, 0.0895, 0.0395, 7]
"""










-----------------------------------------------------------------------------------------------------------------------------
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>                               >  
>                               >
>   Scipy - Scientific Python   >
>                               >
>                               >
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# Scipy 
"""
SciPy is a collection of mathematical algorithms and convenience functions for python, 
whcih makes Python become a data-processing 
and system-prototyping environment rivaling systems such as MATLAB.
"""

# scipy.stats

import scipy
from scipy import stats
scipy.info(stats) # menu


# - stats.itemfreq: count frequency of a var
ringsFre = stats.itemfreq(var[])
ringsFre
"""
array([[  1,   1],
       [  2,   1],
       [  3,  15],
       [  4,  57],
       [  5, 115],
       [  6, 259],
       [  7, 391],
       [  8, 568],
       [  9, 689],
       [ 10, 634],
       [ 11, 487],
       [ 12, 267],
       [ 13, 203],
       [ 14, 126],
       [ 15, 103],
       [ 16,  67],
       [ 17,  58],
       [ 18,  42],
       [ 19,  32],
       [ 20,  26],
       [ 21,  14],
       [ 22,   6],
       [ 23,   9],
       [ 24,   2],
       [ 25,   1],
       [ 26,   1],
       [ 27,   2],
       [ 29,   1]])
"""


# -- stats.describe:  return some basic descriptive statistics including: size, min/max, mean, variance, skewness and kurtosis.
names = ["size", "min/max", "mean", "variance", "skewness", "kurtosis"]
zip(names, stats.describe(var[]))



# -- normal distribution vars
print stats.norm.__doc__ # all options
rvs(loc=0, scale=1, size=1)
stats.norm.rvs(10, 5) # normal distribution  N(10,5)

pdf(x, loc=0, scale=1)
stats.norm.pdf(0) # pdf is short for probability density function

cdf(x, loc=0, scale=1)
stats.norm.cdf(3) - stats.norm.cdf(-3)  # The probability of  −3 ≤ x ≤ 3 if x ∈ N(0,1)  

# ex. normal dist plot
mu = 0
sigma = 1
x = np.arange(mu-5, mu+5, 0.1)
y = stats.norm.pdf(x, mu, sigma)
plt.rcParams['figure.figsize'] = 8, 6
plt.plot(x, y)

theta = np.linspace(-3, 3, 100) 
r = stats.norm.pdf(theta)
theta = np.concatenate([[-3], theta, [3]])
r = np.concatenate([[0], r, [0]])                      

plt.fill(theta, r, alpha = 0.3)
plt.title('Normal: $\mu$=%.2f, $\sigma^2=%.2f$'%(mu, sigma))
plt.text(-1.9, 0.05, '$p(-3 < x < 3)=%.4f$'%(1 - 2*stats.norm.cdf(-3)), size = 15)
plt.xlabel('x')
plt.ylabel('Probability density')
plt.show()





# Poisson Distribution
stats.poisson.pmf(4, 2) 
# equivalent to stats.poisson.cdf(4, 2) - stats.poisson.cdf(3, 2) 
# p(x <= 4) - p(x <= 3) = p(x=4)
lam = 2
sequence = np.arange(0, 10)
plt.bar(sequence, stats.poisson.pmf(sequence, lam), alpha=0.3)
plt.title('Poisson distribution: $\lambda=%s$'%lam)
plt.xlabel('x')
plt.show()




# Binomial Distribution
stats.binom.pmf(2, 10, 0.5) 
n = 10
p = 0.5
sequence = np.arange(0, n+1)
plt.bar(sequence, stats.binom.pmf(sequence, n, p), alpha=0.3)
plt.title('Binomial distribution: $n=%s, p=%.2f$'%(n, p))
plt.xlabel('x')
plt.xlim(0, n+1)
plt.show()






# Hypothesis Testing
"""
Steps of Hypothesis testing:
- Propose a null hypothesis.
- Run an experiment and get a result.
- Calculate the probability of the occured result and a more extremely result occur.
- If the probability is small then a pre-defined threshold, reject the null hypothesis.


In manufacturing industry, people usually sample a center number of products to test the production yield.
Suppose that the production yield should be at least 99%. We sample 1000 products, and 5 of them are rejects.
Are this batch of products qualified?
Null Hypothesis:  H0:p<=0.99H0:p<=0.99 .
pp  is the probability of qualified product.
The number of rejects follows a binomial distribution  B(1000,0.99).
"""
pValue = 1 - stats.binom.cdf(994, 1000, 0.99) # p(n >= 995)
pValue < 0.05
False # Reject




# one-sample t-test
x = np.random.normal(loc = 10, scale = 10, size=100)
# equivalent to x = stats.norm.rvs(loc=10, scale=10, size=100)
stats.ttest_1samp(x, 15) # Is the mean of x equal to 15? Say the null hypothesis is  x¯=15.
" (-16.71046557615076, 4.8063284931973768e-58) " # see second less than 0.05 reject

# Two sample t test
stats.ttest_ind(x, y) 



# Analysis of Variance (ANOVA)
"""" What if we have more than two categories?
Analysis of Variance is a way of treating these kind of cases."""

# ex. plot three continous vars
plt.rcParams['figure.figsize'] = 12, 9
plt.subplot2grid((2, 3), (0, 0), colspan=3)
plt.boxplot([Mheight, Fheight, Iheight])
plt.ylim(0, 0.3)
plt.xticks([1, 2, 3], ('M', 'F', 'I'), fontsize = 15)

plt.subplot2grid((2, 3), (1, 0))
plt.hist(Mheight, color='blue', alpha=0.5)
plt.xlabel('M')
plt.ylim(0, 1200)

plt.subplot2grid((2, 3), (1, 1))
plt.hist(Fheight, color='red', alpha = 0.5)
plt.xlabel('F')
plt.ylim(0, 1200)

plt.subplot2grid((2, 3), (1, 2))
plt.hist(Iheight, color='pink', alpha = 0.5)
plt.xlabel('I')
plt.ylim(0, 1200)


stats.f_oneway(Mheight, Fheight, Iheight) # Anova









-----------------------------------------------------------------------------------------------------------------------------
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>                               >  
>                               >
>            pandas             >
>                               >
>                               >
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Data manipulation in Pandas
import pandas as pd

# Reading Data
pd.read_csv('data/foo.csv') # read comma-separated text files
pd.read_table('data/foo.txt') # read table-separated text files
pd.read_csv('data/foo.txt', sep = '\t') # Or define sep

# separated by , or \t
pd.read_csv('data/foo2.txt', sep = '\t|,') # multiple seps

pd.read_csv('data/foo_noheader.csv', header=None) # headers?
pd.read_csv('data/foo_noheader.csv', names=['a', 'b', 'c', 'd', 'message'], \
                                     index_col=['1','2','3']) # Set headers, index manually

pd.read_csv('data/foo_intro.csv', skiprows=range(6), header=True) # skip the first 6 rows

# -- More arguments
"""
Argument    Description
path    String indicating filesystem location, URL, or file-like object
sep or delimiter    Character sequence or regular expression to use to split fields in each row
header  Row number to use as column names. Defaults to 0 (first row), but should be None if there is no header row
index_col   Column numbers or names to use as the row index in the result. Can be a single name/number or a list of them for a hierarchical index
names   List of column names for result, combine with header=None
skiprows    Number of rows at beginning of file to ignore or list of row numbers (starting from 0) to skip
na_values   Sequence of values to replace with NA
comment Character or characters to split comments off the end of lines
parse_dates Attempt to parse data to datetime; False by default. If True, will attempt to parse all columns. Otherwise can specify a list of column numbers or name to parse. If element of list is tuple or list, will combine multiple columns together and parse to date (for example if date/time split across two columns)
keep_date_col   If joining columns to parse date, drop the joined columns. Default True converters Dict containing column number of name mapping to functions. For example {'foo': f} would apply the function f to all values in the 'foo' column
dayfirst    When parsing potentially ambiguous dates, treat as international format (e.g. 7/6/2012 -> June 7, 2012). Default False
date_parser Function to use to parse dates
nrows   Number of rows to read from beginning of file
iterator    Return a TextParser object for reading file piecemeal chunksize For iteration, size of file chunks
skip_footer Number of lines to ignore at end of file
verbose Print various parser output information, like the number of missing values placed in non-numeric columns
encoding    Text encoding for unicode. For example 'utf-8' for UTF-8 encoded text
squeeze If the parsed data only contains one column return a Series
thousands   Separator for thousands, e.g. ',' or '.'
"""


# Writing Data
data.to_csv('data/tips.csv')
# header: False
# index: False
# delimiter: \t
tips.to_csv('data/tips2.csv', 
            index=False, 
            header=False,
            sep = '\t')


data.to_csv('data/tips.csv', header=True, index=False, encoding='utf-8') # If string encoded in Unicode

# Interacting with Databases: SQLite

"""
In many applications data rarely comes from text files. SQL-based relational databases (such as SQL Server, PostgreSQL, and MySQL) are in wide use.

Python has a built-in sqlite3 driver, which can be used to connect to the in-memory SQLite database.
SQLite is a software library that implements a self-contained, serverless, zero-configuration, transactional SQL database engine.
SQLite is the most widely deployed SQL database engine in the world.
"""

import sqlite3
con = sqlite3.connect(':memory:') 
# ":memory:" to open a database connection to a database that \
# resides in RAM instead of on disk. 
type(con) # a connection object

" Run a query to create a table named test "
# define a query that create a table with four columns
query = """
CREATE TABLE test
    (a VARCHAR(20), 
     b VARCHAR(20),
     c REAL,
     d INTEGER);"""

# execute the query
con.execute(query)
# Commit the current transaction.
con.commit()

"Insert a few rows of data"
data = [('Atlanta', 'Georgia', 1.25, 6), \
        ('Tallahassee', 'Florida', 2.6, 3), \
        ('Sacramento', 'California', 1.7, 5)]
stmt = "INSERT INTO test VALUES(?, ?, ?, ?)"
# Repeatedly executes a SQL statement
# replace ? with the elements in the list
con.executemany(stmt, data)
con.commit()


" Read data from database "
cursor = con.execute('select * from test')
type(cursor)

test = cursor.fetchall() 
# Fetches all rows from the results
# return a list
test

pd.DataFrame(test) # convert to DataFrame





# Interacting with Databases: MySQL
"""
MySQL is another widely used database. I assume you already have MySQL installed.

To connect MySQL, you should install a MySQL driver. The most used package to do so is MySQLdb.

Window users: get a exe of MySQLdb
Linux users: install the package (python-mysqldb). (Ubuntu users run sudo apt-get install python-mysqldb)
Mac users: install MySQLdb here(it also works for Linux).
"""

import MySQLdb
# setting: host, user, password, db
mysql = MySQLdb.connect(host="localhost", # your host, usually localhost
                        user="root",    # your username
                        passwd="****", # your password
                        db="mysql")   # the database

# Firstly, create a Cursor object, which will let
# you execute all the queries you need
cur = mysql.cursor() 
# run a query to show all the tables
cur.execute('show tables;')
# result
cur.fetchall()


# the first row in table data_meeting
cur.execute('select * from data_meeting limit 1;')
# result
cur.fetchall()



# Interacting with Databases with Pandas
"""
Pandas has a read_sql function in its pandas.io.sql module that simplifies the process. 
Just pass the select statement and the connection object:
"""

## read from SQLite
import pandas.io.sql as sql
sql.read_sql('select * from test', con)

## read from MySQL
sql.read_sql('select * from data_meeting limit 5;', mysql)







# Arithmetic operations in Pandas
" One of the most important pandas features is the behavior of arithmetic between objects with different indexes. "

s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
s2 = pd.Series([-1, -2, -3, -4, -5], index=['a', 'c', 'd', 'e', 'f'])
print s1
print s2

"""
a    1
b    2
c    3
d    4
dtype: int64
a   -1
c   -2
d   -3
e   -4
f   -5
dtype: int64
"""


s1 + s2 # add by index
"""
a     0
b   NaN
c     1
d     1
e   NaN
f   NaN
dtype: float64
s1 and s2 were added by their common indices, it return NaN if the indices that don’t overlap.
"""


import numpy as np
df1 = pd.DataFrame(np.arange(9).reshape((3, 3)), 
                   columns=['a', 'b', 'c'],
                   index=['one', 'two', 'three'])
df2 = pd.DataFrame(np.arange(12).reshape((4, 3)), 
                   columns=['b', 'c', 'd'],
                   index=['zero', 'one', 'two', 'three'])
print df1
print df2
"""
       a  b  c
one    0  1  3
two    3  4  5
three  6  7  8
       b   c   d
zero   0   1   2
one    3   4   5
two    6   7   5
three  9  10  11
"""

df1 + df2
#  returns a DataFrame whose index and columns are the unions of the ones in each DataFrame

"""
        a   b   c   d
one     NaN 4   6   NaN
three   NaN 16  18  NaN
two     NaN 10  12  NaN
zero    NaN NaN NaN NaN
"""


# Fill missing (NaN) values with this value. 
# If both DataFrame locations are missing, the result will be missing
df1.add(df2, fill_value=0)
# a: one, two three are from df1
# d: one, two three are from df2
# zero: b, c, d are from df2
# zero: a are missing both in df1 and df2

"""
        a   b   c   d
one     0   4   6   5
three   6   16  18  11
two     3   10  12  8
zero    NaN 0   1   2
"""

# more methods
"""
Method  Description
add     Method for addition (+)
sub     Method for subtraction (-)
div     Method for division (/)
mul     Method for multiplication (*)
"""



# Methods / Functions for DataFrame

# drop
df2 = df2.drop('d', axis=1) 
print df2
# drop column 'd'
# axis = 1 means drop column

df2 = df2.drop('zero', axis=0) 
print df2
# drop row 'zero'
# axis = o means drop row

print df1.drop('a', axis = 1) + df2
# drop column 'a' and add  df2


# apply
" DataFrame’s apply method apply a function on 1D arrays to each column or row. "
df1.apply(lambda x: x) # apply nothing

df1.apply(lambda x: min(x), axis=0) 
# minimum number in each column

df1.apply(lambda x: min(x), axis = 1)
# apply function to each row

def f(x):
    return pd.Series([x.max(), x.min()], index = ['max', 'min'])
df1.apply(f)
# apply function to each column
# return the min and max



# Map
" DataFrame's map method apply a function on each element of a Series. "

df1['var1'].map(lambda x: 10 + x)
# add 10 to each element in the series



# ApplyMap
" DataFrame's ApplyMap method apply a function to each element of a dataframe "

func = lambda x: x+2
df1.applymap(func) # add 2 to each element in the df





# descriptive statistics

df1.min(axis=0) # minimum in each column
df1.min(axis=1) # minimum in each row

"""
Method  Description
count   Number of non-NA values
min, max    Compute minimum and maximum values
argmin, argmax  Compute index locations (integers) at which minimum or maximum value obtained, respectively
idxmin, idxmax  Compute index values at which minimum or maximum value obtained, respectively
quantile    Compute sample quantile ranging from 0 to 1
sum Sum of values
mean    Mean of values
median  Arithmetic median (50% quantile) of values
mad Mean absolute deviation from mean value
var Sample variance of values
std Sample standard deviation of values
skew    Sample skewness (3rd moment) of values
kurt    Sample kurtosis (4th moment) of values
"""

df1.describe()
"""
        a   b   c
count   3.0 3.0 3.0
mean    3.0 4.0 5.0
std     3.0 3.0 3.0
min     0.0 1.0 2.0
25%     1.5 2.5 3.5
50%     3.0 4.0 5.0
75%     4.5 5.5 6.5
max     6.0 7.0 8.0
"""


print tips.head()
# first 5 rows
# same as tips[0:5] 
print tips.tail()
# last 5 rows

tips[0:5] 

print tips.head(3) 
# first 3 rows
# same as tips[0:3] 


print tips['tip'] # When a value was passed, it will be treated as the column name.

# Create Dataframe
df3 = pd.DataFrame({'one': range(3),
                    'two': range(3, 6),
                    'three': range(6, 9)},
                   index = ['one', 'two', 'three'])
print df3
"""
       one  three  two
one      0      6    3
two      1      7    4
three    2      8    5
"""

"""
The .loc method provides a purely label(index/columns) based indexing.
This methods only allows you do selection from a DataFrame by its index and columns.
"""
df3['one':'three'] # rows of indices from 'one' to 'three'
df3.loc['one', ['one', 'three']]
# index: 'one'
# columns: 'one' or 'three'
# return a Series

# index: from 'one' to the last row
# columns: from 'three' to the last column
df3.loc['one':, 'three':]

"""
The .iloc method provides a purely position based indexing.
"""
# first row, first three columns
# return a Series
row1 = df3.iloc[0, :3]

# first row, first three columns
# return a DataFrame
row1 = df3.iloc[0:1, :3]
print type(row1)
row1




# boolean indexing
| # or 
& # and
~ # not
series.isin([1,2,3,4]) # if the value of a series in the list?
s = pd.Series(np.arange(5))
print s
print s.isin([2, 4, 6])

tbl[tbl.color == 'blue'] # rows where color == 'blue'

tbl[(tbl.color == 'blue') & (tbl.value.isin([1, 4]))] 
# rows where color == 'blue' and value in [1, 4]

tbl[(tbl.color == 'blue') & ~(tbl.value.isin([1, 4]))] 
# rows where color == 'blue' and value not in [1, 4]

tbl[(tbl.color == 'blue') | (tbl.value.isin([2, 6]))] 
# rows where color == 'blue' or value in [2, 6]

# loc, iloc
tbl.loc[tbl.value >=5, 'color'] # the color where value >= 5
# tbl.iloc[tbl.value >=5, 1] does not work
tbl.iloc[(tbl.value >=5).values, ]

# Here, tbl.value >= 5 return a Series, which is not supported in the iloc method.
print tbl.value >=5
print (tbl.value >= 5).values
"""
0    False
1    False
2    False
3    False
4     True
Name: value, dtype: bool
[False False False False  True]
"""



# Sorting
"""
- axis = 0: sort by index(default)
- axis = 1: sort by column names
- by = 'name': sort rows by the value of column 'name'
- ascending: sort in ascending order(default)
"""

tbl.sort_index(axis=0) # sort by index [0, 1, 2, 3, 4]


tbl.sort_index(axis=1) 
# sort by column names

tbl.sort_index(by='var') # sort by 'var'


tbl.sort_index(by='var', ascending=False) 
# sort by 'var' in descending order

tbl.sort_index(by=['color', 'value'], ascending=False) 
# sort by 'color' and then 'value' in descending order

tbl.sort_index(by=['color', 'value'], ascending=[False, True]) 
# sort by 'color'(descending) and then 'value'(ascending)



# Group and aggregation

"""
Categorizing a data set and applying a function to each group is often a critical component of a data analysis workflow.

A DataFrame can be grouped on its rows (axis=0) or its columns (axis=1).
After grouped, apply a function to each group, produce a new value(sum, for example).
At last, the results of all those steps are combined into a final result.
"""


group = df.groupby(df.key1)
group.mean()
type(group) # generate a DataFrameGroupBy object

group2 = df.groupby([df.key1, df.key2])
group2.mean()

" It also works for lists, tuples and arrays. "
years = [2005, 2005, 2006, 2005, 2006]
df.groupby(years).mean()
# group by array
df.groupby(np.array(years)).mean()  
# group by tuple
df.groupby((2005, 2005, 2006, 2005, 2006)).mean()



" The GroupBy object supports iteration, generating a sequence of 2-tuples containing the group name along with the chunk of data. "
for key, value in group:
    print 'Group:', key
    print 30 * '-'
    print value, '\n'
"""
Group: a
------------------------------
      data1     data2 key1 key2
0 -1.735067  0.406057    a  one
1 -1.020162  0.223686    a  two
4 -0.286692  0.611796    a  one 

Group: b
------------------------------
      data1     data2 key1 key2
2  0.878318  0.116686    b  one
3 -0.525571 -1.045727    b  two 
"""


for (key1, key2), value in df.groupby([df.key1, df.key2]):
    print 'Group:', key1, '&', key2
    print 30 * '-'
    print value, '\n'
"""
Group: a & one
------------------------------
      data1     data2 key1 key2
0 -1.735067  0.406057    a  one
4 -0.286692  0.611796    a  one 

Group: a & two
------------------------------
      data1     data2 key1 key2
1 -1.020162  0.223686    a  two 

Group: b & one
------------------------------
      data1     data2 key1 key2
2  0.878318  0.116686    b  one 

Group: b & two
------------------------------
      data1     data2 key1 key2
3 -0.525571 -1.045727    b  two 
"""

list(df.groupby('key1')) # Convert a Groupby object to list.
dict(list(df.groupby('key1'))) # Convert a Groupby object to dictionary.

df.groupby(['key1', 'key2']).mean()


# The Groupby object also support column indexing.
data1Mean =  df.groupby(['key1', 'key2'])[['data1']].mean()
# double [] result into a DataFrame
print type(data1Mean)
print data1Mean




# Grouping with Dict and Series¶

print people # dataframe
"""
               a         b         c         d         e
Joe    -1.857129  0.119589  0.971842 -1.640040  0.723110
Steve   1.843463 -0.200584 -0.001003 -0.167259  0.933656
Wes    -1.671633 -1.282425  0.670216 -0.159577  0.326869
Jim    -0.256390 -1.463713  0.006516  1.179782 -0.276399
Travis -1.503408  0.590367  0.984099 -0.314719  1.414260
"""

mapping = {'a': 'red', 'b': 'red', 'c': 'blue', 
           'd': 'blue', 'e': 'red'}
people.groupby(mapping, axis=1).mean()
# group by the values of mapping
# axis = 1: group by columns

"""
        blue        red
Joe     -0.334099   -0.338143
Steve   -0.084131   0.858845
Wes     0.255319    -0.875730
Jim     0.593149    -0.665501
Travis  0.334690    0.167073
"""

# in default, it's group by rows
mapping = {'Joe': 'red', 'Steve': 'red', 'Wes': 'blue', 'Travis': 'blue', 'Jim': 'red'}
people.groupby(mapping).mean()

"""
        a           b           c           d           e
blue    -1.587520   -0.346029   0.827157    -0.237148   0.870564
red     -0.090019   -0.514903   0.325785    -0.209172   0.460122
"""

# the same functionality holds for Series
mapSeries = pd.Series(mapping)
people.groupby(mapSeries).mean()

"""
        a           b           c           d           e
blue    -1.587520   -0.346029   0.827157    -0.237148   0.870564
red     -0.090019   -0.514903   0.325785    -0.209172   0.460122
"""

# Aggregation
# There is a collection of functions for Groupby object.
Function    Description
size        Number of factors
count       Number of non-NA values in the group
sum         Sum of non-NA values
mean        Mean of non-NA values
median      Arithmetic median of non-NA values
std, var    Unbiased (n - 1 denominator) standard deviation and variance
min, max    Minimum and maximum of non-NA values
prod        Product of non-NA values
first, last First and last non-NA values

print tips
"""
   total_bill   tip     sex smoker  day    time  size
0       16.99  1.01  Female     No  Sun  Dinner     2
1       10.34  1.66    Male     No  Sun  Dinner     3
2       21.01  3.50    Male     No  Sun  Dinner     3
3       23.68  3.31    Male     No  Sun  Dinner     2
4       24.59  3.61  Female     No  Sun  Dinner     4
"""

tips.groupby('sex').size()

tips.groupby('sex').count()
# count will counts for all the columns

tips.groupby('sex')['tip'].mean()

tips.groupby('sex')['tip'].max()

# the same
# sort and return the first one
tips = tips.sort_index(by = ['tip'], ascending = False)
tips.groupby('sex').first() 
tips.groupby('sex').head()
# - we can also use 'apply' fun in grouping
def stat(x):
    return pd.Series({'count': len(x), 'sum': np.sum(x), # - define function first
                      'min': np.min(x), 'max': np.max(x),
                      'mean': np.mean(x), 'std': np.std(x)})
tips.groupby('sex')['tip'].apply(stat) # apply fun 

# - // same as above with .agg()
# multiple functions at once
# just give a list of functions
tips.groupby('sex')['tip'].agg(['count', 'sum', 'min', 'max', 'mean', 'std'])
# simpler than tips.groupby('sex')['tip'].apply(stat)
# in which you should define a function at first
"""
        count       sum min max mean    std
sex                     
Female  87  246.51  1   6.5 2.833448    1.159495
Male    157 485.07  1   10.0    3.089618    1.489102
"""
tips.groupby('sex')['tip'].agg(['count', 'sum', 'min', 'max', 'mean', \
                                ('StandardDivation', 'std')])
# replace the name 'std' with 'StandardDivation'
"""
    count   sum min max mean    StandardDivation
sex                     
Female  87  246.51  1   6.5 2.833448    1.159495
Male    157 485.07  1   10.0    3.089618    1.489102
"""


# >> To implement different functions on different columns, you need to pass a dictionary.
colFun = {'tip': ['mean', 'std', 'max'],
          'total_bill': ['mean'],
          'size': ['sum', 'mean']}
tips.groupby('sex').agg(colFun)
"""
        total_bill  tip                             size
        mean        mean        std         max     sum mean
sex                     
Female  18.056897   2.833448    1.159495    6.5     214 2.459770
Male    20.744076   3.089618    1.489102    10.0    413 2.630573
"""
# ** In default, after aggregation the group keys will come back as index.
#    Set as_index = False to convert the keys to columns.
tips.groupby('sex', as_index=False).agg(colFun)


# The transform method applies a function to each group, then places the results in the appropriate locations.
tips['mean>3.1'] = tips.groupby(['sex', 'smoker'])['tip'].transform(lambda x: np.mean(x) >= 3.1)
# add a column 'mean>3.1'
# sex(Male),smoker(No) --> 1(True)
# others ----------------> 0(False)
print tips[['sex', 'smoker', 'tip', 'mean>3.1']].head(10)
"""
        sex smoker    tip  mean>3.1
170    Male    Yes  10.00         0
212    Male     No   9.00         1
23     Male     No   7.58         1
59     Male     No   6.73         1
141    Male     No   6.70         1
214  Female    Yes   6.50         0
183    Male    Yes   6.50         0
47     Male     No   6.00         1
239    Male     No   5.92         1
88     Male     No   5.85         1
"""

# drop the column
tips = tips.drop('mean>3.1', axis = 1)






# >> Pivot Tables and Cross-Tabulation
"""
Function    Description
values      Column name or names to aggregate. By default aggregates all numeric columns
index       Column names or other group keys to group on the rows of the resulting pivot table
columns     Column names or other group keys to group on the columns of the resulting pivot table
aggfunc     Aggregation function or list of functions; 'mean' by default. Can be any function valid in a groupby context
fill_value  Replace missing values in result table
margins Add row/column subtotals and grand total, False by default
"""

print tips
"""
   total_bill   tip     sex smoker  day    time  size
0       16.99  1.01  Female     No  Sun  Dinner     2
1       10.34  1.66    Male     No  Sun  Dinner     3
2       21.01  3.50    Male     No  Sun  Dinner     3
3       23.68  3.31    Male     No  Sun  Dinner     2
4       24.59  3.61  Female     No  Sun  Dinner     4
"""

tips.pivot_table(index=['sex', 'smoker']) 
#  By default aggregates all numeric columns
"""
               size        tip         total_bill
sex    smoker          
Female  No     2.592593    2.773519    18.105185
        Yes    2.242424    2.931515    17.977879
Male    No     2.711340    3.113402    19.791237
        Yes    2.500000    3.051167    22.284500
"""


tips.pivot_table(index=['sex', 'smoker'], columns='day', values='tip',  margins=True)
"""
        day     Fri    Sat    Sun    Thur    All
sex     smoker                  
Female  No  3.125000    2.724615    3.329286    2.459600    2.773519
        Yes 2.682857    2.868667    3.500000    2.990000    2.931515
Male    No  2.500000    3.256563    3.115349    2.941500    3.113402
        Yes 2.741250    2.879259    3.521333    3.058000    3.051167
All         2.734737    2.993103    3.255132    2.771452    2.998279
"""



tips.pivot_table(index=['sex', 'smoker'], 
                 columns='day', values='tip',
                 aggfunc = 'describe')
"""
            day     Fri         Sat         Sun         Thur
sex     smoker                  
Female  No  count   2.000000    13.000000   14.000000   25.000000
            mean    3.125000    2.724615    3.329286    2.459600
            std     0.176777    0.961904    1.282356    1.078369
            min     3.000000    1.000000    1.010000    1.250000
            25%     3.062500    2.230000    2.602500    1.680000
            50%     3.125000    2.750000    3.500000    2.000000
            75%     3.187500    3.000000    3.937500    2.920000
            max     3.250000    4.670000    5.200000    5.170000
        Yes count   7.000000    15.000000   4.000000    7.000000
            mean    2.682857    2.868667    3.500000    2.990000
            std     1.058013    1.461378    0.408248    1.204049
            min     1.000000    1.000000    3.000000    2.000000
            25%     2.250000    2.000000    3.375000    2.005000
            50%     2.500000    2.500000    3.500000    2.500000
            75%     3.240000    3.310000    3.625000    3.710000
            max     4.300000    6.500000    4.000000    5.000000
Male    No  count   2.000000    32.000000   43.000000   20.000000
            mean    2.500000    3.256563    3.115349    2.941500
            std     1.414214    1.839749    1.216401    1.485623
            min     1.500000    1.250000    1.320000    1.440000
            25%     2.000000    2.000000    2.000000    2.000000
            50%     2.500000    2.860000    3.000000    2.405000
            75%     3.000000    3.640000    3.815000    3.550000
            max     3.500000    9.000000    6.000000    6.700000
        Yes count   8.000000    27.000000   15.000000   10.000000
            mean    2.741250    2.879259    3.521333    3.058000
            std     1.166808    1.744338    1.417432    1.111573
            min     1.500000    1.000000    1.500000    2.000000
            25%     1.835000    1.990000    2.500000    2.005000
            50%     2.600000    3.000000    3.500000    2.780000
            75%     3.250000    3.185000    4.000000    4.000000
            max     4.730000    10.000000   6.500000    5.000000
"""



# Cross-Tabulations: Crosstab
# A cross-tabulation (or crosstab for short) is a special case of a pivot table that computes group frequencies.
# It can be computed by the function pandas.crosstab function.
pd.crosstab(tips.sex, tips.smoker)
"""
smoker  No  Yes
sex     
Female  54  33
Male    97  60
"""
# equivalent
tips.pivot_table(index='sex', columns='smoker', values= 'tip', aggfunc='count')
"""
smoker  No  Yes
sex     
Female  54  33
Male    97  60
"""
# equivalent
tips.groupby(['sex', 'smoker']).size()
"""
smoker  No  Yes
sex     
Female  54  33
Male    97  60
"""
# The function crosstab also has arguments rows, cols, and aggfunc, etc. Run help(pd.crosstab) to see the document.



# >> Combine & Merge
"""
Data contained in pandas objects can be combined together in a number of built-in ways:

pandas.concat glues or stacks together objects along an axis.
pandas.merge connects rows in DataFrame objects based on one or more keys. 
This will be familiar to users of SQL or other relational databases, as it implements database join operations.
"""

# concat
print people
"""
               a         b         c         d         e
Joe    -1.857129  0.119589  0.971842 -1.640040  0.723110
Steve   1.843463 -0.200584 -0.001003 -0.167259  0.933656
Wes    -1.671633 -1.282425  0.670216 -0.159577  0.326869
Jim    -0.256390 -1.463713  0.006516  1.179782 -0.276399
Travis -1.503408  0.590367  0.984099 -0.314719  1.414260
"""

people2 = pd.DataFrame({'John': np.random.randn(5)},
                        index = ['a', 'b', 'c', 'd', 'e']) # create a new data for concatenate
print people2
print people2.T
"""
       John
a  0.761151
b  1.198944
c  0.574395
d -0.832043
e  1.449887
"""

"""
             a         b         c         d         e
John  0.761151  1.198944  0.574395 -0.832043  1.449887
"""

## Concatenate people and people2 together by rows
peoples = pd.concat([people, people2.T], axis = 0)
print peoples
"""
               a         b         c         d         e
Joe    -1.857129  0.119589  0.971842 -1.640040  0.723110
Steve   1.843463 -0.200584 -0.001003 -0.167259  0.933656
Wes    -1.671633 -1.282425  0.670216 -0.159577  0.326869
Jim    -0.256390 -1.463713  0.006516  1.179782 -0.276399
Travis -1.503408  0.590367  0.984099 -0.314719  1.414260
John    0.761151  1.198944  0.574395 -0.832043  1.449887 > vy raw default 0
"""



fg = pd.DataFrame(np.random.randn(6, 2), 
                   index = peoples.index,
                   columns = ['f', 'g']) # create another data for joining by column
print fg
"""
               f         g
Joe    -0.821135  0.240469
Steve   1.357890  0.397905
Wes    -0.422386  0.921984
Jim    -0.192974  0.701869
Travis -0.352909  0.250238
John   -0.996313  0.562728
"""


## Concatenate peoples and fg together by columns
peoples = pd.concat([peoples, fg], axis = 1)
print peoples

"""
               a         b         c         d         e         f         g
Joe    -1.857129  0.119589  0.971842 -1.640040  0.723110 -0.821135  0.240469
Steve   1.843463 -0.200584 -0.001003 -0.167259  0.933656  1.357890  0.397905
Wes    -1.671633 -1.282425  0.670216 -0.159577  0.326869 -0.422386  0.921984
Jim    -0.256390 -1.463713  0.006516  1.179782 -0.276399 -0.192974  0.701869
Travis -1.503408  0.590367  0.984099 -0.314719  1.414260 -0.352909  0.250238
John    0.761151  1.198944  0.574395 -0.832043  1.449887 -0.996313  0.562728
"""



# Merge
# merge merge DataFrame objects by performing a database-style join operation by columns or indexes.

"""
Basiclly, there are four kinds of merges:
left: use only keys from left frame (SQL: left outer join)
right: use only keys from right frame (SQL: right outer join)
outer: use union of keys from both frames (SQL: full outer join)
inner: use intersection of keys from both frames (SQL: inner join)
"""

pd.merge(data1, data2, how='outer', # 'inner', 'left', 'right'
               left_on='surname', right_on='name')
help(pd.merge) # more information


































































































































































