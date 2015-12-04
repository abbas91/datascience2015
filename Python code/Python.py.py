print "I will be getting: ", (100*2-23%3)/5
print "yes? -> ", 100 >= 5, 80<4
print 100/3, 100.00/3.00
print 100%3

# you = 100*78 # define you
# me = 19*87/2 # define me
# he = " is not equal "
# print you, he, me
# print you >= me
# print "Hey %s there" % "you"

# score_A = 400*4*3/25%3
# score_B = 600*8/43
# me = "Mark"
# you = "Jason"
# her = "July"
# print "%s is calling %s for %s %d times already. The last time was %d times!.." % (me, you, her, score_A, score_A*score_B)
# print score_A, score_B

# print "yes you are %s" % "mine"
# print "." * 10
# a = "a"
# b = "b"
# c = "c"
# print a + b + c
# print a + c + b


# kkk = "%s %s %s is %s"
# a = (1,2,3,4)
# b = ("s", "d", "j", "o")
# c = (kkk,kkk,kkk,kkk)
# print kkk % a
# print kkk % b
# print kkk % c

# sd = "a\nb\nc\nd"
# print "is is new line: %r" % sd

# print """ asdnfas;dkfaskdm
# sdkfnaksdn;fkasd
# asdlfmasdmf;lasd 
# """

# print "yes we \"do\" it"
# print "as\adf\rfd"
# print "\\"
# print "\'"
# print "\""
# ##print "\a"
# print "\b"
# print "\f"
# print "\n"
# ## print "\N{name}"
# print "\r"
# print "\t"
# ##print "\uxxxx"
# ##print "\Uxxxxxxxxx"
# print "\v"
# ##print "\ooo"
# ##print "\xhh"


# print "\aHow tired you are?",
# tire_degree = int(raw_input())*100
# print "\aYou want to save and quit?",
# Do_you = raw_input()
# print "\aAre you sure?",
# Again = raw_input()
# print "You are %s times tired. \n%s is whether you want to quit. \nWhen I ask you sure? You said \"%s\"." % (tire_degree, Do_you, Again)


# # help?
# linux $pydoc raw_input
# windows $python -m pydoc raw_input
# q # exit

# # From keyboard
# a = raw_input("here yo go, right? ")
# b = raw_input("Again? No? ")
# print "You said %s and then you said %s." % (a, b)

# # From command line - argv (define variables)
# from sys import argv
# scrpit, a, b, c = argv
# print "scrpit is:", scrpit
# print "a is:", a
# print "b is:", b
# print "c is:", c

# # combine argv & raw_input
# from sys import argv
# script, username = argv
# promt = ">> "
# print "Hi %s, how are you? I am the %s script.:)" % (username, script)
# print "I'd like to ask you a few question."
# print "Do you feel bad today %s?" % username
# feel = raw_input(promt)
# print "Ok. Things will change. What you gonna do?"
# do = raw_input(promt)
# print "I see. Good luck with it! You will fine %s. When is it?" % username
# when = raw_input(promt)
# print """
# Ok. %s, even though you feel %s today. 
# You will be fine. Also, eventually, you had a plan that %s on %s.
# Best luck for you! I am script %s.""" % (username, feel, do, when, script)

# # loading text data from cript and print it
# from sys import argv
# script, username, filename = argv
# txt = open(filename)
# print "Hi %s, welcome! Here is your file: %s." % (username, filename)
# print txt.read() # -> give file a command use "."
# print "%s, want to open more file? Please tell me the filename." % username
# file_again = raw_input("> ")
# txt_again = open(file_again)
# print txt_again.read() # -> give file a command use "."
# print "Goodbye %s!" % username

# # Read file, delete content and write new content to a file
# from sys import argv
# script, username, filename = argv
# print "Hi %s, we are going to erase the file: %s." % (username, filename)
# print "If you don't want to, hit CTRL-C"
# print "If you do want that, hit return."
# raw_input("?")
# print "Opening the file..."
# target = open(filename, "w")
# print "Truncating the file..."
# target.truncate()
# print "Now, write 3 lines.."
# line1 = raw_input("line1: ")
# line2 = raw_input("line2: ")
# line3 = raw_input("line3: ")
# print "Writing them to the file..."
# target.write(line1)
# target.write("\n")
# target.write(line2)
# target.write("\n")
# target.write(line3)
# print "We can close it now"
# target.close()

# # Define function
# def print_two(*args):
# 	arg1, arg2 = args # just like argv
# 	print "arg1: %r, arg2: %r" % (arg1, arg2)
# def print_two_again(arg1, arg2): # -> best way
# 	print "arg1: %r, arg2: %r" % (arg1, arg2)
# def print_one(arg1):
# 	print "arg1: %r" % arg1
# def print_none():
# 	print "I got none."
# print_two("ss", "gg")
# print_two_again("ss", "gg")
# print_one("ll")
# print_none()

# # Define variables infunction
# def we_want_sth(arg1, arg2, arg3):
# 	print "we got %d in days!" % arg1
# 	print "\nwe got %d in mothes!" % arg2
# 	print "\nwe got %d in years!" % arg3
# print "lets do insert numbers.."
# we_want_sth(10,30,50)
# print "lets do define vars.."
# a = 10; b = 20; c = 40
# we_want_sth(a, b, c)
# print "lets do a bit math.."
# we_want_sth(a+10, b*6, c/10)
# print "Cool, done!"

# # Define function to process file
# from sys import argv
# script, input_file = argv
# def print_all(f):
# 	print f.read()
# def rewind(f):
# 	f.seek(0)
# def print_a_line(line_count, f):
#     print line_count, f.readline()
# current_file = open(input_file)
# print "First let's print the whole file:\n"
# print_all(current_file)
# print "Now let's rewind, kind of like a tape."
# rewind(current_file)
# print "Let's print three lines:"
# current_line = 1
# print_a_line(current_line, current_file)
# current_line = current_line + 1
# print_a_line(current_line, current_file)
# current_line = current_line + 1
# print_a_line(current_line, current_file)

# # Vraiables can be defined to return function value
# def add(a, b):
# 	print "Adding %d with %d" % (a, b)
# 	return a + b
# def sub(a, b):
# 	print "Substructing %d with %d" % (a, b)
# 	return a - b
# def multi(a, b):
# 	print "Multiplying %d with %d" % (a, b)
# 	return a * b
# print "let's create some fun values.."
# xx = add(10, 80)
# zz = sub(10, 80)
# dd = multi(10, 80)
# print "1 is %d, 2 is %d, 3 is %d." % (xx, zz, dd)
# print "More .. if want see hit return.."
# yes = raw_input("Yes?")
# what = add(xx, sub(dd, multi(zz, 20)))
# print "you get %d from puzzle." % what


# # Define function in file.py then import it in python to use
# def add(a, b, c):
# 	print "adding %d, %d, and %d together:" % (a, b, c)
# 	return a + b + c
# def super_add(a, b, c, d):
# 	print "adding %d, %d, and %d together:" % (a, b, c)
# 	print "Then, divided by %d." % d
# 	return (a + b + c) / d
# $python
# >>> import filename
# >>> filename.add(10, 20, 30)
# >>> filename.super_add(10, 20, 30, 5)
# or
# >>> from filename import *
# >>> add(10, 20, 30)
# >>> super_add(10, 20, 30, 5)





# # Error fixing ----------------- Quiz 1 #
# def break_words(stuff):
#     """This function will break up words for us."""
#     words = stuff.split(' ')
#     return words

# def sort_words(words):
#     """Sorts the words."""
#     return sorted(words)

# def print_first_word(words):
#     """Prints the first word after popping it off."""
#     word = words.poop(0)
#     print word

# def print_last_word(words):
#     """Prints the last word after popping it off."""
#     word = words.pop(-1)
#     print word

# def sort_sentence(sentence):
#     """Takes in a full sentence and returns the sorted words."""
#     words = break_words(sentence)
#     return sort_words(words)

# def print_first_and_last(sentence):
#     """Prints the first and last words of the sentence."""
#     words = break_words(sentence)
#     print_first_word(words)
#     print_last_word(words)

# def print_first_and_last_sorted(sentence):
#     """Sorts the words then prints the first and last one."""
#     words = sort_sentence(sentence)
#     print_first_word(words)
#     print_last_word(words)

# print "Let's practice everything."
# print 'You\'d need to know \'bout escapes with \\ that do \n newlines and \t tabs.'

# poem = """
# \tThe lovely world
# with logic so firmly planted
# cannot discern \n the needs of love
# nor comprehend passion from intuition
# and requires an explantion
# \n\t\twhere there is none.
# """

# print "--------------"
# print poem
# print "--------------"

# five = 10 - 2 + 3 - 5
# print "This should be five: %s" % five

# def secret_formula(started):
#     jelly_beans = started * 500
#     jars = jelly_beans / 1000
#     crates = jars / 100
#     return jelly_beans, jars, crates

# start_point = 10000
# beans, jars, crates = secret_formula(start_point)

# print "With a starting point of: %d" % start_point
# print "We'd have %d jeans, %d jars, and %d crates." % (beans, jars, crates)

# start_point = start_point / 10

# print "We can also do that this way:"
# print "We'd have %d beans, %d jars, and %d crabapples." % secret_formula(start_point)

# sentence = "All god\tthings come to those who weight."
# words = break_words(sentence)
# sorted_words = sort_words(words)
# print_first_word(words)
# print_last_word(words)
# print_first_word(sorted_words)
# print_last_word(sorted_words)
# sorted_words = sort_sentence(sentence)
# print sorted_words
# print_irst_and_last(sentence)
# print_first_a_last_sorted(senence)

# # ----------------------------- End ------------------------------- #


# # Logic 
# print 3 <= 5
# print 4 >= 2
# print 4 > 10
# print True and True
# print False and True
# print 1 == 1 and 2 == 1
# print "test" == "test"
# print 1 == 1 or 2 != 1
# print True and 1 == 1
# print False and 0 != 0
# print True or 1 == 1
# print "test" == "testing"
# print 1 != 0 and 2 == 1
# print "test" != "testing"
# print "test" == 1
# print not (True and False)
# print not (1 == 1 and 0 != 1)
# print not (10 == 1 or 1000 == 1000)
# print not (1 != 10 or 3 == 4)
# print not ("testing" == "testing" and "Zed" == "Cool Guy")
# print 1 == 1 and not ("testing" == 1 or 1 == 0)
# print "chunky" == "bacon" and not (3 == 4 or 3 == 3)
# print 3 == 3 and not ("testing" == "testing" or "Python" == "Fun")


# What - if
# mark = 50
# jason = 40
# her = 80
# if mark > jason:
# 	print "Opps ~"
# if mark < jason:
# 	print "Oh no"
# her -= 20; her += 20
# if her > mark:
# 	print "It is~"
# if her < mark:
# 	print "ok ok"
# if her >= mark:
# 	print "ll"


# # If else logic
# mark = 100; jason = 80; her = 100
# if mark > jason:
# 	print "mark > jason"
# elif mark < jason:
# 	print "Soon.."
# else:
# 	print "Ok lets wait.."
# # -
# if her > mark:
# 	print "her > mark"
# else:
# 	print "not > than mark.. but"



# # Make decisions - multiple if-seld
# print "game starts: choose 1 or 2?"
# choose = raw_input("> ")
# if choose == "1":
# 	print "make another choose: 1 or 2?"
# 	choose_2 = raw_input("> ")
# 	if choose_2 == "1":
# 		print "You die.."
# 	else:
# 		print "you survived. Good job!"
# elif choose == "2":
# 	print "Good. Then choose 2 or 3 next?"
# 	choose_3 = raw_input("> ")
# 	if choose_3 == "2":
# 		print "Oh.. you die"
# 	else:
# 		print "Oh.. you die"
# else:
# 	print "You didn't choose right.. die instantly..;("


# # for Loops and Lists (Only based on object length)
# people = ["mark", "jason", "her"]
# count = [10, 5, 7]
# mix = [10, "mark", 50, "jason", 90, "her"]
# mix_2_dim = [[1,2,3,4],[5,6,7,8]]
# for xx in people:
# 	print "This is %s" % xx
# for i in mix:
# 	print "it is %r" % i # using "%r" cause don't know what it is in mix
# #create emplty list first
# list_A = []
# for i in (10, 20, 30):
# 	print "%d is it" % i
# 	list_A.append(i) # start append it to list
# for i in list_A:
# 	print "%r" % i


# # while - loops (Keep repeating)
# i = 0
# number = []
# while i <= 10: # stoping cirtiria
# 	print "At the top i is %d" % i
# 	number.append(i)
# 	i += 1
# 	print "Numbers now is: ", number
# print "Total is: "
# for x in number:
# 	print x
# #Hold CTRL-c to abort process if go crazy


# # Access elements of lists
# animals = ["k", "d", "l", "o", "r", "p"]
# print animals[0], animals[2], animals[3]
# print [animals[i] for i in (0,3,2)]


# # Connect multiple functions
# from sys import exit, argv

# script, username = argv

# def choose1_B1_B2():
# 	print "Choose Boy1 or Boy2? %d" % username
# 	choose = raw_input("> ")
# 	if choose == "Boy1":
# 		print "How much you would like to spend?"
# 		spend = raw_input("> ")
# 		if spend > 100:
# 			print "Great! You made him stay."
# 		else:
# 			print "Oh oh no he ran off.."
# 			print "Ok, do you want to go back to Boy2 or find Boy3, %s" % username
# 			choose2 = raw_input("> ")
# 			if choose2 == "Boy2"
# 			print "Ok lets see.."

# 			elif choose2 == "Boy3"
# 			exit(0)
# 	if choose == 







































