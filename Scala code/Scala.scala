/* - Scala - */

/* login interperter on command line */ 
$scala
scala> :help
"""
All commands can be abbreviated, e.g., :he instead of :help.
:edit <id>|<line>        edit history
:help [command]          print this summary or command-specific help
:history [num]           show the history (optional num is commands to show)
:h? <string>             search the history
:imports [name name ...] show import history, identifying sources of names
:implicits [-v]          show the implicits in scope
:javap <path|class>      disassemble a file or class name
:line <id>|<line>        place line(s) at the end of history
:load <path>             interpret lines in a file
:paste [-raw] [path]     enter paste mode or paste a file
:power                   enable power user mode
:quit                    exit the interpreter
:replay [options]        reset the repl and replay all previous commands
:require <path>          add a jar to the classpath
:reset [options]         reset the repl to its initial state, forgetting all session entries
:save <path>             save replayable session to a file
:sh <command line>       run a shell command (result is implicitly => List[String])
:settings <options>      update compiler options, if possible; see reset
:silent                  disable/enable automatic printing of results
:type [-v] <expr>        display the type of an expression without evaluating it
:kind [-v] <expr>        display the kind of expression's type
:warnings                show the suppressed warnings from the most recent line which had any
"""

/* [1] Interpreter */

/* ...block */
// ...one line

/* automatically assign a object */
scala> 1 + 2
res0:  Int = 3
scala> res0 * 3
res1: Int = 9

/* Print to console */
scala> printIn("Hello, world!!") 
"Hello, world!!"



/* [2] Define some variables */
scala> val msg: String = "Hello, world !!" /* Inmmutable */
scala> var msg: String = "Hello, world !!" /* mutable (Yet, must be the same type)*/
scala> msg = "new, strings !!" /* re-assign var */
scala> var num: Int = 677 
scala> printIn(msg)
"Hello, world !!"



/* [3] Define some functions */
scala> def max(x: Int = 87, y:Int): Int = {
	exp...x...y...
	return ... /* optional - otherwise use the last line */
}


/* If statement */ 
scala> if (x>y) x else y


/* function takes no parameters, only side effect */
scala> def greet() = printIn("Hello, world!")
greet: ()Unit /* Unit type object */





/* [4] Write some Scala scripts */
/* Run scala script on CLI */
$scala hello.scala 
" Hello, world!!"


/* takes argus into the script */
// -------- Script ------------ //
printIn("Hello, " + args(0) + "!") // argus index starts at '0' 
// ---------------------------- //
$scala hello.scale mark
" Hello, mark!"








/* [5] Loop with While + if */
var i = 0 /* starting term */
while (i < 100) { /* looping while ture */
	printIn(i)
	i += 1 /* update term */
}



/* for loop, foreach loop */
args.foreach((arg: String) => printIn(arg)) // foreach loop

for (arg <- args) // for loop
  printIn(arg)

for (i <- 0 to 10) // for loop - if method takes only one parameter, can call it without . or () -> (0).to(10)
  printIn(i)
// scala doesn't technically has operator overloading
1 + 1 >> (1).+(1)
1 -> "s" >> 1.->("s") // Returns a tuple (1, "s")


/* Function literal */
(x: Int, y: Int) => x + y
// (parameters) 'right arrow' (Function body)


/* [1] Array - mutable 0-based */ 
/* Old way to iniatialze arrary */
val greetStrings: Array[String] = new Array[String](3) // index (#)

greetString(0) = "Hello"
greetString(1) = ", "
greetString(2) = "World!\n"

for (i <- 0 to 10)
  printIn(greetStrings(i))

/* New way */
val greetStrings = Array("A", "B", "C") //Array[String]



/* [2] List - Immutable */
val numml = List(1, 2, 3) //List[Int]
val nummll = List(3, 4, 5)
/* concatenation lists */
val allnum = numml ::: nummll
" List(1, 2, 3, 3, 4, 5)"
/* prepend a element */
val allnum = 7 :: numml
val allnum = 7 :: 1 :: 2 :: 3 :: Nil // Nil or List() (empty list) - initaialize a new list
" List(7, 1, 2, 3)"

// index
allnum(2)
"2"


// count the string has length 4
allnum.count(s => s.length == 4)


// show T/F a value presents
allnum.exists(s => s == "value")


// filter data (only length 4)
allnum.filter(s => s.length == 4)


// evaluate all elements -> return one T/F
allnum.forall(s => s.endsWith("l"))


// evaluate each elements -> return each
allnum.foreach(s => print(s))


// Return the first/last
allnum.head
allnum.last

// Return the all but last
allnum.init

// whether the list is empty
allnum.isEmpty

// length of a list
allnum.length


// apply a fun to each elements
allnum.map(s => s + "y")



// make one string with all elements in a list
allnum.mkString(", ") // in middle of each element



// remove elements by a filter
allnum.remove(s => s.length == 4)


// reverse a list
allnum.reverse



// drop from left (2) elements
allnum.drop(2)


// drop from right (2) elements
allnum.dropRight(2)


// List without first element
allnum.tail





/* [3] Tuples - immutable (contain different type) 1 -based*/
val tutu = (99, "string", 8.0) //Tuple3[Int, String, Float] -> 'Tuple, 3 elements, [type_of_each]'
tutu._1 //index
tutu._2




/* Sets - default immutable*/
import scala.collection.immutable.Set // Scala trait (class)

val moveieSet = Set("Hitch", "Poltergeist")
movieSet += "Shrek" // Create a new set = re-assign
printIn(movieSet)
" Set(Hitch, Poltergeist, Shrek) "

//
import scala.collection.mutable.Set // Scala trait (class)

val moveieSet = Set("Hitch", "Poltergeist")
movieSet += "Shrek" // Append the new elements 
printIn(movieSet)
" Set(Hitch, Poltergeist, Shrek) "




/* Maps - default immutable*/
/* old way */
import scala.collection.immutable.Map

val hill = Map[Int, String]() // format [key, value] '()' empty Map
hill += (1 -> "string1") // 1.->(string1) >> (1, "string1") "Tuple"
hill += (2 -> "string2")
hill += (3 -> "string3")

hill(2) // index access

/* new way */
import scala.collection.immutable.Map

val hill = Map(
   1 -> "string1", 2 -> "string2", 3 -> "string3"
	)

hill(2)

//
import scala.collection.mutable.Map

val hill = Map(
   1 -> "string1", 2 -> "string2", 3 -> "string3"
	)

hill(2)




/* Functional programming vs Imperative programming */

// Imperative programming
def printArgs(args: Arrary[String]): Unit = {
	var i = 0
	while (i < args.length) {
		printIn(args(i))
		i += 1
	}
}
" using vars, side effect - unit()"


// Functional programming
def formatArgs(args: Arrary[String]) = args.mkString("\n")
printIn(formatArgs(args))
" using val, input -> fun -> output"






/* readlines from a file */
// ------------- read.scala ----------------- //
import scala.io.source //pkg for reading file by lines

if (args.length > 0) {

	for (line <- Source.FromFile(args(0)).getLines()) // .getLines() -> returns a Iterator[String] providing one line at each iteration
	  printIn(line.length + " " + line)
}
else
    Console.err.printIn("Please enter filename")
// ------------------------------------------- //

$ read.scala file










/* CLASS / OBJECT */

" CLASS -> is the blue print to an object "
" Once you create an CLASS, you can initiate an object by the CLASS with 'new' "


/* Create a Class */
class my_class { // can also take parameters my_class(my=887)

	// fields
	var my = 90 // var
	val my2 = 80 // val
	private var my3 = 100 // private (Non-accessible from outside)


    // methods - computation with those fields
	def add(b: Byte): Unit = { // b will be a val here
		my += b                // so, assign b = # within fun will not work
	}

    def add2(f: Byte): {my += b} // same as above to return 'Unit()', only : {exp...}, no '=' / Unit() lost all result value


	def sub(s: Int): Int = {
		s * my2
	}
}


// Getting value
val acc = new my_class // Defining a class object
acc.my2 = 80
acc.my3 // won't compile (private)

acc.add(b) // function
//or
import my_class.add
add(b)





/* Semicolon inference */
val s = "Hello"; printIn(s)

//or operator ends
x
+ y // Not working

x +
y // Working





/* Singleton Object */
" inheriate all attributes (include private) from previous Class / same name object as class "
object my_class { // Can't take parameters

	private val my4 = 45 // Object's own val

    def med(acc.my3): Int = { // can access master class' private val

    	val dd = acc.sub(acc.my3)
    	dd * 43

    }

}


// Standalone object
object My_other_class { // Not the same name as other CLASS
	...
}











/* Run A Scala Application */
$ touch Summer.scala
// -------------- Summer.scala ----------------- //
import my_class.add

object Summer {
	def main(args: Arrary[String]) {
		for (arg <- args)
		  printIn(arg +": "+ add(arg))
	}
}
// --------------------------------------------- //

$ touch my_class.scala
// -------------- my_class.scala --------------- //
class my_class { //main class
  ...
}

object my_class { //object
  ...
}
// ---------------------------------------------- //


// Compile App
$ scalac my_class.scala Summer.scala // -> Summer (last)
//or
$ fsc my_class.scala Summer.scala // -> Summer (last)

// Use the APP
$ scala Summer arg1 arg2
" ... "




// Writing Application trait // ??
$ touch Summer2.scala
// -------------- Summer2.scala ----------------- //
import my_class.add

object Summer2 extends Application {
	for (season -> List("summer", "winter", "fall"))
	   printIn(season +": "+ add(season))
}
// --------------------------------------------- //

" limts: can't multi-thead JVM, can't use cli args"










/* Basic Type / Operations */
// data type
"""
Int
Long 
String
Short
Byte
Float
Double
Char
Boolean
"""

//Literals
" write a constant value directly in code "

//P - 74 - 80 ???



//Operators are methods
" Operator Notation "
1 + 2 >> 1.+(2)
S index0f 'o' >> s.index0f(o)


infix operation Notation 1 + 2
pre-fix -1
post-fix 7 Tolong

// P - 81



//Arithmatic Operations
+ - * / %

// P - 84



// Relational / Logical Operations
< > >= <= == != && ||

// P - 85



// Bitwise Operations

???

// P - 87


// Object equality

NULL == 78
False // can test NULL

// P - 88



// Operator Precedence
* / %
+ -
:
= !
<>
&
^
|
(all letters)
(all assignment operators)









/* Functional Objects */

" object that does not have mutable state - use val"
" usually has mutable alternative calss "

// Object-oriented programming
"Class parameters" // n, d
"Constructors"
"methods"
"fields" // numer, denom
"operators"
"private members"
"overriding"
"checking preconditions"
"overloading"
"self references"


// ------- Rational Class --------- //
class Rational(n: Int, d: Int) {

	require(d != 0) // Checking preconditions ( d must not 0 )

	private val g = gcd(n.abs, d.abs) // Define a private field for internal use / can use internal def (Private)
	val numer = n / g // Define local fields, can be get by name.field, other wise only avaiable local in class, name.n (X)
	val denom = d / g

	def this(n: Int) = this(n, 1) // Auxiliary constructors (must invoke another constructor of same class
		                          // as its first action 'this()', or another auxiliary constructor -> evenutally call
		                          // the primary constructor - back to lower )

    def + (that: Rational): Rational =               // overloading a def: define the same fun with different args
       new Rational(                                 //
         numer * that.denom + that.numer * denom,    //
         denom * that.denom                          // Define operater + - ... as def
       	)                                            //
                                                     //
    def + (i: Int): Rational =
       new Rational(numer + i * denom, denom)


    def * (that: Rational): ...
    def * (i: Int): ...

    def / (that: Rational): ...
    def / (i: Int): ...

    def - (that: Rational): ...
    def - (i: Int): ...


    override def toString = numer +"/"+ denom // override -> add a method to class: Rational override previous 'toString'

    private def gcd(a: Int, b: Int): Int = // A private def used internal
      if (b == 0) a else gcd(b, a % b) // recursion


}

// Use case
val oneHalf = new Rational(1, 2)
"oneHalf: Rational = 1/2"

oneHalf * 1
oneHalf * oneHalf
1 * oneHalf


// Implicit conversion 
" define a conversion method automatically "
implicit def intToRational(x: Int) = new Rational(x)

2 * x // Int() * rational() - works


// Identifier notation //
//P - 107
alphanumeric + operator













/* Built-in Control Structure */
if while for try match ...

// IF
//Type 1
var filename = "xxx.txt"
if (!args.isEmpty)
  filename = args(0)

//Type 2
val filename =
  if (!args.isEmpty) { args(0) }
  else { "xxx.txt" }

//Type 3
var filename = "xxx.txt"
  if (!args.isEmpty) {filename = args(0)}
  else if (xxxxx) {xxxxx}
  else if (xxxxx) {xxxxx}
  else {xxxxx}


//Type 3 (if ... generate a value based on one of the branch)
printIn(if (!args.isEmpty) args(0) else "xxx.txt")




// WHILE
while (a != 0) { // will test first and running while TRUE

	....
	a ..update exp
}


do {
	...
	a ..update exp
} while (a != 0) // will run first and test by each iter, running while TRUE
" while result Unit(), Unit() != '' "



// FOR LOOP
val fileHeader = ...

for (file <- fileHeader) // iter a collection
  exp ...

for (file <- 1 to 10) // iter a range
  exp ...

for (file <- 1 until 10) // iter a range -1
  exp ...


// - Filter FOR
for (file <- fileHeader if [statement])
  exp ...

for (file <- fileHeader)
  if ([statement])
  if ([statement]) // can multiple statements
    exp ...


// - Nested iteration
for {
	file <- fileHeader // loop 1
	if ([statement])
	line <- fileLines(file) // loop 2 within each loop 1
	if ([statement])
} exp ...


// - Mid-stream variable bindings
" non-trival computation, save result to a val and only compute one time "
trimed = line.trim // computation saved to a val, 'val' keywaord left out
if ...trimed.xxx()
....trimed


// - Producing a new collection
" use 'for' to generate a collection (Usually used with 'filter' "
" usually 'for' generate a iterable collection, then forget it after done "
for {file <- fileHeader if [exp...]} yield {file} // file - filtered iterable collection
                                     yield {[other var/val]} // filtered var/val




/* Exception handling with try expression */

// throwing exception - throw new [exception] clause
val half = 
  if (n % 2 == 0)
    n / 2 // type: Int
  else
    throw new RuntimeException("n must be even") // an excption throw has type: Nothing

// catching exception - try-catch clause
import java.io.FileReader
import java.io.FileNotFoundException
import java.io.IOException

try {
	val f = new FileReader("input.txt")
	// use and colse file
} catch {
	case ex: FileNotFoundException => // handle missing file
	case ex: IOException => // Handle other I/O error 
}
" if non of the exception meet, will execute further "


// 'finally' clause for cleaning up
import java.io.FileReader

val file = new FileReader("input.txt")
try {
	// use the file
} finally {
	file.close() // Be sure to close the file
}

def f(): Int = try {return 1} finally {return 2} // it returns 2 (X)
def g(): Int = try { 1 } finally { 2 } // it returns 1
" usually 'finally' used to clean up, should not return anything in 'finally' "




/* match patter */
val firstArg = if (!args.isEmpty) args(0) else "" // ! args.isEmpty = T/F

val friend = // a match expression can result a value to form a val/var
  firstArg match {
  	case "salt" => "pepper"
  	case "chip" => "salsa"
  	case "egg" => "bacon"
  	case _ => "huh?" // _ -> wildcard
  }

printIn(friend)






/* Deal without break, continue */

// with 'break', 'continue' [THIS IS IN JAVA]
Int i = 0;
booolean foundIt = False;

while (i < args.length) { // stop if search all didn't fine it
	if (args[i].startswith("-")) { // continue if start with "-"
		i = i + 1;
		continue; // directly to the end
	}
	if (args[i].endsWith(".scala")) { // stop if found '.scala' file
		foundIt = true;
		break; // stop
	}
	i = i + 1; // with 'continue'
}



// Type 1 without [IN SCALA]
" if {} replace continue "
" boolean replace break "

var i = 0
var foundIt = False

while (i < args.length && !foundIt) { // limit(T/F) && foundit(T/F) -> either meet will stop
	if (!args(i).startsWith("-")) { // if this meet - true - reverse'!' - false, jump to 'i = i + 1'
		if (args(i).endWith(".scala")) // if first didn't meet - false - reverse'!' - true - execute this if, no - go to end, yes - return true to stop loop
		  foundIt = True
	}
	i = i + 1
}



// Type 2 without {IN SCALA}
" use recursion + if else if else "

def searchFrom(i: Int): Int = // define a fun for recursion
  if (i >= args.length) -1 // if nothing find, '-1' negative
  else if (args(i).startsWith("-")) searchFrom(i + 1) // if this meet, continue FUN with i + 1 (recursion)
  else if (args(i).endsWith(".scala")) i // if this meet, return i (stop)
  else searchFrom(i + 1) // if nothing meet, continue FUN with i + 1 (recursion) (safty)

val i = searchFrom(0) // initiate a val for the function (starting at 0)




/* variable scope - environment variable */
" { var } - usually define a scope, gobal - {}1st - {}2nd ... scope ends by the closing } "

var y = 60
// y in scope (global)
def printMutiTable() {
	var i = 1
	// only y, i in scope here
	while (i <= 10) {
		var j = 1
		// y, i, j in scope
		while (j <= 10) {
			var prod = (1 * j).toString
			// y, i, j and prod in scope
			var k = prod.length
			// y, i, j, prod and k in scope
		}
	}
}

// same name var in different scope (will complie) (not recommand)
val a = 1;
{
	val a = 2
	printIn(a)
}
printIn(a)
"2 1"







/* Function and Closures */

















/* Control Abstraction */














/* Composition and Inheritance in CLASS */

// [1] Define an class with abstract members (Without implementation)
abstract class Element {
	def contents: Arrary[String]
}
// may have members do not have implementation
// Can initiate an abstract class XXX
new Element ": error message "
// A method 'content' is abstract if no equal sign or body
" Class: Element declares the abstract method, currently no concrete method"

// [2] Defining parameterless methods
abstract class Element {
	def content: Arrary[String] // parameterless methods
	def height: Int = contents.length // parameterless methods
	def width: Int = if (height == 0) 0 else contents(0).length // parameterless methods
}

"Instead of ..."
def width(): Int // empty-paren methods *if the function do operation
"defined as ..."
def width: Int // parameterless methods *if only access to a property

// Also can do... (depends on usage, field pre-define (May faster), fun runs every call)
abstract class Element {
	def content: Arrary[String] // field
	val height = contents.length // field
	val width = 
	  if (height == 0) 0 else contents(0).length // field
}


// [3] Extending CLASS (Create subclass) 
class ArraryElement(Conts: Arrary[String]) extends Element {
	def contents: Arrary[String] = conts
}
" ... extends Element ..."
// ArraryElement inherite all non-private members from CLASS element
// Make type: ArraryElement the sub-type of type: Element
// subclass: ArraryElement - superclass: Element
// Any superclass belongs to "Scala.anyRef" CLASS
" only private memember not inheriate, or subclass has the same name as superclass - overwrite"
" superclass: {abstract memeber A} - subclass: {concrete memeber A} (implemented the abstract memeber A) "
val ae = new ArraryElement(Arrary("hello", "world"))
ae.width // use superclass' method
// a value of subclass can be used whenever a value of superclass is required
val e: Element = new ArraryElement(Arrary("hello"))



// [4] Overriding methods and fields
"""
Scala has two namespaces:

space1: values(fields, methods, packages, and singleton objects) // > can override each other
space2: types(class, traits) // > can override each other
"""
class ArraryElement(Conts: Arrary[String]) extends Element {
	val contents: Arrary[String] = conts // override parameterless fun "content" in the superclass: element
}                                        // good implementation subclass: concrete (field) -> implement -> superclass: abstract (method)

//but same name in same class, NO XXX
class sample {
	private var f = 0  // won't compile
	def f = 1          // won't compile
}



// [5] defining parameteric fields
calss Cat {
	val dangerous = false // superclass
}

class Tiger(override val dangerous: Boolean, // subclass -> use the same name 'dangerous' as parameteric field to define
	        private var age: Int) extends Cat

" same as below: "
class Tiger(param1: Boolean, param2: Int) extends Cat { // nornal parameters
	override val dangerous = param1 // need override for the same name
	private var age = param2
}



// [6] Invoking superclass constructors
class LineElement(s: String) extends ArrayElement(Array(s)) { // pass 'Arrary(s)' to its superclass
	override def width = s.length
	override def height = 1
}
" ... extends superclass([args you want to pass]) {....}"


// [7] Using override modeifiers
override // needed if same name to the memebers (concrete) in a parent class
![override] // ignore if same to the memebers (abstract) in a parent class


// [8] Polymorphism (many forms, many shapes) and dynamic binding
//superclass
" Element has many forms - subclasses "
abstract class Element {
	def demo() {
		printIn("Implementation 1")
	}
}

// inheriate hieracy
class ArraryElement extends Element {
	override def demo() {
		printIn("Implementation 2") // override
	} 
}

class LineElement extends Element {
	override def demo() {
		printIn("Implementation 3") // override ''
	} 
}

class UniformElement extends Element // direct inheriate


//Run - result
def invokeDemo(e: Element) {
	e.demo()
}

invokeDemo(new ArraryElement)
" Implementation 2 " // 

invokeDemo(new LineElement)
" Implementation 3 "

invokeDemo(new UniformElement)
" Implementation 1 " // inheriate superclass implementation



// [8] Declaring final members
class ArraryElement extends Element {
	final override def demo() { // add final to a member to provent from overriden by a subclass
		printIn("Implementation 2") 
	} 
}

final class ArraryElement extends Element { // add final to a class to provent from been subclassed
	override def demo() { 
		printIn("Implementation 2") 
	} 
}



// [9] Using composition and inheritance

//P - 189





/* Scala's class hierarchy */

// P - 211 pic - Any -> AnyVal
//                   -> AnyRef




/* Traits */ // can't take parameters like CLASS
" Encapsulates methods, fields definiations which can be resued when mixed them into classes "

// Create an Traits
trait sample1 {
	def function1() {
		printIn("same texts")
	}

	def function2() {
		printIn("same texts")
	}
}


// Mix a traits to a Class
class CLASS_some extends sample1 {
	override def toString = "green"
}

// - can use trait's fun
val ghm = new CLASS_some
ghm.function1()

//or define a type of it
val ghm: sample1 = bty
ghm.function1()



// create subclass to mix traits
class CLASS_some

class CLASS_single extends CLASS_some with sample1 with sample2 { // if two traits, execute from very right to left
   override def toString = "green"
   override def function1 { // you can override trait's function
   	printIn("another text")
   }
}


// Stackable modificantions
class CLASS_some

class CLASS_single extends CLASS_some

trait sample1 extends CLASS_some { // declare superclassed to CLASS_some
	def function1() {
		printIn("same texts")
	}

	def function2() {
		printIn("same texts")
	}
}

trait sample2 extends CLASS_some { // declare superclassed to CLASS_some
	def function1() {
		printIn("same texts")
	}

	def function2() {
		printIn("same texts")
	}
}


" can only be mixed with class also extends with 'CLASS_some' "
class CLASS_deside extends CLASS_single with sample1

val queue = (new CLASS_single with sample1 with sample2) // call from right to left (Stack modifications)






/* Packages and Imports */
// Group programs into smaller modualers
" elements: class, standalone object, traits"


//create a package cover all file
package name_of_package.class_name
class class_name // a package contains class: class_name

//Create packages for different sections in a file
package pkg1 {
	package pkg2 {

		// In package pkg1.pkg2
		class class_name1

		package pkg3 {

			// In package pkg1.pkg2.pkg3
			class class_name2
		}
	}
}



// access vars in different pkgs
package pkg1 {
	class class_name1
}

package pkg2 {
	package pkg3 {
		package pkg4 {
			class class_name2
		}
	class get_class {
		val class_name1 = new _root_.pkg1.class_name1 // _root_ goes to the very back
		val class_name2 = new pkg4.class_name2
		val class_name3 = new pkg2.class_name3
	    }
	}
	class class_name3
}


// Imports - avaiable 
import pkg2.pkg3._ // access to all memebers in pkg3

import pkg2.{class_name1, class_name2} // access to cls1, cls2 of pkg2

import pkg2.pkg3.{class_name1 => other_name, class_name2} // cls1 as other_name, cls2

import pkg2.pkg3.(class_name1 => _, _) // except class_name1, all import


// private / protect memebers
private // Only avaiale inside the class or same name member
protected // only avaible from subclasses 
public // other will all be public


// Scope of protection
private[pkg1] class class_name1 // class_name1 is visable to all classes or objects in pkg1 {}
protected[pkg1] class class_name1 // class_name1 and its all subclassess is visable to pkg1 {}


// Package Objects
// -------- pkg.scala -------- //
package object sample_object {
	import pkg2._
	def test1() {}
	def test2() {}
}



// ------- some_thing.scala --------- //
package some_thing
import sample_object.test1 // import def like class
import sample_object.test2
test1() // avaible all
test2() // 










/* Assertions and Unit Testing */
// P - 255





/* Case Classes and Pattern Matching */

// [1] Define Case Class
sealed abstract class Expr // sealed with colse all case class, easy to match
  case class class1(a: String) extends Expr
  case class class2(c: String) extends Expr
  case class class3(d: Double) extends Expr
  case class class4(f: String, s: String, t: Expr) extends Expr
// - case class: directly define a object for case class
val v = class1("x")
val p = class2("x")

// - case class: all parameters in a case class has prefixed field names
v.a 
" String = x"

// - case class: has methods - 'toString', 'hashCode', 'equals'
v.a == p.c

// - case class: 'copy' method to make modified copies
val a_copy = a.copy(a="k")


// [2] Pattern matching
def match_all(expr: Expr): Expr = expr match {
	case class1("d") => ["do somethingA"]
	case class2("g") => ["do somethingB"]
	case class3(1.2) => ["do somethingC"]
	case _ => expr // default case matches all cases
}

" selector match { alternatives } "

// - Kind of patterns
// (1) Wildcard pattern
expr match {
	case class1(_) => ["do something"] // match all value of 'a' in class1
	case _ => expr
}

// (2) Constant patterns
x match {
	case 5 => "five"
	case true => "T"
	case "hello" => "hi"
    case _ => x
}

// (3) Variable patterns - matches any object like wildcard
expr match {
	case 0 => "Zero"
	case somethingElse => "not Zero" // 'somethingElse' like a variable to hold the other values
}
" Convention: lowecase 'add' - variable pattern "
" Convention: lowecase 'Add' - constant pattern "
" Convention: backtike `add` - constant pattern "

// (4) Constructor patterns - identify the name designates a case class (deep matches)
expr match {
	case class4("d", "x", class1("g")) => ["do somethingA"] // match the class, each parameters, 
	case _ => printIn("Not found")                          // also the class within class and its paras (deep matches)
}


// (5) Sequence patterns - match sequence type like List, Arrary
expr match {
	case List(0, _, _) => ["do somethingA"]
	case List(4, _*) => ["do something"] // regardless the length of the list
	case _ => expr
}

// (6) Typed pattern - type test / cast
def type_class(x: Any) = x match {
	case s: String => s.length // s holds value as x, not same type, must use s.length, Any.length not exists
	case m: Map[_, _] => m.size // type test => cast
	case _ => -1
}
// or ... with if else (poor style)
x.isInstanceOf[String] // type test
x.asInstanceOf[String] // cast something to 'String'

// ONLY in Array (match type of type)
case a: Array[String] => yes
case _ => "no"
Array("s", "d", "a") // 'Yes'
Array(1, 2, 3) // "No"

// (7) Variable binding ???
expr match {
	case class4("s", 1.4, e @ class1("x")) // bind 'class1("x")' into e with '@'
	case _ =>
}

// (8) Pattern guards
e match {
	case class1(e) if e != "x" => ["do something"] // 'if [condition]' guarded the pattern 'class1(e)'
	case _ =>
}

// (9) Pattern overlaps - execute in the order, case covered all, any below can't reach
// P - 285

// (10) suppress exhaustivity checking for patterns
(e: @unchecked) match {
	..... // not necessary include all cases
}

// (11) Optional type - handle where value can be actual value or may be 'None', continue the loop not break
def show(x: Option[String]) = x match {
	case Some(s) => s // any value s
	case None => "?" // define a case for 'None'
}


// (12) Patterns in variable definiation
val myTuple = (123, "dds")
val (number, string) = myTuple
" number: Int = 123 "
" string: String = 'dds' "

val exp = new class4("s", 1.3, "p")
val class4(str1, number1, str2) = exp
" str1: String = s "
" number1: Int = 1.3 "
" str2: String = p "


// (13) Case sequences as partial function ???
// P - 291
val withDefault: Option[Int] => Int = {
	case Some(x) => x
	case None => 0
}

withDefault(some(9))
" Int = 9 "
withDefault(None)
" Int = 0 "


// (14) Patterns in 'for' expressions
val cap = Map("sss" -> 123, "ddd" -> 456, "kkk" -> None)

for ((v1, v2) <- cap)
  ...v1 ...v2 // case 3 with 'None' will be ignored by for












 /* List - working with */
 // Immutable (No assignment) + same type in a list
val list1 = List(1, 2, 3)
val list2 = List("a", "b", "c")
val list3 = List(List(1,2,3), List("a", "b", "c"))
val list4 = List() // [nothing]

// If S subtype of T, List[S] subtype of List[T]
val num = 1 :: 2 :: 3 :: 4 :: Nil
// same as
List(1, 2, 3, 4)


// [1] basic Operations on List
list1.head // return first element
list1.tail // returns a list contains all but the first
list1.isEmpty // returns TRUE if it is


// [2] List patterns P - 309
val List(x, y, z) = list1
" x: Int = 1 "
" y: Int = 2 "
" z: Int = 3 "

val x :: rest = list1 // use 'rest' represents the length list(1 or greater)
" x: Int = 1 "

case List()
case X :: y 



// [3] First-order methods

// Concatenating two lists
List(1, 2, 3) ::: List(4, 5, 6)
" List(1, 2, 3, 4, 5, 6)"
list(...) ::: list(...) ::: list(...) // start from the right

// Taking the length of a list
List(1, 2, 3).length
" Int = 3 "

// Accessing the end of a list
list1.last // last element
list1.init // all but the last

// Reversing a list
list1.reverse
" List(3, 2, 1)"
list1.reverse.reverse
" same as it is "

// prefix | suffixes
list1 take 2
"List(1, 2)" // take first 2 elements
list1 drop 2
"List(3)" // drop first 2 elements
list1 splitAt 2
"List(List(1, 2), List(3))" // split the list by the n


// apply / indice
list1 apply 2
2
list1(2)
2


// Flatten - take lists of lists, flat into one list (Only apply all lists)
List(List(1, 2), List(3), List()).flatten
" List(1, 2, 3) "
List("apple", "pears").map(_.toCharArray).flatten
" List(a, p, p, l, e, p, e, a, r) "



// zip and unzip (takes two lists and form the pair tuples into a list)
List(1, 2, 3, 4) zip List("a", "b", "c", "d")
" IndexedSeq((1,a), (2,b), (3,c), (4,d)) "
// if List(length 5) zip List(length 6) = only length 5 (extra will be drop)

List("a", "b", "c", "d").zipWithIndex // zip value with its index
" List((a,1), (b,2), (c,3), (d,4)) "

val zipped = List(1, 2, 3, 4) zip List("a", "b", "c", "d")
zipped.unzip
"List(List(1, 2, 3, 4), List("a", "b", "c", "d"))"



// toString and mkString
List(1, 2, 3, 4).toString
" String = List(1, 2, 3, 4) " // returns canonical string representation of a list

List(1, 2, 3, 4) mkString ("/", ",", "/") // use different representation
" String = /1, 2, 3, 4/ "
List(1, 2, 3, 4) mkString
" String = 1234 "

val buf = new StringBuilder
" buf: StringBuilder = "
List(1, 2, 3, 4) addString (buf, "(", ";", ")") // append construct strings
" StringBuilder = (1;2;3;4) "



// Converting list (iterator, toArray, copyToArray)
val arr1 = list1.toArray
" Array(1, 2, 3, 4) "
val list1 = arr1.toList
" List(1, 2, 3, 4) "

val arr2 = new Array[Int](10)
" Array[Int] = Array(0, 0, 0, 0, 0) "

List(1, 2, 3) copyToArray (arr2, 3) // insert arr2 after the 3rd, then convert into Array
" Array[Int] = Array(0, 0, 0, 1, 2, 3, 0, 0) "

val it = List(1, 2, 3, 4).iterator
" Iterator[Int] = non-empty iterator "
it.next
1
it.next
2
...






// [4] Higher-order methods on CLASS list

// Mapping over list (map, flatMap, foreach) - transform every element in a list in a way 
// map (generate new object)
List(1, 2, 3) map (_ + 1)
" List[Int] = List(2, 3, 4) " // every element plus 1
List("add", "ffd", "sdfg") map (_.toList.reverse.mkString) // applying multiple funs on each element
" List[String] = List(dda, dff, gfds) "

//flatMap (generate new object)
List("add", "ffd", "sdfg") map (_.toList) // use - map (Only apply to each element)
" List[List[char]] = List(List(a, d, d), List(f, f, d), List(s, d, f, g)) "
List("add", "ffd", "sdfg") flatMap (_.toList)
" List[char] = List(a, d, d, f, f, d, s, d, f, g) " // use - flatMap (recursive - apply to each element, then flat and apply each elemnt again)

//foreach (takes a procedure - a fun with result type "unit()") simply just apply to original list
var sum = 0
List(1, 2, 3, 4) foreach (sum += _)
" unit() " // nothing generated
sum
" Int = 10 " // being changed in process




// Filtering lists

// filter 
List(1, 2, 3, 4, 5) filter (_ < 3)
" List(1, 2) " // only returns test = True

// partition 
List(1, 2, 3, 4, 5) partition (_ < 3)
" (List(1, 2), List(3, 4, 5))" // split the list into a pair of lists (where = True) (where = False)

// find 
List(1, 2, 3, 4, 5) find (_ < 3)
" Optin[Int] = Some(1) " // like filter but only returns the first element meets the test
" Optin[Int] = None " // If not found, return 'None'

// takeWhile
List(1, 2, 3, 4, 5, 6, 7) takeWhile (_ < 4)
" List(1, 2, 3) " // Only take while the test is True

// dropWhile
List(1, 2, 3, 4, 5, 6, 7) dropWhile (_ < 4)
" List(4, 5, 6, 7) " // Drop while the test is True

// span (combine takeWhile + dropWhile)
List(1, 2, 3, 4, 5, 6, 7) span (_ < 4)
" (List(1, 2, 3), List(4, 5, 6, 7)) " // Split the while True, while False into pair


// [6] Test over all elements of list
List(1, 2, 3, 4, 5) forall (_ < 2)
" Flase " // not all < 2
List(1, 2, 3, 4, 5) exists (_ < 2)
" True " // some < 2



// Folding lists /: and :\
 // z: initial number '/:' list 
// equal -
op(op(op(z,a), b), c)

//example - 
sum(List(a, b, c)) equals 0+a+b+c
def sum(list1: List[Int]): Int = (0 /: list1) (_ + _)

val word: List[String] = List("add", "pear")
("" /: word) (_+ " " +_)
" String =  add pear"

// how op calculated with /:
(z /: List(a, b, c)) (_ op _) equals op(op(op(z,a), b), c)
        op
       /  \
      op   c
     /  \    
    op   b
   /  \
  z    a   
// how op calculated with :\
(List(a, b, c) :\ z) (_ op _) equals op(a, op(b, op(c,z)))
  op
 /  \
a    op
    /  \
   b    op
       /  \
      c    z


// sorting list
// sortWith
List(1, -3, 4, 2, 6) sortWith (_ < _) // ascend order by value
word sortWith (_.length > _.length) // descend order by length of each string







// [5] Methos of List object
List(1, 2, 3) euqlas List.apply(1, 2, 3) // create list from their elements

List.range(1, 5) // from 1 going until 5 (5 - 1 = 4) -default by 1
" List(1, 2, 3, 4) "
List.range(1, 9, 2) // from 1 going until 9 (9 -2 = 7) by 2
" List(1, 3, 5, 7) "
List.range(9, 1, -3) // from 9 going until 1 (stop before 1 = 3) by -3
" List(9, 6, 3) "

List.fill(5)('s')
" List(s,s,s,s,s) " // uniform fill a list
List.fill(2,3)('s')
" List(List(s,s,s), List(s,s,s)) " // 2 X 3 dimensional list

List.tabulate(5)(n => n * n) // List(0, 1, 2, 3, 4) - n = n*n   ??? P - 332
" List(0, 1, 4, 9, 16) "

List.concat(List('a', 'd'), List('z'), List()) // concatnate lists into one list
" List('a', 'b', 'z') "

// Process multiple lists together with map + zipped
(List(10, 20), List(3, 4)).zipped.map(_ * _)
" List(30, 80) " // elements in two list will be paired and calculated with map
(List("abc", "de"), List(3,2)).zipped.forall(_.length == _)
" True " // "abc".length == 3; "de".length == 2
// If List(length = 3), List(length = 4) - extra element will be dropped in the process














































































































