>>>>>>>>>>>>>>>>>>>>>>>>>>
>
>     NLP In Python
>
>>>>>>>>>>>>>>>>>>>>>>>>>>




>>>>>>>>>>>>>>>>>> Package: NLTK

"http://www.nltk.org/"

# Install
import nltk



" Require: "
"          NLTK " # NLP package
"          NLTK-Data " # NLP Corpos / Test
"          Numpy " # Numeric computation / Structure
"          Matplotlib " # Ploting lab
"          NetworkX   " # Storing / manipulating network structures with nodes / visualization
"          Prover     " # This is an automated theorm prover for first-order and equational logic, used to support inference in NLP



" Natrural Language Processing Toolkit (NLTK) "

" Processing Task "                   |      " NLTK Modules "                     |     " Functionality "   

" Accessing corpora "                          nltk.corpus                              " Standarized interfaces to corpora and lexicons "

" String processing "                          nltk.tokenize, nltk.stem                 " Tokenizers, sentence tokenizers, stemmers "

" Collocation discovery "                      nltk.collocations                        " t-test, chi-squared, point-wise mutual information "

" Part-of-speech tagging "                     nltk.tag                                 " n-gram, backoff, brill, HMM, TnT "

" Classification "                             nltk.classify, nltk.cluster              " Decision tree, maximum entropy, naive Bayes, EM, k-means "

" Chunking "                                   nltk.chunk                               " Regular expression, n-gram, named checking "

" Parsing "                                    nltk.parse                               " Chart, feature-based, unification, probabilistic, dependency "

" Semantic interpretation "                    nltk.sem, nltk.inference                 " Lambda calculas, first-order logic, model checking "

" Evaluation metrics "                         nltk.metrics                             " Precision, recall, agreement coefficients "

" Probability and estimation "                 nltk.probability                         " Frequency distributions, smoothed probability distributions "

" Applications "                               nltk.app, nltk.chat                      " Graphical concordancer, parsers, WordNet browser, chatbots "

" Linguistic fieldwork "                       nltk.toolbox                             " Manipulate data in SIL Toolbox format "







>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " [1] Basic Stats & calculation from Text "

from nltk.book import * # Loading all text <text1, text2, ...>
text1
" <text: name 1851> "


1. "Search Text"

text1.concordance("monsters") # Give the text with context
" ....... monster ........ "
" ....... monster ........ "

text1.similar("monster") # Give text has similar context
" Alien, wired, ..... "

* "Different articles same word has diff context "

text1.common_contexts(["monster", "very"]) # allows us to examine just the contexts that are shared by two or more words
" be_glad, am_glad, ..... "

text1.dispersion_plot(['citizens', "monster", "alien"]) # Create a plot with text locations appear across by 0-total words in this article

text1.generate() # Generate random text - it reuses common words and phrases from the source text and gives us a sense of its style and content




2. "Counting Vocabulary"

len(text1) # count total words

*"Token - a sequence of characters like 'hairy','his', ... "

sorted(set(text1)) # Top frequent unique keywords

len(set(text1)) # count total unique words

from __future__ import division
len(text1)/len(set(text1)) # lexical richness - average how many times a word been used

100 * text1.count('smoke') / len(text1) # how % a word accounts for the total words count





3 "Frequency Distribution"
text1 = ['text1','text2','text2',....]
fdist1 = FreqDist(text1)
fdist1
"<freqDist with 26037 outcomes>"
vocabulary1 = fdist1.keys()
Vocabulary1[:50]
"[',','we',.....]"
fdist1['whale']
'906'

fdist1.plot(50, cumulative=True) # Plot cumulative plot of word frequency from top

fdist1.hapaxes() # Return a list of words only appear once

** "Function for FreqDist"
fdist = FreqDist(sample) # Create a freq dist from the sample
fdist.inc(sample) # Increment the count for this sample
fdist['monster'] # count of the number of times a given sample occurs
fdist.freq('monster') # Frequency of a given sample
fdist.N() # Total number of sample
fdist.keys() # The samples sorted in order of decreasing frequency
for sample in fdist: # iterate over the samples, in order of decrasing frequency
fdist.max() # Sample with the greatest count
fdist.tabulate() # Tabulate the frequency distribution
fdist.plot() # Plot freq dist
fdist.plot(cumulative=True) # plot cumulative freq dist
fdist < fdist2 # Test if samples in fdist1 occurs less frequently than in fdist2







3-1 "Function for ConditionalFreqDist"
sample = [(A,B) for A in List_A for B in List_B]
"[('a1','b1'),('a1','b2'),...,(...)]" # Format (Condition, event)
cfd = nltk.ConditionalFreqDist(sample)
cfd.conditions()
"['a1','a2','a3']"
cfd['a1']
"<FreqDist with 2 outcomes>"
list(cfd['a1'])
"['b1','b2']"

A = ['a1','a2','a3']
B = ['b1','b2']

cfd.tabulate(conditions=A, samples=B)
"      b1      b2    "
"   a1 21      44    "
"   a2 12      32    "
"   a3 61      18    " 

cfd = nltk.ConditionalFreqDist(
	       (genre, word)
	       for genre in corpus.categories()
	       for word in corpus.words(categories=genre))
cfd[:10]
"[(condition1, event1),(condition1, event2),(condition1, event3),...]"

cfd.plot() # Plot conditional frequency from top

** "Function for ConditionalFreqDist"
cfdist = ConditionalFreqDist(pairs) # Create a freq dist from the list of pairs
cfdist.conditions() # List all conditions
cfdist[condition] # Check frequency dist for samples in specific condition
cfdist[condition][sample] # Check frequency dist for specific samples in specific condition
cfdist.tabulate() # create contengency table between condition and samples
cfdist.tabulate(samples, conditions) # create contengency table between condition and samples, given samples and conditions
cfdist.plot() # Graphic plot of cfdist (# of lines = # of conditions)
cfdist.plot(samples, conditions) # Graphic plot of cfdist, given samples and conditions 
cfdist1 < cfdist2 # Test if samples in cfdist1 occurs less frequently than in cfdist2






4. "Selection of long words"

V = set(text1)
long_words = [w for w in V if len(w) > 15]
sorted(long_words) # Sort words longer than 15 strings from longest 

fdist5 = FreqDist(text5)
sorted([w for w in set(text5) if len(w) > 7 and fdist[w] > 7]) # all words that are longer than 7 strings and occur more than 7 times




5. "Collocations and Bigrams"

*"Collocation - a sequence of words that occur together unusually often"
*"Bigrams - 'I am the one' = [(I,am),(am,the),(the,one)]"

bigrams(['more','is','said','done'])
"[('more','is'),('is','said'),...()]"
# Create Random Text with Bigrams
def generate_model(cfdist, word, num=15):
	for i in range(num):
		print word, 
		word = cfdist[word].max() # the most frequent keyword 'W2' associated with 'W1' 
		                          # and then the most freq 'KW3' associated with 'KW2' 
		                          # total 15 words

text = nltk.corpus.genesis.words('xxxx.txt')
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams)
print cfd['living']
"<freqDist: 'creature':7, 'thing':4, 'substance':2, ....>"
generate_model(cfd, 'living')
"living creature ......."


text4.collocations()
"['united states', ....]"




6. "Distribution of word length"

[len(w) for w in text1]
"[1,4,4,...,6,9]"
fdist = FreqDist([len(w) for w in text1])
fdist
"<freqDist with 98768698 outcomes>"
fdist.keys() 
"[3,1,4,2,5,.....]" # Most words is length 3, then length 1, then, ...
fdist.items()
"[(3,50223),(1,35234),(4,.....]"
fdist.max()
"3" # length 3
fdist[3]
"50223" # counts of length 3 words
fdist.freq(3)
"0.192554..." # Proportion of length 3 words is about 19%








** "Python string manipulation"
s.startswith(t) # if start with t
s.endswith(t) # if end with t
t in s # if t contained inside s
s.islower() # strings to lower
s.isupper() # strings to upper
s.isalpha() # if all alpha
s.isalnum() # if all alphanumeric
s.isdigit() # if all digits
s.istitle() # if all words have initial title letter captalized



** "Challenges for NLPs"

"Word Sense Disambiguation" # Different meaning for the same word under different context
"Pronoun Resolution" # If multiple noun before, the verb is to which noun
"Generating Language Output" # Answering questions, translation
"Machine Translation(MT)" # Same words have multiple translation --- Text alignment(Gievn dosument in two lang, pair up them)
"Spoken Dialogue System" # Analyze input, generate output --- Turing test (Poeple tell if a machine or ppl)
"Textual Entailment" # 'Recognizing Textual Entailment (RTE) --- Given side information to make decision'

** "Limitation for NLP"

" Can not perform common sense reasoning "

































>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " [2] Accessing Text Corpora and Lexical Resources "

" Text Corpora --> a large body of text | Usually designed to contain a careful balance of material in one or more genres "

" Corpora in NLTK "

1. "Gutenberg Corpus --> from project Gutenberg"
2. "Web and Chat Text --> Firefox discussion forum"
3. "Brown Corpus --> Contain 500 sources like news, religion, etc"
4. "Reuters Corpus --> Contains 10,788 news documents"
5. "Inaugural Address Corpus --> Inaugural address information"
6. "Annotated Text Corpora"
7. "Corpora in other Language"


[1] "Access Corpora"
import nltk
nltk.corpus.gutenberg.fileids() # Check all files in that corpus
"[xxx.txt, xxxx.txt, ...., xxxx.txt]"

nltk.corpus.gutenberg.raw('xxx.txt') # Check content as single string in that file
"xxxxx xxxxx xxxx ..... xxxxxxx xxxx"

nltk.corpus.gutenberg.words('xxx.txt') # Check content as a list of strings in that file
"['xxxx','xxxx','xxxxxx', ...., 'xxxxx']"

nltk.corpus.gutenberg.sents('xxx.txt') # Check content as a list of lists of strings (each sentence) in that file
"[['xxxx','xxxx','.'],['xxxxxx', ....], ['xxxxx','xxxxx','.']]"

nltk.corpus.brown.categories() # Check all categories in that corpus
"['cate1','cate2',..., 'caten']" 

# specify cate or fields
nltk.corpus.brown.words(categories='cate1')
nltk.corpus.brown.words(categories=['cate1','cate1'])
nltk.corpus.brown.words(fileids=['cate1'])

** "In some corpus, two fields may overlap a few categories, two categories may overlap a few fields"
nltk.corpus.brown.categories('xxxxx1.txt')
"['A','B','C']"
nltk.corpus.brown.categories('xxxxx2.txt')
"['A','D','C']"
nltk.corpus.brown.fileids('A')
"['xxxx1.txt','xxxxx2.txt','xxxxx3.txt']"
nltk.corpus.brown.fileids('B')
"['xxxx4.txt','xxxxx2.txt','xxxxx5.txt']"



** 'Text Corpus Structure'

.fileids()                             "The files of the corpus"
.fileids(['categories'])               "The files corresponding to categories"
.categories()                          "The categories of the corpus"
.categories(['fileids'])               "The categories corrsponding to the fileids"
.raw()                                 "The raw content of the corpus"
.raw(fileids=['f1','f2','f3'])         "The raw content of the specified files"
.raw(categories=['c1','c2','c3'])      "The raw content of the specified categories"
.words()                               "The list of strings of the corpus"
.words(fileids=['f1','f2','f3'])       "The list of strings of the specified files"
.words(categories=['c1','c2','c3'])    "The list of strings of the specified categories"
.sents()                               "The list of lists of strings of the corpus"
.sents(fileids=['f1','f2','f3'])       "The list of lists of strings of the specified files"
.sents(categories=['c1','c2','c3'])    "The list of lists of strings of the specified categories"
.abspath('fileids')                    "The location of the given file on disk"
.encoding('fileids')                   "The encoding of the file(if know)"
.open('fileids')                       "Open a stream for reading the given corpus file"
.root()                                "The path to the root of locally installed corpus"
.readme()                              "the content of readme file for the corpus"






[2] "Loading your own ccorpus"

2.1 "NLTK - PlaintextCorpusReader" # Load text file from local
from nltk.corpus import PlaintextCorpusReader
corpus_root = '/usr/share/dict' # Set location
wordlists = PlaintextCorpusReader(corpus_root, '.*') # load all files match '.*'
wordlists.fileids()
"['xxxx.txt','xxxx.txt']"
wordlists.words('xxxx.txt')
"['the','of',....]"

 
2.2 "NLTK - BracketParseCorpusReader" # Load local corpus
from nltk.corpus import BracketParseCorpusReader
corpus_root = r"C:\corpus\sdaasd\corpus"   # Set location
corpus_pattern = r".*/xxx_.*\.mrg"
ptd = BracketParseCorpusReader(corpus_root, corpus_pattern) # load all files match '.*'
wordlists.fileids()
"['xxxx.mrg','xxxx.mrg']"
wordlists.words('xxxx.mrg')
"['the','of',....]"







[3] "Lexical Resources"

" A lexicon, or lexical resource, is a collection of words and/or phrases along with associated information, such as"
" part-of-speech and sense definiations. Lexical resources are secondary to texts, and are usually created and enriched with "
" the help of texts. For example, the Vocabulary and word freqency are the lexical resource for an text."




1. "Terminology"
"saw, [verb], past tense of see."
"saw, [noun], cutting instrument."

"saw-saw"                              : "Headword, or lemma" & "Same spelling -> 'homonyms'"
"[verb]-[noun]"                        : "Part-of-speech, lexical category"
"past tense of see-cutting instrument" : "Sense definiations, or gloss"





2. "Wordlist Corpus" # Nothing more than word list (Lexical resource)

>1 "Use for check un-common words or mis-spelling qwords"

nltk.corpus.words

def unusual_words(text):
	text_vocab = set(w.lower() for w in text if w.isalpha())
	english_vocab = set(w.lower() for w in nltk.corpus.words.words())
	unusual = text_vocab.different(english_vocab)
	return sorted(unusual)


>1-2 "Use for solving word puzzle"

nltk.corpus.words

" E G I "
" V R V "
" O N L "
# How many words can be made has >=6 letters from the grid
# Must contain 'R'
puzzle_letter = nltk.FreqDist('egivrvonl')
obligatory = 'r'
wordlist = nltk.corpus.words.words()
[w for w in wordlist if len(w) >= 6 # length contrains >= 6 letters
                     and obligatory in w # 'r' has to be in the word
                     and nltk.FreqDist(w) <= puzzle_letter] # since each letter in puzzle can only be used one time
                                                            # the freq of each letter in the word can not be more than the freq of letter in puzzle
"['glover', 'gorlin', ....]"





>2 "Use for remove stopwords like 'the, to, also,..'"

nltk.corpus.stopwords

from nltk.corpus import stopwords
stopwords.words('english')
"['a','as', ....]"
def remove_stopwords(text):
	stopwords = nltk.corpus.stopwords.words('english')
	content = [w for w in text if w.lower() not in stopwords]
	print len(content) / len(text)
	return content




>3 "Use for check and compare English names - 8,000 first name by gender"
 
nltk.corpus.names

names = nltk.corpus.names
names.fileids()
"['female.txt','male.txt']"
male_names = names.words('male.txt')
female_names = names.words('female.txt')
# same name for male and female
[w for w in male_names if w in female_names]
"['Alfie', 'Abbie', ....]"




>4 "Use for word pronouncing Dictionary"

nltk.corpus.cmudict

entries = nltk.corpus.cmudict.entries()
len(entries)
"127012"
for entry in entries[39943:38851]:
	print entry

"('fir', ['F','ER1'])" # For each word, this lexicon provides 'phonetic codes' 
"('fire', ['F', 'AY1', 'ERO'])"
"(...........................)"

4-1 "Identify a word with 3 pron, start with 'P' and end with 'T', and then print the word and middle porn"
for word, pron in entries:
	if len(pron) == 3:
		ph1, ph2, ph3 = pron
		if ph1 == 'P' and ph3 == 'T':
			print word, ph2

"pait EY1 pat AE1 ....."


4-2 "Identify word end with certain pronance"
syllable = ['N', 'IHO', 'K', 'S']
[word for word, pron in entries if pron[-4:] == syllable] # last 4 pron match

"['atlantics','audiotronics',.....]"

[word for word, pron in entries if pron[-1] == 'M' and w[-1] == 'n'] # pronanue end with 'M' and word end with 'n'

"['autumn', 'column',...]"




4-3 "Identify by look up particular words"
prondict = nltk.corpus.cmudict.dict()
prondict['fire']
"[['F','AY1','ERO'],['F','FA1','R']]"
# If word not exist
prondict['blog']
"KeyError..."
# Add it / only local / no change in corpus
prondict['blog'] = [['B','L','AA1','G']]
prondict['blog']
"[['B','L','AA1','G']]"








5> "Compare same words with different language - lists of 200 common words in serveral lanuage"

nltk.corpus.swadesh

"Language identified using an ISO 639 two-letter code"

from nltk.corpus import swadesh
swadesh.fileids()
"['be','bg',.....'en',....]" # all language
swadesh.words('en') # all words for 'en'
"['I', 'you (singluar), thou', 'he', .....]"
# Create a small translator
fr2en = swadesh.entries(['fr','en']) # french to English
fr2en
"[('je','I'),('tu','you'),...]"
translate = dict(fr2en)
translate['Hund']
'dog'

# Compare words in various language
language = ['en','fr','es']
swadesh.entries(language)
"[('say','sagen','seggen'),(....),....]"



6> "Most popular tool - 'Toolbox' previously know as 'Shoebox' "

nltk.corpus.toolbox

"Used by linguists to manage data - P66"

from nltk.corpus import toolbox
toolbox.entries('rotokas.dic')
"[....()...([])......([][[]])]" # a series of attribute pairs







3 "WordNet"

nltk.corpus.wordnet

"WordNet is a semantically oriented dictionary of English"
"With 155,287 words and 117,659 synonym sets"

"WordNet Hierarchy"
#######################################################################################
#
#
#                     | artefact |-------------------------- High level words
#                          |
#                          |
#                   |motor vehicle|-------  ---------------- Middle level words
#                     /        |          \
#                    /         |           \
#                   /          |            \
#             |motorcar|     |go-cart|    |truck| ---------- Lower middle level words
#            /       |  \   
#           /        |   \
#          /         |    ----------|
#   |hatch-back|  |compact|   |gas guzzler|  --------------- Low level words (More specific)
#
#
#######################################################################################


"Lexical Relation 1 --> Hyponyms | Hypernyms" # General -- Specific Ex. car --> Benz
"Lexical Relation 2 --> Meronyms | Holonyms" # Whole -- parts Ex. hand --> finger
"Lexical Relation 3 --> entails" # relationship between verbs -- entials (Walk involves the act of stepping)
"Lexical Relation 4 --> antonymy" # the opposite meaning, relationship between Lemmas -- ex. Good - Bad

3-0 "Hyponyms" # ------ A lower level word to its parent word
motorcar = wn.synset('car.n.01')
type_of_motorcar = motorcar.hyponyms()
type_of_motorcar[26] 
"Synset('ambulance.n.01')"
sorted([lemma.name for synset in type_of_motorcar 
	               for lemma in synset.lemmas])
"['model-T', 'S.U.V',....]"


3-0 "Hypernyms" # ------ A higher level word to its parent word
motorcar = wn.synset('car.n.01')
motorcar.hypernyms()
"[Synset('motor_vecgicle.n.01')]"
path = motorcar.hypernym_paths()
len(path)
"2" # 2 path from topest to 'car.n.01'
[synset.name for synset in path[0]] # path 1
"['entity.n.01', .....1... 'car.n.01']"
[synset.name for synset in path[1]] # path 2
"['entity.n.01', .....2... 'car.n.01']"


3-0 "Meronyms"
wn.synset('tree.n.01').part_meronyms() # what parts consist of the 'tree'
"[Synset('burl.n.02'), ....limb.....stump.....trunk....]"
wn.synset('tree.n.01').substance_meronyms() # what made of the 'tree'
"[Synset('heartwood.n.02'), ....sapwood....]"


3-0 "Holonyms"
wn.synset('tree.n.01').member_holonyms() # what main entity the 'tree'  
"[Synset('forest.n.01')]"



3-0 "Entails"
wn.synset('walk.v.01').entailments()
"[Synset('step.v.01')]"
wn.synset('eat.v.01').entailments()
"[Synset('swallow.v.01'), Synset('chew.v.01')]"


3-0 "antonymy"
wn.lemma('supply.n.02.supply').antonymy()
"[Lemma('demand.n.02.demand')]"

** "More relationships? Try: "
dir(wn.synset('harmony.n.02'))





3-1 "Sense and Synonyms" # ----- same level words above

"a1) benz is credited with the invention of the motorcar."
"b1) benz is credited with the invention of the automobile."

"motorcar ~ automobile [synonyms]"


3-2 "Access wordnet"

from nltk.corpus import wordnet as wn
wn.synsets('motorcar') # identify all synonym sets given a word
"[Synset('car.n.01')]" # only one set for 'motorcar'

wn.synsets('car') # More vegue words has more synsets
"[Synset('car.n.01'), Synset('car.n.02').....]" # more meanings for same word

wn.synset('car.n.01').lemma_names # list all words in that synonmy set
"['car','auto','automobile','motorcar',...]"

wn.synset('car.n.01').definiation # definition of this set
"a motor vehicle with four wheels; usually ..."

wn.synset('car.n.01').examples # Some example of the word
"['he needs a car to go to work']"


# lemma
"car.n.01" + "motorcar" = "car.n.01.motorcar" # synset + word = lemma

wn.synset('car.n.01').lemmas # get all lemmas associated with the synset
"[Lemma('car.n.01.car'), Lemma('car.n.01.auto'),.....]"

wn.lemma('car.n.01.automobile') # look up a particular lemma
"Lemma('car.n.01.automobile')"

wn.lemma('car.n.01.automobile').synset # look up synsets associated with this lemma
"Synset('car.n.01')"

wn.lemma('car.n.01.automobile').name # get the name of the lemma
"automobile"

we.lemma('car') # get lemma covers 'car'
"[Lemma('car.n.01.car'), Lemma('car.n.02.car'), ....]"




3-3 "Semantic Similarity"

"Since Network has been linked by complex network and lexical relationship -- words are sementically related"
"Identify two words' lowest shared node, if the node is low in the level, those two words are close"

right = wn.synset('right_whale.n.01')
orca = wn.synset('orca.n.01')
novel = wn.synset('novel.n.01')

right.lowest_common_hypernyms(orca)
"[Synset('baleen_whale.n.01')]"
right.lowest_common_hypernyms(novel)
"[Synset('entity.n.01')]"

wn.synset('baleen_whale.n.01').min_depth()
"14"
wn.synset('entity.n.01').min_depth()
"0"

"So, right is more close to orca than close to novel"

**"Or use 'path_similarity'"

right.path_similarity(orca)
"0.25"
right.path_similarity(novel)
"0.043462354"




































>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " [3] Process Raw Text "

" Massive amount of data exists on Web. Need to know how to access them."

** " Tokenization - We need to break up the strings into words and punctuation"

** " Stemming - "



[1] " Accessing Text from Web and from Disk "

"###############################################"
"#              THE NLP Pipeline               #"
"###############################################"

"## HTML -> ASCII ##"
"Download web page, strip HTML if necssary, trim to desired content"
html = urlopen(url).read()
raw = nltk.clean_html(html)
raw = raw[0:34234224]

"## ASCII -> Text ##"
"Tokenize the text, select tokens of interest, create an NLTK text"
tokens = nltk.wordpunct_tokenize(raw)
tokens = tokens[20:1834]
text = nltk.Text(tokens)

"## Text -> Vocab ##"
"Normalize the words, build the vocabulary"
words = [w.lower() for w in text]
vocab = sorted(set(words))


>>>> 1. "Access from Electronic Books"
from urllib import urlopen
url = "http://www.xxxxx.org/xxx/xxx/xxxx.txt"
raw = urlopen(url).read()
type(raw) 
"<type 'str'>"
len(raw)
"1127651"
raw[:75]
"The project is ......"
raw.find("PART I") # find index of start letter position 
"5031"
raw.rfind("End of Project Gutenberg's Crime")
"1157681"
raw = raw[5031:1157681] # indexing
raw.find("PART I")
"0" # first position after indexing

*"If need to use proxy"
proxies = {'http': 'http://www.proxy.com:3128'}
raw = urlopen(url, proxies=proxies).read()

*"Tokenization"
tokens = nltk.word_tokenize(raw)
type(tokens)
"<type 'list>"
tokens[:75]
"['The','project','is',....]"

*"Futher process it into NLTK corpus"
text = nltk.Text(tokens)
type(text)
"<type 'nltk.text.Text'>"
text[1020:1060]
"['xxxx','xxxx','xxxx',....]"
text.collocations() # Can use all funs in previous chap for corpus
"xxxx xxxxx xxxx xxxx"



>>>> 2. "Dealing with HTML"
url = "http://xxxxx.xx/xxx/xxxx/xxx.stm" # web page
html = urlopen(url).read()
html[:60]
"<!doctype html public '//xx/xxx//xxx'>"
raw = nltk.clean_html(html)
tokens = nltk.word_tokenize(raw)
tokens
"['Header','|',xxxx','|',xxxxx','xxxxxx',....]"
tokens = tokens[94:399]
text = nltk.Text(tokens)
text.concordance('Gene')
".......... Gene .............."
".......... Gene .............."



>>>> 3. "Accessing Search Engine Results"
*"Advantages - Size, its more likely to find meaningful patterns"
*"Disadvantages - Easy to use, usually provided with very convienent tool - API"
*"Shortcoming - search range usually very limited; Result unstable, different given different time and critiria"

" Google Search data API "



>>>> 4. "Access RSS Feeds - blogs"
*"Python library - Universary feed Parser, can access the content of a blog"
import feedparser
llog = feedparser.parse("http://languagelog.ldc.upenn.edu/nll/?feed=atom")
llog['feed']['title']
"u'Language Log'"
len(llog.entries)
"15" # 15 blogs
post = llog.entries[2]
post.title
"u'He's My BF'"
content = post.content[0].value
content[:70]
"u'<p>Today I was chatting ................. "
nltk.word_tokenize(nltk.clean_html(content))
nltk.word_tokenize(nltk.clean_html(llog.entries[2].content[0].value))
"[u'Today',u'is',.....]"

* "u'word' is unicode defined string"



>>>> 5. "Access Local File"
import os
os.chdir('/xxx/xxxx/xxxx/')
f = open('document.txt')
raw = f.read()
raw
"xxx xxxx xxxxx. \nxxxxxxx xxxxxx xxxxx. \nxxxxxx  xxxxxx xxxx."
*"\n means new lines in file. can use 'strip()' to remove them."
*"Use nltk fun to read txt file"
path = nltk.data.find('xxxx/xxx/xxxx.txt')
raw = open(path,'rU').read() # nltk corpus



>>>> 6. "Access from PDF, MSWord, other binary formats"
pypdf, pywin32 "can be used to process those file"
*"Always better to use txt format. So convert to txt before loading is better."



>>>> 7. "Access user input"
s = raw_input("Enter some text: ")
"Enter some text: On a memory day, we went out for lunch."
print "You typed", len(nltk.word_tokenize(s)), "words."
"You typed 11 words."





[2] " Python Strings: Text processing"

word = "xxxxxxxxx"\
       "xxxxxxxxx"
print word
"xxxxxxxxxxxxxxxxxxxxx"

word = ("xxxxxxxxx",
       "xxxxxxxxx")
print word
"xxxxxxxxxxxxxxxxxxxxx"

"xxxxxxx" + "xxxxxxxx"
"xxxxxxx" * 3

- / "Doesn't work on str operation"

print 'xxxx' + 'xxxxx'
print 'xxxxx', 'xxxxxx'

"theworldlist"[0:3] # start from 0
"thew"
"theworldlist"[-4:-1] # strat from -1
"list"

"##### More string operations ######"
s.find(t) # index of first instance of string t inside s (-1 if not find)
s.rfind(t) # index of last instance of string t inside s (-1 if not find)
s.index(t) # index of first instance of string t inside s (rasie ValueError if not find)
s.rindex(t)  # index of last instance of string t inside s (rasie ValueError if not find)
s.join(text) # combine words of text and use 's' in between
s.split(t) # Split a text whenever 't' is appearing
s.splitlines() # Split s into a list of strings, one per line
s.lower() # all to lower
s.upper() # all to upper
s.title() # captialize the first letter of each word
s.strip() # A copy of s without leading/trailing whitespace
s.replace(t,u) # replace t with u in s

* "String is Immutable - can NOT append"
* "List is mutable - can append"





[3] " Text Processing using UNICODE"

" Often deal with different language "

* "Unicode - it supports over a million characters, each character is assigned a number, code point."
'\uXXXX' - "number of four-digit hexadecimal form"

* "Unicode can be process freely as normal string in program. But if need to display in terminal or store in file --> Need to be encoded!" 

* "glyphs (font) - Only glyphs can be print on screen or paper "

"################ Encoding & Decoding ################"

"GB2312" -----------|           |----------- "GB2312"
            decode  |           |   encode
                    |           |
"Latin-2" ----------|           |----------- "Latin-2"
            decode  | "Unicode" |   encode
                    |           |
"UTF-8" ------------|           |----------- "UTF-8"
            decode  |           |   encode
                    |           |
"ASCII" ------------|           |----------- "ASCII"
            
> "File/Terminal"  > "In-memory" >  "File/Terminal"

"#####################################################"


1. "Extracting Encoded Text from files"
path = nltk.data.find('xxxx/xxxx/xxxx.txt')
import codecs
f = codecs.open(path, encoding='latin2')
"......"
f = codecs.open(path,'w',encoding='utf-8')

* "Unicode_escape - is an dummy decoding that coverts all non-ASCII characters into \uXXXX representation"
"string".encode('unicode_escape') # unicode_escape

ord('a') # find integer ordinal of a character using ord()
"97" # the hexadecimal four-digit notation for 97 is 0061
a = u'\u0061'
a
u'a'
print a
'a'

# Non-ASCII
nacute = u'\u0144'
nacute
u'\u0144'
nacute_utf = nacute.encode('utf8')
print nacute_utf # render glyphs
'n;'
print repr(nacute_utf)
'\xc5\x84' # UTF-8 escape sequencces (of form of \xXX)

* "Inspect the propertity of Unicode characters."
import unicodedata
lines = codecs.open(path, encodeing='latin-2').readlines()
line = lines[2]
print line.encode('unicode_escape')

for c in line:
	print '%r %d %s' % (c.encode('utf8'), ord(c), unicodedata.name(c))
"\xc3\xb3 00f3 LATIN SMALL LETTER O WITH ACUTE"

for c in line:
	print '%s %d %s' % (c.encode('utf8'), ord(c), unicodedata.name(c)) # change %r to %s to show glyphs
"o; 00f3 LATIN SMALL LETTER O WITH ACUTE"

** "Using your local encoding in Python"

# ---------- file.py ---------------- #
# -*- coding: utf-8 -*-

import xxxx

xxx = xxxxx
".........."
# ----------------------------------- #






[4] " Regular Expression for detecting Word Patterns "

import re 

# Get sample words
wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]

# Search words match Regex
'$'
[w for w in wordlist if re.search('ed$', w)]
"any word End by ed"

'.'
[w for w in wordlist if re.search('..j..ked$', w)]
"represent an single letter"

'^'
[w for w in wordlist if re.search('^K..j..ked$', w)]
"Start with k"

'?'
[w for w in wordlist if re.search('^K-?..j..ked$', w)]
"- is optional"

'+'
[w for w in wordlist if re.search('^H+M+L+', w)]
"one or moew instance for H, M, L each"

'*'
[w for w in wordlist if re.search('^H*M*L', w)]
"zero or moew instance for H, M each"



** "More ..."

"#####################################################################"
"."              "Wildcard, match any letter"
"^abc"           "Matches some pattern abc at the start of a string"
"abc$"           "Matches some pattern abc at the end of a string"
"[abc]"          "Matches one of a set of letters"
"[A-Z0-9]"       "Matches one of a range of letters"
"ed|ing|s"       "Matches one the specific strings"
"srt(ed|ing|s)"  "Parentheses that indicate the scope of the operators"
"*"              "Zero or more instance of previous letter"
"+"              "One or more instance of previous letter"
"?"              "Zero or one instance of previous letter"
"{n}"            "Exactly n repeats of previous letter"
"{n,}"           "At least n repeats of previous letter"
"{,n}"           "No more than n repeats"
"{m,n}"          "At least m but no more than n repeats"
"#####################################################################"

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



** "avoide mis-interpret strings"

"\band\b" # Mis-interpert
r"\brand\b" # treat as raw strings


** "---------------------------- Useful Application"

1. "Extracting Word Pieces"
# Find vowels
word = 'asdfasefadvbetgbwdvqwefvwefv'
re.findall(r'[aeiou]', word)
"['u','e','a','i','i',......]"

# Find vowels in seuqence of two and present freq table
wsj = sorted(set(nltk.corpus.treebank.words()))
fd = nltk.FreqDist(vs for words in wsj
	                  for vs in re.findall(r'[aeiou]{2,}',word))

fd.items()
"[('io',549),('ea',476),.....]"


# Find words in pattern and glud them together
regexp = r'^[AEIOUaeiou]+|[AEIOUaeiou]'
def compress(word):
	pieces = re.findall(regexp, word)
	return ''.join(pieces)


# Finding Word Stems
def stem(word):
	for suffix in ['ing','ly','ed','ious','ies', 'ive', 'es', 's', 'ment']:
		if word.endswith(suffix):
			return word[:-len(suffix)]
	return word


# Searching Tokenized Text
from nltk.corpus import gutenberg, nps_chat
moby = nltk.Text(gutenberg.words('xxxxxxx.txt'))
moby.findall(r'<a> (<.*>) <man>') # Get three word phrase - 'a xxxx man' and only get (xxxx)
"gentle; good; ...."
chat = nltk.Text(nps_chat.words())
chat.findall(r'<.*> <.*> <bro>') # match three word phrase - 'xxx xxx bro' and get all phrase
"you rule bro; telling you bro; ...."
chat.findall(r'<l.*>{3,}') # Match phrase start with 'l' and followed by any letters not more than 3
"lol lol lol; lmao lol lol; ...."

from nltk.corpus import brown
hobbies_learned = nltk.Text(brown.words(categories=['xxxx','xxxxx']))
hobbies_learned.findall(r'<\w*> <and> <other> <\w*s>') # Matches four words phrase - 'xxxx and other xxxs' and get whole phrase
"speed and other activities; ....... "

** "Regex to tokenize"
raw = "asda asd  fge fv rtg f v sd vs dv sdv sd vdr r gbr f"
re.split(r' ',raw)
raw = "asda asd  fge fv\nrtg f v\nsd vs dv\tsdv sd vdr r gbr f"
re.split(r'[ \t\n]+',raw)
re.split(r'\W+',raw) # Split by non-alphbet

** "NLTK Regex Tokenizer"
text = "That U.S.A poster-print costs $12 ....."
pattern = r'''(?x) # set flag to allow verbose regexps
   ([A-Z]\.)+      # abbrevations like USA
 | (\w+(-\w+)*)    # .................... 
''' 
nltk.regexp_tokenize(text, pattern) 
"['That', 'U.S.A', ....]"



[5] "Normalizing Text"

1. "To a lower form"
set(w.lower() for w in text)

2. 'Stemmers' # retunr to basic form
raw = "asdfsd asdf asdf asdfasd f asdh df gb sdfgsd qwedcvq efb wrvdcv qer "
tokens = nltk.word_tokenize(raw)

porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()

[porter.stem(t) for t in tokens]
[lancaster.stem(t) for t in tokens]


3. "Lemmatization" # A lnown word in dictionary
wnl = nltk.WordNetLemmatizer()
[wnl.lemmatize(t) for t in tokens]


** "List to String / String to List"
' '.join(['xxxx','xxxx',....]) # To string
"xxxxx xxxxx ...........".split(" ") # To list
 
print '%6s' # specify width
print '%2.4f' # float number



[6] "Segmentations"

* "Sentence Segmentation"
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
text = nltk.corpus.gutenberg.raw('xxxxxx.txt')
sents = sent_tokenizer.tokenize(text)
sents[12314:7354345]
# Difficult if like U.S.A appears

* "Word Segmentation"
"Calculation - P114"
def segment(text, segs):
	words = []
	last = 0
	for i in range(len(segs)):
		if segs[i] == '1':
			words.append(text[last:i+1])
			last = i+1
	words.append(text[last:])
	return words


def evaluate(text, segs):
	words = segment(text, segs)
	text_size = len(words)
	lexicon_size = len(' '.join(list(set(words))))
	return text_size + lexicon_size

text = "doyou....................."
seg1 = "00100....................."
seg2 .............................
seg3 .............................

segment(text, seg3)
"['doyou',.......................]"

evaluate(text, seg3)
"46" # The small the better










>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " [4]  Python Tips for NLP"


-1 "Python Basic"

[1] "Assignment"

# String/numeric/.. is just a copy so original change doesn't affect copies
foo = 'Monty'
bar = foo
foo = 'Python'
bar
'Monty'

# structured data - list/array/... is a reference to original so it affacts copies
foo = ['s','a']
bar = foo
foo[1] = 'k'
bar
"['s','k']"

# structured data - ex 2 - change only one element in a list but change all since they all refer the same original
empty = []
nested = [empty, empty, empty]
nested
"[[],[],[]]"
nested[1].append('s') # change 1 element to other value in a list
nested
"[[s],[s],[s]"


# structure data - ex 3 - change only one element in a list not all since they all refer the different originals
nested = [[]]*3 # 3 different [] originals
.....
.....
nested
"[[],[s],[]]"






[2] "Equality"

"Two ways to check equality - 'is' | '==' "

# USe == (values are identical??)
A = 'Python'
B = 'Python'
A == B # any A,B values
"Ture"


# Use is (same object??)
A is B
"False"
C = 'Python'
CC = [C,C,C]
CC[0] is CC[1]
"True"






[3] "Conditionals"

# If If or If elif

if xxxx: # Evaluated no matter what
	xXX 
if xxxx: # Evaluated no matter what
	xxx


if xxxx: # Evaluated
	xxxx
elif xxxx: # Not Evaluated if first meets
	xxxx


# Test a vector of bloo
all(True, True, False)
"False"
any(True, True, False)
"True"






-2 "Sequences"

"Strings" | "lists" | "tuple"

A = 'string1'
B = [1,2,3]; B = list()
C = (1,2,3); C = tuple(); C = 1,2,3


[1] "Operations on Sequences types"

for item in s                         "Iterate over the item of s"
for item in sorted(s)                 "Iterate over the item of sorted s"
for item in set(s)                    "Iterate over the item of set of s"
for item in reversed(s)               "Iterate over the item of reversed s"
for item in set(s).difference(t)      "Iterate over the item of difference of s to t"
for item in random.shuffle(s)         "Iterate over the item of shuffled s"


# Convert FreqDist to list
list(fdist)
"['word1',',','word2',...]"
for key in fdist:
	print fdist[key]
"4,3,2,1,1,...."


# Re-arrange list
words = ['went','i','home','to','.']
words[0], words[1], words[2], words[3], words[4] = words[1], words[0], words[3], words[2], words[4] 
words
"['I', 'went', 'to', 'home', '.'']"


# Zip
A = ['a','b','c']
B = [1,2,3]
zip(A,B)
"[('a',1),('b',2), ...]"

# Enumerate
list(enumerate(A))
"[('a',0),('b',1), ...]"


# Cut the dataset by 90% and 10% for train
cut = int(0.9*len(text))
train, test = text[:cut], text[cut:]
len(train) / len(test)
"9"




[2] "Combining Different Sequences Types"
 words = 'i did something and did some other things'.split()
 wordlens = [(len(word), word) for word in words] # Use mix of two record lens and words
 wordlens.sort()
 ' '.join(w for (_, w) in wordlens) # use '_' represent as convention for the part we don't want

 # Example
 lexicon = [
    ('ss','xx',['dd','pp']),
    ('ff','oo',['dd','uu'])
 ]




[3] "Generator Expressions"

'Normal'
max([w.lower() for w in nltk.word_tokenize(text)]) # first save a object list as the result, then calculate the max from that list object

'exp'
max(w.lower() for w in nltk.word_tokenize(text)) # directly pass result to calculate max - faster





-3 "Questions of Style"

[1] "Python Coding Style"

**"Four spaces for 1 indention level, avoid tab which may confused in different software"

**"Want to break a line"
# add extra ()
if (xxxxxxxxxxx 
	xxxxxxxxxxxxxxxxx):
# use \
if xxxxxxxx \
   xxxxxxxxxxxxxxxxxx:




[2] "Procedural Versus Declarative Style"

"Procedural Style: dicating machine step by steo operation - CPU keep registering objects most meaningless till the end"
tokens = nltk.corpus.xxxxxx
count = 0
total = 0
for token in tokens:
	count += 1
	total += len(token)
print total / count



"Declarative Style: use build-in program on a high abstract level - faster"
total = sum(len(t) for t in tokens)
print total / len(tokens)




[3] "Some Legitimate Uses for Counters"

sent = ['The', 'dog', 'gave', 'John', 'the', 'newspaper']
n = 3
[sent[i:i+n] for i in range(len(sent)-n+1)] # n-gram iterator

"[['the','dog','gave'],"
" ['dog','gave','jhon'],"
"......................"

# Used build-in
bigrams(text); trigrams(text); ngrams(text,n)


# create nested structure using counter
m, n = 3, 7
array = [[set() for i in range(n)] for j in range(m)] # n X m structure
array[2][5].add('A') # only change one value

"if"
array = [[set()]*n]*m
array[2][5].add('A') # all value since same original







-4 "Functions: The Foundation of Structured Programming"

[1] "Function Inputs and Outputs"

** "parameter is not always required"

** "Return values -- never give function two side effects"
def my_sort1(mylist): # Good - modified argument so no return
	mylist.sort()

def my_sort2(mylist): # Good - not modified argument so return
	return sorted(mylist)

def my_sort3(mylist): # Bad - modified and return (People may not realized)
	mylist.sort()
	return mylist



[2] "Parameter Passing"


** "value and structured data assignment is same in function"

def my_func(word, property1): # in func, firs para will be assigned as word, second will be assigned as property1 (Like assignment)
	word = 'local'
	property1.append('A')
	property1 = 5 # 

w = ''
p = []

my_func(w,p) # let function modify w,p
w
'' # unchange - just a copy, no impact on original
p
"['A']" # changed - straured data will be modified original



[3] "Variable Scope"

"LGB rules" -- "local" < "global" < "built-in" 
"Python will first search from local then ...."

** "var decalred in fun is local"
** "var declared outside fun is gloabl"
** "Can esclated local var to global"
global local_var




[4] "Checking Parameter Types"

** "Same error due to wrong para type may not being noticed - need assert (Defensive Programming)"

def tag(word):
	assert isinstance(word, basestring), "argument to tag() must be a string"
	..............
	..............



[5] "Functional Decomposition"

** "Use function to block each process"
** "nest them to use generator expression"

words = func1(func2(func3(text)))



[6] "Documenting Functions"

** "Use 'doctest block' - P149"
def my_func():
	"""
    doctest block..
    <introduction>
    @para1
    @para2
	"""
	..............
	..............



-5 "Doing More with Functions"

[1] "Functions As Arguments"

** "Not only values and structred data, function can also takes function as argument"
sent = [................]
def my_func(func):
	return [func(word) for word in sent]

my_func(len)
my_func(lambda x: x+1)




[2] "Accumulative Functions"

** "Generator vs Iterator"
"Generator -- only generates the data, no storage output -- generator expression"

"Iterator"
def search1(substring, words):
	result = []
	for word in words:
		if substring in word:
			result.append(word)
	return result

"Generator"
def search2(substring, words):
	for word in words:
		if substring in word:
			yield word




[3] "Higher-Order Functions"

"Higher-order function are standard features of functional programming languages"

filter() # Which apply function to each item in the sequence contained in its second
         # parameter, and retains only the items for which the function return True
def my_func(word):
	return word.lower() not in ['a','of','the','and','will']
text = ['Take','the',...............]
filter(my_func,text)

# Normal
[w for w in text if my_func(w)]



map() # Which apply a function to every item in a sequence.
map(len, nltk.corpus.brown.sents(categories='news'))
map(lambda x: x + 'S', text)

# Normal
[len(w) for w in nltk.corpus.brown.sents(categories='news')]



[4] "Named Argument"

** "Named (assign default value) parameters can be in any order and omitted"
def my_func(arg1=1, arg2=4):
	.......................
my_func(arg2=5,arg1=10)


** "Unnamed must be fixed in positions"
def my_func(arg1,arg2):
	......................
my_func(10,5)

** "If use named and unnamed together - unnamed first than named"

** "define arbitrary of unnamed and named parameters"
def my_func(*args, **kwargs): # *args -- arbitrary number of args in a list[]  ;  **kwargs -- arbitrary number of args in a dict{}
	print args 
	print kwargs



-6 "Program  Development"

"Describe the internal structure of a program module and how to organize a multi-module program."



[1] "Structure of a Python Module"

** "Python modules are nothing more than individual .py|.pyc file"
help(nltk.metrics.distance) # distance.py

# Oneline title of module and identifying the authors
# Module level docstring

** "Other modules builds 'classes' main building blocks of functional programming"



[2] "Multimodule Programs"

# ------- M1.py --------- #
def func_m1():
	...............
# ----------------------- #


# ------- M2.py --------- #
def func_m2():
	...............
# ----------------------- #


# ------- M-master.py -------- #
from M1 import func_m1
from M2 import func_m2

def func_master():

	func_m1()....
	func_m2()....
# ---------------------------- #



[3] "Sources of Error"

** "Input data may contains some unexpected characters. -- P157"
** "Supplied function might not behave as expected. -- P157"
** "Our understanding of Python semantics may be faulty -- P157"


[4] "Debugging Techniques"

** "If program produced an 'exception' -- a runtime error -- the interpreter will print stack"

** "Can use debugger"
import pdb
pdb.run("my_func()")
"P-158"


[5] "Defensive Programming"

" In order to aviod some pain of debugging "

** "Build the small pieces first which are known to work"
** "Use 'assert' statement in your code "
** "Keep record / can undo changes"
** "As develop program, extend its functionality, fix any bugs, main a suit of test case -- regression testing"
   doctest



-7 "Algorithm Design"

"selecting an appropriate algorithm for the problem at hand. -- performance with the increase of data scale."

** "Best-known strategy -- 'divide-and-conquer' | we divide a probelm of size n into 2, combine result is easier. (ex. binary search)"
** "Another strategy -- 'transforming' | we transforming current problem into a know problem to solve. "


[1] "Recursion - 'divide-an-conquer' "
def factorial1(n):
	if n == 1:
		return 1
	else:
		return n * factorial1(n-1)
** "navigate deep nested network -- such as word-net"


[2] "Space-Time Trade-offs"
** "We can speed up the data process by using auxiliary data"
# Using index
nltk.Index() # faster lookup

# Using replace token words into integers
def preprocess(tagged_corpus):
   words = set()
   tags = set()
   for sent in tagged_corpus:
       for word, tag in sent:
           words.add(word)
           tags.add(tag)
    wm = dict((w,i) for (i,w) in enumerate(words))
    tm = dict((t,i) for (i,t) in enumerate(tags))
    return [[(wm[w], tm[t]) for (w,t) in snet] for sent in tagged_corpus] 

# check vocabulary set using set rather than list is faster
set(.....) > list(......)

** "Timer run"
from timeit import Timer
"Timer(a statement excuted multiple times, setup code excuted once)"
set_up = "import random; vocab = range(%d)" % vocab_size
statement = "random.randint(0,%d) in vocab" % vocab_size * 2
print Timer(statement, set_up).timeit(1000) # simulate 1000 times and time it
"2.78" # seconds



[3] "Dynamic Programming"

**"When a problem contains overlapping sub-problems, instead of computing solution of the overlapping problem repeatly"
  "we simply store them in a lookup table."

"Example. - S = 1, L = 2 | if 4 how many different combinations?"

def func3(n, lookup={0:[''], 1:['s']}):
	if n not in lookup:
		s = ['S' + p for p in func3(n-1)]
		l = ['L' + p for p in func3(n-2)]
		lookup[n] = s + l
	return lookup[n]

func3(4)
"['SSSS','SSL','SLS','LSS','LL']"



-8 "Main Libraries for NLP"

<Matplotlib>: # visualization

<NetworkX>: # visualizing word-net

<csv>: # manipulate tabeaulate data in csv

<NumPy>: # numeric computing




































>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " [5]  Categorizing the Tagging Words"

" Word class - nouns, verbs, adjectives, adverbs, etc ---- How to tag words by their word classes."

** "Part-of-Speech (POS) tagging | word class - lexical categories | collection of tags - tagset"



-1 "Using a Tagger"
# Use a nltk.pos tagger example
import nltk
text = nltk.word_tokenize("xxxx xxx xxx xxxx")
nltk.pos_tag(text)
"[(xxxx, NN),(xxxxx, VBP),......]"

# Finds ws appears in the similiar context
text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
text.similar('woman')
"man time day .........."




-2 "Tagged Corpora"

" A tagged token is represented using a tuple consisting of token and the tag. "

[1] "Representing Tagged Tokens"
# Creating this representation
tagged_token = nltk.tag.str2tuple('fly/NN')
tagged_token
"('fly','NN')"
tagged_token[0]; tagged_token[1]
"fly"; "NN"

Text = 'fly/NN is/V the/TO'
[nltk.tag.str2tuple(t) for t in Text.split()]
"[('fly','NN'),('is','V'),..........]"



[2] "Reading Tagged Corpora"
" nltk corpus is tagged"
nltk.corpus.brown.tagged_words()
"[(),(),..]"
nltk.corpus.brown.tagged_words(simplify_tags=True) # Simplified POS Tagsets
"[(),(),..]"




[3] "Simplified POS Tagset"

" Tag        Meaning              Examples     "
" ----------------------------------------     "
" ADJ        adjective            new,good,high"
" ADV        adverb               really, already"
" CNJ        conjunction          and, or, but   "
" DET        determiner           the,a,some,every"
" EX         existential          there, there's  "
" FW         foreign words        ??????          "
" MOD        modal verb           will,can,would,may"
" N          noun                 year,home,costs "
" NP         proper noun          Alison, April, Washington"
" MUM        number               four,five "
" PRO        pronoun              he,their,her,its "
" P          preposition          on,of,at,with,by "
" TO         the word to          to "
" UH         interjection         ah,bang,ha,whee,oops "
" V          verb                 is,has,get,do,make "
" VD         past tense           said,took,told "
" VG         present participle   making,going,playing "
" VN         past participle      given,taken,sung "
" WH         wh determiner        who,which,when,what,where "


** "Nouns"

"P-184"

** "Verbs"

"P-185"

** "Adjective & Adverbs"

"P-186"




[4] "Unsimplified Tagset"

"Different variations of the word category"

# Example
"NN" - "Noun" - "Year"
"NN$" - "possessive nouns" - "Year's" 
"NNS" - "plural nouns" - "Years"
"NN-HL" - "in headlines" - "Year xxxxx"
"NN=TL" - "for Title" - "Year xxxxxxx"

"P-187"



[5] "Exploring Tagged Corpora"

**" Study the word 'often' and see how it is used "
brown_learned_text = brown.words(categories='learned')
sorted(set(b for (a,b) in nltk.ibigrams(brown_learned_text) if a == 'often'))
"[',','.','accomplished','call',....]"

**" Firther study tags in the word "
brown_lrnd_tagged = brown.tagged_words(categories='learned', simplify_tags=True)
tags = [b[1] for (a,b) in nltk.ibigrams(brown_lrnd_tagged) if a[0] == 'often']
fd = nltk.FreqDist(tags)
fd.tabulate()
"VN   V   ...."
"15   12  ...."

**" find words invloving particular sequences of tags and words. "
from nltk.corpus import brown
def process(sentence):
	for (w1,t1), (w2,t2), (w3,t3) in nltk.trigrams(sentence):
		if (t1.startwith('V') and t2 == 'TO' and t3.startwith('V')):
			print w1, w2, w3

for tagged_sent in brown.tagged_sents():
	process(tagged_sent)
"combined to achieve"
"continue to place"
"................."

**" Ambiguous words, why such words are tagged, help clarify the distinctions between tags"
brown_news_tagged = brown.tagged_words(categories='news', simplify_tags=True)
data = nltk.ConditionalFreqDist((word.lower(),tag) for (word, tag) in brown_news_tagged)
for word in data.conditions():
	if len(data[word]) > 3:
		tags = data[word].keys()
		print word, ' '.join(tags)
"best ADJ ADV ..."
"better ADJ ADV V ....."
"......................"




-3 "Python: Mapping values using Dictionary"


** "Basic Methods"

" Example                           Description "
" d = {}                            craete a dict"
" d[key] = value                    Assign a value to a key"
" d.keys()                          the list of keys "
" d.values()                        the list of values"
" list(d)                           the list of keys "
" sorted(d)                         The keys sorted "
" key in d                          test if a key in dict "
" for fey in d                      iterate over keys in dict "
" dict([(k1,v1),(k2,v2),...])       Create a dict from a list of tuples "  
" d1.update(d2)                     add all items from d2 to d1 "
" defaultdict(int)                  A dictionary whose default value is 0 " 


# Dictionary in Python
pos = {}
pos['colorless'] = 'ADJ'
pos['ideas'] = 'N'
pos['sleep'] = 'V'
pos
"{'colorless: ADJ', 'ideas': N, 'sleep': V}"
pos['ideas']
"N"
pos.keys()
"['coloerless', 'furours',....]"
pos.values()
"['ADJ',......................]"
pos.items()
"[('colorless','ADJ'),('ideas','N'),()..........]"
pos['sleep'] = ['N','V'] # can multiple 


# To just find the keys
list(pos)
"[ideas, sleep, .....]"
sorted(pos)
"['coloerless', 'furours',....]" 
[w for w in pos if w.endwith('s')]
"['colorless', 'ideas']"


# Use for loop prnting lists
for word in sorted(pos):
	print word + ":", pos[word]
" colorless: ADJ "
" .............. "



# Define a dictionary
pos = {'colorless': 'ADJ', 'ideas': 'N'}
pos = dict(colorless='ADJ',ideas='N')
** "Key in dict must be immutable -- string, tuple"
# Use default dict
pos = nltk.defaultdict(int)
pos['colorless'] = 4
pos['ideas']
0
pos = nltk.defaultdict(int("2"))
pos['colorless'] = 4
pos['ideas']
2
pos = nltk.defaultdict(list)
pos['colorless'] = ['N','V']
pos['ideas']
"[]"
pos = nltk.defaultdict(list("8"))
pos['colorless'] = ['N','V']
pos['ideas']
"[8]"
pos = nltk.defaultdict(lambda: 'N') # function with no argument
pos['colorless'] = ['N','V']
pos['ideas']
"N"
** "Example -- Preprocess a text to replace low-frequency words with a special token 'UNK'"
alice = nltk.corpus.gutenberg.words('carroll-alice.txt')
vocab = nltk.FreqDist(alice) # Sort dict with high freq
v1000 = list(vocab)[:1000] # top 1000 high freq words
mapping = nltk.defaultdict(lambda: 'UNK') # default dict replace with UNK
for v in v1000:
	mapping[v] = v # Create dict with top 1000 valued dict
alice2 = [mapping[v] for v in alice] # use original words in the text to get the value from the dict
                                     # words not in top 1000 freq will be return as 'UNK'
"['UNK','Alice',.....................]" 



# Incrementally Updating a Dictionary
"Each time encounter a tag increment by 1"
counts = nltk.defaultdict(int)
from nltk.corpus import brown 
for (word, tag) in brown.tagged_words(categories='news'):
	counts[tag] += 1
count['N']
22226
list(count)
"['FW','DET',..................]"

from operator import itemgetter
sorted(counts.items(), # dict items [(A,B),...]
	   key=itemgetter(1), # Sorted by the second element (0,1)
	   reverse=True) # From large to small
"[('N',22226), ('WH',10865), .............. ]"
[t for t,c in sorted(counts.items(), # dict items [(A,B),...]
	                 key=itemgetter(1), # Sorted by the second element (0,1)
	                 reverse=True)]
"['N','WH',.................................]"



# Create dict with key descriping word like end by 'ly'
last_letters = nltk.defaultdict(list)
words = nltk.corpus.words.words('en')
for word in words:
	key = word[-2:] # last 2 letters
	last_letters[key].append(word) # default a list can append 

last_letters['ly']
"['xxxxly', 'xxxxxly',..................]"
last_letters['zy']
"['xxxxzy', 'xxxxxzy',..................]"




# Complex Keys and Values
" Study the range of possible tags for a word, "
" given the word itself and the tag of previous word."

pos = nltk.defaultdict(lambda: nltk.defaultdict(int))
brown_news_tagged = brown.tagged_words(categories='news', simplify_tags=True)
for ((w1,t1),(w2,t2)) in nltk.ibigrams(brown_news_tagged):
	pos[(t1, w2)][t2] += 1

pos[('DET','right')]
"defaultdict(<type 'int'>, {'ADV':3, 'ADJ':9, 'N':3})"




# Inverting a Dictionary
" Finding a key given a value -- slow"
[key for (key, value) in dict1.items() if value == 32]
"['key1','key4',...............]"

**" If we do often -- better convert key to value, value to key"
pos2 = dict{(value, key) for (key, value) in pos.items()} # one value to one key

pos2 = nltk.defaultdict(list) # multiple keys to same value
for key, value in pos.items():
	pos2[value].append(key)

pos2['ADV']
"['peacefully','furiously']"






-4 "Automatic Tagging"

" Various way to automatically add POS tags to text. "
" We will see the tag of a word depends on the word and its context within a sentence. "

**" So we will be working with tagged sentence."
from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')



[1] "Default Tagger"
" Assign same tag to every text -- with the most likely tag in a text (basic safe)"
tags = [tag for (word, tag) in brown.tagged_words(categories='news')]
nltk.FreqDist(tags).max()
'NN'
raw = 'I do not like .....................'
tokens = nltk.word_tokenize(raw)
default_tagger = nltk.DefaultTagger('NN') # all to 'NN'
default_tagger.tag(tokens)
"[('I','NN'),('do','NN'),................]"
default_tagger.evaluate(brown_tagged_sents)
0.13089 # Poorly perform if only




[2] "The Regular Expression tagger"
"It assigns tags to tokens on the basis of matching patterns. Ex. -ed very likely past"
patterns = [
   (r'.*ing$', 'VBG'),
   (r'.*ed$', 'VBD'),
   (r'.*es$', 'VBZ'),
   (r'.*ould$', 'MD'),
   (r'.*\'s$', 'NN$'),
   (r'.*s$', 'NNS'),
   (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),
   (r'.*', 'NN')
]
regexp_tagger = nltk.RegexpTagger(patterns)
regexp_tagger.tag(tokens)
regexp_tagger.evaluate(brown_tagged_sents)
0.20326





[3] "The lookup tagger"
" Find the top most freuqnet words and store the most likely tag, then use it as lookup to tag"
fd = nltk.FreqDist(brown.words(categories='news'))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
most_freq_words = fd.keys()[:100]
likely_tags = dict((word, cfd[word].max()) for word in most_freq_words)
baseline_tagger = nltk.UnigramTagger(model=likely_tags) # use uni-gram tagger
baseline_tagger.evaluate(brown_tagged_sents)
0.45235
baseline_tagger.tag(sent)
"[('only','None'),(),..............]" # 'None' means word not in top freq list
** "Backoff process -- Lookup tagger, if no tag then, default tagger"
baseline_tagger = nltk.UnigramTagger(model=likely_tags,
	                                 backoff=nltk.DefaultTagger('NN'))




[4] "Evaluation"

"Emphasis on accuracy score - we evaluate the performance of an tagger relative to a tagging a human expert would assign"
"With an 'Golden Standard Test Data' "

-5 "N-Gram Tagging"

[1] "Uni-Gram Tagging"

"Unigram tagger depends on simple statistical algorithm: for each token, assign the tag that is most likely for that particular token. "
" Behavior like a lookup tagger -- except set up lookup by 'training' which specifying tagged sentence data as a parameter when we initialize the tagger. "
from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')

size = int(len(brown_tagged_sents)*0.9)
size
4160
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
unigram_tagger = nltk.UnigramTagger(train_sents)
unigram_tagger.evaluate(test_sents)
0.81202 # high performance



[2] "General N-Gram Tagging"

" Different from 'unigram' which consider only the token, isolated with larger context, same token regardless context will be tagged the same all the time"

" An N-gram tagger is a generalization of unigram tagger whose context is the current word together with the POS tags of the n-1 preceding tokens."

**"Ngram should not consider words across sentnces. Tagger are design to work with list of sentences."
**"Sparse Data -- If data never saw in training, shows 'none'. As n gets larger, context increases so as the chance a wish tag content not in training set. "
  "As consequence - there is trade-off between the accuracy and the coverage of our results (precision/recall trade-off)"

# unigram
Unigram_tagger = nltk.UnigramTagger(train_sents)

# bigram
Bigram_tagger = nltk.BigramTagger(train_sents)
Bigram_tagger = nltk.BigramTagger(train_sents, cutoff=2) # discard records only shows up 2 times (limit data into tagger train)

# Trigram
Trigram_tagger = nltk.TrigramTagger(train_sents)

# Ngram
Ngram_tagger = nltk.NgramTagger(train_sents,n)



[3] "Combine Taggers"
**"One way to address - 'prescision/recall trade-off' -- use more accurate algorithms when we can"
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
t2.evaluate(test_sents)
0.8449



[4] "Tagging Unknown Words"

Default_tagger | Regular_Expression_tagger

**"Out-Of-Vocabulary"

"Also, tag most freuqunet words, then use Unigram -- usually noun, or use Ngram -- context tags to define the tags"

Ngram_tagger




[5] "Storing Taggers"
from cPickle import dump
onput = open('t2.pkl', 'wb')
dump(t2, output, -1)
output.close()

from cPickle import load
input = open('t2.pkl', 'rb')
tagger = load(input)
input.close()

tagger.tag(tokens)




[6] "Performance Limitations"

**"Measure ambigurity"
cfd = nltk.ConditionalFreqDist(
           ((x[1],y[1],z[0]), z[1])
           for sent in brown_tagged_sents
           for x, y, z in nltk.trigrams(sent)) # trigram data (if train)
ambiguous_contexts = [c for c in cfd.conditions() if len(cfd[c]) > 1]
sum(cfd[c].N() for c in ambiguous_contexts) / cfd.N()
0.049 # 1 out of 20 trigram is ambiguous


**"Mistake Analysis"
test_tags = [tag for sent in brown.sents(categories='editorial')
                 for (word, tag) in t2.tag(sent)]
gold_tags = [tag for (word, tag) in brown.tagged_words(categories='editorial')]
print nltk.ConfusionMatrix(gold, text)


**"Gold Standard with human measures"
"Test Sets"




[7] "Tagging Across Sentence Boundaries"

**"Make sure to train using lists of sentence"
brown_tagged_sents = brown.tagged_sents(categories='news')

size = int(len(brown_tagged_sents) * 0.9)
train = brown_tagged_sents[:size]
test = brown_tagged_sents[size:]

t1 = nltk.BigramTagger(train)




-6 "Transformation-Based Tagging"

**"Ngram tagger needs large data stored, also only consider tags of the context"

"Brill Tagging: guess the tag of each word, then go back and fix the mistakes. Transform a bad tagging into a better one. -- Supervised learning method"
" It develops rules to tag"
nltk.tag.brill.demo()
" P-210 "




-7 "How to determine the word categories of a word"

"How do we decide what category a word belong to in the first place?"

[1] "Morphological Clues"
** "The internal structre of the words -- 'ness', 'ment',etc"

[2] "Syntactic Clues"
** "The contxt of the word -- 'noun' usually after a 'ADJ' "

[3] "Sematic Clues"
** "The meaning of the word is useful too. 'the name of a person'. However, it raises suspicious because hard to formalize."

[4] "New Words"
** "Open class -- noun -- many new words showing up"
   "Close class -- preposition, .. -- ex. above, along, as, ...." 

[5] "Morphology in POS TagSets"
** "morphosyntactic information captured by tags"
   "Go, Goes, Gone, Went -- morphosyntactical Analysis" 

















































>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " [6]  Learning to classify Text"

 "Detecting pattern is the central part of NLP automatically. Using models"



 -1 "Supervised Classification"


***[1]" --- Chooing the right Features --- "
" Choose what features to endcode can impact greatly on the results "
" usually try-and-go fashion "
def gender_features2(name):
	features = {}
	features["firstletter"] = name[0].lower()
	features['lastletter'] = name[-1].lower()
	return features

** "Error Analysis -- initial features, then check what cause error, create more features"









***[2]" --- Gender Identification --- "
def gender_features(word):
	return {'last_letter':word[-1]}
gender_features('Shrek')
"{'last_letter':'k'}"

from nltk.corpus import names
import random
names = ([(name, 'male') for name in names.words('male.txt')] + 
	     [(name, 'female') for names.words('female.txt')])
random.shuffle(names)

featuresets = [(gender_features(n),g) for (n,g) in names] # --- Normal process
train_set, test_set = featuresets[500:], featuresets[:500]

from nltk.classify import apply_features
train_set = apply_features(gender_features, names[500:]) # --- Process large scale data
Test_set = apply_features(gender_features, names[:500])


classifier = nltk.NaiveBayesClassifier.train(train_set)
classifier.classify(gender_features('Neo'))
'male'
classifier.classify(gender_features('Trinity'))
'female'

print nltk.classify.accuracy(classifier, test_set)
0.758

classifier.show_most_informative_features(5)
" last_letter = 'a'     female : male = 38.3 : 1.0 "
" ................................................."








 ***[3]" --- Document Classification --- "

" Classifier that automatically tag new documents "

from nltk.corpus import movie_reviews
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)


all_words = nltk.freqDist(w.lower() for w in movie_reviews.words())
word_features = all_words.keys()[:2000] # Top 2000 freq words

def document_features(document):
	document_words = set(document)
	features = {}
	for words in word_features:
		features['contains(%s)' % word] = (word in document_words)
	return features

print document_features(movie_reviews.words('pos/cv957_8737.txt'))
"{'contains(waste)': False, 'contain(lot)': False, ...}"

featuresets = [(document_features(d),c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print nltk.classify.accuracy(classifier, test_set)
0.81
classifier.show_most_informative_features(5)
" contains(outstanding) = True          pos : neg = 11.1 : 1.0 "
" contains(seagal) = True               pos : neg = 7.7  : 1.0 "
" ............................................................ "








 ***[4]" --- POS tagging --- "

" Instead, we can train a classifier to work out which features are most informative. "

from nltk.corpus import brown
suffix_fdist = nltk.FreqDist()
for word in brown.words():
	word = word.lower()
	suffix_fdist.inc(word[-1:])
	suffix_fdist.inc(word[-2:])
	suffix_fdist.inc(word[-3:])

common_suffixes = suffix_fdist.keys()[:100]
print common_suffixes
"['e',',','.','s','d', ..... ]"

def pos_features(word):
	features = {}
	for suffix in common_suffixes:
		features['endswith(%s)' % suffix] = word.lower().endwith(suffix)
	return features


tagged_words = brown.tagged_words(categories='news')
features = [(pos_features(n), g) for (n,g) in tagged_words]

size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]

classifier = nltk.DecisionTreeClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)
0.627

classifier.classify(pos_features('cats'))
"NNS"

print classifier.pseudocode(depth=4)
" if endwith(,) == True: return ',' "
" if endwith(,) == False: "
"   if ............................. "



# Exploiting Context
" Feature Development "
" Can leverage a variety of other word-internal features: length of word, number of syllables, its prefix "

def pos_features(sentence, i):
	features = {'suffix(1)': sentence[i][-1:], # feature1
	            'suffix(2)': sentence[i][-2:], # feature2
	            'suffix(3)': sentence[i][-3:]} # feature3
	if i == 0:
		features['prev-word'] = "<START>"
	else:
		features['prev-word'] = sentence[i-1]  # feature4
	return features

pos_features(brown.sent()[0], 8)
"{'suffix(3)': 'ion', 'prev-word': 'an', 'suffix(2)': 'on', 'suffix(3)': 'n'}"

tagged_sents = brown.tagged_sents(categories='news')
featuresets = []
for tagged_sent in tagged_sents:
	untagged_sent = nltk.tag.untag(tagged_sent)
	for i, (word, tag) in enumerate(tagged_sent):
		featuresets.append((pos_features(untagged_sent, i), tag) )

size = int(len(features) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)

nltk.classifier.accuracy(classifier, test_set)
0.78915









 ***[5]" --- Sequence Classification --- "

 " in order to capture the dependencies between related classification tasks, we can use 'joint classifier' models, "
 " which choose an appropriate labeling for a collection of related inputs. "

 " Use sequence classifier to jointly choose POS tags for all the words in agiven sentence. "

 ** " Consecutive classification --> find the most likely class label for the first input, "
    " then to use that answer to help label for the next input. The process can be repeat till all being labeled. "

def pos_features(sentence, i, history):
	features = {'suffix(1)': sentence[i][-1:], # feature1
	            'suffix(2)': sentence[i][-2:], # feature2
	            'suffix(3)': sentence[i][-3:]} # feature3
	if i == 0:
		features['prev-word'] = "<START>"
		features['prev-tag'] = "<START>"
	else:
		features['prev-word'] = sentence[i-1]  # feature4
		features['prev-tag'] = history[i-1]    # feature5

	return features

class ConsecutivePosTagger(nltk.TaggerI):

	def __init__(self, train_sents):
		train_set = []
		for tagged_sent in train_sents:
			untagged_sent = nltk.tag.untag(tagged_sent)
			history = []
			for i, (word, tag) in enumerate(tagged_sent):
				featureset = pos_features(untagged_sent, i, history)
				train_set.append((featureset, tag))
				history.append(tag)
		self.classifier = nltk.NaiveBayesClassifier.train(train_set)


    def tag(self, sentence):
    	history = []
    	for i, word in enumerate(sentence):
    		featureset = pos_features(sentence, i, history)
    		tag = self.classifier.classify(featureset)
    		history.append(tag)
    	return zip(sentence, history)


tagged_sents = brown.tagged_sents(categories='news')
size = int(len(tagged_sents) * 0.1)
train_sents, test_sents = tagged_sents[size:], tagged_sents[:size]

tagger = ConsecutivePosTagger(train_sents)
print tagger.evaluate(test_sents)
0.7979

***" Other models for sequence classification --- Hidden Markov Models "












 ** "Further Example of Supervised Classification"

 [1] "Sentence Segmentation"
" Decides whether a sentence ends -- ex. '.', '?' "
# 1. Obtain some data that has already been segmented into 
#    sentence and convert it into a form is suitable for extracting features
sents = nltk.corpus.treebank_raw.sents()
tokens = []
boundaries = set()
offset = 0
for sent in nltk.copus.treebank_raw.sents():
	tokens.extend(sent) # merged list of tokens
	offset += len(sent) # index of all sentence-boundary
	boundaries.add(offset-1)

# Define the feature whether puncuation indicates a sentence bounary
def punct_features(tokens, i):
	return {'next-word-capitalized': tokens[i+1][0].isupper(), # Feature1
	        'prevword': tokens[i-1].lower(),                   # Feature2
	        'punct': tokens[i],                                # Feature3
	        'prev-word-is-one-char': len(tokens[i-1]) == 1}    # Feature4

# We can create a list of labeled featuresets by selecting all the puncuation tokens
# and tagging whether they are bounary tokens or not
featuresets = [(punct_features(tokens, i), (i in boundaries))
               for i in range(1, len(tokens)-1)
               if tokens[i] in '.?!']

# using those features to train classifier evaluate punctuations
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)
0.9741








 [2] "Identifying Dialogue Act Types"
"When processing dialogue, it can be useful to think of utterances as a type of action performed by the speaker."
"Greetings, questions, answer, assertions, or clarification can all be thought of as type of Speech-based Actions"
# getting the post text
posts = nltk.corpus.nps_chat.xml_posts()[:10000]
# Create features - whether contains words
def dialogue_act_features(post):
	features = {}
	for word in nltk.word_tokenize(post):
		features['contains(%s)' % word.lower()] = True
	return features

featuresets = [(dialogue_act_features(post.text), post.get('class'))
                for post in posts]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print nltk.classify.accuray(classifier, Test_set)
0.66











 [3] "Recognizing Textual Entailment"

"Recognizing Textual Entailment(RTE) -- is the task of determining whether a given piece of text T entails another text called the 'hypothesis'."
"Somebody said that China will be actually on SCO memeber from the next year." [Text] ---> "China is a memeber of SCO." [Hypothesis]

# Create features -- word type, overlap word counts
** "RTEFeatureExtractor class builds a bag of words for both the text and the hypothesis after throwing away some stop words, then calculates overlap and difference."
def rte_features(rtepair):
	extractor = nltk.RTEFeatureExtractor(rtepair)
	features = {}
	features['word_overlap'] = len(extractor.overlap('word'))
	features['word_hyp_extra'] = len(extractor.hyp_extra('word'))
	features['ne_overlap'] = len(extractor.overlap('ne'))
	features['ne_hyp_extra'] = len(extractor.hyp_extra('ne'))
	return features
	

rtepair = nltk.corpus.rte.pairs(['rte3_dev.xml'])[33] # get the data
extractor = nltk.RTEFeatureExtractor(rtepair)
print extractor.text_words
"set(['Russia','Organization','Shnghai', ......])"
print extractor.hyp_words
"set(['member','SCO','China'])"
print extractor.overlap('word')
"set([])"
print extractor.overlap('ne')
"set(['SCO','China'])"
print extractor.hyp_extra('word')
"set(['member'])"

**"???? -- RTEFeatureExtractor "




 -2 "Evaluation"

  **"Machine Leanring General Evaluation Methods"

 -3 "Classifiers"

 "<Decision Trees>" -- "P-242"

 "<Naive Bayes Classifier>" -- "P-245"

 "<Maximum Entropy Classifier>" -- "P-250"

 "<More...>"













































>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " [7]  Extracting Information from Text"

"Extracting General meaning from text. Ex. understand nameA relationship nameB. Extract structured data from unstractured data."

# Unstructured Data
"Today I went to a party and ....................... "


# Structured Data
"NameA     Relationship    NameB"
"Mike      likes           Ally "
" ............................  "

"How to capture structured data from unstructred data? "
" Approach 1 -- Information Extraction (Looking for specific information in text) this is in this chap "
" Approach 2 -- Sentence Meaning Analysis (building a very general representation of meaning) Find in later [10] chap "


-1 "Information Extraction"

" -- <Information Extraction Architecture> -- "

               Raw Text                        "XXXXXXXXX"
                   |
(1)                |
##############################################
        Senetence Segmentation                 "List of strings"
##############################################
                   |
(2)                |
##############################################
             Tokenization                      "List of lists of strings"
##############################################
                   |
(3)                |
##############################################
         Part Of Speech Tagging                "List of lists of tuples - tagged"
##############################################
                   |
(4)                |
##############################################
           Entity Recognition                  "List of trees - Chunked Sentences | ""We segment and label the entities that might participate in interesting relations with one another"
############################################## 
                   |
(5)                |
##############################################
           Relation Recognition                "List of tuples - Relations | We search for specific patterns between pairs of entities that occur near one another in the text, use those patterns to build tuples record relations"
##############################################


"To perform the first three " (1) (2) (3)

def ie_preprocess(document):
	sentences = nltk.sent_tokenize(document)                         (1) "Sentence Segmentation"
	sentences = [nltk.word_tokenize(sent) for sent in sentences]     (2) "Tokenization"
	sentences = [nltk.pos_tag(sent) for sent in sentences]           (3) "POS tagging"
    ..............                                                   (4) "Entity Recognition ------ Chunking"
    ..............                                                   (5) "Relation Extraction"




-2 "Chunking"

"Which segments and labels multitoken sequences as below:"

 we      saw      the   yelloe   dog
"PRP"   "VBD"     "DT"   "JJ"    "NN"
-----  -------   ---------------------
"NP"                     "NP"

"Same level DO NOT have overlap words"

# Methodologies to Chunk
"############## Regex Chunker ##############"
[1] "Noun Phrase Chunking | NP-chunking"
" Where we search for chunks corresponding to individual noun phrases."
" NP-chunks design to not contain nested NP chunks -- The market for xxxx1 for xxxxxx2 -- Only 'market' NP though xxxx1 and xxxx2 are nested NP"
**"Use POS tag is good source for building NP-chunks"
sentence = [("the","DT"), ("little", "JJ"),..............]

grammar = "NP: {<DT>?<JJ>*<NN>}" # define a chunk grammar
cp = nltk.RegexpParser(grammar) # Create a parser using the regex
result = cp.parse(sentence) # parse/chunk the sentence
print result

"""
(S
  (NP the/DT little/JJ .......)
  barked/VBD
  at/In
  (NP the/DT cat/NN))
"""
result.draw() # Will plot tree




[2] "Tag Pattern"
"The rules define the a chunk grammar use 'tag patterns' to describe sequences of tagged words."
"A tag pattern is a sequence of POS tags delimited using angle bracket, ex. <DT>?<JJ>*<NN>"

# Example text
"another/DT sharp/JJ dive/NN"
"trade/NN figures/NNS"
"any/DT new/JJ policy/NN measures/NNS"

# Tag patterns
"<DT>?<JJ.*>*<NN.*>+"

** "More complicated patterns -- P-266"





[3] "Chunking with Regular Expressions"

"With multiple rules"

grammar = r"""
   NP: {<DT|PP\$>?<JJ>*<NN>}    # Chunk determiner/possessive, adjectives and nouns
       {<NNP>+}                 # chunk sequences of proper nouns

"""
cp = nltk.RegexpParse(grammar)
sentence = [("Rapunzel","NNP"), ("let","VBD"), .....]

print cp.parse(sentence)

**"If multiple rules matches at an overlap area, the left most match takes precedence"






[4] "Exploring Text Corpora"

**"Use Chunkers"
cp = nltk.RegexpParser("CHUNK: {<V.*> <TO> <V.*>}")
brown = nltk.corpus.brown
for sent in brown.tagged_sents():
	tree = cp.parse(sent)
	for subtree in tree.subtrees():
		if subtree.node == "CHUNK": print subtree

"(CHUNK combined/VBN to/TO achieve/VB)"
"(CHUNK continue/VB to/TO place/VB)"
".................................."


 


[5] "Chinking"

"Sometimes it is easier to define what we want to exclude from a chunk. We can define a chink to be a sequence of tokens that is not included in a chunk."

 we      saw      the   yelloe   dog
"PRP"   "VBD"     "DT"   "JJ"    "NN"
-----            ---------------------
"NP"                     "NP"

" 'saw' is not included in chunks and so that it is chink"

"Chinking -- a process of removing a sequence of tokens from a chunk: (Instead of identify NP chunks, we directly remove those who don't. "
"            If match a entire chunk, whole chunk removed"
"            If match the moddle of a chunk, those tokens will be removed, results two chunks"
"            If match one of the two end of a chunk, those tokens ewmoved and result a smaller chunk"

grammar = r"""
   NP:
      {<.*>+}       # Chunk everything
      }<VBD|IN>+{   # Chink sequences of VBD and IN
"""
sentence = [("the","DT"), ("little", "JJ"),..............]
cp = nltk.RegexpParse(grammar)

print cp.parse(sentence)

"""
(S
  (NP the/DT little/JJ .......)
  barked/VBD 
  at/In
  (NP the/DT cat/NN))
"""






[6] "Representing Chunks: Tags vs Trees"

"CHUNK structure can be represented with either TREEs or TAGs"

**"One popular way -- IOB Tags -- I(inside a Chunk), O(outside a Chunk), B(Begin a Chunk)"

 we         saw      the       yelloe      dog
"PRP"      "VBD"     "DT"       "JJ"       "NN"
------     -----    --------   ------     -----
"B-NP"      "O"      "B-NP"    "I-NP"     "I-NP"
------              ---------------------------
"NP"                           "NP"

# in a file
We   PRP  B-NP
saw  VBD  O
the  DT  B-NP
little  JJ  I-NP
yellow  JJ I-NP
dog  NN I-NP


# in a Tree
  we         saw      the       yelloe      dog
 "PRP"      "VBD"     "DT"       "JJ"       "NN"
" |           \         \          |         /         "
" |            \         \         |        /          "
" NP            \         \--------NP------/           "
" |              \                 /                   "
" |               \               /                    "
" |--------------- S ------------/                     "






-3 "Developing and Evaluating Chunkers"
"Use standrad corpus tagged"
"Mechanics of converting IOB format into NLTK tree, then at how this is done on a larger scale using chunked corpus."
"How to score accuracy of a chunker"

[1] "Reading IOB Format and the CoNLL-2000 Chunking Corpus"
text = '''
he PRP B-NP
accepted VBD B-NP
.........
.........
'''
nltk.chunk.conllstr2tree(text, chunk_types=['NP']).draw() 
**"NP chunk (The market), VP chunks (has already delivered), PP Chunks(because of)" 
"...a nltk tree representation..."

# CoNLL-2000 chunking corpus (standrad)
from nltk.corpus import conll2000
print conll2000.chunked_sents('train.txt')[99]
"""
(S
  (NP the/DT little/JJ .......)
  barked/VBD 
  at/In
  (NP the/DT cat/NN))
.......................
"""




[2] "Simple Evaluation and Baselines"

1.# create a base line
from nltk.corpus import conll2000
cp = nltk.RegexpParse("") # create no chunks
test_sents = conll2000.chunked_sents('test.txt',chunk_types=['NP'])
print cp.evaluate(test_sents)

"CHUNKParse score:         "
"   IOB Accuracy:    43.4% " # all 'O' matches 43% which means 43% are 'O'
"   Precision:       0.0%  "
"   Recall:          0.0%  "
"   F-Measure:       0.0%  "


2.# Initial Chunk Parse
grammar = r"NP: {<[CDJNP].*>+}"
cp = nltk.RegexpParser(grammar)
print cp.evaluate(test_sents)

"CHUNKParse score:          "
"   IOB Accuracy:    87.4%  " # improvement!
"   Precision:       70.0%  "
"   Recall:          67.0%  "
"   F-Measure:       69.0%  "


"############## Ngram Chunker ##############"
3.# Further Improve on chunker
"We use training corpus to find the Chunk tag(IOB) that is most likely for each POS tag"
"Build a chunker using ---" unigram tagger "Instead of trying to determine the correct POS"
"tag for each word, we are trying to determine the correct chunk tag, given each word's POS tag"

class UnigramChunker(nltk.ChunkParserI):

	def __init__(self, train_sents):
		train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
		              for sent in train_sents]
		self.tagger = nltk.UnigramTagger(train_data)

	def parse(self, sentence):
		pos_tags = [pos for (word, pos) in sentence]
		tagged_pos_tags = self.tagger.tag(pos_tags)
		chunktags = [chunktag for (pos, chunking) in tagged_pos_tags]
		conlltags = [(word, pos, chunktag) for ((word,pos),chunktag) in zip(sentence, chunking)]
		return nltk.chunk.conlltags2tree(conlltags)

test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
unigram_chunker = UnigramChunker(train_sents)
print unigram_chunker.evaluate(test_sents)

"CHUNKParse score:          "
"   IOB Accuracy:    92.4%  " # Further improvement!
"   Precision:       79.0%  "
"   Recall:          67.0%  "
"   F-Measure:       69.0%  "




4.# Further further further Improve on chunker
"Replace unigram to bigram "
class BigramChunker(nltk.ChunkParserI):

	def __init__(self, train_sents):
		train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
		              for sent in train_sents]
		self.tagger = nltk.BigramTagger(train_data) # different from unigram

	def parse(self, sentence):
		pos_tags = [pos for (word, pos) in sentence]
		tagged_pos_tags = self.tagger.tag(pos_tags)
		chunktags = [chunktag for (pos, chunking) in tagged_pos_tags]
		conlltags = [(word, pos, chunktag) for ((word,pos),chunktag) in zip(sentence, chunking)]
		return nltk.chunk.conlltags2tree(conlltags)

test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
Bigram_chunker = BigramChunker(train_sents)
print unigram_chunker.evaluate(test_sents)

"CHUNKParse score:          "
"   IOB Accuracy:    93.4%  " # Further further further improvement!
"   Precision:       82.0%  "
"   Recall:          67.0%  "
"   F-Measure:       69.0%  "






[3] "Training Classifier - Based Chunkers"
"############## Classifier-Based Chunker ##############"
"To maximize the chunk performance, we not only looking at POS tag but also make use of the information in the context of words."

"Like the N-gram chunker, classifier based chunker will work by assigning IOB tags to the words in a sentence, and then convert tags to chunks"

class ConsecutiveNPChunkTagger(nltk.TaggerI):

	def __init__(self, train_sents):
		train_sents = []
		for tagged_sent in train_sents:
			untagged_sent = nltk.tag.untag(tagged_sent)
			history = []
			for i, (word, tag) in enumerate(tagged_sent):
				featureset = npchunk_features(untagged_sent, i, history)
				train_set.append((featureset, tag))
				history.append(tag)
		self.classifier = nltk.MaxentClassifier.train(train_set, algorithm='megam',trace=0) # Use Max entropy classifier


	def tag(self, sentence):
		history = []
		for i, word in enumerate(sentence):
			featureset = npchunk_features(sentence, i, history)
			tag = self.classifier.classify(featureset)
			history.append(tag)
		return zip(sentence, history)


# warpper function
class ConsecutiveNPChunker(nltk.ChunkParserI):
	def __init__(self, train_sent):
		tagged_sents = [[((w,t),c) for (w,t,c) in 
		                 nltk.chunk.tree2conlltags(sent)]
		                 for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
    	tagged_sents = self.tagger.tag(sentence)
    	conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
    	return nltk.chunk.conlltags2tree(conlltags)



1.# Classifier based chunker
# Extract features
def npchunk_features(sentence, i, history):
	word, pos = sentence[i]

	return {"pos":pos} # Feature 1

chunker = ConsecutiveNPChunker(train_sents)
print chunker.evaluate(test_sents)

"CHUNKParse score:          "
"   IOB Accuracy:    92.9%  " # similar to Unigram performance
"   Precision:       82.0%  "
"   Recall:          67.0%  "
"   F-Measure:       69.0%  "






2.# Classifier based with more features created
# Extract features
def npchunk_features(sentence, i, history):
	# feature 1,2
	word, pos = sentence[i]
	# feature 3,5
	if i == 0:
		prevword, prevpos = "<START>", "<START>"
	else:
		prevword, prevpos = sentence[i-1]
    # feature 4,6
    if i == len(sentence)-1:
    	nextword, nextpos = "<END>", "<END>"
    else:
        nextword, nextpos = sentence[i+1]
    # feature 7
    def tags_since_dt(sentence, i):
    	tags = set() 
    	for word, pos in sentence[:i]:
    		if pos == 'DT':
    			tags = set()
    		else:
    			tags.add(pos)
    	return '+'.join(sorted(tags))



	return {"pos":pos,                                 # Feature 1
	        "word":word,                               # Feature 2
	        "prevpos":prevpos,                         # Feature 3
	        "nextpos":nextpos,                         # Feature 4
	        "prevpo+pos":'%s+%s' % (prevpos, pos),     # Feature 5
	        "pos+nextpos":'%s+%s' % (pos, nextpos),    # Feature 6
	        "tags-since-dt":tags_since_dt(sentence,i)} # Feature 7

chunker = ConsecutiveNPChunker(train_sents)
print chunker.evaluate(test_sents)


"CHUNKParse score:          "
"   IOB Accuracy:    95.9%  " # Best performance!
"   Precision:       82.0%  "
"   Recall:          67.0%  "
"   F-Measure:       69.0%  "





-4 "Recursion in Linguistic Structure"

[1] "Building Nested Structure with Cascaded Chunkers"

"Previously, our chunk structure has been relatively flat. "

  we         saw      the       yelloe      dog
 "PRP"      "VBD"     "DT"       "JJ"       "NN"        # ------ Level 3 
" |           \         \          |         /         "
" |            \         \         |        /          "
" NP            \         \--------NP------/           "# ------ Level 2
"  \             \                 /                   "
"   \             \               /                    "
"    \------------ S ------------/                     "# ------ Level 1



"However, it is possible to build chunk structures of arbitrary depth"
"Simply create a multistage chunk grammar containing recursive rules."


  we         saw      the       yelloe      dog
 "PRP"      "VBD"     "DT"       "JJ"       "NN"        # ------ Level 4 
" |           |         \          |         /         "
" |           |          \         |        /          "
" |           V           \--------NP------/           "# ------ Level 3
" |            \                   |                   "
" NP            \------------------VP                  "# ------ Level 2
"  \                               /                   "
"   \                             /                    "
"    \------------ S ------------/                     "# ------ Level 1


# Example -- Build Regex Chunker handles NP, VP, PP, and S
grammar = r"""
   NP: {<DT|JJ|NN.*>+}             # Chunk sequences of DT, JJ, NN
   PP: {<IN><NP>}                  # Chunk prepositions followed by NP
   VP: {<VB.*><NP|PP|CLAUSE>+$}    # Chunk verbs and their arguments
   CLAUSE: {<NP><VP>}              # Chunk NP, VP
"""
cp = nltk.RegexpParser(grammar)
sentence = {("Mary","NN"),('saw','VBD'),('the','DT'),........}
print cp.parse(sentence)

"""
(S
   (NP Mary/NN)
   saw/VBD --------------------------- Fail to identify the VP here!!!
   (CLAUSE
   	  (NP the/DT cat/NN)
   	  (VP sit/VB (PP on/IN (NP the/DT mat/NN))))

"""

# Solution!!!

cp = nltk.RegexpParser(grammar, loop=2) # Initiate cascading process/recursive to check tree and subtree
                                        # '2' means the time of times the set of patterns should be run
print cp.parse(sentence)

"""
(S
   (NP Mary/NN)
   (VP
   	 saw/VBD --------------------------- Successfuly identify the VP here!!!
     (CLAUSE
   	   (NP the/DT cat/NN)
   	   (VP sit/VB (PP on/IN (NP the/DT mat/NN)))))

"""



[2] "Trees"

  we         saw      the       yelloe      dog
 "PRP"      "VBD"     "DT"       "JJ"       "NN"        # ------ Level 4 
" |           |         \          |         /         "
" |           |          \         |        /          "
" |           V           \--------NP------/           "# ------ Level 3
" |            \                   |                   "
" NP            \------------------VP                  "# ------ Level 2
"  \                               /                   "
"   \                             /                    "
"    \------------ S ------------/                     "# ------ Level 1


" 'S' is 'parent' of 'NP' and 'VP' "
" 'NP' and 'VP' is the 'child' of 'S' "
" 'NP' and 'VP' are 'siblings' "

# Create a Tree 
tree1 = nltk.Tree('NP',['Alice'])
print tree1
"(NP Alice)"
tree2 = nltk.Tree('NP',['the','rabbit'])
print tree2
"(NP the rabbit)"
tree3 = nltk.Tree('VP',['chased', tree2])
tree4 = nltk.Tree('S',[tree1, tree3])
print tree4
"(S (NP Alice) (VP chased (NP the rabbit)))"

print tree4[1]
"(VP chased (NP the rabbit))"
tree4[1].node
"VP"
tree4.leaves()
"['Alice','chased','the','rabbit']"
tree4[1][1][1]
"rabbit"

tree3.draw()
# Plot tree 3...




[3] "Trees Traversal"

*** "It is standard to use recursive function to traverse a tree."

def traverse(t):
	try:
		t.node
	except AttributeError: # To detect if t is a tree
		print t,

	else:
		# Now we know that t,node is defined
		print '(', t.node, 
			for child in t:
				traverse(child)
			print ')'

t = nltk.Tree('(S (NP Alice) (VP chased (NP the rabbit)))')
traverse(t)
"(S (NP Alice) (VP chased (NP the rabbit)))"






-5 "Named Entity Recognition"

"Named Entities are definite noun pharses that refer to specific types of individuals, such as organizations, persons, ates, and so on."

** "Common used Type of 'NE's "

"  NE Type            Examples       "
" -----------------------------      "
" ORGANIZATION        Geogiz corp.co "
" PERSON              Kacky, president Obama "
" LOCATION            Mount River    "
" DATE .............................. "
" TIME ........ P-281 ............... "
" MONEY ............................. "
" PERCENT ........................... "
" FACILITY .......................... "
" GPE                 South East Asia "


"The goal of 'Named Entity Recognition (NER)' is to identify all textual mentions of named entities which can be break down to two sub-tasks: "
"   (1) - Identfying the boundaries of NE "
"   (2) - Identifying the NE's type "

*** "Challenges -- (1) new NE updates daily | (2) Many names are ambiguous | (3) multi-word names "

"method 1" gazetteer lookup
"method 2 (more perfer)" classifier based

sent = nltk.corpus.treebank.tagged_sents()[22]
print nltk.ne_chunk(sent, binary=True) # Already trainned NE classifier in nltk, binary -- Ne or not, binary=Flase -- detail type of NE







-6 "Relation Extraction"

"P-284"
"Once name entity has been identified in a text. We then want to extract the relations that exist between them. "
" We can look for relationship between specific NE types -- (NE1 relation NE2) which 'relation' is a strings of word intervens between 'NE1' abd 'NE2' "
" Then we use REGEX to get them and check "

# Example 1 --- Only NE tags
IN = re.compile(r'.*\bin\b(?!\b.+ing)') # allow us to disregard strings such as 'success' in 'supervising the transition of' 
for doc in nltk.corpus.ieer.parsed_docs('NYT_19980315'):
	for rel in nltk.sem.extract_rels('ORG','LOC',doc, corpus='ieer',pattern=IN):
	    print nltk.sem.show_raw_rtuple(rel)

"[ORG: 'WHYY'] 'in' [LOC: 'Philadelpha']"
"[ORG: 'Fredom Form'] 'xxxx' [LOC: 'Arlington']"
"[ORG: 'xxxx'] 'xxxx' [LOC: 'xxxxxx']"
" ................................... "


# Example 2 ---- Use CoNLL 2002 -- Both POS tags and NE tags to identify pattern
from nltk.corpus import conll2002
vnv = """
(
is/V|         # 3rd sing present and
was/V|        # past forms of the verb zijn('be')
werd/V|       # and also present
wordt/V       # past of worden('become')
)
.*            # followed by anything
van/Prep      # followed by van ('of')
"""
VAN = re.compile(vnv, re.VERBOSE)
for doc in conll2002.chunked_sents('ned.train'):
	for r in nltk.sem.extract_rels('PER','ORG',doc, corpus='conll2002',pattern=VAN):
		print nltk.sem.show_clause(r, relsym="VAN")



































>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " [8]  Analyzing Sentence Structure"

" We need to deal with the ambiguity that NLP is famous for -- we need to cope with the fact "
" that there are an unlimited number of possible sentences while finite code to analyse them"

" (1) How can i use formal grammar to describe the structure of an unlimited set of sentence? "
" (2) How do we represent the sture of sentences using syntax tree? "
" (3) How do parsers analyze a sentence and automatically build a syntax trees? "



[1] "Some Grammatical Dilemmas"

*** "Linguistic Data and Unlimited Possibilities"

"Its hard to understand sentence with arbitrary length" "P-292"
" Generative Grammars --- in which a language is considered to be nothing more than an enormous collection of all grammatical sentences."
" Grammars use recursive production --- S -> S and S"


*** "Ubiquitous Ambiguity"
# Define a simple grammar
groucho_grammar = nltk.parse_cfg("""
S -> NP VP
PP -> P NP
NP -> Det N | Det N PP | 'I'
VP -> V NP | VP PP
Det -> 'an' | 'my'
N -> 'elephant' | 'pajamas'
V -> 'shot'
P -> 'in'
""")


sent = ['I','shot','an','elephant','in','my','pajamas']
parser = nltk.ChartParser(groucho_grammar)
trees = parser.nbest_parse(sent)
for tree in trees:
	print tree

# produce two trees
"""
(S
   (NP I)
   (VP
      (V shot)
      (NP (Det an) (N elephant) (PP (P in) (NP (Det my) (N pajamas))))))

"""

"""
(S
   (NP I)
   (VP
      (VP (V shot)(NP (Det an) (N elephant) (PP (P in) (NP (Det my) (N pajamas))))))
      (PP (V shot)(NP (Det an) (N elephant) (PP (P in) (NP (Det my) (N pajamas))))))
"""





[2] "What's the Use of Syntax"

# How to tell which wrong with 'word-salad'?
*" makes sense in short sequence but non-sense in long sequence "

a) "The worst part and clumsy looking for whoever heard light"

*"[Coordinate Structure] ---> if 'V1' and 'V2' are both phrases of grammactial category X, then 'V1 and V2' is also a grammatical category X"

Example - 
"(NP the worst part and the best part)"
"(AP slow and clumsy looking)"
* "CAN NOT ----X----- AP and NP"

*"[Constituent Structure] ---> is based on the observation that words combine with other words to form units -- (NP ....) (VP ....) (PP ....) (Norm ......) "

* "We can subsititude longer unit to shorter unit --- 'The bear saw' => 'He saw' "

   The    little    bear     saw      the     fine     fat    trout     in     the      brook
  "Det"   "ADJ"     "N"      "V"     "Det"   "Adj"    "Adj"    "N"     "P"    "Det"     "N"
"   |         \     /         |        |          \_____|______/        |        \______/    "
  "Det"        "Nom"         "V"     "Det"            "Nom"            "P"         "NP"
   The          bear         saw      the             trout             in          it       # --------- Replace unit 2
  "Det"        "Nom"         "V"     "Det"            "Nom"            "P"         "NP"
"    \___________/            |        \________________/                \__________/        "
         "NP"                "V"              "NP"                           "PP"
          He                 saw               it                            there           # --------- Replace unit 2
         "NP"                "V"              "NP"                           "PP"          
"         |                    \_______________/                              |              "
         "NP"                       "VP"                                     "PP"
          He                         ran                                     there           # --------- Replace unit 3 
         "NP"                       "VP"                                     "PP"
"         |                           \_______________________________________/              "
         "NP"                                           "VP"
          He                                             ran                                 # --------- Replace unit 4

"NP - Noun Phrase "
"VP - Verb Phrase "
"PP - Prepositional Phrase "

*"Reverse the subsititution process to create a <syntax tree> "


"               _____________S____________                               "
"              /                          \                              "                          
"         ____NP__                 _______VP_______                      "
"        /        \               /                \                     "
"      Det     __Nom__        ___VP___           ___PP___                "
"       |     /       \      /        \         /        \               "
"     <the>  Adj       N    V       __NP__     P       __NP__            "
"             |        |    |      /      \    |      /      \           "
"         <little>  <bear> <saw>  Det    Nom  <in>   Det     Nom         "
"                                  |    ___|___       |       |          "
"                                <the> /   |   \    <the>  <brook>       "
"                                    Adj  Adj   N                        "
"                                    /     |     \                       "
"                                  <fine> <fat> <trout>                  "
       the little    bear   saw   the fine fat   trout in the brook






[3] "Context-Free Grammar"

*"A Simple Grammar"
nltk.grammar
"Simple Context Free Grammar (CFG)"

grammar1 = nltk.parse_cfg("""
   S -> NP VP
   VP -> V NP | V NP PP
   PP -> P NP
   V -> "saw" | "ate" | "walked"
   NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
   Det -> "a" | "an" | "the" | "my"
   N -> "man" | "dog" | "cat" | "telescope" | "park"
   P -> "in" | "on" | "by" | "with"
""")

# Two production
"VP -> V NP | V NP PP"
"If a sentence has multiple trees productions -- it is 'structurally ambiguous' "

# Can not mix lexical item and grammarcal categories
"PP -> 'of' NP " NOT WORK!


sent = "Mary saw Bob".split()
rd_parser = nltk.RecursiveDescentParser(grammar1)
for tree in rd_parser.nbest_parse(sent):
	print tree
"(S (NP Mary) (VP (V saw) (NP Bob)))"


# Syntactic Categories
"  Symbol     Meaning                   Example        "
" ---------------------------------------------------- "
"  S          sentence                  the man walked "
"  NP         noun phrase               a dog          "
"  VP         verb phrase               saw a park     "
"  PP         prepositional phrase      with a telescope "
"  Det        determiner                the             "
"  N          noun                      dog            "
"  V          verb                      walked         "
"  P          preposition               in             "





*"Writing Your Own Grammar"

# Good to edit your grammar in a txt file
'mygrammar.cfg'

grammar1 = nltk.data.load('file:mygrammar.cfg')
sent = "Mary saw Bob".split()
rd_parser = nltk.RecursiveDescentParser(grammar1)
for tree in rd_parser.nbest_parse(sent):
	print tree



*"Recursion in Syntactic Structure"

"Appear on the left hand side as well as the right hand side"
# Example 1
" Nom -> Adj Nom " 
# Example 2
" S -> NP VP "
" VP -> V S "






[4] "Parsing with Context-Free Grammar"

" A Parser -- process input sentences according to the productions of a grammar, and builds one or more constituent structures that "
" confirm to the grammar. "

" A Grammar -- is a declarative specification of well-formedness - it is actually just string, not a grogram."

" A parser permit a grammar to be evaluated against a collection of test sentence, helping linguuists to discover mistakes in their grammatical analysis. "
" Example: natural language question submitted to a Q-A system to undergo parsing as initial step. "


" Three Parsing Algorithms: "
"                (1) Top-down method:                                      <Recursive Descent Parsing> "
"                (2) Bottom-up method:                                     <Shift-reduce Parsing>      "
"                (3) Top-down method with Bottom-up filtering method:      <Left-Corner Parsing>       "






*"Recursive Descent Parsing"

"Process: top level goal to find 'S', then 'S -> NP VP' enables parser to replace top level goal with subgoals like 'find NP or VP' "
"         Then recursively down (descent) to match specific words in the sequence. If no match, go back and try alternative. Till the end finds "
"         all matches, then back tracking and explore other choice of production in case (ambiguity) "

rd_parser = nltk.RecursiveDescentParser(grammar1)
sent = 'Mary saw a dog'.split()
for t in rd_parser.nbest_parse(sent):
	print t

" CONS:    "
"        (1) left recursive production - 'NP -> NP PP' cause infinite loop. "
"        (2) Waste time considering a lot of words and structures that do not correspond to input sentence. "
"        (3) back tracking discard parsed constituents which needs to be rebuilt again later. "








*"Shift-Reduce Parsing"

"Process: tries to find sequences of words and pahrases that correspond to the righthand side of a grammar production "
"         and then replace (reduce) them with the lefthand side, until the whole sentence is reduced to an 'S'. "
"         (No backtracking performed -- so not guarntee to find one, or if multiple one exist, can only find at most one tree) "

sr_parse = nltk.ShiftReduceParser(grammar1)
sent = 'Mary saw a dog'.split()
print sr_parser.parse(sent)

" CONS:    "
"        (1) Can reach a dead end when search the tree " - "Solution - (a) which reduction to do when there is multiple ones"
"                                                                      (b) whether to shift or reduce when either action is possible"
"                                                                       X - A general version: <Lookahead LR parser> "
" PROS over prev:    "
"        (1) Only build structure that corresponds to the words in the input. "
"        (2) Only build each substructure once -- NP(Det(the), N(man)) "







*"Left-corner Parser"

"Process: is a top-down parser with bottom-up filtter, before starting its works, a left-corner parser preprocesses the "
"         context-free grammar to build a table where each row contains two cells, the first holding a non-terminal, and the second holding "
"         the collection of possible left corners of that non-terminal. Each time a production is considered by the parser, it checks that "
"         the next input word is completible with at least one of the pre-terminal categories in the left-corner table. "

left-corner table
"  Category              left-corners(pre-teminals)  "
"  ------------------------------------------------  "
"  S                     NP                          "
"  NP                    Det, PropN                  "
"  VP                    V                           "
"  PP                    P                           "


" p-306 "

" PROS over prev:       "
"           (1) It not in infinite loop if meet left-recursive production. "








*"Well-Formed Substring Tables - Chart Parser"

+++" All the simple parsers discussed above sufferfrom limitations in both completeness and efficiency. "
" We will apply the algorithm design technique of 'dynamic programming' to parsing problem. "

" Dynamic Programming -- stores intermediate results and reuses them when appropriate, achieving significant efficience gains. "
"                        This technique can be applied to syntactic parsing, allowing us to store partial solutions to the parsing task and then "
"                        look them up as necessary in order to efficiently arrive at a complete solution. "

"                        Allows us to build 'PP in my pajam' just once. The first time we build it we save it in atble, then we look it up when "
"                        we need to use it as subconstituent of either the object NP or the higher VP. This table known as 'Well-formed Substring Table' - WFST"

" <Chart Parsing> "

# 'chart' data structure

"(0)" ---- "(1)" ---- "(2)" ---- "(3)" ---- "(4)" ---- "(5)" ---- "(6)" ---- "(7)"  
"       |          |          |          |          |          |          |      "     
       "I"       "shot"      "an"    "elephant"   "in"        "my"     "pajamas"


# In matrix
"   WFST     1     2     3     4     5     6    7  "
"   0        NP    .     .     S     .     .    S  "
"   1        .     V     .     VP     .     .   VP "
"   2        .     .     Det   NP     .     .   .  "
"   3        .     .     .     N     .     .    .  "
"   4        .     .     .     .     P     .    PP "
"   5        .     .     .     .     .     Det  NP "
"   6        .     .     .     .     .     .    N  "



# code Chart Parser from ground up!
" Hand code -- P-308 "

nltk.app.chartparser()


" CONS:    "
"        (1) Itself is not a parse tree - recognizing a sentence by grammar rather than parsing it. "
"        (2) It requires every non-lexical grammar production to be binary. "
"        (3) As a bottom-up method, it is poyentially wasteful. "









[5] "Dependencies and Dependency Grammar"


"                                                 ______PMOD_______       "
"              ______OBJ___________              |                 |      "
"   __SBJ___  |          _DETMOD   |    _NMOD__  |        DETMOD   |      "
"\|/        | |       \|/       | \|/  |      \|/|      \|/     | \|/     "
 "I"      "shot"     "an"     "elephant"      "in"     "my"    "pajamas"
  |          |         |            |           |        |        |
  |          |         |            |           |        |        |
'Dependent' 'Head'     |       "Dependent"      |        |        |
                       |                        |        |        |
                 "Dependencies"         "Dependencies"   |        |
                                                 "Dependencies"  "Dependencies"


"SBJ -- subject"
"NMOD -- noun modifier"
"OBJ -- object"
"PMOD -- proposition modifier"
"DETMOD -- determinent modifier"


# How to specify dependency in nltk
groucho_dep_grammar = nltk.parse_dependency_grammar("""
'shot' -> 'I' | 'elephant' | 'in'
'elephant' -> 'an' | 'in'
'in' -> 'pajamas'
'pajamas' -> 'my'
""")

pdp = nltk.ProjectuveDependencyParser(groucho_dep_grammar)
sent = 'I shot an elephant in my pajamas'.split()
trees = pdp.parse(sent)
for tree in trees:
	print tree

"(shot I (elephant an (in (pajamas my))))"
"(shot I (elephant an) (in (pajamas my)))"

# Tree structure
      "shot"
"    ____|_____            "
"   /          \           "
  "I"       "elephant"
"               |          "
               "in"
"          _____|_____     "
"         /           \    "
     "pajamas"       "my"



      "shot"
"    ____|__________________            "
"   /          \            \           "
  "I"       "elephant"     "in"
"                      ______|______    "
"                     /             \   "
                 "pajamas"         "my"


**"Critiria to define 'head' and 'dependent' in a construction"
" (1) H determines the distribution class of C; or alternatively, the external syntactic properties of C are due to H "
" (2) H determines the semantic type of C "
" (3) H is obligatory while D may be optional "
" (4) H selects D and determines whether it is obligatory or optional. "
" (5) The morphological form of D is determined by H "

"Although CFGs are not intended to directly capture dependencies, more recent linguistic frameworks have increasingly adopted formalisms which combine aspects of both approaches."








*"Valency and the Lexicon"

"verb ---> dependents"

" Production             Lexical head "
" ----------------------------------- "
" VP -> V Adj            was          "
" VP -> V NP             saw          "
" VP -> V S              thought      "
" VP -> V NP PP          put          "

"(Adj, NP, S and PP are often called 'complements' to verb)"
"(There are strong constraints on what verbs can occur with what complements --- different 'Valencies')"
" <Valency Restriction> is not only apply to verb but also other 'head' "

# Create subcategories to address the difference

"Symbol            Meaning                 Example"
"----------------------------------------------------------------"
"IV                Intranstitive verb      barked "
"TV                Transitive verb         saw a man"
"DatV              Dative verb             gave a dog to a man"
"SV                Sentential verb         said that a dog barked"

# Example
"VP -> TV NP "
"TV -> 'chased' | 'saw' "

"Valency is a property of lexical items"






*"Scaling Up"

"Very hard to create production for large sets!!!! "

"Though its hard, there are some large collaborative projects have achieved interesting and impressive result: "
"          (1) - Lexical Functional Grammar (LFG) "
"          (2) - Head-Driven Phrase Structure Grammar (HPSG) "
"          (3) - Lexicalized Tree Adjoining Grammar (XTAG) "
" P - 315 "






[6] "Grammar Development"
 
"What if we try to scale up this approach to deal with realistic corpus of language"
"Access treebank, look at the challenge of developing broad-coverage grammars"

*"Treebanks and Grammars"

"The 'corpus' module defines the treebank corpus reader, which contains a 10% sample of the Penn Treebank Corpus"

from nltk.corpus import treebank
t = treebank.parsed_sents('wsj_0001.mrg')[0]
print t

1."We can use this data to help develop a grammar."
def filter(tree):
	child_nodes = [child.node for child in tree
	               if isinstance(child, nltk.Tree)]
	return (tree.node == 'VP') and ('S' in child_nodes)

from nltk.corpus import treebank
[subtree for tree in treebank.parsed_sents()
         for subtree in tree.subtrees(filter)]
"[Tree('VP', [Tree('VBN', ['named']), Tree()])]"


2."The 'PP attachment corpus, nltk.corpus.ppattach is another source of information about the valency of particular verbs"
entries = nltk.corpus.ppattach.attachments('training')
table = nltk.defaultdict(lambda: nltk.defaultdict(set))
for entry in entries:
	key = entry.noun1 + '-' + entry.prep + '-' + entry.noun2
	table[key][entry.attachment].add(entry.verb)

for key in sorted(table):
	if len(table[key]) > 1:
		print key, 'N:', sorted(table[key]['N']), 'V:', sorted(table[key]['V'])

** "A collection of a large grammar can be obtained by - "
python -m nltk.downloader large_grammars

"P-316"








*"Pernicious Ambiguity"

1."[Structure Ambiguity]"
" Unfortunately, as the coverage of the grammar increases and the length of the input sentences grows, the "
" number of parse trees grows rapidly. "

grammar = nltk.parse_cfg("""
S -> NP V NP
NP -> NP Sbar
Sbar -> NP V
NP -> 'fish'
V -> 'fish'
""")

tokens = ["fish"] * 5
cp = nltk.ChartParser(grammar)
for tree in cp.nbest_parse(tokens):
	print tree
"tree1 .... "
"tree2 .... "
" ......... "

*" As the length of sentence grows, the number of trees grows more rapidly 1;2;5;14;42;132;429.... (Catalan numbers) "


2."[Lexical Ambiguity]"

"Runs - noun or Runs - verb"

*"Solution --> Weighted Grammar + Probabilistic Parsing"









*"Weighted Grammars   +   Probabilistic Parsing"

"Chart parsers improve the efficiency of computing multiple parses of the same sentences, but they are still overwhelmed by "
"the sheer number of possible parses. "

1."[Weighted Grammar]"
" notion of grammaticality --> gradient"
# Example:
" certain use of 'give' is more preferable. -- Using the Penn Treebank sample, we can examine all instances of prepositional dative and double object "
" construction involving 'give': "
def give(t):
	return t.node == 'VP' and len(t) > 2 and t[1].node == 'NP' \
	       and (t[2].node == 'PP-DTV' or t[2].node == 'NP') \
	       and ('give' in t[0].leaves() or 'give' in t[0].leaves())

def sent(t):
	return ' '.join(token for token in t.leaves() if token[0] not in '*-0')

def print_node(t, width):
	output = "%s %s: %s / %s: %s" % (sent(t[0]), t[1].node, sent(t[1]), t[2].node, sent(t[2]))
	if len(output) > width:
		output = output[:width] + "..."
	print output

for tree in nltk.corpus.treebank.parsed_sents():
	for t in tree.subtrees(give):
		print_node(t, 72)

"gave  NP: the chefs    /  NP: a standing ovation"
"give  NP: advertisers  /  NP: discounts for maintaining or increasing ad sp..."
"give  NP: it           /  PP-DTV: to the politicans"
# We can observe a strong tendency for the shortest complement to appear first. Such preferrance can be a weighted grammar



2."[Probablistic Parsing]"
"A probabilistic context-free grammar (PCFG) is a context-free grammar that associates a probability with each of its productions. "
"It generates the same set of parses for a text that the corresponding context-free grammar does, and assigns a probability to each parse."
"The probability is simply the product of probabilities of productions used to generate it."

grammar = nltk.parse_pcfg("""
S -> NP VP         [1.0]
VP -> TV NP        [0.4]
VP -> IV           [0.3]
VP -> DatV NP NP   [0.3]
TV -> ..................
........................
""")
# Same like 'VP -> ...' sum up to 1
# or
"VP -> TV NP [0.4] | IV [0.3] | DatV NP NP [0.3]"

viterbi_parser = nltk.ViterbiParser(grammar)
print viterbi_parse(['Jack','saw','telescopes'])
"(S  (NP Jack) (VP  (TV saw) (NP telescopes))) (p=0.064)" # Now, we can only select the most probable tree to aviod 'ambiguity' 
























































>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " [9]  Building Feature-Based Grammar"

"Natrural languages have an extensive range of grammatical constructions which are hard to handle with the simple methods described in previous chapter. "
" In order to gain more flexibility, we change our treatment or grammatical categories like 'S', 'NP', 'V'. In place of atomic labels, we decompose them into structures like "
" dictionaries, where features can takes on a range of values. "

*"How can we extend the framework of context-free grammars with features so as to gain more fine-grained control over grammatical categories and production?"

*"What kind of linguistic patterns and grammatical constructions can we now capture with feature-based grammars?"



[1] "Grammatical Features"

"In contrast to feature extractors, which record features that have been automatically detected, we are now going to declare the features of words and phrases. "

# Declare word feature structures
kim = {'CAT': 'NP', 'ORTH': 'Kim', 'REF': 'k'}                 # CAT: category, ORTH: words, REF: referent of kim
chase = {'CAT': 'V', 'ORTH': 'chased', 'REL': 'chase'}         # CAT: category, ORTH: words, REL: relation to base word
# add information for verb using placeholders
chase['AGT'] = 'sbj' 
chase['PAT'] = 'obj'
# Process sentence to bind the verb's agent role 'AGT' to subject and patient role 'PAT' to object
sent = "Kim chased Lee"
tokens = sent.split()
lee = {'CAT': 'NP', 'ORTH': 'Lee', 'REF': 'l'}                 # CAT: category, ORTH: words, REF: referent of lee
# identify the feature structure for each word
def lex2fs(word):
	for fs in [kim, lee, chase]:
		if fs['ORTH'] == word:
			return fs
subj, verb, obj = lex2fs(tokens[0]), lex2fs(tokens[1]), lex2fs(tokens[2])
# bind the verb's agent role 'AGT' to subject and patient role 'PAT' to object
verb['AGT'] = subj['REF'] # Agent of 'chase' is kim
verb['PAT'] = obj['REF']  # Patient of 'chase' is lee
verb['AGT']
"k"
verb['PAT']
'l'

*"Very ac-hoc process, will learn how the framework of context-free grammar and parsing be extended to feature structures"





1. "Syntactic Agreement"

--"Singular vs Plural"--
"The   dog:    noun(singular) -->   runs:   verb(singular)"
"The   dogs:   noun(plural)   -->   run:    verb(plural)"
*"the 'co-variate between noun and verb ---> 'syntactic Agreement' "

--"1st, 2nd, 3rd personal"--
# Further examples:
"                   Singular            Plural   "
" ---------------------------------------------- "
" 1st person        i run               we run   "
" 2nd person        you run             you run  "
" 3rd person        he/she/it runs      they run "

# To identify two sentences
"this dog runs" | "these dogs run"

# explicitly define all versions for each production
# SG: singular - PL: plural
grammar = ("""
S -> NP_SG VP_SG
S -> NP_PL VP_PL

NP_SG -> Det_SG N_SG
NP_PL -> Det_PL N_PL

VP_SG -> V_SG
VP_PL -> V_PL

Det_SG -> 'this'
Det_PL -> 'these'

N_SG -> 'dog'
N_PL -> 'dogs'

V_SG -> 'runs'
V_PL -> 'run'
""")

*"With small program is ok, but hard to generalize to large problems"
 "We then need the blow techniques..."







2. "Using Attributes and Constraint"

"linguistic categories having properties"
# Solution for previous problem
" Add grammatical feature 'NUM' - number, which indicates 'singular or plural' "
" define each 'CAT' property "

grammar = ("""
S -> NP[NUM=?n] VP[NUM=?n]
NP[NUM=?n] -> Det[NUM=?n] N[NUM=?n]
VP[NUM=?n] -> V[NUM=?n]

Det[NUM=sg] -> 'this'
Det[NUM=pl] -> 'these' 
N[NUM=sg] -> 'dog'
N[NUM=pl] -> 'dogs'
V[NUM=sg] -> 'runs'
V[NUM=pl] -> 'run'
""")

"['constriant']"
* "?n as a variable over value of 'NUM', can be either 'sg' or 'pl' within a production -- all 'CAT' (VP, NP, ..) in same production need to take the same value of '?n' "

# Example - 

"        NP[NUM=sg]            "
"     ________|________        "
"    /                 \       "
" Det[NUM=sg]        N[NUM=sg] "
"    |                  |      "
"   this               dog     "


"        NP[NUM=pl]            "
"     ________|________        "
"    /                 \       "
" Det[NUM=pl]        N[NUM=pl] "
"    |                  |      "
"   these               dogs   "

# If some CAT the same for all value of 'NUM'
"""
Det[NUM=sg] -> 'the' | 'some' | 'several'
Det[NUM=pl] -> 'the' | 'some' | 'several'
"""
# or
"""
Det[NUM=?n] -> 'the' | 'some' | 'several'
"""
# or
*"Not specify in the constriaghts"


** # Full Examples
# ------------- feat0.fcfg --------------- # *""" """ not included, just for formatting here, '% start S' - take 'S' as the strt symbal for the grammar 
"""
% start S
########################
# Grammar productions
########################
# S expansion production
S -> NP[NUM=?n] VP[NUM=?n]

# NP expansion productions
NP[NUM=?n] -> PropN[NUM=?n]
NP[NUM=?n] -> Det[NUM=?n] N[NUM=?n]
NP[NUM=pl] -> N[NUM=pl]

# VP expansion productions
VP[TENSE=?t, NUM=?n] -> IV[TENSE=?t, NUM=?n]
VP[TENSE=?t, NUM=?n] -> TV[TENSE=?t, NUM=?n] NP


########################
# Lexical Production
########################
Det[NUM=sg] -> 'this' | 'every'
Det[NUM=pl] -> 'these' | 'all'
Det[NUM=?n] -> 'the' | 'some' | 'several'

PropN[NUM=sg] -> 'Kim' | 'Jody'

N[NUM=sg] -> 'dog' | 'girl' | 'car' | 'child'
N[NUM=pl] -> 'dogs' | 'girls' | 'cars' | 'children'

IV[TENSE=pres, NUM=sg] -> 'disappears' | 'walks'
TV[TENSE=pres, NUM=sg] -> 'sees' | 'likes'

IV[TENSE=pres, NUM=pl] -> 'disappear' | 'walk'
TV[TENSE=pres, NUM=pl] -> 'see' | 'like'

IV[TENSE=past] -> 'disappeared' | 'walked'
TV[TENSE=past] -> 'saw' | 'liked'
"""
# ------------- feat0.fcfg --------------- #


tokens = "Kim likes children".split()
from nltk import load_parser
cp = load_parser('xxx/xxxx/feat0.fcfg', trace=2)
trees = cp.nbest_parse(tokens) # Use Chart Parser -- no, one or more trees
for tree in trees: print tree
"...P-335 "

 






3. "Terminology"

*"features like 'NUM' is 'atomic' since it can't be further decomposed "

*"Some other atomic feature value is 'Boolean' - like 'Auxiliary' --> can, may, will, do "

"""
V[TENSE=pres, +aux] -> 'can'
V[TENSE=pres, -aux] -> 'may'

V[TENSE=pres, -aux] -> 'walks'
V[TENSE=pres, +aux] -> 'likes'
"""

**"feature can take value that themselves are 'feature structure' "
  "[attribute value matrix] (AVM) -- group together other atomic features"
# Example -- group together 'agreement features'
"""
# visual
[POS = N          ]
[                 ]
[AGR = [PER = 3  ]]
[      [NUM = pl ]]
[      [GND = fem]]
"""

grammar = ("""
S -> NP[AGR=?n] VP[AGR=?n]
NP[AGR=?n] -> PropN[AGR=?n]
VP[TENSE=?t, AGR=?n] -> Cop[TENSE=?t, AGR=?n] Adj

Cop[TENSE=pres, AGR=[NUM=sg, PER=3]] -> 'is'
PropN[AGR=[NUM=sg, PER=3]] -> 'Kim'
Adj -> 'happy'
""")






[2] "Processing Features Structures"

*"We use 'FeatStruct()' to construct and manipulate feature structures. "

# Construct feature structure with atomic features
fs1 = nltk.FeatStruct(TENSE='past', NUM='sg')
print fs1
" [ NUM   = 'sg'  ] "
" [ TENSE = 'past'] "

# A feature structure is just a dictionary
print fs1['NUM']
"sg"
fs1['POS'] = 'N'

# Define complex feature structures
fs2 = nltk.FeatStruct(POS='N', AGR=fs1)
print fs2
" [         [ CASE = 'acc'] ] "
" [ AGR =   [ GND  = 'fem'] ] "
" [         [ NUM  = 'pl' ] ] "
" [         [ PER  = 3    ] ] "
" [                         ] "
" [ POS = 'N'               ] "
print fs2['AGR']
"   [ CASE = 'acc']  "
"   [ GND  = 'fem']  "
"   [ NUM  = 'pl' ]  "
"   [ PER  = 3    ]  "
print fs2['AGR']['PER']
"3"


# Other way to define fs using [][][]
*"We can also define fs using bracketed string "
print nltk.FeatStruct("[POS='N', AGR=[PER=3, NUM='pl', GND='fem']]")
" [ AGR =   [ GND  = 'fem'] ] "
" [         [ NUM  = 'pl' ] ] "
" [         [ PER  = 3    ] ] "
" [                         ] "
" [ POS = 'N'               ] "


# Not limited to linguistic category but also can include any features
print nltk.FeatStruct(name='Lee', telno="01 27 86 42 96", GND='fem')
" [  age      =  33                ] "
" [  aname    =  'Lee'             ] "
" [  telno    =  '01 27 86 42 96'  ] "
 

# Sharing Structures
** "Its more helpful to view feature structure as graphs -- direct acyclic graphs (DAGs) "

" P- 338 - 339 "

"<feature path> -- 'AGR' -> 'GND' -> 'fem' "

"<Structure sharing> | <reentrancy> -- 'Name' -> 'Address1' -> .....  "
"                                             -> 'Friend' -> 'Name' -> 'Address2' -> ....."
"                                             :: If 'Address1' == 'Address2' " # Sharing Structure

print nltk.FeatStruct("""
[NAME='Lee', 
 ADDRESS=(1)[NUMBER=74, 
             STREET='rue Pascal'], 
 SPOUSE=[NAME='Kim', 
         ADDRESS->(1)]
]
""")
# (1) - tag / coindex -- can be any number
# Address1 - sharing - address2









1. "Subsumption and Unification"

*"Partial information -- "

" a. [NUMBER = 74] "

" b. [NUMBER = 74] "
"    [STREET = 'rue Pascal'] "

" c. [NUMBER = 74] "
"    [STREET = 'rue Pascal']"
"    [CITY   = 'Pairs']"

" d. [TELNO  = 01 27 86 42 96]"




*"Subsumption -- a more general featire structure 'subsumes' a less general one."

"a. 'subsumes' b. "
"a. and d. are 'incommensurable' "




*"Unification -- merging information from two structures is called 'unification'. "
unify()

fs1 = nltk.FeatStruct(NUMBER=74, STREET='rue Pascal')
fs2 = nltk.FeatStruct(CITY='Paris')
print fs1.unify(fs2)
"  [  NUMBER   =   74            ]   "
"  [  STREET   =   'rue Pascal'  ]   "
"  [  CITY     =   'Pairs'       ]   "
print fs2.unify(fs1) # is symmetric
" Same.... "

print a.unify(c) # 'Subsumption two - more specified one shows '
"...c...."




*"Unification with 'Structure Sharing' "

"(1) --> Separated specified structures -- not affected by 'Unification' "
fs0 = nltk.FeatStruct("""
[NAME=Lee,
 ADDRESS=[NUMBER=74,
          STREET='rue Pascal'],
 SPOUSE=[NAME=Kim,
         ADDRESS=[NUMBER=74,
                  STREET='rue Pascal']]]
""")

fs1 = nltk.FeatStruct("[SPOUSE = [ADDRESS = [CITY = Pairs]]]")
print fs1.unify(fs0) # Only update 'SPOUSE''s features


"(2) --> Sharing structures -- affected by 'Unification' "
fs2 = nltk.FeatStruct("""
[NAME=Lee,
 ADDRESS=(1)[NUMBER=74,
             STREET='rue Pascal'],
 SPOUSE=[NAME=Kim,
         ADDRESS->(1)]]
""")

fs1 = nltk.FeatStruct("[SPOUSE = [ADDRESS = [CITY = Pairs]]]")
print fs1.unify(fs2) # update ALL features


"(3) --> Structure Sharing can also be stated using variables such as '?x' "
fs1 = nltk.FeatStruct("[ADDRESS1=[NUMBER=74, STREET='rue Pascal']]")
fs2 = nltk.FeatStruct('[ADDRESS1=?x, ADDRESS2=?x]')
print fs2.unify(fs1)
"""
[  ADDRESS1  = (1) [ NUMBER = 74          ]]
[                  [ STREET = 'rue Pascal']]
[                                          ]
[  ADDRESS2 -> (1)                         ]
"""






[3] "Extending a Feature-based Grammar"


1. "Subcategorization"

# Example -- Different kind of verb
" VP -> IV "
" VP -> TV NP "
# Need to encode this into the features, subcategories -- following methods:

*"<Phrase Structure Grammar (GPSG)>"
"allowing a lexical categories to bear a SUBCAT feature" | "can apear only on lexical categories -- V, not on NP, VP levels"
# Example
"""
VP[TENSE=?t, NUM=?n] -> V[SUBCAT=intrans, TENSE=?t, NUM=?n]
VP[TENSE=?t, NUM=?n] -> V[SUBCAT=tran, TENSE=?t, NUM=?n] NP
VP[TENSE=?t, NUM=?n] -> V[SUBCAT=clause, TENSE=?t, NUM=?n] SBar

V[SUBCAT=intrans, TENSE=pres, NUM=sg] -> xxx | xxx | xxxx
V[SUBCAT=tran, TENSE=pres, NUM=sg] -> xxx | xxx | xxxx
V[SUBCAT=clause, TENSE=pres, NUM=sg] -> xxx | xxx | xxxx

V.......................................................
V.......................................................
V.......................................................

........................................................
"""

"SBar ---- subordinate clause 'You claim [that you have no water]"

"SBar -> Comp S"
"Comp -> 'that"


*"<Head-driven Phrase Structure Grammar (PATR)>"
"The SUBCAT value directly encodes the valency of a head(the list of arguments that it can combine with' "

"'put' -- V[SUBCAT=<NP,NP,PP>]" # This says the verb can combine with three arguments. NP(subject), NP PP(NP followed by PP)

" ------------------------------- Tree Map -------------------------------- "

        'V[SUBCAT=<>]'
"      _______|_______________                                              "
"     /                       \                                             "
    'NP'                    'V[SUBCAT=<NP>]'
"    |                    _______|____________________                      "
   'Kim'"                /                \           \                     "                        
           'V[SUBCAT=<NP,NP,PP>]'        'NP'         'PP'
"                                          |            |                   "
                                       'the book'  'on the table'

" ------------------------------------------------------------------------- "




2. "Heads Revisited"

'Similary, Ns are heads of NPs.' # noun
'          As are heads of APs.' # adjective
'          Ps are heads of PPs.' # preposition

'We would like our grammar formalism to express the parent/head-child relation where it holds'
"At present, V and VP are just atomic symbols, and we need to find a way to relate them using feature (as we did earlier to relate IV and TV)"

*'[X-bar syntax]' - 'addresses this issue by abstracting out the notion of phrasal level' | 'N - represent the lexical level' | "N' - represent a higher level"

" -------------- X-bar Tree ----------------------------- "

                  " N'' "
"          _________|_____________                        "
"         /                       \                       "
        'Det'                    " N' "
"         |                 _______|_________             "
         'a'"              /                 \            "                       
                         'N'                " P'' "
"                         |                   |           "
                      'student'            'of French'

" ------------------------------------------------------- "




" -------------- Common Tree ----------------------------- "

                  " NP "
"          _________|_____________                        "
"         /                       \                       "
        'Det'                    " Nom "
"         |                 _______|_________             "
         'a'"              /                 \            "                       
                         'N'                " PP "
"                         |                   |           "
                      'student'            'of French'

" ------------------------------------------------------- "

# Production using feature structure
"""
S -> N[BAR=2] V[BAR=2]
N[BAR=2] -> Det N[BAR=1]
N[BAR=1] -> N[BAR=1] P[BAR=2]
N[BAR=1] -> N[BAR=0] P[BAR=2]
"""



3. "Auxiliary Verbs and Inversion"

"Inverted clauses -- where the order of subject and verb is switched -- occur in English interrogatives and also after 'negative' adverbs"

"(1) Do you like children? "  
"(2) Can Jody walk? "


"(1) Rarely do you see Kim. "
"(2) Never have I seen this dog. "

**" Verbs that can be positioned initially in inverted clauses belong to the class known as 'auxiliaries' "
  " :: do, can, can, have, be, will, shall "
  
  # Example - 
  'S[+INV] -> V[+AUX] NP VP'

" ---------------------------------------- Tree -----------------------------------------"

                         'S[+INV]'
"             _______________|_______________________                                    "
"            /                     |                 \                                   "
    'V[+AUX, SUBCAT=3]'           'NP'               'VP'
"            |                     |            _______|__________                       "
            'do'                 'you'"        /                  \                      "        
                                       'V[-AUX,SUBCAT=1]'         'NP'
"                                              |                    |                    "
                                             'like'             'children'


                               'do you like children?'
" -------------------------------------------------------------------------------------- "





4. "Undounded Dependency Constructions"

'You like Jody' - X - '*You like.' # like requires an NP complement, while put requires both a following NP and PP. The complements are obligatory: omitting them leads to ungrammatically

**"[Undounded Dependency Constructions]" -- "that is, afilter-gap dependendency where there is no upper bound on the distance between 'filter' and 'gap' "

"P - 348"

*"[Generalizaed Phrase Structure Grammar]" - "Involves 'slash categories' " -- "Y/XP"

















 





























>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " [10]  Analyzing the Meaning of Sentences"

 " In previous sections, we have 'machinary of parsers' and 'feature-based' grammer --> now can we do anything to analyze the meaning of sentences. "

" Represent a sentence meaning so that computer can process it " | " Associate the meaning representation with unlimited set of sentences " | " Use program connects meaning representation of sentence to store knowledga "




[1] " Natrual Language Understanding "

 1. "Querying a Database"
 
 # Example
 *"Which country is Athens in?" -> *"Greece"
 
 # Represent into a knowledage table using SQL
 # - table
 " City            Country            Population "
 " --------------------------------------------- "
 " althens         greece             1368       "
 " ............................................. "
 
 # - SQL
 "SELECT Country FROM city_table WHERE City = 'athens'"
 
 # - Use the feature-based grammer formalism
 # - sql0.fcfg
 grammar_sql = ("""
 % start S
 ########## Larger Structure ###########
 S[SEM=(?np + WHERE + ?vp)] -> NP[SEM=?np] VP[SEM=?vp]
 
 VP[SEM=(?v + ?pp)] -> IV[SEM=?v] PP[SEM=?pp]
 VP[SEM=(?v + ?ap)] -> IV[SEM=?v] AP[SEM=?ap]
 VP[SEM=(?det + ?n)] -> Det[SEM=?det] N[SEM=?n]
 
 PP[SEM=(?p + ?np)] -> P[SEM=?p] NP[SEM=?np]
 
 AP[SEM=?pp] -> A[SEM=?a] PP[SEM=?pp]

 ########### Automic ###############
 NP[SEM='Country="greece"'] -> 'Greece'
 NP[SEM='Country="china"'] -> 'China'

 Det[SEM='SELECT'] -> 'Which' | 'What'
 N[SEM='City FROM city_table'] -> 'cities'

 IV[SEM=''] -> 'are'
 A[SEM=''] -> 'located'
 P[SEM=''] -> 'in'
 """) # Save into file

 # - Parse a Query into SQL
 from nltk import load_parser
 cp = load_parser('xxx/xxxx/sql0.fcfg')
 query = 'What cities are located in China'
 trees = cp.nbest_parse(query.split())
 answer = trees[0].node['SEM']
 q = ' '.join(answer)
 print q
 "SELECT Country FROM city_table WHERE City = 'athens'"

 *"We defineed a task... (1) all small compounents doesn't have meaning; (2) hand code a lot details "
 *"Alternative below..."






 2. "Natural Language, Semantics, and Logic"

 *"Sentence of natrual language translated into logic instead of an excutable SQL. --> Advantage: Logical formalism is more abstract and therefore more generic."

 "Jack loves Amy" # Sentence

 # (1) Semantics --> declarative sentences are true / false in certain situation
 "A -- loves -- B (Relation) = true"

 # (2) Definite noun / Proper noun --> things in the world
 "A: Jack | B: Amy"


*"We could look at a set of sentences and define whether they can be true together in some situation"

# a. and b. are consistent
"a. A is to the north of B"
"b. B is a republic"


# a. and b. are inconsistent
"a. The captial of A has a population of 9,000"
"b. No city in A has population of 9,000"


*"Basically, logic-based approaches to natrual language semantics focus on those aspects of natrual language that guide our judgments of consistency and inconsistency"
 "Determine 'consistency' can be carried out by a computer --- symbolic manipulation --- In order to pursue this approach, we first want to develop a technique for "
 " representing a possible situation --- 'model' "

*"MODEL for a set of sentences --> is a formal representaion of a situation in which all the sentences in this set are 'true' "

"sent1 - 'k and s are boys'"
"sent2 - 'e is girl"
"sent3 - 's and e are running"

# - Model 
"D{s,k,e}"
################################################################
#               __________
#  'boy' <-----|     |    |
#              |   K | S  |
#              |_____|____|____
#                    |    |    |
#                    | e  |    |----> 'girl'                    |-----------------> D {s, k, e}
#                    |____|____|
#                      |
#                      |
#                  'is running'
#                   
################################################################

*"We will use models to help evaluate the truth or false of English sentence"






[2] "Propositional Logic"

"A logical language is designed to make reasoning formally explicit. As aresult, "
"it can capture aspects of natrual language which determine whether a set of sentences is consistent."


*"We need to develop logical representations of a sentence 'qp' that formally capture the 'truth-conditions' of 'qp'. "
" [A chased B] and [B ran away] " ---> "[1] = qp, [2] = yl" ----> " qp & yl " # logical form 
# Basic
" 'qp', 'yl'                                            --> Propositional Symbols "
" &(and), |(or), -(Not), ->(implies), <->(equivalence)  --> Boolean operators"      nltk.boolean_ops()
" (qp & yl); (qp | yl); (qp -> yl); (qp <-> yl)         --> Well-formed formulas" # infinite set of ... of propositional logic

*"More on 'Boolean operators..."
" Boolean operator                                     Truth condition                                 "
" ---------------------------------------------------------------------------------------------------- "
" negalection(it is not the case that ...)              -qp is true in S         iff    qp is false in S"
" conjunction(...and....)                               (qp & yl) is true in S   iff    qp is true in S so does yl"
" disjunction(...or....)                                (qp | yl) is true in S   iff    qp is true in S or yl is true in S"
" implication(if......then.....)                        (qp -> yl) is true in S  iff    qp is false in S or yl is true in S"
" equivalence(if and only if....)                       (qp <-> yl) is true in S iff    qo and yl are both true in S or both false in S"

lp = nltk.LogicParser()
lp.parse('-(P & Q)')
"<NegatedExpression -(P & Q)>"
lp.parse('P & Q')
"<AndExpression (P & Q)>"
lp.parse('P | (R -> Q)')
"<OrExpression (P | (R -> Q))>"
lp.parse('P <-> -- P')
"<IffExpression (P <-> --P)>"

# (Assumption --> conclusion = Argument)
"A is not to the north of B"                                 # -- Conclusion
"B is to the north of A"                                     # -- Assumption
"Process moving from one or more assumption to conclusion"   # -- Inference

# Valid Argument
" All its premises are true while its conclusion is NOT true "

# Example - Assumption -- THEREFORE -- conclusion
"A is to the north of B"
"THEREFORE B is NOT to the north of A"

*"Propositional logic can NOT look inside automic elements like: "
 " if 'x' is to the north of 'y' the 'y' is not to the north of 'x' "

*"So, we code 'A is to the north of B' -- AnB, 'B is to the north of A' -- BnA "
 "AnB -> -BnA"                      # Propositional Logic
 "[AnB, AnB -> -BnA] / -BnA"        # Propositional Logic (Complete Argument) -- [arg1, arg2, ...] / Conclusion ||| valid argument: all true 

 lp = nltk.LogicParser()
 AnB = lp.parse('AnB')
 NotBnA = lp.parse('-BnA')
 R = lp.parse('AnB -> -BnA')

 prover = nltk.Prover9()
 prover.prove(NotBnA, [AnB,R])
 "True"



*"Recall -- Model <--> we interpret sentences of a logical language relative to a model, which is a very simplified version of the world"
 "                <--> A model for propositional logic needs to assign the values 'True' and 'False' to every possible formula "

 val = nltk.Valuation([('P', True),('Q', True),('R', True)]) # Inducively define value for each formula component
 val['P'] # Like a dictionary
 True
 dom = set([])             # Ignore for now
 g = nltk.Assignment(dom)  # Ignore for now
 m = nltk.Model(dom, val)  # Create model object

 print m.evaluate('(P & Q', g)
 True
 print m.evaluate('-(P & Q', g)
 False


 **"Better we can divid into automic sentence -- qp into 'subject', 'object', 'predicates' "
   " We look into 'First-Order Logic' "











[3] "First-Order Logic"

1. "Syntax"

# Example - 
" Jacky walks "   --> "term: Jacky, | predicate(unary): walks"     --> "walks(Jacky)"       # First-Order logic
" Jacky sees Tom" --> "term: Jacky, Tom | predicate(binary): sees" --> "sees(Jacky, Tom)"   # First-Order logic

# Logical constants
"-(negalection), |(or), &(and) -- Boolean operates"

# Non-logical constants
"Jacky, sees, Tom"

# Inspect the 'types' of expressions
"Jacky, Tom"   --> "e, Entity"
"sees, walks"  --> "t, Type of formula"
*"We can form complex types -- Jacky walks <e,t> | Jacky sees Tom <e,<e,t>> (binary)"

tlp = nltk.LogicParser(type_check=True)
parsed = tlp.parse('walk(Jacky)')
parsed.argument
"<ConstantExpression Jacky>"
parsed.argument.type
"e"
parsed.function
"<ConstantExpression walk>"
parsed.function.type
"<e,?>" # Since its uncertain -- can be <e,t>, <e,e>, <e, <e,t>>

# can specify a signiture
sig = {'walk': '<e,t>'}
parsed = tlp.parse('walk(Jacky)', sig)
parsed.function.type
"<e,t>" # specified

*"How to denote the subject for pronoun like he, she, it?"
"(1) - interprenting a pronoun by pointing a relevant individual in the local context"
"(2) - supply an textual antecedent for pronoun by finding noun in previous sentence"
# Example - 
"a. Jacky is my dog"
"b. He is disappeared" --> "b2. Jacky is disappeared" # We say 'He' is coreferential with the noun 'Jacky'
                                                      # b and b2 are semantically the same

"c. Jacky is my dog and he is disappeared" # 'He' is bound by the indefinite NP 'my dog'
"c2 Jack is my dog and a dog disappeared"  # c and c2 are not semantically the same

# Open Formula
"a. He is a dog and he disappeared"
"b. dog(x) & disappear(x)" # Open formula

# Existential Quantifier
"ex.(dog(x) & disappear(x))" # At least one entity is dog and disappeared or more idiomatically a dog disappeared
exist x.(dog(x) & disappear(x))

# Universal Quantifier
"Ax.(dog(x) -> disappear(x))" # Everything has the property that if it is a dog, it disappeared or more idiomatically every dog disappeared
all x.(dog(x) -> disappear(x)) # If no dog also true => dog(x) can be false in 'dog(x) -> disappear(x)'

# A formula is 'Closed'
*"free -- variable x in a formula qp is 'free' in qp if that x doesn't fall within the scope of 'all x' and 'some x' in qp "
((exist x. dog(x)) -> bark(x)) # free bark(x)

*"closed formula -- no free for variable x"
all x.((exist x. dog(x)) -> bark(x))

*"LogicParser returns objects of class Expression. each expr object comes with a method 'free()' which returns the set of variables that are free in expr"
lp = nltk.LogicParser()
lp.parse('dog(cyril)').free()
"set([])"
lp.parse('dog(x)').free()
"set([Variable('x')])"
lp.parse('((some x. walk(x)) -> sing(x))').free()
"set([Variable('x')])"




2. "First-Order Theorem Proving"

# Revisit the example - 
"a. if x is to the north of y THEREFORE y is not to the north of x" # Which is hard to represent in Propositional Logic but...

"First-Order Logic:" 
all x. all y.(north_of(x, y) -> -north_of(y, x))

# Even better, we can use finite set to inference infinite set
"A |- g" # A is a possible empty list of assumptions, g is the proof goal

NotBnA = lp.parse('-north_of(B,A)') # goal
AnB = lp.parse('north_of(A,B)') # arg1
R = lp.parse('all x. all y. (north_of(x,y) -> -north_of(y,x))') # arg2
prover = nltk.Prover9()
prover.prove(NotBnA, [AnB, R])
True






3. "Summarizing the Language of First-Order Logic"

"Syntactic rules for propositional logic + formation rules for qunatifiers ===> [Syntax of First-Order Logic]"

# New logical constants
"   =      Equality  "
"   !=     Inequality"
"   exist  Existential quantifier"
"   all    Universal quantifier"

*"arity -- n: <e^n, t>"

# x,a,b.. variable | qp, yl ... formula
"(1) - If P is a predicate of type <e^n, t>, and the a1,....an are terms of type e, then P(a1,...an) is of type t"
"(2) - If alpha and beta are both of type e, then (alpha = beta) and (alpha != beta) are of type t"
"(3) - If qp is of type t, then so is -qp"
"(4) - If qp and yl are of type t, then so are (qp &..|..->..<-> yl)"
"(5) - If qp is of type t, and x is a variable of type e, then exists x.qp and all x.qp are of type t"







4. "Truth in Model"

*"Hard to symbolic specify 'true' or 'false' in situation like we did in 'Theorem Proving', so many different situations."
*"In other words, we need to give a truth-conditional semantics to First-Order Logic."


# Using Model
L: "A first-order logic"
M: "A model for L"
<D, Val>: "A pair for M"

D: "An non-empty set called the domain of model"
Val: "A function called the valuation function, which assigns values from D to expressions of L"
     "FUNCTION: "
     "          (1) For every individual constant c in L, Val(c) is an element of D"
     "          (2) For every predicate p in L of 'arity' n >= 0, Val(P) is a function from D^n to {True, False} (If n = 0, Val(P) is simply a true value) " 

# Val(P) is a set of S of pairs
"S = {s|f(s) = True}" # f() is called "characteristic function"

*"Relations are represented semantically in NLTK in the standard set-theoretic way: as sets of tuples."

# we have a domain of discourse consisting of individuals - B.., O.., C.. where B is boy, O is a girl, C is a dog

dom = set(['b','o','c'])
v = """
bertie => b
olive => o
cyril => c

boy => {b}
girl => {o}
dog => {c}

walk => {o,c}
see => {(b,o),(c,b),(o,c)}
"""

val = nltk.parse_valuation(v)

*"Model Structure: " <D, Val> -- <dom, val>

# Test
('o','c') in val['see']
True
('b',) in val['boy']
True









5. "Individual Variables and Assignments"

"Assignment: This is a mapping from individual variables to entities in the domain."

g = nltk.Assignment(dom,[('x','o'),('y','c')]) # Assign variables to constant
g
"{'y':'c', 'x':'o'}"

print g
"g[c/y][o/x]"

g['y']
'c'

# Model checking 
m = nltk.Model(dom, val)
m.evaluate('see(olive, y)', g)
True
m.evaluate('see(y,x)', g)
False
m.evaluate('see(bertie, olive) & boy(bertie) & -walk(bertie)', g)
True

# Clear all binding
g.purge()
g
""

m.evaluate('see(olive, y)', g)
'Undefined'









6. "Quantification"

" The notion of variable satisfaction can be used to provide an interpretation for quantified formulas "
" Evaluate -- exists x.(girl(x) & walk(x)) "

m.evaluate('exists x.(girl(x) & walk(x))', g)
True

# add assignment binds x to u
m.evaluate('girl(x) & walk(x)', g.add('x','o'))
True

# Returns the set of all individuals that satisfy an open formula
fmla1 = lp.parse('girl(x) | boy(x)')
m.satisfiers(fmla1, 'x', g)
"set(['b','o'])"

fmla2 = lp.parse('girl(x) -> walk(x)')
m.satisfiers(fmla2, 'x', g)
"set(['c','b','o'])"

fmla3 = lp.parse('walk(x) -> girl(x)')
m.satisfiers(fmla3, 'x', g)
"set(['b','o'])"








7. "Quantifier Scope Ambiguity"

# Example -- How to give a formal representation of a sentence with two quantifiers?
"Everybody admires someone."
# At least two ways to express in First-Order Logic
"(1) all x.(person(x) -> exists y.(person(y) & admire(x,y)))"
"(2) exists y.(person(y) & all x.(person(x) -> admire(x,y)))"

# (2) is locally stronger than (1)
*"We can claim that this sentence is 'ambiguous' with respect to quantifier scope "

# Examine the ambiguity
v2 = """
bruce => b
cyril => c
elspeth => e
julia => j
matthew => m
person => {b, e, j, m}
admire => {(j,b),(b,b),(m,e),(e,m),(c,a)}
"""
val2 = nltk.parse_valuation(v2)

dom2 = val2.domain

m2 = nltk.Model(dom2, val2)

g2 = nltk.Assignment(dom2)

fmla4 = lp.parse('(person(x) -> exists y.(person(y) & admire(x,y)))') # (1)
m2.satisfiers(fmla4, 'x', g2)
"set(['a','c','b','e','j','m'])"

fmla5 = lp.parse('(person(y) & all x.(person(x) -> admire(x,y)))') 
m2.satisfiers(fmla5, 'y', g2)
"set([])"

fmla6 = lp.parse('(person(y) & all x.((x = bruce | x = julia) -> admire(x,y)))') 
m2.satisfiers(fmla5, 'y', g2)
"set(['b'])"








8. "Model building"

" (1) [Existence proof of the model to define whether the set id consistent] "

" Previously, we have assume that we already have the model and we want to check the truth of a sentence in the model "
" By contrast, in model building -- tries to create a new model, given some set of sentences. If it succeeds, "
" then we know that the set is consistent, since we have an existence proof of the model "

# We invoke the Macc4 model builder by creating an instance of Mace() and calling its build_model() method
# One option is to treat our candidate set of sentences as assumptions, while leaving the goal unspecified
a3 = lp.parse('exists x.(man(x) & walks(x))')
c1 = lp.parse('mortal(socrates)')
c2 = lp.parse('-mortal(socrates)')

mb = nltk.Mace(5) # Create a Mace5 instance

print mb.build_model(None, [a3,c1]) # leave the goal unspecified
True 
print mb.build_model(None, [a3,c2])
True
*"Both [a3,c1] and [a3,c2] are consistent lists, since Mace5 succeeds in building  a model for each, whereas [c1,c2] is inconsistent"
print mb.build_model(None, [c1,c2])
False



" (2) [Use the model builder as an adjunct to the theorem prover] "

" We are trying to prove A |- g, g is locally derived from assumoptions A = [a1,a2,...,an]"

# Given the input Mace5 will try to find a model for the assumptions A together with negalection of g, which is A' = [a1,a2,...,an,-g]
# result 1 - if g fails to follow from S then Mace4 may well return with a counterexample faster than Prover9 conculdes that it cannot find proof
# result 3 - if g is provable from S then Mace4 may takes longtime and eventually failed to find a countermodel

# Example
"assumption = [There is a woman that every man loves, Adam is a man, Eva is a woman]"
"conclusion = [Adam loves Eve]"

a4 = lp.parse('exists y. (woman(y) & all x. (man(x) -> love(x,y)))')
a5 = lp.parse('man(adam)')
a6 = lp.parse('woman(eve)')

g = lp.parse('love(adam,eve)')

mc = nltk.MaceCommand(g, assumptions=[a4,a5,a6]) # We use MaceCommand to inspect the model that has been built
mc.build_model()
True # Mace4 successfully found a countermodel ()

print mc.valuation
"""
{'C1': 'b',
 'adam': 'a',
 'eve': 'a',
 'love':set([('a','b')]),
 'man':set([('a',)]),
 'woman':set([('a',),('b',)])
}
"""

*"'C1' is a 'Skolem constant' that model builder introduces as representative of existential quantifier"
 "Due to 'exists y' part, model builder know there is 'b' statisfy a4, however, it doesn't find 'b'.."
 "because we didn't specify 'man' and 'woman' being a disjoint set, model treat them the same...'a' so there is no 'b'"
 "Below revise..."

a7 = lp.parse('all x. (man(x) -> -woman(x))')

g = lp.parse('love(adam,eve)')

mc = nltk.MaceCommand(g, assumptions=[a4,a5,a6,a7]) # We use MaceCommand to inspect the model that has been built
mc.build_model()
True # Still true...

print mc.valuation
"""
{'C1': 'c',
 'adam':'a',
 'eve':'b',
 'love':set([('a','c')]),
 'man':set([('a',)]),
 'woman':set([('b',),('c',)])
}
"""

*"Even though still found counter model, it is more intuitively correct."
 "There is nothing in our premises says 'eve' is the only woman in the domain of discourse"
 "So, the counter model is actually acceptable, unless we role out like: "
 a8 = lp.parse('y. all x. (woman(x) -> (x = y))') # 'eve' is the only woman






 

[4] "The Semantics of English Sentences"

1. "Compositional Semantics in Feature-Based Grammaer"

" Rather than construct SQL query, we will build a logical form -- One of the guilding ideas for designing such grammars is the "
" 'Principle of Compositionality' --> (also known as 'Frege's Principle') "

* "Principle of Compositionality" --> "The meaning of the whole is a function of the meanings of the parts and of the way they are syntactically combined."

# Example

"        S[SEM=<bark(cyril)>]                     "
"     ________|_________________                  "
"    /                          \                 "
" NP[SEM=<(cyril)>]          VP[SEM=<bark>]       "
"    |                            |               "
"  'Cyril'                    IV[SEM=<\X.bark>]   "
"                            _____|______         "
"                           /            \        "
"                          X           'barks'    "


grammar = ("""

S[SEM=<?vp(?np)>] -> NP[SEM=?np] VP[SEM=?vp]
VP[SEM=?v] -> IV[SEM=?v]                          # - VP rule says the parent's semantics is the same as the head child's semantics. 

NP[SEM=<cyril>] -> 'Cyril'                        # - lexical rule provide non-logical constants to serve as the semantics values of 'Cyril'
IV[SEM=<\x.bark(x)>] -> 'barks'                   # - lexical rule provide non-logical constants to serve as the semantics values of 'barks'

""")



2. "The lambda-calculation"

*"[Lambda Calculation]" -> " This provides us with an invaluable tool for combining expressions of first-order logic as we assemble a meaning representation for English sentence."

"The set of all W such that W is an element of V (the vocabulary) and W has property P"

"{W | W (= V & P(W))}   <----------- lambda conuterpart ---------->  lambda W. (V(W) & P(W))"

*"lambda expressions were original designed by xxxx to represent computable functions and to provide a foundation for mathmatics and logic."
 "The theory in which lambda expressions are studied is known as the 'lambda-calculas'. It is NOT part of First-Order logic. Both can be used independently. "

*"[lambda]" -- "is a binding operator, just as the first-order logic quantifiers are. "

# Example
"(walk(x) & chew_gum(x))"           # open formula
"lambda x.(walk(x) & chew_gum(x))"  # bind variable x with lambda operator
"\x. (walk(x) & chew_gum(x))"       # implementation in nltk

lp = nltk.LogicParser()
e = lp.parse(r'\x.(walk(x) & chew_gum(x))')
e
"<LambdaExpression \x.(walk(x) & chew_gum(x))>"
e.free()
"set([])"


**"[Lambda-abstraction]" -- "The result of binding the variables in an expression -- '\x. (walk(x) & chew_gum(x))' "
                         -- "Meaning --> 'being an x such that x walks and x chews gum' OR 'having the property of walking and chewing gum' "
                         -- "Good representations for verb phrases (subjectiveless clauses), particularly when these occur as arguments in their own right"
                            ^ # Example -- with prop noun
                            " 'To walk and chew gum is hard' => 'hard(\x. (walk(x) & chew_gum(x)))' "
                            ^ # Example -- with individuals
                            " 'Gerald walks and chews gum' => '\x.(walk(x) & chew_gum(x)) (gerald)'"

**"[Beta-reduction]" -- "remove the \x from the expression and replaced all occurances of x in the expression by 'individual' "
                        "'\x.(walk(x) & chew-gum(x)) (gerald)' --- Beat-reduction ---> '(walk(gerald) & chew_gum(gerald))'"
                        "'\x.(walk(x) & chew-gum(x)) [gerald/x] --- Beat expression' : replace x with gerald"
                        # Example - 
                        e = lp.parse(r'\x.(walk(x) & chew_gum(x)) (gerald)')
                        print e
                        "\x.(walk(x) & chew_gum(x)) (gerald)"
                        print e.simplify()                    # Beta reduction
                        "(walk(gerald) & chew_gum(gerald))"
                           

						*"[cases that body of lambda-abstract is NOT an open formula]"
						# Open formula
						'\x.(walk(x) & chew_gum(x))' # unary predicate
						# NOT open formula
						'\x. \y.(dog(x) & own(y,x))' # binary predicate
						print lp.parse(r'\x. \y.(dog(x) & own(y,x)) (cyril)').simplify()
						"\y.(dog(cyril) & own(y,cyril))"
						print lp.parse(r'\x. \y.(dog(x) & own(y,x)) (cyril, angus)').simplify()
						"(dog(cyril) & own(angus,cyril))"
						# Treat one lambda abstract as the argument of another abstract?
						"\y.y(angus) (\x.walk(x))" # '\x.walk(x)' as argument for y
						"XXX -- But since variable y is stipulated to be type 'e', so in expression only type 'e' works but '\x.(walk(x))' is type <e,t>"
						"Soluttion -- we need to abstract over variable on a higher type, so let's use P and Q as variables of type <e,t>, and then: "
						"\P.P(angus) (\x.walk(x))" # it is legal
						"\x.walk(x) (angus)" # 1st simplify (Beta reduction!)
						"walk(angus)" # 2nd simplify (Beta reduction!)

						*"[Some care need to be take for variable]"
						# Example - 
						a. "\y.see(y,x)" -> a. "\P.exists x.P(x) (\y.see(y,x))" -> a. "exists x.see(x,x)" #?? the free variable x in argument is also in existential formula 
						b. "\y.see(y,z)" -> b. "\P.exists x.P(x) (\y.see(y,z))" -> b. "exists x.see(x,z)" #?? Its okay
						"P-389 ??"

						*"[alpha-equivalents]" -- " 'exists x.P(x)' == 'exists y.P(y)' "
						*"[alpha-conversion]" -- " 'exists x.P(x)' => 'exists y.P(y)' " # change from x to y




3. "Quantified NPs (Object NPs)"

"[Challenge] --- 'A dog barks' -> How to build logic form 'exists x.(dog(x) & bark(x))' into feature based grammar? In other words, encode quantifier 'exists all' into the NP term. "

" Building complex semantic representations is 'function application' - make SEM value act as the function expression rather than the argument" --> *"[type raising]"

" We are looking for a way to instantiating '?np' so that '[SEM=<?np(\x.bark(x))>]' is equivalent to '[SEM=<exisits x.(dog(x) & bark(x))>]'"

" Solution: We use lambda M to replace '?np' so that applying M to '\x.bark(x)' we get 'exists x.(dog(x) & bark(x))'"

# Example - 
"(1) Use predicate variable 'P' <e,t> replace '\x.bark(x)' in 'exists x.(dog(x) & bark(x))' " 
'exists x.(dog(x) & bark(x))' --> '\P.exists x.(dog(x) & P(x))' # Quantified NP

"(2) Use predicate variable 'Q' <e,t> replace '\x.dog(x)' in '\P.exists x.(dog(x) & P(x))' " 
'\P.exists x.(dog(x) & P(x))' --> '\Q P.exists x.(Q(x) & P(x))' 

"(3) Apply '\Q P.exists x.(Q(x) & P(x))' to '\x.dog(x)' then '\x.bark(x)' we got: "
'\P.exists x.(dog(x) & P(x)) (\x.bark(x))'

"(4) Carry out 'lambda-reduction' to '\P.exists x.(dog(x) & P(x)) (\x.bark(x))' we got: "
'exists x.(dog(x) & bark(x))'



4. "Transitive Verbs"

"[Challenge] --- Deal sentence contains 'transitive verbs' such as 'Angus chases a dog' which logic form = 'exists x.(dog(x) & chase(angus,x))'"

"(Constriant - 1): Required that semantic representation of 'a dog' be independent of whether the NP acts as subject or object of the sentence. " -- '\P.exists x.(dog(x) & P(x))' 

"(Constriant - 2): VPs should have a uniform type of interpretation, regardless of whether they consist of just an intransitive or transitive verb plus object. " -- '\z.chase(y,z)' # always given type <e,t>

"(1) If we inverse 'lambda-reduction' on '\y.exists x.(dog(x) & chase(y,x))' we got: "
'\y.exists x.(dog(x) & chase(y,x))' -> '\P.exists x.(dog(x) & P(x)) (\z.chase(y,z))' # NP type: <<e,t>,t>

"(2) Replace function expression above by a variable X of the same type as an NP. type <<e,t>,t>"
'\P.exists x.(dog(x) & P(x)) (\z.chase(y,z))' -> 'X(\z.chase(y,z))' 

"(3) The representation of a transitive verb will have to apply to an argument of the type of X to yield a function expression of the type of VPs - <e,t>"
'X(\z.chase(y,z))' --> '\X y.X(\x.chase(y,x))' # substracting over both X and y variables

"(4) Given the senmantic representation of 'chases', apply it to '\P.exists x.(dog(x) & P(x))' we got: "
'(\X y.X(\y.chase(x,y))) (\P.exists x.(dog(x) & P(x)))' : '(semantic rep for 'chases') (quantified NP)' --> '\y.exists x.(dog(x) & chase(y,x))'

lp = nltk.LogicParser()

tvp = lp.parse(r'\X y.X(\y.chase(x,y))') # semantic rep for 'chases'
np = lp.parse(r'(\P.exists x.(dog(x) & P(x)))') # Quantified NP
vp = nltk.sem.ApplicationExpression(tvp, np)
print vp
"(\X y.X(\y.chase(x,y))) (\P.exists x.(dog(x) & P(x)))"
print vp.simplify() # lambda reduction
"\x.exists z2.(dog(z2) & chase(x,z2))" # '\y.exists x.(dog(x) & chase(y,x))'



*"[subject NPs]" -- "Propor Noun, so far we treat them as semantically as individual constants, and these can't applied as functions to expression like '\y.exists x.(dog(x) & chase(y,x))' "
                    "We need to come up with different semantic representation for them." -- "\P.P(angus)" - "denotes the characteristic function corresponding to the set of all properties which"
                                                                                                             "are true of 'Angus' "
 "(1) Converting from individual 'angus' to '\P.P(angus)' is another example of [type rasing]"
 '\P.P(angus)(\x.walk(x))' -- 'lambda-reduction' -- 'walk(angus)'
 # Small set of rules in 'simple-sem.fcfg'
 from nltk import load_parser
 parser = load_parser('grammars/book_grammers/simple-sem.fcfg', trace=0)
 sentence = 'Angus gives a bone to every dog'
 tokens = sentence.split()
 trees = parser.nbest_parse(tokens)

 for tree in trees:
 	print tree.node['SEM']
 "all z2.(dog(z2) -> exists z1.(bone(z1) & give(angus,z1,z2)))"


 *"NLTK native interpreter" -- "some utilities to make it easier to derive and inspect semantic interpretations."
                               "P-392" batch_interpret() # batch interpret a list of input sentences. It builds dict 'd'
                                                         # each input sent - d[sent] is a list of pairs(trees, semantic rep)
 
 **** "We have seen how to convert English sentence to logical forms, and then how logic forms could be checked as true or false in a model."
      "So, put together, we can check the truth value of English sentence in a given model. "

      # Example - 
      v = """
      bertie => b
      olive => o
      cyril => c
      boy => {b}
      girl => {o}
      dog => {c}
      walk => {o,c}
      see => {(b,o),(c,b),(o,c)}
      """
      val = nltk.parse_valuation(v)
      g = nltk.Assignment(val.domain)
      m = nltk.Model(val.domain, val)

      sent = 'Cyril sees every boy'

      grammar_file = 'grammars/book_grammers/simple-sem.fcfg'
      result = nltk.batch_evaluate([sent], grammar_file, m, g)[0]

      for (syntree, semrep, value) in results:
      	print semrep
      	print value

      "all z4.(boy(z4) -> see(cyril,z4))"
      True
 



5. "Quantifier Ambiguity Revisited"

*"[Limitation]" -- "Methods above do not deal with 'Scope ambiguity' - our translation method is syntax-driven, semantic representation is closely coupled with the syntax analysis"
                   "the scope of quantifiers in the semantics therefore reflects the relative scope of the corresponding NPs in the syntactic parse tree."
                   " therefore - some sentence always has two interpretations:"

                               "Every girls chases a dog"
                               "(1) ---- a.all x.(girl(x) -> exists y.(dog(y) & chase(x,y)))"
                               "(2) ---- exists y.(dog9y) & all x.(girl(x) -> chase(x,y)))  "
                    
                    "Many solutions - into the one simplest - " -- "[Cooper Storage]"
                    #########################################################################
                    #     (1)                                          (2)
                    #
                    #     all x.(girl(x) -> qp)                 exists y.(dog(y) & qp)                 # Two are idenitical except we swapped round two quantifiers
                    #                        |                                      |
                    #                        |                                      |
                    #              exists y.(dog(y) & yl)              all x.(girl(x) -> yl)
                    #                                  |                                  |
                    #                                  |                                  |
                    #                           chase(x,y)                         chase(x,y)
                    #
                    #
                    #########################################################################
                    "[Cooper Storage] -- a semantic representation is no longer an expression of 'first-order' logic, but instead a pair consisting "
                    "                    of a 'core' semantic representation "                       - 'chase(x,y)'
                    "                       a list of 'binding operators' (like quantified NPs)"     - '\P.exists y.(dog(y) & P(y))' | '\P.all x.(girl(x) -> P(x))'

                    *"S-Retrieval" - "Combining binding operators with the core: "
                    "(1) - we pick a binding operator off the list, and combine it with core: "
                    -> '\P.exists y.(dog(y) & P(y)) (\z2.chase(z1,z2))'

                    "(2) - then we take the result, and apply the next binding operator from the list to it: "
                    -> '\P.all x.(girl(x) -> P(x)) (\z1.exists x.(dog(x) & chase(z1,x)))' # once the list is empty we have a 'conventional logical form' for the sentence.
                    # If we are carefully to allow every possible order of binding operators (all permulation of the binding opertator list), then we will be able to generate every possible scope 'ordering of quantifiers'

                    **"How to deal 'core + binding oper' representation compositionally?" 
                    'CORE' - 'the core'
                    'STORE' - 'binding operator list'
                    # Example 1 - 
                    "Cyril smiles"
                    "CORE = <\x.smile(x)>"
                    "STORE = (<bo(\P.P(cyril),@x)>)"



                    grammar = """
                    S[SEM=[CORE=<?vp(?np)>, STORE=(?b1+?b2)]] -> NP[SEM=[CORE=?np, STORE=?b1]]   VP[SEM=[CORE=?vp, STORE=?b2]]

                    VP[SEM=?s] -> IV[SEM=?s]

                    IV[SEM=[CORE=<\x.smile(x)>, STORE=(/)]] -> 'smiles'
                    NP[SEM=[CORE=<@x>, STORE=(<bo(\P.P(cyril),@x)>)]] -> 'Cyril'
                    """

                    "------------------------------------------------------- Tree Map ------------------------------------------------------------------"

                                                                         'Cyril smiles'
                    "                                                           |                                                                       "
		                                                   'S[SEM=[CORE=<?vp(?np)>, STORE=(?b1+?b2)]]'
		            "                               ____________________________|__________________________________                                     "                                  
		            "                              /                                                               \                                    "
		            "          'NP[SEM=[CORE=?np, STORE=?b1]]'                                        'VP[SEM=[CORE=?vp, STORE=?b2]]'                   "
		            "                             |                                                                     |                               "
		                'NP[SEM=[CORE=<@x>, STORE=(<bo(\P.P(cyril),@x)>)]]'                                       'VP[SEM=?s]' 
                    "                             |                                                                     |                               "
                                               'Cyril'                                                            'IV[SEM=?s]'
                    "                                                                                                   |                               "                            
                                                                                                      'IV[SEM=[CORE=<\x.smile(x)>, STORE=(/)]]'
                    "                                                                                                   |                               "
                                                                                                                     'smiles'

                    "-----------------------------------------------------------------------------------------------------------------------------------"

                    # Example 2 - 
                    "Every girls chases a dog"
                    "CORE = <chase(z1,z2)>"
                    "STORE = (bo(\P.all x.(girl(x) -> P(x)),z1), bo(\P.exists x.(dog(x) & P(x)),z2))"
                    *"Address @x why it is important - P - 396"

                    # More formal example -
                    ":: the module nltk.sem.cooper_storage deals with the task of turning storage-style semantic representation into standard logical forms."
                    from nltk.sem import cooper_storage as cs
                    sentence = 'every girl chases a dog'
                    trees = cs.parse_with_bindops(sentence, grammar='grammars/book_grammars/storage.fcfg')
                    semrep = trees[0].node['SEM']
                    cs_semrep = cs.CooperStore(semrep)
                    print cs_semrep.core
                    "chase(z1,z2)"
                    for bo in cs_semrep.store:
                    	print bo
                    "bo(............)"
                    "bo(............)"
                    # then we call s_retrieve() and check the reading
                    cs_semrep.s_retrieve(trace=True)
                    "permutation 1"
                    "    (\P.all x.(girl(x) -> P(x))) (\z1.chase(z1,z2))"
                    "    (\P.exists x.(dog(x) & P(x))) (\z2.all x......)"
                    "permutation 2"
                    "    (.............................................)"
                    "    (.............................................)"
                    for reading in cs_semrep.readings:
                    	print reading
                    "exists x.(dog(x) & all z3.(girl(z3) -> chase(z3,x)))"
                    "all x.(girl(x) -> exists z4.(dog(z4) & chase(x,z4)))"



[5] "Discourse Semantics"

"[Discourse]" - "a sequence of sentences" - "Angus owns a dog. It bit Irene."

1. "Discourse Representation Theory"
"The scope of quantifier could go beyound one sentence." --> "Angus owns a dog. It bit Irene." --> 'exist x.(dog(x) & own(Angus,x) & bite(x,Irene))' # NP 'a dog' acts like a qunatifier which binds 'it' in the second sentence

**"[Discourse Representation Structure - DRS]" - "presents the meaning of discourse in terms of a list of discourse referents and a list of conditions."
                                                 "[discourse referents]" - "are the things under discussion in the discourse - individual variables of first-order logic"
                                                 "[DRS conditions]" - "apply to those discourse referents, and correspond to atomic open formulas of first-order logic."

                                                 "---------------------------------- DRS ----------------------------------"

                                                      'x y'                                         'x y u z'

                                                      'Angus(x)'                                    'Angus(x)'
                                                      'dog(x)'                                      'dog(x)'
                                                      'own(x,y)'                                    'won(x,y)'
                                                                                                    ' u = y '
                                                                                                    'Irene(z)'
                                                                                                    'bite(u,z)'
 
                                                    'Angus owns a dog'                    'Angus owns a dog. It bits Irene.'
                                                 "-------------------------------------------------------------------------"

                                                 "'it' in the sceond sentence trigger search for '[anaphoric antecedent]' like 'dog' "
                                                 *"In order to process DRSs computationally, need to convert then into linear format: "
                                                                  "List of referents: [x,y] + List of conditions: [angus(x), dog(x), own(x,y)]" - "([x,y],[angus(x), dog(x), own(x,y)])"
                                                                  # Build DRS object in nltk use strings:
                                                                  dp = nltk.DrtParser()
                                                                  drs1 = dp.parse('([x,y],[angus(x), dog(x), own(x,y)])')
                                                                  drs1.draw()
                                                                  "plot..."
                                                                  print drs1.fol() # convert back to first order logic
                                                                  "exists x y.((angus(x) & dog(y)) & own(x,y))"
                                                                  # Use DRS concatenation operator + two drs (automatically alpha-convert)
                                                                  drs2 = dp.parse('([x],[walk(x)]) + ([y],[run(y)])')
                                                                  print drs2.simplify() # lambda-reduction
                                                                  "([x,y],[walk(x),run(y)])"
                                                                  # embed one DRS within another - this how univeral quantification is handled
                                                                  drs3 = dp.parse('([], [(([x],[dog(x)]) -> ([y],[ankle(y),bite(x,y)]))])')
                                                                  print drs3.fol()
                                                                  "all x.(dog(x) -> exists y.(ankle(y) & bite(x,y)))"
                                                                  # DRT allows design to anaphoric pronouns 'it' to be interpreted by linking to exisiting discourse referents [...]
                                                                  drs4 = dp.parse('([x,y], [angus(x),dog(x),own(x,y)])')
                                                                  drs5 = dp.parse('([u,z], [PRO(u),irene(z),bite(u,z)])')
                                                                  drs6 = drs4 + drs5
                                                                  print drs6.simplify()
                                                                  "([x,y,u,z],[angus(x),dog(x),won(x),PRO(u),irene(z),bite(u,z)])"
                                                                  print drs6.simplify().resolve_anaphora() # allows... [....] | can use alternative process for more intelligence
                                                                  "([x,y,u,z],[angus(x),dog(x),won(x),(u = [x,y,z]),irene(z),bite(u,z)])"

                                                 *"DRS is fullycompatible with the existing machinary for handling 'lambda-reduction' and so it is striaght-forward to build compostional semantic "
                                                  "representation that based on DRT rather than First-Order Logic."
                                                  # Example - 
                                                  "----------------- Example Tree Map ---------------------------- "

                                                                            'a dog' 
                                                  "                            |                                   "
                                                           'NP[NUM=sg,SEM=<\Q.(([x],[dog(x)]) + Q(x))>]'                     # result of lower 2 abstracts
                                                  "                 |                       |                      "
                                                  "                 |                       |                      "
                         'Det[NUM=sg,SEM=<\P Q.((([x],[]) + P(x)) + Q(x))>]'        'Nom[NUM=sg,SEM=<\x.([],[dog(x)]>]'      # 2 abstract combine and go up
                         	                      "                 |                       |                       "
                         	                                       'a'                'N[NUM=sg,SEM=<\x.([],[dog(x)])>]'
                                                  "                                         |                       "
                                                                                          'dog'
                                                  # divide into:
                                                  ** 'lambda-abstract' - '\x.([],[dog(x)])' -> 'leads to' -> '\Q.(([x],[]) + ([],[dog(x)]) + Q(x))' -> 'simply so that' -> '\Q.(([x],[dog(x)]) + Q(x))' # go up in the tree
                                                  # in nltk
                                                  # parse with grammar - drt.fcfg
                                                  from nltk import load_parser
                                                  parser = load_parser('grammars/book_grammars/drt.fcfg', logic_parser=nltk.DrtParser()) # using DRT
                                                  trees = parser.nbest_parse('Angus owns a dog'.split())
                                                  print trees[0].node['SEM'].simplify() # lambda reduction
                                                  '([x,z2],[Angus(x),dog(z2),own(x,z2)])'




2. "Discourse Processing"

"Though we learnt rich set of context representations, determined by preceding context and our background assumptions. "
"DRT provides a theory of how the meaning of a sentence is integrated into a representation of the prior discourse."

'[Missed Topic #1]' - 'There has been no attempt to incorporate any kind of inference.'
'[Missed Topic #2]' - 'We have only process individual sentences.'


'Addressed by nltk:' nltk.inference.discourse

"If a discourse = a sequence s1,...,sn of sentences."
*'[discourse thread]' - "is a sequence s1-ri,...,sn-rj "
"The nltk module processes sentence incrementally, keeping track of all possible threads when there is ambiguity."

dt = nltk.DiscourseTester(['A student dances','Every student is a person'])
dt.readings()
"s0 readings: s0-r0: exists x.(student(x) & dance(x))"
"s1 readings: s1-r0: all x.(student(x) -> person(x))"

# When a new sentence is added to the current discourse
dt.add_sentence('No person dances', consistchk=True)
"Inconsistent discourse d0 ['s0-r0','s1-r0','s2-r0']: "
"s0-r0: exists x.(student(x) & dance(x))"
"s1-r0: all x.(student(x) -> person(x))"
"s2-r0: -exists x.(person(x) & dance(x))"
dt.retract_sentence('No person dances', verbose=True) # remove sentence
"current sentences are"
"s0: A srtudent dances"
"s1: Every student is a person"

# Use 'informchk=True' to check whether a new sentence is informative relative to current discourse
dt.add_sentence('A person dances', informchk=True)
"Sentence 'A person dances' under reading 'exists x.(person(x) & dance(x))"
"Not informative relative to thread 'd0' "
# Also possible to pass in an additional set of assumptions as background knowledage and use these to filter out inconsistent readings
'http://www.nltk.org/howto'

# The discourse module can accommodate semantic ambiguity and filter out readings that are not admissible
"P-401 details..."
'Every dog chases a boy' | 'He runs'
from nltk.tag import RegexpTagger
tagger = RegexpTagger(
	[('^(chases|runs)$','VB'),
	 ('^(a)$','ex_quant'),
	 ('^(every)$','univ_quant'),
	 ('^(dog|boy)$','NN'),
	 ('^(He)$','PRP')]
	)
rc = nltk.DrtGlueReadingCommand(depparser=nltk.MaltParser(tagger=tagger))
dt = nltk.DiscourseTester(['Every dog chases a boy','He runs'], rc)
dt.readings()
"s0 readings: "
"s0-r0: ([],[(([x],[dog(x)]) -> ([z3],[boy(z3), chases(x,z3)]))])" # first sentence has two possible readings
"s0-r1: ([z4],[boy(z4),(([x],[boy(x)]) -> ([],[chases(x,z4)]))])"

"s1 readings: "
"s1-r0: ([x],[PRO(x),runs(x)])"

dt.readings(show_thread_readings=True) # check which reading works or not
"d0: ['s0-r0','s1-r0']: INVALID: AnaphoraResolutionException"
"d1: ['s0-r1','s1-r0']: ([z6,z10],[boy(z6),(([x],[dog(x)]) -> ([],[chases(x,z6)])),(z10 =z6), runs(z10)])"

# Filter out inadmissible readings by passing parameter 'filter=True'
dt.readings(show_thread_readings=True, filter=True)
"d1: ['s0-r1','s1-r0']: ([z6,z10],[boy(z6),(([x],[dog(x)]) -> ([],[chases(x,z6)])),(z10 =z6), runs(z10)])"








>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " [11]  Managing Linguistic Data "































