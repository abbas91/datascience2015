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







>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " [1] Basic Stats from Text "

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

















