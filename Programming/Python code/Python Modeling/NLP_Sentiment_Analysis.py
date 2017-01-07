# NLP - Sentiment Analysis #

" Classify documents based on author's opion and emotions "


##################################################
#                                                #
#                                                #
#               Bag-of-words Model               #
#                                                #
#                                                #
##################################################

" Create token of each vocablary - count frequency "


# -------- Download py lib and dataset --------- #
$ tar -zxf aclImdb_v1.tar.gz # download and decompress it
$ pip install pyprind # downliad py lib - progess indicator

# >>>>>>> Use of Pyprind >>>>>> #
# [1]
n = 10000                # initiate progress bar
pbar = pyprind.ProgBar(n)
for i in range(n):
	n += 1
	pbar.update() # // call update on basic inter
"0%                 100%"
"[############         ]"

# [2]
n = 10000                # initiate progress percent bar
pperc = pyprind.ProgPercent(n)
for i in range(n):
	n += 1
	pperc.update() # // call update on basic inter




# -------- Load Libraries ----------- #
import pyprind
import pandas as pd
import os


# -------- Loading download tar.gz fileball ---------- #
pbar = pyprind.ProgBar(50000) # // initiate progess iters = number of files needed to load
labels = {'pos':1, 'neg':0} # // create dict for convert label to numbers
df = pd.DataFrame() # // create empty df to store loaded data

for s in ('test', 'train'): # // in loop of test, train folders
    for l in ('pos', 'neg'): # // in loop of pos, neg folders within test and train folders
        path = './aclImdb/%s/%s' % (s, l) # // build the dir path for the combo 4 nest folders
        for file in os.listdir(path): # // for all files in each combo folder
            with open(os.path.join(path, file), 'r') as infile: # // open the file object
                txt = infile.read() # // read the whole file as txt
                df = df.append([[txt, labels[l]]], ignore_index=True) # // append the txt and "l" - the label(number) into empty df
                pbar.update() # // update progess bar done with each file
df.colnums = ['review', 'sentiment'] # // rename colnme when all file appended into df



# --------- write into one file ------------ #
import numpy as np
np.random.seed(0) # // set seed
df = df.reindex(np.random.permutation(df.index)) # // shuffle the original index and applied it to df
df.to_csv('./movie_data.scv') # // write into a single file

df = pd.read_csv('./movie_data.scv') # // read the new file back in





# --------- Preprocess data ------------ #
import re # // regex
def preprocessor(text):
	text = re.sub('<[^>]*>', '', text) # // remove HTML markups
	emotions = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P', text) # // split the emotion pucuations
	text = re.sub('[\W]+', ' ', text.lower()).join(emotions).replace('-', '') # // remove all none words pucuations but + emtions pucuations + ':-) -> :)'
	return text

df['review'] = df['review'].apply(preprocessor) # apply preprocess to all sets



# -------- Tokenize data ----------- #
# [1] Simply split by space
def tokenizer(text):
	return text.split()



# [2] use word stemming (transform words into their roots)
" Other stemming algorithm - Porter, Snowball, Lancaster "
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
	return [porter.stem(word) for word in text.split()] # porter.stem() on single word



# [3] also remove 'meaningless' words - and, he, she, etc
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop = stopwords.words('english')
[w for w in tokenizer_porter('xxxxxxx') if w not in stop]




# ------------ Transforming words into features vectors ----------------- #
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer() # // ex.   CountVectorizer(ngram_range=(2,2)) : use different n-gram model ??
" The sun is shining "
" 1-gram: 'the', 'sun', 'is', 'shining' "
" 2=gram: 'the sun', 'sun is', 'is shining' " # different n used for different application, for spam 3-4 gram best

docs = np.array([
	'we have one tree',
	'we have two trees',
	'we have three trees'])
bag = count.fit_transform(docs)
count.vocabulary_ # // single voca with index
" {'we':2, 'have':3, ...}"
bag.toarray() # // by column = index 'we' = 3rd column (index=2) 
""" [[0 1 1 ...]
     [1 0 1 ...]
     [0 2 1 ...]] """

# Assess word relevancy via term frequency-inverse document frequency -> HIgh frequency words have everywhere, meaningless, reweighted
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
np.set_printoptions(precision=2)
bag_new = tfidf.fit_transform(bag).toarray()
""" [[0. 0.43 0.56 ...]
     [0. 0.31, 0.  ...]
     [0.18, 0. 0.  ...]] """ 






# ---------------------- Split datasets ------------------------- #
X_train = df.loc[:25000, 'review'].values
Y_train = df.loc[:25000, 'sentiment'].values

X_test = df.loc[25000:, 'review'].values
Y_test = df.loc[25000:, 'sentiment'].values





# ---------------------- Modeling Pipeline ------------------------- #
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer # // combine(CountVectorizer() + TfidfTransformer())

tfidf = TfidfVectorizer(strip_accents=None, # // set up tokenization object
	                    lowercase=False,
	                    preprocessor=None)

param_grid = [{'vect__ngram_range': [(1,1)], # // gird search for all pareas
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},

               {'vect__ngram_range': [(1,1)],
	            'vect__stop_words': [stop, None],
	            'vect__tokenizer': [tokenizer, tokenizer_porter],
	            'vect__use_idf': [False], # tfidf (Default = True)
	            'vect__norm': [None], # tfidf (Default = True)
	            'clf__penalty': ['l1', 'l2'],
	            'clf__C': [1.0, 10.0, 100.0]}]

lr_tfidf = Pipeline([('vect', tfidf),
	                 ('clf', LogisticRegression(random_state=0))]) # // put together estimator pipeline

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1) # // put together GS pipeline

gs_lr_tfidf.fit(X_train, Y_train) # Train model

gs_lr_tfidf.best_params_ # best combinations params

gs_lr_tfidf.best_score_ # Training accuracy

clf = gs_lr_tfidf.best_estimator_
clf.score(X_test, Y_test) # Testing accuracy






# ------------------- Bouns: Online algorithms & Out-of-core learning ----------------------------- #
" stochastic gradient descent - updating model's weights using one ssample at a time "
" Use partial fit - GDS - using min-batch data "



# ---- Define 'tokenizer function to clean unprocessed data' ------ #
import numpy as np
import re
from nltk.corpus import stopwords
stop = stopwords.words('english')
def tokenizer(text):
	text = re.sub('<[^>]*>', '', text) # // remove HTML markups
	emotions = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P', text) # // split the emotion pucuations
	text = re.sub('[\W]+', ' ', text.lower()).join(emotions).replace('-', '') # // remove all none words pucuations but + emtions pucuations + ':-) -> :)'
	tokenized = [w for w in text.split() if w not in stop] # remove stop-words
	return tokenized



# ---- Define a generator function reads in and returns one document at a time ---- #
def stream_docs(path):
	with open(path, 'r', encoding='utf-8') as csv:
		next(csv) # skip header
		for line in csv:
			text, lablel = line[:-3], int(line[-2])
			yield text, label # // use yield in a loop -> form a generator (Not store iterable in memory, use and discard, only use once)

"test"
next(stream_docs(path='./movie_data.csv'))
" ('xxxxxxxxxxxx', 1) "


# ---- Define an function takes a document stream and return particular number of documents ---- #
def get_minibatch(doc_stream, size):
	docs, y = [], []
	  try:
	  	for _ in range(size):
	  		text, label = next(doc_stream)
	  		docs.append(text)
	  		y.append(label)
	  except StopIteration:
	  	return None, None
	  return docs, y



# ------ Create vectorizer to transform to numeric features ------- #
# ------ Initiate model with SGD -------- #
" CountVectorizer, TfidfVectorizer can't be used since need to load all feature vectors of training in memorry "
" Use 'HashingVectorizer' - 32-bit murmurhash3 algorithm "
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifer

vect = HashingVectorizer(decode_error='ignore',
	                     n_features=2**21, #// number of features to 2^21
	                     preprocessor=None,
	                     tokenizer=tokenizer) #// Use previous fun

clf = SGDClassifier(loss='log', random_state=1, n_iter=1) # // 'loss' - logistic Regression
doc_stream = stream_docs(path='./movie_data.csv')


# ---- Start out-of-core learning ---- #
import pyprind
pbar = pyprind.ProgBar(45) # 45 iterations

classes = np.array([0, 1])
for _ in range(45): # for each iteration
  X_train, Y_train = get_minibatch(doc_stream, size=1000) # read 1000 line for each iteration and split into train, test
  if not X_train:
  	break
  X_train = vect.transform(X_train) #// vectorized + tokenized trainset
  clf.partial_fit(X_train, Y_train, classes=classes) # // partial fit the model
  pbar.update()





# ---- Test the model performance ---- #
X_test, Y_test = get_minibatch(doc_stream, size=5000) # // the line has recorded, go forward
X_test = vect.transform(X_test) # // vectorized + tokenized trainset
clf.score(X_test, Y_test) # sccuracy rate on the test
# -- finally add the last 5000 test set to fit the model -- #
clf = clf.partial_fit(X_test, Y_test)














##################################################
#                                                #
#                                                #
#           Latent Dirichlet allocation          #
#                                                #
#                                                #
##################################################












































































































##################################################
#                                                #
#                                                #
#               word2vec (Google)                #
#                                                #
#                                                #
##################################################


























