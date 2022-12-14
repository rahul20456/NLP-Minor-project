# NLP-TWITTER SENTIMENT ANALYSIS


Dataset

We have curated a subsample of Twitter Sentiment Analysis dataset and this
subsample is attached for your reference (link). This dataset has a total of 4k samples. We also
curated a subset from these tweets and created a test set. This test set will be later used for
predictions.


TEXT PREPROCESSING 

Whenever we have textual data, we need to apply several preprocessing steps to transform text
into numerical features that work with ML algorithms. The preprocessing steps for a problem
depend mainly on the domain and the problem itself.
a. Tokenization 
b. Spelling correction 
c. Stemming/Lemmatization 
d. Punctuations removal 
e. Using regex remove stopwords 
f. Using regex remove extra whitespaces 
g. Using regex remove URL and HTML tag 


VISUALIZATION

Data visualization shows how the data looks like and what kind of correlation is held by the
attributes of data. A word cloud is a text visualization that displays the most used words in a text
from small to large, according to how often each appears.


RULE-BASED SENTIMENT ANALYSIS

Sentiment Analysis is the process of ‘computationally’ determining whether a piece of writing is
positive, negative, or neutral. VADER (Valence Aware Dictionary and sEntiment Reasoner) is a
lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments
expressed in social media.


Using bi-gram language model (LM) on the given dataset, generated 500 sentiment-oriented
sentences. The probability of the generated samples should be higher than neutral sentences,
i.e., Prob(S1) > Prob(S2), where S1 is a sentiment-oriented sentence and S2 is a non-sentiment-oriented (or neutral) sentence.


Model training

we trained  ML model of our choice. A sample code snippet is provided below for Naive Bayes classifier from scikit-learn. we 
use this code for this evaluation  We will treat the ML model as a blackbox where
it will return an accuracy for the test set, i.e., f(Trainset, Testset) =
accuracy_Testset.
a. Train a ML model on dataset A and compute accuracy (Acc_A) on the test set.
b. Train the same model again but this time using dataset B and compute accuracy
(Acc_B) on the test set.



Sample Snippet for ML model and accuracy calculator using Sklearn.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
def train_and_evaluate(train_sentences, train_labels,
test_sentences, test_labels):
'''
parameters:
train_sentences : list of training sentences
train_labels : list of training labels
test_sentences : list of test sentences
test_labels : list of test labels
output:
accuracy : accuracy of the test set
'''
# Model building
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
# Training the model with the training data
model.fit(train_sentences, train_labels)
# Predicting the test data categories
predicted_test_labels = model.predict(test_sentences)
return accuracy_score(test_labels, predicted_test_labels)


