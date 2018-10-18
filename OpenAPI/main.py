import os
import yaml
import sys
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from string import ascii_lowercase
import string
import re, nltk
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import pca
from sklearn.grid_search import GridSearchCV
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)

class LemmaCountVectorizer(CountVectorizer):
	def build_analyzer(self):
		lemm = WordNetLemmatizer()
		analyzer = super(LemmaCountVectorizer, self).build_analyzer()
		return lambda doc: (lemm.lemmatize(w) for w in analyzer(doc))

# You should change this variable to whenever your API yaml is.
PATH_YAMLS = 'C:/Users/L/Documents/Corpus/APIs'
SAVED_CORPUS = 'corpus.txt'
NUM_TOPICS = 3
TF_VECTORIZER = LemmaCountVectorizer(max_df=0.95, 
										 min_df=2,
										 stop_words='english',
										 decode_error='ignore')

# We need this work-around for some documents, because the yaml loader couldn't handle some cases										 
def value_constructor(loader, node):
	value = loader.construct_scalar(node)
	return value

# Generating Corpus using the API
def read_yamls_by_doc(path):
	data = []
	
	for base, dirs, files in os.walk(path):
		for f in files:
			text_document = ''
			
			if f == "swagger.yaml" or f == "openapi.yaml":

				print(base[32:], f)
				yaml.add_constructor(u'tag:yaml.org,2002:value', value_constructor)
				yaml_data = yaml.load(open(os.path.join(base,f), encoding="utf8"))

				if 'description' in list(yaml_data['info'].keys()):
					text_document = str(yaml_data['info']['description']) + ' '

				for api in yaml_data['paths'].keys():
					for methodHTTP in yaml_data['paths'][api].keys():
						if (type(yaml_data['paths'][api][methodHTTP]) == dict) and 'summary' in list(yaml_data['paths'][api][methodHTTP].keys()):
							text_document += str(yaml_data['paths'][api][methodHTTP]['summary']) + ' '

				text_document = text_document[0:len(text_document)-1] #Remove the last space added to the end of the string
				data.append(text_document)

	return data

# This method saves the created corpus in a file, so it's faster to get it to a variable later
def save_data(data, file):
	with open(file, "wb") as fp:
		pickle.dump(data, fp)

# This method loads the created corpus from a file
def load_data(file):
	with open(file, "rb") as fp:
		data = pickle.load(fp)
		return data


def tokenization(data_list):
	#Tokenization with NLTK
	tokenized_data = []
	for sentence in data_list:
		tokenized_data.append(word_tokenize(sentence.lower()))
	
	return tokenized_data

def stop_words_removal(tokenized_data):
	#Removing Stopwords
	#nltk.download('stopwords')
	stpwords = stopwords.words('english')
	stpwords.extend(['WWW','www','HTTP', 'http', 'HTTPS', 'https', 'api', 'API','apis', 'APIS', 'REST', 'rest','RESTfull',
					  'RESTful', 'restfull', 'restful', 'service', 'services', 'user', 'users','get','post','put','delete',
					  'request', 'response', 'url', 'URL', 'curl','token', 'json', 'crud', 'create', 'update', 'data',
					  'code', 'list', 'tokens', 'urls','html', 'html5','yaml', 'httpclient','httprequest','httpresponse',
					  'client','clients', 'lenguage', 'return', 'type']
					  + list(ascii_lowercase))

	words_filtered=[]
	
	for sentence in tokenized_data:
		sentences=[]
		for word in sentence:
			if word not in stpwords and word not in string.punctuation and re.match('[a-zA-Z\\-][a-zA-Z\\-]{3,12}', word):
				sentences.append(word)
		words_filtered.append(sentences)
		
	#print(words_filtered[:2])	

	#Only one doc for file
	new_data = []
	for words in words_filtered:
		text = ' '.join(words)
		new_data.append(text)
	#print(new_data[:2])
	
	return new_data
		
def vectorization(new_data):		
	#nltk.download('wordnet')
	data_vectorized = TF_VECTORIZER.fit_transform(new_data)

	#print("The features are: \n\n {}".format(TF_VECTORIZER.get_feature_names()[:100]) + "\n")
	#print(data_vectorized.toarray())
	#return data_vectorized, tf_vectorizer
	return data_vectorized
	
def pre_processing(data_list):
	tokenized_data = tokenization(data_list)
	new_data = stop_words_removal(tokenized_data)
	data_vectorized = vectorization(new_data)
	
	return data_vectorized

# Modelling 	

def fit_model_lda(data_vectorized):
	# Define Search Param
	search_params = {'n_components': [3, 5, 10, 15], 'learning_decay': [.5, .7, .9], 'max_iter':[10, 20, 30]}

	# Init the Model
	lda = LatentDirichletAllocation()

	# Init Grid Search Class
	model = GridSearchCV(lda, param_grid=search_params)

	# Do the Grid Search
	model.fit(data_vectorized)
	
	return model

def best_model(model, data_vectorized):
	# Best Model
	best_lda_model = model.best_estimator_

	# Model Parameters
	print("Best parameters for model: ", model.best_params_)

	# Log Likelihood Score
	print("Log Likelihood Score: ", model.best_score_)

	# Perplexity
	print("Model perplexity: ", best_lda_model.perplexity(data_vectorized))
	
	return best_lda_model

# Print LDA Topics
def print_top_words(model, feature_names, n_top_words):
    for index, topic in enumerate(model.components_):
        message = "\nTopic #{}:".format(index)
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1 :-1]])
        print(message)
        print("="*90)
		
def top_topics(best_lda_model, n_top_words):
	print("\nTopics in LDA model: ")
	tf_feature_names = TF_VECTORIZER.get_feature_names()
	print_top_words(best_lda_model, tf_feature_names, n_top_words)
	
	
def fit_model_lsa(data_vectorized):
	# Init the Model
	lsi = TruncatedSVD(n_components=NUM_TOPICS)

	# Init Grid Search Class
	lsi_Z = lsi.fit_transform(data_vectorized)
	
	return lsi

# Print LSA Topics
def print_topics(model, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(TF_VECTORIZER.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])
	

	
def main():
	#data_list = read_yamls_by_doc(PATH_YAMLS)
	data_list = load_data(SAVED_CORPUS)
	data_vectorized = pre_processing(data_list)
	
	# LDA - Training the model and printing the top topics
	model_lda = fit_model_lda(data_vectorized)
	best_model_lda = best_model(model_lda, data_vectorized)
	
	top_topics(best_model_lda, 35)
	
	# LSA training the model and printing the top topics
	model_lsa = fit_model_lsa(data_vectorized)
	print_topics(model_lsa)
	
if __name__ == "__main__":
    main()