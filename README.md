# Disaster-Tweet-Classification

With the proliferation of internet-based communication technologies, people are increasingly using social media during disasters. Hence, disaster tweet classification plays a significant role to support situation awareness and disaster response broadly. Therefore, this project aims at classifying disaster tweets in to two categories, namely, Related and Not-Related.


![Disaster Tweet Classification](https://github.com/nilani-rangika/Disaster-Tweet-Classification/blob/main/Tweet_classification.png)

## The dataset

| Description   | Article       | 
| ------------- | ------------- |
| [CrisisLexT26](https://crisislex.org/data-collections.html#CrisisLexT26) | What to expect when the unexpected happens: Social media communications across crises | 
| [CrisisLexT6](https://crisislex.org/data-collections.html#CrisisLexT6)  | Crisislex: A lexicon for collecting and filtering microblogged communications in crises| 
| [CrisisNLP-Resource#1](https://crisisnlp.qcri.org/lrec2016/lrec2016.html) | Twitter as a lifeline: Human-annotated twitter corpora for NLP of crisis-related messages  |
| [CrisisNLP-Resource#2](https://crisisnlp.qcri.org/)  | Practical extraction of disaster-relevant information from social media  | 
| [CrisisNLP-Resource#5](https://crisisnlp.qcri.org/crisismmd)  | Crisismmd: Multimodal twitter datasets from natural disasters  | 
| [CrisisNLP-Resource#10](https://crisisnlp.qcri.org/)  | Graph based semi-supervised learning with convolution neural networks to classify crisis related tweets  | 
| [Kaggle Real or Not? NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started/overview)  | - | 
| [Appen Disaster Response Messages](https://appen.com/datasets/combined-disaster-response-data/)  | -  | 
| [Kaggle Disasters on social media](https://www.kaggle.com/jannesklaas/disasters-on-social-media)  | -  | 

## The models

We evaluate the performance of twelve Machine Learning models and two Deep Learning models using three different word embeddings for the classification task.

1.	Linear SVM
2.	RidgeClassifier
3.	Logistic Regression
4.	Decision Tree
5.	k-Nearest Neighbors
6.	Gradient Boosting Classifier
7.	Random Forest
8.	AdaBoost
9.	Naïve Bayes
10.	Perceptron
11.	xgboost
12.	catboost
13.	LSTM-fasttext
14.	CNN-fasttext
15.	LSTM-GloVe
16.	CNN-Glove
17.	LSTM-W2V
18.	CNN-W2V

## Adopted CNN and Bi-LSTM architectures

![Image of CNN and LSTM](https://github.com/nilani-rangika/Disaster-Tweet-Classification/blob/main/DL2.png)

## Noise Reduction and Preprocessing Tweets
### Remove Non-English Tweets
```python
def isEnglish(s):
    return s.isascii()
```

### Remove ReTweets
```python
df = df[~df.tweet_text.str.startswith('RT')]
```

### Remove Duplicate Tweets
```python
df = df.drop_duplicates(subset=['tweet_text'])
```

### Preprocessing
```python
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt

df['cleaned'] = np.vectorize(remove_pattern)(df['tweet_text'], "@[\w]*") #remove @user mensions
df['cleaned'] = df['cleaned'].str.replace("http\S+|www.\S+", " ") #remove http address

from spacy.lang.en import English
nlp = English()
df['cleaned'] = df['cleaned'].apply(lambda row: " ".join([w.lemma_ if w.lemma_ !='-PRON-' else w.lower_ for w in nlp(row)])) #lematization
df['cleaned'] = df['cleaned'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3])) #remove words less than 3 characters 
df = df[~df.cleaned.str.count('\s+').lt(3)] #remove shorter tweets having less than 3 words
```

## Using Weights
```python
import pickle
loaded_vectorizer = pickle.load(open('earthquake/vectorizer_all.sav', 'rb'))
loaded_model_Linear_SVM = pickle.load(open('earthquake/classifier_svm_all.sav', 'rb'))
loaded_model_RidgeClassifier = pickle.load(open('earthquake/classifier_RidgeClassifier_all.sav', 'rb'))
loaded_model_Logistic_Regression = pickle.load(open('earthquake/classifier_LRegression_all.sav', 'rb'))
loaded_model_Decision_Tree = pickle.load(open('earthquake/classifier_DTClassifier_all.sav', 'rb'))
loaded_model_KNN = pickle.load(open('earthquake/classifier_knn_all.sav', 'rb'))
loaded_model_GradientBoostingClassifierM = pickle.load(open('earthquake/classifier_GradientBoostingClassifier_all.sav', 'rb'))
loaded_model_Random_Forest = pickle.load(open('earthquake/classifier_RandomForestClassifier_all.sav', 'rb'))
loaded_model_AdaBoost = pickle.load(open('earthquake/classifier_AdaBoost_all.sav', 'rb'))
loaded_model_MNB = pickle.load(open('earthquake/classifier_MNB_all.sav', 'rb'))
loaded_model_Perceptron = pickle.load(open('earthquake/classifier_ANN_all.sav', 'rb'))
loaded_model_xgboost = pickle.load(open('earthquake/classifier_xgboost_all.sav', 'rb'))
loaded_model_catboost = pickle.load(open('earthquake/classifier_catboost_all.sav', 'rb'))
```
