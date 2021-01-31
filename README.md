# Disaster-Tweet-Classification

With the proliferation of internet-based communication technologies, people are increasingly using social media during disasters. Hence, disaster tweet classification plays a significant role to support situation awareness and disaster response broadly. Therefore, this project aims at classifying disaster tweets in to two categories, namely, Related and Not-Related.

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
