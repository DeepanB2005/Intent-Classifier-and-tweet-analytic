# Intent-Classifier-and-tweet-analytic
An NLP-based intent classification system that identifies customer queries from social media posts and categorizes them into predefined support intents. The project compares TF-IDF + traditional ML models with Sentence Transformer embeddings + classifiers such as Linear SVM and Logistic Regression.

Social-media text preprocessing
TF-IDF-based intent classification
Sentence Transformer embeddings
Linear SVM / Logistic Regression classification
Evaluation using Accuracy, Precision, Recall, and Macro F1
Designed to handle large-scale customer support datasets
Kaggle-ready training and inference pipeline

Tech Stack: Python · NLP · Scikit-learn · Sentence Transformers · Pandas · PyTorch



I have used 2 feature extraction techniques for converting the input data into numerical features 
the numerical veectors represent the importance of the every token in the input text data
and also the vectors will be in numerical format so that the model can understand and process it 

1.Term-frequency and Inverse Document Frequence
2.used a sentence transformer- All miniL LM v6

after this the both classifiers has used Linear SVM model as i have trained logistic regression,SVC also but linear SVM shows some promising performance

