# SMS Spam Detection using NLP and Machine Learning

## Project Overview

This project classifies SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and Machine Learning techniques.

The dataset contains real SMS messages labeled as spam or ham. The messages are preprocessed, converted into numerical features using TF-IDF, and then classified using Machine Learning models.

---

## Dataset

Dataset: SMS Spam Collection Dataset

Columns:

* **v1** → Label (spam / ham)
* **v2** → SMS message

---

## Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-Learn
* Matplotlib

---

## NLP Pipeline

1. Data Loading
2. Text Cleaning

   * Lowercasing
   * Removing special characters
   * Removing digits
   * Removing extra spaces
3. Tokenization
4. Stop Word Removal
5. Lemmatization
6. TF-IDF Vectorization

---

## Machine Learning Models

### Supervised Learning

* Logistic Regression
* Naive Bayes

### Unsupervised Learning

* KMeans Clustering
* Elbow Method for selecting clusters

---

## Evaluation Metrics

The following metrics were used to evaluate model performance:

* Accuracy Score
* Precision Score
* Confusion Matrix

---

## Sample Prediction

Input Message:

Free entry win cash now

Output:

spam

---

## Project Workflow

Dataset
→ Text Cleaning
→ Tokenization
→ Stop Word Removal
→ Lemmatization
→ TF-IDF
→ Train-Test Split
→ Model Training
→ Evaluation
→ Prediction

---

## Learning Outcomes

Through this project, I learned:

* Text preprocessing techniques in NLP
* TF-IDF feature extraction
* Spam message classification
* Logistic Regression and Naive Bayes
* Model evaluation using accuracy, precision, and confusion matrix
* Basic clustering using KMeans

---

## Future Improvements

* Add more Machine Learning models
* Create a web interface using Flask
* Deploy the model as a web application
* Compare TF-IDF with Word Embeddings

---

## Author

Mariya Ann

