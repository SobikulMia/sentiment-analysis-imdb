# IMDb Movie Review Sentiment Analysis

This project performs sentiment classification on IMDb movie reviews using Natural Language Processing (NLP) techniques and machine learning models.  
The goal is to classify movie reviews into positive or negative sentiments based on the text content.

---

##  Libraries and Tools Used
- Python 3.x
- Pandas, NumPy
- Matplotlib, Seaborn, WordCloud (for visualization)
- NLTK (for text preprocessing and lemmatization)
- Scikit-learn (for machine learning models and evaluation)
- Regex (for text cleaning)
- Jupyter Notebook (for experimentation)

---

##  Dataset

The project uses the **IMDb Dataset** of 50,000 movie reviews, available on Kaggle:  
[IMDb Dataset on Kaggle](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

> **Note:** The dataset is **not included** in this repository due to size restrictions.  
> Please download it manually and place the CSV file in the root directory or a `data/` folder.

---

##  Project Overview

1. **Data Loading and Sampling:**  
   Loaded a random sample of 20,000 reviews from the full dataset for faster experimentation.

2. **Exploratory Data Analysis (EDA):**  
   - Analyzed sentiment distribution (positive vs negative).  
   - Created word clouds for positive and negative reviews to visualize common words.

3. **Text Preprocessing:**  
   - Converted text to lowercase.  
   - Expanded contractions (e.g., "can't" → "can not").  
   - Removed HTML tags, digits, punctuations, URLs, emails, and short words.  
   - Handled negations by joining "not" with the next word (e.g., "not good" → "not_good").  
   - Removed stopwords and performed lemmatization.

4. **Feature Extraction:**  
   - Applied TF-IDF vectorization with unigrams, bigrams, and trigrams.  
   - Limited to top 3000 features.

5. **Model Training and Hyperparameter Tuning:**  
   - Models trained: Logistic Regression, Multinomial Naive Bayes, Random Forest, Support Vector Classifier (SVC), Gradient Boosting Classifier.  
   - Used GridSearchCV with 5-fold cross-validation to find best hyperparameters.

6. **Evaluation:**  
   - Reported Accuracy, Classification Report, Confusion Matrix, and F1 Score on the test set.  
   - Selected the best performing model based on cross-validation accuracy.


