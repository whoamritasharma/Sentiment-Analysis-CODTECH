# Sentiment-Analysis-CODTECH

## CODTECH IT Solutions - Internship Task 2


## 📋 Project Overview

This project implements a **Sentiment Analysis** system on customer reviews using Natural Language Processing (NLP) techniques. The model classifies reviews as either **Positive** or **Negative** using **TF-IDF Vectorization** and **Logistic Regression**.

**Internship:** CODTECH IT Solutions  
**Task:** Task 2 - Sentiment Analysis with NLP  
**Objective:** Perform sentiment analysis on customer reviews dataset  

---

## 🎯 Objectives

- Preprocess customer review text data
- Extract features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- Train a Logistic Regression model for binary sentiment classification
- Evaluate model performance using various metrics
- Predict sentiments on new, unseen reviews

---

## 🛠️ Technologies Used

### Programming Language
- **Python 3.8+**

### Libraries & Frameworks
- **Data Processing:** pandas, numpy
- **NLP:** nltk (Natural Language Toolkit)
- **Machine Learning:** scikit-learn
- **Visualization:** matplotlib, seaborn, wordcloud

---

## 📂 Project Structure

```
sentiment-analysis-nlp/
│
├── sentiment_analysis.ipynb    # Main Jupyter notebook with complete implementation
├── README.md                    # Project documentation (this file)
├── requirements.txt             # Python dependencies
└── .gitignore                   # Git ignore file
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/sentiment-analysis-nlp.git
cd sentiment-analysis-nlp
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### 5. Launch Jupyter Notebook
```bash
jupyter notebook sentiment_analysis.ipynb
```

---

## 📊 Dataset

The project includes a sample dataset of customer reviews with the following features:
- **review:** The customer review text
- **sentiment:** The sentiment label (positive/negative)

You can replace the sample dataset with your own CSV file containing customer reviews.

**Dataset Format:**
```csv
review,sentiment
"This product is amazing!",positive
"Terrible experience.",negative
```

---

## 🔍 Methodology

### 1. **Data Preprocessing**
   - Convert text to lowercase
   - Remove special characters and numbers
   - Remove stopwords
   - Lemmatization using WordNet Lemmatizer

### 2. **Feature Extraction**
   - TF-IDF Vectorization with:
     - Max features: 1000
     - N-gram range: (1, 2) - unigrams and bigrams
     - Min document frequency: 1
     - Max document frequency: 0.9

### 3. **Model Training**
   - Algorithm: Logistic Regression
   - Train-test split: 80-20
   - Solver: liblinear
   - Regularization parameter (C): 1.0

### 4. **Model Evaluation**
   - Accuracy Score
   - Precision, Recall, F1-Score
   - Confusion Matrix
   - ROC-AUC Curve

---

## 📈 Results

The model achieves the following performance metrics (on the sample dataset):

| Metric | Score |
|--------|-------|
| **Training Accuracy** | ~95-100% |
| **Testing Accuracy** | ~90-100% |
| **AUC-ROC Score** | ~0.95+ |

### Confusion Matrix
The confusion matrix shows the model's prediction performance:
- True Positives (TP): Correctly predicted positive reviews
- True Negatives (TN): Correctly predicted negative reviews
- False Positives (FP): Negative reviews predicted as positive
- False Negatives (FN): Positive reviews predicted as negative

---

## 💡 Key Features

✅ **Text Preprocessing Pipeline**
- Comprehensive text cleaning and normalization

✅ **TF-IDF Vectorization**
- Captures word importance across documents

✅ **Logistic Regression Model**
- Fast, efficient, and interpretable

✅ **Visualizations**
- Word clouds for positive/negative sentiments
- Confusion matrix heatmap
- ROC curve
- Feature importance plots

✅ **Interactive Predictions**
- Predict sentiment on custom reviews
- Confidence scores for predictions

---

## 🎨 Visualizations

The notebook includes the following visualizations:

1. **Sentiment Distribution** - Bar chart and pie chart
2. **Word Clouds** - Separate clouds for positive and negative reviews
3. **Confusion Matrix** - Heatmap showing prediction results
4. **ROC Curve** - Model performance visualization
5. **Feature Importance** - Top positive and negative indicators

---

## 📝 Usage Example

```python
# Import required libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Preprocess text
cleaned_text = preprocess_text("This product is amazing!")

# Vectorize
tfidf_vector = tfidf_vectorizer.transform([cleaned_text])

# Predict
prediction = lr_model.predict(tfidf_vector)
sentiment = 'Positive' if prediction[0] == 1 else 'Negative'

print(f"Sentiment: {sentiment}")
```

---

## 🔮 Future Enhancements

- [ ] Add support for multi-class sentiment classification (positive, negative, neutral)
- [ ] Implement deep learning models (LSTM, BERT)
- [ ] Create a web interface using Flask/Streamlit
- [ ] Add aspect-based sentiment analysis
- [ ] Expand dataset with real-world customer reviews
- [ ] Implement model deployment using Docker

---

## 📚 Learning Outcomes

Through this project, I learned:

✔️ Text preprocessing techniques for NLP  
✔️ TF-IDF feature extraction methodology  
✔️ Binary classification using Logistic Regression  
✔️ Model evaluation and performance metrics  
✔️ Visualization of ML results  
✔️ End-to-end ML pipeline development  


## 🙏 Acknowledgments

- **CODTECH IT Solutions** for the internship opportunity
- **scikit-learn** documentation and community
- **NLTK** for natural language processing tools
- All open-source contributors

---

## 📞 Contact

For any queries or feedback, please reach out:

- **Email:** whoamritasharma@gmail.com

---

## ⭐ Support

If you found this project helpful, please give it a ⭐ on GitHub!

---

**Last Updated:** February 2026  
**Status:** ✅ Completed
