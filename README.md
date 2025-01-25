# 🚀 Fake News Detection

In today’s digital era, **fake news** has become a growing concern, leading to widespread misinformation and societal consequences. This project leverages **Machine Learning (ML)** and **Natural Language Processing (NLP)** techniques to accurately detect fake news articles, ensuring better dissemination of authentic information.

---

## 🌟 Features

- **Dataset Loading and Cleaning**: Efficiently loads datasets (`Fake.csv` and `True.csv`), removes null values, and cleans text for analysis.
- **Text Preprocessing**: Prepares raw text using techniques such as tokenization, stop word removal, stemming/lemmatization, and vectorization.
- **Machine Learning Models**: Implements advanced classification models:
  - Logistic Regression
  - Random Forest
  - Support Vector Machines (SVM)
- **Model Evaluation**: Uses key performance metrics for evaluation:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- **Interactive Interface**: User-friendly application built with **Streamlit**, providing real-time predictions.
- **Deployment**: Fully hosted and accessible online using **Streamlit**.

---

## 📊 Dataset

- **Source**: Datasets sourced from reliable repositories (e.g., Kaggle or public datasets).
- **Format**: CSV files with the following structure:
  - `Fake.csv`: Labeled fake news articles.
  - `True.csv`: Labeled authentic news articles.

### 📁 Key Dataset Details:
| **Attribute** | **Description**            |
|---------------|----------------------------|
| `Title`       | Headline of the news       |
| `Text`        | Full news content          |
| `Label`       | Classification (Fake/True) |

---

## 🧠 AI Techniques

### 🔍 **Natural Language Processing (NLP)**
- **Preprocessing**:
  - Tokenization
  - Stop word removal
  - Stemming/Lemmatization
- **Vectorization**:
  - TF-IDF
  - Count Vectorizer

### 🤖 **Machine Learning Models**
- Logistic Regression
- Random Forest
- Support Vector Machines (SVM)

### 📈 **Performance Metrics**
- **Accuracy**: Measures the percentage of correctly predicted labels.
- **Precision**: Evaluates the correctness of positive predictions.
- **Recall**: Measures how many actual positive cases were identified.
- **F1-Score**: Provides a balance between precision and recall.

---

## ⚙️ Tools and Technologies

| **Category**         | **Technologies Used**                          |
|-----------------------|-----------------------------------------------|
| **Programming**       | Python                                       |
| **Libraries**         | NLTK, spaCy, Scikit-learn, TensorFlow/Keras  |
| **Data Analysis**     | Pandas, NumPy                                |
| **Visualization**     | Matplotlib, Seaborn                          |
| **Deployment**        | Streamlit                                    |
| **Environment**       | Jupyter Notebook                             |

---

## 🛠️ System Architecture

Below is an architectural representation of the project:

![System Architecture](https://github.com/user-attachments/assets/406838dc-87db-4042-8603-4039c22c8857)


