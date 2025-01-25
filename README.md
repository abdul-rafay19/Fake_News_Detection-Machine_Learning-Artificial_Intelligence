# Fake News Detection

Fake news has become a major concern in the digital age, spreading misinformation and causing significant societal impacts. This project aims to leverage **Machine Learning** and **Natural Language Processing (NLP)** techniques to efficiently and accurately detect fake news articles.

## Features

- **Dataset Loading and Cleaning**: Handles Fake.csv and True.csv datasets by removing null values and cleaning text for processing.
- **Text Preprocessing**: Includes tokenization, stop word removal, stemming/lemmatization, and vectorization for converting text into numerical features.
- **Machine Learning Models**: Implements models like Logistic Regression, Random Forest, and Support Vector Machines (SVM) for binary classification.
- **Model Evaluation**: Assesses model performance using metrics such as accuracy, precision, recall, and F1-score.
- **User-Friendly Interface**: Provides an interactive web-based application developed using Streamlit.
- **Deployment**: Hosts the model online via Streamlit for easy access.

## Dataset

- **Source**: Datasets were obtained from credible repositories (e.g., Kaggle or public research sources).
- **Format**: CSV files containing fields like title, text, and label.
- **Files**:
  - `Fake.csv`: Labeled fake news articles.
  - `True.csv`: Labeled true news articles.

## AI Techniques

1. **Natural Language Processing**:
   - Text preprocessing (tokenization, stemming, stop word removal).
   - Vectorization using TF-IDF or Count Vectorizer.

2. **Machine Learning Models**:
   - Logistic Regression
   - Random Forest
   - Support Vector Machines (SVM)

3. **Performance Metrics**:
   - Accuracy
   - Precision
   - Recall
   - F1-Score

## Tools and Technologies

- **Programming Language**: Python
- **Libraries**:
  - NLP: NLTK, spaCy
  - Modeling: Scikit-learn, TensorFlow/Keras (optional)
  - Data Analysis: Pandas, NumPy
  - Visualization: Matplotlib, Seaborn
  - Deployment: Streamlit
- **Environment**: Jupyter Notebook

## System Architecture

![Project Diagram](path-to-diagram-image) <!-- Replace with an actual link to your diagram -->

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Fake-News-Detection.git
