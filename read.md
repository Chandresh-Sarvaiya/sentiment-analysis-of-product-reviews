# 🚀 Product Review Sentiment Analyzer

An end-to-end Natural Language Processing (NLP) project that classifies Amazon product reviews as positive or negative.



---

## 🎯 Overview

The goal of this project is to automate the analysis of customer feedback. By classifying reviews into **positive** and **negative** sentiments, businesses can quickly understand customer opinions, identify product issues, and improve service. This model was trained on the [Amazon Fine Food Reviews dataset from Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews).

---

## 🛠️ Tech Stack

- **Language:** `Python 3.9`
- **Libraries:**
  - **Data Manipulation:** `Pandas`, `NumPy`
  - **NLP & Preprocessing:** `NLTK`, `scikit-learn`
  - **Data Visualization:** `Matplotlib`, `Seaborn`, `WordCloud`
  - **Notebook:** `Jupyter Notebook`

---

## ⚙️ Project Workflow

1.  **Data Loading & Cleaning:** Loaded the dataset and handled missing values. Converted the 1-5 star ratings into a binary sentiment classification:
    - **Positive (1):** Score > 3
    - **Negative (0):** Score < 3
    - Neutral reviews (Score = 3) were excluded to create a clear binary problem.

2.  **Exploratory Data Analysis (EDA):** Visualized the sentiment distribution and generated word clouds to identify the most frequent words in positive and negative reviews.

3.  **Text Preprocessing:** Implemented a comprehensive text cleaning pipeline:
    - Removed HTML tags and non-alphabetic characters.
    - Converted text to lowercase.
    - **Tokenized** text into individual words.
    - Removed common English **stopwords**.
    - Applied **lemmatization** to reduce words to their base form (e.g., "running" -> "run").

4.  **Feature Engineering & Modeling:**
    - Converted the cleaned text data into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization, considering the top 5,000 features.
    - Trained a **Multinomial Naive Bayes** classification model on the resulting features.



---

## 🚀 How to Run Locally

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Chandresh-Sarvaiya/sentiment-analysis-of-product-reviews.git](https://github.com/Chandresh-Sarvaiya/sentiment-analysis-of-product-reviews.git)
    cd sentiment-analysis-of-product-reviews
    ```

2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

---

## ✨ Future Improvements

- **Advanced Models:** Experiment with more complex models like LSTMs or pre-trained transformers (e.g., BERT from Hugging Face) to potentially improve performance on nuanced reviews.
- **Error Analysis:** Perform a detailed analysis of the reviews that the model misclassifies to understand its weaknesses.
- **Expand Features:** Incorporate additional features like review length or helpfulness counts into the model.