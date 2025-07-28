Twitter Sentiment Analysis with Naive Bayes
Welcome to my sentiment analysis project! This repository demonstrates how to process and analyze Twitter data using machine learning, natural language processing (NLP), and visualization techniques in Python.

# Project Highlights
 .Preprocess raw tweets using regex, NLTK, and stemming

 . Clean and transform text using TF-IDF vectorization

 . Train a Multinomial Naive Bayes model

.  Visualize sentiment distribution and evaluate model performance

. Display accuracy, precision, recall, F1-score, and a confusion matrix

## Dataset
. The project uses two datasets:

. twitter_training.csv – training set

. twitter_validation.csv – validation set

. Each file contains:

. Tweet ID

. Entity (e.g., company or person mentioned)

. Sentiment label (Positive, Negative, Neutral, Irrelevant)

. Tweet text

## Technologies Used
. Tool/Library	Purpose
. pandas	Data handling & cleaning
. re	Regex-based text cleaning
. nltk	Stopword removal & stemming
. sklearn	Model training & evaluation
. matplotlib & seaborn	Data visualization

## How to Run
Install required libraries:

Copy
Edit
pip install pandas nltk scikit-learn matplotlib seaborn
Download NLTK data (automatically handled in script):

python
Copy
Edit
import nltk
nltk.download('stopwords')
nltk.download('punkt')
Place CSV files in the same directory as sentimentAnalysis.py.

Run the script:

Copy
Edit
python sentimentAnalysis.py
## Sample Output
Sentiment distribution bar charts

Cleaned tweet text samples

TF-IDF matrix shapes

Model accuracy, precision, recall, F1-score

Confusion matrix heatmap

## Results
. The Naive Bayes model provides a solid baseline for tweet sentiment classification. This project can be expanded with:

. Deep learning (LSTM, BERT)

. Emoji/emoticon sentiment

. Multilingual support

. Real-time sentiment tracking via Twitter API

## Screenshots

<img width="1920" height="1080" alt="Screenshot 2025-07-27 181029" src="https://github.com/user-attachments/assets/256b5e74-5d1a-4572-9962-4085d6f0e11c" />
<img width="1920" height="1080" alt="Screenshot 2025-07-27 181038" src="https://github.com/user-attachments/assets/b806b559-0844-4b76-8d27-704b7d11ef77" />
<img width="1920" height="1080" alt="Screenshot 2025-07-27 181047" src="https://github.com/user-attachments/assets/3cc5e2bf-3f6f-4e50-8fb8-74fef4c7c302" />

