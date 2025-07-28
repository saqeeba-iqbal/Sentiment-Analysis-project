import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure you have the necessary NLTK data downloaded
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt') # Correct path for punkt
except LookupError:
    nltk.download('punkt')
print("Starting Tweet Sentiment Extraction Project...")

# --- 1. Data Loading ---
print("\n--- 1. Data Loading ---")
# Load the datasets. Assuming the columns are ID, Information, Sentiment, TweetText
# We'll re-name them for clarity.
try:
    # Training Data
    train_df = pd.read_csv('twitter_training.csv', header=None, names=['TweetID', 'Entity', 'Sentiment', 'TweetText'])
    print("twitter_training.csv loaded successfully.")
    print("Shape of training data:", train_df.shape)
    print("First 5 rows of training data:")
    print(train_df.head())
    print("\nMissing values in training data:")
    print(train_df.isnull().sum())

    # Validation Data
    val_df = pd.read_csv('twitter_validation.csv', header=None, names=['TweetID', 'Entity', 'Sentiment', 'TweetText'])
    print("\ntwitter_validation.csv loaded successfully.")
    print("Shape of validation data:", val_df.shape)
    print("First 5 rows of validation data:")
    print(val_df.head())
    print("\nMissing values in validation data:")
    print(val_df.isnull().sum())

except FileNotFoundError:
    print("Error: Make sure 'twitter_training.csv' and 'twitter_validation.csv' are in the same directory as this script.")
    exit() # Exit if files are not found

# Drop rows with missing 'TweetText' as they are crucial for sentiment analysis
train_df.dropna(subset=['TweetText'], inplace=True)
val_df.dropna(subset=['TweetText'], inplace=True)
print("\nMissing values after dropping NaN rows (Training):")
print(train_df.isnull().sum())
print("Missing values after dropping NaN rows (Validation):")
print(val_df.isnull().sum())


# --- 2. Data Preprocessing ---
print("\n--- 2. Data Preprocessing ---")
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return "" # Handle non-string inputs, though dropna should mostly prevent this

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize and remove stopwords, then stem
    tokens = text.split()
    cleaned_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned_tokens)

print("Applying preprocessing to training and validation data...")
train_df['CleanedTweetText'] = train_df['TweetText'].apply(preprocess_text)
val_df['CleanedTweetText'] = val_df['TweetText'].apply(preprocess_text)

print("\nFirst 5 rows of training data with cleaned text:")
print(train_df[['TweetText', 'CleanedTweetText', 'Sentiment']].head())
print("\nFirst 5 rows of validation data with cleaned text:")
print(val_df[['TweetText', 'CleanedTweetText', 'Sentiment']].head())


# --- 3. Exploratory Data Analysis (EDA) ---
print("\n--- 3. Exploratory Data Analysis (EDA) ---")

# Sentiment Distribution in Training Data
plt.figure(figsize=(8, 6))
sns.countplot(x='Sentiment', data=train_df, palette='viridis')
plt.title('Sentiment Distribution in Training Data')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.show()

# Sentiment Distribution in Validation Data
plt.figure(figsize=(8, 6))
sns.countplot(x='Sentiment', data=val_df, palette='viridis')
plt.title('Sentiment Distribution in Validation Data')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.show()

# You could add more EDA like word clouds for each sentiment, but this is a good start.


# --- 4. Model Training ---
print("\n--- 4. Model Training ---")

# Feature Extraction (TF-IDF)
print("Converting text data into numerical features using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000) # Limit features to top 5000 to prevent overfitting and manage memory

X_train = vectorizer.fit_transform(train_df['CleanedTweetText'])
y_train = train_df['Sentiment']

X_val = vectorizer.transform(val_df['CleanedTweetText']) # Use transform, not fit_transform for validation set
y_val = val_df['Sentiment']

print("Shape of TF-IDF matrix for training data:", X_train.shape)
print("Shape of TF-IDF matrix for validation data:", X_val.shape)

# Train a Classifier (Multinomial Naive Bayes is a good baseline for text classification)
print("Training Multinomial Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train, y_train)
print("Model training complete.")


# --- 5. Model Evaluation ---
print("\n--- 5. Model Evaluation ---")

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Confusion Matrix
print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(y_val, y_pred, labels=model.classes_) # Ensure labels are in consistent order
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Sentiment')
plt.ylabel('True Sentiment')
plt.show()

print("\nProject execution complete. Check the generated plots for sentiment distribution and confusion matrix.")