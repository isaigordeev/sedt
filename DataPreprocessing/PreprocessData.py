from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
import re
import nltk
import os
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """
    Preprocess text by lowercasing, removing punctuation, numbers, stopwords, and lemmatizing.

    Args:
        text (str): Text to be preprocessed.

    Returns:
        str: Preprocessed text.
    """
    # Lowercasing
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenization
    words = text.split()

    # Remove stopwords
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def format_number_to_string(number):
    """
    Formats a number into a 3-character string, adding leading zeros if necessary.

    Args:
        number (int): A number between 0 and 999.

    Returns:
        str: A string of length 3.
    """
    if not (0 <= number <= 999):
        raise ValueError("The number must be in the range 0 to 999.")

    return f"{number:03d}"

def process_csv(file_path, l, with_period_id, with_event_type):
    """
    Process a CSV file to extract and tokenize data.

    Args:
        file_path (str): Path to the CSV file.
        l (int): Desired length of token arrays for the 'Tweet' column.
        with_period_id (bool): Whether to include the 'PeriodID' in the tweet text.

    Returns:
        pd.DataFrame: Processed DataFrame with columns 'PeriodID', 'EventType', and 'Tweet'.
    """
    # Load the tokenizer (default tokenizer from transformers)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Extract required columns
    if with_event_type:
        df = df[['PeriodID', 'EventType', 'Tweet']]
    else:
        df = df[['PeriodID', 'Tweet']]

    if with_period_id:
      # Preprocess text and concatenate with formatted PeriodID
        df['Tweet'] = df.apply(
            lambda row: f"{format_number_to_string(row['PeriodID'])} {preprocess_text(row['Tweet'])}",
            axis=1
        )
    else:
      # Apply preprocessing to each tweet
      df['Tweet'] = df['Tweet'].apply(preprocess_text)

    # Tokenize the 'Tweet' column and pad/truncate to length l
    def tokenize_tweet(tweet):
        tokens = tokenizer.encode(tweet, truncation=True, padding="max_length", max_length=l, add_special_tokens=True)
        return tokens

    df['Tweet'] = df['Tweet'].apply(tokenize_tweet)

    if with_event_type:
        df = df[['EventType', 'Tweet']]
    else:
        df = df[['Tweet']]

    return df

# function of reading the csv file and return the processed data
def read_csv(folder_path, l, with_period_id, with_event_type):
    """
    Read all CSV files in a folder and process them.

    Args:
        folder_path (str): Path to the folder containing the CSV files.
        l (int): Desired length of token arrays for the 'Tweet' column.
        with_period_id (bool): Whether to include the 'PeriodID' in the tweet text.

    Returns:
        pd.DataFrame: Processed DataFrame with columns 'PeriodID', 'EventType' (optional), and 'Tweet' (tokenized).
    """
    li = []
    for filename in os.listdir(folder_path):
        df = process_csv(folder_path + filename, l, with_period_id, with_event_type)
        li.append(df)
    df = pd.concat(li, ignore_index=True)
    return df