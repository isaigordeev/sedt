import csv
from transformers import AutoTokenizer
import numpy as np
from nltk.stem import WordNetLemmatizer
import re
import nltk
import os
nltk.download('stopwords')
nltk.download('wordnet')
import matplotlib.pyplot as plt

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
        with_event_type (bool): Whether to include 'EventType' in the output.

    Returns:
        tuple: A tuple of tokenized tweets (NumPy array), IDs, and EventTypes (if included).
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tweets, ids, event_types = [], [], []

    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                tweet = row['Tweet']
                tweet_id = row['ID']
                period_id = int(row['PeriodID']) if with_period_id else None
                event_type = row['EventType'] if with_event_type else None

                if with_period_id:
                    tweet = f"{format_number_to_string(period_id)} {preprocess_text(tweet)}"
                else:
                    tweet = preprocess_text(tweet)

                tokens = tokenizer.encode(tweet, truncation=True, padding="max_length", max_length=l, add_special_tokens=True)
                tweets.append(tokens)
                ids.append(tweet_id)
                if with_event_type:
                    event_types.append(event_type)
            except KeyError as e:
                print(f"Missing expected column in row: {row}")
                continue

    tweets = np.array(tweets, dtype=np.int32)
    return tweets, np.array(ids), np.array(event_types) if with_event_type else None

# function of reading the csv file and return the processed data
def read_csv(folder_path, l, with_period_id, with_event_type):
    """
    Read all CSV files in a folder and process them.

    Args:
        folder_path (str): Path to the folder containing the CSV files.
        l (int): Desired length of token arrays for the 'Tweet' column.
        with_period_id (bool): Whether to include the 'PeriodID' in the tweet text.
        with_event_type (bool): Whether to include 'EventType' in the output.

    Returns:
        tuple: A tuple of tokenized tweets (NumPy array), IDs, and EventTypes (if included).
    """
    all_tweets, all_ids, all_event_types = [], [], []

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            tweets, ids, event_types = process_csv(file_path, l, with_period_id, with_event_type)
            all_tweets.append(tweets)
            all_ids.append(ids)
            if with_event_type:
                all_event_types.append(event_types)

    all_tweets = np.concatenate(all_tweets, axis=0)
    all_ids = np.concatenate(all_ids, axis=0)
    all_event_types = np.concatenate(all_event_types, axis=0) if with_event_type else None
    return all_tweets, all_ids, all_event_types

# function to count number of PAD token, tokenizer defined in the function
def count_pad_tokens(tokens, pad_token_id): 
    return tokens.count(pad_token_id)