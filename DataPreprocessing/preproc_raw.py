import gc

import pandas as pd

def process_csv_for_eval_boosting(file_path, l, with_period_id, with_event_type):
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

    df['Tweet'] = df['Tweet'].apply(preprocess_text)

    # Tokenize the 'Tweet' column and pad/truncate to length l
    def tokenize_tweet(tweet):
        tokens = tokenizer.encode(tweet, truncation=True, padding="max_length", max_length=l, add_special_tokens=False)
        return tokens

    df['Tweet'] = df['Tweet'].apply(tokenize_tweet)

    df = df.groupby(['PeriodID'])['Tweet'].apply(list).reset_index()

    return df


def embed(x):
    tens = torch.tensor(x).to('cuda')
    embeddings = embeddings_layer(tens)

    del tens
    embeddings = embeddings.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()

    return embeddings.numpy()

def average_token_embed(x):
    tens = torch.tensor(x).to('cuda')
    embeddings = embeddings_layer(tens)

    del tens
    embeddings = embeddings.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()

    return embeddings.numpy().mean(axis=1)

def average_tweet_token_embed(x):
    tens = torch.tensor(x).to('cuda')
    embeddings = embeddings_layer(tens)

    del tens
    embeddings = embeddings.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()

    return embeddings.numpy().mean(axis=1).mean(axis=0)