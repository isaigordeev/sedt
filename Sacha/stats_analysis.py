import os
import pandas as pd
import matplotlib.pyplot as plt

def read_data(directory):
    """
    Reads all CSV files in a specified directory and stores them in a dictionary with
    filenames as keys and DataFrames as values.
    """
    data = {}
    try:
        files = os.listdir(directory)
        print(f"Files in directory: {files}")
    except FileNotFoundError:
        print("Directory not found. Check the directory path.")
        return data

    for filename in files:
        if filename.endswith(".csv"):
            try:
                file_path = os.path.join(directory, filename)
                df = pd.read_csv(file_path, encoding='utf-8')
                data[filename] = df
                print(f"Loaded data from {filename}.")
            except Exception as e:
                print(f"Could not read file {filename}: {e}")
    
    return data

def calculate_statistics(data):
    """
    Calculates the mean and variance of EventType, correlation between EventType and PeriodID,
    and maximum PeriodID for each file.
    """
    stats = {
        'mean': [],
        'variance': [],
        'correlation': [],
        'max_periodid': [],
        'filename': []
    }

    all_eventtype_values = []
    all_periodid_values = []
    
    for filename, df in data.items():
        if 'EventType' in df.columns and 'PeriodID' in df.columns:
            # Mean and Variance of EventType
            mean_eventtype = df['EventType'].mean()
            variance_eventtype = df['EventType'].var()

            # Correlation between EventType and PeriodID
            correlation = df[['EventType', 'PeriodID']].corr().iloc[0, 1]

            # Max of PeriodID
            max_periodid = df['PeriodID'].max()

            # Append to stats
            stats['mean'].append(mean_eventtype)
            stats['variance'].append(variance_eventtype)
            stats['correlation'].append(correlation)
            stats['max_periodid'].append(max_periodid)
            stats['filename'].append(filename)

            # Collect all values for overall statistics
            all_eventtype_values.extend(df['EventType'])
            all_periodid_values.extend(df['PeriodID'])
    
    # Calculate overall mean, variance, and correlation
    overall_mean_eventtype = pd.Series(all_eventtype_values).mean()
    overall_variance_eventtype = pd.Series(all_eventtype_values).var()
    overall_correlation = pd.Series(all_eventtype_values).corr(pd.Series(all_periodid_values))

    return pd.DataFrame(stats), overall_mean_eventtype, overall_variance_eventtype, overall_correlation

def plot_statistics(statistics_df, overall_mean_eventtype, overall_variance_eventtype, overall_correlation):
    """
    Creates three plots: 
    1. Mean and variance of EventType for each file, with overall mean and variance as horizontal lines.
    2. Correlation of EventType and PeriodID for each file, with overall correlation as a horizontal line.
    3. Maximum PeriodID for each file.
    """
    # Plot 1: Mean and Variance of EventType
    fig, ax1 = plt.subplots(figsize=(10, 6))
    statistics_df.plot(x='filename', y='mean', kind='bar', ax=ax1, color='skyblue', position=0, width=0.4, label='Mean')
    statistics_df.plot(x='filename', y='variance', kind='bar', ax=ax1, color='salmon', position=1, width=0.4, label='Variance')
    ax1.axhline(overall_mean_eventtype, color='blue', linestyle='--', linewidth=1, label='Overall Mean')
    ax1.axhline(overall_variance_eventtype, color='red', linestyle='--', linewidth=1, label='Overall Variance')
    ax1.set_title("Mean and Variance of EventType for Each File")
    ax1.set_xlabel("Files")
    ax1.set_ylabel("Value")
    ax1.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot 2: Correlation of EventType and PeriodID
    fig, ax2 = plt.subplots(figsize=(10, 6))
    statistics_df.plot(x='filename', y='correlation', kind='bar', ax=ax2, color='green')
    ax2.axhline(overall_correlation, color='darkgreen', linestyle='--', linewidth=1, label='Overall Correlation')
    ax2.set_title("Correlation of EventType and PeriodID for Each File")
    ax2.set_xlabel("Files")
    ax2.set_ylabel("Correlation")
    ax2.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot 3: Maximum PeriodID for each file
    fig, ax3 = plt.subplots(figsize=(10, 6))
    statistics_df.plot(x='filename', y='max_periodid', kind='bar', ax=ax3, color='orange')
    ax3.set_title("Maximum PeriodID for Each File")
    ax3.set_xlabel("Files")
    ax3.set_ylabel("Max PeriodID")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def calculate_binned_mean_eventtype(data, bin_size=10):
    """
    Calculates the average EventType for binned PeriodID values, aggregated across all files.
    Returns a DataFrame containing binned PeriodID and the mean EventType.
    """
    combined = pd.DataFrame()  # To combine data from all files

    for filename, df in data.items():
        if 'EventType' in df.columns and 'PeriodID' in df.columns:
            # Bin PeriodID into groups of bin_size
            df['PeriodID_bin'] = (df['PeriodID'] // bin_size) * bin_size
            # Group by PeriodID_bin and calculate the mean of EventType
            binned_mean = df.groupby('PeriodID_bin')['EventType'].mean().reset_index()
            binned_mean.rename(columns={'EventType': f'EventType_{filename}'}, inplace=True)
            
            if combined.empty:
                combined = binned_mean
            else:
                # Merge data by PeriodID_bin
                combined = pd.merge(combined, binned_mean, on='PeriodID_bin', how='outer')

    # Calculate the average of all EventType_* columns
    combined['MeanEventType'] = combined.iloc[:, 1:].mean(axis=1)  # Average across all EventType_* columns
    return combined[['PeriodID_bin', 'MeanEventType']]

def plot_binned_mean_eventtype(mean_distribution, bin_size):
    """
    Plots the averaged distribution of EventType against binned PeriodID.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(mean_distribution['PeriodID_bin'], mean_distribution['MeanEventType'], marker='o', linestyle='-')
    plt.title(f"Averaged Distribution of Mean EventType by PeriodID (Bin Size = {bin_size})")
    plt.xlabel(f"PeriodID (binned by {bin_size})")
    plt.ylabel("Mean EventType")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Specify the directory
directory = 'challenge_data/train_tweets'

# Read data from CSV files
data = read_data(directory)

# Calculate statistics and overall metrics
statistics_df, overall_mean_eventtype, overall_variance_eventtype, overall_correlation = calculate_statistics(data)

# Plot statistics with overall lines
plot_statistics(statistics_df, overall_mean_eventtype, overall_variance_eventtype, overall_correlation)

# Set bin size
bin_size = 10

# Calculate the binned averaged distribution
mean_distribution = calculate_binned_mean_eventtype(data, bin_size=bin_size)

# Plot the binned distribution
plot_binned_mean_eventtype(mean_distribution, bin_size)