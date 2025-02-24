import pandas as pd
import tqdm
import numpy as np
import json
from sklearn.model_selection import train_test_split

def cold_processed(seq_length, file_path="./data/lsapp.tsv"):
    """
    Process the LSApp dataset with a cold-start split.

    Parameters:
        seq_length (int): The length of the app usage sequence.
        file_path (str): Path to the dataset file.

    Saves:
        Processed train, validation, and test datasets, along with user and app mappings.
    """

    # Load data
    df_usage = pd.read_csv(file_path, sep="\t")

    # Keep only "Opened" events
    df_usage = df_usage[df_usage["event_type"] == "Opened"].copy()
    df_usage.drop(columns=["event_type"], inplace=True)

    # Convert timestamps to datetime format
    df_usage["timestamp"] = pd.to_datetime(df_usage["timestamp"])

    # Sort by user ID and timestamp
    df_usage = df_usage.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)

    # Initialize storage
    all_user_seq, all_app_seq, all_time_seq = [], [], []
    user_list, timestamp_list, next_app_list = [], [], []

    prev_user, prev_time = None, None

    for i in tqdm.tqdm(range(len(df_usage))):
        user = df_usage.iloc[i]['user_id']
        app = df_usage.iloc[i]['app_name']
        time = df_usage.iloc[i]['timestamp']
        
        if prev_user != user:
            app_seq, time_seq = [app], [time]
        else:
            if (time - prev_time).total_seconds() // 60 <= 5:
                if len(app_seq) == seq_length:
                    all_user_seq.append(user)
                    all_app_seq.append(app_seq.copy())
                    all_time_seq.append([(prev_time - t).total_seconds() // 60 for t in time_seq])
                    
                    user_list.append(user)
                    timestamp_list.append(time)
                    next_app_list.append(app)

                    app_seq = app_seq[1:] + [app]
                    time_seq = time_seq[1:] + [time]
                else:
                    app_seq.append(app)
                    time_seq.append(time)
            else:
                app_seq, time_seq = [app], [time]

        prev_user = user
        prev_time = time

    # Create DataFrame
    df_processed = pd.DataFrame({
        "user_id": user_list,
        "timestamp": timestamp_list,
        "app_seq": all_app_seq,
        "time_seq": all_time_seq,
        "next_app": next_app_list
    })

    # Remove users with fewer than 50 records
    df_processed = df_processed[df_processed.groupby("user_id")["user_id"].transform("count").ge(50)]

    # Convert timestamp to weekday-hour encoding
    df_processed['time'] = df_processed['timestamp'].apply(lambda x: f"{x.weekday()}_{x.hour}")

    # Get all users
    users = df_processed['user_id'].unique()

    # Split cold-start test users
    train_users, test_users = train_test_split(users, test_size=0.1, random_state=42)  # 90% train, 10% cold-start test users

    # Further split train users into train/validation (90%/10%)
    #train_users, valid_users = train_test_split(train_users, test_size=0.1, random_state=42)

    # Split data by users
    train_data = df_processed[df_processed['user_id'].isin(train_users)].copy()
    #valid_data = df_processed[df_processed['user_id'].isin(valid_users)].copy()
    test_data = df_processed[df_processed['user_id'].isin(test_users)].copy()  # Cold-start test users

    # Map user IDs (only for training users)
    user2id = {u: i for i, u in enumerate(sorted(train_users))}
    max_user_id = len(user2id)  # Maximum user ID for cold start

    # Process user IDs
    train_data["user"] = train_data["user_id"].map(user2id)
    #valid_data["user"] = valid_data["user_id"].apply(lambda x: user2id.get(x, max_user_id))  # Unknown users mapped to max ID
    test_data["user"] = test_data["user_id"].apply(lambda x: user2id.get(x, max_user_id))  # Cold-start users mapped to max ID

    # Map application IDs
    app_set = set()
    for s in df_processed['app_seq'].values:
        app_set.update(s)
    app2id = {a: i for i, a in enumerate(sorted(app_set))}

    # Process `app_seq`
    train_data["app_seq"] = train_data["app_seq"].apply(lambda x: [app2id[c] for c in x])
    #valid_data["app_seq"] = valid_data["app_seq"].apply(lambda x: [app2id.get(c, 0) for c in x])
    test_data["app_seq"] = test_data["app_seq"].apply(lambda x: [app2id.get(c, 0) for c in x])

    train_data["app"] = train_data["next_app"].map(app2id)
    #valid_data["app"] = valid_data["next_app"].apply(lambda x: app2id.get(x, 0))
    test_data["app"] = test_data["next_app"].apply(lambda x: app2id.get(x, 0))

    # Remove `timestamp` and `user_id`
    train_data.drop(columns=['timestamp', 'user_id'], inplace=True)
    #valid_data.drop(columns=['timestamp', 'user_id'], inplace=True)
    test_data.drop(columns=['timestamp', 'user_id'], inplace=True)

    # Save datasets
    train_data.to_csv('./data/cold/train.txt', sep='\t', index=False)
    #valid_data.to_csv('./data/cold/valid.txt', sep='\t', index=False)
    test_data.to_csv('./data/cold/test.txt', sep='\t', index=False)

    # Save user2id and app2id mappings
    def dict2file(dic, filename):
        with open(filename, 'w') as f:
            for k, v in dic.items():
                f.write("{}\t{}\n".format(k, v))

    dict2file(user2id, "./data/cold/user2id.txt")
    dict2file(app2id, "./data/cold/app2id.txt")
