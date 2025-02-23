import pandas as pd
import tqdm

def stand_processed(seq_length, file_path="./data/lsapp.tsv"):
    """
    Process the LSApp dataset with a standard split.

    Parameters:
        seq_length (int): The length of the app usage sequence.
        file_path (str): Path to the dataset file.

    Saves:
        Processed train, validation, and test datasets, along with user and app mappings.
    """
    
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
            app_seq, time_seq, user_seq = [app], [time], [user]
        else:
            if (time - prev_time).total_seconds() // 60 <= 5:
                if len(app_seq) == seq_length:
                    all_user_seq.append(user_seq.copy())
                    all_app_seq.append(app_seq.copy())
                    all_time_seq.append([(prev_time - t).total_seconds() // 60 for t in time_seq])

                    user_list.append(user)
                    timestamp_list.append(time)
                    next_app_list.append(app)

                    # Sliding window update
                    app_seq = app_seq[1:] + [app]
                    time_seq = time_seq[1:] + [time]
                    user_seq = user_seq[1:] + [user]
                else:
                    app_seq.append(app)
                    time_seq.append(time)
                    user_seq.append(user)
            else:
                app_seq, time_seq, user_seq = [app], [time], [user]

        prev_user, prev_time = user, time

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

    # Encode users and apps
    user2id = {u: i for i, u in enumerate(sorted(df_processed['user_id'].unique()))}
    app_set = set()
    for s in df_processed['app_seq'].values:
        app_set.update(s)
    app2id = {a: i for i, a in enumerate(sorted(app_set))}

    # Save mappings
    def dict2file(dic, filename):
        with open(filename, 'w') as f:
            for k, v in dic.items():
                f.write("{}\t{}\n".format(k, v))

    dict2file(user2id, "./data/stand/user2id.txt")
    dict2file(app2id, "./data/stand/app2id.txt")

    # Map users and apps to IDs
    df_processed['user'] = df_processed['user_id'].apply(lambda x: user2id[x])
    df_processed['app_seq'] = df_processed['app_seq'].apply(lambda x: [app2id[c] for c in x])
    df_processed['app'] = df_processed['next_app'].apply(lambda x: app2id[x])

    # Create final dataset
    df_dataset = df_processed[['user', 'timestamp', 'time', 'app_seq', 'time_seq', 'app']]

    # Remove users with fewer than 2 records
    df_dataset = df_dataset[df_dataset.groupby("user")["user"].transform("count") >= 2]

    # Data split
    train_list, valid_list, test_list = [], [], []

    for user, user_data in df_dataset.groupby("user"):
        user_data = user_data.sort_values(by="timestamp")

        n = len(user_data)
        train_end = int(n * 0.7)
        valid_end = int(n * 0.8)

        train_list.append(user_data.iloc[:train_end])
        valid_list.append(user_data.iloc[train_end:valid_end])
        test_list.append(user_data.iloc[valid_end:])

    # Merge data
    train = pd.concat(train_list)
    validation = pd.concat(valid_list)
    test = pd.concat(test_list)

    # Remove timestamp if not needed
    train.drop(columns=['timestamp'], inplace=True)
    validation.drop(columns=['timestamp'], inplace=True)
    test.drop(columns=['timestamp'], inplace=True)

    # Save final datasets
    train.to_csv('./data/stand/train.txt', sep='\t', index=False)
    validation.to_csv('./stand/data/validation.txt', sep='\t', index=False)
    test.to_csv('./data/stand/test.txt', sep='\t', index=False)
