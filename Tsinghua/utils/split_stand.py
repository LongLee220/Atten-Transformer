import pandas as pd
import datetime
import tqdm

# Read and preprocess data
def stand_processed(seq_length):
    df_usage = pd.read_csv('./data/App_usage_trace.txt', sep=' ', names=['user', 'time', 'location', 'app', 'traffic'])
    df_usage = df_usage[['user', 'time', 'app', 'traffic']]

    # Merge consecutive usage records of the same app within one minute
    df_usage['time'] = df_usage['time'].apply(lambda x: str(x)[:-2])
    df_usage.drop_duplicates(inplace=True)

    # Remove apps used fewer than 5 times across all users
    df_usage = df_usage[df_usage.groupby('app')['app'].transform('count').ge(5)]

    # Normalize traffic
    max_traffic = df_usage['traffic'].max()
    df_usage['traffic'] /= max_traffic

    # Convert time format
    df_usage['timestamp'] = df_usage['time'].apply(lambda x: datetime.datetime.strptime(x, '%Y%m%d%H%M'))

    # Sort by user and timestamp
    df_usage = df_usage.sort_values(by=['user', 'timestamp']).reset_index(drop=True)

    # Sliding window to extract sequences
    all_user_seq, all_app_seq, all_time_seq, all_traffic_seq = [], [], [], []
    user_list, timestamp_list, next_app_list = [], [], []

    prev_user, prev_time = None, None
    app_seq, time_seq, traffic_seq = [], [], []

    for i in tqdm.tqdm(range(len(df_usage))):
        user = df_usage.iloc[i]['user']
        app = df_usage.iloc[i]['app']
        time = df_usage.iloc[i]['timestamp']
        traffic = df_usage.iloc[i]['traffic']
        
        if prev_user != user:
            app_seq, time_seq, traffic_seq = [app], [time], [traffic]
        else:
            # Limit time gap to ≤ 5 minutes
            if (time - prev_time).total_seconds() // 60 <= 5:
                if len(app_seq) == seq_length:
                    all_user_seq.append(user)
                    all_app_seq.append(app_seq.copy())
                    all_time_seq.append([(prev_time - t).total_seconds() // 60 for t in time_seq])
                    all_traffic_seq.append(traffic_seq.copy())
                    
                    user_list.append(user)
                    timestamp_list.append(time)
                    next_app_list.append(app)

                    # Sliding window
                    app_seq = app_seq[1:] + [app]
                    time_seq = time_seq[1:] + [time]
                    traffic_seq = traffic_seq[1:] + [traffic]
                else:
                    app_seq.append(app)
                    time_seq.append(time)
                    traffic_seq.append(traffic)
            else:
                app_seq, time_seq, traffic_seq = [app], [time], [traffic]

        prev_user, prev_time = user, time

    # Create DataFrame
    df_processed = pd.DataFrame({
        "user": user_list,
        "timestamp": timestamp_list,
        "app_seq": all_app_seq,
        "time_seq": all_time_seq,
        "traffic_seq": all_traffic_seq,
        "app": next_app_list
    })

    # Remove users with fewer than 50 records
    df_processed = df_processed[df_processed.groupby("user")["user"].transform("count").ge(50)]

    # Encode time
    df_processed['time'] = df_processed['timestamp'].apply(lambda x: f"{x.weekday()}_{x.hour}")

    # Encode users & apps
    user2id = {u: i for i, u in enumerate(sorted(df_processed['user'].unique()))}
    app_set = set()
    for s in df_processed['app_seq'].values:
        app_set.update(s)
    app2id = {a: i for i, a in enumerate(sorted(app_set))}

    # Save user2id and app2id mappings
    def dict2file(dic, filename):
        with open(filename, 'w') as f:
            for k, v in dic.items():
                f.write("{}\t{}\n".format(k, v))

    dict2file(user2id, "./data/stand/user2id.txt")
    dict2file(app2id, "./data/stand/app2id.txt")

    # Map user and app sequences
    df_processed['user'] = df_processed['user'].apply(lambda x: user2id[x])
    df_processed['app_seq'] = df_processed['app_seq'].apply(lambda x: [app2id[c] for c in x])
    df_processed['app'] = df_processed['app'].apply(lambda x: app2id.get(x, 0))  # Handle missing values

    # Split data by time sequence for each user
    train_list, valid_list, test_list = [], [], []

    for user, user_data in df_processed.groupby("user"):
        user_data = user_data.sort_values(by="timestamp")  # Sort by timestamp
        
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

    # Drop timestamp column
    train.drop(columns=['timestamp'], inplace=True)
    validation.drop(columns=['timestamp'], inplace=True)
    test.drop(columns=['timestamp'], inplace=True)

    # Save processed data
    train.to_csv('./data/stand/train.txt', sep='\t', index=False)
    validation.to_csv('./data/stand/validation.txt', sep='\t', index=False)
    test.to_csv('./data/stand/test.txt', sep='\t', index=False)
