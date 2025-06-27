import pandas as pd
import datetime
import tqdm
from sklearn.model_selection import train_test_split

# Read and preprocess data
def cold_processed(seq_length):
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
        "user_id": user_list,
        "timestamp": timestamp_list,
        "app_seq": all_app_seq,
        "time_seq": all_time_seq,
        "next_app": next_app_list
    })

    # Remove users with fewer than 50 records
    df_processed = df_processed[df_processed.groupby("user_id")["user_id"].transform("count").ge(50)]

    # Process time format
    df_processed['time'] = df_processed['timestamp'].apply(lambda x: f"{x.weekday()}_{x.hour}")

    # Get all users
    users = df_processed['user_id'].unique()

    # Split cold-start test users (90% train, 10% test)
    train_users, test_users = train_test_split(users, test_size=0.1, random_state=42)

    # Split train users into train (90%) and validation (10%) (optional)
    #train_users, valid_users = train_test_split(train_users, test_size=0.1, random_state=42)

    # Split data by user
    train_data = df_processed[df_processed['user_id'].isin(train_users)].copy()
    #valid_data = df_processed[df_processed['user_id'].isin(valid_users)].copy()
    test_data = df_processed[df_processed['user_id'].isin(test_users)].copy()  # Cold-start users

    # User ID mapping (only for training users)
    user2id = {u: i for i, u in enumerate(sorted(train_users))}
    max_user_id = len(user2id)  # Max user ID for cold start

    # Map user IDs
    train_data["user"] = train_data["user_id"].map(user2id)
    #valid_data["user"] = valid_data["user_id"].apply(lambda x: user2id.get(x, max_user_id))
    test_data["user"] = test_data["user_id"].apply(lambda x: user2id.get(x, max_user_id))

    # App ID mapping
    app_set = set()
    for s in df_processed['app_seq'].values:
        app_set.update(s)
    app2id = {a: i for i, a in enumerate(sorted(app_set))}

    # Map app sequences
    train_data["app_seq"] = train_data["app_seq"].apply(lambda x: [app2id[c] for c in x])
    #valid_data["app_seq"] = valid_data["app_seq"].apply(lambda x: [app2id.get(c, 0) for c in x])
    test_data["app_seq"] = test_data["app_seq"].apply(lambda x: [app2id.get(c, 0) for c in x])

    train_data["app"] = train_data["next_app"].map(app2id)
    #valid_data["app"] = valid_data["next_app"].apply(lambda x: app2id.get(x, 0))
    test_data["app"] = test_data["next_app"].apply(lambda x: app2id.get(x, 0))

    # Remove unnecessary columns
    train_data.drop(columns=['timestamp', 'user_id'], inplace=True)
    #valid_data.drop(columns=['timestamp', 'user_id'], inplace=True)
    test_data.drop(columns=['timestamp', 'user_id'], inplace=True)

    # Save processed data
    train_data.to_csv('./data/cold/train.txt', sep='\t', index=False)
    #valid_data.to_csv('data/cold/valid.txt', sep='\t', index=False)
    test_data.to_csv('./data/cold/test.txt', sep='\t', index=False)

    # Save user2id and app2id mappings
    def dict2file(dic, filename):
        with open(filename, 'w') as f:
            for k, v in dic.items():
                f.write("{}\t{}\n".format(k, v))

    dict2file(user2id, "./data/cold/user2id.txt")
    dict2file(app2id, "./data/cold/app2id.txt")
