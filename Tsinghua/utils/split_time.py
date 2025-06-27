import pandas as pd
import datetime
import tqdm

# Process time-based split
def time_processed(seq_length):
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

    prev_user = -1
    prev_time = -1
    app_seq, time_seq, traffic_seq = [], [], []
    all_app_seq, all_time_seq, all_traffic_seq = [], [], []

    # Convert time format and process sequences
    for i in tqdm.tqdm(range(len(df_usage))):
        user = df_usage.iloc[i]['user']
        app = df_usage.iloc[i]['app']
        time = df_usage.iloc[i]['time']
        time = datetime.datetime.strptime(time, '%Y%m%d%H%M')
        traffic = df_usage.iloc[i]['traffic']
        
        if prev_user != user:
            app_seq, time_seq, traffic_seq = [app], [time], [traffic]
            all_app_seq.append([])
            all_time_seq.append([])
            all_traffic_seq.append([])
        else:
            # Ensure time gap is ≤ 5 minutes
            if (time - prev_time).total_seconds() // 60 <= 5:
                if len(app_seq) == seq_length:
                    all_app_seq.append(app_seq)
                    all_time_seq.append([(prev_time - x).total_seconds() // 60 for x in time_seq])
                    all_traffic_seq.append(traffic_seq)

                    app_seq = app_seq[1:] + [app]
                    time_seq = time_seq[1:] + [time]
                    traffic_seq = traffic_seq[1:] + [traffic]
                else:
                    app_seq.append(app)
                    time_seq.append(time)
                    traffic_seq.append(traffic)
                    all_app_seq.append([])
                    all_time_seq.append([])
                    all_traffic_seq.append([])
            else:
                app_seq, time_seq, traffic_seq = [app], [time], [traffic]
                all_app_seq.append([])
                all_time_seq.append([])
                all_traffic_seq.append([])
        
        prev_user = user
        prev_time = time

    df_usage['app_seq'] = all_app_seq
    df_usage['time_seq'] = all_time_seq
    df_usage['traffic_seq'] = all_traffic_seq

    # Filter out empty sequences
    valid_indices = [i for i, seq in enumerate(all_app_seq) if len(seq) != 0]
    all_app_seq = [all_app_seq[i] for i in valid_indices]
    all_time_seq = [all_time_seq[i] for i in valid_indices]
    all_traffic_seq = [all_traffic_seq[i] for i in valid_indices]

    df_usage = df_usage[df_usage['app_seq'].map(len) != 0]

    # Remove users with fewer than 50 sequences
    df_usage = df_usage[df_usage.groupby('user')['user'].transform('count').ge(50)]

    # Encode time as weekday and hour
    def prep_time(t):
        t = t[:-2]
        weekday = datetime.datetime.strptime(t[:-2], '%Y%m%d').weekday()
        return '{}_{}'.format(weekday, t[-2:])

    df_usage['time'] = df_usage['time'].apply(lambda x: prep_time(x))

    # Encode user and app IDs
    user2id = {u: i for i, u in enumerate(sorted(df_usage['user'].unique()))}
    app_set = set()
    for s in df_usage['app_seq'].values:
        app_set.update(s)
    app2id = {a: i for i, a in enumerate(sorted(app_set))}

    # Save user and app mappings
    def dict2file(dic, filename):
        with open(filename, 'w') as f:
            for k, v in dic.items():
                f.write("{}\t{}\n".format(k, v))

    dict2file(user2id, "./data/time/user2id.txt")
    dict2file(app2id, "./data/time/app2id.txt")

    # Create dataset
    df_dataset = pd.DataFrame()
    df_dataset['user'] = df_usage['user'].apply(lambda x: user2id[x])
    df_dataset['time'] = df_usage['time']
    df_dataset['app_seq'] = df_usage['app_seq'].apply(lambda x: [app2id[c] for c in x])
    df_dataset['time_seq'] = df_usage['time_seq']
    df_dataset['app'] = df_usage['app'].apply(lambda x: app2id.get(x, 0))  # Handle missing values
    df_dataset['traffic_seq'] = df_usage['traffic_seq']

    # Convert time into actual dates
    def get_week_and_hour(t):
        parts = t.split('_')
        return int(parts[0]), int(parts[1])

    start_date = datetime.datetime(2016, 4, 20)  # Define a reference start date

    df_dataset['date'] = df_dataset['time'].apply(
        lambda t: start_date + datetime.timedelta(days=get_week_and_hour(t)[0], hours=get_week_and_hour(t)[1])
    )

    train_data = []
    test_data = []

    # Split data based on date
    for user, group in df_dataset.groupby('user'):
        dates = sorted(group['date'].unique())
        if len(dates) >= 7:
            train_dates = dates[:-1]
            test_date = dates[-1]
            
            train_data.append(group[group['date'].isin(train_dates)])
            test_data.append(group[group['date'] == test_date])

    train_df = pd.concat(train_data)
    test_df = pd.concat(test_data)

    # Save training and test data
    train_df.to_csv('./data/time/train.txt', sep='\t', index=False)
    test_df.to_csv('./data/time/test.txt', sep='\t', index=False)
