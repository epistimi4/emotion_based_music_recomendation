def unify_classes(df):
    # The parent mood unhappy contains the moods angry,sad and nervous
    df['mood'] = df['mood'].replace({'angry': 'unhappy', 'sad': 'unhappy', 'nervous': 'unhappy'})
    # The parent mood annoyed contains the moods bored and sleepy
    df['mood'] = df['mood'].replace({'bored': 'annoyed', 'sleepy': 'annoyed'})
    # The parent mood relaxed contains the moods calm,relaxed and peaceful
    df['mood'] = df['mood'].replace({'calm': 'relaxed', 'relaxed': 'relaxed', 'peaceful': 'relaxed'})
    # The parent mood happy contains the moods excited,happy and pleased
    df['mood'] = df['mood'].replace({'excited': 'happy', 'happy': 'happy', 'pleased': 'happy'})

    # The parent period midday contains the periods evening and afternoon
    df['period'] = df['period'].replace({'evening': 'midday', 'afternoon': 'midday'})

    df['activity'] = df['activity'].replace({'commuting': 'other'})
    return df