import pandas as pd, numpy as np
def build_options_sentiment(df_options: pd.DataFrame, rolling_windows=(5,10,20)) -> pd.DataFrame:
    df = df_options.copy().sort_values('Date')
    for col in ['PutVol','CallVol','PutOI','CallOI','PutCallVolRatio','SentimentIndex']:
        if col not in df.columns: df[col] = np.nan
    df['put_call_oi_ratio'] = df['PutOI'] / df['CallOI'].replace({0: np.nan})
    df['total_option_vol'] = df['PutVol'].fillna(0) + df['CallVol'].fillna(0)
    for w in rolling_windows:
        df[f'sentiment_rollmean_{w}'] = df['SentimentIndex'].rolling(w).mean()
        df[f'sentiment_rollstd_{w}'] = df['SentimentIndex'].rolling(w).std()
        df[f'pcvr_rollmean_{w}'] = df['PutCallVolRatio'].rolling(w).mean()
        df[f'pcvr_rollstd_{w}'] = df['PutCallVolRatio'].rolling(w).std()
    return df
