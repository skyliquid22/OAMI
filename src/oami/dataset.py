import pandas as pd
from .features import FeatureBuilder
from .options_features import build_options_sentiment, compute_sentiment_scores


def build_unified_dataset(df_market: pd.DataFrame,
                          df_options: pd.DataFrame,
                          feature_cfg: dict,
                          target: str = 'next_return') -> pd.DataFrame:
    fb = FeatureBuilder(df_market)
    fb.add_indicators(
        sma_windows=tuple(feature_cfg.get('sma_windows',[10,20,50])),
        ema_windows=tuple(feature_cfg.get('ema_windows',[10,20,50])),
        rsi_window=feature_cfg.get('rsi_window',14),
        macd_fast=feature_cfg.get('macd_fast',12),
        macd_slow=feature_cfg.get('macd_slow',26),
        macd_signal=feature_cfg.get('macd_signal',9),
        bb_window=feature_cfg.get('bb_window',20),
        bb_std=feature_cfg.get('bb_std',2),
    ).add_lags(lags=tuple(feature_cfg.get('lags',[1,2,3,5]))).add_rolling(
        windows=tuple(feature_cfg.get('rolling_windows',[5,10,20]))
    )
    mkt = fb.finalize(dropna=False)

    # Target
    mkt['return'] = mkt['Close'].pct_change()
    mkt['next_return'] = mkt['return'].shift(-1)

    # Options sentiment
    opt = build_options_sentiment(
        df_options,
        near_term_days=feature_cfg.get('near_term_days', 7),
        far_term_days=feature_cfg.get('far_term_days', 30),
    )
    if not opt.empty:
        sentiment_scores = compute_sentiment_scores(opt)
        opt = opt.merge(sentiment_scores, on='Date', how='left')

    # Merge
    merged = pd.merge(mkt, opt.drop_duplicates(subset=['Date']), on='Date', how='left')

    # Clean
    merged = merged.dropna(subset=['next_return']).reset_index(drop=True)
    return merged
