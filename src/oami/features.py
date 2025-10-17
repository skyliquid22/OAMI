from __future__ import annotations
import pandas as pd, numpy as np, ta
class FeatureBuilder:
    def __init__(self, df: pd.DataFrame): self.df = df.copy().sort_values('Date')
    def add_indicators(self, sma_windows=(10,20,50), ema_windows=(10,20,50), rsi_window=14, macd_fast=12, macd_slow=26, macd_signal=9, bb_window=20, bb_std=2):
        c, h, l = self.df['Close'], self.df['High'], self.df['Low']
        for w in sma_windows: self.df[f'sma_{w}'] = c.rolling(w).mean()
        for w in ema_windows: self.df[f'ema_{w}'] = c.ewm(span=w, adjust=False).mean()
        self.df['rsi'] = ta.momentum.RSIIndicator(c, window=rsi_window).rsi()
        macd = ta.trend.MACD(c, window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal)
        self.df['macd'], self.df['macd_signal'], self.df['macd_hist'] = macd.macd(), macd.macd_signal(), macd.macd_diff()
        bb = ta.volatility.BollingerBands(c, window=bb_window, window_dev=bb_std); self.df['bb_width'] = bb.bollinger_wband()
        self.df['atr'] = ta.volatility.AverageTrueRange(h, l, c).average_true_range(); return self
    def add_lags(self, cols=None, lags=(1,2,3,5)):
        if cols is None: cols = [c for c in self.df.columns if c not in ['Date','Open','High','Low','Close','Volume']]
        for col in cols:
            for L in lags: self.df[f'{col}_lag{L}'] = self.df[col].shift(L)
        return self
    def add_rolling(self, cols=None, windows=(5,10,20)):
        if cols is None: cols = ['Close','Volume']
        for col in cols:
            for w in windows:
                self.df[f'{col}_rollmean_{w}'] = self.df[col].rolling(w).mean()
                self.df[f'{col}_rollstd_{w}'] = self.df[col].rolling(w).std()
        return self
    def finalize(self, dropna=True): return self.df.dropna() if dropna else self.df
