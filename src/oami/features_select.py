import pandas as pd
from sklearn.decomposition import PCA


class FeatureSelector:
    def __init__(self, df: pd.DataFrame): self.df = df.copy()

    def apply_pca(self, n_components: int = 5) -> pd.DataFrame:
        numeric_df = self.df.select_dtypes('number').dropna()
        if numeric_df.shape[1] == 0:
            return self.df
        comps = PCA(n_components=min(n_components, max(1, numeric_df.shape[1]))).fit_transform(numeric_df)
        import pandas as pd
        comp_df = pd.DataFrame(comps, columns=[f'pca_{i+1}' for i in range(comps.shape[1])])
        return pd.concat([self.df.reset_index(drop=True), comp_df], axis=1)
