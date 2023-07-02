import datetime
import random
from enum import Enum
from typing import List, Tuple, Union, Dict, Optional
import numpy as np
import pandas as pd
import sys
from sklearn.cluster import KMeans, DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.decomposition import PCA
from hurst import compute_Hc
from statsmodels.tsa.stattools import coint
array_1d = Union[List, Tuple, pd.Series, np.ndarray]
date_obj = Union[datetime.datetime, datetime.date]
from sif.siftools import abstractalpha
from sif.siftools import operators as op
    
class Cluster(Enum):
    NONE = 0
    KMEANS = 1
    DBSCAN = 2
    OPTICS = 3
    AGGLOMERATIVE = 4

class ParentPairsTrader(abstractalpha.AbstractAlpha):
    """
    Includes all standard functionality that a more specific Pairs Trading 
    Algorithm would need.
    """
    def __init__(self, reset: int, 
                 npairs: Optional[int] = sys.maxsize, 
                 enter: Optional[int] = None,
                 exit: Optional[int] = None, 
                 stop: Optional[int] = None, 
                 hurst: Optional[bool] = False, 
                 use_random: Optional[bool] = False,
                 full_universe: Optional[bool] = True,
                 cluster: Optional[Cluster] = Cluster.NONE, 
                 hurst_sort: Optional[bool] = False):
        # Required variables for AbstractAlpha
        self.name = None
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        # Variables for Pairs Trading Alpha
        self.pairs = None
        self.reset = reset
        self.npairs = npairs
        self.enter = enter
        self.exit = exit
        self.stop = stop
        self.use_hurst = hurst
        self.use_random = use_random
        self.cluster = cluster
        self.hurst_sort = hurst_sort
        self.full_universe = full_universe
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
    
    def reduce_dimensionality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Implements principal component analysis.
        :param df: close price data
        :return: 90% of the preserved information
        """
        # Obtain returns, drop NaN values, and transpose.
        rets = df.pct_change().dropna()
        rets = rets.T
        # Use Sklearn implementation.
        pca = PCA(n_components=0.9, svd_solver="full")
        transformed_data = pca.fit_transform(rets)
        return transformed_data
    
    ### CHOICE OF CLUSTERING ALGORITHMS ###
    

    def cluster_kmeans(self, transformed_data: pd.DataFrame, 
                       nclusters: int) -> List[int]:
        """
        Implements kmeans clustering.
        :param transformed_data: output of principal component analysis
        :param nclusters: number of clusters
        :return: labels
        """
        kmeans = KMeans(n_clusters=nclusters)
        labels = kmeans.fit_predict(transformed_data)
        return labels

    def cluster_dbscan(self, transformed_data: pd.DataFrame, 
                       epsilon: float, min_samples: int) -> List[int]:
        """
        Implements dbscan clustering.
        :param transformed_data: output of principal component analysis
        :param epsilon: distance
        :param min_samples: minimum samples to be a cluster
        :return: labels
        """
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        labels = dbscan.fit_predict(transformed_data)
        return labels
    
    def cluster_optics(self, transformed_data: pd.DataFrame, 
                       min_samples: int) -> List[int]:
        """
        Implements optics clustering.
        :param: transformed_data: output of principal component analysis
        :param min_samples: minimm samples to be a cluster
        :return: labels
        """
        optics = OPTICS(min_samples=min_samples)
        labels = optics.fit_predict(transformed_data)
        return labels

    def cluster_hierarchical(self, transformed_data: pd.DataFrame, 
                             nclusters: int) -> List[int]:
        """
        Implements agglomerative clustering.
        :param transformed_data: output of principal component analysis
        :param nclusters: number of clusters:
        :return: labels
        """
        agglomerative = AgglomerativeClustering(n_clusters=nclusters)
        labels = agglomerative.fit_predict(transformed_data)
        return labels

    ### CHOICE OF PAIRS IDENTIFICATION ###

    def hurst(self, p, lag_num):
        lags = range(2, lag_num)
        variancetau, tau = [], []
        for lag in lags: 
            tau.append(lag)
            pp = np.subtract(p[lag:], p[:-lag])
            variancetau.append(np.var(pp)) 
        try:
            m = np.polyfit(np.log10(tau), np.log10(variancetau), 1)
            hurst = m[0] / 2
            return hurst
        except:
            return 1.0

    def engle_granger_cointegration(self, df: pd.DataFrame, 
                                    labels: List[int]) -> Dict:
        """
        If two stocks are in the same cluster, as specified by the labels 
        array, then perform the Engel Granger Test for Cointegration.
        :param df: close data
        :param labels: output from clustering
        :return: Dictionary with pairs as keys and pvalues as values
        """
        # n is the number of assets in our universe
        n = df.shape[1]
        # For earlier alphas, if we choose not to cluster, then just set n = 50.
        if not self.full_universe and self.cluster == Cluster.NONE:
            n = 50
        # Keys are their column index. 
        keys = df.keys()
        # Initalize the dictionary that we will return.
        pairs_dict = {}
        # Iterate through all possible combinations of pairs.
        for i in range(n):
            for j in range(i + 1, n):
                # Proceed if the stocks are in the same cluster and not outliers.
                if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                    # Isolate the two stocks and drop NaNs.
                    S1 = df[keys[i]]
                    S2 = df[keys[j]]
                    coint_df = pd.DataFrame({"S1": S1, "S2": S2}).dropna()
                    S1 = coint_df["S1"]
                    S2 = coint_df["S2"]
                    # Ensure that the two time series are not constant.
                    if S1.nunique() > 1 and S2.nunique() > 1:
                        pvalue = coint(S1, S2)[1]
                        # Include this pair if it rejects the null hypothesis.
                        if pvalue < 0.05:
                            # Validate the mean reversion of the spread with hurst.
                            if self.use_hurst:
                                diff = list(S1 - S2)
                                lags = min(int(4 + len(diff) / 3), 100)
                                h_exp = self.hurst(pd.DataFrame(diff), lags)
                                if h_exp < 0.5:
                                    if self.hurst_sort:  
                                        pairs_dict[(i, j)] = h_exp
                                    else:
                                        pairs_dict[(i, j)] = pvalue   
                        # Else just add it to the dictionary.
                        else:
                            pairs_dict[(i, j)] = pvalue
        return pairs_dict

    def other_cointegration(self):
        pass

    ### CHOICE OF PAIRS SELECTION ###

    def lowest(self, pairs_dict: Dict) -> List[Tuple]:
        """
        Returns the list of pairs sorted by lowest pvalue or hurst.
        :param pairs_dict: Dictionary with keys as pairs and values as pvalues or hurst exponents.
        :return: list of pairs sorted by lowest pvalue or hurst exponent.
        """
        # Obtain the keys and values of the pairs dictionary.
        keys = list(pairs_dict.keys())
        values = list(pairs_dict.values())
        # Argsort the values such that we can obtain the keys in that order.
        args = np.argsort(values)
        return [keys[i] for i in args]
    
    def randomize(self, pairs_dict: Dict) -> List[Tuple]:
        pairs = list(pairs_dict.keys())
        random.shuffle(pairs)
        return pairs

    def remove_duplicates(self, pairs: List[Tuple]) -> List[Tuple]:
        """
        Ensures that one stock can only appear in our list of pairs once to 
        prevent mishaps with our self.holdings array.
        :param: list of pairs
        :return: list of pairs without duplicate stocks.
        """
        res = []
        seen = set()
        # Iterate through all cointegrated pairs.
        for (i, j) in pairs:
            # Only proceed if both stocks have not been seen before.
            if i not in seen and j not in seen:
                res.append((i, j))
                seen.add(i)
                seen.add(j)
        return res

    ### TRADING METHODOLOGY FUNCTIONS ###

    def zscore(self, series: pd.Series) -> float:
        """
        Normalizes the spread of the two cointegrated stocks.
        :param: the spread
        :return: the zscore
        """
        return (series - series.mean()) / np.std(series)
    
    def simple_mean_reversion(self, data: pd.DataFrame) -> List[float]:
        """
        Mechanically trade using fixed trading thresholds.
        :param df: close price data
        :return: weights
        """
        # Create new pairs after self.reset days.
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data["close"])
        # Isolate only the close price data.
        data = pd.DataFrame(data["close"])
        for p in self.pairs:
            # These are the two stocks, corresponding to column indices.
            FIRST = p[0]
            SECOND = p[1]
            # Calculate the current zscore of the spread.
            spread = data[FIRST] - data[SECOND]
            zscore = self.zscore(spread).tail(1).values[0]
            # Enter a trade here.
            if zscore >= self.enter and zscore <= self.stop:
                self.holdings[FIRST] = -1
                self.holdings[SECOND] = 1
            # Exit the trade here.
            elif abs(zscore) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0
            # Consider our stop loss.
            elif abs(zscore) > self.stop:
                S1 = data[FIRST]
                S2 = data[SECOND]
                coint_df = pd.DataFrame({"S1": S1, "S2": S2}).dropna()
                S1 = coint_df["S1"]
                S2 = coint_df["S2"]
                if S1.nunique() > 1 and S2.nunique() > 1:
                    pvalue = coint(S1, S2)[1]
                    if pvalue >= 0.05:
                        self.holdings[FIRST] = 0
                        self.holdings[SECOND] = 0
                        self.pairs.remove(p)
            # Enter a trade here too.
            elif zscore <= -self.enter and zscore >= -self.stop:
                self.holdings[FIRST] = 1
                self.holdings[SECOND] = -1
        # Decrement day counter at the end of the day.
        self.day_counter -= 1
        return op.weight(self.holdings)
    
    def arima(self):
        pass

    def xgboost(self):
        pass

    def sentiment_analysis(self):
        pass
    
    ### FUNCTIONS THE RESEARCHER MUST IMPLEMENT ### 

    def form_pairs(self, df: pd.DataFrame) -> List[int]:
        """
        Allows the researcher to easily make their choice of clustering, 
        pairs identification, selection, and validation.
        :param df: close data
        :return: a list of trading pairs
        """
        raise NotImplementedError
    

    def generate_day(self, day: datetime, data: pd.DataFrame) -> List[float]:
        """
        Allows the researcher to easily make their choice of trading 
        methodologies including simple mean reversion (SMR), ARIMA, XGBoost,
        or sentiment analysis.
        :param day: the current day in the backtester
        :data: close price data
        :return: weights
        """
        raise NotImplementedError
    
class SMRPairsTrader(ParentPairsTrader):
    """
    This type of pairs trading algorithm has customizable trading thresholds,
    but they are not dynamic. We call this simple mean reversion (SMR).
    """
    def __init__(self, reset: int, 
                 npairs: int, 
                 enter: int, 
                 exit: int,
                 stop: Optional[int] = sys.maxsize, 
                 cluster: Optional[Cluster] = Cluster.NONE, 
                 use_random: Optional[bool] = False,
                 n_clusters: Optional[int] = None, 
                 hurst: Optional[bool] = False, 
                 full_universe: Optional[bool] = True,
                 hurst_sort: Optional[bool] = False):
        # Required variables for AbstractAlpha
        self.name = "SMR"
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        # Variables for Pairs Trading Alpha
        self.pairs = None
        self.reset = reset
        self.npairs = npairs
        self.enter = enter
        self.exit = exit
        self.stop = stop
        self.use_hurst = hurst
        self.hurst_sort = hurst_sort
        self.cluster = cluster
        self.use_random = use_random
        self.full_universe = full_universe
        self.n_clusters = n_clusters
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0

    def form_pairs(self, df: pd.DataFrame) -> List[int]:
        """
        We will use principal component analysis, optics clustering, and 
        cointegration tests for the most part.
        :param df: close price data
        :return: a list of pairs
        """
        df = pd.DataFrame(df).dropna()
        n = df.shape[1]
        # Principal component analysis.
        transformed_data = self.reduce_dimensionality(df)
        # Researcher's choice of clustering algorithm.
        if self.cluster == Cluster.KMEANS:
            labels = self.cluster_kmeans(transformed_data, 60)
        elif self.cluster == Cluster.DBSCAN:
            labels = self.cluster_dbscan(transformed_data, 0.05, 2)
        elif self.cluster == Cluster.OPTICS:
            labels = self.cluster_optics(transformed_data, 2)
        elif self.cluster == Cluster.AGGLOMERATIVE:
            labels = self.cluster_hierarchical(transformed_data, self.n_clusters)
        else:
            labels = list(np.zeros(n))
        # Pairs identification with or without hurst exponent validation.
        pairs_dict = self.engle_granger_cointegration(df, labels)
        # Pairs selection.
        if self.use_random:
            pairs = self.randomize(pairs_dict)
        else:
            pairs = self.lowest(pairs_dict)
        pairs = self.remove_duplicates(pairs)
        return pairs if self.npairs > len(pairs) else pairs[0: self.npairs]
    
    def generate_day(self, day: datetime, data: pd.DataFrame) -> List[float]:
        """
        See simple mean reversion implementation in ParentPairsTrader class.
        :param day:
        :param data: close price data
        :return: weights
        """
        return self.simple_mean_reversion(data)