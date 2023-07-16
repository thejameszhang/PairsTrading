import datetime
import random
from typing import List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import coint
array_1d = Union[List, Tuple, pd.Series, np.ndarray]
date_obj = Union[datetime.datetime, datetime.date]
import sys
sys.path.append('../')
from siftools import abstractalpha
from siftools import operators as op

"""
This alphas.py file consists of all of the alphas that we have developed and 
tested throughout our iterative development process. We include these alphas
in a separate file instead of inside the notebook in order to use the SIF
Backtester's multiprocessing functionality and for better organization.

The below section of brute force alphas only consider 50 stocks in the 
universe, and essentially have no efficiency optimizations made. Both of the 
following alphas test all possible combinations of pairs for cointegration 
using the  Engle-Granger test. They differ in their methodology of selecting 
profitable pairs from this lsit. All alphas, unless noted otherwise, 
implement a Fully Invested Weighting Scheme.
"""

class Random(abstractalpha.AbstractAlpha):
    """
    Runs the Engle-Granger test for cointegration, and randomly selects
    10 pairs from the list.
    """
    def __init__(self, reset, npairs, exit):
        self.name = 'Random'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = True
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
    
    def form_pairs(self, df):
        df = pd.DataFrame(df)
        n = df.shape[1]
        
        # creating an adjacency matrix sort of for cointegration scores and pvalues
        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        for i in range(50):
            for j in range(i + 1, 50):
                S1 = df[keys[i]]
                S2 = df[keys[j]]
                coint_df = pd.DataFrame({'S1': S1, 'S2': S2}).dropna()
                S1 = coint_df['S1']
                S2 = coint_df['S2']
                result = coint(S1, S2)
                pvalue = result[1]
                pvalue_matrix[i, j] = pvalue
                if pvalue < 0.05 and pvalue != 0:
                    pairs.append([i, j])
        
        # a stock should only be included once in our holdings to prevent mishaps
        new_pairs, seen = [], set()
        for (i, j) in pairs:
            if i not in seen and j not in seen:
                new_pairs.append([i, j])
                seen.add(i)
                seen.add(j)

        new_pairs = new_pairs if len(new_pairs) < self.npairs else random.sample(new_pairs, self.npairs)
        return new_pairs
    
    def zscore(self, series: array_1d) -> float:
        return (series - series.mean()) / np.std(series)
        
    def generate_day(self, day, data):
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
      
        data = pd.DataFrame(data['close'])
        global ex_data
        if self.print:
            ex_data = data
            self.print = False

        for p in self.pairs:
            # FIRST and SECOND are indices of the stocks
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            if z_score >= 1.0:
                self.holdings[FIRST] = -1
                self.holdings[SECOND] = 1
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0

            # enter the trade; long the FIRST, short SECOND
            elif z_score <= -1.0:
                self.holdings[FIRST] = 1
                self.holdings[SECOND] = -1
            
        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return op.weight(self.holdings)

class Lowest_PValue(abstractalpha.AbstractAlpha):
    """
    After running the Engle-Granger Test for cointegration, sort all of the 
    cointegrated pairs by lowest pvalue. A lower pvalue rejects the null 
    hypothesis that the two time series are not cointegrated. Therefore, the 
    lower the pvalue, the more confident we are in that the pairs will be 
    mean reverting.
    """
    def __init__(self, reset, npairs, exit):
        self.name = 'Lowest PValue'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = False
        self.keyPairs = []
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def form_pairs(self, df):
        df = pd.DataFrame(df)
        n = df.shape[1]
        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        pairsDict = {} #maps from pvalue to pair
        for i in range(50):
            for j in range(i + 1, 50):
                S1 = df[keys[i]]
                S2 = df[keys[j]]
                coint_df = pd.DataFrame({'S1': S1, 'S2': S2}).dropna()
                S1 = coint_df['S1']
                S2 = coint_df['S2']
                result = coint(S1, S2)
                pvalue = result[1]
                pvalue_matrix[i, j] = pvalue
                
                # used a hashmap, sorted keys (pvalues) in ascending order in order
                # to get the smallest pvalues
                if pvalue < 0.05 and pvalue != 0:
                    pairsDict[(i,j)] = pvalue
        
        # sort the pairs by lowest p-value
        keys, values = list(pairsDict.keys()), list(pairsDict.values())
        sorted_value_index = np.argsort(values)
        sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

        # a stock should only be included once in our holdings to prevent mishaps
        new_dict, seen = {}, set()
        for (i, j) in sorted_dict:
            if i not in seen and j not in seen:
                new_dict[(i, j)] = sorted_dict[(i,j)]
                seen.add(i)
                seen.add(j)
        
        # obtain top npairs pairs if possible
        pairs = list(new_dict.keys()) if len(new_dict) < self.npairs else list((new_dict.keys()))[0:self.npairs]
        return pairs
    
    def zscore(self, series):
        return (series - series.mean()) / np.std(series)

    def generate_day(self, day, data):
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
      
        data = pd.DataFrame(data['close'])
        for p in self.pairs:
            # FIRST and SECOND are indices of the stocks
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            if z_score >= 1.0:
                self.holdings[FIRST] = -1
                self.holdings[SECOND] = 1
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0

            # enter the trade; long the FIRST, short SECOND
            elif z_score <= -1.0:
                self.holdings[FIRST] = 1
                self.holdings[SECOND] = -1
            
        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return op.weight(self.holdings)

# -------------------------------------------------------------------------- #

"""
This section takes the previous brute force alphas with a universe size of
50 and implements PCA and OPTICS clustering such that we can consider a
universe size of 500. We denote these new alphas as enhanced alphas.
"""

class Enhanced_Random(abstractalpha.AbstractAlpha):
    """
    This alpha still randomly selects 10 pairs from the list of cointegrated
    pairs, but uses PCA and OPTICS to obtain these pairs.
    """
    def __init__(self, reset, npairs, exit):
        self.name = 'Enhanced Random Sample of Pairs'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = True
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def reduce_dimensionality(self, df: pd.DataFrame) -> pd.DataFrame:
        rets = df.pct_change().dropna()
        rets = rets.T
        pca = PCA(n_components = 0.9, svd_solver='full')
        transformed_data = pca.fit_transform(rets)
        return transformed_data
    
    def cluster_optics(self, transformed_data: array_1d) -> array_1d:
        optics = OPTICS(min_samples = 2)
        labels = optics.fit_predict(transformed_data)
        return labels

    def form_pairs(self, df):
        df = pd.DataFrame(df)
        clustering_df = df.dropna()
        n = df.shape[1]
        
        # obtain transformed data and clusters using above functions
        transformed_data = self.reduce_dimensionality(clustering_df)
        labels = self.cluster_optics(transformed_data)  

        # creating an adjacency matrix sort of for cointegration scores and pvalues
        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                    S1 = df[keys[i]]
                    S2 = df[keys[j]]
                    coint_df = pd.DataFrame({'S1': S1, 'S2': S2}).dropna()
                    S1 = coint_df['S1']
                    S2 = coint_df['S2']
                    result = coint(S1, S2)
                    pvalue = result[1]
                    pvalue_matrix[i, j] = pvalue
                    if pvalue < 0.05:
                        pairs.append([i, j])
        
        # a stock should only be included once in our holdings to prevent mishaps
        new_pairs, seen = [], set()
        for (i, j) in pairs:
            if i not in seen and j not in seen:
                new_pairs.append([i, j])
                seen.add(i)
                seen.add(j)

        new_pairs = new_pairs if len(new_pairs) < self.npairs else random.sample(new_pairs, self.npairs)
        return new_pairs
    
    def zscore(self, series: array_1d) -> float:
        return (series - series.mean()) / np.std(series)

    def generate_day(self, day, data):
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
      
        data = pd.DataFrame(data['close'])
        global ex_data
        if self.print:
            ex_data = data
            self.print = False

        for p in self.pairs:
            # FIRST and SECOND are indices of the stocks
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            if z_score >= 1.0:
                self.holdings[FIRST] = -1
                self.holdings[SECOND] = 1
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0

            # enter the trade; long the FIRST, short SECOND
            elif z_score <= -1.0:
                self.holdings[FIRST] = 1
                self.holdings[SECOND] = -1
            
        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return op.weight(self.holdings)

class Enhanced_Lowest_PValue(abstractalpha.AbstractAlpha):
    """
    This alpha uses PCA and OPTICS and selects the 10 pairs with lowest pvalues.
    """
    def __init__(self, reset, npairs, exit):
        self.name = 'Enhanced Lowest PValue Pairs'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = False
        self.keyPairs = []
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def zscore(self, series):
        return (series - series.mean()) / np.std(series)
    
    def reduce_dimensionality(self, df: pd.DataFrame) -> pd.DataFrame:
        rets = df.pct_change().dropna()
        rets = rets.T
        pca = PCA(n_components = 0.9, svd_solver='full')
        transformed_data = pca.fit_transform(rets)
        return transformed_data
        
    def cluster_optics(self, transformed_data: array_1d) -> array_1d:
        optics = OPTICS(min_samples = 2)
        labels = optics.fit_predict(transformed_data)
        return labels

    def form_pairs(self, df):
        df = pd.DataFrame(df)
        clustering_df = df.dropna()
        n = df.shape[1]

        # obtain transformed data and clusters using above functions
        transformed_data = self.reduce_dimensionality(clustering_df)
        labels = self.cluster_optics(transformed_data)

        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        pairsDict = {} #maps from pvalue to pair
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                    S1 = df[keys[i]]
                    S2 = df[keys[j]]
                    coint_df = pd.DataFrame({'S1': S1, 'S2': S2}).dropna()
                    S1 = coint_df['S1']
                    S2 = coint_df['S2']
                    result = coint(S1, S2)
                    pvalue = result[1]
                    pvalue_matrix[i, j] = pvalue
                
                    # used a hashmap, sorted keys (pvalues) in ascending order in order
                    # to get the smallest pvalues
                    if pvalue < 0.05:
                        pairsDict[(i,j)] = pvalue
        
        # sort the pairs by lowest p-value
        keys, values = list(pairsDict.keys()), list(pairsDict.values())
        sorted_value_index = np.argsort(values)
        sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

        # a stock should only be included once in our holdings to prevent mishaps
        new_dict, seen = {}, set()
        for (i, j) in sorted_dict:
            if i not in seen and j not in seen:
                new_dict[(i, j)] = sorted_dict[(i,j)]
                seen.add(i)
                seen.add(j)
        
        # obtain top npairs pairs if possible
        pairs = list(new_dict.keys()) if len(new_dict) < self.npairs else list((new_dict.keys()))[0:self.npairs]
        return pairs
    
    def generate_day(self, day, data):
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
      
        data = pd.DataFrame(data['close']).dropna()
        for p in self.pairs:
            # FIRST and SECOND are indices of the stocks
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            if z_score >= 1.0:
                self.holdings[FIRST] = -1
                self.holdings[SECOND] = 1
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0

            # enter the trade; long the FIRST, short SECOND
            elif z_score <= -1.0:
                self.holdings[FIRST] = 1
                self.holdings[SECOND] = -1
            
        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return op.weight(self.holdings)

# -------------------------------------------------------------------------- #

"""
We notice that in a few cases there are rapid declines in our returns, 
predominantly with the lowest pvalue alphas. to  counter this, we 
propose including a stop loss condition, such that if the
spread of a pair diverges beyond a certain threshold, we will immediately 
exit the trade to minimize our losses and remove the current pair from
future consideration. this section takes the 8 enhanced alphas and 
adds a stop loss condition
"""

# brute force random alphas with a stop loss condition

class Enhanced_Noncolinear_Stop_Random(abstractalpha.AbstractAlpha):
    def __init__(self, reset, npairs, exit, stop):
        self.name = 'Enhanced Noncolinear Random Sample of Pairs with Stop - Fully Invested'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = True
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.stop = stop
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def form_pairs(self, df):
        df = pd.DataFrame(df).dropna()
        n = df.shape[1]
        
        # obtain transformed data and clusters using above functions
        transformed_data = self.reduce_dimensionality(df)
        labels = self.cluster_optics(transformed_data)

        # creating an adjacency matrix sort of for cointegration scores and pvalues
        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                    S1 = df[keys[i]]
                    S2 = df[keys[j]]
                    coint_df = pd.DataFrame({'S1': S1, 'S2': S2}).dropna()
                    S1 = coint_df['S1']
                    S2 = coint_df['S2']
                    result = coint(S1, S2)
                    pvalue = result[1]
                    pvalue_matrix[i, j] = pvalue
                    if pvalue < 0.05 and pvalue != 0:
                        pairs.append([i, j])
        
        # a stock should only be included once in our holdings to prevent mishaps
        new_pairs, seen = [], set()
        for (i, j) in pairs:
            if i not in seen and j not in seen:
                new_pairs.append([i, j])
                seen.add(i)
                seen.add(j)

        new_pairs = new_pairs if len(new_pairs) < self.npairs else random.sample(new_pairs, self.npairs)
        return new_pairs
    
    def zscore(self, series: array_1d) -> float:
        return (series - series.mean()) / np.std(series)
    
    def reduce_dimensionality(self, df: pd.DataFrame) -> pd.DataFrame:
        rets = df.pct_change().dropna()
        rets = rets.T
        pca = PCA(n_components = 0.9, svd_solver='full')
        transformed_data = pca.fit_transform(rets)
        return transformed_data
        
    def cluster_optics(self, transformed_data: array_1d) -> array_1d:
        optics = OPTICS(min_samples = 2)
        labels = optics.fit_predict(transformed_data)
        return labels

    def generate_day(self, day, data):
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
      
        data = pd.DataFrame(data['close'])
        global ex_data
        if self.print:
            ex_data = data
            self.print = False

        for p in self.pairs:
            # FIRST and SECOND are indices of the stocks
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            if z_score >= 1.0 and z_score <= self.stop:
                self.holdings[FIRST] = -1
                self.holdings[SECOND] = 1
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0
            
            # ensure that the pair is still cointegrated, otherwise eat the losses before it gets worse
            elif abs(z_score) > self.stop:
                temp = data.dropna()
                S1, S2 = temp[FIRST], temp[SECOND]
                result = coint(S1, S2)
                pvalue = result[1]
                if pvalue >= 0.05:
                    self.holdings[FIRST] = 0
                    self.holdings[SECOND] = 0
                    self.pairs.remove(p)

            # enter the trade; long the FIRST, short SECOND
            elif z_score <= -1.0 and z_score >= -self.stop:
                self.holdings[FIRST] = 1
                self.holdings[SECOND] = -1
            
        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return op.weight(self.holdings)

# brute force lowest pvalue alphas with a stop loss condition

class Enhanced_Noncolinear_Stop_Lowest_PValue(abstractalpha.AbstractAlpha):
    def __init__(self, reset, npairs, exit, stop):
        self.name = 'Enhanced Noncolinear Lowest PValue Pairs with Stop - Fully Invested'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = False
        self.keyPairs = []
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.stop = stop
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def zscore(self, series):
        return (series - series.mean()) / np.std(series)
    
    def form_pairs(self, df):
        df = pd.DataFrame(df)
        clustering_df = df.dropna()
        n = df.shape[1]
        
        # obtain transformed data and clusters using above functions
        transformed_data = self.reduce_dimensionality(clustering_df)
        labels = self.cluster_optics(transformed_data)

        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        pairsDict = {} #maps from pvalue to pair
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                    S1 = df[keys[i]]
                    S2 = df[keys[j]]
                    coint_df = pd.DataFrame({'S1': S1, 'S2': S2}).dropna()
                    S1 = coint_df['S1']
                    S2 = coint_df['S2']
                    result = coint(S1, S2)
                    pvalue = result[1]
                    pvalue_matrix[i, j] = pvalue
                    
                    # used a hashmap, sorted keys (pvalues) in ascending order in order
                    # to get the smallest pvalues
                    if pvalue < 0.05 and pvalue != 0:
                        pairsDict[(i,j)] = pvalue
        
        # sort the pairs by lowest p-value
        keys, values = list(pairsDict.keys()), list(pairsDict.values())
        sorted_value_index = np.argsort(values)
        sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

        # a stock should only be included once in our holdings to prevent mishaps
        new_dict, seen = {}, set()
        for (i, j) in sorted_dict:
            if i not in seen and j not in seen:
                new_dict[(i, j)] = sorted_dict[(i,j)]
                seen.add(i)
                seen.add(j)
        
        # obtain top npairs pairs if possible
        pairs = list(new_dict.keys()) if len(new_dict) < self.npairs else list((new_dict.keys()))[0:self.npairs]
        return pairs
    
    def reduce_dimensionality(self, df: pd.DataFrame) -> pd.DataFrame:
        rets = df.pct_change().dropna()
        rets = rets.T
        pca = PCA(n_components = 0.9, svd_solver='full')
        transformed_data = pca.fit_transform(rets)
        return transformed_data
        
    def cluster_optics(self, transformed_data: array_1d) -> array_1d:
        optics = OPTICS(min_samples = 2)
        labels = optics.fit_predict(transformed_data)
        return labels

    def generate_day(self, day, data):
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
      
        data = pd.DataFrame(data['close'])
        for p in self.pairs:
            # FIRST and SECOND are indices of the stocks
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            if z_score >= 1.0 and z_score <= self.stop:
                self.holdings[FIRST] = -1
                self.holdings[SECOND] = 1
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0
            
            # ensure that the pair is still cointegrated, otherwise eat the losses before it gets worse
            elif abs(z_score) > self.stop:
                temp = data.dropna()
                S1, S2 = temp[FIRST], temp[SECOND]
                result = coint(S1, S2)
                pvalue = result[1]
                if pvalue >= 0.05:
                    self.holdings[FIRST] = 0
                    self.holdings[SECOND] = 0
                    self.pairs.remove(p)

            # enter the trade; long the FIRST, short SECOND
            elif z_score <= -1.0 and z_score >= -self.stop:
                self.holdings[FIRST] = 1
                self.holdings[SECOND] = -1
            
        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return op.weight(self.holdings)

# -------------------------------------------------------------------------- #

"""
These two alphas are used in the time complexity analysis.
"""

class Brute_Force(abstractalpha.AbstractAlpha):
    def __init__(self, reset, npairs, exit):
        self.name = 'Brute Force Strategy'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = False
        self.keyPairs = []
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def form_pairs(self, df):
        df = pd.DataFrame(df)
        n = df.shape[1]
        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        pairsDict = {} #maps from pvalue to pair
        for i in range(n):
            for j in range(i + 1, n):
                S1 = df[keys[i]]
                S2 = df[keys[j]]
                coint_df = pd.DataFrame({'S1': S1, 'S2': S2}).dropna()
                S1 = coint_df['S1']
                S2 = coint_df['S2']
                result = coint(S1, S2)
                pvalue = result[1]
                pvalue_matrix[i, j] = pvalue
                
                # used a hashmap, sorted keys (pvalues) in ascending order in order
                # to get the smallest pvalues
                if pvalue < 0.05 and pvalue != 0:
                    pairsDict[(i,j)] = pvalue
        
        # sort the pairs by lowest p-value
        keys, values = list(pairsDict.keys()), list(pairsDict.values())
        sorted_value_index = np.argsort(values)
        sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

        # a stock should only be included once in our holdings to prevent mishaps
        new_dict, seen = {}, set()
        for (i, j) in sorted_dict:
            if i not in seen and j not in seen:
                new_dict[(i, j)] = sorted_dict[(i,j)]
                seen.add(i)
                seen.add(j)
        
        # obtain top npairs pairs if possible
        pairs = list(new_dict.keys()) if len(new_dict) < self.npairs else list((new_dict.keys()))[0:self.npairs]
        return pairs
    
    def zscore(self, series):
        return (series - series.mean()) / np.std(series)

    def generate_day(self, day, data):
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
      
        data = pd.DataFrame(data['close'])
        for p in self.pairs:
            # FIRST and SECOND are indices of the stocks
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            if z_score >= 1.0:
                self.holdings[FIRST] = -1
                self.holdings[SECOND] = 1
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0

            # enter the trade; long the FIRST, short SECOND
            elif z_score <= -1.0:
                self.holdings[FIRST] = 1
                self.holdings[SECOND] = -1
            
        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return op.weight(self.holdings)

class Enhanced(abstractalpha.AbstractAlpha):
    def __init__(self, reset, npairs, exit, stop):
        self.name = 'Enhanced Strategy'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = False
        self.keyPairs = []
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.stop = stop
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def zscore(self, series):
        return (series - series.mean()) / np.std(series)
    
    def form_pairs(self, df):
        df = pd.DataFrame(df)
        clustering_df = df.dropna()
        n = df.shape[1]
        
        # obtain transformed data and clusters using above functions
        transformed_data = self.reduce_dimensionality(clustering_df)

        # obtain transformed data and clusters using above functions
        transformed_data = self.reduce_dimensionality(df)
        labels = self.cluster_optics(transformed_data)

        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        pairsDict = {} #maps from pvalue to pair
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                    S1 = df[keys[i]]
                    S2 = df[keys[j]]
                    coint_df = pd.DataFrame({'S1': S1, 'S2': S2}).dropna()
                    S1 = coint_df['S1']
                    S2 = coint_df['S2']
                    result = coint(S1, S2)
                    pvalue = result[1]
                    pvalue_matrix[i, j] = pvalue
                    
                    # used a hashmap, sorted keys (pvalues) in ascending order in order
                    # to get the smallest pvalues
                    if pvalue < 0.05 and pvalue != 0:
                        pairsDict[(i,j)] = pvalue
        
        # sort the pairs by lowest p-value
        keys, values = list(pairsDict.keys()), list(pairsDict.values())
        sorted_value_index = np.argsort(values)
        sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

        # a stock should only be included once in our holdings to prevent mishaps
        new_dict, seen = {}, set()
        for (i, j) in sorted_dict:
            if i not in seen and j not in seen:
                new_dict[(i, j)] = sorted_dict[(i,j)]
                seen.add(i)
                seen.add(j)
        
        # obtain top npairs pairs if possible
        pairs = list(new_dict.keys()) if len(new_dict) < self.npairs else list((new_dict.keys()))[0:self.npairs]
        return pairs
    
    def reduce_dimensionality(self, df: pd.DataFrame) -> pd.DataFrame:
        rets = df.pct_change().dropna()
        rets = rets.T
        pca = PCA(n_components = 0.9, svd_solver='full')
        transformed_data = pca.fit_transform(rets)
        return transformed_data
        
    def cluster_optics(self, transformed_data: array_1d) -> array_1d:
        optics = OPTICS(min_samples = 2)
        labels = optics.fit_predict(transformed_data)
        return labels

    def generate_day(self, day, data):
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
      
        data = pd.DataFrame(data['close'])
        for p in self.pairs:
            # FIRST and SECOND are indices of the stocks
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            if z_score >= 1.0 and z_score <= self.stop:
                self.holdings[FIRST] = -1
                self.holdings[SECOND] = 1
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0
            
            # ensure that the pair is still cointegrated, otherwise eat the losses before it gets worse
            elif abs(z_score) > self.stop:
                S1, S2 = data[FIRST], data[SECOND]
                result = coint(S1, S2)
                pvalue = result[1]
                if pvalue >= 0.05:
                    self.holdings[FIRST] = 0
                    self.holdings[SECOND] = 0
                    self.pairs.remove(p)

            # enter the trade; long the FIRST, short SECOND
            elif z_score <= -1.0 and z_score >= -self.stop:
                self.holdings[FIRST] = 1
                self.holdings[SECOND] = -1
            
        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return op.weight(self.holdings)

# -------------------------------------------------------------------------- #

"""
These alphas were used to compare the three clustering methods we tried.
"""

class KMeans_Alpha(abstractalpha.AbstractAlpha):
    def __init__(self, reset, npairs, exit):
        self.name = 'KMeans'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = False
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def zscore(self, series: array_1d) -> float:
        return (series - series.mean()) / np.std(series)

    def reduce_dimensionality(self, df: pd.DataFrame) -> pd.DataFrame:
        
        #note that we have to transpose rets to obtain a dataframe of shape n_samples, n_features
        rets = df.pct_change().dropna()
        rets = rets.T

        # use pca to reduce dimensionality to preserve at least 90% of original information
        pca = PCA(n_components = 0.9, svd_solver='full')
        transformed_data = pca.fit_transform(rets)
        return transformed_data
    
    def cluster_kmeans(self, transformed_data: array_1d) -> array_1d:
        kmeans = KMeans(n_clusters = 60)
        labels = kmeans.fit_predict(transformed_data)
        return labels
    
    def form_pairs(self, df: pd.DataFrame) -> array_1d:
        df = pd.DataFrame(df)
        clustering_df = df.dropna()
        n = df.shape[1]
        
        # obtain transformed data and clusters using above functions
        transformed_data = self.reduce_dimensionality(clustering_df)
        labels = self.cluster_kmeans(transformed_data)
        keys = df.keys()
        pairs = []
        pairsDict = {}
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                    S1 = df[keys[i]]
                    S2 = df[keys[j]]
                    coint_df = pd.DataFrame({'S1': S1, 'S2': S2}).dropna()
                    S1 = coint_df['S1']
                    S2 = coint_df['S2']
                    result = coint(S1, S2)
                    pvalue = result[1]
                    if pvalue < 0.05 and pvalue != 0:
                        pairsDict[(i, j)] = pvalue
        
        # sort the dictionary by lowest pvalue
        keys, values = list(pairsDict.keys()), list(pairsDict.values())
        sorted_value_index = np.argsort(values)
        sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

        # a stock should only be included once in our holdings to prevent mishaps
        new_dict, seen = {}, set()
        for (i, j) in sorted_dict:
            if i not in seen and j not in seen:
                new_dict[(i, j)] = sorted_dict[(i,j)]
                seen.add(i)
                seen.add(j)
        
        if len(new_dict) < self.npairs:
            pairs = list(new_dict.keys())
        else:
            pairs = list((new_dict.keys()))[0:self.npairs]
    
        return pairs
    
    def generate_day(self, day: date_obj, data: pd.DataFrame) -> array_1d:
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
            return op.weight(self.holdings)
      
        data = pd.DataFrame(data["close"])
        for p in self.pairs:
            spread = data[p[0]] - data[p[1]]
            z_score = self.zscore(spread).tail(1).values[0]
            if z_score > 1.0:
                self.holdings[p[0]] = -1
                self.holdings[p[1]] = 1
            elif abs(z_score) < self.exit:
                self.holdings[p[0]] = 0
                self.holdings[p[1]] = 0
            elif z_score < -1.0:
                self.holdings[p[0]] = 1
                self.holdings[p[1]] = -1
            
        self.day_counter -= 1
        return op.weight(self.holdings)

class DBSCAN_Alpha(abstractalpha.AbstractAlpha):
    def __init__(self, reset, npairs, exit):
        self.name = 'DBSCAN'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = False
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def zscore(self, series: array_1d) -> float:
        return (series - series.mean()) / np.std(series)

    def reduce_dimensionality(self, df: pd.DataFrame) -> pd.DataFrame:
        
        #note that we have to transpose rets to obtain a dataframe of shape n_samples, n_features
        rets = df.pct_change().dropna()
        rets = rets.T

        # use pca to reduce dimensionality to preserve at least 90% of original information
        pca = PCA(n_components = 0.9, svd_solver='full')
        transformed_data = pca.fit_transform(rets)
        return transformed_data
    
    def cluster_dbscan(self, transformed_data: array_1d) -> array_1d:
        db = DBSCAN(eps = 0.03, min_samples = 2)
        labels = db.fit_predict(transformed_data)
        return labels
    
    def form_pairs(self, df: pd.DataFrame) -> array_1d:
        df = pd.DataFrame(df)
        clustering_df = df.dropna()
        n = df.shape[1]
        
        # obtain transformed data and clusters using above functions
        transformed_data = self.reduce_dimensionality(clustering_df)
        labels = self.cluster_dbscan(transformed_data)
        keys = df.keys()
        pairs = []
        pairsDict = {}
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                    S1 = df[keys[i]]
                    S2 = df[keys[j]]
                    coint_df = pd.DataFrame({'S1': S1, 'S2': S2}).dropna()
                    S1 = coint_df['S1']
                    S2 = coint_df['S2']
                    result = coint(S1, S2)
                    pvalue = result[1]
                    if pvalue < 0.05 and pvalue != 0:
                        pairsDict[(i, j)] = pvalue
        
        # sort dict by lowest pvalue
        keys, values = list(pairsDict.keys()), list(pairsDict.values())
        sorted_value_index = np.argsort(values)
        sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

        # a stock should only be included once in our holdings to prevent mishaps
        new_dict, seen = {}, set()
        for (i, j) in sorted_dict:
            if i not in seen and j not in seen:
                new_dict[(i, j)] = sorted_dict[(i,j)]
                seen.add(i)
                seen.add(j)

        if len(new_dict) < self.npairs:
            pairs = list(new_dict.keys())
        else:
            pairs = list((new_dict.keys()))[0:self.npairs]
        return pairs
    
    def generate_day(self, day: date_obj, data: pd.DataFrame) -> array_1d:
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
            return op.weight(self.holdings)
      
        data = pd.DataFrame(data["close"])
        for p in self.pairs:
            spread = data[p[0]] - data[p[1]]
            z_score = self.zscore(spread).tail(1).values[0]
            if z_score > 1.0:
                self.holdings[p[0]] = -1
                self.holdings[p[1]] = 1
            elif abs(z_score) < self.exit:
                self.holdings[p[0]] = 0
                self.holdings[p[1]] = 0
            elif z_score < -1.0:
                self.holdings[p[0]] = 1
                self.holdings[p[1]] = -1
            
        self.day_counter -= 1
        return op.weight(self.holdings)

class OPTICS_Alpha(abstractalpha.AbstractAlpha):
    def __init__(self, reset, npairs, exit):
        self.name = 'OPTICS'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = False
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def zscore(self, series: array_1d) -> float:
        return (series - series.mean()) / np.std(series)

    def reduce_dimensionality(self, df: pd.DataFrame) -> pd.DataFrame:
        
        #note that we have to transpose rets to obtain a dataframe of shape n_samples, n_features
        rets = df.pct_change().dropna()
        rets = rets.T

        # use pca to reduce dimensionality to preserve at least 90% of original information
        pca = PCA(n_components = 0.9, svd_solver='full')
        transformed_data = pca.fit_transform(rets)
        return transformed_data
    
    def cluster_optics(self, transformed_data: array_1d) -> array_1d:
        optics = OPTICS(min_samples = 2)
        labels = optics.fit_predict(transformed_data)
        return labels
    
    def form_pairs(self, df: pd.DataFrame) -> array_1d:
        df = pd.DataFrame(df)
        clustering_df = df.dropna()
        n = df.shape[1]
        
        # obtain transformed data and clusters using above functions
        transformed_data = self.reduce_dimensionality(clustering_df)
        labels = self.cluster_optics(transformed_data)
        keys = df.keys()
        pairs = []
        pairsDict = {}
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                    S1 = df[keys[i]]
                    S2 = df[keys[j]]
                    coint_df = pd.DataFrame({'S1': S1, 'S2': S2}).dropna()
                    S1 = coint_df['S1']
                    S2 = coint_df['S2']
                    result = coint(S1, S2)
                    pvalue = result[1]
                    if pvalue < 0.05 and pvalue != 0:
                        pairsDict[(i, j)] = pvalue
        
        # sort dict by lowest pvalue
        keys, values = list(pairsDict.keys()), list(pairsDict.values())
        sorted_value_index = np.argsort(values)
        sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

        # a stock should only be included once in our holdings to prevent mishaps
        new_dict, seen = {}, set()
        for (i, j) in sorted_dict:
            if i not in seen and j not in seen:
                new_dict[(i, j)] = sorted_dict[(i,j)]
                seen.add(i)
                seen.add(j)

        if len(new_dict) < self.npairs:
            pairs = list(new_dict.keys())
        else:
            pairs = list((new_dict.keys()))[0:self.npairs]
        return pairs
    
    def generate_day(self, day: date_obj, data: pd.DataFrame) -> array_1d:
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
            return op.weight(self.holdings)
      
        data = pd.DataFrame(data["close"])
        pairs_selected = []
        for p in self.pairs:
            spread = data[p[0]] - data[p[1]]
            z_score = self.zscore(spread).tail(1).values[0]
            if z_score > 1.0:
                self.holdings[p[0]] = -1
                self.holdings[p[1]] = 1
            elif abs(z_score) < self.exit:
                self.holdings[p[0]] = 0
                self.holdings[p[1]] = 0
            elif z_score < -1.0:
                self.holdings[p[0]] = 1
                self.holdings[p[1]] = -1
            
        self.day_counter -= 1
        return op.weight(self.holdings)

"""
Below is our current best performing alpha, which uses PCA, OPTICS, and 
has customizable parameters. This is the alpha we use for the hyperparameter
optimization.
"""

class Testing2(abstractalpha.AbstractAlpha):
    def __init__(self, reset, npairs, enter, exit, stop):
        self.name = 'Testing2'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = False
        self.keyPairs = []
        self.reset = reset
        self.npairs = npairs
        self.enter = enter
        self.exit = exit
        self.stop = stop
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def zscore(self, series):
        return (series - series.mean()) / np.std(series)
    
    def form_pairs(self, df):
        df = pd.DataFrame(df).dropna()
        n = df.shape[1]

        # obtain transformed data and clusters using above functions
        transformed_data = self.reduce_dimensionality(df)
        labels = self.cluster_optics(transformed_data)

        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        pairsDict = {} #maps from pvalue to pair
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                    S1 = df[keys[i]]        
                    S2 = df[keys[j]]
                    coint_df = pd.DataFrame({'S1': S1, 'S2': S2}).dropna()
                    S1 = coint_df['S1']
                    S2 = coint_df['S2']
                    result = coint(S1, S2)
                    result = coint(S1, S2)
                    pvalue = result[1]
                    pvalue_matrix[i, j] = pvalue
                    
                    # used a hashmap, sorted keys (pvalues) in ascending order in order
                    # to get the smallest pvalues
                    if pvalue < 0.05 and pvalue != 0:
                        pairsDict[(i,j)] = pvalue
        
        # sort the pairs by lowest p-value
        keys, values = list(pairsDict.keys()), list(pairsDict.values())
        sorted_value_index = np.argsort(values)
        sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

        # a stock should only be included once in our holdings to prevent mishaps
        new_dict, seen = {}, set()
        for (i, j) in sorted_dict:
            if i not in seen and j not in seen:
                new_dict[(i, j)] = sorted_dict[(i,j)]
                seen.add(i)
                seen.add(j)
        
        # obtain top npairs pairs if possible
        pairs = list(new_dict.keys()) if len(new_dict) < self.npairs else list((new_dict.keys()))[0:self.npairs]
        return pairs
    
    def reduce_dimensionality(self, df: pd.DataFrame) -> pd.DataFrame:
        rets = df.pct_change().dropna()
        rets = rets.T
        pca = PCA(n_components = 0.9, svd_solver='full')
        transformed_data = pca.fit_transform(rets)
        return transformed_data
        
    def cluster_optics(self, transformed_data: array_1d) -> array_1d:
        optics = OPTICS(min_samples = 2)
        labels = optics.fit_predict(transformed_data)
        return labels

    def generate_day(self, day, data):
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
      
        data = pd.DataFrame(data['close'])
        for p in self.pairs:
            # FIRST and SECOND are indices of the stocks
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            if z_score >= self.enter and z_score <= self.stop:
                self.holdings[FIRST] = -1
                self.holdings[SECOND] = 1
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0
            
            # ensure that the pair is still cointegrated, otherwise eat the losses before it gets worse
            elif abs(z_score) > self.stop:
                temp = data.dropna()
                S1, S2 = temp[FIRST], temp[SECOND]
                result = coint(S1, S2)
                pvalue = result[1]
                if pvalue >= 0.05:
                    self.holdings[FIRST] = 0
                    self.holdings[SECOND] = 0
                    self.pairs.remove(p)

            # enter the trade; long the FIRST, short SECOND
            elif z_score <= -self.enter and z_score >= -self.stop:
                self.holdings[FIRST] = 1
                self.holdings[SECOND] = -1
            
        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return op.weight(self.holdings)

# -------------------------------------------------------------------------- #

"""
These alphas implement Hurst Exponent, a stochastic process that indicates
whether a time series exhibits mean reversion, brownian motion / random walk, 
or trending behavior. 
"""

class HURST(abstractalpha.AbstractAlpha):
    def __init__(self, reset, npairs, enter, exit, stop, hurstt):
        self.name = 'HURST'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = False
        self.reset = reset
        self.npairs = npairs
        self.enter = enter
        self.exit = exit
        self.stop = stop
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
        
        self.hurst = hurstt # MAXIMUM Hurst Exponent. Recall 0.5 means that the series reflects brownian motion
        
    def hurst(p):
        lags = range(2,100)
        variancetau, tau = [], []
        for lag in lags: 
            tau.append(lag) # Write the different lags into a vector to compute a set of tau or lags
            
            # Compute the log returns on all days, then compute the variance on the difference in log returns
            # call this pp or the price difference
            pp = np.subtract(p[lag:], p[:-lag])
            variancetau.append(np.var(pp))

        # we now have a set of tau or lags and a corresponding set of variances.
        # plot the log of those variance against the log of tau and get the slope
        m = np.polyfit(np.log10(tau), np.log10(variancetau), 1)
        hurst = m[0] / 2
        return hurst

    def zscore(self, series: array_1d) -> float:
        return (series - series.mean()) / np.std(series)

    def reduce_dimensionality(self, df: pd.DataFrame) -> pd.DataFrame:
        
        #note that we have to transpose rets to obtain a dataframe of shape n_samples, n_features
        rets = df.pct_change().dropna()
        rets = rets.T

        # use pca to reduce dimensionality to preserve at least 90% of original information
        pca = PCA(n_components = 0.9, svd_solver='full')
        transformed_data = pca.fit_transform(rets)
        
        if self.print:
            print("Shape of our data:", df.shape)
            print("Preserved Information:", sum(pca.explained_variance_ratio_))
            print("Number of Components:", pca.n_components_)
            print("Reduced dimensions shape of our new data:", transformed_data.shape)
        return transformed_data
    
    def cluster_optics(self, transformed_data: array_1d) -> array_1d:
        optics = OPTICS(min_samples = 2)
        labels = optics.fit_predict(transformed_data)

        if self.print:
            # Number of clusters in labels, ignoring noise if present.
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            clusters = {}
            for i in range(len(labels)):
                if labels[i] != -1:
                    clusters[labels[i]] = clusters.get(labels[i], 0) + 1
            print("Estimated number of clusters:", n_clusters)
            print("Estimated number of noise points:", n_noise)
            print("Number of securities per cluster:", clusters)
        return labels
    
    def form_pairs(self, df: pd.DataFrame) -> array_1d:
        df = pd.DataFrame(df).dropna()
        n = df.shape[1]

        # obtain transformed data and clusters using above functions
        transformed_data = self.reduce_dimensionality(df)
        labels = self.cluster_optics(transformed_data)

        # creating an adjacency matrix sort of for cointegration scores and pvalues
        keys = df.keys()
        pairs = []
        pairsDict = {} #maps from pvalue to pair
        for i in range(n):
            for j in range(i + 1, n):
                
                # assert that both stocks are in the same cluster and not outliers
                if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                    S1 = df[keys[i]]        
                    S2 = df[keys[j]]
                    result = coint(S1, S2)
                    pvalue = result[1]
                    
                    # used a hashmap, sorted keys (pvalues) in ascending order in order
                    # to get the smallest pvalues
                    if pvalue < 0.05:
                        # NEW: CHECK HURST EXPONENT
                        h_exp = hurst(list(S1-S2))
                        pairsDict[(i, j)] = h_exp
        
        # sort the pairs by lowest p-value
        keys, values = list(pairsDict.keys()), list(pairsDict.values())
        sorted_value_index = np.argsort(values)
        sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

        # a stock should only be included once in our holdings to prevent mishaps
        new_dict, seen = {}, set()
        for (i, j) in sorted_dict:
            if i not in seen and j not in seen:
                new_dict[(i, j)] = sorted_dict[(i,j)]
                seen.add(i)
                seen.add(j)
        
        # choose 10 pairs if possible
        if len(new_dict) < self.npairs:
            pairs = list(new_dict.keys())
        else:
            pairs = list((new_dict.keys()))[0:self.npairs]
            
        if self.print:
            print(f"PValue Cointegrated Pairs - Round {self.count} - Pairs are {pairs}")
            print()
            self.count += 1
        return pairs
    
#     def ratio_function(self, num1: float, num2: float) -> array_1d:
#         gcd = math.gcd(int(num1), int(num2))
#         return num1/gcd, num2/gcd
    
    def generate_day(self, day: date_obj, data: pd.DataFrame) -> array_1d:
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.pairs = self.form_pairs(data['close'])
            self.holdings = np.zeros(self.universe_size)
      
        data = pd.DataFrame(data["close"])
        pairs_selected = []
        for p in self.pairs:
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            # enter the trade, short the FIRST, long SECOND
            if z_score > self.enter and z_score <= self.stop:
                self.holdings[FIRST] = -1
                self.holdings[SECOND] = 1
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0
            
            # ensure that the pair is still cointegrated, otherwise eat the losses before it gets worse
            elif abs(z_score) > self.stop:
                temp = data.dropna()
                S1, S2 = temp[FIRST], temp[SECOND]
                result = coint(S1, S2)
                pvalue = result[1]
                if pvalue >= 0.05:
                    self.holdings[FIRST] = 0
                    self.holdings[SECOND] = 0
                    self.pairs.remove(p)

            # enter the trade; long the FIRST, short SECOND
            elif z_score < -self.enter and z_score >= -self.stop:
                self.holdings[FIRST] = 1
                self.holdings[SECOND] = -1
            
        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return op.weight(self.holdings)

class HURST_Sort(abstractalpha.AbstractAlpha):
    def __init__(self, reset, npairs, enter, exit, stop, hurstt):
        self.name = 'HURST Sort'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = False
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.enter = enter
        self.stop = stop
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
        
        self.hurst = hurstt # MAXIMUM Hurst Exponent. Recall 0.5 means that the series reflects brownian motion

    def zscore(self, series: array_1d) -> float:
        return (series - series.mean()) / np.std(series)

    def reduce_dimensionality(self, df: pd.DataFrame) -> pd.DataFrame:
        
        #note that we have to transpose rets to obtain a dataframe of shape n_samples, n_features
        rets = df.pct_change().dropna()
        rets = rets.T

        # use pca to reduce dimensionality to preserve at least 90% of original information
        pca = PCA(n_components = 0.9, svd_solver='full')
        transformed_data = pca.fit_transform(rets)
        
        if self.print:
            print("Shape of our data:", df.shape)
            print("Preserved Information:", sum(pca.explained_variance_ratio_))
            print("Number of Components:", pca.n_components_)
            print("Reduced dimensions shape of our new data:", transformed_data.shape)
        return transformed_data
    
    def cluster_optics(self, transformed_data: array_1d) -> array_1d:
        optics = OPTICS(min_samples = 2)
        labels = optics.fit_predict(transformed_data)

        if self.print:
            # Number of clusters in labels, ignoring noise if present.
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            clusters = {}
            for i in range(len(labels)):
                if labels[i] != -1:
                    clusters[labels[i]] = clusters.get(labels[i], 0) + 1
            print("Estimated number of clusters:", n_clusters)
            print("Estimated number of noise points:", n_noise)
            print("Number of securities per cluster:", clusters)
        return labels
    
    def form_pairs(self, df: pd.DataFrame) -> array_1d:
        df = pd.DataFrame(df).dropna()
        n = df.shape[1]

        # obtain transformed data and clusters using above functions
        transformed_data = self.reduce_dimensionality(df)
        labels = self.cluster_optics(transformed_data)

        # creating an adjacency matrix sort of for cointegration scores and pvalues
        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        pairsDict = {} #maps from pvalue to pair
        for i in range(n):
            for j in range(i + 1, n):
                
                # assert that both stocks are in the same cluster and not outliers
                if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                    S1 = df[keys[i]]        
                    S2 = df[keys[j]]
                    result = coint(S1, S2)
                    pvalue = result[1]
                    
                    # used a hashmap, sorted keys (pvalues) in ascending order in order
                    # to get the smallest pvalues
                    if pvalue < 0.05:
                        # NEW: CHECK HURST EXPONENT
                        h_exp = hurst(list(S1-S2))
                        pairsDict[(i, j)] = h_exp
        
        # sort the pairs by lowest p-value
        keys, values = list(pairsDict.keys()), list(pairsDict.values())
        sorted_value_index = np.argsort(values)
        sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

        # a stock should only be included once in our holdings to prevent mishaps
        new_dict, seen = {}, set()
        for (i, j) in sorted_dict:
            if i not in seen and j not in seen:
                new_dict[(i, j)] = sorted_dict[(i,j)]
                seen.add(i)
                seen.add(j)
        
        # choose 10 pairs if possible
        if len(new_dict) < self.npairs:
            pairs = list(new_dict.keys())
        else:
            pairs = list((new_dict.keys()))[0:self.npairs]
            
        if self.print:
            print(f"PValue Cointegrated Pairs - Round {self.count} - Pairs are {pairs}")
            print()
            self.count += 1
        return pairs
    
#     def ratio_function(self, num1: float, num2: float) -> array_1d:
#         gcd = math.gcd(int(num1), int(num2))
#         return num1/gcd, num2/gcd


    def hurst(p):
        lags = range(2,100)
        variancetau, tau = [], []
        for lag in lags: 
            tau.append(lag) # Write the different lags into a vector to compute a set of tau or lags
            
            # Compute the log returns on all days, then compute the variance on the difference in log returns
            # call this pp or the price difference
            pp = np.subtract(p[lag:], p[:-lag])
            variancetau.append(np.var(pp))

        # we now have a set of tau or lags and a corresponding set of variances.
        # plot the log of those variance against the log of tau and get the slope
        m = np.polyfit(np.log10(tau), np.log10(variancetau), 1)
        hurst = m[0] / 2
        return hurst
    
    def generate_day(self, day: date_obj, data: pd.DataFrame) -> array_1d:
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.pairs = self.form_pairs(data['close'])
            self.holdings = np.zeros(self.universe_size)
      
        data = pd.DataFrame(data["close"])
        pairs_selected = []
        for p in self.pairs:
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            # enter the trade, short the FIRST, long SECOND
            if z_score > 1.0 and z_score <= self.stop:
                self.holdings[FIRST] = -1
                self.holdings[SECOND] = 1
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0
            
            # ensure that the pair is still cointegrated, otherwise eat the losses before it gets worse
            elif abs(z_score) > self.stop:
                temp = data.dropna()
                S1, S2 = temp[FIRST], temp[SECOND]
                result = coint(S1, S2)
                pvalue = result[1]
                if pvalue >= 0.05:
                    self.holdings[FIRST] = 0
                    self.holdings[SECOND] = 0
                    self.pairs.remove(p)

            # enter the trade; long the FIRST, short SECOND
            elif z_score < -1.0 and z_score >= -self.stop:
                self.holdings[FIRST] = 1
                self.holdings[SECOND] = -1
            
        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return op.weight(self.holdings)

# -------------------------------------------------------------------------- #

"""
DEPRECATED ALPHAS
"""

"""
These alphas that allow colinear pairs when conducting pairs selection. 
According to the Statsmodels documentation, when two time series are almost
perfectly colinear, the test becomes unstable and the t-statistic is set to 
infinity and the pvalue is set to 0. If two securities are perfectly 
colinear, then they will almost never diverge from their historical mean, thus
decreasing profitability and our alpha's activity.
"""

class Random1(abstractalpha.AbstractAlpha):
    def __init__(self, reset, npairs, exit):
        self.name = 'Random Sample of Pairs - Fully Invested'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = True
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def form_pairs(self, df):
        df = pd.DataFrame(df).dropna()
        n = df.shape[1]
        
        # creating an adjacency matrix sort of for cointegration scores and pvalues
        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        for i in range(50):
            for j in range(i + 1, 50):
                S1 = df[keys[i]]        
                S2 = df[keys[j]]
                result = coint(S1, S2)
                pvalue = result[1]
                pvalue_matrix[i, j] = pvalue
                if pvalue < 0.05:
                    pairs.append([i, j])
        
        # a stock should only be included once in our holdings to prevent mishaps
        new_pairs, seen = [], set()
        for (i, j) in pairs:
            if i not in seen and j not in seen:
                new_pairs.append([i, j])
                seen.add(i)
                seen.add(j)

        new_pairs = new_pairs if len(new_pairs) < self.npairs else random.sample(new_pairs, self.npairs)
        return new_pairs
    
    def zscore(self, series: array_1d) -> float:
        return (series - series.mean()) / np.std(series)
        
    def generate_day(self, day, data):
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
      
        data = pd.DataFrame(data['close'])
        global ex_data
        if self.print:
            ex_data = data
            self.print = False

        for p in self.pairs:
            # FIRST and SECOND are indices of the stocks
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            if z_score >= 1.0:
                self.holdings[FIRST] = -1
                self.holdings[SECOND] = 1
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0

            # enter the trade; long the FIRST, short SECOND
            elif z_score <= -1.0:
                self.holdings[FIRST] = 1
                self.holdings[SECOND] = -1
            
        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return op.weight(self.holdings)

class Lowest_PValue1(abstractalpha.AbstractAlpha):
    def __init__(self, reset, npairs, exit):
        self.name = 'Lowest PValue Pairs - Fully Invested'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = False
        self.keyPairs = []
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def zscore(self, series):
        return (series - series.mean()) / np.std(series)
    
    def form_pairs(self, df):
        df = pd.DataFrame(df).dropna()
        n = df.shape[1]
        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        pairsDict = {} #maps from pvalue to pair
        for i in range(50):
            for j in range(i + 1, 50):
                S1 = df[keys[i]]        
                S2 = df[keys[j]]
                result = coint(S1, S2)
                pvalue = result[1]
                pvalue_matrix[i, j] = pvalue
                
                # used a hashmap, sorted keys (pvalues) in ascending order in order
                # to get the smallest pvalues
                if pvalue < 0.05:
                    pairsDict[(i,j)] = pvalue
        
        # sort the pairs by lowest p-value
        keys, values = list(pairsDict.keys()), list(pairsDict.values())
        sorted_value_index = np.argsort(values)
        sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

        # a stock should only be included once in our holdings to prevent mishaps
        new_dict, seen = {}, set()
        for (i, j) in sorted_dict:
            if i not in seen and j not in seen:
                new_dict[(i, j)] = sorted_dict[(i,j)]
                seen.add(i)
                seen.add(j)
        
        # obtain top npairs pairs if possible
        pairs = list(new_dict.keys()) if len(new_dict) < self.npairs else list((new_dict.keys()))[0:self.npairs]
        return pairs
    
    def generate_day(self, day, data):
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
      
        data = pd.DataFrame(data['close'])
        for p in self.pairs:
            # FIRST and SECOND are indices of the stocks
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            if z_score >= 1.0:
                self.holdings[FIRST] = -1
                self.holdings[SECOND] = 1
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0

            # enter the trade; long the FIRST, short SECOND
            elif z_score <= -1.0:
                self.holdings[FIRST] = 1
                self.holdings[SECOND] = -1
            
        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return op.weight(self.holdings)

class Enhanced_Random1(abstractalpha.AbstractAlpha):
    def __init__(self, reset, npairs, exit):
        self.name = 'Enhanced Random Sample of Pairs - Fully Invested'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = True
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def reduce_dimensionality(self, df: pd.DataFrame) -> pd.DataFrame:
        rets = df.pct_change().dropna()
        rets = rets.T
        pca = PCA(n_components = 0.9, svd_solver='full')
        transformed_data = pca.fit_transform(rets)
        return transformed_data
    
    def cluster_optics(self, transformed_data: array_1d) -> array_1d:
        optics = OPTICS(min_samples = 2)
        labels = optics.fit_predict(transformed_data)
        return labels

    def form_pairs(self, df):
        df = pd.DataFrame(df).dropna()
        n = df.shape[1]
        
        # obtain transformed data and clusters using above functions
        transformed_data = self.reduce_dimensionality(df)
        labels = self.cluster_optics(transformed_data)  

        # creating an adjacency matrix sort of for cointegration scores and pvalues
        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                    S1 = df[keys[i]]        
                    S2 = df[keys[j]]
                    result = coint(S1, S2)
                    pvalue = result[1]
                    pvalue_matrix[i, j] = pvalue
                    if pvalue < 0.05:
                        pairs.append([i, j])
        
        # a stock should only be included once in our holdings to prevent mishaps
        new_pairs, seen = [], set()
        for (i, j) in pairs:
            if i not in seen and j not in seen:
                new_pairs.append([i, j])
                seen.add(i)
                seen.add(j)

        new_pairs = new_pairs if len(new_pairs) < self.npairs else random.sample(new_pairs, self.npairs)
        return new_pairs
    
    def zscore(self, series: array_1d) -> float:
        return (series - series.mean()) / np.std(series)

    def generate_day(self, day, data):
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
      
        data = pd.DataFrame(data['close'])
        global ex_data
        if self.print:
            ex_data = data
            self.print = False

        for p in self.pairs:
            # FIRST and SECOND are indices of the stocks
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            if z_score >= 1.0:
                self.holdings[FIRST] = -1
                self.holdings[SECOND] = 1
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0

            # enter the trade; long the FIRST, short SECOND
            elif z_score <= -1.0:
                self.holdings[FIRST] = 1
                self.holdings[SECOND] = -1
            
        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return op.weight(self.holdings)

class Enhanced_Lowest_PValue_Noncolinear1(abstractalpha.AbstractAlpha):
    def __init__(self, reset, npairs, exit):
        self.name = 'Noncolinear Enhanced Lowest PValue Pairs - Fully Invested'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = False
        self.keyPairs = []
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def zscore(self, series):
        return (series - series.mean()) / np.std(series)
    
    def reduce_dimensionality(self, df: pd.DataFrame) -> pd.DataFrame:
        rets = df.pct_change().dropna()
        rets = rets.T
        pca = PCA(n_components = 0.9, svd_solver='full')
        transformed_data = pca.fit_transform(rets)
        return transformed_data
        
    def cluster_optics(self, transformed_data: array_1d) -> array_1d:
        optics = OPTICS(min_samples = 2)
        labels = optics.fit_predict(transformed_data)
        return labels

    def form_pairs(self, df):
        df = pd.DataFrame(df).dropna()
        n = df.shape[1]

        # obtain transformed data and clusters using above functions
        transformed_data = self.reduce_dimensionality(df)
        labels = self.cluster_optics(transformed_data)

        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        pairsDict = {} #maps from pvalue to pair
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                    S1 = df[keys[i]]        
                    S2 = df[keys[j]]
                    result = coint(S1, S2)
                    pvalue = result[1]
                    pvalue_matrix[i, j] = pvalue
                
                    # used a hashmap, sorted keys (pvalues) in ascending order in order
                    # to get the smallest pvalues
                    if pvalue < 0.05 and pvalue != 0:
                        pairsDict[(i,j)] = pvalue
        
        # sort the pairs by lowest p-value
        keys, values = list(pairsDict.keys()), list(pairsDict.values())
        sorted_value_index = np.argsort(values)
        sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

        # a stock should only be included once in our holdings to prevent mishaps
        new_dict, seen = {}, set()
        for (i, j) in sorted_dict:
            if i not in seen and j not in seen:
                new_dict[(i, j)] = sorted_dict[(i,j)]
                seen.add(i)
                seen.add(j)
        
        # obtain top npairs pairs if possible
        pairs = list(new_dict.keys()) if len(new_dict) < self.npairs else list((new_dict.keys()))[0:self.npairs]
        return pairs
    
    def generate_day(self, day, data):
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
      
        data = pd.DataFrame(data['close']).dropna()
        for p in self.pairs:
            # FIRST and SECOND are indices of the stocks
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            if z_score >= 1.0:
                self.holdings[FIRST] = -1
                self.holdings[SECOND] = 1
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0

            # enter the trade; long the FIRST, short SECOND
            elif z_score <= -1.0:
                self.holdings[FIRST] = 1
                self.holdings[SECOND] = -1
            
        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return op.weight(self.holdings)

class Enhanced_Stop_Random1(abstractalpha.AbstractAlpha):
    def __init__(self, reset, npairs, exit, stop):
        self.name = 'Enhanced Random Sample of Pairs with Stop - Fully Invested'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = True
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.stop = stop
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def form_pairs(self, df):
        df = pd.DataFrame(df).dropna()
        n = df.shape[1]
        
        # obtain transformed data and clusters using above functions
        transformed_data = self.reduce_dimensionality(df)
        labels = self.cluster_optics(transformed_data)

        # creating an adjacency matrix sort of for cointegration scores and pvalues
        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                    S1 = df[keys[i]]        
                    S2 = df[keys[j]]
                    result = coint(S1, S2)
                    pvalue = result[1]
                    pvalue_matrix[i, j] = pvalue
                    if pvalue < 0.05:
                        pairs.append([i, j])
        
        # a stock should only be included once in our holdings to prevent mishaps
        new_pairs, seen = [], set()
        for (i, j) in pairs:
            if i not in seen and j not in seen:
                new_pairs.append([i, j])
                seen.add(i)
                seen.add(j)

        new_pairs = new_pairs if len(new_pairs) < self.npairs else random.sample(new_pairs, self.npairs)
        return new_pairs
    
    def zscore(self, series: array_1d) -> float:
        return (series - series.mean()) / np.std(series)
    
    def reduce_dimensionality(self, df: pd.DataFrame) -> pd.DataFrame:
        rets = df.pct_change().dropna()
        rets = rets.T
        pca = PCA(n_components = 0.9, svd_solver='full')
        transformed_data = pca.fit_transform(rets)
        return transformed_data
        
    def cluster_optics(self, transformed_data: array_1d) -> array_1d:
        optics = OPTICS(min_samples = 2)
        labels = optics.fit_predict(transformed_data)
        return labels

    def generate_day(self, day, data):
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
      
        data = pd.DataFrame(data['close'])
        global ex_data
        if self.print:
            ex_data = data
            self.print = False

        for p in self.pairs:
            # FIRST and SECOND are indices of the stocks
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            if z_score >= 1.0 and z_score <= self.stop:
                self.holdings[FIRST] = -1
                self.holdings[SECOND] = 1
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0
            
            # ensure that the pair is still cointegrated, otherwise eat the losses before it gets worse
            elif abs(z_score) > self.stop:
                temp = data.dropna()
                S1, S2 = temp[FIRST], temp[SECOND]
                result = coint(S1, S2)
                pvalue = result[1]
                if pvalue >= 0.05:
                    self.holdings[FIRST] = 0
                    self.holdings[SECOND] = 0
                    self.pairs.remove(p)

            # enter the trade; long the FIRST, short SECOND
            elif z_score <= -1.0 and z_score >= -self.stop:
                self.holdings[FIRST] = 1
                self.holdings[SECOND] = -1
            
        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return op.weight(self.holdings)

class Enhanced_Stop_Lowest_PValue1(abstractalpha.AbstractAlpha):
    def __init__(self, reset, npairs, exit, stop):
        self.name = 'Enhanced Lowest PValue Pairs with Stop - Fully Invested'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = False
        self.keyPairs = []
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.stop = stop
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def zscore(self, series):
        return (series - series.mean()) / np.std(series)
    
    def form_pairs(self, df):
        df = pd.DataFrame(df).dropna()
        n = df.shape[1]

        # obtain transformed data and clusters using above functions
        transformed_data = self.reduce_dimensionality(df)
        labels = self.cluster_optics(transformed_data)

        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        pairsDict = {} #maps from pvalue to pair
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                    S1 = df[keys[i]]        
                    S2 = df[keys[j]]
                    result = coint(S1, S2)
                    pvalue = result[1]
                    pvalue_matrix[i, j] = pvalue
                    
                    # used a hashmap, sorted keys (pvalues) in ascending order in order
                    # to get the smallest pvalues
                    if pvalue < 0.05:
                        pairsDict[(i,j)] = pvalue
        
        # sort the pairs by lowest p-value
        keys, values = list(pairsDict.keys()), list(pairsDict.values())
        sorted_value_index = np.argsort(values)
        sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

        # a stock should only be included once in our holdings to prevent mishaps
        new_dict, seen = {}, set()
        for (i, j) in sorted_dict:
            if i not in seen and j not in seen:
                new_dict[(i, j)] = sorted_dict[(i,j)]
                seen.add(i)
                seen.add(j)
        
        # obtain top npairs pairs if possible
        pairs = list(new_dict.keys()) if len(new_dict) < self.npairs else list((new_dict.keys()))[0:self.npairs]
        return pairs
    
    def reduce_dimensionality(self, df: pd.DataFrame) -> pd.DataFrame:
        rets = df.pct_change().dropna()
        rets = rets.T
        pca = PCA(n_components = 0.9, svd_solver='full')
        transformed_data = pca.fit_transform(rets)
        return transformed_data
        
    def cluster_optics(self, transformed_data: array_1d) -> array_1d:
        optics = OPTICS(min_samples = 2)
        labels = optics.fit_predict(transformed_data)
        return labels

    def generate_day(self, day, data):
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
      
        data = pd.DataFrame(data['close'])
        for p in self.pairs:
            # FIRST and SECOND are indices of the stocks
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            if z_score >= 1.0 and z_score <= self.stop:
                self.holdings[FIRST] = -1
                self.holdings[SECOND] = 1
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0
            
            # ensure that the pair is still cointegrated, otherwise eat the losses before it gets worse
            elif abs(z_score) > self.stop:
                temp = data.dropna()
                S1, S2 = temp[FIRST], temp[SECOND]
                result = coint(S1, S2)
                pvalue = result[1]
                if pvalue >= 0.05:
                    self.holdings[FIRST] = 0
                    self.holdings[SECOND] = 0
                    self.pairs.remove(p)

            # enter the trade; long the FIRST, short SECOND
            elif z_score <= -1.0 and z_score >= -self.stop:
                self.holdings[FIRST] = 1
                self.holdings[SECOND] = -1
            
        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return op.weight(self.holdings)

"""
These alphas implement a Committed Capital Weighting Scheme instead of a 
Fully Invested Weighting Scheme, which we have implemented above. Throughout
testing, committed capital alphas performed significantly worse, probably
because pairs trading should already be low risk, so there's no need to 
even more reserved with our weighting.
"""

class Random_Noncolinear2(abstractalpha.AbstractAlpha):
    def __init__(self, reset, npairs, exit):
        self.name = 'Noncolinear Random Sample of Pairs - Committed Capital'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = True
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def form_pairs(self, df):
        df = pd.DataFrame(df).dropna()
        n = df.shape[1]
        
        # creating an adjacency matrix sort of for cointegration scores and pvalues
        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        for i in range(50):
            for j in range(i + 1, 50):
                S1 = df[keys[i]]        
                S2 = df[keys[j]]
                result = coint(S1, S2)
                pvalue = result[1]
                pvalue_matrix[i, j] = pvalue
                if pvalue < 0.05 and pvalue != 0:
                    pairs.append([i, j])
        
        # a stock should only be included once in our holdings to prevent mishaps
        new_pairs, seen = [], set()
        for (i, j) in pairs:
            if i not in seen and j not in seen:
                new_pairs.append([i, j])
                seen.add(i)
                seen.add(j)

        new_pairs = new_pairs if len(new_pairs) < self.npairs else random.sample(new_pairs, self.npairs)
        return new_pairs
    
    def zscore(self, series: array_1d) -> float:
        return (series - series.mean()) / np.std(series)
        
    def generate_day(self, day, data):
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
      
        data = pd.DataFrame(data['close'])
        if len(self.pairs) > 0:
            capital_per_pair = 1 / (len(self.pairs) * 2)
        
        for p in self.pairs:
            # FIRST and SECOND are indices of the stocks
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            if z_score >= 1.0:
                self.holdings[FIRST] = -capital_per_pair
                self.holdings[SECOND] = capital_per_pair
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0

            # enter the trade; long the FIRST, short SECOND
            elif z_score <= -1.0:
                self.holdings[FIRST] = capital_per_pair
                self.holdings[SECOND] = -capital_per_pair

        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return self.holdings

class Lowest_PValue2(abstractalpha.AbstractAlpha):
    def __init__(self, reset, npairs, exit):
        self.name = 'Lowest PValue Pairs - Committed Capital'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = True
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def form_pairs(self, df):
        df = pd.DataFrame(df).dropna()
        n = df.shape[1]
        
        # creating an adjacency matrix sort of for cointegration scores and pvalues
        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        pairsDict = {}
        for i in range(50):
            for j in range(i + 1, 50):
                S1 = df[keys[i]]        
                S2 = df[keys[j]]
                result = coint(S1, S2)
                pvalue = result[1]
                pvalue_matrix[i, j] = pvalue
                if pvalue < 0.05:
                    pairsDict[(i,j)] = pvalue
        
        # sort the pairs by lowest p-value and then obtain 10 pairs
        keys, values = list(pairsDict.keys()), list(pairsDict.values())
        sorted_value_index = np.argsort(values)
        sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

        # a stock should only be included once in our holdings to prevent mishaps
        new_dict, seen = {}, set()
        for (i, j) in sorted_dict:
            if i not in seen and j not in seen:
                new_dict[(i, j)] = sorted_dict[(i,j)]
                seen.add(i)
                seen.add(j)
        
        # obtain top npairs pairs if possible
        pairs = list(new_dict.keys()) if len(new_dict) < self.npairs else list((new_dict.keys()))[0:self.npairs]
        return pairs
    
    def zscore(self, series: array_1d) -> float:
        return (series - series.mean()) / np.std(series)
        
    def generate_day(self, day, data):
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
      
        data = pd.DataFrame(data['close'])
        if len(self.pairs) > 0:
            capital_per_pair = 1 / (len(self.pairs) * 2)
        
        for p in self.pairs:
            # FIRST and SECOND are indices of the stocks
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            if z_score >= 1.0:
                self.holdings[FIRST] = -capital_per_pair
                self.holdings[SECOND] = capital_per_pair
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0

            # enter the trade; long the FIRST, short SECOND
            elif z_score <= -1.0:
                self.holdings[FIRST] = capital_per_pair
                self.holdings[SECOND] = -capital_per_pair
            
        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return self.holdings

class Lowest_PValue_Noncolinear2(abstractalpha.AbstractAlpha):
    def __init__(self, reset, npairs, exit):
        self.name = 'Noncolinear Lowest PValue Pairs - Committed Capital'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = True
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def form_pairs(self, df):
        df = pd.DataFrame(df).dropna()
        n = df.shape[1]
        
        # creating an adjacency matrix sort of for cointegration scores and pvalues
        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        pairsDict = {}
        for i in range(50):
            for j in range(i + 1, 50):
                S1 = df[keys[i]]        
                S2 = df[keys[j]]
                result = coint(S1, S2)
                pvalue = result[1]
                pvalue_matrix[i, j] = pvalue
                if pvalue < 0.05 and pvalue != 0:
                    pairsDict[(i,j)] = pvalue
        
        # sort the pairs by lowest p-value and then obtain 10 pairs
        keys, values = list(pairsDict.keys()), list(pairsDict.values())
        sorted_value_index = np.argsort(values)
        sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

        # a stock should only be included once in our holdings to prevent mishaps
        new_dict, seen = {}, set()
        for (i, j) in sorted_dict:
            if i not in seen and j not in seen:
                new_dict[(i, j)] = sorted_dict[(i,j)]
                seen.add(i)
                seen.add(j)
        
        # obtain top npairs pairs if possible
        pairs = list(new_dict.keys()) if len(new_dict) < self.npairs else list((new_dict.keys()))[0:self.npairs]
        return pairs
    
    def zscore(self, series: array_1d) -> float:
        return (series - series.mean()) / np.std(series)
        
    def generate_day(self, day, data):
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
      
        data = pd.DataFrame(data['close'])
        if len(self.pairs) > 0:
            capital_per_pair = 1 / (len(self.pairs) * 2)
        
        for p in self.pairs:
            # FIRST and SECOND are indices of the stocks
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            if z_score >= 1.0:
                self.holdings[FIRST] = -capital_per_pair
                self.holdings[SECOND] = capital_per_pair
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0

            # enter the trade; long the FIRST, short SECOND
            elif z_score <= -1.0:
                self.holdings[FIRST] = capital_per_pair
                self.holdings[SECOND] = -capital_per_pair
            
        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return self.holdings

class Enhanced_Random2(abstractalpha.AbstractAlpha):
    def __init__(self, reset, npairs, exit):
        self.name = 'Enhanced Random Sample of Pairs - Committed Capital'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = True
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def reduce_dimensionality(self, df: pd.DataFrame) -> pd.DataFrame:
        rets = df.pct_change().dropna()
        rets = rets.T
        pca = PCA(n_components = 0.9, svd_solver='full')
        transformed_data = pca.fit_transform(rets)
        return transformed_data
        
    def cluster_optics(self, transformed_data: array_1d) -> array_1d:
        optics = OPTICS(min_samples = 2)
        labels = optics.fit_predict(transformed_data)
        return labels

    def form_pairs(self, df):
        df = pd.DataFrame(df).dropna()
        n = df.shape[1]
        
        # obtain transformed data and clusters using above functions
        transformed_data = self.reduce_dimensionality(df)
        labels = self.cluster_optics(transformed_data)

        # creating an adjacency matrix sort of for cointegration scores and pvalues
        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                    S1 = df[keys[i]]        
                    S2 = df[keys[j]]
                    result = coint(S1, S2)
                    pvalue = result[1]
                    pvalue_matrix[i, j] = pvalue
                    if pvalue < 0.05:
                        pairs.append([i, j])
        
        # a stock should only be included once in our holdings to prevent mishaps
        new_pairs, seen = [], set()
        for (i, j) in pairs:
            if i not in seen and j not in seen:
                new_pairs.append([i, j])
                seen.add(i)
                seen.add(j)

        new_pairs = new_pairs if len(new_pairs) < self.npairs else random.sample(new_pairs, self.npairs)
        return new_pairs
    
    def zscore(self, series: array_1d) -> float:
        return (series - series.mean()) / np.std(series)
        
    def generate_day(self, day, data):
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
      
        data = pd.DataFrame(data['close']).dropna()
        if len(self.pairs) > 0:
            capital_per_pair = 1 / (len(self.pairs) * 2)
        
        for p in self.pairs:
            # FIRST and SECOND are indices of the stocks
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            if z_score >= 1.0:
                self.holdings[FIRST] = -capital_per_pair
                self.holdings[SECOND] = capital_per_pair
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0

            # enter the trade; long the FIRST, short SECOND
            elif z_score <= -1.0:
                self.holdings[FIRST] = capital_per_pair
                self.holdings[SECOND] = -capital_per_pair

        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return self.holdings

class Enhanced_Random_Noncolinear2(abstractalpha.AbstractAlpha):
    def __init__(self, reset, npairs, exit):
        self.name = 'Noncolinear Enhanced Random Sample of Pairs - Committed Capital'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = True
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def reduce_dimensionality(self, df: pd.DataFrame) -> pd.DataFrame:
        rets = df.pct_change().dropna()
        rets = rets.T
        pca = PCA(n_components = 0.9, svd_solver='full')
        transformed_data = pca.fit_transform(rets)
        return transformed_data
        
    def cluster_optics(self, transformed_data: array_1d) -> array_1d:
        optics = OPTICS(min_samples = 2)
        labels = optics.fit_predict(transformed_data)
        return labels

    def form_pairs(self, df):
        df = pd.DataFrame(df).dropna()
        n = df.shape[1]
        
        # obtain transformed data and clusters using above functions
        transformed_data = self.reduce_dimensionality(df)
        labels = self.cluster_optics(transformed_data)

        # creating an adjacency matrix sort of for cointegration scores and pvalues
        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                    S1 = df[keys[i]]        
                    S2 = df[keys[j]]
                    result = coint(S1, S2)
                    pvalue = result[1]
                    pvalue_matrix[i, j] = pvalue
                    if pvalue < 0.05 and pvalue != 0:
                        pairs.append([i, j])
        
        # a stock should only be included once in our holdings to prevent mishaps
        new_pairs, seen = [], set()
        for (i, j) in pairs:
            if i not in seen and j not in seen:
                new_pairs.append([i, j])
                seen.add(i)
                seen.add(j)

        new_pairs = new_pairs if len(new_pairs) < self.npairs else random.sample(new_pairs, self.npairs)
        return new_pairs
    
    def zscore(self, series: array_1d) -> float:
        return (series - series.mean()) / np.std(series)
        
    def generate_day(self, day, data):
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
      
        data = pd.DataFrame(data['close']).dropna()
        if len(self.pairs) > 0:
            capital_per_pair = 1 / (len(self.pairs) * 2)
        
        for p in self.pairs:
            # FIRST and SECOND are indices of the stocks
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            if z_score >= 1.0:
                self.holdings[FIRST] = -capital_per_pair
                self.holdings[SECOND] = capital_per_pair
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0

            # enter the trade; long the FIRST, short SECOND
            elif z_score <= -1.0:
                self.holdings[FIRST] = capital_per_pair
                self.holdings[SECOND] = -capital_per_pair

        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return self.holdings

class Enhanced_Lowest_PValue2(abstractalpha.AbstractAlpha):
    def __init__(self, reset, npairs, exit):
        self.name = 'Enhanced Lowest PValue Pairs - Committed Capital'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = True
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def reduce_dimensionality(self, df: pd.DataFrame) -> pd.DataFrame:
        rets = df.pct_change().dropna()
        rets = rets.T
        pca = PCA(n_components = 0.9, svd_solver='full')
        transformed_data = pca.fit_transform(rets)
        return transformed_data
        
    def cluster_optics(self, transformed_data: array_1d) -> array_1d:
        optics = OPTICS(min_samples = 2)
        labels = optics.fit_predict(transformed_data)
        return labels

    def form_pairs(self, df):
        df = pd.DataFrame(df).dropna()
        n = df.shape[1]
        
        # obtain transformed data and clusters using above functions
        transformed_data = self.reduce_dimensionality(df)
        labels = self.cluster_optics(transformed_data)

        # creating an adjacency matrix sort of for cointegration scores and pvalues
        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        pairsDict = {}
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                    S1 = df[keys[i]]        
                    S2 = df[keys[j]]
                    result = coint(S1, S2)
                    pvalue = result[1]
                    pvalue_matrix[i, j] = pvalue
                    if pvalue < 0.05:
                        pairsDict[(i,j)] = pvalue
        
        # sort the pairs by lowest p-value and then obtain 10 pairs
        keys, values = list(pairsDict.keys()), list(pairsDict.values())
        sorted_value_index = np.argsort(values)
        sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

        # a stock should only be included once in our holdings to prevent mishaps
        new_dict, seen = {}, set()
        for (i, j) in sorted_dict:
            if i not in seen and j not in seen:
                new_dict[(i, j)] = sorted_dict[(i,j)]
                seen.add(i)
                seen.add(j)
        
        # obtain top npairs pairs if possible
        pairs = list(new_dict.keys()) if len(new_dict) < self.npairs else list((new_dict.keys()))[0:self.npairs]
        return pairs
    
    def zscore(self, series: array_1d):
        return (series - series.mean()) / np.std(series)
        
    def generate_day(self, day, data):
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
      
        data = pd.DataFrame(data['close']).dropna()
        if len(self.pairs) > 0:
            capital_per_pair = 1 / (len(self.pairs) * 2)
        
        for p in self.pairs:
            # FIRST and SECOND are indices of the stocks
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            if z_score >= 1.0:
                self.holdings[FIRST] = -capital_per_pair
                self.holdings[SECOND] = capital_per_pair
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0

            # enter the trade; long the FIRST, short SECOND
            elif z_score <= -1.0:
                self.holdings[FIRST] = capital_per_pair
                self.holdings[SECOND] = -capital_per_pair
            
        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return self.holdings

class Enhanced_Lowest_PValue_Noncolinear2(abstractalpha.AbstractAlpha):
    def __init__(self, reset, npairs, exit):
        self.name = 'Noncolinear Enhanced Lowest PValue Pairs - Committed Capital'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = True
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def reduce_dimensionality(self, df: pd.DataFrame) -> pd.DataFrame:
        rets = df.pct_change().dropna()
        rets = rets.T
        pca = PCA(n_components = 0.9, svd_solver='full')
        transformed_data = pca.fit_transform(rets)
        return transformed_data
        
    def cluster_optics(self, transformed_data: array_1d) -> array_1d:
        optics = OPTICS(min_samples = 2)
        labels = optics.fit_predict(transformed_data)
        return labels

    def form_pairs(self, df):
        df = pd.DataFrame(df).dropna()
        n = df.shape[1]
        
        # obtain transformed data and clusters using above functions
        transformed_data = self.reduce_dimensionality(df)
        labels = self.cluster_optics(transformed_data)

        # creating an adjacency matrix sort of for cointegration scores and pvalues
        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        pairsDict = {}
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                    S1 = df[keys[i]]        
                    S2 = df[keys[j]]
                    result = coint(S1, S2)
                    pvalue = result[1]
                    pvalue_matrix[i, j] = pvalue
                    if pvalue < 0.05 and pvalue != 0:
                        pairsDict[(i,j)] = pvalue
        
        # sort the pairs by lowest p-value and then obtain 10 pairs
        keys, values = list(pairsDict.keys()), list(pairsDict.values())
        sorted_value_index = np.argsort(values)
        sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

        # a stock should only be included once in our holdings to prevent mishaps
        new_dict, seen = {}, set()
        for (i, j) in sorted_dict:
            if i not in seen and j not in seen:
                new_dict[(i, j)] = sorted_dict[(i,j)]
                seen.add(i)
                seen.add(j)
        
        # obtain top npairs pairs if possible
        pairs = list(new_dict.keys()) if len(new_dict) < self.npairs else list((new_dict.keys()))[0:self.npairs]
        return pairs
    
    def zscore(self, series: array_1d):
        return (series - series.mean()) / np.std(series)
        
    def generate_day(self, day, data):
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
      
        data = pd.DataFrame(data['close']).dropna()
        if len(self.pairs) > 0:
            capital_per_pair = 1 / (len(self.pairs) * 2)
        
        for p in self.pairs:
            # FIRST and SECOND are indices of the stocks
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            if z_score >= 1.0:
                self.holdings[FIRST] = -capital_per_pair
                self.holdings[SECOND] = capital_per_pair
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0

            # enter the trade; long the FIRST, short SECOND
            elif z_score <= -1.0:
                self.holdings[FIRST] = capital_per_pair
                self.holdings[SECOND] = -capital_per_pair
            
        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return self.holdings

class Enhanced_Stop_Random2(abstractalpha.AbstractAlpha):
    def __init__(self, reset, npairs, exit, stop):
        self.name = 'Enhanced Random Sample of Pairs with Stop - Committed Capital'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = True
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.stop = stop
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def form_pairs(self, df):
        df = pd.DataFrame(df).dropna()
        n = df.shape[1]
        
        # obtain transformed data and clusters using above functions
        transformed_data = self.reduce_dimensionality(df)
        labels = self.cluster_optics(transformed_data)

        # creating an adjacency matrix sort of for cointegration scores and pvalues
        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                    S1 = df[keys[i]]        
                    S2 = df[keys[j]]
                    result = coint(S1, S2)
                    pvalue = result[1]
                    pvalue_matrix[i, j] = pvalue
                    if pvalue < 0.05:
                        pairs.append([i, j])
        
        # a stock should only be included once in our holdings to prevent mishaps
        new_pairs, seen = [], set()
        for (i, j) in pairs:
            if i not in seen and j not in seen:
                new_pairs.append([i, j])
                seen.add(i)
                seen.add(j)

        new_pairs = new_pairs if len(new_pairs) < self.npairs else random.sample(new_pairs, self.npairs)
        return new_pairs
    
    def zscore(self, series: array_1d) -> float:
        return (series - series.mean()) / np.std(series)
    
    def reduce_dimensionality(self, df: pd.DataFrame) -> pd.DataFrame:
        rets = df.pct_change().dropna()
        rets = rets.T
        pca = PCA(n_components = 0.9, svd_solver='full')
        transformed_data = pca.fit_transform(rets)
        return transformed_data
        
    def cluster_optics(self, transformed_data: array_1d) -> array_1d:
        optics = OPTICS(min_samples = 2)
        labels = optics.fit_predict(transformed_data)
        return labels

    def generate_day(self, day, data):
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
      
        data = pd.DataFrame(data['close'])
        if len(self.pairs) > 0:
            capital_per_pair = 1 / (len(self.pairs) * 2)
        
        for p in self.pairs:
            # FIRST and SECOND are indices of the stocks
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            if z_score >= 1.0 and z_score <= self.stop:
                self.holdings[FIRST] = -capital_per_pair
                self.holdings[SECOND] = capital_per_pair
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0
            
            # ensure that the pair is still cointegrated, otherwise eat the losses before it gets worse
            elif abs(z_score) > self.stop:
                temp = data.dropna()
                S1, S2 = temp[FIRST], temp[SECOND]
                result = coint(S1, S2)
                pvalue = result[1]
                if pvalue >= 0.05:
                    self.holdings[FIRST] = 0
                    self.holdings[SECOND] = 0
                    self.pairs.remove(p)

            # enter the trade; long the FIRST, short SECOND
            elif z_score <= -1.0 and z_score >= -self.stop:
                self.holdings[FIRST] = capital_per_pair
                self.holdings[SECOND] = -capital_per_pair

        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return self.holdings

class Enhanced_Noncolinear_Stop_Random2(abstractalpha.AbstractAlpha):
    def __init__(self, reset, npairs, exit, stop):
        self.name = 'Enhanced Noncolinear Random Sample of Pairs with Stop - Committed Capital'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = True
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.stop = stop
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def form_pairs(self, df):
        df = pd.DataFrame(df).dropna()
        n = df.shape[1]
        
        # obtain transformed data and clusters using above functions
        transformed_data = self.reduce_dimensionality(df)
        labels = self.cluster_optics(transformed_data)

        # creating an adjacency matrix sort of for cointegration scores and pvalues
        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                    S1 = df[keys[i]]        
                    S2 = df[keys[j]]
                    result = coint(S1, S2)
                    pvalue = result[1]
                    pvalue_matrix[i, j] = pvalue
                    if pvalue < 0.05 and pvalue != 0:
                        pairs.append([i, j])
        
        # a stock should only be included once in our holdings to prevent mishaps
        new_pairs, seen = [], set()
        for (i, j) in pairs:
            if i not in seen and j not in seen:
                new_pairs.append([i, j])
                seen.add(i)
                seen.add(j)

        new_pairs = new_pairs if len(new_pairs) < self.npairs else random.sample(new_pairs, self.npairs)
        return new_pairs
    
    def zscore(self, series: array_1d) -> float:
        return (series - series.mean()) / np.std(series)
    
    def reduce_dimensionality(self, df: pd.DataFrame) -> pd.DataFrame:
        rets = df.pct_change().dropna()
        rets = rets.T
        pca = PCA(n_components = 0.9, svd_solver='full')
        transformed_data = pca.fit_transform(rets)
        return transformed_data
        
    def cluster_optics(self, transformed_data: array_1d) -> array_1d:
        optics = OPTICS(min_samples = 2)
        labels = optics.fit_predict(transformed_data)
        return labels

    def generate_day(self, day, data):
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
      
        data = pd.DataFrame(data['close'])
        if len(self.pairs) > 0:
            capital_per_pair = 1 / (len(self.pairs) * 2)
        
        for p in self.pairs:
            # FIRST and SECOND are indices of the stocks
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            if z_score >= 1.0 and z_score <= self.stop:
                self.holdings[FIRST] = -capital_per_pair
                self.holdings[SECOND] = capital_per_pair
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0
            
            # ensure that the pair is still cointegrated, otherwise eat the losses before it gets worse
            elif abs(z_score) > self.stop:
                temp = data.dropna()
                S1, S2 = temp[FIRST], temp[SECOND]
                result = coint(S1, S2)
                pvalue = result[1]
                if pvalue >= 0.05:
                    self.holdings[FIRST] = 0
                    self.holdings[SECOND] = 0
                    self.pairs.remove(p)

            # enter the trade; long the FIRST, short SECOND
            elif z_score <= -1.0 and z_score >= -self.stop:
                self.holdings[FIRST] = capital_per_pair
                self.holdings[SECOND] = -capital_per_pair

        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return self.holdings

class Enhanced_Stop_Lowest_PValue2(abstractalpha.AbstractAlpha):
    def __init__(self, reset, npairs, exit, stop):
        self.name = 'Enhanced Lowest PValue Pairs with Stop - Committed Capital'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = True
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.stop = stop
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def form_pairs(self, df):
        df = pd.DataFrame(df).dropna()
        n = df.shape[1]
        
        # obtain transformed data and clusters using above functions
        transformed_data = self.reduce_dimensionality(df)
        labels = self.cluster_optics(transformed_data)

        # creating an adjacency matrix sort of for cointegration scores and pvalues
        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        pairsDict = {}
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                    S1 = df[keys[i]]        
                    S2 = df[keys[j]]
                    result = coint(S1, S2)
                    pvalue = result[1]
                    pvalue_matrix[i, j] = pvalue
                    if pvalue < 0.05:
                        pairsDict[(i,j)] = pvalue
        
        # sort the pairs by lowest p-value and then obtain 10 pairs
        keys, values = list(pairsDict.keys()), list(pairsDict.values())
        sorted_value_index = np.argsort(values)
        sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

        # a stock should only be included once in our holdings to prevent mishaps
        new_dict, seen = {}, set()
        for (i, j) in sorted_dict:
            if i not in seen and j not in seen:
                new_dict[(i, j)] = sorted_dict[(i,j)]
                seen.add(i)
                seen.add(j)
        
        # obtain top npairs pairs if possible
        pairs = list(new_dict.keys()) if len(new_dict) < self.npairs else list((new_dict.keys()))[0:self.npairs]
        return pairs
    
    def zscore(self, series: array_1d) -> float:
        return (series - series.mean()) / np.std(series)
    
    def reduce_dimensionality(self, df: pd.DataFrame) -> pd.DataFrame:
        rets = df.pct_change().dropna()
        rets = rets.T
        pca = PCA(n_components = 0.9, svd_solver='full')
        transformed_data = pca.fit_transform(rets)
        return transformed_data
        
    def cluster_optics(self, transformed_data: array_1d) -> array_1d:
        optics = OPTICS(min_samples = 2)
        labels = optics.fit_predict(transformed_data)
        return labels

    def generate_day(self, day, data):
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
      
        data = pd.DataFrame(data['close'])
        if len(self.pairs) > 0:
            capital_per_pair = 1 / (len(self.pairs) * 2)
        
        for p in self.pairs:
            # FIRST and SECOND are indices of the stocks
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            if z_score >= 1.0 and z_score <= self.stop:
                self.holdings[FIRST] = -capital_per_pair
                self.holdings[SECOND] = capital_per_pair
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0
            
            # ensure that the pair is still cointegrated, otherwise eat the losses before it gets worse
            elif abs(z_score) > self.stop:
                temp = data.dropna()
                S1, S2 = temp[FIRST], temp[SECOND]
                result = coint(S1, S2)
                pvalue = result[1]
                if pvalue >= 0.05:
                    self.holdings[FIRST] = 0
                    self.holdings[SECOND] = 0
                    self.pairs.remove(p)

            # enter the trade; long the FIRST, short SECOND
            elif z_score <= -1.0 and z_score >= -self.stop:
                self.holdings[FIRST] = capital_per_pair
                self.holdings[SECOND] = -capital_per_pair
            
        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return self.holdings

class Enhanced_Noncolinear_Stop_Lowest_PValue2(abstractalpha.AbstractAlpha):
    def __init__(self, reset, npairs, exit, stop):
        self.name = 'Enhanced Noncolinear Lowest PValue Pairs with Stop - Committed Capital'
        self.lookback = reset
        self.factor_list = ['close']
        self.universe_size = 500
        
        self.pairs = None
        self.print = True
        self.reset = reset
        self.npairs = npairs
        self.exit = exit
        self.stop = stop
        self.holdings = np.zeros(self.universe_size)
        self.day_counter = 0
        self.count = 0
    
    def form_pairs(self, df):
        df = pd.DataFrame(df).dropna()
        n = df.shape[1]
        
        # obtain transformed data and clusters using above functions
        transformed_data = self.reduce_dimensionality(df)
        labels = self.cluster_optics(transformed_data)

        # creating an adjacency matrix sort of for cointegration scores and pvalues
        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        pairsDict = {}
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                    S1 = df[keys[i]]        
                    S2 = df[keys[j]]
                    result = coint(S1, S2)
                    pvalue = result[1]
                    pvalue_matrix[i, j] = pvalue
                    if pvalue < 0.05 and pvalue != 0:
                        pairsDict[(i,j)] = pvalue
        
        # sort the pairs by lowest p-value and then obtain 10 pairs
        keys, values = list(pairsDict.keys()), list(pairsDict.values())
        sorted_value_index = np.argsort(values)
        sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

        # a stock should only be included once in our holdings to prevent mishaps
        new_dict, seen = {}, set()
        for (i, j) in sorted_dict:
            if i not in seen and j not in seen:
                new_dict[(i, j)] = sorted_dict[(i,j)]
                seen.add(i)
                seen.add(j)
        
        # obtain top npairs pairs if possible
        pairs = list(new_dict.keys()) if len(new_dict) < self.npairs else list((new_dict.keys()))[0:self.npairs]
        return pairs
    
    def zscore(self, series: array_1d) -> float:
        return (series - series.mean()) / np.std(series)
    
    def reduce_dimensionality(self, df: pd.DataFrame) -> pd.DataFrame:
        rets = df.pct_change().dropna()
        rets = rets.T
        pca = PCA(n_components = 0.9, svd_solver='full')
        transformed_data = pca.fit_transform(rets)
        return transformed_data
        
    def cluster_optics(self, transformed_data: array_1d) -> array_1d:
        optics = OPTICS(min_samples = 2)
        labels = optics.fit_predict(transformed_data)
        return labels

    def generate_day(self, day, data):
        
        # creating new pairs
        if self.day_counter == 0:
            self.day_counter = self.reset
            self.holdings = np.zeros(self.universe_size)
            self.pairs = self.form_pairs(data['close'])
      
        data = pd.DataFrame(data['close'])
        if len(self.pairs) > 0:
            capital_per_pair = 1 / (len(self.pairs) * 2)
        
        for p in self.pairs:
            # FIRST and SECOND are indices of the stocks
            FIRST, SECOND = p[0], p[1]
            spread = data[FIRST] - data[SECOND]
            
            #zscore tells us how far from away from the mean a data point is
            z_score = self.zscore(spread).tail(1).values[0]
            
            if z_score >= 1.0 and z_score <= self.stop:
                self.holdings[FIRST] = -capital_per_pair
                self.holdings[SECOND] = capital_per_pair
                
            # exit the trade
            elif abs(z_score) <= self.exit:
                self.holdings[FIRST] = 0
                self.holdings[SECOND] = 0
            
            # ensure that the pair is still cointegrated, otherwise eat the losses before it gets worse
            elif abs(z_score) > self.stop:
                temp = data.dropna()
                S1, S2 = temp[FIRST], temp[SECOND]
                result = coint(S1, S2)
                pvalue = result[1]
                if pvalue >= 0.05:
                    self.holdings[FIRST] = 0
                    self.holdings[SECOND] = 0
                    self.pairs.remove(p)

            # enter the trade; long the FIRST, short SECOND
            elif z_score <= -1.0 and z_score >= -self.stop:
                self.holdings[FIRST] = capital_per_pair
                self.holdings[SECOND] = -capital_per_pair
            
        # at the end of the trading day, decrement day_counter
        self.day_counter -= 1
        return self.holdings










































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































