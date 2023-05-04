import numpy as np
import pandas as pd
from abc import *
from typing import Tuple, List, Dict, Any

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed

# thanks to : https://github.com/Anylee2142/


class Discretizer(BaseEstimator, TransformerMixin, metaclass=ABCMeta):
    def __init__(self, numjobs=1, msglvl=50):

        self.cutpoints = dict()
        self.numjobs=numjobs
        self.msglvl=msglvl


    def fit(self, X, y, detail=False):
        '''
        Get cutpoints, then discretize each feature
        :param X: pd.DataFrame with features
        :param y: pd.Series with target
        :n_jobs, verbose: joblib
        :return: self
        '''
        from .d_ppdf import prep_df
        from .d_nbhg import hgbins

        prebin_df, col_list, lowuniq = prep_df(X)

        if isinstance(y, np.ndarray):
            sy = pd.Series(data=y, name='target')
        else:
            sy = y

        if sy.dtype == 'O':
            sy = pd.Series(data=LabelEncoder().fit_transform(sy),name=sy.name)

        self.cutpoints = self._prep_get_cutpoints(prebin_df, sy, col_list)

# log10(32) and sqrt(7) are first to give num_bins > 2
# TODO with refactoring - uv < 32 for log; uv < 7 for sqrt, ten 
        print('\nUnique Values < 32 - Always binned with =hgrm=')
        for feature_name in lowuniq:
            print('\t',feature_name,':: Unique Values =',len(prebin_df[feature_name].unique()))
            feature = prebin_df[feature_name].values
            bin_edges = hgbins(feature)
            cutpoints = bin_edges[1:-1].tolist()
            self.cutpoints.update({feature_name: cutpoints})

        self.binned_df = prebin_df.copy()
        for feature_name, cutpoints in self.cutpoints.items():
            if len(cutpoints) < 2:
                print('\nWARNING: No',self.mkbins,'cutpoints could be calculated for',feature_name)
                print('           Falling back to =hgrm=')

                feature = X[feature_name].values
                bin_edges = hgbins(feature)
                cutpoints = bin_edges[1:-1].tolist()

            self.binned_df[feature_name] = pd.Series(
                data=np.searchsorted(cutpoints, X.loc[:, feature_name]),
                name=feature_name)

        self.binned_df.reset_index(inplace = True, drop = True)
        return self


    def transform(self, X, detail=False):
        '''
        :detail: report outcome of binning
        :return: discretized X
        '''
        if detail:
            colwid = max(len(n) for n in self.binned_df.columns.values)    # +i for padding
            print('\nUnique value count: Original ::> Binned\n  Same for all features except:')
            for col in self.binned_df.columns:
                lcx = len(X[col].unique())
                lcb = len(self.binned_df[col].unique())
                if lcx != lcb:
                    print("{: <{colwid}} {: >5} {} {: <5}".format(
                        col,lcx,'::>',lcb,colwid=colwid))

        return self.binned_df


    def fit_transform(self, X, y, detail=False):
        return self.fit(X, y).transform(X, detail=detail)


    @abstractmethod
    def _get_cutpoints(self):
        '''
        This method should describe how to get cutpoints according to each algorithm
        '''
        raise NotImplementedError


    def _prep_get_cutpoints(self, features, target, col_list):
        '''
        Get cutpoints from every continuous feature
        :param features, col_list: pd.Dataframe as col_list
        :param target_: pd.Series with target
        :return: cutpoints for every continuous feature {feature name: List of cutpoints}
        '''
        parallel = Parallel(n_jobs=self.numjobs, verbose=self.msglvl)

        work = parallel( 
            delayed(self._get_cutpoints)(
                feature=features.loc[:, feature_name],
                target=target,
                feature_name=feature_name
            ) 
            for feature_name in col_list
        )

        cutpoints_each_feature = {k: v for (k, v) in work} 
        return cutpoints_each_feature



