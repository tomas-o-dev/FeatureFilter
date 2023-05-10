import numpy as np
import pandas as pd
from abc import *
from typing import Tuple, List, Dict, Any

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed

# thanks to : https://github.com/Anylee2142/


class Discretizer(BaseEstimator, TransformerMixin, metaclass=ABCMeta):
    def __init__(self, numjobs= -2, msglvl=5):

        self.cutpoints = dict()
        self.numjobs=numjobs
        self.msglvl=msglvl


    def fit(self, X, y):
        '''
        call the preprocessor
        :param X: pd.DataFrame with features
        :param y: pd.Series with target
        :return: self
        '''
        from .d_ppdf import prep_df

## log10(32) and sqrt(7) are first to give num_bins > 2
        self.prebin_df, self.flist, self.vi, self.xxxi = prep_df(X)

        if isinstance(y, np.ndarray):
            sy = pd.Series(data=y, name='target')
        else:
            sy = y

        if sy.dtype == 'O':
            sy = pd.Series(data=LabelEncoder().fit_transform(sy),name=sy.name)

        self.sy = sy

        return self


    def transform(self, mkbins='hgrm', detail=False):
        '''
        Get cutpoints, then discretize each feature
        :detail: report outcome of binning
        :return: discretized X
        '''
        from .d_nbhg import hgbins

        if mkbins in ['unif-ten', 'unif-log', 'unif-sqrt', 
                      'mdlp-ten', 'mdlp-log', 'mdlp-sqrt', 
                      'chim-ten', 'chim-log', 'chim-sqrt']:
            self.mkbins = mkbins
        else:
            self.mkbins='hgrm'

## log10(32) and sqrt(7) are first to give num_bins > 2
        col_list = [x for x in self.flist]
        lowuniq = [x for x in self.vi]
        if self.mkbins[5:] == 'log':
            lowuniq.extend(x for x in self.xxxi)
        else:
            col_list.extend(x for x in self.xxxi)

        self.cutpoints = self._prep_get_cutpoints(col_list)

        if len(lowuniq):
            print('\n Few Unique Values - Always binned with = hgrm =')
            for feature_name in lowuniq:
                print('\t',feature_name,':: Unique Values =',len(self.prebin_df[feature_name].unique()))
                feature = self.prebin_df[feature_name].values
                bin_edges = hgbins(feature)
                cutpoints = bin_edges[1:-1].tolist()
                self.cutpoints.update({feature_name: cutpoints})

        self.binned_df = self.prebin_df.copy()
        for feature_name, cutpoints in self.cutpoints.items():
            if len(cutpoints) < 2:
                print('\nWARNING: No',self.mkbins,'cutpoints could be calculated for',feature_name)
                print('           Falling back to = hgrm =')

                feature = self.prebin_df[feature_name].values
                bin_edges = hgbins(feature)
                cutpoints = bin_edges[1:-1].tolist()

            self.binned_df[feature_name] = pd.Series(
                data=np.searchsorted(cutpoints, self.prebin_df.loc[:, feature_name]),
                name=feature_name)

        self.binned_df.reset_index(inplace = True, drop = True)

        if detail:
            colwid = max(len(n) for n in self.binned_df.columns.values)    # +i for padding
            print('\nUnique value count: Original ::> Binned with',self.mkbins,'\n  Same for all features except:')
            for col in self.binned_df.columns:
                lcx = len(self.prebin_df[col].unique())
                lcb = len(self.binned_df[col].unique())
                if lcx != lcb:
                    print("{: <{colwid}} {: >5} {} {: <5}".format(
                        col,lcx,'::>',lcb,colwid=colwid))

        return self.binned_df

##  --  --  --  --  ##

    def fit_transform(self, X, y, mkbins='hgrm', detail=False):
        return self.fit(X, y).transform(mkbins, detail)

##  --  --  --  --  ##

    def _prep_get_cutpoints(self, col_list):
        '''
        Get cutpoints from every continuous feature
        :features, col_list: pd.Dataframe as col_list
        :target_: pd.Series with target
        :return: cutpoints for every continuous feature {feature name: List of cutpoints}
        '''
        features = self.prebin_df
        target = self.sy

        numjobs = self.numjobs
        msglvl = self.msglvl

        if self.mkbins[:4] in ['unif', 'hgrm']:
            from .d_nbhg import get_cutpoints
            numjobs = 1
            msglvl = 0
        elif self.mkbins[:4] == 'chim':
            from .d_chim import get_cutpoints
        elif self.mkbins[:4] == 'mdlp':
            from .d_mdlp import get_cutpoints

        parallel = Parallel(n_jobs=numjobs, verbose=msglvl)

        work = parallel( 
            delayed(get_cutpoints)(
                self.mkbins[5:],
                feature=features.loc[:, feature_name],
                target=target,
                feature_name=feature_name
            ) 
            for feature_name in col_list
        )

        cutpoints_each_feature = {k: v for (k, v) in work} 
        return cutpoints_each_feature



