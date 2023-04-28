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


    def fit(self, X, y, numjobs, msglvl, detail=False):
        '''
        Get cutpoints, then discretize each feature
        :param X: pd.DataFrame with features
        :param y: pd.Series with target
        :n_jobs, verbose: joblib
        :return: self
        '''

        prebin_df, col_list = self._prep_df(X)

        if isinstance(y, np.ndarray):
            sy = pd.Series(data=y, name='target')
        else:
            sy = y

        if sy.dtype == 'O':
            sy = pd.Series(data=LabelEncoder().fit_transform(sy),name=sy.name)

        self.cutpoints = self._prep_get_cutpoints(prebin_df, sy, col_list, numjobs, msglvl)

        self.binned_df = X.copy()
        for feature_name, cutpoints in self.cutpoints.items():
            if len(cutpoints) < 2:
                print('\nWARNING: No',self.mkbins,'cutpoints could be calculated for',feature_name)
                print('           Falling back to mkbins=ten')

                feature = X[feature_name]
                bin_edges = np.linspace(feature.min(), feature.max(), 11)
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


    def fit_transform(self, X, y, numjobs, msglvl, detail=False):
        return self.fit(X, y, numjobs, msglvl).transform(X, detail=detail)


    @abstractmethod
    def _get_cutpoints(self):
        '''
        This method should describe how to get cutpoints according to each algorithm
        '''
        raise NotImplementedError


    def _prep_get_cutpoints(self, features, target, col_list, numjobs, msglvl):
        '''
        Get cutpoints from every continuous feature
        :param features, col_list: pd.Dataframe as col_list
        :param target_: pd.Series with target
        :return: cutpoints for every continuous feature {feature name: List of cutpoints}
        '''
        parallel = Parallel(n_jobs=numjobs, verbose=msglvl)

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


    def _prep_df(self, indf):
        print('Using only numeric datatypes',end='')
        dzdf = indf.select_dtypes(
            include=['number','bool','boolean'],exclude=['timedelta64','complexfloating']).copy()

        dxc=[]
        dxc.extend(x for x in indf.columns.values if x not in dzdf.columns.values)
        if len(dxc) > 0:
            print('\nDropping ineligible features (text features should be one-hot encoded)')
            for c in range(len(dxc)):
                print('\t',dxc[c],' [ dtype =',indf.dtypes[dxc[c]].name,']')
        else:
            print(' [ All features are in ]')

# all signed numbers
        sn = dzdf.select_dtypes(
            include=['signedinteger','floating'],exclude=['timedelta64']).columns.values                    
        for c in range(len(sn)):
            if dzdf[sn[c]].min() < 0:
                dzdf[sn[c]] += abs(dzdf[sn[c]].min())

# all pyTF columns
        pytf = dzdf.select_dtypes(include=['bool','boolean']).columns.values
        for c in range(len(pytf)):
            dzdf[sn[c]] = dzdf[sn[c]].replace({True: 1, False: 0})

# list of column names for get_cutpoints       
        ftc=[]

        lblenc = LabelEncoder()
        for col in dzdf.columns:
            uv = len(dzdf[col].unique())
            if uv == 1:
                print('WARNING: Dropping single-valued feature', dzdf[col].name)
                dzdf.drop(dzdf[col].name, axis=1, inplace=True)
            elif uv == 2: 
                if (dzdf[col].min() != 0) or (dzdf[col].max() != 1):
                    dzdf[col] = dzdf[col].replace({dzdf[col].max(): 1, dzdf[col].min(): 0})
            else:
                ftc.append(dzdf[col].name)                

        dzdf.reset_index(inplace = True, drop = True)
        return dzdf, ftc



