import numpy as np
import pandas as pd
from abc import *
from typing import Tuple, List, Dict, Any

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class qlcfFilter(BaseEstimator, TransformerMixin, metaclass=ABCMeta):
    def __init__(self):
        init = True

    def fit(self, X, y, filters, plvl=0.5, minpc=0.1, minsu=0.01, hipc=0.7, hisu=0.7):
        '''
        :param X: binned pd.DataFrame with features
        :param y: target labels
        :param filters in ['Floor', 'FDR', 'FWE', 'FCBF-SU', 'FCBF-PC']
               processed in order with progressive filtering
        optionals
            Floor: 
                minpc : threshold for pearson correlation    default=0.1
                minsu : threshold for symmetric uncertainty  default=0.01 
            FDR, FWE: 
                plvl : threshold (alpha) for chi_sq test  default=0.5
            FCBF-SU, FCBF-PC:
                hipc : threshold for "high" f2f pearson correlation  default=0.7
                hisu : threshold for "high" f2f su correlation       default=0.7
        :return: self
        '''
        from .f_fltr import filter_fcy, filter_fdr, filter_fcc
        from .f_rpts import get_filter

        try: 
            assert (isinstance(filters, list) or isinstance(filters, tuple)), "filters should be a list\n\tone or more of ['Floor', 'FDR', 'FWE', 'FCBF-SU', 'FCBF-PC']"
        except AssertionError as e: 
            print(e)
            self.flist = ['FDR', 'FWE']
        else:
            self.flist = filters

        binned_df = X.copy()
        binned_df.reset_index(inplace = True, drop = True)

        if isinstance(y, np.ndarray):
            sy = pd.Series(data=y, name='target')
        else:
            sy = y

        if sy.dtype == 'O':
            sy = pd.Series(data=LabelEncoder().fit_transform(sy),name=sy.name)

        self.QLCFFilter = []
        self.cyd = dict()
        self.cyk = dict()

#filters in ['Floor', 'FDR', 'FWE', 'FCBF-SU', 'FCBF-PC']

        print('Filters are applied in order:',self.flist)
        for f in range(len(self.flist)):
            filtr = self.flist[f]

            if filtr == 'Floor':
                print('filter',(f+1),filtr)
                nfd, nfk = filter_fcy(binned_df, sy, minpc=minpc, minsu=minsu)
                if (len(nfd) > 1):
                    nfdrop = get_filter(nfd)
                else:
                    nfdrop = []

                self.cyd.update({filtr: nfd})
                self.cyk.update({filtr: nfk})

                ffdrop = nfdrop
                ffkl = len(nfk)

            elif filtr in ['FDR', 'FWE']:
                print('filter',(f+1),filtr)

                if filtr == 'FDR':
                    fdrd, fdrk = filter_fdr(binned_df, sy, usefdr=True, plvl=plvl)
                    if (len(fdrd) > 1):
                        fdrdrop = get_filter(fdrd)
                    else:
                        fdrdrop = []

                    self.cyd.update({filtr: fdrd})
                    self.cyk.update({filtr: fdrk})

                    ffdrop = fdrdrop
                    ffkl = len(fdrk)
                else:
                    fwed, fwek = filter_fdr(binned_df, sy, usefdr=False, plvl=plvl)
                    if (len(fwed) > 1):
                        fwedrop = get_filter(fwed)
                    else:
                        fwedrop = []

                    self.cyd.update({filtr: fwed})
                    self.cyk.update({filtr: fwek})

                    ffdrop = fwedrop
                    ffkl = len(fwek)

            elif filtr in ['FCBF-SU', 'FCBF-PC']:
                print('filter',(f+1),filtr)
                if filtr == 'FCBF-SU':
                    sud, suk, self.suff = filter_fcc(binned_df, sy, hipc=hipc, hisu=hisu, usesu=True)
                    if (len(sud) > 1):
                        sudrop = get_filter(sud)
                    else:
                        sudrop = []

                    self.cyd.update({filtr: sud})
                    self.cyk.update({filtr: suk})

                    ffdrop = sudrop
                    ffkl = len(suk)
                else:
                    pcd, pck, self.pcff = filter_fcc(binned_df, sy, hipc=hipc, hisu=hisu, usesu=False)
                    if (len(pcd) > 1):
                        pcdrop = get_filter(pcd)
                    else:
                        pcdrop = []

                    self.cyd.update({filtr: pcd})
                    self.cyk.update({filtr: pck})

                    ffdrop = pcdrop
                    ffkl = len(pck)
            else:
                print("Filter",filtr,"not in ['Floor', 'FDR', 'FWE', 'FCBF-SU', 'FCBF-PC']")
                continue

            if (len(ffdrop) > 1):
                print('\tProgressive_filtering: Dropping',len(ffdrop),
                     'features, Keeping',ffkl, end='')
                if filtr in ['FCBF-SU', 'FCBF-PC']:
                    print(' as proxies')
                else:
                    print('')
                binned_df.drop(ffdrop, axis = 1, inplace = True)
                self.QLCFFilter.extend(x for x in ffdrop if x not in self.QLCFFilter)
            else:
                print('\n  - No features were selected to drop')

        return self


#feature-to-label (f2y) correlations 

    def get_f2y_report(self, filters, kd='drop'):
        from .f_rpts import rpt_ycor

        if kd == 'keep':
            for filtr, ycor in self.cyk.items():
                print('\nKept:',filtr,'Layer')
                rpt_ycor(ycor)
        else:
            for filtr, ycor in self.cyd.items():
                print('\nDropped:',filtr,'Layer')
                rpt_ycor(ycor)


    def get_f2y_dict(self, kd='drop'):
        if kd == 'keep':
            return self.cyk
        else:
            return self.cyd


#feature to feature (f2f) correlations above threshold report

    def get_f2f_report(self, filters):
        from .f_rpts import rpt_fcor
        print('Only available for FCBF-SU and FCBF-PC')

        for f in range(len(filters)):
            filtr = filters[f]
 
            if filtr not in self.flist:
                continue
            if filtr not in ['FCBF-SU', 'FCBF-PC']:
                continue

            print('\nHighly correlated features:',filtr,'Layer')

            if filtr == 'FCBF-SU':
                rpt_fcor(self.suff)

            if filtr == 'FCBF-PC':
                rpt_fcor(self.pcff)


    def get_f2f_dict(self, filters):
        hcfd = dict()
        for f in range(len(filters)):
            filtr = filters[f]
 
            if filtr not in self.flist:
                continue
            if filtr not in ['FCBF-SU', 'FCBF-PC']:
                continue

            if filtr == 'FCBF-SU':
                hcfd.update({filtr: self.suff})

            if filtr == 'FCBF-PC':
                hcfd.update({filtr: self.pcff})

        return hcfd


    def transform(self, trudf):
        '''
        apply consolidated drop list
        :trudf: actual pd.dataframe for clf.fit_predict()
        :return: filtered trudf
        '''
        filtered_df = trudf.copy()
        filtered_df.reset_index(inplace = True, drop = True)
        filtered_df.drop(self.QLCFFilter, axis = 1, inplace = True)
        return filtered_df

##  --  --  --  --  ##

    def fit_transform(self, X, y, filters, trudf, 
                      plvl=0.5, 
                      minpc=0.035, minsu=0.0025, 
                      hipc=0.82, hisu=0.7):
        return self.fit(X, y, filters, 
                   plvl=plvl, 
                   minpc=minpc, minsu=minsu, 
                   hipc=hipc, hisu=hisu).transform(trudf)

##  --  --  --  --  ##
