import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from scipy.stats import chi2

## from .c_dctzr import Discretizer

# fork of: https://github.com/Anylee2142/ChiMerge

# Kerber R. 1992. Chimerge: Discretization of numeric attributes. 
# Proc. 10th National Conference on Artifcial Intelligence (AAAI'92), 123–128

# ChiMerge Algorithm
# 1. sorting continuous features
# 2. make each example as one interval
# 3. compute chi2 value for every pair of adjacent two intervals
# 4. merge certain pair of intervals that have lowest chi2 value
# 5. go 3 until chi2 values from all pairs exceed threshold of 
#    pre-defined significance level (until all alternative hypothesises become passed)

## class ChiMerge(Discretizer):
##    def __init__(self, mkbins='chim-ten', numjobs=1, msglvl=50, plvl=0.1):
##        assert 1 > plvl > 0, 'plvl (significance level) should be 0.01, 0.05, or 0.1'



def chi_score(intervals):
    '''
        Compute chi square statistics from paper
        :param intervals: List of pd.Series with frequencies of each class
        :return: chi square from paper
    '''

    for interval in intervals:
        assert len(interval) > 0, 'each interval should not have length of 0'

    A = np.array([
        interval.values for interval in intervals
    ])

    R = np.array([
        [np.sum(row)] for row in A
    ])

    C = np.array([
        np.sum(col) for col in A.T
    ])

    N = A.sum()

    try:
        E = np.multiply(R, C) / N

        chi2 = np.nansum(
            np.power(A - E, 2) / E
        )

    except ValueError as ve:
        # To catch unidentical dimension error, R <-> C, A <-> E
        print(ve)

    return chi2


def get_cutpoints(nbinz, feature, target, feature_name):
    '''
        Get ChiMerge cutpoints from one continuous feature
        :param feature: continuous column to be discretized
        :param target: target column considered
        :return: List of cutpoints
        :        passthru feature_name
    '''
    from .d_nbhg import nbins

    # max_cutpoints
    max_cutpoints = nbins(feature, nbinz)

    feature_target = pd.concat([feature, target], axis=1)
    feature_target.sort_values(by=[feature.name], inplace=True)
    feature_target.reset_index(drop=True, inplace=True)

    # unique value : target frequencies
    frequencies = feature_target\
        .groupby(by=[feature.name, target.name])\
        .size()\
        .unstack()\
        .fillna(0)

    intervals = [unique_value[1] for unique_value in frequencies.iterrows()]

    num_of_classes = len(target.unique())
##
#    chi2_threshold = chi2.ppf(1 - self.plvl, num_of_classes - 1)
    plvl=0.1
    chi2_threshold = chi2.ppf(1 - plvl, num_of_classes - 1)
##
    mode = False
    while True:
        chi2_scores = list()

        # TODO: don't iterate, but keep them in matrix for one go
        for idx in range(len(intervals)-1):
            chi2_scores.append(chi_score(
                intervals=intervals[idx:idx+2]
            ))

        # 1. Merge intervals until all statistics are bigger than threshold
        if mode is False and all(np.array(chi2_scores) > chi2_threshold):
            mode = True
            continue

        # 2. If upper condition is satisfied, merge them until num_intervals < max_cutpoints
        if mode is True and len(intervals) <= max_cutpoints:
            break

        # Merge intervals with lowest chi2 score, which means they have similar class frequencies
        lowest_idx = np.argmin(chi2_scores)
        merged_interval = intervals[lowest_idx] + intervals[lowest_idx+1]
        merged_interval.name = '{} ~ {}'.format(intervals[lowest_idx].name, intervals[lowest_idx+1].name)
        intervals = intervals[:lowest_idx] + [merged_interval] + intervals[lowest_idx+2:]

        # If one feature merged into one interval, then it's irrelavant feature
        if len(intervals) == 1:
            return feature_name, []

    ranges = [interval.name.replace(' ', '').split('~')[0] if isinstance(interval.name, str)
              else str(interval.name)
              for interval in intervals]

    return feature_name, ranges
