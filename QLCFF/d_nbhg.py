import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

from .c_dctzr import Discretizer


def hgbins(feature):
    bnz, egz = np.histogram(feature, bins='auto')
    egz = egz[np.nonzero(bnz)]
    if feature.min != 0:
        egz[0:1] = 0

    while (len(egz) /2) > 8:
        egz[1::2] = [0]*(len(egz)//2)
        egz = egz[np.nonzero(egz)]
        egz = np.insert(egz, 0, 0)

    edges = np.append(egz,[feature.max()])
    return edges


def nbins(feature, mkbins):
    n = len(np.unique(feature)) 

# floor() always rounds down, rint() rounds to the nearest even value ...
    if mkbins == 'sqrt':
        bins = np.rint(np.sqrt(n)).astype(int)
    elif mkbins == 'log':
        bins = max(1, 2*(np.rint(np.log10(n)).astype(int)))
    else:
        bins = 10

    return bins



# thanks to : https://github.com/Anylee2142/

class unifhgm(Discretizer):
    def __init__(self, mkbins='hgrm', numjobs=1, msglvl=0):

        Discretizer.__init__(
            self=self,
            numjobs=1,
            msglvl=0
        )

        if mkbins in ['sqrt', 'log', 'ten', 'hgrm']:
            self.mkbins = mkbins
        else:
            self.mkbins='hgrm'


    def _get_cutpoints(self, feature, target, feature_name):
        '''
        Get cutpoints from one continuous feature
        :param feature: continuous column to be discretized
        :param target: target column considered
        :param mkbins: 'ten' [1],[2]; 'log' [1]; 'sqrt' [3], 'auto' [4]

    [1] J. Dougherty, R. Kohavi, and M. Sahami, “Supervised and unsupervised
    discretization of continuous features,” in ICML 1995, pp. 194–202
    [2] Y. Yang and G. I. Webb, “A comparative study of discretization methods 
    for naive-bayes classifiers,” in Proc. PKAW 2002, pp. 159-173
    [3] Y. Yang and G. I. Webb, “Proportional k-interval discretization 
    for naive-bayes classifiers,” in Machine learning: ECML 2001, pp. 564–575
    [4] https://numpy.org/doc/stable/reference/generated/numpy.histogram.html

        :return: List of cutpoints
        :        passthru feature_name
        '''
        if self.mkbins == 'hgrm':
            bin_edges = hgbins(feature)
        else:
            numbins = nbins(feature, self.mkbins)
            bin_edges = np.linspace(feature.min(), feature.max(), numbins + 1)

        return feature_name, bin_edges[1:-1].tolist()
