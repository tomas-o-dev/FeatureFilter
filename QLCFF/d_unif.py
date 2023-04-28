import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

from .c_dctzr import Discretizer

# thanks to : https://github.com/Anylee2142/

class uniform(Discretizer):
    def __init__(self, mkbins='ten', numjobs=1, msglvl=0):

        Discretizer.__init__(
            self=self,
            numjobs=1,
            msglvl=0
        )

        self.mkbins = mkbins


    def _get_cutpoints(self, feature, target, feature_name):
        '''
        Get cutpoints from one continuous feature
        :param feature: continuous column to be discretized
        :param target: target column considered
        :param mkbins: 'ten' [1],[2]; 'log' [1]; 'sqrt' [3]

    [1] J. Dougherty, R. Kohavi, and M. Sahami, “Supervised and unsupervised
    discretization of continuous features,” in ICML 1995, pp. 194–202
    [2] Y. Yang and G. I. Webb, “A comparative study of discretization methods 
    for naive-bayes classifiers,” in Proc. PKAW 2002, pp. 159-173
    [3] Y. Yang and G. I. Webb, “Proportional k-interval discretization 
    for naive-bayes classifiers,” in Machine learning: ECML 2001, pp. 564–575

        :return: List of cutpoints
        :        passthru feature_name
        '''
        n = len(np.unique(feature)) 

        if self.mkbins == 'sqrt':
            nbins = round(np.sqrt(n))

        elif self.mkbins == 'log':
            nbins = max(2, 2*(round(np.log10(n))))

        else:
            nbins = 10

        bin_edges = np.linspace(feature.min(), feature.max(), nbins + 1)
        return feature_name, bin_edges[1:-1].tolist()
