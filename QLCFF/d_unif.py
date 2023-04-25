import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

from .c_dctzr import Discretizer

# thanks to : https://github.com/Anylee2142/

class uniform(Discretizer):
    def __init__(self, method='lin', maxbins=10, numjobs=1, msglvl=0):
        assert maxbins > 1, 'maxbins should be bigger than 1'

        Discretizer.__init__(
            self=self,
            maxbins=maxbins,
            numjobs=1,
            msglvl=0
        )

        self.method = method
        self.maxbins = maxbins


    def _get_cutpoints(self, feature, target, feature_name):
        '''
        Get cutpoints from one continuous feature
        :param feature: continuous column to be discretized
        :param target: target column considered
        :param method: 'lin' [1],[2]; 'log' [1]; 'sqrt' [3]

    [1] J. Dougherty, R. Kohavi, and M. Sahami, “Supervised and unsupervised
    discretization of continuous features,” in ICML 1995, pp. 194–202
    [2] Y. Yang and G. I. Webb, “A comparative study of discretization methods 
    for naive-bayes classifiers,” in Proc. PKAW 2002, pp. 159-173
    [3] Y. Yang and G. I. Webb, “Proportional k-interval discretization 
    for naive-bayes classifiers,” in Machine learning: ECML 2001, pp. 564–575

        :return: List of cutpoints
        :        passthru feature_name
        '''

        if self.method == 'sqrt':
            n = len(np.unique(feature)) 
            nbins = min(self.maxbins, round(np.sqrt(n)))
            bin_edges = np.linspace(feature.min(), feature.max(), nbins + 1)

        elif self.method == 'log':
            n = len(np.unique(feature)) 
            nbins = min(self.maxbins, max(1, 2*(round(np.sqrt(n)))))
            if feature.max() < (1+np.nextafter(1,1)):
                bin_edges = np.logspace(feature.min(), feature.max(), nbins + 1)
                bin_edges = bin_edges / 10
                if feature.min() == 0:
                    bin_edges[bin_edges == 0.1] = 0
            else:
                bin_edges = np.logspace(np.log10(feature.min()), np.log10(feature.max()), nbins + 1)

        else:
            bin_edges = np.linspace(feature.min(), feature.max(), self.maxbins + 1)

        return feature_name, bin_edges[1:-1].tolist()
