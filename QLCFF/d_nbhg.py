import numpy as np
import pandas as pd

def hgbins(feature):
    uv = np.unique(feature)

    bnz, egz = np.histogram(uv, bins='auto')
    egz = egz[np.nonzero(bnz)]
    if uv.min != 0:
        egz[0:1] = 0

    while (len(egz) /2) > 6:
        egz[1::2] = [0]*(len(egz)//2)
        egz = egz[np.nonzero(egz)]
        egz = np.insert(egz, 0, 0)

    edges = np.append(egz,[feature.max()])
    return edges


def nbins(feature, nb):
    n = len(np.unique(feature)) 

#   floor() always rounds down, 
#           rint() rounds 0.5 to the nearest even value ...
    if nb == 'sqrt':
        bins = np.rint(np.sqrt(n)).astype(int)
    elif nb == 'log':
        bins = max(1, 2*(np.rint(np.log10(n)).astype(int)))
    else:
        bins = 10

    return bins


def get_cutpoints(nbinz, feature, target, feature_name):
    '''
    Get cutpoints from one continuous feature
        :param feature: continuous column to be discretized
        :param target: target column considered
        :param nbinz: 'ten' [1],[2]; 'log' [1]; 'sqrt' [3], 'auto' [4]

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
    if nbinz in ['ten', 'log', 'sqrt']:
        numbins = nbins(feature, nbinz)
        bin_edges = np.linspace(feature.min(), feature.max(), numbins + 1)
    else:
        bin_edges = hgbins(feature)

    return feature_name, bin_edges[1:-1].tolist()
