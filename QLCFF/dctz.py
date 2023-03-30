#!/usr/bin/env python
# coding: utf-8

# binning: requires dataframe, returns dataframe
# nb =  maximum number of bins, default 20
# detail = print report, boolean, default False
# R doc for miMRMR asserts 10 bins is consistent with the literature
# 20 bins with uniform strategy means each bin is 5% of observed values
#   (ideally - it is really max_bins, uniform means all are the same size)

def mkbins(indf, nb=20, detail=False):
    import numpy
    import pandas
    from sklearn.preprocessing import KBinsDiscretizer

    dzdf = indf.copy()
    for col in indf.columns:
        uv = len(indf[col].unique())
        if uv > nb:
            arr = numpy.array(indf[col])
            binz = KBinsDiscretizer(strategy='uniform', encode='ordinal', 
                                    n_bins=nb).fit(arr.reshape(-1, 1))
            dzdf[col] = binz.transform(arr.reshape(len(arr),1))
        
    if detail:
#        data = indf.columns.values
        colwid = max(len(n) for n in indf.columns.values)    # +i for padding
        print('Unique value count: Original ::> Binned\n  Same for all features except:')
        for col in indf.columns:
            lctrn = len(indf[col].unique())
            lctst = len(dzdf[col].unique())
            if lctrn != lctst:
                print("{: <{colwid}} {: >5} {} {: <5}".format(
                    col,lctrn,'::>',lctst,colwid=colwid))

    return dzdf

