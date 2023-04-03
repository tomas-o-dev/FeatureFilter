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

    print('Using only numeric datatypes',end='')
    dzdf = indf.select_dtypes(
        include=["number",'bool','boolean'],exclude=["timedelta64","complexfloating"]).copy()
    dxc=[]
    dxc.extend(x for x in indf.columns.values if x not in dzdf.columns.values)
    if len(dxc) > 0:
        print('\nDropping ineligible features (text features should be one-hot encoded)')
        for c in range(len(dxc)):
            print('\t',dxc[c],' [ dtype =',indf.dtypes[dxc[c]].name,']')
    else:
        print(' [ All features are in ]')
        
# all pyTF columns
    pytf = dzdf.select_dtypes(include=['bool','boolean']).columns.values
    for c in range(len(pytf)):
        dzdf[c] = dzdf[c].replace({True: 1, False: 0})

# signed number columns
    sn = dzdf.select_dtypes(
        include=["signedinteger","floating"],exclude=["timedelta64"]).columns.values

    for col in dzdf.columns:
        uv = len(dzdf[col].unique())
        if uv > nb:
            arr = numpy.array(dzdf[col])
            binz = KBinsDiscretizer(strategy='uniform', encode='ordinal', 
                                    n_bins=nb).fit(arr.reshape(-1, 1))
            dzdf[col] = binz.transform(arr.reshape(len(arr),1))
        else:
            if dzdf[col].name in sn:
                if dzdf[col].min() < 0:
                    dzdf[col] += abs(dzdf[col].min())
        
    if detail:
        colwid = max(len(n) for n in indf.columns.values)    # +i for padding
        print('Unique value count: Original ::> Binned\n  Same for all features except:')
        for col in indf.columns:
            lctrn = len(indf[col].unique())
            lctst = len(dzdf[col].unique())
            if lctrn != lctst:
                print("{: <{colwid}} {: >5} {} {: <5}".format(
                    col,lctrn,'::>',lctst,colwid=colwid))

    return dzdf

