#!/usr/bin/env python
# coding: utf-8


# selects features to keep based on an upper bound on 
# the expected false discovery rate
#       standard thresholds are 0.01, 0.05, 0.1
#       sklearn alpha default=0.05
# requires dataframe, gt_labels
#    features must be discrete and non-negative
# returns a list of features most likely to have high FPR
#     or an empty list if all are likely to have high FPR

def uvtcsq(indf, ingt, t=0.05, usefdr=True):
    import numpy
    import pandas
    from sklearn.feature_selection import chi2, SelectFdr, SelectFwe

    if usefdr:    
        FD = SelectFdr(score_func=chi2, alpha=t).fit(indf, ingt)
    else:
        FD = SelectFwe(score_func=chi2, alpha=t).fit(indf, ingt)

# features selected (sklearn 1.0 or better)
# FD.get_support(indices=False)  # t/f mask or ints
# FD.get_feature_names_out()     # names

    todrop = []
    try:
        fdd=FD.get_feature_names_out() 
    except:
        print('No features selected')
    else:
        dfc=indf.columns.values
        for p in range(len(dfc)):
            if dfc[p] not in fdd:
                todrop.append(dfc[p])
            
    return todrop        

