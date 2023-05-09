#!/usr/bin/env python
# coding: utf-8

# https://github.com/ctlab/ITMO_FS/blob/master/ITMO_FS/utils/information_theory.py
# ITMO implementations are nice, use np.apply_along_axis()

import numpy as np
import pandas as pd

from collections import Counter  #https://www.guru99.com/python-counter-collections-example.html#5
from operator import itemgetter  
#https://docs.python.org/3/library/operator.html
from itertools import groupby
from math import log, fsum


#    Parameters
#    x : array-like, shape (n,)
#    y : array-like, shape (n,)

def entropy(x):
    """Calculate the entropy (H(X)) of an array.
    Returns      float : H(X) value
    """
    return log(len(x)) - fsum(v * log(v) for v in Counter(x).values()) / len(x)


def conditional_entropy(x, y):
    """Calculate the conditional entropy (H(Y|X)) 
                 between two arrays.
    Returns      float : H(Y|X) value
    """
    buf = [[e[1] for e in g] for _, g in 
           groupby(sorted(zip(x, y)), itemgetter(0))]
    return fsum(entropy(group) * len(group) for group in buf) / len(x)


def mutual_info(x, y):
    """Calculate the mutual information (I(X;Y) = H(Y) - H(Y|X)) 
                 between two arrays.
    Returns      float : I(X;Y) value
    """
    return entropy(y) - conditional_entropy(x, y)


# su_measure(X, y) 
# https://github.com/ctlab/ITMO_FS/blob/master/ITMO_FS/filters/univariate/measures.py
# https://github.com/ctlab/ITMO_FS/blob/master/ITMO_FS/filters/multivariate/FCBF.py

def symm_uncert(x, y):
    """Calculate the symmetric uncertainty U(X,Y) = 2* I(X;Y) / (H(X)+H(Y)) 
                 between two arrays.
    Returns      float : U(X,Y) value
    """
    entropy_x = entropy(x)
    entropy_y = entropy(y)
    if (entropy_x != 0) or (entropy_y != 0):
        suv = 2 * ((entropy_x - conditional_entropy(y, x)) / (entropy_x + entropy_y))
    else:
        suv = np.nextafter(0, 1)
    return suv


### - joblib for sucorr
def updatecmx(indf,cmx,i,j):
    item = cmx.iloc[j:(j+1), (i+1):(i+2)]
    col = item.columns
    row = item.index
    su = symm_uncert(indf[col[0]], indf[row[0]])
    return [i,j,su]


def mksucm(dfin, numjobs= -2, msglvl=0):
    from joblib import Parallel, delayed

    cmx = dfin.corr()
    pb='%'
    sulist=[]

    with Parallel(n_jobs=numjobs, verbose=msglvl) as parallel:
        for i in range(len(cmx.columns) - 1):
            work = parallel( 
                delayed(updatecmx)(
                dfin,cmx,i,j
                )    
                for j in range(i+1)
            )
            sulist.extend(work)
            print(pb,end='')

    print('  --Done, Updating...')
    for x in range(len(sulist)):
        i=sulist[x][0]
        j=sulist[x][1]
        v=sulist[x][2]
        cmx.iloc[j:(j+1), (i+1):(i+2)] = v

    return cmx


## call this --
# requires pandas.dataframe, gt_labels
# the standard threshold for multicolliniarity is > 0.7
#     experiments show PC can go a bit higher

def mulcol(indf, ingt, hipc=0.82, hisu=0.7, usesu=False):

# create correlation matrix
#----
    if usesu:
        threshold = hisu
        print('Calculating the SU correlation matrix takes some time ...')
        corr_matrix = mksucm(indf, numjobs= -2, msglvl=0)  
    else:
        threshold = hipc
        corr_matrix = indf.corr()
#----
# correlations > threshold
    drop_cor = []
    for i in range(len(corr_matrix.columns) - 1):
            for j in range(i+1):
                item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
                col = item.columns
                row = item.index
                val = abs(item.values)
                if val >= threshold:
                    dd=[col.values[0],row.values[0],round(val[0][0], 4)]
                    drop_cor.append(dd)

# make the frequency table
    chekt = []
    vfrqt = []
    for i in range(len(drop_cor)):
        v1c=drop_cor[i][0]
        v2c=drop_cor[i][1]
        if v1c not in chekt:
            chekt.append(v1c)
            vfrqt.append([v1c,0])
        if v2c not in chekt:
            chekt.append(v2c)
            vfrqt.append([v2c,0])

        for r in range(len(vfrqt)):
            if v1c == vfrqt[r][0]:
                vfrqt[r][1]+=1
            if v2c == vfrqt[r][0]:
                vfrqt[r][1]+=1

    dropc = []     # multiple covariates
    for r in range(len(vfrqt)):
        if vfrqt[r][1] > 1:
            dropc.append(vfrqt[r][0])

    keepc = []     # single covariate
    for r in range(len(chekt)):
        if chekt[r] not in dropc:
            keepc.append(chekt[r])

# potential proxies for multiple covariates
    prokz = []
    for p in range(len(dropc)):
        pkz = dropc[p]
        tmpp =[]
        for i in range(len(drop_cor)):
            v1c=drop_cor[i][0]
            v2c=drop_cor[i][1]
            if v1c == pkz:
                tmpp.append(drop_cor[i][1])
            if v2c == pkz:
                tmpp.append(drop_cor[i][0])
        prokz.append([pkz,tmpp])
    
# keep or proxy: initialise
    korp = []
    for p in range(len(prokz)):
        FL = prokz[p][1]
        for i in range(len(FL)):
            if FL[i] not in korp: 
                korp.append(FL[i])

# start with the longest list of covariates
    keepp = []
    dropc = []

### Loop 
    while (len(prokz) > 0):
        ln = 0
        for p in range(len(prokz)):
            if len(prokz[p][1]) > ln:
                ln=len(prokz[p][1])
                tgt=prokz[p][0]
                tgf=prokz[p][1]

### loop thru tgf, find higher cwt
        ctgt = tgt
        vcf = indf[tgt].values

        if usesu:
            bc=symm_uncert(vcf, ingt)
        else:
            bc = abs(np.corrcoef(vcf, ingt)[0,1])
                
### swap highest into tgt
        for p in range(len(tgf)):
            if (tgf[p] not in keepp) and (tgf[p] not in dropc):
                vcf = indf[tgf[p]].values

                if usesu:
                    cwt=symm_uncert(vcf, ingt)
                else:
                    cwt = abs(np.corrcoef(vcf, ingt)[0,1])

                if cwt > bc:
                    ctgt = tgf[p]
                    bc = cwt                

        if ctgt != tgt:  
### drop ctgt from tgf
            tgf[:] = [x for x in tgf if x != ctgt]
### add tgt to tgf
            tgf.append(tgt)
### swap highest into tgt
            tgt = ctgt                

## good to go ...
        keepp.append(tgt)

# add everything in tgf to dropc - they have a proxy
        for n in range(len(tgf)):
            if tgf[n] not in dropc:
               dropc.append(tgf[n])

#### list comprehension magic (pythonic)
## With list comprehension, it is tempting to build a new list and 
## assign it the same name as the old list. This will get the desired result, 
## but it does not remove the old list in place.

## To make sure the reference remains the same, you must assign to a list slice. 
## This will replace the contents with the same Python list object, so the reference 
## remains the same, preventing some bugs if it is being referenced elsewhere.

# remove anything in tgf from keepc
        for n in range(len(tgf)):
           keepc[:] = [x for x in keepc if x != tgf[n]]

# rm from korp
        korp[:] = [x for x in korp if x != tgt]
        for n in range(len(tgf)):
           korp[:] = [x for x in korp if x != tgf[n]]

# rm from prokz
        prokz[:] = [x for x in prokz if x[0] != tgt]
        for n in range(len(tgf)):
           prokz[:] = [x for x in prokz if x[0] != tgf[n]]

# len prokz==0
# singletons - keep one of
# if both are in keepc: drop the one with lower abs(corr) to target var
    keep1 = []
    drop1 = []
    corr1 = []
    for i in range(len(drop_cor)):
        v1c=drop_cor[i][0]
        v2c=drop_cor[i][1]

        if (v1c in keepc) and (v2c in keepc):
            corr1.append([v1c,v2c])

    for i in range(len(corr1)):
        v1c=corr1[i][0]
        v2c=corr1[i][1]

        vals1=indf[v1c].values
        cowt1 = np.corrcoef(vals1,ingt)[0,1]
        vals2=indf[v2c].values
        cowt2 = np.corrcoef(vals2,ingt)[0,1]

        if (abs(cowt1) > abs(cowt2)):
            keep1.append(v1c)
            drop1.append(v2c)
        else:
            drop1.append(v1c)
            keep1.append(v2c)

# other correlated feature is on the drop list already
    for n in range(len(keepc)):
        if (keepc[n] not in keep1) and (keepc[n] not in drop1) and (keepc[n] not in keepp):
            keepp.append(keepc[n])

    for i in range(len(keep1)):
        keepp.append(keep1[i])

    for i in range(len(drop1)):
        dropc.append(drop1[i]) 

    return dropc, keepp, drop_cor, corr_matrix

