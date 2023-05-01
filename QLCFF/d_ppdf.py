#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

def prep_df(indf):
    print('Using only numeric datatypes',end='')
    dzdf = indf.select_dtypes(
        include=['number','bool','boolean'],exclude=['timedelta64','complexfloating']).copy()

    dxc=[]
    dxc.extend(x for x in indf.columns.values if x not in dzdf.columns.values)
    if len(dxc) > 0:
        print('\nDropping ineligible features (text features should be one-hot encoded)')
        for c in range(len(dxc)):
            print('\t',dxc[c],' [ dtype =',indf.dtypes[dxc[c]].name,']')
    else:
        print(' [ All features are in ]')

# all signed numbers
    sn = dzdf.select_dtypes(
        include=['signedinteger','floating'],exclude=['timedelta64']).columns.values                    
    for c in range(len(sn)):
        if dzdf[sn[c]].min() < 0:
            dzdf[sn[c]] += abs(dzdf[sn[c]].min())

# all pyTF columns
    pytf = dzdf.select_dtypes(include=['bool','boolean']).columns.values
    for c in range(len(pytf)):
        dzdf[sn[c]] = dzdf[sn[c]].replace({True: 1, False: 0})


# list of column names for only histogram
    hgc=[]

# list of column names for get_cutpoints
    ftc=[]

    for col in dzdf.columns:
        uv = len(dzdf[col].unique())
        if uv == 1:
            print('WARNING: Dropping single-valued feature', dzdf[col].name)
            dzdf.drop(dzdf[col].name, axis=1, inplace=True)
        elif uv == 2: 
            if (dzdf[col].min() != 0) or (dzdf[col].max() != 1):
                dzdf[col] = dzdf[col].replace({dzdf[col].max(): 1, dzdf[col].min(): 0})
        elif (uv > 2) and (uv < 32):        # log10(32) is first to give numbins>2
            hgc.append(dzdf[col].name)
        else:
            ftc.append(dzdf[col].name)                

    dzdf.reset_index(inplace = True, drop = True)
    return dzdf, ftc, hgc
