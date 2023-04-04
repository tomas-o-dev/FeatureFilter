#!/usr/bin/env python
# coding: utf-8

# > **Essential ML process for Intrusion Detection**
# <br>` python  3.7.13    scikit-learn  1.0.2 `
# <br>`numpy   1.19.5          pandas  1.3.5`

# **Import the main libraries**

# In[ ]:


import numpy
import pandas

import os
data_path = '../datasets'


# **Import the Dataset**

# In[ ]:


# Using boosted Train and preprocessed Test

data_file = os.path.join(data_path, 'NSL_ppTrain.csv') 
train_df = pandas.read_csv(data_file)
print('Train Dataset: {} rows, {} columns'.format(train_df.shape[0], train_df.shape[1]))

data_file = os.path.join(data_path, 'NSL_ppTest.csv') 
test_df = pandas.read_csv(data_file)
print('Test Dataset: {} rows, {} columns'.format(test_df.shape[0], test_df.shape[1]))


# ***
# **Data Preparation and EDA** (consistency checks)

# * _Check column names of numeric attributes_

# In[ ]:


trnn = train_df.select_dtypes(include=['float64','int64']).columns
tstn = test_df.select_dtypes(include=['float64','int64']).columns
trndif = numpy.setdiff1d(trnn, tstn)
tstdif = numpy.setdiff1d(tstn, trnn)

print("Numeric features in the train_set that are not in the test_set: ",end='')
if len(trndif) > 0:
    print('\n',trndif)
else:
    print('None')

print("Numeric features in the test_set that are not in the train_set: ",end='')
if len(tstdif) > 0:
    print('\n',tstdif)
else:
    print('None')

print()
# correct any differences here


# * _Check column names of categorical attributes_

# In[ ]:


trnn = train_df.select_dtypes(include=['object']).columns
tstn = test_df.select_dtypes(include=['object']).columns
trndif = numpy.setdiff1d(trnn, tstn)
tstdif = numpy.setdiff1d(tstn, trnn)

print("Categorical features in the train_set that are not in the test_set: ",end='')
if len(trndif) > 0:
    print('\n',trndif)
else:
    print('None')

print("Categorical features in the test_set that are not in the train_set: ",end='')
if len(tstdif) > 0:
    print('\n\t',tstdif)
else:
    print('None')

print()
# correct any differences here


# * _Drop columns with only one value_

# In[ ]:


n_eq_one = []
for col in train_df.columns:
    lctrn = len(train_df[col].unique())
    lctst = len(test_df[col].unique())
    if (lctrn == 1) and (lctrn == lctst): 
        n_eq_one.append(train_df[col].name)

if len(n_eq_one) > 0:
    print('Dropping single-valued features')
    print(n_eq_one)
    train_df.drop(n_eq_one, axis=1, inplace=True)
    test_df.drop(n_eq_one, axis=1, inplace=True)


# * _Check categorical feature values:<br>
# differences will be resolved by one-hot encoding the combined test and train sets_

# In[ ]:


trnn = train_df.select_dtypes(include=['object']).columns
for col in trnn:
    tr = train_df[col].unique()
    ts = test_df[col].unique()
    trd = numpy.setdiff1d(tr, ts)
    tsd = numpy.setdiff1d(ts, tr)
    
    print(col,'::> ')
    print("\tUnique text values in the train_set that are not in the test_set: ",end='')
    if len(trd) > 0:
        print('\n\t',trd)
    else:
        print('None')
    
    print("\tUnique text values in the test_set that are not in the train_set: ",end='')
    if len(tsd) > 0:
        print('\n\t',tsd)
    else:
        print('None')


# * _Combine for processing classification target and text features_

# In[ ]:


combined_df = pandas.concat([train_df, test_df])
print('Combined Dataset: {} rows, {} columns'.format(
    combined_df.shape[0], combined_df.shape[1]))


# In[ ]:


# Classification Target feature:
# two columns of labels are available 
#    * Two-class: labels     * Multiclass: atakcat

# Two-class: Reduce the detailed attack labels to 'normal' or 'attack'
labels_df = combined_df['label'].copy()
labels_df[labels_df != 'normal'] = 'attack'

# drop target features 
combined_df.drop(['label'], axis=1, inplace=True)
combined_df.drop(['atakcat'], axis=1, inplace=True)


# ***
# **QLCFF: Quick Layered Correlation-based Feature Filter**<br>
# > **_library requirements:_**<br>
#     * Dataframe with only numeric columns<br>
#     * Numeric class labels in "array-like" with shape (n,1)<br>
#     * Binary classification (not multiclass or multilabel)

# In[ ]:


# one-Hot encoding the remaining text features
categori = combined_df.select_dtypes(include=['object']).columns
category_cols = categori.tolist()

features_df = pandas.get_dummies(combined_df, columns=category_cols)
features_df.info(verbose=False)


# In[ ]:


# numeric values for the target feature
from sklearn.preprocessing import LabelEncoder
ynum = LabelEncoder().fit_transform(labels_df)


# ***
# _**import the local library**_

# In[ ]:


# add parent folder path where lib folder is
import sys
if ".." not in sys.path:import sys; sys.path.insert(0, '..') 


# In[ ]:


## or ## from QLCFF import *
# 
from QLCFF import mkbins
from QLCFF import filter_fcy, filter_fdr, filter_fcc
from QLCFF import get_filter, rpt_ycor, rpt_fcor

# mkbins: applies sklearn KBinsDiscretizer(strategy='uniform', encode='ordinal') 
#       https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-discretization

# filter_fcy: naive filter, feature-to-label (f2y) correlations  
#       filter all with low correlation to target
# filter_fdr: sklearn univariate chi-square test: FDR or FWE
#       https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
# filter_fcc: FCBF-style, filter on feature-to-feature (f2f) correlations
#       using Pearson correlation (PC) or symmetric uncertainty (SU)

# get_filter: returns list of features from f2y report
# rpt_ycor: print feature-to-label (f2y) correlations 
# rpt_fcor: print feature-to-feature (f2f) correlations 


#  ***

# _**mkbins(): FDR/FWE and SU requre binning**_

# In[ ]:


### def mkbins(indf, nb=20, detail=False):
# Requires: features dataframe
#    nb =  maximum number of bins, default 20
#    detail = print report, boolean, default False
# Returns: binned (discretized) dataframe

# applies sklearn KBinsDiscretizer(strategy='uniform', encode='ordinal')
# R doc for miMRMR asserts 10 bins is consistent with the literature
# 20 bins with uniform strategy means each bin is 5% of observed values
#   (ideally - it is really max_bins, uniform means all are the same size) 

tmpdf = mkbins(features_df, detail=True)
# tmpdf.info(verbose=False)


#  ***

# In[ ]:


# naive filter and univariate tests both filter features
#    on the basis that low correlation with the target labels
#    means low utility for distinguishing class membership
# chi_sq is a formal test for independence, fwe will select more to drop
#    than fdr, lower threshold will select more than higher
# naive filter will select all from either univariate test, and more


# <br>_**filter_fcy(): naive filter**_

# In[ ]:


### def filter_fcy(indf, ingt, minpc=0.1, minsu=0.01):
# naive filter: keep if f2y_pc >= minpc or f2y_su >= minsu
# Requires: features_df, numeric_labels
#    minpc = threshold (alpha) for pearson correlation
#    minsu = threshold (alpha) for symmetric uncertainty   
# Returns: f2y report for features to drop

print('\nNaive filter (low correlation with target)')
nfdrop_yc, nfkeep_yc = filter_fcy(tmpdf, ynum)

if (len(nfdrop_yc) > 1):
    nfdrop = get_filter(nfdrop_yc)
    nfkeep = get_filter(nfkeep_yc)
    
    print('To Keep:',len(nfkeep),'features')
#    rpt_ycor(nfkeep_yc)
    print('To Drop:',len(nfdrop),'features')
#    rpt_ycor(nfdrop_yc)
else:
    print('No features were selected')


# In[ ]:


# apply the "keep" filter to the real dataset
# pandas has a lot of rules about returning a 'view' vs. a copy
#        so we force it to create a new dataframe 
# python assigns by reference (namespaces) so keeping a reference
#        to the original is minimal overhead

# features_df_original = features_df
filtered_df = features_df[nfkeep].copy()
filtered_df.info(verbose=False)
# features_df = filtered_df


# <br>_**filter_fdr(): univariate chi_sq layer**_

# In[ ]:


### def filter_fdr(dfin, gtin, t=0.01, usefdr=True):
# Requires: features_df, numeric_labels
#    t = threshold (alpha) for chi_sq test, sklearn default is 0.5
#    usefdr = test, boolean, fdr if True, else fwe  
# Returns: f2y report for features to drop

# set test:
# tst = 'FDR'
# ufd = True
# = or =
tst = 'FWE'
ufd = False

print('\nUnivariate chi-sq',tst,'test')
uvdrop_yc = filter_fdr(tmpdf, ynum, usefdr=ufd)

if (len(uvdrop_yc) > 1):
    uvdrop = get_filter(uvdrop_yc)

    print('Progressive_filtering: Dropping',len(uvdrop),'features')
    tmpdf.drop(uvdrop, axis = 1, inplace = True)

    print('\nFiltered:',tst,'Layer')
    rpt_ycor(uvdrop_yc)
else:
    uvdrop = uvdrop_yc 
    print('\n',tst,'Layer: No features were selected')


#  ***

# In[ ]:


# FCBF-Pearson and FCBF-SU Layers use the same code:  filter_fcc()
#            metric depends on the boolean argument:  usesu
# just use appropriate names for the return values
#      features to keep are called "predominant features" in the FCBF paper; 
#      they act as proxies for the highly correlated features to drop
#      see Lei Yu & Huan Liu, Proc. 20th ICML 2003


# <br>_**filter_fcc(): FCBF-SU layer**_

# In[ ]:


### def filter_fcc(dfin, ingt, t=0.7, usesu=False):
# Requires: features_df, numeric_labels
#    t = threshold (alpha) for "high" f2f correlation
#        standard for detecting multicollinearity is 0.7
#    usesu = metric, boolean, su if True, else pearson  
# Returns: f2y report for features to drop
#          f2y report for features to keep
#          f2f above threshold report 

sudrop_yc, sukeep_yc, su_hicorr = filter_fcc(tmpdf, ynum, usesu=True)

if (len(sudrop_yc) > 1):
    sudrop = get_filter(sudrop_yc)

    print('Progressive_filtering: Dropping',len(sudrop),'features')
    tmpdf.drop(sudrop, axis = 1, inplace = True)

    print('\nKept: FCBF (SU) Layer')
    rpt_ycor(sukeep_yc)
    print('\nFiltered: FCBF (SU) Layer')
    rpt_ycor(sudrop_yc)
    print('\nHighly correlated features: FCBF (SU) Layer')
    rpt_fcor(su_hicorr)
else:
    sudrop = sudrop_yc 
    print('\nFCBF (SU) Layer\nNo features were selected')


# <br>_**filter_fcc(): FCBF-Pearson layer**_

# In[ ]:


### def filter_fcc(dfin, ingt, t=0.7, usesu=False):
# Requires: features_df, numeric_labels
#    t = threshold (alpha) for "high" f2f correlation
#        standard for detecting multicollinearity is 0.7
#    usesu = metric, boolean, su if True, else pearson 
# Returns: f2y report for features to drop
#          f2y report for features to keep
#          f2f above threshold report 

pcdrop_yc, pckeep_yc, pc_hicorr = filter_fcc(tmpdf, ynum)

if (len(pcdrop_yc) > 1):
    pcdrop = get_filter(pcdrop_yc)

    print('Progressive_filtering: Dropping',len(pcdrop),'features')
    tmpdf.drop(pcdrop, axis = 1, inplace = True)

    print('\nKept: Pearson Layer')
    rpt_ycor(pckeep_yc)
    print('\nFiltered: Pearson Layer')
    rpt_ycor(pcdrop_yc)
    print('\nHighly correlated features: Pearson Layer')
    rpt_fcor(pc_hicorr)
else:
    pcdrop = pcdrop_yc 
    print('\nPearson Layer\nNo features were selected')


#  ***

# In[ ]:


## QLCFF: Quick Layered Correlation-based Feature Filter
## full layered drop filter for real DF 
QLCFFilter = []
QLCFFilter.extend(x for x in uvdrop if x not in QLCFFilter)
QLCFFilter.extend(x for x in pcdrop if x not in QLCFFilter)
QLCFFilter.extend(x for x in sudrop if x not in QLCFFilter)


# In[ ]:


# apply the "drop" filter to the real dataset

# features_df_original = features_df
filtered_df = features_df.drop(QLCFFilter, axis = 1)
filtered_df.info(verbose=False)
# features_df = filtered_df


#  ***

# In[ ]:




