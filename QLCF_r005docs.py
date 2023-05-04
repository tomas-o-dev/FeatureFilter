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


# In[ ]:


# one-Hot encoding the remaining text features
categori = combined_df.select_dtypes(include=['object']).columns
category_cols = categori.tolist()

features_df = pandas.get_dummies(combined_df, columns=category_cols)
features_df.info(verbose=False)


# In[ ]:


# Restore the train // test split: slice 1 Dataframe into 2 
# pandas has a lot of rules about returning a 'view' vs. a copy from slice
# so we force it to create a new dataframe [avoiding SettingWithCopy Warning]
features_train = features_df.iloc[:len(train_df),:].copy()    # X_train
features_test = features_df.iloc[len(train_df):,:].copy()     # X_test

# Restore the train // test split: slice 1 Series into 2 
labels_train = labels_df[:len(train_df)]               # y_train
labels_test = labels_df[len(train_df):]                # y_test


# In[ ]:





# ***
# **QLCFF: Quick Layered Correlation-based Feature Filter**<br>
# > **_library requirements:_**<br>
#     * Dataframe of features (text values may be one-hot encoded)<br>
#     * Class labels in np.ndarray or pd.Series with shape (n,1)<br>
#     * Binary classification (not multiclass or multilabel)<br><br>
# > **_workflow:_**<br>
#     1. Instantiate a discretizer<br>
#     2. get the binned dataframe from the discretizer<br>
#     3. Apply filters to the binned dataset<br>
#     4. Apply drop (or keep) lists to the real features dataset

# ***
# _**import the local library**_

# In[ ]:


# add parent folder path where lib folder is
import sys
if ".." not in sys.path:import sys; sys.path.insert(0, '..') 


# In[ ]:


from QLCFF import unifhgm, MDLP, ChiMerge
# three distinct discretizers can be instantiated

from QLCFF import filter_fcy, filter_fdr, filter_fcc
# filter_fcy: floor filter, feature-to-label (f2y) correlations  
#       filter all with low correlation to target
# filter_fdr: sklearn univariate chi-square test: FDR or FWE
#       https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
# filter_fcc: FCBF-style, filter on feature-to-feature (f2f) correlations
#       using Pearson correlation (PC) or symmetric uncertainty (SU)

from QLCFF import get_filter, rpt_ycor, rpt_fcor
# get_filter: returns list of features from f2y report
# rpt_ycor: print feature-to-label (f2y) correlations 
# rpt_fcor: print feature-to-feature (f2f) correlations


# ***
# _new in v0.05_
# >   1. Instantiate a discretizer<br>
# 2. get the binned dataframe from the discretizer<br>

# In[ ]:


# three distinct discretizers can be instantiated:
#    unifhgm: uniform (np.linspace()) or histogram
#        Optional : mkbins in ['ten','sqrt','log','hgrm']
#        default is hgrm: applies np.histogram(feature, bins='auto')
#    MDLP algorithm  [1]
#        Optional : mkbins in ['ten','sqrt','log']
#                   joblib processes, verbose level 
#                          defaults: numjobs=1, msglvl=0 
#    ChiMerge algorithm  [2]
#        Optional : mkbins in ['ten','sqrt','log']
#                   joblib processes, verbose level 
#                          defaults: numjobs=1, msglvl=0 
#    ten:  number of bins is always ten - default for MDLP and ChiMerge [3,4]
#    sqrt: number of bins is sqrt(len(np.unique(feature)))   [5]
#    log:  number of bins is log10(len(np.unique(feature)))  [3]


# <br>_**binning: uniform or histogram**_

# In[ ]:


# IMPORTANT: instantiate, then call fit() or fit_transform()
hgmb = unifhgm(mkbins='hgrm')


# In[ ]:


# fit() calls the preprocessor and the discretizer
# Requires:
#    features as pd.dataframe, labels as array-like
# Optional:
#    print detailed report (boolean) - default False
# preprocessor:
#    1. selects only column dtypes np.number and pd or np boolean
#    2. nomalizes all columns with signed dtypes to positive numbers
#    3. nomalizes all columns with boolean dtypes to zero//one
#    text labels are converted with sklearn LabelEncoder()

hgmb.fit(features_test, labels_test,
         detail=True)


# In[ ]:


# binned bataframe is an attribute
#     transform() is just a getter method, with optional detail

X_hgmbinz = hgmb.transform(features_test, detail=True)
#X_hgmbinz = hgmb.binned_df


# In[ ]:


X_hgmbinz.info(verbose=False)


# In[ ]:


# detailed list of bin edges is an attribute
# hgmb.cutpoints


# <br>_**binning: MDLP algorithm**_

# In[ ]:


# IMPORTANT: instantiate, then call fit() or fit_transform()
mdlpb = MDLP(mkbins='log',
            numjobs= -2, msglvl=5)


# In[ ]:


X_mdlbinz = mdlpb.fit_transform(features_test, labels_test,
                                detail=True)


# In[ ]:


X_mdlbinz.info(verbose=False)


# <br>_**binning: ChiMerge algorithm**_

# In[ ]:


# IMPORTANT: instantiate, then call fit() or fit_transform()
chi_merge = ChiMerge(mkbins='sqrt',
                    numjobs= -2, msglvl=5)


# In[ ]:


X_chibinz = chi_merge.fit_transform(features_test, labels_test, 
                                    detail=True)


# In[ ]:


X_chibinz.info(verbose=False)


# ***
# _same as v0.04, with minor changes to signatures_
# >   3. Apply filters to the binned dataset

# In[ ]:


# filter_fcy: floor filter, feature-to-label (f2y) correlations  
#       filter all with low correlation to target
# filter_fdr: sklearn univariate chi-square test: FDR or FWE
#       https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection

# floor filter and univariate tests both filter features
#    on the basis that low correlation with the target labels
#    means low utility for distinguishing class membership

# chi_sq is a formal test for independence, fwe will select more to drop than fdr,
#    standard thresholds are 0.1, 0.05, 0.01; lower will select more to drop 
# floor filter will select all from either univariate test, and more

# one of these can be applied before
#     filter_fcc: FCBF-style, filter on feature-to-feature (f2f) correlations
#           using Pearson correlation (PC) or symmetric uncertainty (SU)
# to create layered feature selection filters


# In[ ]:


# filters still require numeric targets
from sklearn.preprocessing import LabelEncoder
ynum = LabelEncoder().fit_transform(labels_test)


# In[ ]:


# dataframe to use for filters
binned_df = X_hgmbinz


# <br>_**filter_fcy(): floor filter**_ - as single layer

# In[ ]:


# filter_fcy: floor filter, feature-to-label (f2y) correlations  
#       drop if f2y_pc < minpc or f2y_su < minsu

# Requires: binned_df, numeric_labels
# Optional:
#    minpc : threshold for pearson correlation    default=0.1
#    minsu : threshold for symmetric uncertainty  default=0.01 
# Returns: f2y report for features to drop
#          f2y report for features to keep

print('\nFloor filter (low correlation with target)')
nfdrop_yc, nfkeep_yc = filter_fcy(binned_df, ynum)

if (len(nfdrop_yc) > 1):
    nfdrop = get_filter(nfdrop_yc)
    nfkeep = get_filter(nfkeep_yc)
    
    print('To Keep:',len(nfkeep),'features')
#    rpt_ycor(nfkeep_yc)
    print('To Drop:',len(nfdrop),'features')
    rpt_ycor(nfdrop_yc)
else:
    print('No features were selected')


# _same as v0.04_
# >   4. Apply drop (or keep) list to the real features dataset

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


# ***
# <br>_**FDR + SU + Pearson**_ - multiple layers

#  ***

# _**filter_fdr(): univariate chi_sq layer**_

# In[ ]:


# filter_fdr: sklearn univariate chi-square test: FDR or FWE
#    fwe will select more to drop than fdr,
#    standard thresholds are 0.1, 0.05, 0.01; lower will select more to drop

# Requires: binned_df, numeric_labels
# Optional:
#    plvl : threshold (alpha) for chi_sq test  default=0.5
#    usefdr : boolean, fdr if True, else fwe   default=True
# Returns: f2y report for features to drop

# set test:
# tst = 'FDR'
# ufd = True
# = or =
tst = 'FWE'
ufd = False

print('\nUnivariate chi-sq',tst,'test')
uvdrop_yc = filter_fdr(binned_df, ynum, usefdr=ufd, plvl=0.01)

if (len(uvdrop_yc) > 1):
    uvdrop = get_filter(uvdrop_yc)

    print('Progressive_filtering: Dropping',len(uvdrop),'features')
    binned_df.drop(uvdrop, axis = 1, inplace = True)

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
#     features to keep are called "predominant features" in the FCBF paper [6]; 
#     they act as proxies for the highly correlated features to drop


# <br>_**filter_fcc(): FCBF-SU layer**_

# In[ ]:


# filter_fcc: FCBF-style, filter on feature-to-feature (f2f) correlations

# Requires: binned_df, numeric_labels
# Optional:
#    hipc : threshold for "high" f2f pearson correlation  default=0.7
#    hisu : threshold for "high" f2f su correlation       default=0.7
#    usesu : boolean, use su as metric if True, else pearson
# Returns: f2y report for features to drop
#          f2y report for features to keep
#          f2f above threshold report 

sudrop_yc, sukeep_yc, su_hicorr = filter_fcc(binned_df, ynum, usesu=True)

if (len(sudrop_yc) > 1):
    sudrop = get_filter(sudrop_yc)

    print('Progressive_filtering: Dropping',len(sudrop),'features')
    binned_df.drop(sudrop, axis = 1, inplace = True)

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


# filter_fcc: FCBF-style, filter on feature-to-feature (f2f) correlations

# Requires: binned_df, numeric_labels
# Optional:
#    hipc : threshold for "high" f2f pearson correlation  default=0.7
#    hisu : threshold for "high" f2f su correlation       default=0.7
#    usesu : boolean, use su as metric if True, else pearson
# Returns: f2y report for features to drop
#          f2y report for features to keep
#          f2f above threshold report

pcdrop_yc, pckeep_yc, pc_hicorr = filter_fcc(binned_df, ynum)

if (len(pcdrop_yc) > 1):
    pcdrop = get_filter(pcdrop_yc)

#    print('Progressive_filtering: Dropping',len(pcdrop),'features')
#    binned_df.drop(pcdrop, axis = 1, inplace = True)

    print('\nKept: Pearson Layer')
    rpt_ycor(pckeep_yc)
    print('\nFiltered: Pearson Layer')
    rpt_ycor(pcdrop_yc)
    print('\nHighly correlated features: Pearson Layer')
    rpt_fcor(pc_hicorr)
else:
    pcdrop = pcdrop_yc 
    print('\nPearson Layer\nNo features were selected')


# ***
# <i>same as v0.04</i>
# >   4. Apply drop lists to the real features dataset

# In[ ]:


## QLCFF: Quick Layered Correlation-based Feature Filter
## full layered drop filter for real DF 
QLCFFilter = []
QLCFFilter.extend(x for x in uvdrop if x not in QLCFFilter)
QLCFFilter.extend(x for x in pcdrop if x not in QLCFFilter)
QLCFFilter.extend(x for x in sudrop if x not in QLCFFilter)


# In[ ]:


# features_df_original = features_df
filtered_df = features_df.drop(QLCFFilter, axis = 1)
filtered_df.info(verbose=False)
# features_df = filtered_df


# In[ ]:





#  ***

#  ***
[1] Fayyad, U. M., and Irani, K. B. (1993). "Multiinterval discretization of 
    continuous-valued attributes for classifcation learning", Proc. 13th 
    Int. Joint Conference on Artifcial Intelligence, pp. 1022-1027
[2] Kerber R. (1992). "Chimerge: Discretization of numeric attributes", 
    Proc. 10th National Conference on Artifcial Intelligence (AAAI'92), pp. 123–128
[3] Dougherty J., Kohavi, R., and Sahami, M. (1995), “Supervised and unsupervised
    discretization of continuous features”, Proc. ICML 1995, pp. 194–202
[4] Yang, Y. and Webb, G. I. (2002), “A comparative study of discretization methods 
    for naive-bayes classifiers”, Proc. PKAW 2002, pp. 159-173
[5] Yang, Y. and Webb, G. I. (2001), “Proportional k-interval discretization 
    for naive-bayes classifiers”, in Machine learning: ECML 2001, pp. 564–575
[6] Lei Yu and Huan Liu (2003), "Feature Selection for High-Dimensional Data: 
    A Fast Correlation-Based Filter Solution", Proc. 20th ICML 2003, pp. 856-863