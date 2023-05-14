#!/usr/bin/env python
# coding: utf-8

# > **Essential ML process for Intrusion Detection**
# <br>` python  3.7.13    scikit-learn  1.0.2 `
# <br>`numpy   1.19.5          pandas  1.3.5`

# **Import the main libraries**

# In[ ]:


import numpy
import pandas

from time import time

import os
data_path = '../datasets'


# **Import the Dataset**

# In[ ]:


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
features_train.reset_index(inplace=True, drop=True)

features_test = features_df.iloc[len(train_df):,:].copy()     # X_test
features_test.reset_index(inplace=True, drop=True)

# Restore the train // test split: slice 1 Series into 2 
labels_train = labels_df[:len(train_df)]               # y_train
labels_train.reset_index(inplace=True, drop=True)

labels_test = labels_df[len(train_df):]                # y_test
labels_test.reset_index(inplace=True, drop=True)


# In[ ]:





# ***
# **QLCFF: Quick Layered Correlation-based Feature Filter**<br>
# > **_library requirements:_**<br>
#     * Dataframe of features (text values may be one-hot encoded)<br>
#     * Class labels in np.ndarray or pd.Series with shape (n,1)<br>
#     * Binary classification (not multiclass or multilabel)<br><br>
# > **_workflow:_**<br>
# Workflow: Correlation-based feature filtering has four steps: preprocessing, discretization, calculating correlations, and feature reduction.
# Here the first two steps are implemented in the Discretizer class, and the second two steps in the qlcfFilter class.

# ***
# _**import the local library**_

# In[ ]:


# add parent folder path where lib folder is
import sys
if ".." not in sys.path:import sys; sys.path.insert(0, '..') 


# In[ ]:


from QLCFF import Discretizer, qlcfFilter


# ***
# _**the discretizer**_

# In[ ]:


dtzr = Discretizer(numjobs= -2, msglvl=5)   # Initialise
# Requires : none
# Optional : joblib Parallel(n_jobs=, verbose=)


# In[ ]:


dtzr.fit(features_test, labels_test)    # Calls the preprocessor

# Requires : features as pd.dataframe, labels as array-like
# Optional : none
#  X : preprocessor
#    1. selects only column dtypes np.number and pd or np boolean
#    2. normalizes all columns with signed dtypes to positive numbers
#    3. normalizes all columns with boolean dtypes to zero//one
#  y : Text labels are converted with sklearn LabelEncoder()


# In[ ]:


# After fit(), the preprocessed dataframe is an attribute
dtzr.prebin_df.head()


# In[ ]:


# the discretized dataframe is an attribute after transform()
_ = dtzr.transform(mkbins='hgrm', detail=True)

# Returns  : discretized df
# Requires : none
# Optional : binning strategy, default or one of
#     'unif-ten'  'unif-log'  'unif-sqrt'
#     'mdlp-ten'  'mdlp-log'  'mdlp-sqrt'
#     'chim-ten'  'chim-log'  'chim-sqrt'
# Optional : (boolean) print binning report

# Binning Strategy
# The default value mkbins=hgrm applies numpy.histogram(feature, bins='auto'), 
# and repeatedly folds lower bins into the next higher one until there are a 
# maximum of 12 for the feature.

# Otherwise, the valid values combine an algorithm for calculating the bin  
# edges (cutpoints) with a method for determining the maximum number of bins.
#     calculate edges
#         unif: uniform [numpy.linspace()]
#         mdlp: MDLP algorithm 
#         chim: ChiMerge algorithm
#     number of bins
#         ten: always ten 
#         sqrt: sqrt(len(feature)) 
#         log: log10(len(feature))


# In[ ]:


## After transform():

# the discretized dataframe is an attribute
dtzr.binned_df.head()


# In[ ]:


# the dict of bin edges is an attribute
dtzr.cutpoints


# In[ ]:


# note: distribution of values within bins
for col in dtzr.binned_df.columns:
    print(col, numpy.bincount(dtzr.binned_df[col].values))


# ***
# _**the feature filter**_

# In[ ]:


ffltr = qlcfFilter()   #Initialise
# Requires : none
# Optional : none


# In[ ]:


# Create layered feature selection filters 

# most informative
fltrs = ['FDR', 'FWE', 'Floor', 'FCBF-SU', 'FCBF-PC']

# quick way to drop the most
#fltrs = ['Floor', 'FCBF-PC']

ffltr.fit(dtzr.binned_df, labels_test, fltrs)


# In[ ]:


## ffltr.fit(X, y, filters, plvl=0.5, minpc=0.035, minsu=0.0025, hipc=0.82, hisu=0.7)

# Requires : discretizer.binned_df, labels as array-like, list of one or more filters
# Optional : *varies depending on filters selected

# Filters
# A list with one or more of 
#     'Floor', 'FDR', 'FWE', 'FCBF-SU', 'FCBF-PC'
# The list is processed in order with progressive filtering

## 'Floor': filters on the basis that low correlation with the target labels (f2y) 
#          means low utility for distinguishing class membership. Keeps features that have 
#          f2y correlation greater than a threshold value.
#          Optional :
#              minpc : threshold for pearson correlation
#              minsu : threshold for symmetric uncertainty
## 'FDR', 'FWE': sklearn univariate chi-square test; selects features to keep 
#          based on an upper bound on the expected false discovery rate. 
#          fwe will select more to drop than fdr, 
#              lower thresholds will also select more to drop. 
#          The floor filter will select all from either univariate test, and more.
#          Optional :
#              plvl : chi-square threshold (alpha), standard values are 0.01, 0.05, 0.1
## 'FCBF-SU', 'FCBF-PC': FCBF-style, filter on feature-to-feature (f2f) correlations. 
#          Given a group of features with high cross-correlations, keep the one with 
#          the highest (f2y) as a proxy for the others. 
#          Optional :
#              hipc : threshold for "high" f2f pearson correlation
#              hisu : threshold for "high" f2f symmetric uncertainty


# In[ ]:


# After fit() -

# the consolidated drop list is an attribute
ffltr.QLCFFilter


# In[ ]:


# reporting methods are available

# print feature-to-label (f2y) correlations
# Optional : kd = 'keep' or 'drop'
ffltr.get_f2y_report(kd='drop')


# In[ ]:


# returns a dict of correlations for each filter
# Optional : kd = 'keep' or 'drop'
fyd = ffltr.get_f2y_dict(kd='drop')


# In[ ]:


# print feature to feature (f2f) correlations above threshold report
# only available for 'FCBF-SU' or 'FCBF-PC'
ffltr.get_f2f_report()


# In[ ]:


# returns a dict of f2f correlations checked by each filter
# only available for 'FCBF-SU' or 'FCBF-PC'
ffd = ffltr.get_f2f_dict()


# In[ ]:


# apply the consolidated drop list
reduced_df = ffltr.transform(features_test)

# Requires : actual pd.dataframe for clf.fit_predict()
# Optional : none

reduced_df.info(verbose=False)


# ***
# _**fit_transform**_<br>
# > _instantiate separately if you want attributes & reports_

# In[ ]:


# fit_transform()
dtzdf = Discretizer(numjobs= -2, msglvl=5).fit_transform(features_test,
                                                         labels_test,
                                                         mkbins='mdlp-log',
                                                         detail=False)


# In[ ]:


fltrs = ['Floor']
filtered_df = qlcfFilter().fit_transform(dtzdf,
                                         labels_test,
                                         fltrs,
                                         features_test)


# In[ ]:


filtered_df.info(verbose=False)


#  ***

#  ***
