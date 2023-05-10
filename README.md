# FeatureFilter
Quick Layered Correlation-based Feature Filtering

General library requirements (Release 1.0):
* Dataframe of features (text values may be one-hot encoded)
* Class labels in np.ndarray or pd.Series with shape (n,1)
* Binary classification (not multiclass or multilabel)

Workflow:
Correlation-based feature filtering has four steps: preprocessing, 
discretization, calculating correlations, and feature reduction.

Here the first two steps are implemented in the Discretizer class, 
and the second two steps in the qlcfFilter class. Discretizer and 
qlcfFilter work SciKit-Learn style (instantiate, fit, transform) 
and can be used in a pipeline.

#### Quick Start: 
```
# import the local library:
# add parent folder path where lib folder is

import sys
if ".." not in sys.path:import sys; sys.path.insert(0, '..') 

from QLCFF import Discretizer, qlcfFilter

dzdf = Discretizer(numjobs= -2, msglvl=5).fit_transform(features_train, 
                                                        labels_train, 
                                                        mkbins='mdlp-log', 
                                                        detail=True)

fltrs = ['FDR', 'FWE', 'FCBF-PC']
ffdf = qlcfFilter().fit_transform(dzdf, 
                                  labels_train, 
                                  fltrs, 
                                  features_train)
```
Examples are in QLCF_docs .py and .ipynb


#### The Discretizer Class

* ` dtzr = Discretizer(numjobs= -2, msglvl=5)   #Initialise `
  - Requires : none
  - Optional : joblib Parallel(n_jobs=, verbose=)
 
* ` dtzr.fit(X, y)    # Calls the preprocessor `
  - Requires : features as pd.dataframe, labels as array-like
  - Optional : none
  - X : preprocessor  
    1. selects only column dtypes np.number and pd or np boolean
    2. nomalizes all columns with signed dtypes to positive numbers
    3. nomalizes all columns with boolean dtypes to zero//one
  - y : Text labels are converted with sklearn LabelEncoder()

* After fit(), the preprocessed dataframe is an attribute<br>` dtzr.prebin_df.head() `

* ` _ = dtzr.transform(mkbins='hgrm', detail=False) `
  - Returns  : discretized df
  - Requires : none
  - Optional : binning strategy, default or one of
    > ` 'unif-ten'  'unif-log'  'unif-sqrt' `<br> 
      ` 'mdlp-ten'  'mdlp-log'  'mdlp-sqrt' `<br> 
      ` 'chim-ten'  'chim-log'  'chim-sqrt' `
  - Optional : (boolean) print binning report  
  - #### Binning Strategy
    > The default value ` mkbins=hgrm ` applies ` numpy.histogram(feature, bins='auto') `.<br> 
      Otherwise, the valid values combine an algorithm for calculating the bin 
      edges (cutpoints) with a method for determining the maximum number of bins. 
      
     calculate edges | number of bins
      --------------- | ---------------
      unif: uniform [numpy.linspace()] |ten:  always ten [3,4]
      mdlp: MDLP algorithm  [1]        |sqrt: sqrt(len(np.unique(feature)))   [5]
      chim: ChiMerge algorithm  [2]    |log:  log10(len(np.unique(feature)))  [3]

* After transform():
  - the processed dataframe is an attribute<br>` dtzr.binned_df.head() `
  - the dict of bin edges is an attribute<br>` dtzr.cutpoints `
  - note: distribution of values within bins<br>` numpy.bincount(dtzr['num_compromised'].values) `
