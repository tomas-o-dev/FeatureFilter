# FeatureFilter
#### Quick Layered Correlation-based Feature Filtering

General library requirements (Release 1.0):
* Dataframe of features (text values may be one-hot encoded)
* Class labels in np.ndarray or pd.Series with shape (n,1)
* Binary classification (not multiclass or multilabel)

Workflow:
Correlation-based feature filtering has four steps: preprocessing, 
discretization, calculating correlations, and feature reduction.

Here the first two steps are implemented in the ` Discretizer ` class, 
and the second two steps in the ` qlcfFilter ` class. 
They work SciKit-Learn style (instantiate, fit, transform) 
and can be used in a pipeline.

#### Quick Start: 
```
# import the local library:
# add parent folder path where lib folder is
import sys
if ".." not in sys.path:import sys; sys.path.insert(0, '..') 

from QLCFF import Discretizer, qlcfFilter

dzdf = Discretizer().fit_transform(features_train, labels_train) 

fltrs = ['FDR', 'FWE', 'FCBF-PC']
ffdf = qlcfFilter().fit_transform(dzdf, labels_train, 
                                  fltrs, features_train)
```
Examples are in QLCF_demo .py and .ipynb


#### The Discretizer Class

* ` dtzr = Discretizer(numjobs= -2, msglvl=5)   #Initialise `
  - Requires : none
  - Optional : joblib Parallel(n_jobs=, verbose=)
 
* ` dtzr.fit(X, y)    # Calls the preprocessor `
  - Requires : features as pd.dataframe, labels as array-like
  - Optional : none
  - X : preprocessor  
    1. selects only column dtypes np.number and pd or np boolean
    2. normalizes all columns with signed dtypes to positive numbers
    3. normalizes all columns with boolean dtypes to zero//one
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
    > The default value ` mkbins=hgrm ` applies ` numpy.histogram(feature, bins='auto') `, 
    > and repeatedly folds lower bins into the next higher one until there are a maximum of 12 for the feature.<br> 
      Otherwise, the valid values combine an algorithm for calculating the bin 
      edges (cutpoints) with a method for determining the maximum number of bins. 
      
     calculate edges | number of bins
      --------------- | ---------------
      unif: uniform [numpy.linspace()] |ten:  always ten [3,4]
      mdlp: MDLP algorithm  [1]        |sqrt: sqrt(len(feature))   [5]
      chim: ChiMerge algorithm  [2]    |log:  log10(len(feature))  [3]

* After transform():
  - the processed dataframe is an attribute<br>` dtzr.binned_df.head() `
  - the dict of bin edges is an attribute<br>` dtzr.cutpoints `
  - note: distribution of values within bins<br>` numpy.bincount(dtzr['num_compromised'].values) `

#### The qlcfFilter Class

* ` ffltr = qlcfFilter()   #Initialise `
  - Requires : none
  - Optional : none
 
* ` ffltr.fit(X, y, filters, plvl=0.5, minpc=0.035, minsu=0.0025, hipc=0.82, hisu=0.7) `
  - Requires : discretizer.binned_df, labels as array-like, list of one or more filters
  - Optional : *varies depending on filters selected
  - #### Filters
    A list with one or more of ` 'Floor', 'FDR', 'FWE', 'FCBF-SU', 'FCBF-PC' `<br>
    The list is processed in order with progressive filtering
    - ` 'Floor' `: filters on the basis that low correlation with the target labels (f2y) means low utility for distinguishing class membership. Keeps features that have correlation > a threshold (the defaults were selected through experimentation).
      - Optional : 
        - ` minpc ` : threshold for pearson correlation
        - ` minsu ` : threshold for symmetric uncertainty 
    - ` 'FDR', 'FWE' `: sklearn univariate chi-square test; selects features to keep based on an upper bound on the expected false discovery rate. ` fwe ` will select more to drop than ` fdr `, and lower thresholds will also select more to drop. The floor filter will select all from either univariate test, and more.
      - Optional : 
        - ` plvl ` : chi-square threshold (alpha), standard values are 0.01, 0.05, 0.1
    - ` 'FCBF-SU', 'FCBF-PC' `: FCBF-style, filter on feature-to-feature (f2f) correlations. Given a group of features with high cross-correlations, keep the one with the highest (f2y) as a proxy for the others (FCBF paper [6] calls this the "dominant feature"). The standard threshold for multicolliniarity is > 0.7, the defaults were selected through experimentation. 
      - Optional : 
        - ` hipc ` : threshold for "high" f2f pearson correlation
        - ` hisu ` : threshold for "high" f2f symmetric uncertainty 

    To create layered feature selection filters, apply either ` 'Floor' ` or ` 'FDR', 'FWE' ` before ` 'FCBF-SU' ` and/or  ` 'FCBF-PC' `
 
 * After fit():
   - the consolidated drop list is an attribute<br>` ffltr.QLCFFilter `
   - reporting methods are available:
     - ` ffltr.get_f2y_report(kd='drop') `<br>print feature-to-label (f2y) correlations 
     - ` fyd = ffltr.get_f2y_dict(kd='drop') `<br>returns a dict of correlations for each filter
       - Optional : ` kd = 'keep' ` or ` 'drop' `
     - ` ffltr.get_f2f_report() `<br>print feature to feature (f2f) correlations above threshold report
     - ` ffd = ffltr.get_f2f_dict() `<br>returns a dict of f2f correlations checked by each filter
       - f2f is only available for ` 'FCBF-SU' ` or ` 'FCBF-PC' `

* ` reduced_df = ffltr.transform(Xdf) `
  - Returns  : Xdf after applying the consolidated drop list
  - Requires : actual pd.dataframe for clf.fit_predict()
  - Optional : none

Examples are in QLCF_demo .py and .ipynb
    
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
