# FeatureFilter
Quick Layered Correlation-based Feature Filtering

General library requirements (Release 0.05):
* Dataframe of features (text values may be one-hot encoded)
* Class labels in np.ndarray or pd.Series with shape (n,1)
* Binary classification (not multiclass or multilabel)

Workflow:
1. Instantiate a discretizer
2. Get the binned dataframe from the discretizer
3. Apply filters to the binned dataset
4. Apply drop (or keep) lists to the real features dataset

Use: 
```
# import the local library
# add parent folder path where lib folder is
import sys
if ".." not in sys.path:import sys; sys.path.insert(0, '..') 

from QLCFF import unifhgm, MDLP, ChiMerge
# three distinct discretizers can be instantiated

from QLCFF import filter_fcy, filter_fdr, filter_fcc
# filter_fcy: floor filter, feature-to-label (f2y) correlations  
# filter_fdr: sklearn univariate chi-square test: FDR or FWE
# filter_fcc: FCBF-style, filter on feature-to-feature (f2f) correlations
#       using Pearson correlation (PC) or symmetric uncertainty (SU)

from QLCFF import get_filter, rpt_ycor, rpt_fcor
# get_filter: returns list of features from f2y report
# rpt_ycor: print feature-to-label (f2y) correlations 
# rpt_fcor: print feature-to-feature (f2f) correlations
```
Examples are in /QLCFFdemo/: .py and .ipynb

Discretizers
* unifhgm: uniform (np.linspace()) or histogram<br>
  - Optional : 
  - mkbins in `['ten','sqrt','log','hgrm']`<br>
  default is `hgrm`: applies `np.histogram(feature, bins='auto')`
* MDLP algorithm  [1]<br>
  - Optional : 
    - mkbins in `['ten','sqrt','log']`, 
    - joblib processes, verbose level; defaults: `numjobs=1, msglvl=0` 
* ChiMerge algorithm  [2]<br>
  - Optional : 
    - mkbins in `['ten','sqrt','log']`, 
    - joblib processes, verbose level; defaults: `numjobs=1, msglvl=0` 
* ten [3,4]:  number of bins is always ten - default for MDLP and ChiMerge
* sqrt [5]: number of bins is `sqrt(len(np.unique(feature)))`
* log [3]:  number of bins is `log10(len(np.unique(feature)))`
```
# IMPORTANT: instantiate, then call fit() or fit_transform()
hgmb = unifhgm(mkbins='hgrm')

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
         
# binned bataframe is an attribute
X_hgmbinz = hgmb.binned_df
#     transform() is just a getter method, with optional detail

# detailed list of bin edges is an attribute
hgmb.cutpoints
```
Filters
* floor filter `filter_fcy()` and univariate tests `filter_fdr()` both filter features on the basis that low correlation with the target labels means low utility for distinguishing class membership. 
* chi_sq is a formal test for independence, `fwe` will select more to drop than `fdr`, and lower thresholds will also  select more to drop. The floor filter will select all from either univariate test, and more.
* to create layered feature selection filters, apply one of these 3 before<br>
`filter_fcc()` : FCBF-style, filter on feature-to-feature (f2f) correlations<br>using Pearson correlation (PC) or symmetric uncertainty (SU)

<br>`filter_fcy()` : floor filter, feature-to-label (f2y) correlations, drop<br>
`if f2y_pc < minpc or f2y_su < minsu`
* Requires: binned_df, numeric_labels
* Optional:
  - minpc : threshold for pearson correlation    default=0.1
  - minsu : threshold for symmetric uncertainty  default=0.01 
* Returns: 
  - f2y report for features to drop
  - f2y report for features to keep

<br>`filter_fdr()` : sklearn univariate chi-square test: FDR or FWE
* Requires: binned_df, numeric_labels
* Optional:
  - plvl : threshold (alpha) for chi_sq test  default=0.5<br>
standard thresholds are 0.1, 0.05, 0.01; lower will select more to drop
  - usefdr : boolean, fdr if True, else fwe   default=True
* Returns: 
  - f2y report for features to drop

<br>`filter_fcc()`: FCBF-style, filter on feature-to-feature (f2f) correlations<br>
FCBF-Pearson and FCBF-SU layers use the same code, so be sure to use appropriate names for the return values
* Requires: binned_df, numeric_labels
* Optional:
  - hipc : threshold for "high" f2f pearson correlation  default=0.7
  - hisu : threshold for "high" f2f su correlation       default=0.7
  - usesu : boolean, use su as metric if True, else pearson
* Returns: 
  - f2y report for features to drop
  - f2y report for features to keep
  - f2f above threshold report

Examples are in /QLCFFdemo/: .py and .ipynb
