# FeatureFilter
Quick Layered Correlation-based Feature Filtering

General library requirements:
* Dataframe with only numeric columns
* Numeric class labels in "array-like" with shape (n,1)
* Binary classification (not multiclass or multilabel)"
  
Examples are in /QLCFFdemo/: .py and .ipynb

-- Use: --
```
# import the local library
# add parent folder path where lib folder is
import sys
if ".." not in sys.path:import sys; sys.path.insert(0, '..') 

## or ## from QLCFF import *
# 
from QLCFF import mkbins
from QLCFF import filter_fcy, filter_fdr, filter_fcc
from QLCFF import get_filter, rpt_ycor, rpt_fcor
```

-- Functions --

`mkbins()`: applies sklearn KBinsDiscretizer(strategy='uniform', encode='ordinal')<br>
https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-discretization

R doc for `miMRMR` asserts 10 bins is consistent with the literature
20 bins with uniform strategy means each bin is 5% of observed values
   (ideally - it is really max_bins, uniform means all are the same size) 

```
### def mkbins(indf, nb=20, detail=False):
# Requires: features dataframe
#    nb =  maximum number of bins, default 20
#    detail = print report, boolean, default False
# Returns: binned (discretized) dataframe
```

`filter_fcy()`: naive filter, drop all with low feature-to-label (f2y) correlations

`filter_fdr()`: sklearn univariate chi-square test: FDR or FWE<br>
https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection

naive filter and univariate tests both filter features on the premise that
low correlation with the target labels means low utility for distinguishing class membership

chi_sq is a formal test for independence:<br>FWE will select more to drop than FDR; lower threshold will select more than higher<br>naive filter will select all from either univariate test, and more

```
### def filter_fcy(indf, ingt, minpc=0.1, minsu=0.01):
# naive filter: keep if f2y_pc >= minpc or f2y_su >= minsu
# Requires: features_df, numeric_labels
#    minpc = threshold (alpha) for pearson correlation
#    minsu = threshold (alpha) for symmetric uncertainty   
# Returns: f2y report for features to drop
```
```
### def filter_fdr(dfin, gtin, t=0.01, usefdr=True):
# Requires: features_df, numeric_labels
#    t = threshold (alpha) for chi_sq test, sklearn default is 0.5
#    usefdr = test, boolean, fdr if True, else fwe  
# Returns: f2y report for features to drop
```
`filter_fcc()`: FCBF-style, filter on feature-to-feature (f2f) correlations<br>
using Pearson correlation (PC) or symmetric uncertainty (SU)

FCBF-Pearson and FCBF-SU Layers use the same code, metric depends on the boolean argument: `usesu`<br>
(just use appropriate names for the return values)

Features to keep are called "predominant features" in the FCBF paper; they act as proxies for the highly correlated features to drop (see Lei Yu & Huan Liu, Proc. 20th ICML 2003)
```
### def filter_fcc(dfin, ingt, t=0.7, usesu=False):
# Requires: features_df, numeric_labels
#    t = threshold (alpha) for "high" f2f correlation
#        standard for detecting multicollinearity is 0.7
#    usesu = metric, boolean, su if True, else pearson 
# Returns: f2y report for features to drop
#          f2y report for features to keep
#          f2f above threshold report 
```

These functions process the return values of the filter_()<br>
`get_filter()`: returns list of features from f2y report<br>
`rpt_ycor()`: print feature-to-label (f2y) correlations<br> 
`rpt_fcor()`: print feature-to-feature (f2f) correlations 

Examples are in /QLCFFdemo/: .py and .ipynb
