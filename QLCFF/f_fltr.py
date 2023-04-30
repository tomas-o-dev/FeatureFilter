#!/usr/bin/env python
# coding: utf-8

import numpy
import pandas

# only internal: get_ycor,get_fcor
from .f_mcor import mutual_info, symm_uncert 

def get_ycor(flist, indf, ingt):
    ctbl=[]
    for f in range(len(flist)):
        arr=numpy.array(indf[flist[f]])
        mi=mutual_info(arr, ingt)  
        su=symm_uncert(arr, ingt)
        bc = numpy.corrcoef(arr, ingt)[0,1]
        
        tmp=[flist[f],round(bc,4),round(su,4),round(mi,4)]
        if tmp not in ctbl:
            ctbl.append(tmp)
    return ctbl


def get_fcor(hicorr,indf,corrmtx,sucmx):
##  Note:
##  the UPPER HALF of the su=True CM is SU correlations 
##  the LOWER HALF is Pearson correlations

    ctbl=[]
    for x in range(len(hicorr)):
        arr=numpy.array(indf[hicorr[x][0]])
        brr=numpy.array(indf[hicorr[x][1]])

        pcc = corrmtx.loc[hicorr[x][0],hicorr[x][1]]
        if sucmx:
            suc = corrmtx.loc[hicorr[x][1],hicorr[x][0]]
        else:
            suc = symm_uncert(arr, brr)
        mi = mutual_info(arr, brr)

        tmp=[hicorr[x][0],hicorr[x][1],round(pcc,4),round(suc,4),round(mi,4)]

        if tmp not in ctbl:
            ctbl.append(tmp)
    return ctbl


# import external: filter_fcy,filter_fdr,filter_fcc

# floor filter: 
# keep if suy > 0.01 or pcy > 0.1

def filter_fcy(indf, ingt, minpc=0.1, minsu=0.01):
    ykeep = []
    ydrop = []
    
    cl = list(indf.columns)
    rr = get_ycor(cl, indf, ingt)
    for j in range(len(rr)):
        if (rr[j][2] > minpc) or (rr[j][2] > minsu):
            ykeep.append(rr[j])
        else:
            ydrop.append(rr[j])

    return ydrop, ykeep


def filter_fdr(dfin, gtin, t=0.01, usefdr=True):
    from .f_uvfs import uvtcsq

    highfdr=uvtcsq(dfin, gtin, t, usefdr)
    if len(highfdr) > 1:
        fdr_yc=get_ycor(highfdr, dfin, gtin)
    else:
        fdr_yc=highfdr
        
    return fdr_yc     


def filter_fcc(dfin, ingt, t=0.7, usesu=False):
    from .f_mcor import mulcol

    dfdrop, proxies, hicorr, cormtx = mulcol(dfin, ingt, t, usesu)
    if len(dfdrop) > 1:
        fccd_yc=get_ycor(dfdrop, dfin, ingt)
        fccp_yc=get_ycor(proxies, dfin, ingt)
        fcc_fxc=get_fcor(hicorr, dfin, cormtx, usesu)
    else:
        fccd_yc=dfdrop
        fccp_yc=dfdrop
        fcc_fxc=dfdrop

    return fccd_yc, fccp_yc, fcc_fxc     

