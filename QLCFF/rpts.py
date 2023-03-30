#!/usr/bin/env python
# coding: utf-8


def rpt_ycor(data):
    colwid = max(len(str(word)) for row in data for word in row)+1    # +i for padding
    print("{: <{colwid}} {: >12} {: >12} {: >12}".format(
        '--Feature--','PCy   ','SUy   ','MIy   ',colwid=colwid))
    for k in range(len(data)):
        print("{: <{colwid}} {: >12} {: >12} {: >12}".format(
            data[k][0],data[k][1],data[k][2],data[k][3],colwid=colwid))


def rpt_fcor(data):
    colwid = max(len(str(word)) for row in data for word in row)+1    # +i for padding
    print("{: <{colwid}} {: <8} {: <8}  {: <8}   {: <{colwid}}".format(
        '--Feature--','  PCf ','  SUf',' MIf','--Feature--',colwid=colwid))
    for k in range(len(data)):
        print("{: <{colwid}} {: >8} {: >8} {: >8}   {: <{colwid}}".format(
            data[k][0],data[k][2],data[k][3],data[k][4],data[k][1],colwid=colwid))
#    print ("  ".join(str(word).ljust(col_width) for word in row))  # NB: delim.join 


def get_filter(ctbl):
    flist=[]
    for k in range(len(ctbl)):    
        if ctbl[k][0] not in flist:
            flist.append(ctbl[k][0])
    return flist

