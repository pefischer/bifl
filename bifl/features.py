"""
Extract feature batteries from gauss pyramids
"""

import cv
import logging
from utils import saveIm
from mods import *
from cpy import *

stageLogger = logging.getLogger("features.stage")
def stage(lum, sat, rg, by):
    pixel_per_degree = 41 
    wsize =  5 * pixel_per_degree # original bilc value: ws = 251
    stageLogger.debug("Kernel size for texture contrast: %s" % wsize)
    lumc = contrast(lum)
    lumt = contrast(lumc, wsize) # texture contrast
    sats = smooth(sat)
    satc = contrast(sat)
    satt = contrast(satc, wsize)
    rgc = contrast(rg)
    rgt = contrast(rgc, wsize)
    byc = contrast(by)
    byt = contrast(byc, wsize)
    sob = sobel(lum)
    sobs = smooth(sob)
    lums = smooth(lum)
    rgs = smooth(rg)
    bys = smooth(by)
    id0, id1, id2 = intdim(lum)
    idX = add(zscale(id0), zscale(id2))
    return dict(lumc=lumc, lumt=lumt, satc=satc, satt=satt, rgc=rgc, rgt=rgt,
            byc=byc, byt=byt, sobs=sobs, lums=lums, id0=id0, id1=id1, id2=id2,
            rgs=rgs, sats=sats, bys=bys, idX=idX,)


def noscale(indict):
    return indict


def zscaledict(indict):
    return dict((n, zscale(m)) for n, m in indict.items())


def histeqdict(indict):
    def eq(inmat):
        m = zscale(inmat)
        return equalize(m)
    return dict((n, eq(m)) for n, m in indict.items())


def pyramid(lsrb, count=3, scaler=noscale):   #called from extract, with zscaledict as scaler
    """
    run stage in a downwards pyramid for ``count`` times,
    scale each map with ``scaler``,
    return list with one dict per pyramid level
    """
    features = [scaler(stage(*lsrb))]  # zTransform all images that come from stage
    if count == 1:
        return features
    lsrb = list(pyrsdown(*lsrb))
    features += pyramid(lsrb, count - 1, scaler)
    return features


def base(im, layers):
    """make sure im's dimensions are multiples of 2**layers"""
    mod = 2 ** layers  # 2³ makes 8
    if im.width % mod != 0 or im.height % mod != 0:
        im = cv.GetSubRect(im, (  # returns matrix header corresponding to a specific rectangle of the input array
            0, 0,
            im.width - im.width % mod,  # it allows the user to treat a rectangular part of input array as a standalone array
            im.height - im.height % mod,))
    return cv.GetImage(im)


def extract(image, pyr_levels=3, scaler=zscaledict):
    # This is called by Niklas' featmat.py in _extract_feature()
    """extract features from ``image``"""
    image = base(image, pyr_levels)
    lsrb = colorsplit(image)
    return pyramid(lsrb, pyr_levels, scaler=scaler)
