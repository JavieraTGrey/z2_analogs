from astropy.coordinates.matching import match_coordinates_sky
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS, utils
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def match(detcoord, catalogcoord):
    """Matches every detection to the closest detection in the catalog.
    """
    index, sep2d, _ = match_coordinates_sky(detcoord, catalogcoord)
    return index, sep2d


def sep_constraint(sep, constraint):
    ''' Separation constraint on the separation list between sources.'''
    over = sep > constraint
    under = sep < constraint
    return under, over


def cut_catalog(matchcoord, catalogcoord, index, sep):
    '''Cut the catalog according to the match index and the
    separation constraint given'''
    catalog = catalogcoord[index[sep]]
    match = matchcoord[sep]
    return catalog, match


def catalog_comparison(max_sep, matchcoord, catalogcoord):
    """ Compare the catalog matches and the detections by a separation
        constraint. The function looks for the matches between the catalog
        and the detections, and by the given constraint looks for the perfect
        matches, the detections without matches, and the catalog sources
        that are unmacthed.
    """
    index, sep1 = match(matchcoord, catalogcoord)
    index2, sep2 = match(catalogcoord, matchcoord)

    under_sep, over_sep = sep_constraint(sep1, max_sep)
    _, over_switch = sep_constraint(sep2, max_sep)

    # All matches
    catalog_match = catalogcoord[index]

    # Perfect match
    under = cut_catalog(matchcoord, catalogcoord, index, under_sep)

    # Detections without matches
    over = cut_catalog(matchcoord, catalogcoord, index, over_sep)

    # Catalog sources unmatched with detections
    unmatched = cut_catalog(catalogcoord, matchcoord, index2, over_switch)

    return catalog_match, under, over, unmatched
