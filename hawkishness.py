from random import random, choice, sample
from statistics import median
from itertools import combinations
import csv
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from cycler import cycler
from functools import lru_cache as memoize
import colorsys
import itertools

def p_win(pA, pB):
    return ( pA / (pA + pB), pB / (pA + pB) )
'''
epsilon = 1e-15
def utility(c, a):
    interim = -np.exp(-c*a) / a
    interim[~np.isfinite(interim)] = c
    return interim
'''

from scipy.stats import norm
def eval(riskAs, riskBs, hawkAs, hawkBs, powerAs, powerBs):
    war_a = hawkAs * norm.cdf(powerAs * np.exp(-riskAs), loc=powerBs,scale=powerBs)
    war_b = hawkBs * norm.cdf(powerBs * np.exp(-riskBs), loc=powerAs,scale=powerAs)
    return 1 - (1 - war_a)*(1 - war_b)

prec=0.02

def color(war_prob):
    #color = (not_peace, not_peace - war, 1.0 - not_peace)
    #color = (not_peace, 1.0 - war, 0)
    return np.array([war_prob, 1.0 - war_prob, 0])


def draw(plot=True, read=False, save=False, file='data.csv', powera=1, powerb=1, vriska=True, vriskb=False, vhawka=True, vhawkb=False):
    assert not save and not read
    powera = np.array([powera])
    powerb = np.array([powerb])
    assert sum([vriska, vriskb, vhawka, vhawkb]) == 2
    labels = np.array(["Country 1 Risk Aversion", "Country 2 Risk Aversion", "Country 1 Hawkishness", "Country 2 Hawkishness"])[[vriska, vriskb, vhawka, vhawkb]]
    assert len(labels) == 2
    riska = np.arange(0.0,1.0 + prec,prec) if vriska else np.array([0])
    riskb = np.arange(0.0,1.0 + prec,prec) if vriskb else np.array([0])
    hawka = np.arange(0.0,1.0 + prec,prec) if vhawka else np.array([1])
    hawkb = np.arange(0.0,1.0 + prec,prec) if vhawkb else np.array([1])

    if read:
        pass
    #    X = np.loadtxt(file, delimiter=',')
    else:
        riskAs, riskBs, hawkAs, hawkBs, powerAs, powerBs = np.meshgrid(riska, riskb, hawka, hawkb, powera, powerb, indexing='ij')
        X = eval(riskAs, riskBs, hawkAs, hawkBs, powerAs, powerBs)
        #if save:
        #    np.savetxt(file, X, fmt='%.3f', delimiter=',')
    if plot:
        index = (slice(None) if vriska else 0, slice(None) if vriskb else 0, slice(None) if vhawka else 0, slice(None) if vhawkb else 0, 0,0)
        Y = np.vectorize(color, signature='()->(n)')(X[index])
        plt.close()
        fig, ax = plt.subplots()
        ax.imshow(Y, origin='lower', extent=(0.0, 1.0, 0.0, 1.0))
        ax.set_xlabel(labels[0], size=14)
        ax.set_ylabel(labels[1], size=14)
        ax.text(-0.02,0,'Low', size=12, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
        ax.text(-0.02,1,'High', size=12, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
        ax.text(0,-0.02,'Low', size=12, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
        ax.text(1,-0.02,'High', size=12, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
        ax.tick_params(bottom=False, left=False, top=False, right=False, labelbottom=False, labelleft=False)
        plt.subplots_adjust(left=0.06, bottom=0.06, right=0.76, top=0.98)
        #plt.legend(handles=[mpatches.Patch(color='red', label='War'), mpatches.Patch(color='gold', label='Standoff'), mpatches.Patch(color='limegreen', label='Peace')], loc=(1.05, 0.8))
        plt.legend(handles=[mpatches.Patch(color='red', label='War'), mpatches.Patch(color='limegreen', label='Peace')], loc=(1.05, 0.8))
        plt.show()
