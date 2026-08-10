# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 21:19 2022

@author: BN
"""

"Berni's color palette is for tritanopia I think"
c = ["#00B9E4", "#FF0035", "#613AD6", "#FFC602"]

"z: Zesty, c: corporate, e: elegant, r: retro" 
"p: protanopia (red), d: deuteranopia (green)"

cz = ["#F5793A", "#A95AA1", "#85C0F9", "#0F2080"]
czp = ["#AE9C45", "#6073B1", "#A7B8F8", "#052955"]
czd = ["#C59434", "#6F7498", "#A3B7F9", "#092C48"]

cc = ["#BDB8AD", "#EBE7E0", "#C6D4E1", "#44749D"]
ccp = ["#BDB6AB", "#EDE6DE", "#D1D0DE", "#636D97"]
ccd = ["#CDB1AD", "#FADFE2", "#DECBE3", "#5D6E9E"]

ce = ["#ABC3C9", "#E0DCD3", "#CCBE9F", "#382119"]
cep = ["#BEBCC5", "#E2DAD1", "#C9BD9E", "#2E2B21"]
ced = ["#CAB8CB", "#F4D4D4", "#DCB69F", "#342A1F"]

cr = ["#601A4A", "#EE442F", "#63ACBE", "#F9F4EC"]
crp = ["#2A385B", "#8B7F47", "#9C9EB5", "#FAF2EA"]
crd = ["#383745", "#A17724", "#9E9CC2", "#FDF0F2"]


import matplotlib.pyplot as plt

def texPlot(state):
    if state == "on":
        plt.rcParams.update({
        'lines.color':'black',
        'font.family':'serif',
        'font.weight':'normal',
        'text.color':'black',
        'text.usetex':True,
        'axes.edgecolor':'black',
        'axes.linewidth':1.0,
        'axes.titlesize':'x-large',
        'axes.labelsize':20,
        'axes.labelcolor':'black',
        'xtick.labelsize':'x-large',
        'xtick.minor.width':1.0,
        'xtick.major.width':1.0,
        'ytick.major.size':7,
        'ytick.minor.size':4,
        'ytick.major.pad':8,
        'ytick.minor.pad':8,
        'ytick.labelsize':'x-large',
        'ytick.minor.width':1.0,
        'ytick.major.width':1.0,
        'legend.numpoints':1,
        'legend.fontsize':'x-large',
        'legend.shadow':False,
        'legend.frameon':False})
    if state == "off":
        plt.rcParams.update(plt.rcParamsDefault)