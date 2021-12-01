#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 16:27:49 2021

@author: tom
"""


###################
'''For Srayan:
The red boxes are only on the final graph where I've put all the constants together, it was just easier doing them all with a for loop so I'll work on putting it in the individual plots. The boxes use matplotlib.patches.Polygon, it was easier drawing squares than using fill_between on both axes. It did mean I had to put in each coordinate of the 4 corners of the square which will be a bit messy to re-adjust (especially since I always manage to make messy code)
'''
##############################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import iminuit
from iminuit import Minuit
import json

si_preds = {}
pi_preds = {}
with open("All_Data_CSV/std_predictions_si.json","r") as _f:
    si_preds = json.load(_f)
with open("All_Data_CSV/std_predictions_pi.json","r") as _f:
    pi_preds = json.load(_f)

si_pred_list = []
pi_pred_list = []

for _binNo in si_preds.keys():
    si_frame = pd.DataFrame(si_preds[_binNo]).transpose()
    si_pred_list.append(si_frame)

for _binNo in pi_preds.keys():
    pi_frame = pd.DataFrame(pi_preds[_binNo]).transpose()
    si_pred_list.append(si_frame)

si_pred_list[0]

si_pred_list[0].loc["FL","val"]
#          bin no   var  val/err

print(iminuit.__version__)
#%%
#loading the data
df = pd.read_pickle("Toy_Data_PKL/sig.pkl")

# %% bin splitting
# I'm sure there's a more elegant solution but I can't think of one right now.

bin0 = df[(df['q2'] > 0.1) & (df['q2'] <= 0.98)]
bin1 = df[(df['q2'] > 1.1) & (df['q2'] <= 2.5)]
bin2 = df[(df['q2'] > 2.5) & (df['q2'] <= 4.0)]
bin3 = df[(df['q2'] > 4.0) & (df['q2'] <= 6.0)]
bin4 = df[(df['q2'] > 6.0) & (df['q2'] <= 8.0)]
bin5 = df[(df['q2'] > 15.0) & (df['q2'] <= 17.0)]
bin6 = df[(df['q2'] > 17.0) & (df['q2'] <= 19.0)]
bin7 = df[(df['q2'] > 11.0) & (df['q2'] <= 12.5)]
bin8 = df[(df['q2'] > 1.0) & (df['q2'] <= 6.0)] # ???
# we're only given toy data up to bin 6, so how can we check that this isn't a mistake?
bin9 = df[(df['q2'] > 15.0) & (df['q2'] <= 17.9)]

bins = [bin0, bin1, bin2, bin3, bin4, bin5, bin6, bin7, bin8, bin9]

bins[0].head()


#%%

#THETAL
plt.hist(bins[3]['costhetal'], bins=25, density=True)
plt.xlabel(r'$cos(\theta_l)$')
plt.ylabel(r'Number of candidates')
plt.grid()
plt.show()

def d2gamma_p_d2q2_dcosthetal(fl, afb, cos_theta_l):
    """
    Returns the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param cos_theta_l: cos(theta_l)
    :return:
    """
    ctl = cos_theta_l
    c2tl = 2 * ctl ** 2 - 1
    acceptance = 0.5  # acceptance "function"
    scalar_array = 3/8 * (3/2 - 1/2 * fl + 1/2 * c2tl * (1 - 3 * fl) + 8/3 * afb * ctl) * acceptance
    normalised_scalar_array = scalar_array * 2  # normalising scalar array to account for the non-unity acceptance function
    return normalised_scalar_array

#THETAL
def log_likelihoodthetal(fl, afb, _bin): #FOR THETA_L
    """
    Returns the negative log-likelihood of the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param _bin: number of the bin to fit
    :return:
    """
    _bin = bins[int(_bin)]
    ctl = _bin['costhetal']
    normalised_scalar_array = d2gamma_p_d2q2_dcosthetal(fl=fl, afb=afb, cos_theta_l=ctl)
    return - np.sum(np.log(normalised_scalar_array))

_test_bin = 1  #FOR THETA_L
_test_afb = 0.7
_test_fl = 0.0

x = np.linspace(-1, 1, 500)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
ax1.plot(x, [log_likelihoodthetal(fl=i, afb=_test_afb, _bin=_test_bin) for i in x])
ax1.set_title(r'$A_{FB}$ = ' + str(_test_afb))
ax1.set_xlabel(r'$F_L$')
ax1.set_ylabel(r'$-\mathcal{L}$')
ax1.grid()
ax2.plot(x, [log_likelihoodthetal(fl=_test_fl, afb=i, _bin=_test_bin) for i in x])
ax2.set_title(r'$F_{L}$ = ' + str(_test_fl))
ax2.set_xlabel(r'$A_{FB}$')
ax2.set_ylabel(r'$-\mathcal{L}$')
ax2.grid()
plt.tight_layout()
plt.show()
#%%
#Alternative plot - AFB and FL together
plt.plot(x, [log_likelihoodthetal(fl=i, afb=_test_afb, _bin=_test_bin) for i in x],label=r'$F_L$, defined at $A_{FB}$=0.7')
x_Afb=[]

plt.plot(x, [log_likelihoodthetal(fl=_test_fl, afb=i, _bin=_test_bin) for i in x],label=r'$A_{FB}$, defined at $F_L$=0')

plt.xlabel(r'Angular Distribution Constants Values')
plt.ylabel(r'$-\mathcal{L}$')
plt.legend()

plt.grid()
plt.show()

#%%

bin_number_to_check = 0  # bin that we want to check in more details in the next cell
bin_results_to_check = None

log_likelihoodthetal.errordef = Minuit.LIKELIHOOD
decimal_places = 3
starting_point = [-0.1,0.0]
fls, fl_errs = [], []
afbs, afb_errs = [], []
for i in range(len(bins)):
    m = Minuit(log_likelihoodthetal, fl=starting_point[0], afb=starting_point[1], _bin=i)
    m.fixed['_bin'] = True  # fixing the bin number as we don't want to optimize it
    m.limits=((-1.0, 1.0), (-1.0, 1.0), None)
    m.migrad()
    m.hesse()
    if i == bin_number_to_check:
        bin_results_to_check = m
    fls.append(m.values[0])
    afbs.append(m.values[1])
    fl_errs.append(m.errors[0])
    afb_errs.append(m.errors[1])
    print(f"Bin {i}: {np.round(fls[i], decimal_places)} pm {np.round(fl_errs[i], decimal_places)},", f"{np.round(afbs[i], decimal_places)}")

plt.figure(figsize=(8, 5))
plt.subplot(221)
bin_results_to_check.draw_mnprofile('afb', bound=3)
plt.subplot(222)
bin_results_to_check.draw_mnprofile('fl', bound=3)
plt.tight_layout()
plt.show()
          
#%%
bin_to_plot = 3
number_of_bins_in_hist = 25
cos_theta_l_bin = bins[bin_to_plot]['costhetal']
hist, _bins, _ = plt.hist(cos_theta_l_bin, bins=number_of_bins_in_hist,alpha=0.75,edgecolor='black', linewidth=1.2)
x = np.linspace(-1, 1, number_of_bins_in_hist)
pdf_multiplier = np.sum(hist) * (np.max(cos_theta_l_bin) - np.min(cos_theta_l_bin)) / number_of_bins_in_hist
y = d2gamma_p_d2q2_dcosthetal(fl=fls[bin_to_plot], afb=afbs[bin_to_plot], cos_theta_l=x) * pdf_multiplier
plt.plot(x, y,'--', label=f'Fit for bin {bin_to_plot}',color='red')
plt.xlabel(r'$cos(\theta_l)$')
plt.ylabel(r'Number of candidates')
plt.legend()
plt.grid()
plt.show()

#%%
fig, ax = plt.subplots(1, 2, figsize=(7, 3))
fig = plt.figure()
gs = fig.add_gridspec(1,2, wspace=0)
(ax1, ax2) = gs.subplots(sharex=False, sharey=True)

#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3),sharey=True)
#gs = fig.add_gridspec(1,2, wspace=0)
l1=ax1.plot(np.linspace(0, len(bins) - 1, len(bins)), fls,'o',ms=2,label=r'$F_L$', color='red')
l2=ax2.plot(np.linspace(0, len(bins) - 1, len(bins)), afbs, 'o', ms=2, label=r'$A_{FB}$', color='blue')
ax1.errorbar(np.linspace(0, len(bins) - 1, len(bins)), fls, yerr=fl_errs, ls='none',color='black',capsize=1.5,lw=0.8)
ax2.errorbar(np.linspace(0, len(bins) - 1, len(bins)), afbs, yerr=afb_errs,ls='none',color='black',capsize=1.5,lw=0.8)
ax1.grid()
ax2.grid()
ax1.set_ylabel('')
#ax2.set_ylabel(r'$A_{FB}$')
ax1.set_xlabel(r'Bin number')
ax2.set_xlabel(r'Bin number')

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels,loc='upper center')

plt.tight_layout()
plt.show()

#%%
#Phi
#PHI
plt.hist(bins[6]['phi'], bins=25, density=True)
plt.xlabel(r'$\phi$')
plt.ylabel(r'Number of candidates')
plt.grid()
plt.show()

def d2gamma_p_d2q2_dphi(s3, s9, phi):
    """
    Returns the pdf defined above
    :param s3: s3 observable
    :param s9: s9 observable
    :param phi: phi
    :return:
    """

    acceptance = 0.5  # acceptance "function"
    scalar_array = 1/(2*np.pi)*(1+s3*np.cos(2*phi)+s9*np.sin(2*phi)) * acceptance
    normalised_scalar_array = scalar_array * 2  # normalising scalar array to account for the non-unity acceptance function
    return normalised_scalar_array

def log_likelihoodphi(s3, s9, _bin): #FOR PHI
    """
    Returns the negative log-likelihood of the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param _bin: number of the bin to fit
    :return:
    """
    _bin = bins[int(_bin)]
    Phi = _bin['phi']
    normalised_scalar_array = d2gamma_p_d2q2_dphi(s3=s3, s9=s9, phi=Phi)
    return - np.sum(np.log(normalised_scalar_array))

_test_bin = 6  #FOR PHI
_test_s3 = 0.7
_test_s9 = 0

x = np.linspace(-1, 1, 500)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
ax1.plot(x, [log_likelihoodphi(s3=i, s9=_test_s9, _bin=_test_bin) for i in x])
ax1.set_title(r'$S_9$ = ' + str(_test_s9))
ax1.set_xlabel(r'$S_3$')
ax1.set_ylabel(r'$-\mathcal{L}$')
ax1.grid()
ax2.plot(x, [log_likelihoodphi(s3=_test_s3, s9=i, _bin=_test_bin) for i in x])
ax2.set_title(r'$S_3$ = ' + str(_test_s3))
ax2.set_xlabel(r'$S_9$')
ax2.set_ylabel(r'$-\mathcal{L}$')
ax2.grid()
plt.tight_layout()
plt.show()

#%%
bin_number_to_check = 6  # bin that we want to check in more details in the next cell
bin_results_to_check = None

log_likelihoodphi.errordef = Minuit.LIKELIHOOD
decimal_places = 3
starting_point = [-0.25,0.0]
s3s, s3_errs = [], []
s9s, s9_errs = [], []
for i in range(len(bins)):
    m = Minuit(log_likelihoodphi, s3=starting_point[0], s9=starting_point[1], _bin=i)
    m.fixed['_bin'] = True  # fixing the bin number as we don't want to optimize it
    m.limits=((-1, 1), (-1, 1), None)
    m.migrad()
    m.hesse()
    if i == bin_number_to_check:
        bin_results_to_check = m
    s3s.append(m.values[0])
    s9s.append(m.values[1])
    s3_errs.append(m.errors[0])
    s9_errs.append(m.errors[1])
    print(f"Bin {i}: S3={np.round(s3s[i], decimal_places)} pm {np.round(s3_errs[i], decimal_places)},", f" S9={np.round(s9s[i], decimal_places)} pm {np.round(s9_errs[i], decimal_places)}. Function minimum considered valid: {m.fmin.is_valid}")

plt.figure(figsize=(8, 5))
plt.subplot(221)
bin_results_to_check.draw_mnprofile('s9', bound=3)
plt.subplot(222)
bin_results_to_check.draw_mnprofile('s3', bound=3)
plt.tight_layout()
plt.show()       
          
#%%
bin_to_plot = 6
number_of_bins_in_hist = 25
phi_bin = bins[bin_to_plot]['phi']
hist, _bins, _ = plt.hist(phi_bin, bins=number_of_bins_in_hist,alpha=0.75,edgecolor='black', linewidth=1.2)
x = np.linspace(-np.pi, np.pi, number_of_bins_in_hist)
pdf_multiplier = np.sum(hist) * (np.max(phi_bin) - np.min(phi_bin)) / number_of_bins_in_hist
y = d2gamma_p_d2q2_dphi(s3=s3s[bin_to_plot], s9=s9s[bin_to_plot], phi=x) * pdf_multiplier
plt.plot(x, y, '--',label=f'Fit for bin {bin_to_plot}',color='red')
plt.xlabel(r'$\phi$')
plt.ylabel(r'Number of candidates')
plt.legend()
plt.grid()
plt.show()

#%%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
ax1.plot(np.linspace(0, len(bins) - 1, len(bins)), s3s,'o', ms=2, label=r'$S_3$', color='red')
ax2.plot(np.linspace(0, len(bins) - 1, len(bins)), s9s,'o', ms=2, label=r'$S_9$', color='red')
ax1.errorbar(np.linspace(0, len(bins) - 1, len(bins)), s3s, yerr=s3_errs,ls='none', color='black',capsize=1.5,lw=0.8)
ax2.errorbar(np.linspace(0, len(bins) - 1, len(bins)), s9s, yerr=s9_errs, ls='none', color='black',capsize=1.5,lw=0.8)
ax1.grid()
ax2.grid()
ax1.set_ylabel(r'$S_3$')
ax2.set_ylabel(r'$S_9$')
ax1.set_xlabel(r'Bin number')
ax2.set_xlabel(r'Bin number')
plt.tight_layout()
plt.show()

#%%


#THETAK
plt.hist(bins[1]['costhetak'], bins=25, density=True)
plt.xlabel(r'$\cos(\theta_k)$')
plt.ylabel(r'Number of candidates')
plt.grid()
plt.show()
    
#%%      
def d2gamma_p_d2q2_dcosthetak(fl, cos_theta_k):
    """
    Returns the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param cos_theta_k: cos(theta_k)
    :return:
    """
    ctk = cos_theta_k
    acceptance = 0.5  # acceptance "function"
    scalar_array = 3/4 * ((3*fl-1)*ctk**2 +(1-fl)) * acceptance
    normalised_scalar_array = scalar_array * 2  # normalising scalar array to account for the non-unity acceptance function
    return normalised_scalar_array

def log_likelihoodthetak(fl, _bin): #FOR THETA_K
    """
    Returns the negative log-likelihood of the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param _bin: number of the bin to fit
    :return:
    """
    _bin = bins[int(_bin)]
    ctk = _bin['costhetak']
    normalised_scalar_array = d2gamma_p_d2q2_dcosthetak(fl=fl, cos_theta_k=ctk)
    return - np.sum(np.log(normalised_scalar_array))          
          
_test_bin = 1  #FOR THETA_K


x = np.linspace(-1, 1, 500)
plt.plot(x, [log_likelihoodthetak(fl=i, _bin=_test_bin) for i in x])
plt.xlabel(r'$F_L$')
plt.ylabel(r'$-\mathcal{L}$')
plt.grid()
plt.tight_layout()
plt.show()

#%%
bin_number_to_check = 0  # bin that we want to check in more details in the next cell
bin_results_to_check = None

log_likelihoodthetak.errordef = Minuit.LIKELIHOOD
decimal_places = 3
starting_point = [0.5]
fls, fl_errs = [], []

for i in range(len(bins)):
    m = Minuit(log_likelihoodthetak, fl=starting_point[0], _bin=i)
    m.fixed['_bin'] = True  # fixing the bin number as we don't want to optimize it
    m.limits=((-1.0, 1.0), None)
    m.migrad()
    m.hesse()
    if i == bin_number_to_check:
        bin_results_to_check = m
    fls.append(m.values[0])
    fl_errs.append(m.errors[0])
    print(f"Bin {i}: {np.round(fls[i], decimal_places)} pm {np.round(fl_errs[i], decimal_places)}. Function minimum considered valid: {m.fmin.is_valid}")
          
        
plt.plot(figsize=(1, 15))
plt.subplot(222)
bin_results_to_check.draw_mnprofile('fl', bound=3)
plt.tight_layout()
plt.show()


#%%
bin_to_plot = 3
number_of_bins_in_hist = 25
cos_theta_k_bin = bins[bin_to_plot]['costhetak']
hist, _bins, _ = plt.hist(cos_theta_k_bin, bins=number_of_bins_in_hist,alpha=0.75,edgecolor='black', linewidth=1.2)
x = np.linspace(-1, 1, number_of_bins_in_hist)
pdf_multiplier = np.sum(hist) * (np.max(cos_theta_k_bin) - np.min(cos_theta_k_bin)) / number_of_bins_in_hist
y = d2gamma_p_d2q2_dcosthetak(fl=fls[bin_to_plot], cos_theta_k=x) * pdf_multiplier
plt.plot(x, y,'--', label=f'Fit for bin {bin_to_plot}',color='red')
plt.xlabel(r'$cos(\theta_k)$')
plt.ylabel(r'Number of candidates')
plt.legend()
plt.grid()
plt.show()

#%%

fig, (ax1) = plt.subplots(1, figsize=(5, 4))
ax1.plot(np.linspace(0, len(bins) - 1, len(bins)), fls,'o', ms=2, label=r'$F_L$', color='red')
ax1.errorbar(np.linspace(0, len(bins) - 1, len(bins)), fls, yerr=fl_errs,color='black',lw=0.8,capsize=1.5,ls='none')
ax1.grid()
ax1.set_ylabel(r'$F_L$')
ax1.set_xlabel(r'Bin number')

plt.tight_layout()
plt.show()     
        

#%% #THETAK + PHI

def d2gamma_p_d2q2_dphi_dcosthetak(fl,s3, s4, s5, s7, s8, s9, phi,cos_theta_k):
    """
    Returns the pdf defined above
    :param s3: s3 observable
    :param s9: s9 observable
    :param phi: phi
    :return:
    """
    ctk2=cos_theta_k**2
    stk2=1-cos_theta_k**2
    ct2k=ctk2-stk2
    st2k=np.sqrt(1-ct2k**2)
    
    acceptance = 0.5  # acceptance "function"
    scalar_array = (9/(32*np.pi))*(1.5*(1-fl)*(stk2)+2*fl*ctk2-(1/6)*(1-fl)*stk2+(2/3)*fl*ctk2+(4/3)*s3*stk2*np.cos(2*phi)+(2/3)*s4*np.cos(phi)*st2k+(np.pi/2)*s5*st2k*np.cos(phi)+(np.pi/2)*s7*st2k*np.sin(phi)+(2/3)*s8*st2k*np.pi+(4/3)*s9*stk2*np.sin(2*phi)  ) * acceptance
    normalised_scalar_array = scalar_array * 2  # normalising scalar array to account for the non-unity acceptance function
    return normalised_scalar_array

def log_likelihoodphithetak(fl,s3,s4,s5,s7,s8, s9, _bin):
    """
    Returns the negative log-likelihood of the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param _bin: number of the bin to fit
    :return:
    """
    _bin = bins[int(_bin)]
    Phi = _bin['phi']
    ctk= _bin['costhetak']
    normalised_scalar_array = d2gamma_p_d2q2_dphi_dcosthetak(fl=fl,s3=s3,s4=s4,s5=s5,s7=s7,s8=s8, s9=s9, phi=Phi,cos_theta_k=ctk)
    return - np.sum(np.log(normalised_scalar_array))

bin_number_to_check = 7  # bin that we want to check in more details in the next cell
bin_results_to_check = None

log_likelihoodphithetak.errordef = Minuit.LIKELIHOOD
decimal_places = 3
starting_point = [0.32,-0.25,-0.31,-0.22,-0.00056,0.000119,0.000169]
fls, fl_errs= [],[]
s3s, s3_errs = [], []
s4s, s4_errs = [], []
s5s, s5_errs = [], []
s7s, s7_errs = [], []
s8s,s8_errs=[],[]
s9s, s9_errs = [], []
for i in range(len(bins)):
    m = Minuit(log_likelihoodphithetak, fl=starting_point[0], s3=starting_point[1],s4=starting_point[2],s5=starting_point[3],s7=starting_point[4],s8=starting_point[5], s9=starting_point[6], _bin=i)
    m.fixed['_bin'] = True  # fixing the bin number as we don't want to optimize it
    m.limits=((-1, 1),(-0.6, 0.6),(-0.6, 0.6),(-0.6, 0.6),(-0.6, 0.6),(-0.05, 0.05), (-0.05, 0.05), None)
    m.migrad()
    m.hesse()
    if i == bin_number_to_check:
        bin_results_to_check = m
    
    fls.append(m.values[0])
    s3s.append(m.values[1])
    s4s.append(m.values[2])
    s5s.append(m.values[3])
    s7s.append(m.values[4])
    s8s.append(m.values[5])
    s9s.append(m.values[6])
    
    fl_errs.append(m.errors[0])
    s3_errs.append(m.errors[1])
    s4_errs.append(m.errors[2])
    s5_errs.append(m.errors[3])
    s7_errs.append(m.errors[4])
    s8_errs.append(m.errors[5])
    s9_errs.append(m.errors[6])
    
    print(f"Bin {i}:Function minimum considered valid: {m.fmin.is_valid}")
#%%    
plt.figure(figsize=(8, 5))
plt.subplot(222)
bin_results_to_check.draw_mnprofile('s8', bound=3)
plt.tight_layout()
plt.show()

#%%
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14/1.5, 6/1.5))
ax1.plot(np.linspace(0, len(bins) - 1, len(bins)), fls, 'o',ms=2, label=r'$F_L$', color='red')
ax2.plot(np.linspace(0, len(bins) - 1, len(bins)), s3s, 'o',ms=2, label=r'$S_3$', color='red')
ax1.errorbar(np.linspace(0, len(bins) - 1, len(bins)), fls, yerr=fl_errs, color='black',lw=0.8,capsize=1.5,ls='none')
ax2.errorbar(np.linspace(0, len(bins) - 1, len(bins)), s3s, yerr=s3_errs, color='black',lw=0.8,capsize=1.5,ls='none')
ax1.grid()
ax2.grid()
ax1.set_ylabel(r'$F_L$')
ax2.set_ylabel(r'$S_3$')
ax1.set_xlabel(r'Bin number')
ax2.set_xlabel(r'Bin number')
plt.tight_layout()
plt.show()

#%%
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14/1.5, 6/1.5))
ax1.plot(np.linspace(0, len(bins) - 1, len(bins)), s4s,'o',ms=2, label=r'$S_4$', color='red')
ax2.plot(np.linspace(0, len(bins) - 1, len(bins)), s5s,'o', ms=2, label=r'$S_5$', color='red')
ax1.errorbar(np.linspace(0, len(bins) - 1, len(bins)), s4s, yerr=s4_errs, color='black',lw=0.8,capsize=1.5,ls='none')
ax2.errorbar(np.linspace(0, len(bins) - 1, len(bins)), s5s, yerr=s5_errs, color='black',lw=0.8,capsize=1.5,ls='none')
ax1.grid()
ax2.grid()
ax1.set_ylabel(r'$S_4$')
ax2.set_ylabel(r'$S_5$')
ax1.set_xlabel(r'Bin number')
ax2.set_xlabel(r'Bin number')
plt.tight_layout()
plt.show()

#%%
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14/1.5, 6/1.5))
ax1.plot(np.linspace(0, len(bins) - 1, len(bins)), s7s, 'o',ms=2, label=r'$S_7$', color='red')
ax2.plot(np.linspace(0, len(bins) - 1, len(bins)), s8s, 'o', ms=2, label=r'$S_8$', color='red')
ax1.errorbar(np.linspace(0, len(bins) - 1, len(bins)), s7s, yerr=s7_errs, color='black',lw=0.8,capsize=1.5,ls='none')
ax2.errorbar(np.linspace(0, len(bins) - 1, len(bins)), s8s, yerr=s8_errs, color='black',lw=0.8,capsize=1.5,ls='none')
ax1.grid()
ax2.grid()
ax1.set_ylabel(r'$S_7$')
ax2.set_ylabel(r'$S_8$')
ax1.set_xlabel(r'Bin number')
ax2.set_xlabel(r'Bin number')
plt.tight_layout()
plt.show()

#%%
fig, (ax1) = plt.subplots(1,figsize=(14/3, 12/3))
ax1.plot(np.linspace(0, len(bins) - 1, len(bins)), s9s, 'o',ms=2, label=r'$S_9$', color='red')
ax1.errorbar(np.linspace(0, len(bins) - 1, len(bins)), s9s, yerr=s9_errs, color='black',lw=0.8,capsize=1.5,ls='none')
ax1.grid()
ax1.set_ylabel(r'$S_9$')
ax1.set_xlabel(r'Bin number')
plt.tight_layout()
plt.show()

#%%
fig = plt.figure(figsize=(8, 6))



ax1= fig.add_subplot(4,3,4)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

ax2= fig.add_subplot(4,3,5)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

ax3= fig.add_subplot(4,3,6)
ax3.errorbar(np.linspace(0, len(bins) - 1, len(bins)), s3s, yerr=s3_errs, color='black',lw=0.8,capsize=1.5,ls='none')
ax3.plot(np.linspace(0, len(bins) - 1, len(bins)), s3s, 'o',ms=2, label=r'$S_3$', color='red')
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

ax4= fig.add_subplot(4,3,7)
ax4.plot(np.linspace(0, len(bins) - 1, len(bins)), s4s,'o',ms=2, label=r'$S_4$', color='red')
ax4.errorbar(np.linspace(0, len(bins) - 1, len(bins)), s4s, yerr=s4_errs, color='black',lw=0.8,capsize=1.5,ls='none')
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

ax5= fig.add_subplot(4,3,8)
ax5.plot(np.linspace(0, len(bins) - 1, len(bins)), s5s,'o', ms=2, label=r'$S_5$', color='red')
ax5.errorbar(np.linspace(0, len(bins) - 1, len(bins)), s5s, yerr=s5_errs, color='black',lw=0.8,capsize=1.5,ls='none')
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

ax6= fig.add_subplot(4,3,9)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

ax7= fig.add_subplot(4,3,10)
ax7.plot(np.linspace(0, len(bins) - 1, len(bins)), s7s, 'o',ms=2, label=r'$S_7$', color='red')
ax7.errorbar(np.linspace(0, len(bins) - 1, len(bins)), s7s, yerr=s7_errs, color='black',lw=0.8,capsize=1.5,ls='none')
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

ax8= fig.add_subplot(4,3,11)
ax8.errorbar(np.linspace(0, len(bins) - 1, len(bins)), s8s, yerr=s8_errs, color='black',lw=0.8,capsize=1.5,ls='none')
ax8.plot(np.linspace(0, len(bins) - 1, len(bins)), s8s, 'o', ms=2, label=r'$S_8$', color='red')
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

ax9= fig.add_subplot(4,3,12)
ax9.plot(np.linspace(0, len(bins) - 1, len(bins)), s9s, 'o',ms=2, label=r'$S_9$', color='red')
ax9.errorbar(np.linspace(0, len(bins) - 1, len(bins)), s9s, yerr=s9_errs, color='black',lw=0.8,capsize=1.5,ls='none')
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
ax9.set_xticklabels([])

ax10= fig.add_subplot(4,2,1)
ax10.plot(np.linspace(0, len(bins) - 1, len(bins)), fls, 'o',ms=2, label=r'$F_L$', color='red')
ax10.errorbar(np.linspace(0, len(bins) - 1, len(bins)), fls, yerr=fl_errs, color='black',lw=0.8,capsize=1.5,ls='none')
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

ax11= fig.add_subplot(4,2,2)
ax11.plot(np.linspace(0, len(bins) - 1, len(bins)), afbs, 'o', ms=2, label=r'$A_{FB}$', color='red')
ax11.errorbar(np.linspace(0, len(bins) - 1, len(bins)), afbs, yerr=afb_errs,ls='none',color='black',capsize=1.5,lw=0.8)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

axs=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11]
Saxs=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]
USaxs=[ax1,ax2,ax3,ax4,ax5,ax6]

for i in axs:
    i.grid(True)
    i.legend(loc='upper center',prop={'size': 7})
    
for i in Saxs:
    i.set_xlim(-0.5,9.5)
    
for i in USaxs:
    i.set_xticklabels([])

fig.suptitle('Angular Distribution Constants')
plt.show()
        
#%%
fig = plt.figure(figsize=(8, 6))
bins=np.linspace(0, len(bins) - 1, len(bins))
b0,b0err=[(0.98+0.1)/2,(0.98-0.1)/2]
b1,b1err=[(1.1+2.5)/2,(2.5-1.1)/2]
b2,b2err=[(2.5+4)/2,(4-2.5)/2]
b3,b3err=[(6+4)/2,(6-4)/2]
b4,b4err=[(6+8)/2,(8-6)/2]
b5,b5err=[(17+15)/2,(17-15)/2]
b6,b6err=[(17+19)/2,(19-17)/2]
b7,b7err=[(11+12.5)/2,(12.5-11)/2]
b8,b8err=[(1+6)/2,(6-1)/2]
b9,b9err=[(15+17.9)/2,(17.9-15)/2]
qsquared=[b0,b1,b2,b3,b4,b5,b6,b7,b8,b9]
qsquarederr=[b0err,b1err,b2err,b3err,b4err,b5err,b6err,b7err,b8err,b9err]

ax1= fig.add_subplot(4,3,4)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

ax2= fig.add_subplot(4,3,5)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

ax3= fig.add_subplot(4,3,6)
ax3.errorbar(qsquared, s3s, yerr=s3_errs,xerr=qsquarederr, color='black',lw=0.8,capsize=1.5,ls='none')
ax3.plot(qsquared, s3s, 'o',ms=2, label=r'$S_3$', color='red')
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

ax4= fig.add_subplot(4,3,7)
ax4.plot(qsquared, s4s,'o',ms=2, label=r'$S_4$', color='red')
ax4.errorbar(qsquared, s4s, yerr=s4_errs,xerr=qsquarederr, color='black',lw=0.8,capsize=1.5,ls='none')
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

ax5= fig.add_subplot(4,3,8)
ax5.plot(qsquared, s5s,'o', ms=2, label=r'$S_5$', color='red')
ax5.errorbar(qsquared, s5s, yerr=s5_errs,xerr=qsquarederr, color='black',lw=0.8,capsize=1.5,ls='none')
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

ax6= fig.add_subplot(4,3,9)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

ax7= fig.add_subplot(4,3,10)
ax7.plot(qsquared, s7s, 'o',ms=2, label=r'$S_7$', color='red')
ax7.errorbar(qsquared, s7s, yerr=s7_errs,xerr=qsquarederr, color='black',lw=0.8,capsize=1.5,ls='none')
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

ax8= fig.add_subplot(4,3,11)
ax8.errorbar(qsquared, s8s, yerr=s8_errs,xerr=qsquarederr, color='black',lw=0.8,capsize=1.5,ls='none')
ax8.plot(qsquared, s8s, 'o', ms=2, label=r'$S_8$', color='red')
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

ax9= fig.add_subplot(4,3,12)
ax9.plot(qsquared, s9s, 'o',ms=2, label=r'$S_9$', color='red')
ax9.errorbar(qsquared, s9s, yerr=s9_errs,xerr=qsquarederr, color='black',lw=0.8,capsize=1.5,ls='none')
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
ax9.set_xticklabels([])

ax10= fig.add_subplot(4,2,1)
ax10.plot(qsquared, fls, 'o',ms=2, label=r'$F_L$', color='red')
ax10.errorbar(qsquared, fls, yerr=fl_errs,xerr=qsquarederr, color='black',lw=0.8,capsize=1.5,ls='none')
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

ax11= fig.add_subplot(4,2,2)
ax11.plot(qsquared, afbs, 'o', ms=2, label=r'$A_{FB}$', color='red')
ax11.errorbar(qsquared, afbs, yerr=afb_errs,xerr=qsquarederr,ls='none',color='black',capsize=1.5,lw=0.8)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


s1s,s2s,s6s,s1_errs,s2_errs,s6_errs=[],[],[],[],[],[]
plots=[s1s,s2s,s3s,s4s,s5s,s6s,s7s,s8s,s9s,fls,afbs]
errors=[s1_errs,s2_errs,s3_errs,s4_errs,s5_errs,s6_errs,s7_errs,s8_errs,s9_errs,fl_errs,afb_errs]

Usedax=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11]

for j in range(len(Usedax)):
    if j!=0 and j!=1 and j!=5:
        patches=[]
        for i in range(0,10):
            square=np.array([[qsquared[i]-qsquarederr[i],plots[j][i]-errors[j][i]],[qsquared[i]+qsquarederr[i],plots[j][i]-errors[j][i]],[qsquared[i]+qsquarederr[i],plots[j][i]+errors[j][i]],[qsquared[i]-qsquarederr[i],plots[j][i]+errors[j][i]]])
            patches.append(Polygon(square, color='red',alpha=0.01))
            p=PatchCollection(patches, match_original=True)
            Usedax[j].add_collection(p)
    

axs=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11]
Saxs=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]
USaxs=[ax1,ax2,ax3,ax4,ax5,ax6]

for i in axs:
    i.grid(True)
    i.legend(loc='upper center',prop={'size': 7})
    
for i in Saxs:
    i.set_xlim(-0.5,9.5)
    
for i in USaxs:
    i.set_xticklabels([])

fig.suptitle('Angular Distribution Constants')
plt.show()
        