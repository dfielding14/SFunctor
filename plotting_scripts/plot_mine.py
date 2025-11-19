#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import subprocess

# Add texlive to PATH
texlive_path = "/sw/andes/spack-envs/base/opt/linux-rhel8-x86_64/gcc-8.3.1/texlive-20210325-ari2ztcowrqldrwjblfqpdiphkzd3nhu/bin/x86_64-linux"
os.environ['PATH'] = f"{texlive_path}:{os.environ['PATH']}"

# Verify latex is available
try:
    result = subprocess.run(['which', 'latex'], capture_output=True, text=True)
    print(f"LaTeX found at: {result.stdout.strip()}")
except:
    print("Warning: LaTeX not found in PATH")

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cmasher as cmr


import matplotlib
matplotlib.rc('font', family='serif', size=12)
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True
matplotlib.rcParams['xtick.minor.visible'] = True
matplotlib.rcParams['ytick.minor.visible'] = True
matplotlib.rcParams['lines.dash_capstyle'] = 'round'
matplotlib.rcParams['figure.dpi'] = 200
matplotlib.rcParams['text.usetex'] = True

# In[ ]:


data = np.load("/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/results_Turb_5120_beta1_dedt025_plm/ndisp10_000_nrand1_000_nell128_sw2_job3652554/sf_results_all_slices.npz", allow_pickle=True)
hist_mag = data['hist_mag']
hist_other = data['hist_other']
mag_channels = data['mag_channels']
other_channels = data['other_channels']
ell_bin_edges = data['ell_bin_edges']
theta_bin_edges = data['theta_bin_edges']
phi_bin_edges = data['phi_bin_edges']
sf_channel_bin_edges = data['sf_channel_bin_edges']
product_bin_edges = data['product_bin_edges']
metadata = data['metadata']
slice_metadata = data['slice_metadata']


# In[9]:


def plot_histogram_from_file(npz_file, title):
    """
    Load the given .npz file and plot the histogram as in the original code.

    Parameters
    ----------
    npz_file : str
        Path to the .npz file to load.
    title : str
        Title for the plot.
    """
    data = np.load(npz_file, allow_pickle=True)
    hist_mag = data['hist_mag']
    hist_other = data['hist_other']
    mag_channels = data['mag_channels']
    other_channels = data['other_channels']
    ell_bin_edges = data['ell_bin_edges']
    theta_bin_edges = data['theta_bin_edges']
    phi_bin_edges = data['phi_bin_edges']
    sf_channel_bin_edges = data['sf_channel_bin_edges']
    product_bin_edges = data['product_bin_edges']
    metadata = data['metadata']
    slice_metadata = data['slice_metadata']

    fig, ax = plt.subplots(figsize=(5,4), constrained_layout=True)
    plot = ax.pcolormesh(
        ell_bin_edges,
        sf_channel_bin_edges[-1],
        ((np.sum(hist_mag[10], axis=(1,2)).T / np.sum(hist_mag[10], axis=(1,2,3))).T / np.diff(np.log10(sf_channel_bin_edges[-1]))).T,
        norm=LogNorm(vmin=1e-4),
        cmap=cmr.ocean_r
    )
    cb = fig.colorbar(plot)
    cb.set_label(r'$\frac{N(q_\ell \big| \ell) }{N(\ell)} \frac{1}{\Delta \log q_\ell}$')
    ax.loglog()
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$q_\ell = \delta B (\ell) / B_{\rm mean, loc}(\ell)$')
    ax.set_ylim(1e-4,1e1)
    ax.axhline(1, color='k', linestyle='--')
    ax.set_title(title)

    return fig, ax, (((np.sum(hist_mag[10], axis=(1,2)).T / np.sum(hist_mag[10], axis=(1,2,3))).T / np.diff(np.log10(sf_channel_bin_edges[-1]))))[16], sf_channel_bin_edges

# In[10]:



fig, ax, q16_beta1, sf_channel_bin_edges_beta1 = plot_histogram_from_file(
    "/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/results_Turb_5120_beta1_dedt025_plm/ndisp10_000_nrand1_000_nell128_sw2_job3652554/sf_results_all_slices.npz",
    r"$\delta B / B_0 = 1/3$"
)
fig.savefig('dBB_loc_beta1_5120.png')

fig, ax, q16_beta6, sf_channel_bin_edges_beta6 = plot_histogram_from_file(
    "/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/results_Turb_5120_beta6_dedt025_plm/ndisp10_000_nrand1_000_nell128_sw2_job3650342/sf_results_all_slices.npz",
    r"$\delta B / B_0 = 1$"
)
fig.savefig('dBB_loc_beta6_5120.png')

fig, ax, q16_beta25, sf_channel_bin_edges_beta25 = plot_histogram_from_file(
    "/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/results_Turb_5120_beta25_dedt025_plm/ndisp10_000_nrand1_000_nell128_sw2_job3652555/sf_results_all_slices.npz",
    r"$\delta B / B_0 = 2$"
)
fig.savefig('dBB_loc_beta25_5120.png')

fig, ax, q16_beta100, sf_channel_bin_edges_beta100 = plot_histogram_from_file(
    "/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/results_Turb_5120_beta100_dedt025_plm/ndisp10_000_nrand1_000_nell128_sw2_job3652559/sf_results_all_slices.npz",
    r"$\delta B / B_0 = 4$"
)
fig.savefig('dBB_loc_beta100_5120.png')









# In[11]:


plt.loglog(np.sqrt(sf_channel_bin_edges_beta1[-1][1:]*sf_channel_bin_edges_beta1[-1][:-1]), q16_beta1, label=r'$\delta B / B_0 = 1/3$', color=cmr.guppy(0))
plt.loglog(np.sqrt(sf_channel_bin_edges_beta6[-1][1:]*sf_channel_bin_edges_beta6[-1][:-1]), q16_beta6, label=r'$\delta B / B_0 = 1$', color=cmr.guppy(0.333))
plt.loglog(np.sqrt(sf_channel_bin_edges_beta25[-1][1:]*sf_channel_bin_edges_beta25[-1][:-1]), q16_beta25, label=r'$\delta B / B_0 = 2$', color=cmr.guppy(0.666))
plt.loglog(np.sqrt(sf_channel_bin_edges_beta100[-1][1:]*sf_channel_bin_edges_beta100[-1][:-1]), q16_beta100, label=r'$\delta B / B_0 = 4$', color=cmr.guppy(0.999))
plt.legend()
plt.xlabel(r'$\delta B (16 \Delta x) / B_{\rm mean, loc}(16 \Delta x)$')
plt.ylabel(r'$\frac{N(q_{16 \Delta x} \big| {16 \Delta x}) }{N({16 \Delta x})} \frac{1}{\Delta \log q_{16 \Delta x}}$')
plt.xlim(1e-3,1e3)
plt.tight_layout()
plt.savefig('dBB_loc_16dx_comparison_5120.png')


# In[16]:



fig, ax, q16_10240, sf_channel_bin_edges_10240 = plot_histogram_from_file(
    "/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/results_Turb_10240_beta25_dedt025_plm/ndisp100_000_nrand100_000_nell128_sw3_job3663362/sf_results_all_slices.npz",
    ""
    # r"$\delta B / B_0 = 2 \qquad N_{\rm res} = 10{,}240$"
)
ax.set_ylim(top=1e2)
fig.savefig('dBB_loc_beta25_10240.png')





# In[ ]:




# In[87]:


data = np.load("/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/results_Turb_10240_beta25_dedt025_plm/ndisp100_000_nrand100_000_nell128_sw3_job3663362/sf_results_all_slices.npz", allow_pickle=True)
hist_mag = data['hist_mag']
hist_other = data['hist_other']
mag_channels = data['mag_channels']
other_channels = data['other_channels']
ell_bin_edges = data['ell_bin_edges']
theta_bin_edges = data['theta_bin_edges']
phi_bin_edges = data['phi_bin_edges']
sf_channel_bin_edges = data['sf_channel_bin_edges']
product_bin_edges = data['product_bin_edges']
metadata = data['metadata']
slice_metadata = data['slice_metadata']

q =     ((np.sum(hist_mag[10], axis=(1,2)).T / np.sum(hist_mag[10], axis=(1,2,3))).T / np.diff(np.log10(sf_channel_bin_edges[-1]))).T


# In[92]:


qbins = np.sqrt(sf_channel_bin_edges[-1][1:]*sf_channel_bin_edges[-1][:-1])
fig, ax = plt.subplots(figsize=(5,4), constrained_layout=True)
for i in range(len(ell_bin_edges)-1):
    ax.plot(qbins, q.T[i], color=cmr.guppy(i/(len(ell_bin_edges)-1)))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$q_\ell$')
ax.set_ylabel(r'$P(q_\ell)$')
# ax.set_xlim(1e-3,1e3)
# ax.set_ylim(1e-3,1e3)
# ax.set_title(r'$\delta B / B_0 = 2 \qquad N_{\rm res} = 10{,}240$')
# plt.savefig('dBB_loc_beta25_10240.png')

# In[108]:


from scipy.optimize import curve_fit
slopes = np.zeros(len(ell_bin_edges)-1)
qbins = np.sqrt(sf_channel_bin_edges[-1][1:]*sf_channel_bin_edges[-1][:-1])
fig, ax = plt.subplots(figsize=(5,4), constrained_layout=True)
for i in range(len(ell_bin_edges)-1):
    ax.plot(qbins/qbins[np.argmax(q.T[i])], q.T[i], color=cmr.guppy(i/(len(ell_bin_edges)-1)))
    qbinnormed = qbins/qbins[np.argmax(q.T[i])]
    j_40 = np.argmin(np.abs(qbinnormed - 40))
    j_200 = np.argmin(np.abs(qbinnormed - 200))
    # do a best fit to a power law between j_40 and j_200
    popt, pcov = curve_fit(lambda x, a, b: a*x**b, qbinnormed[j_40:j_200], q.T[i][j_40:j_200])
    # ax.plot(qbinnormed[j_40:j_200], popt[0]*qbinnormed[j_40:j_200]**popt[1], color=cmr.guppy(i/(len(ell_bin_edges)-1)), linestyle='--')
    slopes[i] = popt[1]

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$q_\ell / q_\ell^{\rm max}$')
ax.set_ylabel(r'$P(q_\ell)$')
ax.set_xlim(1e-3,1e3)
plt.show()

plt.semilogx(ell_bin_edges[:-1], slopes)
plt.xlabel(r'$\ell$')
plt.ylabel(r'slope')
plt.show()

# In[113]:



plt.plot(np.log10(ell_bin_edges[:-1]), -slopes)
plt.plot(np.log10(ell_bin_edges[:-1]), 2+np.log10(ell_bin_edges[:-1])/3)
plt.xlabel(r'$\ell$')
plt.ylabel(r'slope')
plt.show()

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:


beta = 25
B0 = np.sqrt(2/beta)
Nres = 10240
def load_data(data_file):
    data = np.load(data_file, allow_pickle=True)

    hist_mag = data['hist_mag']
    mag_channels = data['mag_channels']

    hist_other = data['hist_other']
    other_channels = data['other_channels']

    ell_bin_edges = data['ell_bin_edges']
    theta_bin_edges = data['theta_bin_edges']
    phi_bin_edges = data['phi_bin_edges']
    sf_bin_edges = data['sf_bin_edges']
    product_bin_edges = data['product_bin_edges']

    print("sf_bin_edges", np.min(sf_bin_edges), np.max(sf_bin_edges), len(sf_bin_edges))
    print("product_bin_edges", np.min(product_bin_edges), np.max(product_bin_edges), len(product_bin_edges))


    ell_bin_centers = 0.5*(ell_bin_edges[1:] + ell_bin_edges[:-1])
    theta_bin_centers = 0.5*(theta_bin_edges[1:] + theta_bin_edges[:-1])
    phi_bin_centers = 0.5*(phi_bin_edges[1:] + phi_bin_edges[:-1])
    sf_bin_centers = 0.5*(sf_bin_edges[1:] + sf_bin_edges[:-1])
    product_bin_centers = 0.5*(product_bin_edges[1:] + product_bin_edges[:-1])

    ell = ell_bin_centers/Nres
    dB  = (sf_bin_centers/np.sqrt(2))
    hist = np.sum(hist_mag[1], axis=(1,2)).T/np.sum(np.sum(hist_mag[1], axis=(1,2)).T,axis=0)
    bsf_median = np.array([ np.interp(0.5, np.cumsum(np.sum(hist_mag[1], axis=(1,2))[i]) / np.sum(hist_mag[1], axis=(1,2,3))[i], (sf_bin_centers/np.sqrt(2))) for i in range(hist_mag[1].shape[0])])

    return ell, dB, hist, bsf_median


# In[4]:


data_file = "/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/results_Turb_10240_beta25_dedt025_plm/ndisp10000_nrand1000_nell128_sw2_job3039703/sf_results_all_slices.npz"
ell_H, dB_H, hist_H, bsf_median_H = load_data(data_file)

data_file = "/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/results_Turb_2560_beta25_dedt025_plm/ndisp3000_nrand10000_nell128_sw2_job3039926/sf_results_all_slices.npz"
ell_M, dB_M, hist_M, bsf_median_M = load_data(data_file)

data_file = "/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/results_Turb_640_beta25_dedt025_plm/ndisp10000_nrand10000_nell128_sw2_job3039955/sf_results_all_slices.npz"
ell_L, dB_L, hist_L, bsf_median_L = load_data(data_file)

# In[32]:


median_color = '#ff3399'
median_color_M = '#ff6699'
median_color_L = '#ff9999'
B0_color = '#ffcc66'

fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
plot = ax.pcolormesh(
    ell_H,
    dB_H,
    hist_H,
    norm=LogNorm(0.8e-4, 1.2e-1),
    cmap=cmr.get_sub_cmap('cmr.arctic_r',0, 0.9),
    zorder=-1,
    rasterized=True
)

# Make axis spines thinner
for spine in ax.spines.values():
    spine.set_linewidth(0.5)

# Make major and minor ticks thinner
ax.tick_params(axis='both', which='both', width=0.5, length=2)
ax.tick_params(axis='both', which='minor', width=0.5, length=1)

# Create inset axes for horizontal colorbar in lower right corner
# Position: [left, bottom, width, height] in axes coordinates
# 40% width (0.4) and 5% height (0.05) in lower right
inset_axes = ax.inset_axes([0.5, 0.05, 0.45, 0.05])
cb = fig.colorbar(plot, cax=inset_axes, orientation='horizontal')
cb.set_label(r"$f_V(\delta B|\ell)$", labelpad=5)
cb.ax.xaxis.set_label_position('top')
cb.ax.xaxis.tick_top()
cb.ax.xaxis.set_ticks_position('top')
cb.ax.tick_params(axis='x', labelsize=10, width=0.5, length=2, direction='in', which='major')
cb.ax.tick_params(axis='x', width=0.5, length=1, direction='in', which='minor')

# Make colorbar outline thinner
for spine in cb.ax.spines.values():
    spine.set_linewidth(0.7)

ax.loglog(ell_H, bsf_median_H, color=median_color, lw=2, label = r"${\rm median} \, |\delta B(\ell)|$")
ax.loglog(ell_M*4, bsf_median_M, color=median_color_M, lw=1)
ax.loglog(ell_L*16, bsf_median_L, color=median_color_L, lw=0.5)

ax.text(
    ell_H[0]/1.05, bsf_median_H[0]/1.05, r"$L/\Delta x = 10{,}240$",
    color=median_color, ha='left', va='top',
    rotation=0, rotation_mode='anchor', fontsize=10
)
ax.text(
    ell_M[0]*4/1.05, bsf_median_M[0]/1.05, r"$L/\Delta x = 2560$",
    color=median_color_M, ha='left', va='top',
    rotation=0, rotation_mode='anchor', fontsize=10
)
ax.text(
    ell_L[0]*16/1.05, bsf_median_L[0]/1.05, r"$L/\Delta x = 640$",
    color=median_color_L, ha='left', va='top',
    rotation=0, rotation_mode='anchor', fontsize=10
)

# plot 2/5 power law line
line_color = 'white'
x_line = ell_H[20:-20]
y_line = 2.5 * B0 * (x_line / (ell_H[-20])) ** (2/5)
ax.loglog(x_line, y_line, color=line_color, lw=0.5, ls=':')

# Compute midpoint and angle for label
mid = 1*len(x_line) // 9
x_mid, y_mid = x_line[mid], y_line[mid] * 1.18
pt1, pt2 = [x_line[0], y_line[0]], [x_line[-1], y_line[-1]]
angle = np.degrees(np.arctan2(*(ax.transData.transform(pt2) - ax.transData.transform(pt1))[::-1]))

ax.text(
    x_mid, y_mid, r"$\propto \ell^{2/5}$",
    color=line_color, va='center', ha='center', fontsize=10,
    rotation=angle, rotation_mode='anchor'
)

# plot 1/3 power law line
line_color = 'white'
x_line = ell_H[60:-20]
y_line = 2.5 * B0 * (x_line / (ell_H[-20])) ** (1/3)
ax.loglog(x_line, y_line, color=line_color, lw=0.5, ls=':')

# Compute midpoint and angle for label
mid = 4*len(x_line) // 6
x_mid, y_mid = x_line[mid], y_line[mid] * 1.18
pt1, pt2 = [x_line[0], y_line[0]], [x_line[-1], y_line[-1]]
angle = np.degrees(np.arctan2(*(ax.transData.transform(pt2) - ax.transData.transform(pt1))[::-1]))

ax.text(
    x_mid, y_mid, r"$\propto \ell^{1/3}$",
    color=line_color, va='center', ha='center', fontsize=10,
    rotation=angle, rotation_mode='anchor'
)

# plot B0 line
ax.axhline(B0, color=B0_color, linestyle='--')
# Add label 5% from the left edge above the line
ax.text(ell_H[-1]*0.9, B0/1.2, r"$B_0$", color=B0_color, va='top', ha='right', fontsize=12)


# add short vertical lines to indicate where bsfp[2] and bsf_median cross B0
x_cross = ell_H[np.argmin(np.abs(bsf_median_H - B0))]
ax.axvline(x_cross, color=median_color, linestyle='--', lw=1, ymin=0.775, ymax=1.0)
# Add label above the top of the line, at y=0.3 in axes coordinates
ax.text(
    x_cross/1.05, 0.9,  # y in axes coordinates
    r"$\ell_{A}$",
    color=median_color, ha='right', va='bottom', fontsize=12,
    transform=ax.get_xaxis_transform()  # x in data, y in axes
)


# ax.set_ylabel(r'$|\delta B^{(2pt)}(\ell)| \doteq \frac{|\mathbf{B}(x{+}\ell) {-} \mathbf{B}(x)|}{\sqrt{2}}$')#, loc='top')
ax.set_ylabel(r'$|\delta B (\ell)|$')#, loc='top')
ax.set_xlabel(r'$\ell / L$')
ax.set_ylim(8e-4, 1.8)
ax.legend(
    loc='upper left',
    fontsize=10,
    frameon=False,
    handlelength=1.5,      # make handle length smaller
    handletextpad=0.5      # make space between handle and text smaller
    # bbox_to_anchor=(1, 0.1)
)



fig.savefig("bsf_median_comparison.pdf", dpi=300, bbox_inches='tight')

# ax.text(
#     0.02, 0.98,
#     r"$\delta B / B_0 = 2$"+"\n"+r"$L/\Delta x = 10{,}240$",
#     transform=ax.transAxes,
#     va='top',
#     ha='left',
#     fontsize=11,
#     color='black'
# )


# In[ ]:




# In[ ]:




# In[ ]:



