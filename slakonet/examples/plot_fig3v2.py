import pandas as pd
from jarvis.db.jsonutils import loadjson
from sklearn.metrics import mean_absolute_error
from jarvis.db.figshare import data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from matplotlib.patches import Rectangle

# Set the style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")

# Enhanced color palette
colors = {
    'scatter': '#2E86AB',      # Professional blue
    'perfect': '#A23B72',      # Deep magenta for perfect prediction line
    'text_bg': 'white',        # Background for text boxes
    'grid': '#E5E5E5'         # Light grid color
}

# Load and prepare data (keeping your original data loading logic)
dataset = data("dft_3d")

def get_data(jid=''):
    for i in dataset:
        if i['jid'] == jid:
            return i

df = pd.read_csv("pred.csv")
mem = []
for i, ii in df.iterrows():
    info = {}
    jid = ii.id
    sk = ii.prediction
    exp = ii.target
    dat = get_data(jid)
    opt_gap = dat["optb88vdw_bandgap"]
    mbj_gap = dat["mbj_bandgap"]
    info['bandgap'] = float(sk)
    info['opt_gap'] = opt_gap
    info['mbj_gap'] = mbj_gap
    info['formula'] = dat["formula"]
    info['jid'] = jid
    info['target'] = exp
    mem.append(info)

# Create the enhanced figure
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('white')

# Use a more sophisticated grid layout
gs = GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.25)

# Set global font parameters
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.minor.width': 1.0,
    'ytick.minor.width': 1.0,
})

def create_enhanced_subplot(ax, x_data, y_data, xlabel, ylabel, title, xlim, ylim, text_pos=(1.5, 7)):
    """Create an enhanced subplot with consistent styling"""
    
    # Add subtle grid
    ax.grid(True, alpha=0.3, color=colors['grid'], linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Scatter plot with enhanced styling
    scatter = ax.scatter(x_data, y_data, 
                        alpha=0.6, 
                        s=25, 
                        c=colors['scatter'],
                        edgecolors='white', 
                        linewidth=0.5,
                        zorder=3)
    
    # Perfect prediction line with enhanced styling
    perfect_line = ax.plot(xlim, xlim, 
                          color=colors['perfect'], 
                          linewidth=2.5, 
                          linestyle='-', 
                          alpha=0.8,
                          label='Perfect prediction',
                          zorder=2)
    
    # Calculate MAE
    mae = mean_absolute_error(x_data, y_data)
    
    # Enhanced text box for MAE
    textstr = f'MAE = {mae:.2f} eV'
    props = dict(boxstyle='round,pad=0.5', 
                facecolor=colors['text_bg'], 
                alpha=0.9, 
                edgecolor='gray',
                linewidth=1)
    ax.text(text_pos[0], text_pos[1], textstr, 
           fontsize=11, 
           bbox=props,
           fontweight='bold')
    
    # Enhanced labels and title
    ax.set_xlabel(xlabel, fontsize=13, fontweight='bold', labelpad=8)
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold', labelpad=8)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    # Set limits and ticks
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Enhanced tick parameters
    ax.tick_params(axis='both', which='major', labelsize=11, 
                   length=6, width=1.5, direction='in')
    ax.tick_params(axis='both', which='minor', length=3, width=1, direction='in')
    
    # Add minor ticks
    ax.minorticks_on()
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    return mae

# Create DataFrame
dfc = pd.DataFrame(mem)

# Subplot (a) - OPT vs Experimental
ax1 = fig.add_subplot(gs[0, 0])
dfc_clean = dfc.copy()
mae1 = create_enhanced_subplot(ax1, dfc_clean.target, dfc_clean.opt_gap,
                              'Experimental gap (eV)', 'OPT gap (eV)', 
                              '(a) OptB88vdW vs Experimental', 
                              [-0.1, 8], [-0.1, 8])

# Subplot (b) - TBmBJ vs Experimental
ax2 = fig.add_subplot(gs[0, 1])
dfc_clean = dfc.copy()
dfc_clean['mbj_gap'] = dfc_clean['mbj_gap'].replace('na', np.nan)
dfc_clean = dfc_clean.dropna(subset=['mbj_gap', 'bandgap'])
mae2 = create_enhanced_subplot(ax2, dfc_clean.target, dfc_clean.mbj_gap,
                              'Experimental gap (eV)', 'TBmBJ gap (eV)',
                              '(b) TBmBJ vs Experimental',
                              [-0.1, 8], [-0.1, 8])

# Subplot (c) - SK vs Experimental
ax3 = fig.add_subplot(gs[1, 0])
dfc_clean = dfc.copy()
mae3 = create_enhanced_subplot(ax3, dfc_clean.target, dfc_clean.bandgap,
                              'Experimental gap (eV)', 'SK gap (eV)',
                              '(c) SlaKoNet vs Experimental',
                              [-0.1, 8], [-0.1, 8])

# Subplot (d) - SK vs TBmBJ
ax4 = fig.add_subplot(gs[1, 1])
pred = loadjson('pred.json')
x = []
y = []
for i in pred:
    if i["mbj_gap"] != 'na':
        x.append(i["mbj_gap"])
        y.append(i["bandgap"])

mae4 = create_enhanced_subplot(ax4, x, y,
                              'TBmBJ gap (eV)', 'SK gap (eV)',
                              '(d) SlaKoNet vs TBmBJ',
                              [-0.1, 21], [-0.1, 21], 
                              text_pos=(1.5, 19))

# Add a main title
#fig.suptitle('Bandgap Prediction Comparison Across Methods', 
#             fontsize=16, fontweight='bold', y=0.95)

# Print MAE values for reference
print(f'MAE values:')
print(f'OptB88vdW vs Experimental: {mae1:.3f} eV')
print(f'TBmBJ vs Experimental: {mae2:.3f} eV') 
print(f'SlaKoNet vs Experimental: {mae3:.3f} eV')
print(f'SlaKoNet vs TBmBJ: {mae4:.3f} eV')

# Save with high quality
plt.savefig('DFTvsSlaKoNet_enhanced.pdf', 
           dpi=300, 
           bbox_inches='tight', 
           facecolor='white',
           edgecolor='none')

plt.savefig('DFTvsSlaKoNet_enhanced.png', 
           dpi=300, 
           bbox_inches='tight', 
           facecolor='white',
           edgecolor='none')

plt.show()
plt.close()
