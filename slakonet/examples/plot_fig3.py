import pandas as pd
from jarvis.db.jsonutils import loadjson
from sklearn.metrics import mean_absolute_error
from jarvis.db.figshare import data
import numpy as np
dataset=data("dft_3d")
def get_data(jid=''):
    for i in dataset:
        if i['jid']==jid:
            return i

df=pd.read_csv("pred.csv")
mem=[]
for i,ii in df.iterrows():
    info={}
    jid=ii.id
    sk=ii.prediction
    exp=ii.target
    dat=get_data(jid)
    opt_gap=dat["optb88vdw_bandgap"]
    mbj_gap=dat["mbj_bandgap"]
    info['bandgap']=float(sk)
    info['opt_gap']=opt_gap
    info['mbj_gap']=mbj_gap
    info['formula']=dat["formula"]
    info['jid']=jid
    info['target']=exp
    mem.append(info)

from sklearn.metrics import mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

the_grid = GridSpec(2, 2)
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(10,8))

plt.subplot(the_grid[1, 0])

dfc=pd.DataFrame(mem)
dfc_clean = dfc.copy()
# dfc_clean['mbj_gap'] = dfc_clean['mbj_gap'].replace('na', np.nan)
# dfc_clean = dfc_clean.dropna(subset=['mbj_gap', 'bandgap'])

plt.scatter(dfc_clean.target,dfc_clean.bandgap,alpha=.5,s=15)
plt.plot(dfc_clean.target,dfc_clean.target,'-',c='black')
#plt.plot(dfc_clean.target,dfc_clean.mbj_gap,'-',c='black')
mae=mean_absolute_error(dfc_clean.target,dfc_clean.bandgap)
print('mae',mae)
plt.xlabel('Exp gap (eV)')
plt.ylabel('TBmBJ gap (eV)')
txt='MAE(eV):'+str(round(mae,2))
plt.text(1.5,7,txt)
plt.xlabel('Exp gap (eV)')
plt.ylabel('SK gap (eV)')
plt.xlim([-0.1,8])
plt.ylim([-0.1,8])
plt.title('(c)')
plt.tight_layout()


plt.subplot(the_grid[0, 1])
plt.title('(b)')
dfc=pd.DataFrame(mem)
dfc_clean = dfc.copy()
dfc_clean['mbj_gap'] = dfc_clean['mbj_gap'].replace('na', np.nan)
dfc_clean = dfc_clean.dropna(subset=['mbj_gap', 'bandgap'])
plt.scatter(dfc_clean.target,dfc_clean.mbj_gap,alpha=.5,s=18)
#plt.plot(dfc_clean.target,dfc_clean.mbj_gap,'-',c='black')
plt.plot(dfc_clean.target,dfc_clean.target,'-',c='black')
mae=mean_absolute_error(dfc_clean.target,dfc_clean.mbj_gap)
print('mae',mae)
plt.xlabel('Exp gap (eV)')
plt.ylabel('TBmBJ gap (eV)')
txt='MAE(eV):'+str(round(mae,2))
plt.text(1.5,7,txt)
plt.xlim([-0.1,8])
plt.ylim([-0.1,8])
#plt.tight_layout()


plt.subplot(the_grid[0, 0])
plt.title('(a)')
dfc=pd.DataFrame(mem)
dfc_clean = dfc.copy()
# dfc_clean['mbj_gap'] = dfc_clean['mbj_gap'].replace('na', np.nan)
# dfc_clean = dfc_clean.dropna(subset=['mbj_gap', 'bandgap'])
plt.scatter(dfc_clean.target,dfc_clean.opt_gap,alpha=.5,s=18)
plt.plot(dfc_clean.target,dfc_clean.target,'-',c='black')
mae=mean_absolute_error(dfc_clean.target,dfc_clean.opt_gap)
print('mae',mae)
plt.xlabel('Exp gap (eV)')
plt.ylabel('OPT gap (eV)')
txt='MAE(eV):'+str(round(mae,2))
plt.text(1.5,7,txt)
plt.xlim([-0.1,8])
plt.ylim([-0.1,8])

pred=loadjson('pred.json')
plt.subplot(the_grid[1, 1])

x=[]
y=[]
for i in pred:
    if i["mbj_gap"]!='na':
        x.append(i["mbj_gap"])
        y.append(i["bandgap"])

plt.scatter(x,y,alpha=.5,s=18)
plt.plot(x,x,'-',c='black')
mae=mean_absolute_error(x,y)
print('mae',mae)
plt.xlabel('TBmBJ gap (eV)')
plt.ylabel('SK gap (eV)')
txt='MAE(eV):'+str(round(mae,2))
plt.text(1.5,19,txt)
plt.xlim([-0.1,21])
plt.ylim([-0.1,21])
plt.tight_layout()

plt.savefig('DFTvsSlaKoNet_2.pdf')
plt.close()




