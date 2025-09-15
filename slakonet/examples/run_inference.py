from slakonet.predict_slakonet import get_properties,plot_band_dos_atoms
import pandas as pd
from tqdm import tqdm
from slakonet.optim import (
    MultiElementSkfParameterOptimizer,
    get_atoms,
    kpts_to_klines,
    default_model,
)
from jarvis.db.figshare import data
dataset=data("dft_3d")

df=pd.read_csv("ES-SinglePropertyPrediction-bandgap-dft_3d-test-mae.csv")
f=open('pred.csv','w')
f.write("id,target,prediction\n")
model=default_model()
for ii,i in tqdm(df.iterrows(),total=len(df)):
   print('id',i.id)
   properties, atoms, kpoints=get_properties(model=model,jid=i.id,dataset=dataset)
   
   bandgap = properties["bandgap"].squeeze().detach().cpu().numpy().tolist()
   line=str(i.id)+","+str(i.target)+str(",")+str(round(bandgap,4))+"\n"
   f.write(line)
   print(line)
f.close()
