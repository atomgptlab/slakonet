#%matplotlib inline
from tqdm import tqdm
import time
from jarvis.core.kpoints import Kpoints3D as Kpoints
from slakonet.atoms import Geometry
from slakonet.optim import get_atoms
import matplotlib.pyplot as plt
from slakonet.optim import (
    MultiElementSkfParameterOptimizer,
    get_atoms,
    kpts_to_klines,
    default_model,
)
from slakonet.main import generate_shell_dict_upto_Z65
plt.rcParams.update({'font.size': 22})
import torch
model_best=default_model()

atoms,_,_=get_atoms(jid='JVASP-1002')
shell_dict = generate_shell_dict_upto_Z65()
kpoints=torch.tensor([1,1,1])

scells=[1,2,3,4,5,6,7,8,9,10]
#scells=[1,2,3,4,5]
times_gpu=[]
times_cpu=[]
nums=[]
for i in scells:
    s=atoms.make_supercell_matrix([i,i,i])
    geometry=Geometry.from_ase_atoms([s.ase_converter()])
    with torch.no_grad():  # No gradients needed for inference
        t1=time.time()
        properties, success = model_best.compute_multi_element_properties(
            geometry=geometry, shell_dict=shell_dict, kpoints=kpoints,get_energy=True,device="cpu"
        )
        en=properties["total_energy"]
        t2=time.time()
        times_cpu.append(t2-t1)
        #print(times_cpu[-1])
        nums.append(s.num_atoms)
    with torch.no_grad():  # No gradients needed for inference
        t1=time.time()
        properties, success = model_best.compute_multi_element_properties(
            geometry=geometry, shell_dict=shell_dict, kpoints=kpoints,get_energy=True,device="cuda"
        )
        en=properties["total_energy"]
        t2=time.time()
        times_gpu.append(t2-t1)
        #print(times_gpu[-1])
      

    print("i,num,cpu,gpu",i,s.num_atoms,times_cpu[-1],times_gpu[-1])
print("nums",nums)
print("times_cpu",times_cpu)
print("times_gpu",times_gpu)

