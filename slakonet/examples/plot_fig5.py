# %matplotlib inline
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

plt.rcParams.update({"font.size": 22})
import torch
import sys


nums = [2, 16, 54, 128, 250, 432, 686, 1024, 1458, 2000]
times_cpu = [
    4.981168746948242,
    4.553197622299194,
    4.88469934463501,
    5.553440093994141,
    7.160606145858765,
    13.558997869491577,
    34.77856516838074,
    99.5997314453125,
    255.79310011863708,
    631.9005651473999,
]
times_gpu = [
    5.525652647018433,
    4.9758460521698,
    4.890452861785889,
    5.127196788787842,
    5.881839990615845,
    7.848914384841919,
    12.485913515090942,
    22.772614002227783,
    41.18002128601074,
    75.67503428459167,
]

plt.plot(nums, times_cpu)
plt.plot(nums, times_gpu)
plt.tight_layout("times.png")
plt.close()

sys.exit()

model_best = default_model()

atoms, _, _ = get_atoms(jid="JVASP-1002")
shell_dict = generate_shell_dict_upto_Z65()
kpoints = torch.tensor([1, 1, 1])

scells = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# scells=[1,2,3,4,5]
times_gpu = []
times_cpu = []
nums = []
for i in scells:
    s = atoms.make_supercell_matrix([i, i, i])
    geometry = Geometry.from_ase_atoms([s.ase_converter()])
    with torch.no_grad():  # No gradients needed for inference
        t1 = time.time()
        properties, success = model_best.compute_multi_element_properties(
            geometry=geometry,
            shell_dict=shell_dict,
            kpoints=kpoints,
            get_energy=True,
            device="cpu",
        )
        en = properties["total_energy"]
        t2 = time.time()
        times_cpu.append(t2 - t1)
        # print(times_cpu[-1])
        nums.append(s.num_atoms)
    with torch.no_grad():  # No gradients needed for inference
        t1 = time.time()
        properties, success = model_best.compute_multi_element_properties(
            geometry=geometry,
            shell_dict=shell_dict,
            kpoints=kpoints,
            get_energy=True,
            device="cuda",
        )
        en = properties["total_energy"]
        t2 = time.time()
        times_gpu.append(t2 - t1)
        # print(times_gpu[-1])

    print("i,num,cpu,gpu", i, s.num_atoms, times_cpu[-1], times_gpu[-1])
print("nums", nums)
print("times_cpu", times_cpu)
print("times_gpu", times_gpu)
