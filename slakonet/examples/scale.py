from slakonet.main import SimpleDftb
import gc
from slakonet.atoms import Geometry
import torch
from jarvis.io.vasp.inputs import Poscar
from slakonet.atoms import Geometry, Periodic
import torch
import numpy as np
from slakonet.slaterkoster import fermi, hs_matrix
from slakonet.optim import MultiElementSkfParameterOptimizer
from slakonet.main import run_calc, SlakoNetCalculator
from slakonet.utils import eighb, create_feeds, generate_shell_dict_upto_Z65
from slakonet.basis import Basis
import time

model_path = "../tests/Si_only.pt"
model = MultiElementSkfParameterOptimizer.load_ultra_compact(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
updated_skfs = model.get_updated_skfs()

# Test 1: Unit verification
poscar = """Si2
1.0
3.3641499856336465 -2.5027128e-09 1.94229273881412
1.121382991333525 3.1717517190189715 1.9422927388141193
-2.5909987e-09 -1.8321133e-09 3.884586486670313
Si
2
direct
0.875 0.875 0.875
0.125 0.125 0.125
"""

atoms = Poscar.from_string(poscar).atoms
shell_dict = generate_shell_dict_upto_Z65()
h_feed = create_feeds(updated_skfs, shell_dict, "H")
s_feed = create_feeds(updated_skfs, shell_dict, "S")
times = []
cells = list(range(1, 19))
print(cells)
for i in cells:
    t1 = time.time()
    new_atoms = atoms.make_supercell([i])
    ase_atoms = new_atoms.ase_converter()
    geometry = Geometry.from_ase_atoms([ase_atoms])
    kpoints = torch.tensor([1, 1, 1])
    with torch.no_grad():
        s = SimpleDftb(
            geometry, kpoints=kpoints, model=model, compute_forces=False
        )
        ##s = SimpleDftb(geometry,klines=klines,model=model)
        # print('ele',s.nelectron)
        res = s.calculate()
        H = res["hamiltonian"]
    """
    periodic = Periodic(
        geometry,
        geometry.cell,
        cutoff=7.1,
        kpoints=torch.tensor([2, 2, 2]),  # Use stored original
    )
    basis = Basis(geometry.atomic_numbers, shell_dict)

    # Build Hamiltonian and overlap matrices
    H = hs_matrix(periodic, basis, h_feed)
    S = hs_matrix(periodic, basis, s_feed)
    """

    t2 = time.time()
    tot_time = t2 - t1
    print(new_atoms.num_atoms, H.shape, tot_time)
    times.append([new_atoms.num_atoms, tot_time])
    print(times)
    print()
    torch.cuda.empty_cache()
    gc.collect()
