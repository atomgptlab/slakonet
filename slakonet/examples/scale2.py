"""Memory-optimized scaling test."""

from slakonet.main import SimpleDftb
from slakonet.atoms import Geometry
import torch
from jarvis.io.vasp.inputs import Poscar
from slakonet.optim import MultiElementSkfParameterOptimizer
import time
import gc

# Setup
model_path = "../tests/Si_only.pt"
model = MultiElementSkfParameterOptimizer.load_ultra_compact(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

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
times = []
cells = list(range(1, 29))

print("Starting scaling benchmark...")
print(f"Cells to test: {cells}")
print()

for i in cells:
    # Aggressive memory cleanup
    torch.cuda.empty_cache()
    gc.collect()

    t1 = time.time()

    new_atoms = atoms.make_supercell([i])
    n_atoms = new_atoms.num_atoms
    ase_atoms = new_atoms.ase_converter()
    geometry = Geometry.from_ase_atoms([ase_atoms])

    # Adaptive k-points based on system size
    if n_atoms < 500:
        kpoints = torch.tensor([2, 2, 2])
    elif n_atoms < 2000:
        kpoints = torch.tensor([1, 1, 1])
    else:
        kpoints = torch.tensor([1, 1, 1])

    # Print memory status before calculation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mem_alloc = torch.cuda.memory_allocated() / (1024**3)
        mem_reserved = torch.cuda.memory_reserved() / (1024**3)
        print(
            f"Before calc: GPU mem allocated={mem_alloc:.1f}GB, reserved={mem_reserved:.1f}GB"
        )

    try:
        # Critical: no gradients!
        with torch.no_grad():
            s = SimpleDftb(geometry, kpoints=kpoints, model=model)
            res = s.calculate()
            H = res["hamiltonian"]

            # Explicitly detach and move to CPU to free GPU
            H_cpu = H.detach().cpu()

            t2 = time.time()
            tot_time = t2 - t1

            print(f"✓ {n_atoms:5d} atoms: {tot_time:7.2f}s, H={H.shape}")
            times.append([n_atoms, tot_time])

            # Clean up
            del s, res, H, H_cpu, geometry, ase_atoms

    except MemoryError as e:
        print(f"❌ {n_atoms:5d} atoms: Memory limit reached")
        print(f"   {str(e)}")
        break

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"❌ {n_atoms:5d} atoms: GPU OOM")
            print(f"   Last successful: {times[-1][0] if times else 0} atoms")
            break
        else:
            print(f"❌ {n_atoms:5d} atoms: Error - {e}")
            raise

    # Aggressive cleanup after each iteration
    torch.cuda.empty_cache()
    gc.collect()

    print()

print("\n" + "=" * 60)
print("SCALING RESULTS")
print("=" * 60)
for n, t in times:
    print(f"{n:5d} atoms: {t:7.2f}s  ({t/n*1000:.2f} ms/atom)")

print("\nFinal results:", times)
