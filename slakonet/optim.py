#!/usr/bin/env python3
"""
Extension to train on multiple vasprun.xml files
Add this to your existing MultiElementSkfParameterOptimizer class
"""

import glob
from pathlib import Path
import random
import torch
import pickle
import json
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
import os
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data
from collections import defaultdict
from jarvis.core.kpoints import Kpoints3D as Kpoints
from itertools import combinations_with_replacement
from jarvis.core.specie import atomic_numbers_to_symbols
from slakonet.skf import Skf
from slakonet.main import SimpleDftb, generate_shell_dict_upto_Z65
from slakonet.skfeed import SkfFeed, _get_hs_dict, _get_onsite_dict
from slakonet.interpolation import PolyInterpU
from slakonet.atoms import Geometry
from jarvis.io.vasp.outputs import Vasprun
import matplotlib.pyplot as plt
import matplotlib
from slakonet.fermi import fermi_search, fermi_smearing, fermi_dirac
from jarvis.core.specie import atomic_numbers_to_symbols
import random
from tqdm import tqdm
import zipfile
import requests
import io

matplotlib.rcParams["figure.max_open_warning"] = 50
# torch.set_default_dtype(torch.float32)
# torch.set_default_dtype(torch.float32)

random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
try:
    import torch_xla.core.xla_model as xm

    xm.set_rng_state(random_seed)
except ImportError:
    pass
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(random_seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(":4096:8")
torch.use_deterministic_algorithms(True)


def get_atoms(jid="", dataset=None, id_tag="jid"):
    if dataset is None:
        dataset = data("dft_3d")
    for i in dataset:
        if i[id_tag] == jid:
            return (
                Atoms.from_dict(i["atoms"]),
                i["optb88vdw_bandgap"],
                i["mbj_bandgap"],
            )


def kpts_to_klines(kpts, default_points=10):
    """
    Convert a sequence of k-points into segments for band path plotting.

    Args:
        kpts (list[list[float]] or torch.Tensor): List of k-points (Nx3)
        default_points (int): Number of interpolation points between each segment

    Returns:
        torch.Tensor: Tensor of shape (num_segments, 7)
                      Each row is [kx1, ky1, kz1, kx2, ky2, kz2, n_points]
    """
    if not isinstance(kpts, torch.Tensor):
        kpts = torch.tensor(kpts).type(torch.get_default_dtype())
        # kpts = torch.tensor(kpts, dtype=torch.float32)

    num_pairs = (kpts.shape[0] - 1) // 2 + ((kpts.shape[0] - 1) % 2 == 0)
    segments = []

    for i in range(0, kpts.shape[0] - 1, 2):
        k1 = kpts[i]
        k2 = kpts[i + 1]
        seg = torch.cat(
            [
                k1,
                k2,
                torch.tensor([default_points]).type(torch.get_default_dtype()),
            ]
            # [k1, k2, torch.tensor([default_points], dtype=torch.float32)]
        )
        segments.append(seg)

    return torch.stack(segments, dim=0)


def get_klines_example(
    jid="JVASP-1002", model=None, plot=False, default_points=2, line_density=20
):
    # jid='JVASP-14636'
    atoms, opt_gap, mbj_gap = get_atoms(
        jid
    )  # Atoms.from_dict(get_jid_data(jid=jid,dataset='dft_3d')['atoms'])
    # atoms=Atoms.from_poscar("tests/POSCAR")
    # atoms=Atoms.from_poscar("tests/POSCAR-SiC.vasp")
    geometry = Geometry.from_ase_atoms([atoms.ase_converter()])
    # Generate shell dictionary
    shell_dict = generate_shell_dict_upto_Z65()
    kpoints = Kpoints().kpath(atoms, line_density=line_density)
    labels = kpoints.labels
    xticks = []
    xtick_labels = []
    kps = []
    for ii, i in enumerate(labels):
        kps.append(kpoints.kpts[ii])
        lbl = "$" + i + "$"
        # lbl=lbl.replace("\\G","\G")
        if ii == 0 and lbl != "$$":
            xticks.append(ii * int(default_points / 2))
            xtick_labels.append(lbl)

        if lbl != "$$" and labels[ii] != labels[ii - 1]:
            xticks.append(ii * int(default_points / 2))
            xtick_labels.append(lbl)
            # kps.append(kpoints.kpts[ii])

    # print(xtick_labels)
    formula = atoms.composition.reduced_formula
    klines = kpts_to_klines(kpoints.kpts, default_points=default_points)
    return klines


class MultiElementSkfParameterOptimizer(nn.Module):
    """Enhanced Universal SKF parameter optimizer for multi-element systems"""

    def __init__(
        self,
        skf_directory,
        geometry=None,
        vasprun_path=None,
        available_skf_pairs=None,
        universal_params_file=None,
        elements_in_system=["Si", "C"],
    ):
        super().__init__()

        self.skf_directory = skf_directory
        self.element_pairs = set()
        self.skf_optimizers = nn.ModuleDict()
        self.elements_in_system = set(elements_in_system)
        # self.elements_in_system = set()

        # Atomic number to symbol mapping
        zz = [i for i in range(1, 100)]
        z = atomic_numbers_to_symbols(zz)
        self.atomic_num_to_symbol = dict(zip(zz, z))

        # Detect elements from geometry FIRST
        if geometry is not None:
            self.elements_in_system = self._extract_elements_from_geometry(
                geometry
            )
            print(
                f"Elements detected in geometry: {sorted(self.elements_in_system)}"
            )

        # Check for universal parameters file
        universal_file = universal_params_file or os.path.join(
            skf_directory, "universal_initial_params.pt"
        )

        if os.path.exists(universal_file):
            print(
                f"🔄 Loading from universal parameters file: {universal_file}"
            )
            self._load_from_universal_params(universal_file)
        else:
            print(f"📁 Loading from individual SKF files...")
            detected_elements = self._detect_elements(
                vasprun_path, geometry, available_skf_pairs
            )

            if not detected_elements:
                raise ValueError(
                    "No elements detected. Please provide geometry, vasprun_path, or available_skf_pairs"
                )

            print(f"All detected elements: {sorted(detected_elements)}")

            # Generate all possible element pairs for detected elements
            self.element_pairs = set(
                combinations_with_replacement(sorted(detected_elements), 2)
            )
            print(f"Element pairs to optimize: {sorted(self.element_pairs)}")

            self._initialize_skf_optimizers()

    def save_model(self, save_path, method="state_dict"):
        """
        Save the model using different methods

        Args:
            save_path: Path to save the model
            method: 'state_dict', 'full_model', or 'universal_params'
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if method == "state_dict":
            self._save_state_dict_method(save_path)
        elif method == "compact":
            self.save_ultra_compact(save_path)
        elif method == "full_model":
            self._save_full_model_method(save_path)
        elif method == "universal_params":
            self._save_universal_params_method(save_path)
        else:
            raise ValueError(f"Unknown save method: {method}")

    def _save_state_dict_method(self, save_path):
        """Save using state_dict + metadata (most reliable)"""
        # Create save directory
        save_dir = save_path.with_suffix("")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save the state dict
        torch.save(self.state_dict(), save_dir / "model_state.pt")

        # Save metadata needed for reconstruction
        metadata = {
            "skf_directory": self.skf_directory,
            "elements_in_system": list(self.elements_in_system),
            "element_pairs": [list(pair) for pair in self.element_pairs],
            "available_pairs": list(self.skf_optimizers.keys()),
            "class_name": "MultiElementSkfParameterOptimizer",
        }

        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save original SKF data for each optimizer
        skf_data = {}
        for pair_key, optimizer in self.skf_optimizers.items():
            skf_data[pair_key] = {
                "skf_dict": optimizer.skf_dict,
                "original_h_params": {
                    k: v.tolist()
                    for k, v in optimizer.original_h_params.items()
                },
                "original_s_params": {
                    k: v.tolist()
                    for k, v in optimizer.original_s_params.items()
                },
            }

        torch.save(skf_data, save_dir / "skf_data.pt")

        print(f"✅ Model saved using state_dict method to: {save_dir}")
        # print(f"   - model_state.pt: PyTorch state dict")
        # print(f"   - metadata.json: Model configuration")
        # print(f"   - skf_data.pt: Original SKF data")

    def _save_full_model_method(self, save_path):
        """Save full model (less reliable due to pickle issues)"""
        save_file = save_path.parent / f"{save_path.name}.pt"
        torch.save(self, save_file)
        print(f"⚠️  Model saved using full model method to: {save_file}")
        print("   Note: May have issues loading if class definition changes")

    def _save_universal_params_method(self, save_path):
        """Save as universal parameters file"""
        universal_params = {}

        for pair_key, optimizer in self.skf_optimizers.items():
            universal_params[pair_key] = {
                "h_params": {
                    k: v.detach().clone()
                    for k, v in optimizer.h_params.items()
                },
                "s_params": {
                    k: v.detach().clone()
                    for k, v in optimizer.s_params.items()
                },
                "skf_dict": optimizer.skf_dict,
            }

        # Fix: Use proper path construction instead of with_suffix
        save_file = save_path.parent / f"{save_path.name}_universal_params.pt"
        torch.save(universal_params, save_file)
        print(f"✅ Universal parameters saved to: {save_file}")

    @classmethod
    def load_model(cls, load_path, method="state_dict", skf_directory=None):
        """
        Load the model using different methods

        Args:
            load_path: Path to load the model from
            method: 'state_dict', 'full_model', or 'universal_params'
            skf_directory: SKF directory (needed for some methods)
        """
        load_path = Path(load_path)

        if method == "state_dict":
            return cls._load_state_dict_method(load_path)
        elif method == "full_model":
            return cls._load_full_model_method(load_path)
        elif method == "compact":
            return cls.load_ultra_compact(load_path)
        elif method == "universal_params":
            return cls._load_universal_params_method(load_path, skf_directory)
        else:
            raise ValueError(f"Unknown load method: {method}")

    @classmethod
    def _load_state_dict_method(cls, load_path):
        """Load using state_dict + metadata (most reliable)"""
        print("Loading model ...")
        t1 = time.time()
        if load_path.is_file():
            # If it's a file, assume it's the directory name
            load_dir = load_path.with_suffix("")
        else:
            load_dir = load_path

        # Load metadata
        with open(load_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Create new instance with minimal initialization
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)

        # Restore basic attributes
        instance.skf_directory = metadata["skf_directory"]
        instance.elements_in_system = set(metadata["elements_in_system"])
        instance.element_pairs = set(
            tuple(pair) for pair in metadata["element_pairs"]
        )

        # Load SKF data
        skf_data = torch.load(load_dir / "skf_data.pt")

        # Recreate atomic number mapping

        zz = [i for i in range(1, 100)]
        z = atomic_numbers_to_symbols(zz)
        instance.atomic_num_to_symbol = dict(zip(zz, z))

        # Recreate SKF optimizers
        instance.skf_optimizers = nn.ModuleDict()
        for pair_key, data in skf_data.items():
            # Create SkfParameterOptimizer manually
            optimizer = SkfParameterOptimizer.__new__(SkfParameterOptimizer)
            nn.Module.__init__(optimizer)

            # Restore attributes
            optimizer.skf_dict = data["skf_dict"]
            optimizer.original_h_params = {
                k: torch.tensor(v)
                for k, v in data["original_h_params"].items()
            }
            optimizer.original_s_params = {
                k: torch.tensor(v)
                for k, v in data["original_s_params"].items()
            }

            # Create parameter dicts
            h_param_dict = {
                k: nn.Parameter(torch.tensor(v))
                for k, v in data["original_h_params"].items()
            }
            s_param_dict = {
                k: nn.Parameter(torch.tensor(v))
                for k, v in data["original_s_params"].items()
            }

            optimizer.h_params = nn.ParameterDict(h_param_dict)
            optimizer.s_params = nn.ParameterDict(s_param_dict)

            # Restore other attributes
            optimizer.grid = optimizer.skf_dict.get("grid", None)
            optimizer.atomic_data = optimizer.skf_dict.get("atomic_data", None)
            optimizer.atom_pair = optimizer.skf_dict.get("atom_pair", None)
            optimizer.hs_cutoff = optimizer.skf_dict.get("hs_cutoff", None)

            instance.skf_optimizers[pair_key] = optimizer

        # Load the trained state dict
        state_dict = torch.load(load_dir / "model_state.pt")
        instance.load_state_dict(state_dict)

        t2 = time.time()
        print(f"✅ Model loaded using state_dict method from: {load_dir}")
        print("Time taken:", round(t2 - t1, 3))
        return instance

    @classmethod
    def _load_full_model_method(cls, load_path):
        """Load full model (may fail due to pickle issues)"""
        try:
            load_file = load_path.parent / f"{load_path.name}.pt"
            model = torch.load(load_file)
            print(f"✅ Model loaded using full model method from: {load_file}")
            return model
        except Exception as e:
            print(f"❌ Failed to load full model: {e}")
            raise

    @classmethod
    def _load_universal_params_method(cls, load_path, skf_directory):
        """Load from universal parameters file"""
        if skf_directory is None:
            raise ValueError(
                "skf_directory must be provided for universal_params method"
            )

        load_file = load_path.parent / f"{load_path.name}_universal_params.pt"
        universal_params = torch.load(load_file)

        # Create new instance
        instance = cls(skf_directory, universal_params_file=str(load_file))

        print(
            f"✅ Model loaded using universal_params method from: {load_file}"
        )
        return instance

    def _load_from_universal_params(self, universal_file):
        """Load from universal parameters file (used in __init__)"""
        universal_params = torch.load(universal_file)

        self.skf_optimizers = nn.ModuleDict()

        for pair_key, param_data in universal_params.items():
            # Create optimizer instance
            optimizer = SkfParameterOptimizer.__new__(SkfParameterOptimizer)
            nn.Module.__init__(optimizer)

            # Set attributes
            optimizer.skf_dict = param_data["skf_dict"]

            # Create parameter dicts from saved parameters
            h_param_dict = {}
            for key, param_tensor in param_data["h_params"].items():
                h_param_dict[key] = nn.Parameter(param_tensor.detach().clone())

            s_param_dict = {}
            for key, param_tensor in param_data["s_params"].items():
                s_param_dict[key] = nn.Parameter(param_tensor.detach().clone())

            optimizer.h_params = nn.ParameterDict(h_param_dict)
            optimizer.s_params = nn.ParameterDict(s_param_dict)

            # Set original parameters as copies
            optimizer.original_h_params = {
                k: v.detach().clone() for k, v in optimizer.h_params.items()
            }
            optimizer.original_s_params = {
                k: v.detach().clone() for k, v in optimizer.s_params.items()
            }

            # Set other attributes
            optimizer.grid = optimizer.skf_dict.get("grid", None)
            optimizer.atomic_data = optimizer.skf_dict.get("atomic_data", None)
            optimizer.atom_pair = optimizer.skf_dict.get("atom_pair", None)
            optimizer.hs_cutoff = optimizer.skf_dict.get("hs_cutoff", None)

            self.skf_optimizers[pair_key] = optimizer

            # Update element tracking
            elements = pair_key.split("-")
            self.elements_in_system.update(elements)

        print(
            f"Loaded {len(self.skf_optimizers)} optimizers from universal parameters"
        )

    def save_ultra_compact(self, save_path):
        """
        Save everything in a single .pt file with minimal redundancy
        Only stores trained parameters once, reconstructs skf_dict on load
        """
        save_file = Path(save_path).with_suffix(".pt")
        save_file.parent.mkdir(parents=True, exist_ok=True)

        # Get current trained parameters
        state_dict = self.state_dict()

        compact_data = {
            "metadata": {
                "skf_directory": self.skf_directory,
                "elements_in_system": list(self.elements_in_system),
                "element_pairs": [list(pair) for pair in self.element_pairs],
                "available_pairs": list(self.skf_optimizers.keys()),
                "class_name": "MultiElementSkfParameterOptimizer",
                "ultra_compact": True,
            },
            "trained_parameters": state_dict,  # Only store trained params once
            "skf_metadata": {},  # Only store non-parameter data from skf_dict
        }

        # Store only non-parameter metadata from each SKF
        for pair_key, optimizer in self.skf_optimizers.items():
            skf_dict = optimizer.skf_dict.copy()

            # Remove parameter data (we have it in state_dict)
            skf_dict.pop("hamiltonian", None)
            skf_dict.pop("overlap", None)

            compact_data["skf_metadata"][pair_key] = skf_dict

        torch.save(compact_data, save_file)

        # Calculate size savings
        original_h_size = sum(
            len(opt.skf_dict.get("hamiltonian", {}))
            for opt in self.skf_optimizers.values()
        )
        original_s_size = sum(
            len(opt.skf_dict.get("overlap", {}))
            for opt in self.skf_optimizers.values()
        )
        total_eliminated = original_h_size + original_s_size

        print(f"✅ Compact model saved to: {save_file}")
        print(f"   Eliminated {total_eliminated} duplicate parameter copies")
        print(f"   Estimated ~75% size reduction from eliminating redundancy")

    @classmethod
    def load_ultra_compact(cls, load_path):
        """
        Load ultra-compact model and reconstruct skf_dict from trained parameters
        """
        t1 = time.time()
        load_file = Path(load_path).with_suffix(".pt")
        compact_data = torch.load(load_file)

        if not compact_data["metadata"].get("ultra_compact", False):
            raise ValueError("This is not an ultra-compact model file")

        metadata = compact_data["metadata"]
        state_dict = compact_data["trained_parameters"]
        skf_metadata = compact_data["skf_metadata"]

        # Create new instance
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)

        # Restore basic attributes
        instance.skf_directory = metadata["skf_directory"]
        instance.elements_in_system = set(metadata["elements_in_system"])
        instance.element_pairs = set(
            tuple(pair) for pair in metadata["element_pairs"]
        )

        # Recreate atomic number mapping
        from jarvis.core.specie import atomic_numbers_to_symbols

        zz = [i for i in range(1, 100)]
        z = atomic_numbers_to_symbols(zz)
        instance.atomic_num_to_symbol = dict(zip(zz, z))

        # Recreate SKF optimizers
        instance.skf_optimizers = nn.ModuleDict()

        for pair_key in metadata["available_pairs"]:
            # Create optimizer
            optimizer = SkfParameterOptimizer.__new__(SkfParameterOptimizer)
            nn.Module.__init__(optimizer)

            # Get the metadata (everything except hamiltonian/overlap)
            skf_dict = skf_metadata[pair_key].copy()

            # Extract trained parameters for this pair from state_dict
            h_params = {}
            s_params = {}

            for key, value in state_dict.items():
                if key.startswith(f"skf_optimizers.{pair_key}.h_params."):
                    param_name = key.replace(
                        f"skf_optimizers.{pair_key}.h_params.", ""
                    )
                    h_params[param_name] = value
                elif key.startswith(f"skf_optimizers.{pair_key}.s_params."):
                    param_name = key.replace(
                        f"skf_optimizers.{pair_key}.s_params.", ""
                    )
                    s_params[param_name] = value

            # Reconstruct full skf_dict with trained parameters
            skf_dict["hamiltonian"] = h_params
            skf_dict["overlap"] = s_params

            optimizer.skf_dict = skf_dict

            # Create parameter dicts
            optimizer.h_params = nn.ParameterDict(
                {k: nn.Parameter(v.clone()) for k, v in h_params.items()}
            )
            optimizer.s_params = nn.ParameterDict(
                {k: nn.Parameter(v.clone()) for k, v in s_params.items()}
            )

            # Set other attributes
            optimizer.grid = skf_dict.get("grid", None)
            optimizer.atomic_data = skf_dict.get("atomic_data", None)
            optimizer.atom_pair = skf_dict.get("atom_pair", None)
            optimizer.hs_cutoff = skf_dict.get("hs_cutoff", None)

            instance.skf_optimizers[pair_key] = optimizer

        # Load the state dict (this should work since we reconstructed the structure)
        instance.load_state_dict(state_dict)
        t2 = time.time()

        print(f"✅ Compact model loaded from: {load_file}")
        print("Time taken:", round(t2 - t1, 3))
        return instance

    def save_without_orig(self, save_path):
        """
        Save model without original parameters to reduce file size

        Args:
            save_path: Path to save the model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Create save directory
        save_dir = save_path.with_suffix("")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save the state dict
        torch.save(self.state_dict(), save_dir / "model_state.pt")

        # Save metadata needed for reconstruction
        metadata = {
            "skf_directory": self.skf_directory,
            "elements_in_system": list(self.elements_in_system),
            "element_pairs": [list(pair) for pair in self.element_pairs],
            "available_pairs": list(self.skf_optimizers.keys()),
            "class_name": "MultiElementSkfParameterOptimizer",
            "compact_version": True,  # Flag to indicate this is the compact version
        }

        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save SKF data WITHOUT original parameters
        skf_data = {}
        for pair_key, optimizer in self.skf_optimizers.items():
            skf_data[pair_key] = {
                "skf_dict": optimizer.skf_dict
                # No original_h_params or original_s_params stored
            }

        torch.save(skf_data, save_dir / "skf_data.pt")

        print(
            f"✅ Compact model saved (without original params) to: {save_dir}"
        )

        # Calculate size reduction estimate
        total_params = sum(p.numel() for p in self.parameters())
        print(
            f"   Reduced storage by ~{total_params} parameter values (~50% size reduction)"
        )

    @classmethod
    def load_without_orig(cls, load_path):
        """
        Load model saved without original parameters

        Args:
            load_path: Path to load the model from
        """
        t1 = time.time()
        load_path = Path(load_path)

        if load_path.is_file():
            load_dir = load_path.with_suffix("")
        else:
            load_dir = load_path

        # Load metadata
        with open(load_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Check if this is a compact version
        if not metadata.get("compact_version", False):
            print("⚠️  This doesn't appear to be a compact model")

        # Create new instance with minimal initialization
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)

        # Restore basic attributes
        instance.skf_directory = metadata["skf_directory"]
        instance.elements_in_system = set(metadata["elements_in_system"])
        instance.element_pairs = set(
            tuple(pair) for pair in metadata["element_pairs"]
        )

        # Load SKF data
        skf_data = torch.load(load_dir / "skf_data.pt")

        # Recreate atomic number mapping
        from jarvis.core.specie import atomic_numbers_to_symbols

        zz = [i for i in range(1, 100)]
        z = atomic_numbers_to_symbols(zz)
        instance.atomic_num_to_symbol = dict(zip(zz, z))

        # Recreate SKF optimizers WITHOUT original parameters
        instance.skf_optimizers = nn.ModuleDict()
        for pair_key, data in skf_data.items():
            # Create SkfParameterOptimizer manually
            optimizer = SkfParameterOptimizer.__new__(SkfParameterOptimizer)
            nn.Module.__init__(optimizer)

            # Restore attributes
            optimizer.skf_dict = data["skf_dict"]

            # Create parameter dicts directly from skf_dict (no original params)
            h_param_dict = {
                k: nn.Parameter(
                    torch.tensor(v).type(torch.get_default_dtype())
                )
                # k: nn.Parameter(torch.tensor(v, dtype=torch.float32))
                for k, v in optimizer.skf_dict["hamiltonian"].items()
            }
            s_param_dict = {
                k: nn.Parameter(
                    torch.tensor(v).type(torch.get_default_dtype())
                )
                # k: nn.Parameter(torch.tensor(v, dtype=torch.float32))
                for k, v in optimizer.skf_dict["overlap"].items()
            }

            optimizer.h_params = nn.ParameterDict(h_param_dict)
            optimizer.s_params = nn.ParameterDict(s_param_dict)

            # Set other attributes
            optimizer.grid = optimizer.skf_dict.get("grid", None)
            optimizer.atomic_data = optimizer.skf_dict.get("atomic_data", None)
            optimizer.atom_pair = optimizer.skf_dict.get("atom_pair", None)
            optimizer.hs_cutoff = optimizer.skf_dict.get("hs_cutoff", None)

            instance.skf_optimizers[pair_key] = optimizer

        # Load the trained state dict
        state_dict = torch.load(load_dir / "model_state.pt")
        instance.load_state_dict(state_dict)
        t2 = time.time()

        print(f"✅ Compact model loaded from: {load_dir}")
        print("Time taken", round(t2 - t1, 3))
        return instance

    def get_available_pairs(self):
        """Get available element pairs"""
        return list(self.skf_optimizers.keys())

    def debug_feed_coverage(self, geometry):
        """Debug function to check what interactions are needed vs available"""
        print("\n" + "=" * 50)
        print("FEED COVERAGE DEBUG")
        print("=" * 50)

        # Extract atomic numbers from geometry
        if hasattr(geometry, "atomic_numbers"):
            atomic_nums = geometry.atomic_numbers.flatten().unique().tolist()
        elif hasattr(geometry, "Z"):
            atomic_nums = geometry.Z.flatten().unique().tolist()
        else:
            print("Cannot extract atomic numbers from geometry")
            return

        print(f"Atomic numbers in geometry: {atomic_nums}")

        # Get shell information
        shell_dict = generate_shell_dict_upto_Z65()

        # Check what interactions we need
        needed_interactions = []
        for i, z1 in enumerate(atomic_nums):
            for j, z2 in enumerate(atomic_nums):
                for l1 in shell_dict[z1]:
                    for l2 in shell_dict[z2]:
                        interaction = (z1, z2, l1, l2)
                        needed_interactions.append(interaction)

        print(f"Total interactions needed: {len(needed_interactions)}")
        print("Sample needed interactions:")
        for interaction in needed_interactions[:10]:
            print(f"  {interaction}")

        # Check what we have in our SKFs
        print("\nAvailable SKF files and their atomic numbers:")
        for pair_key, optimizer in self.skf_optimizers.items():
            skf_dict = optimizer.skf_dict
            elements = skf_dict.get("atom_pair", [])
            atomic_numbers = []
            for elem in elements:
                for num, symbol in self.atomic_num_to_symbol.items():
                    if symbol == elem:
                        atomic_numbers.append(num)
                        break
            print(
                f"  {pair_key}: elements={elements}, atomic_nums={atomic_numbers}"
            )

        # Test creating feeds
        print("\nTesting feed creation...")
        try:
            updated_skfs = self.get_updated_skfs()
            h_feed = self._create_comprehensive_feed(
                updated_skfs, shell_dict, "H"
            )
            print("✓ Hamiltonian feed created successfully")

            print(f"H feed off_site_dict keys (first 10):")
            for i, key in enumerate(h_feed.off_site_dict.keys()):
                if i < 10:
                    print(f"  {key}")

        except Exception as e:
            print(f"✗ Error creating Hamiltonian feed: {e}")

        print("=" * 50)  #!/usr/bin/env python3

    def _extract_elements_from_geometry(self, geometry):
        """Extract unique elements from geometry object"""
        elements = set()
        try:
            if hasattr(geometry, "atomic_numbers"):
                atomic_nums = (
                    geometry.atomic_numbers.flatten().unique().tolist()
                )
            elif hasattr(geometry, "Z"):
                atomic_nums = geometry.Z.flatten().unique().tolist()
            else:
                print(
                    "Warning: Could not extract atomic numbers from geometry"
                )
                return elements

            for atomic_num in atomic_nums:
                if atomic_num in self.atomic_num_to_symbol:
                    elements.add(self.atomic_num_to_symbol[atomic_num])
                else:
                    print(f"Warning: Unknown atomic number {atomic_num}")

        except Exception as e:
            print(f"Error extracting elements from geometry: {e}")

        return elements

    def _detect_elements(self, vasprun_path, geometry, available_skf_pairs):
        """Enhanced element detection with better multi-element support"""
        elements = set()

        # Method 1: From geometry (highest priority for training system)
        if geometry is not None:
            geom_elements = self._extract_elements_from_geometry(geometry)
            elements.update(geom_elements)
            self.elements_in_system.update(geom_elements)
            print(f"Elements from geometry: {geom_elements}")

        # Method 2: From available SKF pairs
        if available_skf_pairs:
            for pair in available_skf_pairs:
                elements.update(pair)
            print(f"Elements from available_skf_pairs: {elements}")

        # Method 3: From vasprun.xml
        if vasprun_path and os.path.exists(vasprun_path):
            try:
                vasprun = Vasprun(vasprun_path)
                structure = vasprun.all_structures[-1]
                composition = structure.composition.to_dict()
                vasp_elements = set(composition.keys())
                elements.update(vasp_elements)
                self.elements_in_system.update(vasp_elements)
                print(f"Elements from vasprun.xml: {vasp_elements}")
            except Exception as e:
                print(f"Could not extract elements from vasprun.xml: {e}")

        # Method 4: Scan SKF directory
        if os.path.exists(self.skf_directory):
            try:
                skf_files = [
                    f
                    for f in os.listdir(self.skf_directory)
                    if f.endswith(".skf")
                ]
                scanned_elements = set()

                for skf_file in skf_files:
                    name_part = skf_file.replace(".skf", "")
                    if "-" in name_part:
                        elem1, elem2 = name_part.split("-", 1)
                        scanned_elements.update([elem1, elem2])

                elements.update(scanned_elements)
                print(f"Elements from SKF directory scan: {scanned_elements}")

            except Exception as e:
                print(f"Could not scan SKF directory: {e}")

        return elements

    def _initialize_skf_optimizers(self):
        """Initialize SKF optimizers for all element pairs - handles both orientations"""
        successful_pairs = []

        # Scan directory first to see what files actually exist
        available_files = {}
        if os.path.exists(self.skf_directory):
            for filename in os.listdir(self.skf_directory):
                if filename.endswith(".skf"):
                    name_part = filename.replace(".skf", "")
                    if "-" in name_part:
                        elem1, elem2 = name_part.split("-", 1)
                        available_files[f"{elem1}-{elem2}"] = filename

        print(f"Available SKF files: {list(available_files.keys())}")

        # Create optimizers for available files (not just element pairs)
        for pair_key, filename in available_files.items():
            skf_path = os.path.join(self.skf_directory, filename)
            try:
                print(f"Loading SKF optimizer for {pair_key} from {skf_path}")
                self.skf_optimizers[pair_key] = SkfParameterOptimizer(skf_path)
                successful_pairs.append(pair_key)
            except Exception as e:
                print(f"Failed to load {pair_key}: {e}")

        if not self.skf_optimizers:
            raise ValueError("No valid SKF files found")

        print(
            f"Successfully initialized {len(self.skf_optimizers)} SKF optimizers"
        )
        print(f"Available pairs: {successful_pairs}")

    def compute_multi_element_properties(
        self,
        geometry=None,
        shell_dict=None,
        kpoints=None,
        klines=None,
        phonons=False,
        get_fermi=False,
        get_energy=False,
        get_forces=False,
        get_bulk_mod=False,
        device=None,
        with_eigenvectors=False,
        pairwise_cutoff_length=7,
    ):
        """Compute DFTB properties for multi-element systems using ALL available optimizers"""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # Get all updated SKFs
        updated_skfs = self.get_updated_skfs()

        # Create comprehensive HS feeds that include ALL element pairs
        h_feed = self._create_comprehensive_feed(updated_skfs, shell_dict, "H")
        s_feed = self._create_comprehensive_feed(updated_skfs, shell_dict, "S")

        # Calculate total electron count for the system
        nelectron = self._calculate_system_electrons(geometry, updated_skfs)
        # print("nelectron2",nelectron)
        # TODO: Specify kpoint option
        # Setup k-lines for band structure
        # klines = self._get_default_klines()

        # Create calculator with comprehensive feeds
        if klines is not None:
            calc = SimpleDftb(
                geometry,
                shell_dict=shell_dict,
                klines=klines,
                h_feed=h_feed,
                s_feed=s_feed,
                nelectron=nelectron,
                device=device,
                with_eigenvectors=with_eigenvectors,
                pairwise_cutoff_length=pairwise_cutoff_length,
            )
        else:
            calc = SimpleDftb(
                geometry,
                shell_dict=shell_dict,
                kpoints=kpoints,
                h_feed=h_feed,
                s_feed=s_feed,
                nelectron=nelectron,
                device=device,
                with_eigenvectors=with_eigenvectors,
                pairwise_cutoff_length=pairwise_cutoff_length,
            )

        # Compute properties
        eigenvalues = calc()
        properties = calc.get_properties_dict(
            include_bulk_modulus=get_bulk_mod,
            include_dos_data=True,
        )
        # """
        if get_fermi:
            kT = 0.025
            H2E = 27.211
            kT_hartree = kT / H2E
            # print("nelectron1", nelectron)
            fermi_energy = fermi_search(
                # fermi_energy = fermi_search(
                eigenvalues=eigenvalues,
                n_electrons=nelectron,
                k_weights=calc.k_weights,
                # k_weights=self.k_weights,
            )
            print("fermi energy", fermi_energy)

            Ef_expanded = fermi_energy.view(-1, 1, 1)  # shape [batch, 1, 1]

            # Identify occupied/unoccupied bands
            occ = eigenvalues <= Ef_expanded
            unocc = eigenvalues > Ef_expanded

            # Replace invalid entries with extreme values
            vbm = torch.where(
                occ,
                eigenvalues,
                torch.tensor(
                    float("-inf"),
                    dtype=eigenvalues.dtype,
                    device=eigenvalues.device,
                ),
            )
            cbm = torch.where(
                unocc,
                eigenvalues,
                torch.tensor(
                    float("inf"),
                    dtype=eigenvalues.dtype,
                    device=eigenvalues.device,
                ),
            )

            # Max occupied and min unoccupied
            vbm_val = vbm.max(dim=-1)[0].max(dim=-1)[0]  # [batch]
            cbm_val = cbm.min(dim=-1)[0].min(dim=-1)[0]  # [batch]

            # Bandgap
            bandgap = (cbm_val - vbm_val).clamp(min=0.0)
            properties["efermi"] = Ef_expanded * H2E  # fermi_energy
            properties["bandgap"] = bandgap * H2E
            properties["calc"] = calc

        if phonons:
            print("Running phonons")
            freqs, ds = calc.calculate_phonon_modes()
            properties["ph_frequencies"] = freqs
            properties["ph_dos"] = ds
        if get_energy:
            total_energy = calc._calculate_electronic_energy()
            properties["total_energy"] = total_energy
        if get_forces:
            forces = calc._compute_forces_finite_diff()
            properties["forces"] = forces
        # """
        properties["eigenvalues"] = eigenvalues
        return properties, True

    def _create_comprehensive_feed(
        self, updated_skfs, shell_dict, integral_type
    ):
        """Create comprehensive feed that includes all element interactions with proper orientation handling"""
        interpolator = PolyInterpU

        # Initialize dictionaries
        hs_dict = {}
        onsite_hs_dict = {}

        # Track which atomic number pairs we have covered
        covered_interactions = set()

        # print(f"Creating {integral_type} feed for {len(updated_skfs)} SKF files...")

        # Process each SKF file
        for pair_key, skf in updated_skfs.items():
            # print(f"Processing {pair_key} for {integral_type} integrals...")

            # Get HS dict for this pair
            hs_dict = _get_hs_dict(hs_dict, interpolator, skf, integral_type)

            # Track the atomic numbers involved
            elements = skf.to_dict()["atom_pair"]
            if len(elements) >= 2:
                # Get atomic numbers
                atomic_nums = []
                for elem_symbol in elements:
                    for num, symbol in self.atomic_num_to_symbol.items():
                        if symbol == elem_symbol:
                            atomic_nums.append(num)
                            break

                if len(atomic_nums) == 2:
                    interaction = tuple(sorted(atomic_nums))
                    covered_interactions.add(interaction)
                    # print(f"  Added interaction: {interaction} ({elements})")

            # Get onsite dict for homo-nuclear pairs
            elements = skf.to_dict()["atom_pair"]
            if (
                len(elements) >= 2 and elements[0] == elements[1]
            ):  # Same element pair
                onsite_hs_dict = _get_onsite_dict(
                    onsite_hs_dict, skf, shell_dict, integral_type
                )
                # print(f"  Added onsite terms for {elements[0]}")

        # print(f"Covered interactions: {covered_interactions}")
        # print(f"Final HS dict keys: {len(hs_dict)}")
        # print(f"Final onsite dict keys: {list(onsite_hs_dict.keys()) if onsite_hs_dict else 'None'}")

        # Create the feed with all interactions
        feed = SkfFeed(hs_dict, onsite_hs_dict, shell_dict)
        return feed

    def _calculate_system_electrons(self, geometry, updated_skfs):
        """Calculate total number of electrons in the system"""
        total_electrons = 0.0

        # Extract atomic numbers from geometry
        if hasattr(geometry, "atomic_numbers"):
            atomic_nums = geometry.atomic_numbers.flatten()
        elif hasattr(geometry, "Z"):
            atomic_nums = geometry.Z.flatten()
        else:
            raise ValueError("Cannot extract atomic numbers from geometry")

        # Count electrons for each atom
        for atomic_num in atomic_nums:
            element_symbol = self.atomic_num_to_symbol.get(atomic_num.item())
            if element_symbol:
                electrons_for_atom = self._get_electrons_for_element(
                    element_symbol, updated_skfs
                )
                total_electrons += electrons_for_atom

        return torch.tensor([total_electrons])

    def _get_electrons_for_element(self, element_symbol, updated_skfs):
        """Get electron count for a specific element from SKF data"""
        # Look for homo-nuclear pair first
        pair_key = f"{element_symbol}-{element_symbol}"
        if pair_key in updated_skfs:
            skf_dict = updated_skfs[pair_key].to_dict()
            if "atomic_data" in skf_dict and skf_dict["atomic_data"]:
                occupations = skf_dict["atomic_data"]["occupations"]
                return sum(occupations)  # Factor of 2 for spin
                # return 2 * sum(occupations)  # Factor of 2 for spin

        # Fallback: look in any pair containing this element
        for pair_key, skf in updated_skfs.items():
            elements = skf.to_dict()["atom_pair"]
            if element_symbol in elements:
                atomic_data = skf.to_dict().get("atomic_data", {})
                if atomic_data:
                    occupations = atomic_data.get("occupations", [])
                    if occupations:
                        return 2 * sum(occupations)

        # Default fallback based on atomic number
        atomic_num = None
        for num, symbol in self.atomic_num_to_symbol.items():
            if symbol == element_symbol:
                atomic_num = num
                break

        if atomic_num:
            return float(atomic_num)  # Approximation

        return 4.0  # Conservative default

    def _get_default_klines(self):
        """Get default k-lines for band structure calculation"""
        return torch.tensor(
            [
                [0.0, 0.0, 0.0, -0.5, 0.5, 0.0, 10],
                [-0.5, 0.5, 0.0, -0.5, 0.5, -0.07654977, 10],
                [
                    -0.5,
                    0.5,
                    -0.07654977,
                    -0.28827489,
                    0.28827489,
                    -0.28827489,
                    10,
                ],
                [-0.28827489, 0.28827489, -0.28827489, 0.0, 0.0, 0.0, 10],
                [0.0, 0.0, 0.0, 0.5, 0.5, -0.5, 10],
                [0.5, 0.5, -0.5, 0.28827489, 0.71172511, -0.71172511, 10],
                [0.28827489, 0.71172511, -0.71172511, 0.0, 0.5, -0.5, 10],
                [0.0, 0.5, -0.5, -0.25, 0.75, -0.25, 10],
                [-0.25, 0.75, -0.25, 0.07654977, 0.92345023, -0.5, 10],
                [0.07654977, 0.92345023, -0.5, 0.5, 0.5, -0.5, 10],
                [0.5, 0.5, -0.5, -0.5, 0.5, 0.0, 10],
                [-0.5, 0.5, 0.0, -0.25, 0.75, -0.25, 10],
            ]
        )

    def get_updated_skfs(self):
        """Get updated SKF objects for all element pairs"""
        updated_skfs = {}
        for pair_key, optimizer in self.skf_optimizers.items():
            updated_skfs[pair_key] = optimizer.get_updated_skf()
        return updated_skfs

    def apply_constraints(self):
        """Apply constraints to all SKF optimizers"""
        for optimizer in self.skf_optimizers.values():
            optimizer.apply_constraints()

    def get_system_elements(self):
        """Get elements present in the training system"""
        return sorted(self.elements_in_system)

    def debug_feed_coverage(self, geometry):
        """Debug function to check what interactions are needed vs available"""
        print("\n" + "=" * 50)
        print("FEED COVERAGE DEBUG")
        print("=" * 50)

        # Extract atomic numbers from geometry
        if hasattr(geometry, "atomic_numbers"):
            atomic_nums = geometry.atomic_numbers.flatten().unique().tolist()
        elif hasattr(geometry, "Z"):
            atomic_nums = geometry.Z.flatten().unique().tolist()
        else:
            print("Cannot extract atomic numbers from geometry")
            return

        print(f"Atomic numbers in geometry: {atomic_nums}")

        # Get shell information
        shell_dict = generate_shell_dict_upto_Z65()

        # Check what interactions we need
        needed_interactions = []
        for i, z1 in enumerate(atomic_nums):
            for j, z2 in enumerate(atomic_nums):
                for l1 in shell_dict[z1]:
                    for l2 in shell_dict[z2]:
                        interaction = (z1, z2, l1, l2)
                        needed_interactions.append(interaction)

        print(f"Total interactions needed: {len(needed_interactions)}")
        print("Sample needed interactions:")
        for interaction in needed_interactions[:10]:
            print(f"  {interaction}")

        # Check what we have in our SKFs
        print("\nAvailable SKF files and their atomic numbers:")
        for pair_key, optimizer in self.skf_optimizers.items():
            skf_dict = optimizer.skf_dict
            elements = skf_dict.get("atom_pair", [])
            atomic_numbers = []
            for elem in elements:
                for num, symbol in self.atomic_num_to_symbol.items():
                    if symbol == elem:
                        atomic_numbers.append(num)
                        break
            print(
                f"  {pair_key}: elements={elements}, atomic_nums={atomic_numbers}"
            )

        # Test creating feeds
        print("\nTesting feed creation...")
        try:
            updated_skfs = self.get_updated_skfs()
            h_feed = self._create_comprehensive_feed(
                updated_skfs, shell_dict, "H"
            )
            print("✓ Hamiltonian feed created successfully")

            print(f"H feed off_site_dict keys (first 10):")
            for i, key in enumerate(h_feed.off_site_dict.keys()):
                if i < 10:
                    print(f"  {key}")

        except Exception as e:
            print(f"✗ Error creating Hamiltonian feed: {e}")

        print("=" * 50)

    def print_multi_element_summary(self):
        """Print detailed summary for multi-element system"""
        print("\n" + "=" * 70)
        print("MULTI-ELEMENT SKF PARAMETER OPTIMIZER SUMMARY")
        print("=" * 70)
        print(f"SKF Directory: {self.skf_directory}")
        print(f"Elements in system: {sorted(self.elements_in_system)}")
        print(f"All detected element pairs: {sorted(self.element_pairs)}")
        print(f"Available optimizers: {len(self.skf_optimizers)}")

        print("\nActual SKF Files Loaded:")
        for pair_key, optimizer in self.skf_optimizers.items():
            elements = pair_key.split("-")
            coverage = (
                "✓ SYSTEM"
                if all(e in self.elements_in_system for e in elements)
                else "  EXTRA"
            )
            h_params = len(optimizer.h_params)
            s_params = len(optimizer.s_params)
            total_params = sum(p.numel() for p in optimizer.parameters())
            print(
                f"  {coverage} {pair_key}: {h_params}H + {s_params}S = {total_params} params"
            )

        # Check for missing interactions
        print("\nInteraction Coverage Check:")
        system_elements = sorted(self.elements_in_system)
        for i, elem1 in enumerate(system_elements):
            for j, elem2 in enumerate(system_elements):
                if i <= j:  # Only check unique pairs
                    pair1 = f"{elem1}-{elem2}"
                    pair2 = f"{elem2}-{elem1}"

                    if pair1 in self.skf_optimizers:
                        print(f"  ✓ {pair1}: Available")
                    elif pair2 in self.skf_optimizers:
                        print(f"  ✓ {pair2}: Available (reverse)")
                    else:
                        print(f"  ✗ {elem1}-{elem2}: MISSING")

        total_trainable = sum(p.numel() for p in self.parameters())
        system_pairs = [
            k
            for k in self.skf_optimizers.keys()
            if all(e in self.elements_in_system for e in k.split("-"))
        ]

        print(f"\nTotal trainable parameters: {total_trainable}")
        print(f"System-relevant pairs: {len(system_pairs)}")
        print(
            f"Additional pairs: {len(self.skf_optimizers) - len(system_pairs)}"
        )
        print("=" * 70)


# Keep original SkfParameterOptimizer class
class SkfParameterOptimizer(nn.Module):
    """Trainable SKF parameters for fitting to DFT data with constraints"""

    def __init__(self, skf_path):
        super().__init__()

        # Load initial SKF parameters
        self.skf = Skf.from_skf(skf_path)
        self.skf_dict = self.skf.to_dict()

        # Store original parameters for reference
        self.original_h_params = {}
        self.original_s_params = {}

        # Make Hamiltonian and overlap parameters trainable
        h_param_dict = {}
        for key, value in self.skf_dict["hamiltonian"].items():
            original_tensor = torch.tensor(value).type(
                torch.get_default_dtype()
            )
            # original_tensor = torch.tensor(value, dtype=torch.float32)
            self.original_h_params[key] = original_tensor.clone()
            h_param_dict[key] = nn.Parameter(original_tensor)

        s_param_dict = {}
        for key, value in self.skf_dict["overlap"].items():
            original_tensor = torch.tensor(value).type(
                torch.get_default_dtype()
            )
            # original_tensor = torch.tensor(value, dtype=torch.float32)
            self.original_s_params[key] = original_tensor.clone()
            s_param_dict[key] = nn.Parameter(original_tensor)

        self.h_params = nn.ParameterDict(h_param_dict)
        self.s_params = nn.ParameterDict(s_param_dict)

        # Store other fixed parameters
        self.grid = self.skf_dict.get("grid", None)
        self.atomic_data = self.skf_dict.get("atomic_data", None)
        self.atom_pair = self.skf_dict.get("atom_pair", None)
        self.hs_cutoff = self.skf_dict.get("hs_cutoff", None)

    def apply_constraints(self, c=[0.9, 0.7, 0.95, 0.9]):
        """Apply physics-aware constraints"""
        # Create original params lazily if they don't exist (for ultra-compact loaded models)
        if not hasattr(self, "original_h_params"):
            self.original_h_params = {
                k: v.clone().detach() for k, v in self.h_params.items()
            }
        if not hasattr(self, "original_s_params"):
            self.original_s_params = {
                k: v.clone().detach() for k, v in self.s_params.items()
            }

        # These are quite arbitrary
        with torch.no_grad():
            for key, param in self.h_params.items():
                original = self.original_h_params[key]
                if key.split("-")[0] == key.split("-")[1]:  # Diagonal terms
                    param.data = torch.clamp(
                        param.data,
                        original * c[0],
                        original * (1 + (1 - c[0])),
                    )
                else:  # Off-diagonal terms
                    param.data = torch.clamp(
                        param.data,
                        original * c[1],
                        original * (1 + (1 - c[1])),
                    )

            for key, param in self.s_params.items():
                original = self.original_s_params[key]
                if key.split("-")[0] == key.split("-")[1]:  # Diagonal terms
                    param.data = torch.clamp(
                        param.data,
                        torch.maximum(original * c[2], torch.tensor(0.1)),
                        original * (1 + (1 - c[2])),
                    )
                else:  # Off-diagonal terms
                    param.data = torch.clamp(
                        param.data,
                        original * c[3],
                        original * (1 + (1 - c[3])),
                    )

    def get_updated_skf(self):
        """Create updated SKF with current parameters"""
        updated_dict = self.skf_dict.copy()
        updated_h = {key: param for key, param in self.h_params.items()}
        updated_s = {key: param for key, param in self.s_params.items()}
        updated_dict["hamiltonian"] = updated_h
        updated_dict["overlap"] = updated_s
        return Skf.from_dict(updated_dict)


class MultiVaspDataLoader:
    """Data loader for multiple VASP calculations"""

    def __init__(self, vasprun_paths):
        """
        Initialize with multiple vasprun.xml files

        Args:
            vasprun_paths: List of paths to vasprun.xml files or glob pattern
            geometry_paths: Optional list of geometry files (if different from vasprun)
        """
        # Handle glob patterns
        if isinstance(vasprun_paths, str):
            if "*" in vasprun_paths:
                self.vasprun_paths = sorted(glob.glob(vasprun_paths))
            else:
                self.vasprun_paths = [vasprun_paths]
        else:
            self.vasprun_paths = list(vasprun_paths)

        print(f"Found {len(self.vasprun_paths)} VASP calculations:")
        for i, path in enumerate(self.vasprun_paths):
            print(f"  {i+1:2d}. {path}")

        # Load and validate all data
        self.datasets = []
        self._load_all_datasets()

    def _load_all_datasets(self):
        """Load all VASP datasets and validate them"""
        successful_loads = 0

        for i, (vasp_path) in enumerate((self.vasprun_paths)):
            try:
                dataset = self._load_single_dataset(vasp_path, i)
                if dataset is not None:
                    self.datasets.append(dataset)
                    successful_loads += 1
            except Exception as e:
                print(f"❌ Failed to load {vasp_path}: {e}")

        if successful_loads == 0:
            raise ValueError("No valid datasets could be loaded!")

        print(
            f"✅ Successfully loaded {successful_loads}/{len(self.vasprun_paths)} datasets"
        )

    def _load_single_dataset(self, vasprun_path, index):
        """Load a single VASP dataset"""
        # Load VASP data
        vasprun = Vasprun(vasprun_path)

        # Extract geometry (try from VASP first)
        structure = vasprun.all_structures[-1]  # Final structure
        geometry = self._structure_to_geometry(structure)

        # Extract target properties
        target_energy = vasprun.final_energy
        target_bandgap = vasprun.get_indir_gap[0]
        target_dos = torch.tensor(vasprun.total_dos[1])  # spin up
        dos_energies = torch.tensor(vasprun.total_dos[0])

        # Get elements for this system
        composition = structure.composition.to_dict()
        elements = set(composition.keys())

        dataset = {
            "index": index,
            "vasprun_path": vasprun_path,
            "geometry": geometry,
            "target_energy": target_energy,
            "target_bandgap": target_bandgap,
            "target_dos": target_dos,
            "dos_energies": dos_energies,
            "elements": elements,
            "composition": composition,
            "metadata": {
                "natoms": structure.num_atoms,
                "formula": structure.composition.reduced_formula,
                "volume": structure.volume,
            },
        }
        print("_load_single_dataset", dataset)
        print(
            f"  ✓ Dataset {index}: {dataset['metadata']['formula']} "
            f"({dataset['metadata']['natoms']} atoms, {len(elements)} elements)"
        )

        return dataset

    def _structure_to_geometry(self, structure):
        """Convert structure to slakonet Geometry"""
        # Extract atomic numbers
        # print('structure)',structure)
        geometry = Geometry.from_ase_atoms([structure.ase_converter()])
        # print('structure2',structure)
        return geometry

    def get_all_elements(self):
        """Get all unique elements across all datasets"""
        all_elements = set()
        for dataset in self.datasets:
            all_elements.update(dataset["elements"])
        return sorted(all_elements)

    def get_batch(self, batch_size=None, shuffle=True):
        """Get a batch of datasets"""
        datasets = self.datasets.copy()
        if shuffle:
            random.shuffle(datasets)

        if batch_size is None:
            return datasets
        else:
            return datasets[:batch_size]

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        return self.datasets[idx]


def train_multi_vasp_skf_parameters(
    multi_element_optimizer,
    vasprun_paths,  # List of vasprun.xml files or glob pattern
    # geometry_paths=None,
    num_epochs=100,
    learning_rate=0.00001,
    batch_size=None,  # None = use all datasets each epoch
    plot_frequency=5,
    save_directory="multi_vasp_optimization_all",
    weight_by_system_size=True,
    early_stopping_patience=20,
    pairwise_cutoff_length=7,
):
    """
    Enhanced training function for multiple VASP datasets

    Args:
        multi_element_optimizer: The MultiElementSkfParameterOptimizer instance
        vasprun_paths: List of vasprun.xml paths or glob pattern like "tests/vasprun*.xml"
        geometry_paths: Optional separate geometry files
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        batch_size: Number of systems to use per epoch (None = use all)
        plot_frequency: How often to print progress
        save_directory: Directory to save results
        weight_by_system_size: Weight loss by number of atoms
        early_stopping_patience: Stop if no improvement for this many epochs
        pairwise_cutoff_length: Maximim length for pairwise interactions measured in ANGSTROMS
    """

    os.makedirs(save_directory, exist_ok=True)

    # Load multiple VASP datasets
    print("=" * 70)
    print("LOADING MULTIPLE VASP DATASETS")
    print("=" * 70)

    data_loader = MultiVaspDataLoader(vasprun_paths)

    if len(data_loader) == 0:
        raise ValueError("No valid datasets found!")

    # Check element coverage
    all_elements = data_loader.get_all_elements()
    optimizer_elements = multi_element_optimizer.get_system_elements()

    print(f"\nElement Coverage Analysis:")
    print(f"Elements in datasets: {all_elements}")
    print(f"Elements in optimizer: {optimizer_elements}")

    missing_elements = set(all_elements) - set(optimizer_elements)
    if missing_elements:
        print(f"⚠️  Missing SKF files for elements: {missing_elements}")
        print(
            "Consider adding SKF files for these elements or filtering datasets"
        )

    # Print detailed summary
    multi_element_optimizer.print_multi_element_summary()

    # Setup training
    shell_dict = generate_shell_dict_upto_Z65()
    kpoints = torch.tensor([5, 5, 5])
    # kpoints = torch.tensor([11, 11, 11])

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        multi_element_optimizer.parameters(), lr=learning_rate
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=10
    )

    print(f"\nStarting multi-VASP training:")
    print(f"  Datasets: {len(data_loader)}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size or 'all'}")
    print(
        f"  Total parameters: {sum(p.numel() for p in multi_element_optimizer.parameters())}"
    )

    # Training tracking
    best_loss = float("inf")
    epochs_without_improvement = 0
    loss_history = []
    dataset_losses = defaultdict(list)  # Track per-dataset performance

    klines = get_klines_example()
    for epoch in range(num_epochs):
        t1 = time.time()
        optimizer.zero_grad()

        # Apply constraints
        multi_element_optimizer.apply_constraints()

        # Get batch of datasets
        batch_datasets = data_loader.get_batch(
            batch_size=batch_size, shuffle=True
        )
        epoch_losses = []
        total_weight = 0.0

        # Process each dataset in the batch
        for dataset in batch_datasets:
            try:
                # Compute properties for this system
                print("geometry=", dataset["geometry"])
                properties, success = (
                    multi_element_optimizer.compute_multi_element_properties(
                        geometry=dataset["geometry"],
                        shell_dict=shell_dict,
                        klines=klines,
                        # dataset["geometry"], shell_dict, kpoints,
                        pairwise_cutoff_length=pairwise_cutoff_length,
                    )
                )

                if not success:
                    print(
                        f"⚠️  Failed to compute properties for dataset {dataset['index']}"
                    )
                    continue

                # Extract computed values
                # print("properties",properties,"\n")
                # print("dataset",dataset)
                target_bandgap = dataset["target_bandgap"]
                print("pred band_gap_eV", properties["band_gap_eV"])
                print("target band_gap_eV", target_bandgap)
                computed_dos = properties["dos_values_tensor"]
                target_dos = dataset["target_dos"].to(computed_dos.device)
                bandgap_weight = 1.0
                dos_weight = 0.0
                # Compute losses for this dataset
                mse_loss = torch.mean((computed_dos - target_dos) ** 2)
                mae_loss = torch.mean(torch.abs(computed_dos - target_dos))
                bandgap_mae_loss = torch.mean(
                    torch.abs(target_bandgap - properties["band_gap_eV"])
                )

                # Peak matching
                peak_mask = target_dos > target_dos.max() * 0.1
                if peak_mask.sum() > 0:
                    peak_loss = torch.mean(
                        ((computed_dos - target_dos) * peak_mask.float()) ** 2
                    )
                else:
                    peak_loss = torch.tensor(0.0, device=computed_dos.device)

                # Dataset-specific loss
                dataset_loss = mse_loss + 0.5 * mae_loss + 2.0 * peak_loss
                dataset_loss = bandgap_mae_loss
                # Weight by system size if requested
                if weight_by_system_size:
                    weight = dataset["metadata"]["natoms"]
                else:
                    weight = 1.0

                weighted_loss = dataset_loss * weight
                epoch_losses.append(weighted_loss)
                total_weight += weight

                # Track per-dataset performance
                dataset_losses[dataset["index"]].append(dataset_loss.item())

            except Exception as e:
                print(f"❌ Error processing dataset {dataset['index']}: {e}")
                continue

        if not epoch_losses:
            print(f"Epoch {epoch}: No valid computations, skipping...")
            continue
        t2 = time.time()

        # Combine losses across all datasets in batch
        batch_loss = (
            sum(epoch_losses) / total_weight
            if total_weight > 0
            else sum(epoch_losses) / len(epoch_losses)
        )

        # Add regularization across ALL optimizers
        total_h_reg = sum(
            sum(torch.sum(param**2) for param in opt.h_params.values())
            for opt in multi_element_optimizer.skf_optimizers.values()
        )
        total_s_reg = sum(
            sum(torch.sum(param**2) for param in opt.s_params.values())
            for opt in multi_element_optimizer.skf_optimizers.values()
        )
        regularization = 1e-10 * (total_h_reg + total_s_reg)

        # Final loss
        total_loss = batch_loss + regularization

        if torch.isnan(total_loss):
            print(f"Epoch {epoch}: NaN loss detected, skipping...")
            continue

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            multi_element_optimizer.parameters(), max_norm=2.0
        )
        optimizer.step()
        scheduler.step(total_loss)

        # Track progress
        loss_history.append(
            {
                "epoch": epoch,
                "total_loss": total_loss.item(),
                "batch_loss": batch_loss.item(),
                "regularization": regularization.item(),
                "datasets_used": len(batch_datasets),
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        # Check for improvement
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            epochs_without_improvement = 0

            # Save best model
            save_path = os.path.join(save_directory, "best_model")
            multi_element_optimizer.save_model(save_path, method="state_dict")
            # multi_element_optimizer.save_model(
            #    save_path, method="universal_params"
            # )
        else:
            epochs_without_improvement += 1

        # Print progress
        if epoch % plot_frequency == 0:
            avg_dataset_loss = (
                sum(sum(losses[-5:]) for losses in dataset_losses.values())
                / sum(len(losses[-5:]) for losses in dataset_losses.values())
                if dataset_losses
                else 0.0
            )
            tot_time = round(t2 - t1, 3)
            print(
                f"Epoch {epoch:3d}: Loss={total_loss.item():.6f}, "
                f"Batch={batch_loss.item():.6f}, AvgDataset={avg_dataset_loss:.6f}, "
                f"LR={optimizer.param_groups[0]['lr']:.6f}, Used={len(batch_datasets)} datasets"
                f" Time={tot_time} sec."
            )

        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"\nEarly stopping: No improvement for {early_stopping_patience} epochs"
            )
            break

    # Final save
    final_save_path = os.path.join(save_directory, "final_model")
    multi_element_optimizer.save_model(final_save_path, method="state_dict")
    multi_element_optimizer.save_model(final_save_path, method="compact")

    # Save training history
    history_file = os.path.join(save_directory, "training_history.json")
    with open(history_file, "w") as f:
        json.dump(loss_history, f, indent=2)

    # Print final statistics
    print(f"\n{'='*70}")
    print("MULTI-VASP TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Total epochs: {len(loss_history)}")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Datasets processed: {len(data_loader)}")
    print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    # Per-dataset performance summary
    print(f"\nPer-dataset performance (last 5 epochs avg):")
    for dataset_idx, losses in dataset_losses.items():
        if len(losses) >= 5:
            avg_loss = sum(losses[-5:]) / 5
            dataset = data_loader[dataset_idx]
            print(
                f"  Dataset {dataset_idx} ({dataset['metadata']['formula']}): {avg_loss:.6f}"
            )

    print(f"\nResults saved to: {save_directory}")
    print(f"{'='*70}")

    return multi_element_optimizer, loss_history, data_loader


def multi_vasp_training(
    vasprun_files=["tests/vasprun-1002.xml", "tests/vasprun-107.xml"],
    model="",
    num_epochs=2,
    batch_size=None,
    save_directory="slakonet_universal",
    pairwise_cutoff_length=7,
):
    """Example demonstrating training on multiple VASP calculations"""

    print("=" * 70)
    print("MULTI-VASP SKF PARAMETER OPTIMIZATION")
    print("=" * 70)

    # multi_optimizer = MultiElementSkfParameterOptimizer.load_model(
    #    "tests/slakonet_v1_sic"
    # )

    trained_optimizer, history, data_loader = train_multi_vasp_skf_parameters(
        multi_element_optimizer=model,
        vasprun_paths=vasprun_files,
        num_epochs=num_epochs,
        learning_rate=0.001,
        batch_size=batch_size,  # Use all datasets each epoch
        plot_frequency=5,
        save_directory=save_directory,
        weight_by_system_size=True,
        early_stopping_patience=20,
        pairwise_cutoff_length=pairwise_cutoff_length,
    )

    print("\n✅ Multi-VASP training completed successfully!")
    print(f"Trained on {len(data_loader)} VASP calculations")

    return trained_optimizer, history, data_loader


# Additional utility functions
def analyze_multi_vasp_performance(
    data_loader, trained_optimizer, save_directory, pairwise_cutoff_length=7
):
    """Analyze performance across different systems"""

    print("\n" + "=" * 50)
    print("MULTI-VASP PERFORMANCE ANALYSIS")
    print("=" * 50)

    shell_dict = generate_shell_dict_upto_Z65()
    kpoints = torch.tensor([5, 5, 5])
    # kpoints = torch.tensor([11, 11, 11])

    results = []

    for dataset in data_loader:
        try:
            # Compute properties
            properties, success = (
                trained_optimizer.compute_multi_element_properties(
                    dataset["geometry"], shell_dict, kpoints,
                    pairwise_cutoff_length=pairwise_cutoff_length,
                )
            )

            if success:
                computed_dos = properties["dos_values_tensor"]
                target_dos = dataset["target_dos"].to(computed_dos.device)

                # Compute metrics
                mse = torch.mean((computed_dos - target_dos) ** 2).item()
                mae = torch.mean(torch.abs(computed_dos - target_dos)).item()

                results.append(
                    {
                        "dataset_index": dataset["index"],
                        "formula": dataset["metadata"]["formula"],
                        "natoms": dataset["metadata"]["natoms"],
                        "elements": list(dataset["elements"]),
                        "mse": mse,
                        "mae": mae,
                        "success": True,
                    }
                )

                print(
                    f"✓ {dataset['metadata']['formula']:10s}: MSE={mse:.6f}, MAE={mae:.6f}"
                )
            else:
                results.append(
                    {
                        "dataset_index": dataset["index"],
                        "formula": dataset["metadata"]["formula"],
                        "success": False,
                    }
                )
                print(f"✗ {dataset['metadata']['formula']:10s}: Failed")

        except Exception as e:
            print(f"✗ Dataset {dataset['index']}: Error - {e}")

    # Save analysis
    analysis_file = os.path.join(save_directory, "performance_analysis.json")
    with open(analysis_file, "w") as f:
        json.dump(results, f, indent=2)

    # Summary statistics
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        avg_mse = sum(r["mse"] for r in successful_results) / len(
            successful_results
        )
        avg_mae = sum(r["mae"] for r in successful_results) / len(
            successful_results
        )

        print(f"\nSummary Statistics:")
        print(
            f"  Successful calculations: {len(successful_results)}/{len(results)}"
        )
        print(f"  Average MSE: {avg_mse:.6f}")
        print(f"  Average MAE: {avg_mae:.6f}")

    return results


def default_model(dir_path=None, model_name="slakonet_v0"):
    """
    More direct version - modify load function to accept BytesIO
    """
    if dir_path is None:
        dir_path = str(os.path.join(os.path.dirname(__file__), model_name))
    dir_path = os.path.abspath(dir_path)

    # Check for cached .pt file first (extracted from previous run)
    cached_model_file = os.path.join(dir_path, f"{model_name}.pt")
    if os.path.exists(cached_model_file):
        print(f"Loading cached model from {cached_model_file}")
        return MultiElementSkfParameterOptimizer.load_ultra_compact(
            cached_model_file
        )

    # Check if zip file already exists
    zip_file = os.path.join(dir_path, f"{model_name}.zip")
    if os.path.exists(zip_file):
        print(f"Found existing zip file: {zip_file}")
        # Load from existing zip file
        with zipfile.ZipFile(zip_file, "r") as zf:
            pt_files = [f for f in zf.namelist() if f.endswith(".pt")]
            if not pt_files:
                raise FileNotFoundError(f"No .pt file found in {zip_file}")

            # Load model data from zip
            with zf.open(pt_files[0]) as model_file:
                model_data = model_file.read()

            # Cache for future use
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            with open(cached_model_file, "wb") as cache_file:
                cache_file.write(model_data)

            # Load the model
            return MultiElementSkfParameterOptimizer.load_ultra_compact(
                cached_model_file
            )

    # If we get here, need to download
    url = "https://figshare.com/ndownloader/files/57945370"

    print(f"Downloading and loading {model_name} model from zip...")
    response = requests.get(url, stream=True)

    # Read zip data into memory
    zip_data = io.BytesIO()
    total_size = int(response.headers.get("content-length", 0))

    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
    for chunk in response.iter_content(chunk_size=1024):
        zip_data.write(chunk)
        progress_bar.update(len(chunk))
    progress_bar.close()

    zip_data.seek(0)  # Reset to beginning

    # Process zip from memory
    with zipfile.ZipFile(zip_data, "r") as zf:
        pt_files = [f for f in zf.namelist() if f.endswith(".pt")]
        if not pt_files:
            raise FileNotFoundError("No .pt file found in downloaded zip")

        # Load model data
        with zf.open(pt_files[0]) as model_file:
            model_data = model_file.read()

        # Cache for future use
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        cached_path = os.path.join(dir_path, f"{model_name}.pt")
        with open(cached_path, "wb") as cache_file:
            cache_file.write(model_data)

        # Load the model
        return MultiElementSkfParameterOptimizer.load_ultra_compact(
            cached_path
        )


# """
if __name__ == "__main__":
    # Run multi-VASP training example
    # trained_optimizer, loss_history, data_loader = (
    #    multi_vasp_training()
    # )

    # Analyze performance
    # performance_results = analyze_multi_vasp_performance(
    #    data_loader, trained_optimizer, "multi_vasp_results"
    # )
    multi_vasp_training()

# """
