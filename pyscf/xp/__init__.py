# pyscf-forge/pyscf/xp/__init__.py
"""
XP module for pyscf-forge
"""

import importlib.util
import sys
from pathlib import Path

import pyscf.data as _pyscf_data

def _load_local_module(module_name, relative_path):
	module_path = Path(__file__).resolve().parents[1] / relative_path
	spec = importlib.util.spec_from_file_location(module_name, module_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Cannot load {module_name} from {module_path}")

	module = importlib.util.module_from_spec(spec)
	sys.modules[module_name] = module
	spec.loader.exec_module(module)
	return module

_pyscf_data.radii = _load_local_module('pyscf.data.radii', 'data/radii.py') # Temp solution
from .xppcm import *
