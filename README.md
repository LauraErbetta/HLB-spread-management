# HLB Model Package

A Cython-based model for HLB epidemic dynamics, packaged with an example setup for running single-region simulations.

---

## Folder Structure

```text
.
├── Additional_data/    # Folder containing default estimated proportions at different prevalences for survey 
├── hlb_model/          # Core package (Python scripts, Cython modules, Data)
├── example_run.py      # Example script to execute a simulation
├── requirements.txt    # Required Python packages
├── pyproject.toml      # Build system specification
├── setup.py            # Cython compilation setup
└── README.md           # Documentation
```

---

## 1. Prerequisites

A C/C++ compiler is required to compile the Cython extensions during installation.

---

## 2. Installation

1. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv

   # On Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies and the `hlb_model` package**
   ```bash
   pip install -r requirements.txt      # dependencies
   pip install -e .                     # hlb model
   ```
   
---

## 3. Running example_run.py

Paramters can be changed manually. If changed in the cython files, the package may need to be recompiled.
The script allows the creation of maps for one simulation.

---
