# LOGO Analysis

This repository provides a Python script to perform **Leave-One-Group-Out (LOGO) analysis** on combinatorial **catalyst/substrate MLR models**.  
It interfaces directly with the general MLR functions in `mlr_utils.py` from the Sigman Lab’s [python-modeling](https://github.com/SigmanGroup/python-modeling) environment.

---

## 📦 Environment

The scripts are compatible with the standard **Sigman Lab python-modeling environment**.  
If you encounter dependency issues, a clean conda environment can be created from the included `logo_env.yml`:

```bash
conda env create -f logo_env.yml
conda activate logo
```

---

## ▶️ Usage

Run locally:

```bash
python logo.py input.xlsx > logo.out &
```

Run on the **University of Utah CHPC (Notchpeak)** cluster with the included submission wrapper:

```bash
submit_cli.sh python logo.py input.xlsx
```

Numerically unstable values can cause issues: Make sure too high feature values are filtered out first (>10^10)

---

## ⚡ Parallelization

As the last step of the bidirectional stepwise regression of mlr_utils.py is not parallelized and takes the majority of the time in model training, LOGO analysis of big datasaets and big models can take very long! This script parallelizes the model training of the independent LOGO splits and therefore reduces the computation time significantly. On a 32 cores node the parallel LOGO analysis was 8 times faster than a serial LOGO analysis (using all cores for MLR training of a single LOGO split). The script uses **two levels of parallelism**:

1. **Outer parallelization** → distributes **LOGO groups** (leave-one-group-out folds) across workers.  
   - Controlled by the number of CPUs you request in SLURM (`--cpus-per-task`) or by available cores on a local machine (`total_cores` variable)
   - Each group runs independently and writes its own log file (`<LeftOutGroup>.log`).

2. **Inner parallelization** → inside each LOGO worker, the **bidirectional stepwise MLR search** is parallelized across candidate models.  
   - This passes the `inner_threads` input to the `n_processors` variable inside `mlr_utils.py`.
   - By default, the script automatically uses at least 1/8 of cores per inner job and minimum 2. 

### Adjusting cores manually

In `logo.py`, the split between outer and inner threads is defined like this:

```python
total_cores = multiprocessing.cpu_count()

# Inner: number of threads per LOGO worker (minimum 2 and 1/8 of the total cores)
inner_threads = max(2, total_cores // 8)

# Outer: number of LOGO workers
n_logo_workers = max(1, total_cores // inner_threads)
```

- Increase `n_logo_workers` if you want to run more groups in parallel.
- Decrease it if you want to give each LOGO worker more cores for the inner MLR search.
- On the cluster, `total_cores` will automatically respect your SLURM allocation (`SLURM_CPUS_PER_TASK`).

---

## 📊 Output

For each left-out group, the script generates:

- A `.log` file with detailed training/testing results and model parameters
- A `.png` parity plot (predicted vs. measured)
- A row in the final `LOGO_results.csv` summary file

The final CSV is sorted by **LOGO MAE** for quick comparison.

---

