# hs_index

Reusable function wrapping the proven `hhsa_sohail_n34` M0–M5 experiment for any monthly one-dimensional index.

```python
from src import run_hhsa_sohail
run_hhsa_sohail(time, index, "my_index", "outputs/my_index", forecast_end="2028-12-01")
```

CSV command:

```bash
python src/hs_index.py --input data/pmm_monthly.csv --time-column date --columns pmm_sst pmm_wind --output outputs/pmm --forecast-end 2028-12-01
```

The experiment retains M0 Persistence, M1 Direct NN, M2 HHSA single ResNet, M3 Event/Strength component NNs, M4 physical `A*cos(phase)` reconstruction, and M5 fusion ResNet. Outputs contain models, CSV, PNG, leakage audit, and CPU/memory records. Full-record decomposition is offline/diagnostic, not causal real-time validation.

Reference: Sohail, Zika, and Ehmen, *How accurate are salinity measurements around Antarctica? A machine learning based approach*, DOI [10.1088/3049-4753/ae7113](https://doi.org/10.1088/3049-4753/ae7113). Sohail supplies the NN inspiration; HHSA and M0–M5 are this project's design.
