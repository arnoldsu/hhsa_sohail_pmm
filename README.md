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


## PMM end-to-end test

The standard function was tested using both columns in `data/pmm_monthly.csv`:

- `pmm_sst`, used as the PMM SST/ts index;
- `pmm_wind`, used as the PMM zonal-wind/uas index.

Both records end in February 2026. The function generated 34 monthly forecasts from March 2026 through December 2028. Each future-prediction figure contains M0 Persistence, M1 Direct NN, M2 HHSA single ResNet, M4 Dual physical reconstruction, and M5 Dual fusion ResNet.

M3 is not drawn as a separate index forecast because it consists of the intermediate Event NN and individual AM-IMF Strength NNs. Its outputs are recombined by M4 and supplied to M5.

### December 2028 predictions

| Method | PMM SST | PMM UAS |
|---|---:|---:|
| M0 Persistence | 1.920 | -3.820 |
| M1 Direct NN | 2.035 | 0.810 |
| M2 HHSA single ResNet | -0.490 | -6.213 |
| M4 Dual physical reconstruction | 2.091 | -13.849 |
| M5 Dual fusion ResNet | 2.037 | -6.633 |

Generated figures:

- `outputs/pmm/pmm_sst/figures/03_future_forecast_all_methods.png`
- `outputs/pmm/pmm_wind/figures/03_future_forecast_all_methods.png`

Numerical forecasts:

- `outputs/pmm/pmm_sst/results/future_forecast_all_methods.csv`
- `outputs/pmm/pmm_wind/results/future_forecast_all_methods.csv`

These future trajectories cannot determine which method is correct before observations become available. They are experimental projections based on full-record, offline decomposition and are not evidence of operational forecast skill.

Reference: Sohail, Zika, and Ehmen, *How accurate are salinity measurements around Antarctica? A machine learning based approach*, DOI [10.1088/3049-4753/ae7113](https://doi.org/10.1088/3049-4753/ae7113). Sohail supplies the NN inspiration; HHSA and M0–M5 are this project's design.
