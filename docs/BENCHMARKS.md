# Benchmark Notes

All tables and charts linked from the README are derived from evaluation pickles already
stored under `files/data/`. Regeneration:

```bash
python scripts/generate_docs_assets.py --figures-only
```

## Primary configuration

- Machines: 2
- Recipes: 2
- Buffer size: 3
- Episodes: 1000
- Artifacts:
  - `dqn_data_seco_2m_2r_3b_0.6g_1000.pkl`
  - `edd_data_seco_2m_2r_3b_1000.pkl`
  - `fifo_data_seco_2m_2r_3b_1000.pkl`
  - `heuristic_data_seco_2m_2r_3b_1000.pkl`
  - `a2c_data_seco_2m_2r_3b_seco_recipes_1000.pkl`

## Interpretation guidance

- **Late rate** = mean over episodes of `JNOT / (JOT + JNOT)`.
- High on-time counts with high late counts (all-purpose setting) indicate a throughput–tardiness trade-off, not a single scalar “best” agent.
- A2C can show low late rate while completing very few jobs; always read tardiness together with throughput.
