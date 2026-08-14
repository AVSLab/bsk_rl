# Research Focus I results (pre-run status)

No final numerical comparison is reported yet. The six requested 48-hour exploratory
training jobs and paired Monte Carlo campaign have not been run on Alpine, so inserting
performance values here would be scientifically unsupported.

After the cluster campaign, run:

```bash
python examples/prospectus_rfi/analyze.py \
  --input-root /scratch/alpine/$USER/prospectus_rfi
```

That command writes the completed file to
`/scratch/alpine/$USER/prospectus_rfi/prospectus_results.md`, including numerical paired
differences, 95% bootstrap intervals, Holm-adjusted tests, practical-equivalence language,
and paths to every PDF figure. Until three independent training seeds are available for a
configuration, the generated text explicitly labels all architecture conclusions
exploratory.

