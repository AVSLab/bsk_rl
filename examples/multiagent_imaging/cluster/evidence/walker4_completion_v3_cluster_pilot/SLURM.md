# Walker-4 Slurm accounting

Accounting was captured with `sacct -X -P --units=G` after both jobs completed.

| Job/step | State | Elapsed | MaxRSS | AllocCPUS | Requested memory | Node |
|---|---|---:|---:|---:|---:|---|
| 32528119 | COMPLETED | 00:46:42 | — | 9 | 32 GiB | c3cpu-e2-u2 |
| 32528119.batch | COMPLETED | 00:46:42 | 5.46 GiB | 9 | — | c3cpu-e2-u2 |
| 32530057 | COMPLETED | 03:59:44 | — | 9 | 32 GiB | c3cpu-e2-u15 |
| 32530057.batch | COMPLETED | 03:59:44 | 12.36 GiB | 9 | — | c3cpu-e2-u15 |

The Slurm script requested one node and eight CPUs. The partition accounted nine
allocated CPUs for each job, so nine is the resource value used in the report.
The learner was CPU-only and each Torch/BLAS process used one thread.
