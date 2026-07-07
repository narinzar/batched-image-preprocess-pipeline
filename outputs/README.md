# outputs/

Generated artifacts land here. This directory is gitignored except for this
file.

- `throughput.json` - the worker-count sweep table from
  `scripts/01_bench_workers.py`.
- `throughput.png` - images/sec vs worker count plot from the same script.

Run the benchmark to populate these:

```
python scripts/01_bench_workers.py --images-dir data/images
```
