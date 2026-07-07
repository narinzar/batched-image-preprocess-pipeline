# data/

This directory holds the images the pipeline reads. It is gitignored except for
this file.

Populate it with synthetic images (no license concerns) by running:

```
python scripts/00_make_sample_images.py --count 4000 --size 256
```

That writes PNGs to `data/images/`. The benchmark and autotune scripts read from
`data/images/` by default. You can also point `--images-dir` at any directory of
`*.png` files of your own.
