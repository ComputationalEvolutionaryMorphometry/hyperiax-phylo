# Data

The private shape datasets are not included in this repository.

Place local copies in this layout before running the pipeline:

```text
data/butterflies/
  lmks.csv
  tree.nwk
  data.h5

data/beaks/
  lmks.csv
  tree.nwk
  data.h5
```

`data.h5` can be rebuilt from `lmks.csv` and `tree.nwk` with
`scripts.build_data`.
