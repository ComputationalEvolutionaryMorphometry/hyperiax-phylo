# Data

This directory contains the Procrustes aligned landmark data file (lmks.csv), original phylogenetic tree topology file (tree.nwk). The data can be reproduced by running the proprocessing scripts.

The file structure follows:

```text
data/butterflies/
  lmks.csv
  tree.nwk

data/beaks/
  lmks.csv
  tree.nwk
```

`data.h5` can be rebuilt from `lmks.csv` and `tree.nwk` with
`scripts.build_data`.
