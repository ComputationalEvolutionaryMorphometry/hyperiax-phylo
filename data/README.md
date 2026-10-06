# Data

This directory contains the Procrustes aligned landmark data file (data.h5), original phylogenetic tree topology file (tree.nwk), and drawn phylogenetic tree figures for both butterfly wings and bird beaks. The data can be reproduced by running the proprocessing scripts.

The file structure follows:

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
