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

The butterfly data is further described in Michael Lind Severinsen. Stochastic Morphometry: Applying stochastic processes to model infinite dimensional shape evolution. PhD thesis, University of Copenhagen, 2026.

The beak landmark are from the Mark My Bird citizen science project first described in Cooney et al. (2017) and the phylogenetic tree is described in Stiller et al. (2024). We are grateful to Gavin Thomas (University of Sheffield) for access to the bird beak landmark data from Mark My Bird.
