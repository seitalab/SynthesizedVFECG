# SynthesizedVFECG


## FEDformer

`git clone https://github.com/MAZiqing/FEDformer`

Move to `repo` directory.

Need to fix `layers/AutoCorrelation.py`.

`.cuda()` to `.to(values.device)`.