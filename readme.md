bins should contain the executables for:
 - raxml-ng https://codeberg.org/amkozlov/raxml-ng/releases/tag/2.0.3
 - codeml https://github.com/abacus-gene/paml/releases/download/v4.10.10/paml-4.10.10-linux-x86_64.tar.gz

 data/raw/ folder should contain the raw data files, including:
 - orthomam trees (trees/) (nucletide trees for each gene unzipped from orthomam)
 - orthomam msas (msas/) (unfiltered amino acid alignments for each gene zipped from orthomam)


 run all scripts from data/, e.g. cd data && python ../benchmarks/...”.