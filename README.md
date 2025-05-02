# Protein Engineering with Large Language Models
This repository contains source code for the MLDE method **Bayesian Optimization in Embedding Space (BOES)**.<br>
See the accompanying paper https://ieeexplore.ieee.org/abstract/document/10822356.

Additionally, this repository also includes source code for two other MLDE methods developed in the previous work: https://dspace.cvut.cz/handle/10467/115759.

- The BOES MLDE method is implemented in `code/de/boes`.

- The two other MLDE methods proposed in the previous work are implemented in `code/de/nsde` and `code/de/perceptron`.

- The two traditional DE benchmarks are implemented in `code/single_mutation_walk` and `code/recombine_mutation`.

- Folder `code/llm` contains code from initial exploratory experiments, which is not essential to any of the methods.

- Folder `code/plot` and `code/dimred` contain code used to generate the included graphics.

## Installation
To start using the proposed MLDE methods implemented in `code/de`, please follow these steps:
1) Install Julia.
2) Install Conda.
3) In `code/de/setup_pycall.jl` set `CONDA_PATH` to the path to the Conda root folder.
4) In bash, run:
```
cd code/de
source setup.sh
```

## Acknowledgements
All of the proposed MLDE methods require a pre-trained protein language model as an embedding extractor.<br/>
If you use the ESM-1b model, cite the original paper.

- **ESM-1b:** Rives, Alexander, et al. "Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences." Proceedings of the National Academy of Sciences 118.15 (2021): e2016239118.

The used datasets are included in `data`. If you use them, don't forget to cite the original papers.<br/>
Correct citations are included with each dataset in a `CITE_AS.txt` file with a corresponding BibTeX template.

- **GB1:** Wu, Nicholas C., et al. "Adaptation in protein fitness landscapes is facilitated by indirect paths." Elife 5 (2016): e16965.

- **PhoQ:** Podgornaia, Anna I., and Michael T. Laub. "Pervasive degeneracy and epistasis in a protein-protein interface." Science 347.6222 (2015): 673-677.

- **TrpB:** Johnston, Kadina E., et al. "A combinatorially complete epistatic fitness landscape in an enzyme active site." Proceedings of the National Academy of Sciences 121.32 (2024): e2400439121.
