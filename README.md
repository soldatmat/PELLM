# Protein Engineering with Large Language Models
This repository contains source code developed as part of my diploma thesis available at https://dspace.cvut.cz/handle/10467/115759.

- The three proposed MLDE methods are implemented in `code/de`.

- The two tradintional DE benchmarks are implemented in `code/single_mutation_walk` and `code/recombine_mutation`.

- Folder `code/llm` contains code from initial exploratory experiments, which is not essential to any of the methods.

- Folder `code/plot` and `code/dimred` contain code used to generate included graphics.

## Acknowledgements
All of the proposed MLDE methods require a pre-trained protein language model as an embedding extractor.<br/>
If you use the ESM-1b model, cite the original paper.

- **ESM-1b:** Alexander Rives et al. “Biological structure and function emerge from scaling unsu-
pervised learning to 250 million protein sequences”. In: Proceedings of the National
Academy of Sciences 118.15 (2021), e2016239118.

The used datasets are included in `data`. If you use them, don't forget to cite the original papers.<br/>
Correct citations are included with each dataset in a `CITE_AS.txt` file with corresponding BibTeX template.

- **GB1:** Nicholas C Wu et al. “Adaptation in protein fitness landscapes is facilitated by
indirect paths”. In: Elife 5 (2016), e16965.

- **PhoQ:** Anna I Podgornaia and Michael T Laub. “Pervasive degeneracy and epistasis in a
protein-protein interface”. In: Science 347.6222 (2015), pp. 673–677.
