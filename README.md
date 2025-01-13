# Single-cell perturbation effects prediciton benchmark
## Introduction
Various computational methods have been developed to predict perturbation effects, while despite claims of promising performance, great concerns on the true efficacy of these models have been raised, particularly when evaluated across diverse unseen cellular contexts and unseen perturbations. To this end, a comprehensive benchmark of 21 single cell perturbation response prediction methods including genetic and chemical perturbations, with 29 datasets for the method generalizations in unseen cellular contexts and unseen perturbations, using different evaluation metrics, were conducted. Tips regarding the method limitations, method generalizations and method selections were presented. Finally, an applicable solution by leveraging prior knowledge through cellular context embedding to achieve improved model generalization in new cellular context is presented. 

## Workflow
![Workflow](imgs/fig1_v3.png)


## Cellular transfer scenario
In the cellular context transfer scenario, we evaluate the prediction of known perturbations in previously unobserved cellular contexts. Specifically, we assessed the accuracy of 10 published methods and the trainMean baseline model across 12 datasets using four evaluation metrics: MSE, PCC-delta, E-distance, and common DEGs. The cellular transfer scenario can be further divided into two distinct test settings based on the partitioning of the training and test datasets: i.i.d (independent and identically distributed or in-distribution) and o.o.d (out-of-distribution) setting. [i.i.d](https://github.com/bm2-lab/scPerturBench/tree/main/cellular_transfer/i.i.d) contained the script used in the i.i.d setting. [o.o.d](https://github.com/bm2-lab/scPerturBench/tree/main/cellular_transfer/o.o.d) contained the script used in the o.o.d setting. [calPerformance](https://github.com/bm2-lab/scPerturBench/tree/main/cellular_transfer/calPerformance_delta.py) and [Utils](https://github.com/bm2-lab/scPerturBench/tree/main/cellular_transfer/myUtil.py) is the script for performance calculation and generic function。


## Perturbation transfer scenario
In the perturbation transfer scenario, we assess the ability of models to predict the effects of previously unobserved perturbations within a specific cellular context. Depending on the type of perturbation, this scenario can be further divided into two categories: genetic perturbation effects prediction and chemical perturbation effects prediction. (1) Genetic perturbation effect prediction. (2) Chemical perturbation effect prediction. [Genetic perturbation effect prediction](https://github.com/bm2-lab/scPerturBench/tree/main/perturbation_transfer/genetic) contained the script used in the genetic setting. [Chemical perturbation effect prediction](https://github.com/bm2-lab/scPerturBench/tree/main/perturbation_transfer/chemical) contained the script used in the chemical setting. [calPerformance](https://github.com/bm2-lab/scPerturBench/tree/main/perturbation_transfer/calPerformance.py) and [Utils](https://github.com/bm2-lab/scPerturBench/tree/main/perturbation_transfer/myUtil1.py) is the script for performance calculation and generic function。

## Benchmark datasets summary
All datasets analyzed in our study are listed in the [Workflow](imgs/fig1_v3.png). The processed datasets are available in a public Figshare repostiory [link](https://figshare.com/articles/dataset/SCMMIB_Register_Report_Stage_2_processed_datasets/27161451/1).

## Benchmark Methods
All benchmark methods analyzed in SCMMIB study are listed below. Details of these methods were available in our Register Report Stage 1 manuscript in [figshare folder](https://springernature.figshare.com/articles/journal_contribution/Benchmarking_single-cell_multi-modal_data_integrations/26789572).

| Method                                                                         | Article                                                                   | Time |
|--------------------------------------------------------------------------------|---------------------------------------------------------------------------|------|
| [bioLord](https://github.com/nitzanlab/biolord)                               | [Nature Biotechnology](https://www.nature.com/articles/s41587-023-02079-x)                        | 2024 |
| [CellOT](https://github.com/satijalab/seurat)                            | [Cell](https://doi.org/10.1016/j.cell.2019.05.031)                        | 2019 |
| [inVAE](https://github.com/satijalab/seurat)                           | [Cell](https://doi.org/10.1016/j.cell.2019.05.031)                        | 2019 |
| [scDisInFact](https://github.com/SydneyBioX/CiteFuse)                             | [Bioinformatics](https://doi.org/10.1093/bioinformatics/btaa282)          | 2020 |
| [cGen](https://github.com/gtca/mofaplus-shiny)                                | [Genome Biology](https://doi.org/10.1186/s13059-020-02015-1)              | 2020 |
| [scPRAM](https://github.com/sqjin/scAI)                                          | [Genome Biology](https://doi.org/10.1186/s13059-020-1932-8)               | 2020 |
| [scPreGAN](https://github.com/caokai1073/UnionCom)                             | [Bioinformatics](https://doi.org/10.1093/bioinformatics/btaa443)          | 2020 |
| [SCREEN](https://github.com/boyinggong/cobolt/blob/master/docs/tutorial.ipynb) | [Genome Biology](https://doi.org/10.1186/s13059-021-02556-z)              | 2021 |
| [scVIDR](https://github.com/cmzuo11/DCCA)                                        | [Bioinformatics](https://doi.org/10.1093/bioinformatics/btab403)          | 2021 |
| [trVAE](https://github.com/gao-lab/GLUE)                                        | [Nature Biotechnology](https://doi.org/10.1101/2021.08.22.457275)         | 2021 |
| [AttentionPert](https://github.com/welch-lab/liger)                        | [Nature Biotechnology](https://doi.org/10.1038/s41587-021-00867-x)        | 2021 |
| [CPA]( https://github.com/kimmo1019/scDEC)                                   | [Nature machine intelligence](https://doi.org/10.1038/s42256-021-00333-y) | 2021 |
| [GEARS](https://github.com/kodaim1115/scMM)                                     | [Cell Reports Methods](https://doi.org/10.1016/j.crmeth.2021.100071)      | 2021 |
| [GenePert](https://github.com/cmzuo11/scMVAE)                                    | [Briefings in Bioinformatics](https://doi.org/10.1093/bib/bbaa287)        | 2021 |
| [linearModel](https://github.com/satijalab/seurat)                            | [Cell](https://doi.org/10.1016/j.cell.2021.04.048)                        | 2021 |
| [scGPT](https://scvi-tools.org/)                                             | [Nature Methods](https://doi.org/10.1038/s41592-020-01050-x)              | 2021 |
| [scFoundation](https://github.com/Teichlab/MultiMAP)                               | [Genome Biology](https://doi.org/10.1186/s13059-021-02565-y)              | 2021 |
| [chemCPA](https://github.com/KChen-lab/bindSC)                                  | [Genome Biology](https://doi.org/10.1186/s13059-022-02679-x)              | 2022 |
| [scouter](https://github.com/welch-lab/liger)                              | [Nature Communications](https://doi.org/10.1038/s41467-022-28431-4)       | 2022 |



