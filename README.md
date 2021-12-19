# Reproducible Experiment

To reproduce the evaluation presented in the CS592 paper:

First, login to the cuda server `cuda.cs.purdue.edu`, make sure you have access to this server

```bash
ssh username@cuda.cs.purdue.edu
```
Then, clone the code from github, and enter the repo directory

```bash
git clone git@github.com:Kuigesi/PipelineExecution-Reproducible.git
cd PipelineExecution-Reproducible
```

To conduct the Pipeline Execution evaluation, run

```bash
bash ./runtest.sh
```
This will produce the following files:
- `./benchmark/data/benchmark.csv`, which is the collected results of the running time of different parallel settigs.


- 2 figures `./benchmark/pictures/pipelineparallelruntime.pdf`, `./benchmark/pictures/pipelineparallelspeedup.pdf` will be plotted to illustrate the runtime and speed up of different parallel seetings.

To check out the generated figures, the pdf file should be transfered to your local computer.