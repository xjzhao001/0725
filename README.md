
```bash
conda create -n pn-gnn python=3.10
conda install -c conda-forge -c pytorch -c pyg
conda install ninja
conda install ogb easydict pyyaml -c conda-forge
```

Run the following commands on transductive real datasets.

```bash
python script/run.py -c config/transductive fb15k237.yaml --gpus [0]
```

Run the following commands on inductive real datasets.

```bash
python script/run.py -c config/inductive fb15k237.yaml --gpus [0] --v1
```
