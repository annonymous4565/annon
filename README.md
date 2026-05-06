# Dependency-Aware Discrete Diffusion for Scene Graph Generation

**Under Review**

### Environment

```javascript
conda env create -f environment.yml
```
Or

```javascript
pip install -r requirements.txt
```

### Datasets
Please download datasets [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html), [COCO-Stuff](https://github.com/nightrome/cocostuff), and [LAION-SG](https://github.com/mengcye/LAION-SG) and place them in folder [data](./data).

### Training Script

```javascript
bash ./scripts/exp_discretesg_run_all.sh
```

### Evaluation Script

```javascript
bash ./scripts/exp_master_eval_metrics_run_all.sh
```

### Image Generators

We use [SDXL-SG](https://github.com/mengcye/LAION-SG) and [GLIGEN](https://github.com/gligen/GLIGEN) for SG-to-Image and Layout-to-Image generation respectively.
