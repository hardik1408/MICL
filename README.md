# MICL

This repository contains all code and dataset for the MICL experiments.

### Experiment 1: Artistic style transfer
To run the experiment and generate the descriptions:
```
python -m gemma.artist --dataset gemma/results/keeffe_descriptions.json --k_shots 1   
```

To run evaluation, use `eval.py`
```
python eval.py --dataset generated/top_1_keeffe_descriptions_gemma.json    
```