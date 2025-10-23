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

To run the server:
```
python -m vllm.entrypoints.openai.api_server     --model meta-llama/Llama-3.2-11B-Vision --port 8000 --gpu-memory-utilization 0.45
--max_num_seqs 16 --max-model-len 4000 --chat-template llama3.2_json.jinja --limit-mm-per-prompt '{"image":2}'
```
To run Qwen:
```
python -m vllm.entrypoints.openai.api_server     --model Qwen/Qwen2.5-VL-7B-Instruct     --host 0.0.0.0     --port 8000
```
