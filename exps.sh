python main.py --config-name text-train/act_train_llama_8b_lora.yaml
python main.py --config-name text-train/react_train_llama_8b_lora.yaml

python main.py --config-name text-evals/act_eval_llama8b_zero_shot.yaml
python main.py --config-name text-evals/act_eval_llama8b_lora.yaml
python main.py --config-name text-evals/act_eval_llama8b_few_shot.yaml

python main.py --config-name text-evals/react_eval_llama8b_zero_shot.yaml
python main.py --config-name text-evals/react_eval_llama8b_lora.yaml
python main.py --config-name text-evals/react_eval_llama8b_few_shot.yaml

python main.py --config-name text-evals/act_eval_llama70B_few_shot.yaml
python main.py --config-name text-evals/react_eval_llama70b_few_shot.yaml

python main.py --config-name text-evals/act_eval_gpt4o_mini_few_shot.yaml
python main.py --config-name text-evals/react_eval_gpt4o_mini_few_shot.yaml
