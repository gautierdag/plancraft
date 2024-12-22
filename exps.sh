# generate all results from paper

# Oracle benchmark
uv run main.py --config-name evals/oracle.yaml ++evals.plancraft.valid_actions='[move,smelt,impossible]'

# Text only - GPT4o-mini
uv run main.py --config-name evals/gpt4o_mini.yaml ++evals.plancraft.valid_actions='[move,smelt]'
uv run main.py --config-name evals/gpt4o_mini.yaml ++evals.plancraft.valid_actions='[move,smelt,think]'
uv run main.py --config-name evals/gpt4o_mini.yaml ++evals.plancraft.valid_actions='[move,smelt,think,search]'
uv run main.py --config-name evals/gpt4o_mini.yaml ++evals.plancraft.valid_actions='[move,smelt,think,search,impossible]'

# image inputs
uv run main.py --config-name evals/gpt4o_mini.yaml ++evals.plancraft.valid_actions='[move,smelt]' ++evals.plancraft.use_images=True ++evals.plancraft.use_text_inventory=False
# both image and text inputs
uv run main.py --config-name evals/gpt4o_mini.yaml ++evals.plancraft.valid_actions='[move,smelt]' ++evals.plancraft.use_images=True
# use fasterrcnn
uv run main.py --config-name evals/gpt4o_mini.yaml ++evals.plancraft.valid_actions='[move,smelt]' ++evals.plancraft.use_images=True ++evals.plancraft.use_text_inventory=False ++evals.plancraft.use_fasterrcnn=True



uv run main.py --config-name evals/dummy.yaml