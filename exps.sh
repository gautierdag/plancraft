# generate all results from paper

# Oracle benchmark
uv run main.py --config-name evals/oracle.yaml ++evals.plancraft.valid_actions='[move,smelt,impossible]'

# Text only - GPT4o-mini
uv run main.py --config-name evals/gpt4o_mini.yaml ++evals.plancraft.valid_actions='[move,smelt]'
uv run main.py --config-name evals/gpt4o_mini.yaml ++evals.plancraft.valid_actions='[move,smelt,think]'
uv run main.py --config-name evals/gpt4o_mini.yaml ++evals.plancraft.valid_actions='[move,smelt,think,search]'
uv run main.py --config-name evals/gpt4o_mini.yaml ++evals.plancraft.valid_actions='[move,smelt,think,search,impossible]'

# Text only - Llama 8B
uv run main.py --config-name evals/llama8B.yaml ++evals.plancraft.valid_actions='[move,smelt]'
uv run main.py --config-name evals/llama8B.yaml ++evals.plancraft.valid_actions='[move,smelt,think]'
uv run main.py --config-name evals/llama8B.yaml ++evals.plancraft.valid_actions='[move,smelt,think,search]'
uv run main.py --config-name evals/llama8B.yaml ++evals.plancraft.valid_actions='[move,smelt,think,search,impossible]'

# Text only - Llama 70B
uv run main.py --config-name evals/llama70B.yaml ++evals.plancraft.valid_actions='[move,smelt]'
uv run main.py --config-name evals/llama70B.yaml ++evals.plancraft.valid_actions='[move,smelt,think]'
uv run main.py --config-name evals/llama70B.yaml ++evals.plancraft.valid_actions='[move,smelt,think,search]'
uv run main.py --config-name evals/llama70B.yaml ++evals.plancraft.valid_actions='[move,smelt,think,search,impossible]'

# Text only - Llama 8B FT
uv run main.py --config-name evals/llama8B_FT.yaml ++evals.plancraft.valid_actions='[move,smelt]'
uv run main.py --config-name evals/llama8B_FT.yaml ++evals.plancraft.valid_actions='[move,smelt,think]'
uv run main.py --config-name evals/llama8B_FT.yaml ++evals.plancraft.valid_actions='[move,smelt,think,search]'
uv run main.py --config-name evals/llama8B_FT.yaml ++evals.plancraft.valid_actions='[move,smelt,think,search,impossible]'


# Image results
# use fasterrcnn
uv run main.py --config-name evals/gpt4o_mini.yaml ++evals.plancraft.valid_actions='[move,smelt]' ++evals.plancraft.use_images=True ++evals.plancraft.use_text_inventory=False ++evals.plancraft.use_fasterrcnn=True
uv run main.py --config-name evals/llama8B.yaml ++evals.plancraft.valid_actions='[move,smelt]' ++evals.plancraft.use_images=True ++evals.plancraft.use_text_inventory=False ++evals.plancraft.use_fasterrcnn=True
uv run main.py --config-name evals/llama70B.yaml ++evals.plancraft.valid_actions='[move,smelt]' ++evals.plancraft.use_images=True ++evals.plancraft.use_text_inventory=False ++evals.plancraft.use_fasterrcnn=True

# using images only
uv run main.py --config-name evals/gpt4o_mini.yaml ++evals.plancraft.valid_actions='[move,smelt]' ++evals.plancraft.use_images=True ++evals.plancraft.use_text_inventory=False