
import json
import os
import argparse
from typing import List, Dict

# Constants for estimation
AVG_SYSTEM_PROMPT_TOKENS = 500
AVG_USER_OBSERVATION_TOKENS = 300
AVG_MODEL_RESPONSE_TOKENS = 50
AVG_SEARCH_RESULT_TOKENS = 100

def estimate_tokens(
    dataset_path: str,
    arch: str,
    steps: int,
    num_agents: int = 3,
    rounds: int = 2
) -> int:
    """
    Estimate total tokens (input + output) for a single example
    running for `steps` turns.
    """
    
    # 1. Calculate per-turn context growth
    # Each turn adds: User Obs + Model Resp
    # Context at step k approx: System + k * (Obs + Resp)
    
    total_tokens = 0
    
    # Simulation loop
    current_context = AVG_SYSTEM_PROMPT_TOKENS
    
    for step in range(1, steps + 1):
        # Input tokens for this step
        input_tokens = current_context + AVG_USER_OBSERVATION_TOKENS
        
        # Output tokens
        output_tokens = AVG_MODEL_RESPONSE_TOKENS
        
        # Arch-specific multipliers
        if arch == "single":
            # 1 call per step
            total_tokens += (input_tokens + output_tokens)
            
        elif arch == "independent":
            # N calls per step (parallel but independent)
            # Each agent sees same context
            total_tokens += num_agents * (input_tokens + output_tokens)
            
        elif arch == "centralized":
            # 2 calls per step: Orchestrator + Worker
            # Orchestrator sees context
            orch_input = input_tokens
            orch_output = AVG_MODEL_RESPONSE_TOKENS # Plan
            
            # Worker sees context + plan + directive
            worker_input = input_tokens + orch_output + 50 # directive overhead
            worker_output = AVG_MODEL_RESPONSE_TOKENS # Action
            
            total_tokens += (orch_input + orch_output) + (worker_input + worker_output)
            
        elif arch == "decentralized":
            # 1. Initial Proposals (N agents)
            total_tokens += num_agents * (input_tokens + output_tokens)
            
            # 2. Debate Rounds (N agents * R rounds)
            # Context grows with debate history
            debate_context_overhead = 0
            for r in range(rounds):
                # Each agent sees other agents' previous proposals
                # Debate context approx: N * Avg_Resp
                debate_overhead = num_agents * AVG_MODEL_RESPONSE_TOKENS
                
                # Input for this round
                debate_input = input_tokens + debate_context_overhead + debate_overhead
                total_tokens += num_agents * (debate_input + output_tokens)
                
                # Accumulate history for next round
                debate_context_overhead += debate_overhead

        elif arch == "hybrid":
             # 1. Manager (1 call)
            total_tokens += (input_tokens + output_tokens)
            
            # 2. Workers Debate (N agents * 1 round)
            debate_input = input_tokens + output_tokens # Manager directive
            total_tokens += num_agents * (debate_input + output_tokens)
            
            # 3. Manager Final (1 call)
            # Sees worker proposals
            final_input = input_tokens + output_tokens + (num_agents * AVG_MODEL_RESPONSE_TOKENS)
            total_tokens += (final_input + output_tokens)

        # Update context for next step
        current_context += (AVG_USER_OBSERVATION_TOKENS + AVG_MODEL_RESPONSE_TOKENS)
        
    return total_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="plancraft/data/val.small.easy.json")
    parser.add_argument("--steps", type=int, default=10, help="Avg steps to solve")
    parser.add_argument("--limit", type=int, default=20, help="Number of examples")
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"Dataset not found: {args.dataset}")
        # Try relative path from scripts/
        args.dataset = os.path.join("..", args.dataset)
        if not os.path.exists(args.dataset):
             print(f"Dataset still not found: {args.dataset}")
             return

    with open(args.dataset) as f:
        data = json.load(f)
        
    num_examples = min(len(data), args.limit)
    print(f"📊 Estimating tokens for {num_examples} examples (Avg {args.steps} steps/ex)\n")
    
    archs = ["single", "independent", "centralized", "decentralized", "hybrid"]
    
    print(f"{'Architecture':<15} | {'Per Ex (k)':<10} | {'Total (M)':<10} | {'Cost ($)*':<10}")
    print("-" * 55)
    
    for arch in archs:
        tokens_per_ex = estimate_tokens(args.dataset, arch, args.steps)
        total_tokens = tokens_per_ex * num_examples
        
        # gemini-1.5-flash pricing (approx $0.075 / 1M input, negligible output)
        # Input is dominant in agent workloads
        cost = (total_tokens / 1_000_000) * 0.075
        
        print(f"{arch:<15} | {tokens_per_ex/1000:<10.1f}k | {total_tokens/1000000:<10.2f}M | ${cost:<10.2f}")
        
    print("\n* Cost estimated using Gemini 1.5 Flash pricing ($0.075/1M tokens)")
    print("  For Copilot SDK with gpt-5-mini: $0.00 (0 premium request cost)")
    print("  Actual usage may vary based on exact prompt length and retries.")

if __name__ == "__main__":
    main()
