"""
Plancraft Evaluation Agent powered by GitHub Copilot SDK.

Uses the Copilot SDK (github-copilot-sdk) to call LLMs through the
Copilot CLI. Supports all 5 agent architectures.

Usage:
    python eval_copilot.py [--split val.small] [--max-steps 30] [--max-examples 0] [--model gpt-5-mini] [--architecture single]
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime

from plancraft.simple import PlancraftGymWrapper, get_plancraft_examples
from plancraft.environment.actions import (
    MoveActionHandler,
    SmeltActionHandler,
    ImpossibleActionHandler,
)
from plancraft.agents.copilot_llm import get_copilot_client, DEFAULT_MODEL
from plancraft.agents.copilot_single import CopilotSingleAgent
from plancraft.agents.copilot_multi import (
    CopilotIndependentAgent,
    CopilotCentralizedAgent,
    CopilotDecentralizedAgent,
    CopilotHybridAgent,
)


def get_agent(
    architecture: str,
    model_name: str,
    client,
    num_agents: int = 3,
    rounds: int = 1,
    peer_rounds: int = 1,
):
    if architecture == "single":
        return CopilotSingleAgent(model_name, client)
    elif architecture == "independent":
        return CopilotIndependentAgent(model_name, client, num_agents=num_agents)
    elif architecture == "centralized":
        return CopilotCentralizedAgent(model_name, client, num_agents=num_agents, rounds=rounds)
    elif architecture == "decentralized":
        return CopilotDecentralizedAgent(model_name, client, num_agents=num_agents, rounds=rounds)
    elif architecture == "hybrid":
        return CopilotHybridAgent(model_name, client, num_agents=num_agents, rounds=rounds, peer_rounds=peer_rounds)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


async def run_evaluation(
    split: str = "val.small",
    max_steps: int = 30,
    max_examples: int = 0,
    model_name: str = DEFAULT_MODEL,
    architecture: str = "single",
    num_agents: int = 3,
    rounds: int = 1,
    peer_rounds: int = 1,
):
    # ---- Setup Copilot client ----
    print("🔌 Starting Copilot SDK client...")
    client = await get_copilot_client()
    print("✅ Copilot client started")

    try:
        # ---- Load dataset ----
        examples = get_plancraft_examples(split=split)
        if max_examples > 0:
            examples = examples[:max_examples]

        print(f"📦 Loaded {len(examples)} examples from split '{split}'")
        print(f"🤖 Model: {model_name} (via Copilot SDK)")
        print(f"🏗️  Architecture: {architecture} (n={num_agents}, r={rounds}, p={peer_rounds})")
        print(f"🔄 Max steps per example: {max_steps}")
        print("-" * 60)

        results = []
        total_start = time.time()

        for idx, example in enumerate(examples):
            print(
                f"\n[{idx+1}/{len(examples)}] Example {example.id} | "
                f"Target: {example.target} | "
                f"Impossible: {example.impossible} | "
                f"Complexity: {example.complexity}"
            )

            # Create environment
            env = PlancraftGymWrapper(
                example=example,
                actions=[
                    MoveActionHandler(),
                    SmeltActionHandler(),
                    ImpossibleActionHandler(),
                ],
                max_steps=max_steps,
                use_text_inventory=True,
            )

            # Initialize Agent
            agent = get_agent(architecture, model_name, client, num_agents=num_agents, rounds=rounds, peer_rounds=peer_rounds)
            agent.reset(example.id, example.target)

            # Get initial observation
            observation, reward, terminated, truncated, info = env.step("")

            step_count = 0
            while not (terminated or truncated):
                step_count += 1

                # Agent decides action (async)
                try:
                    action_text = await agent.act(observation["text"])
                except Exception as e:
                    print(f"  ⚠ Agent error: {e}")
                    action_text = "impossible: Agent error"

                print(f"  Step {step_count}: {action_text}")

                # Execute action
                observation, reward, terminated, truncated, info = env.step(action_text)

            # Record result
            success = env.success
            status = "✅" if success else "❌"
            reason = info.get("reason", "unknown")
            print(f"  {status} Result: success={success} | reason={reason} | steps={step_count}")

            results.append(
                {
                    "example_id": example.id,
                    "target": example.target,
                    "impossible": example.impossible,
                    "complexity": example.complexity,
                    "success": success,
                    "steps": step_count,
                    "reason": reason,
                }
            )

        # ---- Summary ----
        elapsed = time.time() - total_start
        total = len(results)
        successes = sum(r["success"] for r in results)
        success_rate = successes / total if total > 0 else 0

        possible = [r for r in results if not r["impossible"]]
        impossible = [r for r in results if r["impossible"]]
        possible_success = sum(r["success"] for r in possible) if possible else 0
        impossible_success = sum(r["success"] for r in impossible) if impossible else 0

        print("\n" + "=" * 60)
        print("📊 EVALUATION RESULTS (Copilot SDK)")
        print("=" * 60)
        print(f"  Model:          {model_name}")
        print(f"  Architecture:   {architecture}")
        print(f"  Split:          {split}")
        print(f"  Total examples: {total}")
        print(f"  Overall:        {successes}/{total} = {success_rate:.1%}")
        if possible:
            print(
                f"  Possible tasks: {possible_success}/{len(possible)} = "
                f"{possible_success/len(possible):.1%}"
            )
        if impossible:
            print(
                f"  Impossible:     {impossible_success}/{len(impossible)} = "
                f"{impossible_success/len(impossible):.1%}"
            )
        print(f"  Time elapsed:   {elapsed:.1f}s ({elapsed/total:.1f}s per example)")
        print("=" * 60)

        # ---- Save results ----
        output_dir = "plancraft/agents/benchmark/output"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            output_dir, f"copilot_{architecture}_{split}_{timestamp}.json"
        )
        with open(output_file, "w") as f:
            json.dump(
                {
                    "model": model_name,
                    "backend": "copilot-sdk",
                    "architecture": architecture,
                    "split": split,
                    "max_steps": max_steps,
                    "total": total,
                    "successes": successes,
                    "success_rate": success_rate,
                    "elapsed_seconds": elapsed,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"\n💾 Results saved to {output_file}")

    finally:
        # Always clean up the Copilot client
        print("🔌 Stopping Copilot client...")
        await client.stop()
        print("✅ Copilot client stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plancraft Copilot SDK Evaluation")
    parser.add_argument("--split", default="val.small", help="Dataset split to evaluate on")
    parser.add_argument("--max-steps", type=int, default=30, help="Max steps per example")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Max examples to evaluate (0 = all)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--architecture",
        default="single",
        choices=["single", "independent", "centralized", "decentralized", "hybrid"],
        help="Agent architecture to use",
    )
    # MAS hyperparameters (paper §3.1 / Table 2)
    parser.add_argument("--num-agents", type=int, default=3, help="Number of sub-agents (n)")
    parser.add_argument("--rounds", type=int, default=1, help="Orchestrator / debate rounds (r or d)")
    parser.add_argument("--peer-rounds", type=int, default=1, help="Lateral peer rounds for Hybrid (p)")
    args = parser.parse_args()

    asyncio.run(
        run_evaluation(
            split=args.split,
            max_steps=args.max_steps,
            max_examples=args.max_examples,
            model_name=args.model,
            architecture=args.architecture,
            num_agents=args.num_agents,
            rounds=args.rounds,
            peer_rounds=args.peer_rounds,
        )
    )
