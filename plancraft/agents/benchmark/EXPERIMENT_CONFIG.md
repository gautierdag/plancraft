# Experiment Configuration

This document specifies the exact hyperparameters, configurations, and environment settings used for the Copilot Multi-Agent architectures benchmark run.

## General Settings
- **Model**: `gpt-5-mini`
- **Backend API**: `copilot-sdk`
- **Dataset Split**: `val.small`
- **Total Examples Evaluation**: 110 (the complete val.small subset)
- **Max Action Steps per Example**: 30

## Architecture-Specific Configurations

All multi-agent architectures were run using the default hyperparameter values defined in the CLI parser for `eval_copilot.py`.

### 1. Single Agent (SAS)
- **Agents**: 1
- **Rounds**: N/A
- **LLM Temperature**: 0.0 (Deterministic)
- **Topology**: No coordination (Direct environment loop).
- **Description**: The baseline single agent acts in a sequential loop, maintaining standard conversation history without any orchestration or peer communication overhead.

### 2. Independent MAS
- **Agents (n)**: 3 parallel workers
- **Rounds**: N/A
- **LLM Temperature**: 0.7 for workers, 0.0 for final aggregator
- **Topology**: Agent $\rightarrow$ Aggregator only (No peer communication)
- **Description**: 3 agents explore the same state in parallel to generate diverse proposals. A final, single LLM call evaluates all concatenated proposals and synthesises one concrete action. 

### 3. Centralized MAS
- **Workers (n)**: 3
- **Orchestrator**: 1
- **Orchestration Rounds (r)**: 1
- **Topology**: Star (Orchestrator $\leftrightarrow$ Workers)
- **Description**: A hierarchical setup. The single orchestrator analyzes the state and issues a planning directive. The 3 workers execute the directive in parallel. The orchestrator checks their parallel outputs and selects/synthesises the best action into a single step.

### 4. Decentralized MAS
- **Agents (n)**: 3
- **Debate Rounds (d)**: 1 (Configured under the `--rounds` flag)
- **Topology**: Fully connected mesh (All-to-All Peer)
- **Description**: All 3 agents propose initial actions. Over 1 debate round, each agent receives a summary of everyone else's proposals and updates its own position based on peer input. The environment action is selected via a simple majority vote (Consensus).

### 5. Hybrid MAS
- **Workers (n)**: 3
- **Orchestrator**: 1
- **Orchestration Rounds (r)**: 1
- **Lateral Peer Rounds (p)**: 1
- **Topology**: Hierarchical Star + Lateral Peer communication
- **Description**: The orchestrator parses the environment and issues a directive. Workers propose initial actions. **Unlike Centralized, these workers exchange their proposals directly for 1 lateral peer round to self-correct/refine.** The orchestrator then performs final synthesis on those *peer-refined* proposals.

## CLI Execution Command Reference
The benchmark was executed using exact terminal commands formatted as follows, utilizing the implicit fallback to default hyperparameters (`n=3`, `r=1`, `p=1`):

```bash
python3 -u eval_copilot.py \
  --split val.small \
  --max-examples 110 \
  --max-steps 30 \
  --model gpt-5-mini \
  --architecture [single|independent|centralized|decentralized|hybrid]
```
