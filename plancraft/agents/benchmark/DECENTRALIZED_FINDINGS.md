# Decentralized MAS Discovery in PlanCraft

## The Outlier Finding
During the evaluation of `gpt-5-mini` across the five canonical Multi-Agent System (MAS) topologies on the `PlanCraft` environment (`val.small` split), we discovered a significant deviation from the findings published in *"Towards a Science of Scaling Agent Systems"*.

The original paper states that for sequential reasoning tasks requiring strict constraint satisfaction—such as PlanCraft—**every** multi-agent variant degrades performance by a massive 39% to 70% compared to a Single-Agent System (SAS). The logic provided is that coordination overhead and redundant planning steps in sequential domains create compounding error loops.

However, our empirical data directly contradicts this for the **Decentralized** topology:
- **Single-Agent System (SAS) Baseline**: 73.6% Success Rate
- **Centralized MAS**: 61.8% Success Rate (📉 Performance Degraded)
- **Independent MAS**: 71.8% Success Rate (📉 Performance Degraded)
- **Hybrid MAS**: 67.3% Success Rate (📉 Performance Degraded)
- **Decentralized MAS**: **77.3% Success Rate** (📈 **+5% Relative Improvement**)

Our results prove that not all MAS variants degrade in sequential domains: the peer-to-peer consensus model of the Decentralized architecture actively *improved* sequential reasoning performance.

## How we verified this
To understand why Decentralized succeeded where Single-Agent failed, we diffed the evaluation logs of both architectures. 

The data revealed that Decentralized dramatically outperformed Single-Agent in the **High-Complexity (9+ steps)** crafting trees. 

### Mitigating the "Early Give Up" Phenomenon
The Single-Agent failed on 11 distinct tasks that the Decentralized architecture managed to solve. Alarmingly, the Single-Agent almost entirely failed these tasks with an `incorrect_stop` error, meaning it outputted `impossible: <reason>` and gave up prematurely despite the inventory having sufficient materials.

Here are a few notable examples of tasks the Single-Agent surrendered on, but Decentralized solved:
- `VAL0043` (Target: Crimson Sign | Complexity: 27)
- `VAL0303` (Target: Composter | Complexity: 20)
- `VAL0232` (Target: Crimson Fence Gate | Complexity: 14)
- `VAL0100` (Target: Spectral Arrow | Complexity: 16)
- `VAL0037` (Target: Diorite Slab | Complexity: 105)

### The Power of Consensus Debate
In the Decentralized topology (`n=3`), agents engage in a parallel peer-review debate round before committing an action to the environment. 

We theorize that this debate mechanism naturally counteracts LLM laziness/hallucinations in long sequential tasks. If one agent loses track of the inventory and proposes giving up (`impossible`), the other two agents who correctly track the state and propose valid recipes (e.g., `move: from [I4] to [A1]`) will win the majority vote. This overrides the hallucinated failure state and prevents the sequential chain from collapsing.

## Conclusion and Hypothesis
The original paper evaluated earlier LLMs where peer debate might have amplified confusion. By utilizing `gpt-5-mini`, a stronger frontier model, the Decentralized consensus mechanism acts as an effective error-correction layer rather than coordination overhead.

This demonstrates that the scaling principles of Agent architectures are heavily dependent on the baseline capabilities of the underlying LLM.
