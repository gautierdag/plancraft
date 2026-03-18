# Hybrid Architecture: Anthropic-Style Orchestrator-Worker Pattern

> **File:** `copilot_hybrid.py` → `CopilotHybridAgent`
>
> Inspired by Anthropic's *"How we built our multi-agent research system"* (June 2025)
> and Kim et al., *"Towards a Science of Scaling Agent Systems"* (2025, arXiv:2512.08296, §3.1).

---

## Table of Contents

1. [Motivation: Why a New Hybrid?](#1-motivation-why-a-new-hybrid)
2. [What Anthropic Described](#2-what-anthropic-described)
3. [What the Old Hybrid Actually Did](#3-what-the-old-hybrid-actually-did)
4. [Gap Analysis: Old Hybrid vs. Anthropic](#4-gap-analysis-old-hybrid-vs-anthropic)
5. [New Hybrid Architecture](#5-new-hybrid-architecture)
   - [5.1 Architecture Overview](#51-architecture-overview)
   - [5.2 The Four Phases](#52-the-four-phases)
   - [5.3 Component Deep-Dive](#53-component-deep-dive)
6. [Execution Trace: Side-by-Side](#6-execution-trace-side-by-side)
7. [Code Walkthrough](#7-code-walkthrough)
   - [7.1 SharedMemory](#71-sharedmemory)
   - [7.2 AutonomousWorker](#72-autonomousworker)
   - [7.3 CopilotHybridAgent](#73-copilothybridagent)
8. [Prompt Design](#8-prompt-design)
9. [Configuration & Hyperparameters](#9-configuration--hyperparameters)
10. [Design Decisions & Trade-offs](#10-design-decisions--trade-offs)
11. [Relationship to Other Architectures](#11-relationship-to-other-architectures)

---

## 1. Motivation: Why a New Hybrid?

The original `CopilotHybridAgent` in `copilot_multi.py` implemented a Hybrid MAS
following Kim et al. (2025) — an orchestrator issuing directives, workers proposing
actions, and peer rounds for lateral refinement. On the surface it looked like an
orchestrator-worker pattern, but under the hood **every "agent" was a single LLM call
with the same prompt**, differentiated only by role-play framing in the system message.

Anthropic's production Research system (June 2025) revealed what a real
orchestrator-worker pattern looks like at scale. The gap was significant enough to
warrant a ground-up reimplementation.

---

## 2. What Anthropic Described

Key architectural principles from Anthropic's blog post:

### 2.1 Subagents Are Autonomous Loops

> *"Each subagent independently performs web searches, evaluates tool results using
> interleaved thinking, and returns findings to the LeadResearcher."*

Workers are not single LLM calls. Each worker is a **full agentic loop** — an LLM
calling tools repeatedly across many turns, making decisions, adapting to tool results.
A subagent might make 3–15 tool calls on its own before returning.

### 2.2 Task Decomposition With Unique Instructions

> *"The lead agent decomposes queries into subtasks and describes them to subagents.
> Each subagent needs an objective, an output format, guidance on the tools and
> sources to use, and clear task boundaries."*

Each worker gets a **different, specific task description** with a distinct objective.
Without this, workers duplicate work or leave gaps.

### 2.3 Own Context Windows

> *"Subagents facilitate compression by operating in parallel with their own context
> windows, exploring different aspects of the question simultaneously before condensing
> the most important tokens for the lead research agent."*

Each subagent has a **separate, clean context window** that it fills through its own
tool calls. It then **compresses** its findings before passing them back.

### 2.4 Dynamic Scaling

> *"Simple fact-finding requires just 1 agent with 3-10 tool calls, direct comparisons
> might need 2-4 subagents with 10-15 calls each, and complex research might use more
> than 10 subagents."*

The orchestrator **decides how many subagents** to spawn based on query complexity.
It can also spawn additional subagents after seeing initial results.

### 2.5 External Memory / Filesystem

> *"Subagent output to a filesystem to minimize the 'game of telephone.' Rather than
> requiring subagents to communicate everything through the lead agent, implement
> artifact systems where specialized agents can create outputs that persist
> independently."*

Workers write to persistent storage; the orchestrator reads lightweight summaries.
This avoids information loss from multi-stage message passing.

### 2.6 Iterative Re-planning

> *"The LeadResearcher synthesizes these results and decides whether more research is
> needed — if so, it can create additional subagents or refine its strategy."*

The orchestrator is not a one-shot planner. It reviews worker output and can
dynamically spawn more workers or adjust strategy.

---

## 3. What the Old Hybrid Actually Did

The original `CopilotHybridAgent` (in `copilot_multi.py`) followed this flow:

```
For r rounds:
  1. Orchestrator LLM call → directive string
  2. n workers LLM call (parallel, same prompt, same context) → proposals
  3. p peer rounds (parallel, broadcast all proposals) → refined proposals
  4. Orchestrator LLM call → synthesise from refined proposals
```

**Every "agent" was a single `call_copilot_with_retry()` call.** The orchestrator,
workers, and peer-round participants were all the same model, same prompt, same context
— differentiated only by appending strings like `"[Orchestrator] Analyse the situation"`
or `"Worker: propose an action based on the directive."` to the conversation.

### Old Hybrid: What a "Worker" Looked Like

```python
# copilot_multi.py — old Hybrid, Step 2
worker_msgs = context + [
    {"role": "model", "content": directive},
    {"role": "user", "content": "Worker: propose an action based on the directive."},
]
worker_tasks = [
    call_copilot_with_retry(
        self.client, self.model_name, worker_msgs, SYSTEM_PROMPT, temperature=0.7
    )
    for _ in range(self.num_agents)
]
results = await asyncio.gather(*worker_tasks, return_exceptions=True)
```

Every worker:
- Received the **same directive**
- Used the **same system prompt** (`SYSTEM_PROMPT`)
- Shared the **same context** (`worker_msgs`)
- Made **one LLM call** and returned a string
- Had **no tool access** (could not search, could not reason across turns)
- Was **not specialised** (all clones hoping temperature sampling produces diversity)

### Old Hybrid: What "Peer Rounds" Looked Like

```python
# copilot_multi.py — old Hybrid, Step 3
for p in range(self.peer_rounds):
    peer_summary = "Peer proposals:\n" + "\n".join(
        [f"- Worker {j+1}: {prop}" for j, prop in enumerate(proposals)]
    )
    peer_question = (
        peer_summary
        + "\nConsidering your peers' proposals, output your refined action."
    )
    peer_tasks = [
        call_copilot_with_retry(
            self.client, self.model_name,
            worker_msgs + [{"role": "user", "content": peer_question}],
            SYSTEM_PROMPT, temperature=0.7,
        )
        for _ in range(self.num_agents)
    ]
```

This was **structurally identical** to the Decentralized agent's debate rounds — same
broadcast-all-proposals → parallel-refine → overwrite-proposals pattern. The only
difference was that Hybrid's peers debated on top of `worker_msgs` (which included the
orchestrator's directive), while Decentralized's agents debated on top of raw
`self.conversation`.

---

## 4. Gap Analysis: Old Hybrid vs. Anthropic

| Feature | Anthropic's System | Old Hybrid (`copilot_multi.py`) |
|---|---|---|
| **Worker autonomy** | Multi-turn agentic loops with tools | Single LLM call, no tools |
| **Task specialisation** | Each worker gets a unique subtask | All workers get the same prompt |
| **Own context windows** | Each subagent has independent context | All share `worker_msgs` |
| **Dynamic scaling** | Orchestrator decides worker count | Fixed `num_agents` at init |
| **Tool access for workers** | Workers search, browse, call APIs | Workers just generate text |
| **External memory** | Persistent filesystem/artifact storage | Flat `self.conversation` list |
| **Output bypass** | Workers write to storage directly | Everything funnels through messages |
| **Error recovery** | Resume from checkpoints, graceful degradation | If a worker fails, it's just filtered out |
| **Iterative re-planning** | Orchestrator can spawn more workers after seeing results | Fixed number of rounds |
| **Prompt differentiation** | Distinct prompts per role | One shared `SYSTEM_PROMPT` for everyone |

### The Fundamental Problem

The old Hybrid was not really an orchestrator-worker system. It was **the same model
role-playing different roles in single shots**, with all "coordination" happening
through string concatenation in shared conversation history. The peer rounds were
copy-paste equivalent to Decentralized debate. The orchestrator's "directive" was just
a string injected into the same context every worker saw.

If you removed the orchestrator directive and synthesis from the old Hybrid and replaced
the final step with a majority vote, you would get **Decentralized**. If you removed
the peer rounds, you would get **Centralized**. The old Hybrid was a mechanical
composition of the two, not a genuinely distinct architecture.

---

## 5. New Hybrid Architecture

### 5.1 Architecture Overview

```
                         ┌──────────────┐
                         │ Observation  │
                         └──────┬───────┘
                                │
                    ┌───────────▼────────────┐
                    │   ORCHESTRATOR (Plan)   │  Phase 1
                    │  • analyse state        │  — unique prompt
                    │  • judge complexity      │  — dynamic worker count
                    │  • decompose subtasks    │  — JSON structured output
                    └───┬───────┬─────────┬──┘
                        │       │         │
                  (unique)  (unique)  (unique)    ← each gets different instructions
                        │       │         │
               ┌────────▼──┐ ┌─▼───────┐ ┌▼──────────┐
               │ Worker 1  │ │ Worker 2│ │ Worker 3  │  Phase 2
               │ recipe    │ │ grid    │ │ inventory │  — parallel
               │ researcher│ │ planner │ │ checker   │  — autonomous loops
               │ (≤5 steps)│ │(≤5 stp) │ │ (≤5 stp) │  — own context
               │ [search]  │ │[analyse]│ │[analyse]  │  — tool access
               │ [reason]  │ │[search] │ │[recommend]│
               │ [recommend│ │[recomm.]│ │           │
               └────┬──────┘ └──┬──────┘ └──┬────────┘
                    │           │            │
                    ▼           ▼            ▼
              ┌──────────────────────────────────┐
              │        SHARED MEMORY             │  Workers write here
              │  WorkerFinding(compressed)       │  (not conversation history)
              │  • status, recommendations       │
              │  • search results, step count    │
              └──────────────┬───────────────────┘
                             │
                  ┌──────────▼───────────┐
                  │ ORCHESTRATOR (Replan) │  Phase 3
                  │ • reads memory summary│  — decides: enough info?
                  │ • enough info?        │  — can spawn more workers
                  │ • spawn more workers? │  — up to max_replan_rounds
                  └──────────┬───────────┘
                             │
                  ┌──────────▼──────────────┐
                  │ ORCHESTRATOR (Synthesise)│  Phase 4
                  │ • reads all findings     │  — single final action
                  │ • picks best action      │  — from memory, not chat
                  └──────────┬──────────────┘
                             │
                         ┌───▼───┐
                         │Action │
                         └───────┘
```

### 5.2 The Four Phases

| Phase | Orchestrator Role | Workers | Communication Channel |
|---|---|---|---|
| **Phase 1: Plan** | Analyse state → produce JSON plan with subtasks | — | — |
| **Phase 2: Execute** | — | Run autonomous agentic loops in parallel | Workers → SharedMemory |
| **Phase 3: Replan** | Review findings → spawn more workers if needed | (additional workers) | SharedMemory → Orchestrator |
| **Phase 4: Synthesise** | Read all findings → produce single action | — | SharedMemory → Orchestrator |

### 5.3 Component Deep-Dive

#### SharedMemory

An in-memory data store that decouples worker output from conversation history.

**Why it exists:** In the old Hybrid, everything flowed through `self.conversation` — a
flat list of message dicts. Worker proposals were stuffed into conversation messages,
then peer proposals were stuffed in again, then the orchestrator saw all of it.
Anthropic warns about this: *"Subagent output to a filesystem to minimize the 'game of
telephone.'"*

SharedMemory provides:
- `store_plan(plan)` — orchestrator writes its decomposition
- `store_finding(finding)` — workers write compressed results
- `get_findings_summary()` — orchestrator reads a structured summary
- `get_plan_summary()` — for re-planning context
- `clear()` — reset between environment steps

Workers write `WorkerFinding` objects containing:
- `worker_id`, `role`, `objective` (identity)
- `status` — `"success"` / `"partial"` / `"failed"`
- `finding` — the compressed output text
- `recommendations` — list of proposed actions
- `search_results` — any recipe lookups performed
- `steps_taken` — how many agentic loop iterations ran

The orchestrator never sees the workers' raw conversation history. It reads a compressed
summary, which prevents context bloat and information distortion.

#### AutonomousWorker

A worker that runs its own multi-turn agentic loop.

**Critical difference from old Hybrid:** This is NOT a single LLM call. The worker
loops up to `max_steps` times, dispatching tools and building its own private context
window.

The worker's agentic loop:

```
Initialise own context (independent from all other workers)
For step in range(max_steps):
    Call LLM with own context + own unique system prompt
    Parse response for tool commands:
        "search: <item>"    → invoke oracle search, add result to context, continue
        "analyse: <text>"   → acknowledge, nudge toward recommendation, continue
        "recommend: <action>" → record recommendation, STOP (terminal)
        (no tool match)     → nudge toward recommendation, continue
    Write WorkerFinding to SharedMemory
```

Each worker gets a unique system prompt built from its subtask assignment:

```
## Your Assignment
- Role: {role}             ← e.g. "recipe_researcher"
- Objective: {objective}   ← e.g. "Find the recipe for iron_pickaxe"
- Instructions: {instructions} ← e.g. "Search for iron_pickaxe, identify materials"
```

The worker's available tools:
- `search: <item_name>` — invokes `_oracle_search()`, returns recipe text, adds to
  context, continues the loop
- `analyse: <text>` — non-terminal reasoning step, worker is nudged to continue
- `recommend: <action>` — terminal action, stops the loop and records the recommendation

#### CopilotHybridAgent

The main agent class orchestrating the full pipeline.

```
A = {a_orch, a_1, ..., a_n}   (n chosen dynamically per step)
C = star via shared memory     (no game of telephone)
Ω = hierarchical + iterative re-planning
```

---

## 6. Execution Trace: Side-by-Side

### Scenario

Target: craft `iron_pickaxe`. Inventory has iron ingots and sticks.

### Old Hybrid (rounds=1, peer_rounds=1, num_agents=3)

**Call 1: Orchestrator directive (single)**
```
Input:   self.conversation (raw observation)
Prompt:  SYSTEM_PROMPT + "[Orchestrator] Analyse the situation..."
Output:  "We need to place 3 iron ingots across the top row and 2 sticks down
          the middle column..."
→ directive: D (a string)
```

**Call 2–4: Workers propose (parallel, identical context)**
```
Input:   self.conversation + [model: D, user: "Worker: propose an action..."]
Prompt:  SYSTEM_PROMPT (same for all 3)
Temp:    0.7
Output:  Worker 1: "move: from [I1] to [A1] with quantity 1"
         Worker 2: "move: from [I1] to [A1] with quantity 1"
         Worker 3: "move: from [I2] to [A1] with quantity 1"
→ proposals: [P1, P2, P3]  (likely similar due to shared context)
```

**Call 5–7: Peer round (parallel, broadcast all proposals)**
```
Input:   worker_msgs + [user: "Peer proposals: - Worker 1: P1 - Worker 2: P2 ..."]
Prompt:  SYSTEM_PROMPT (same for all 3)
Temp:    0.7
Output:  3 refined proposals (likely converged to same answer)
→ refined: [P1', P2', P3']
```

**Call 8: Orchestrator synthesis (single)**
```
Input:   worker_msgs + [model: summary, user: "Select the single best action."]
Prompt:  SYSTEM_PROMPT
Output:  "move: from [I1] to [A1] with quantity 1"
```

**Total: 8 LLM calls. 0 tool uses. 0 specialisation. All workers saw the same thing.**

---

### New Hybrid (max_workers=3, max_worker_steps=5, max_replan_rounds=1)

**Call 1: Orchestrator plans (single)**
```
Input:   self.conversation (raw observation)
Prompt:  ORCHESTRATOR_PLAN_PROMPT (unique orchestrator prompt)
Temp:    0.0
Output:  {
           "complexity": "moderate",
           "analysis": "Need iron_pickaxe recipe. Have iron ingots and sticks.",
           "subtasks": [
             {
               "id": 1,
               "role": "recipe_researcher",
               "objective": "Find the exact crafting recipe for iron_pickaxe",
               "instructions": "Search for iron_pickaxe and report the grid layout",
               "tools_hint": "search"
             },
             {
               "id": 2,
               "role": "inventory_analyst",
               "objective": "Map inventory slots to required materials",
               "instructions": "Identify which inventory slots contain iron_ingot
                                and stick, report exact slot positions",
               "tools_hint": "inventory_check"
             }
           ]
         }
→ plan stored in SharedMemory
→ dynamic scaling: complexity "moderate" → 2 workers (not fixed 3)
```

**Worker 1: recipe_researcher (autonomous loop, own context)**
```
Step 0:  LLM call → "search: iron_pickaxe"
         Tool:    oracle search invoked
         Result:  "Recipes to craft iron_pickaxe: recipe 1:
                   iron_ingot at [A1], iron_ingot at [A2], iron_ingot at [A3],
                   stick at [B2], stick at [C2]"
         → added to worker's own context

Step 1:  LLM call → "recommend: move: from [I1] to [A1] with quantity 1"
         → STOP. Finding written to SharedMemory.

Steps taken: 2 LLM calls + 1 tool call
```

**Worker 2: inventory_analyst (autonomous loop, own context, in parallel)**
```
Step 0:  LLM call → "analyse: Inventory contains iron_ingot in [I1] (qty 3)
                     and stick in [I5] (qty 2). Need 3 iron_ingot and 2 stick."

Step 1:  LLM call → "recommend: move: from [I1] to [A1] with quantity 1"
         → STOP. Finding written to SharedMemory.

Steps taken: 2 LLM calls + 0 tool calls
```

**Call ~6: Orchestrator re-plans (single)**
```
Input:   self.conversation
Prompt:  ORCHESTRATOR_REPLAN_PROMPT (with findings summary from SharedMemory)
         "Worker 1 — recipe_researcher: Found recipe. Recommends move I1→A1.
          Worker 2 — inventory_analyst: Mapped slots. Recommends move I1→A1."
Output:  {"ready": true, "reasoning": "Both workers agree. Recipe and inventory
          confirmed.", "additional_subtasks": []}
→ No additional workers needed.
```

**Call ~7: Orchestrator synthesises (single)**
```
Input:   self.conversation
Prompt:  ORCHESTRATOR_SYNTHESIZE_PROMPT (with findings from SharedMemory)
Temp:    0.0
Output:  "move: from [I1] to [A1] with quantity 1"
```

**Total: ~7 LLM calls. 1 tool use (search). 2 specialised roles. Independent contexts.
Dynamic scaling chose 2 workers instead of 3.**

---

### Comparison Summary

| Metric | Old Hybrid | New Hybrid |
|---|---|---|
| **LLM calls** | 8 (fixed) | ~7 (variable) |
| **Tool calls by workers** | 0 | 1 (search) |
| **Worker specialisation** | None — all clones | recipe_researcher + inventory_analyst |
| **Worker autonomy** | 1 call each | Up to 5 steps each |
| **Context isolation** | All share `worker_msgs` | Each has own `self.context` |
| **Peer rounds** | 1 round of broadcast+refine | None (replaced by specialisation) |
| **Information channel** | Conversation history (telephone) | SharedMemory (compressed) |
| **Worker count** | Fixed at init (3) | Dynamic per step (2 here) |
| **Re-planning** | None | Orchestrator reviewed, decided ready |
| **Orchestrator prompts** | 1 shared `SYSTEM_PROMPT` | 3 distinct role prompts |

---

## 7. Code Walkthrough

### 7.1 SharedMemory

```python
@dataclass
class WorkerFinding:
    """A single finding written by a worker to shared memory."""
    worker_id: int
    role: str
    objective: str
    status: str          # "success" | "partial" | "failed"
    finding: str         # the compressed output
    recommendations: list[str] = field(default_factory=list)
    search_results: list[str] = field(default_factory=list)
    steps_taken: int = 0


class SharedMemory:
    def __init__(self):
        self.findings: list[WorkerFinding] = []
        self.plan: dict | None = None
```

The `get_findings_summary()` method produces a compressed text representation:

```
[Worker 1 — recipe_researcher]
  Objective: Find the exact crafting recipe for iron_pickaxe
  Status: success
  Steps taken: 2
  Finding: move: from [I1] to [A1] with quantity 1
  Recipes found:
    Recipes to craft iron_pickaxe: recipe 1: iron_ingot at [A1]...
  Recommendations:
    → move: from [I1] to [A1] with quantity 1
```

This is what the orchestrator sees — not the worker's raw multi-turn conversation.

### 7.2 AutonomousWorker

The agentic loop in `AutonomousWorker.run()`:

```python
async def run(self, observation: str) -> WorkerFinding:
    # 1. Build UNIQUE system prompt from subtask
    system_prompt = WORKER_PROMPT_TEMPLATE.format(
        role=self.subtask.get("role", "general"),
        objective=self.subtask.get("objective", "..."),
        instructions=self.subtask.get("instructions", "..."),
        observation=observation,
    )

    # 2. Own context — independent from all other workers
    self.context = [{"role": "user", "content": observation}]

    # 3. Agentic loop
    for step in range(self.max_steps):
        response = await call_copilot_with_retry(...)
        self.context.append({"role": "model", "content": response})

        # Tool dispatch
        if "search:" in response.lower():
            result = self.oracle_search(...)
            self.context.append({"role": "user", "content": result})
            continue  # ← loop continues with tool result in context

        if "recommend:" in response.lower():
            # Terminal — extract recommendation, stop loop
            break

        if "analyse:" in response.lower():
            # Non-terminal — nudge toward recommendation
            self.context.append({"role": "user", "content": "...continue..."})
            continue

    # 4. Write to shared memory (NOT conversation history)
    finding = WorkerFinding(...)
    self.memory.store_finding(finding)
    return finding
```

Key properties:
- **Own context**: `self.context` is created fresh per worker, populated only with the
  observation and the worker's own LLM responses and tool results.
- **Tool access**: Workers can invoke `search:` which calls `_oracle_search()` and
  feeds the result back into the worker's context for the next loop iteration.
- **Graceful termination**: If the worker never produces a `recommend:`, the loop
  nudges it toward one. On the final step, whatever output exists is captured as a
  `"partial"` finding.

### 7.3 CopilotHybridAgent

The main `act()` method ties all four phases together:

```python
async def act(self, observation_text: str | None) -> str:
    # Clear per-step memory
    self.shared_memory.clear()

    # Phase 1: Orchestrator plans
    plan = await self._orchestrator_plan(context)
    # Dynamic scaling
    subtasks = plan["subtasks"][:max_allowed_for_complexity]

    # Phase 2: Spawn autonomous workers
    findings = await self._spawn_workers(subtasks, observation)

    # Phase 3: Re-plan if needed
    for replan_round in range(self.max_replan_rounds):
        replan_result = await self._orchestrator_replan(context)
        if replan_result["ready"]:
            break
        # Spawn additional workers from replan_result["additional_subtasks"]

    # Phase 4: Synthesise
    action = await self._orchestrator_synthesize(context)
    return action
```

The `_spawn_workers()` method creates `AutonomousWorker` instances and runs them in
parallel:

```python
async def _spawn_workers(self, subtasks, observation, start_id=1):
    workers = [
        AutonomousWorker(
            worker_id=start_id + i,
            subtask=subtask,          # ← UNIQUE per worker
            client=self.client,
            model_name=self.model_name,
            oracle_search_fn=self._oracle_search,
            shared_memory=self.shared_memory,
            max_steps=self.max_worker_steps,
        )
        for i, subtask in enumerate(subtasks)
    ]
    # All workers run in parallel with independent context windows
    results = await asyncio.gather(*[w.run(observation) for w in workers])
    return [r for r in results if isinstance(r, WorkerFinding)]
```

---

## 8. Prompt Design

The new Hybrid uses **four distinct prompt types** instead of one shared `SYSTEM_PROMPT`:

### 8.1 `ORCHESTRATOR_PLAN_PROMPT`

**Role:** Analyse state, judge complexity, decompose into subtasks.

**Key design choices:**
- Requires **structured JSON output** (not free text) so the plan can be parsed
  programmatically.
- Includes **complexity guidelines** with concrete examples so the LLM can calibrate
  worker count appropriately:
  - `simple` (1 worker) — direct craft, ingredients available
  - `moderate` (2 workers) — recipe lookup + planning, or intermediate crafting
  - `complex` (3 workers) — multi-step chain, unknown recipe

**Output schema:**
```json
{
  "complexity": "simple | moderate | complex",
  "analysis": "brief situation analysis",
  "subtasks": [
    {
      "id": 1,
      "role": "worker role name",
      "objective": "what this worker must accomplish",
      "instructions": "specific step-by-step instructions",
      "tools_hint": "search | grid_placement | inventory_check"
    }
  ]
}
```

### 8.2 `WORKER_PROMPT_TEMPLATE`

**Role:** Execute a specific subtask autonomously.

**Key design choices:**
- **Parameterised per worker** — `{role}`, `{objective}`, `{instructions}` are filled
  from the orchestrator's plan. No two workers see the same prompt.
- **Tool documentation** is embedded in the prompt so the worker knows what actions
  it can take within its agentic loop (`search`, `analyse`, `recommend`).
- **Observation is embedded** — the worker doesn't need to navigate conversation
  history, the current game state is right in its prompt.

### 8.3 `ORCHESTRATOR_REPLAN_PROMPT`

**Role:** Review worker findings and decide whether to proceed or spawn more workers.

**Key design choices:**
- Receives `{findings}` and `{plan}` from SharedMemory, not from conversation history.
- Outputs structured JSON with `ready` boolean and optional `additional_subtasks`.

### 8.4 `ORCHESTRATOR_SYNTHESIZE_PROMPT`

**Role:** Read all findings and produce the single best action.

**Key design choices:**
- Includes the full action specification (move/smelt/impossible/search) with slot
  names, so the orchestrator can produce a valid environment action.
- Receives findings from SharedMemory, not from raw worker conversations.
- Uses `temperature=0.0` for deterministic final output.

### Prompt Comparison: Old vs. New

| Aspect | Old Hybrid | New Hybrid |
|---|---|---|
| Orchestrator prompt | `SYSTEM_PROMPT + "[Orchestrator] Analyse..."` | `ORCHESTRATOR_PLAN_PROMPT` (dedicated) |
| Worker prompt | `SYSTEM_PROMPT` (same for all) | `WORKER_PROMPT_TEMPLATE` (unique per worker) |
| Synthesis prompt | `SYSTEM_PROMPT` | `ORCHESTRATOR_SYNTHESIZE_PROMPT` (dedicated) |
| Replan prompt | N/A (no re-planning) | `ORCHESTRATOR_REPLAN_PROMPT` (dedicated) |
| Output format | Free text | JSON (plan, replan) / Action string (synthesis) |

---

## 9. Configuration & Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `max_workers` | 3 | Maximum number of workers the orchestrator can spawn (across all rounds including re-planning) |
| `max_worker_steps` | 5 | Maximum agentic loop iterations per worker (LLM calls + tool uses) |
| `max_replan_rounds` | 1 | How many times the orchestrator can review findings and spawn additional workers |

### Dynamic Scaling Map

The orchestrator's `complexity` judgement maps to worker count:

| Complexity | Workers Spawned |
|---|---|
| `simple` | 1 |
| `moderate` | 2 |
| `complex` | 3 (capped by `max_workers`) |

This is a **per-step decision** — the orchestrator might spawn 1 worker for a simple
move-from-output step and 3 workers for an unknown multi-step recipe.

### Old Hybrid Hyperparameters (for reference)

| Parameter | Default | Description |
|---|---|---|
| `num_agents` | 3 | Fixed worker count (always 3, regardless of task) |
| `rounds` | 1 | Orchestrator rounds |
| `peer_rounds` | 1 | Lateral peer refinement rounds |

---

## 10. Design Decisions & Trade-offs

### 10.1 No Peer Rounds

The new Hybrid **removes peer rounds entirely**. In the old Hybrid, peer rounds were
the distinguishing feature — workers broadcast proposals and refined them through
lateral communication. In the new Hybrid, this is replaced by **task specialisation**.

**Rationale:**
- Peer rounds were structurally identical to Decentralized debate (see §3–4 above).
  They didn't add anything that task decomposition + specialisation doesn't provide
  better.
- Anthropic's system has no peer communication between subagents. Workers report to
  the orchestrator only (star topology via shared memory).
- If workers are doing **different subtasks**, broadcasting their proposals to each
  other doesn't make sense — a recipe researcher's intermediate search result is not
  useful context for an inventory analyst.

### 10.2 JSON-Structured Orchestrator Output

The orchestrator plan and replan use structured JSON instead of free text.

**Trade-off:**
- **Pro:** Plans are machine-parseable. Dynamic scaling and subtask routing work
  programmatically.
- **Con:** LLMs sometimes fail to produce valid JSON. The `_parse_json()` utility
  handles markdown fences and brace extraction as fallbacks, and the system has a
  fallback plan if parsing fails entirely (single generic worker).

### 10.3 SharedMemory vs. Conversation History

Workers write to SharedMemory; the orchestrator reads compressed summaries.

**Trade-off:**
- **Pro:** Prevents context bloat. The orchestrator doesn't see 5 turns of worker
  reasoning — it sees a structured summary with status, findings, and recommendations.
  This is Anthropic's "minimise the game of telephone" principle.
- **Con:** Information loss. If a worker discovered something subtle in step 3 of its
  loop that it didn't include in its final recommendation, the orchestrator won't see
  it. The compression is lossy by design.

### 10.4 Worker Temperature (0.3 vs. 0.7)

Workers use `temperature=0.3` (old Hybrid used `0.7`).

**Rationale:** In the old Hybrid, diversity came from temperature sampling because all
workers had the same prompt. In the new Hybrid, diversity comes from **different
subtasks and prompts**. Workers don't need high temperature to explore — they're
already exploring different aspects of the problem. Lower temperature makes each
worker more reliable at its specific task.

### 10.5 Graceful Degradation

Every phase has fallbacks:
- **Plan parsing fails:** Single generic worker is spawned.
- **All workers fail:** Returns `"impossible: All workers failed"`.
- **Worker never produces `recommend:`:** Last-step output is captured as `"partial"`.
- **Replan parsing fails:** Proceeds directly to synthesis.
- **Worker budget exhausted during re-planning:** Proceeds with existing findings.

---

## 11. Relationship to Other Architectures

### Where New Hybrid Sits in the Taxonomy

| Architecture | Workers | Worker Autonomy | Communication | Coordination |
|---|---|---|---|---|
| **Single (SAS)** | 0 | N/A | None | Direct |
| **Independent** | n (fixed) | None (1 call) | Worker→Aggregator | Synthesis only |
| **Centralized** | n (fixed) | None (1 call) | Star (orch↔worker) | Hierarchical |
| **Decentralized** | n (fixed) | None (1 call) | All-to-all mesh | Consensus vote |
| **Old Hybrid** | n (fixed) | None (1 call) | Star + peer mesh | Hierarchical + lateral |
| **New Hybrid** | n (dynamic) | Full (multi-turn loop) | Star via SharedMemory | Hierarchical + re-planning |

### What You Lose (vs. Old Hybrid)

- **Peer rounds / lateral communication.** Workers no longer refine each other's
  proposals. This is a deliberate choice — specialisation replaces debate.
- **Simplicity.** The old Hybrid was ~130 lines. The new Hybrid is ~725 lines. More
  components, more failure modes, more prompts to maintain.
- **Predictable LLM call count.** The old Hybrid always made exactly
  `1 + n + (p × n) + 1 = 8` calls (with defaults). The new Hybrid makes a variable
  number depending on task complexity and worker loop lengths.

### What You Gain (vs. Old Hybrid)

- **Genuine task decomposition.** Workers do different things, not the same thing with
  temperature diversity.
- **Worker autonomy.** Workers can search, reason, and iterate — not just produce one
  string.
- **Dynamic resource allocation.** Simple tasks get 1 worker; complex tasks get 3.
- **Information fidelity.** SharedMemory avoids the context-bloat and distortion of
  passing everything through conversation history.
- **Iterative re-planning.** The orchestrator adapts based on what workers actually
  found, rather than being locked into a fixed round structure.
- **Independent context windows.** Each worker's reasoning doesn't pollute other
  workers' context, enabling genuine parallel exploration of different aspects.

### Ablation Guide

To test whether specific components contribute to performance:

| Ablation | How | Tests |
|---|---|---|
| Remove re-planning | Set `max_replan_rounds=0` | Does iterative re-planning help? |
| Single worker | Set `max_workers=1` | Is decomposition better than single-agent? |
| Remove worker tools | Skip search dispatch in `AutonomousWorker.run()` | Does worker tool access matter? |
| Single-step workers | Set `max_worker_steps=1` | Is multi-turn autonomy better than single-shot? |
| Shared context | Replace SharedMemory with conversation passing | Does memory isolation help? |

---

## References

- Anthropic. *"How we built our multi-agent research system."* Anthropic Engineering Blog, June 2025.
  https://www.anthropic.com/engineering/multi-agent-research-system

- Kim, Y. et al. *"Towards a Science of Scaling Agent Systems."* arXiv:2512.08296, 2025, §3.1 / Table 2.