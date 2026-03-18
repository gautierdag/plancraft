"""
Hybrid MAS — Anthropic-style Orchestrator-Worker Pattern for Plancraft.

Inspired by Anthropic's "How we built our multi-agent research system" (2025)
and Kim et al., "Towards a Science of Scaling Agent Systems" (2025), §3.1.

Key differences from the naive Hybrid in copilot_multi.py:
  1. Workers are autonomous agentic loops (multi-turn with tool use), not
     single LLM calls.
  2. The orchestrator decomposes tasks and gives each worker a UNIQUE subtask
     with specific objectives, output format, and tool guidance.
  3. Dynamic scaling — the orchestrator decides how many workers to spawn
     based on task complexity analysis.
  4. External memory (scratchpad) — workers write findings to shared storage;
     the orchestrator reads lightweight summaries instead of passing everything
     through conversation history (avoids "game of telephone").
  5. Iterative re-planning — the orchestrator can spawn additional workers
     after reviewing initial findings.
  6. Workers have tool access (oracle search) and can make multiple calls
     in their own agentic loops.

Architecture:
    A = {a_orch, a_1, ..., a_n}   (n determined dynamically)
    C = star topology: orchestrator ↔ workers via shared memory
    Ω = hierarchical with iterative re-planning

    Per step:
      Phase 1: Orchestrator analyses observation → produces a plan with
               subtasks and complexity estimate.
      Phase 2: Autonomous workers execute subtasks in parallel, each with
               its own agentic loop (up to max_worker_steps tool calls).
      Phase 3: Orchestrator reviews worker findings from memory, decides
               whether to re-plan or synthesise a final action.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field

from copilot import CopilotClient

from .copilot_base import CopilotBaseAgent
from .copilot_llm import call_copilot_with_retry

# ═══════════════════════════════════════════════════════════════════════════════
# System prompts — each role gets a distinct prompt
# ═══════════════════════════════════════════════════════════════════════════════

ORCHESTRATOR_PLAN_PROMPT = """\
You are the ORCHESTRATOR of a Minecraft crafting multi-agent system. Your job \
is to analyse the current game state and decompose the crafting task into \
subtasks for specialised worker agents.

## Your Responsibilities
1. Analyse the inventory and target item.
2. Determine task complexity (simple / moderate / complex).
3. Decompose the work into subtasks, each with a clear objective.
4. Assign each subtask to a worker with specific instructions.

## Complexity Guidelines
- **simple** (1 worker): Direct craft — ingredients already in inventory, \
recipe is well-known (e.g. planks from logs).
- **moderate** (2 workers): Need to look up a recipe AND plan grid placement, \
OR need one intermediate crafting step.
- **complex** (3 workers): Multi-step crafting chain, recipe unknown, or \
inventory analysis is non-trivial.

## Output Format
You MUST respond with valid JSON and nothing else:
{
  "complexity": "simple" | "moderate" | "complex",
  "analysis": "<brief situation analysis>",
  "subtasks": [
    {
      "id": 1,
      "role": "<worker role name>",
      "objective": "<what this worker must accomplish>",
      "instructions": "<specific step-by-step instructions>",
      "tools_hint": "<which tools to use: search | grid_placement | inventory_check>"
    }
  ]
}
"""

ORCHESTRATOR_REPLAN_PROMPT = """\
You are the ORCHESTRATOR reviewing findings from your worker agents. Based on \
their results, decide whether you have enough information to produce a final \
action, or whether you need additional workers.

## Worker Findings
{findings}

## Original Plan
{plan}

## Output Format
Respond with valid JSON and nothing else:
{{
  "ready": true | false,
  "reasoning": "<why you are or aren't ready>",
  "additional_subtasks": [
    {{
      "id": <next_id>,
      "role": "<worker role name>",
      "objective": "<what this worker must accomplish>",
      "instructions": "<specific instructions>",
      "tools_hint": "<search | grid_placement | inventory_check>"
    }}
  ]
}}

If "ready" is true, "additional_subtasks" should be an empty list [].
"""

ORCHESTRATOR_SYNTHESIZE_PROMPT = """\
You are the ORCHESTRATOR of a Minecraft crafting multi-agent system. Your \
workers have completed their subtasks. Using their findings, produce the \
single best action to take right now.

## Worker Findings
{findings}

## Available Actions
1. **move** – move items between slots
   `move: from [Source] to [Target] with quantity N`
2. **smelt** – smelt an item (e.g. ores → ingots)
   `smelt: from [Source] to [Target] with quantity N`
3. **impossible** – declare the task impossible
   `impossible: <reason>`
4. **search** – look up how to craft an item
   `search: <item_name>`

## Slot names
- Crafting grid: [A1] [A2] [A3] / [B1] [B2] [B3] / [C1] [C2] [C3]
- Crafting output: [0]
- Inventory: [I1] through [I36]

## Important
- Respond with ONLY the action, nothing else.
- Use exact slot names like [I1], [A1], [0], etc.
- Quantities must be between 1 and 64.
- If workers found a recipe, place items in the EXACT grid positions.
- If multiple steps are needed, output only the NEXT single step.
"""

WORKER_PROMPT_TEMPLATE = """\
You are a specialised WORKER agent in a Minecraft crafting system. You have \
been assigned a specific subtask by the orchestrator.

## Your Assignment
- **Role**: {role}
- **Objective**: {objective}
- **Instructions**: {instructions}

## Available Tools
You can use these tools by outputting the corresponding command:

1. **search** – look up how to craft an item
   `search: <item_name>`
   Use this when you need to discover a recipe.

2. **analyse** – reason about the current state
   `analyse: <your analysis>`
   Use this to record your reasoning about inventory, grid placement, etc.

3. **recommend** – propose an action for the orchestrator
   `recommend: <action>`
   Use this when you've determined what action should be taken.

## Current Game State
{observation}

## Instructions
- Work through your assignment step by step.
- Use `search:` if you need recipe information.
- After gathering information, use `recommend:` to propose your action.
- If your task is impossible, use `recommend: impossible: <reason>`.
- Be concise and precise.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Shared Memory — external scratchpad workers write to, orchestrator reads from
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class WorkerFinding:
    """A single finding written by a worker to shared memory."""

    worker_id: int
    role: str
    objective: str
    status: str  # "success" | "partial" | "failed"
    finding: str  # the compressed output
    recommendations: list[str] = field(default_factory=list)
    search_results: list[str] = field(default_factory=list)
    steps_taken: int = 0


class SharedMemory:
    """
    External memory store that decouples worker output from conversation
    history. Workers write findings here; the orchestrator reads summaries.
    Avoids the 'game of telephone' problem.
    """

    def __init__(self):
        self.findings: list[WorkerFinding] = []
        self.plan: dict | None = None

    def store_plan(self, plan: dict):
        self.plan = plan

    def store_finding(self, finding: WorkerFinding):
        self.findings.append(finding)

    def get_findings_summary(self) -> str:
        """Compressed summary for the orchestrator — not raw conversation."""
        if not self.findings:
            return "No findings yet."
        parts = []
        for f in self.findings:
            part = (
                f"[Worker {f.worker_id} — {f.role}]\n"
                f"  Objective: {f.objective}\n"
                f"  Status: {f.status}\n"
                f"  Steps taken: {f.steps_taken}\n"
                f"  Finding: {f.finding}"
            )
            if f.search_results:
                part += "\n  Recipes found:\n"
                for sr in f.search_results:
                    part += f"    {sr}\n"
            if f.recommendations:
                part += "  Recommendations:\n"
                for rec in f.recommendations:
                    part += f"    → {rec}\n"
            parts.append(part)
        return "\n\n".join(parts)

    def get_plan_summary(self) -> str:
        if self.plan is None:
            return "No plan."
        return json.dumps(self.plan, indent=2)

    def clear(self):
        self.findings = []
        self.plan = None


# ═══════════════════════════════════════════════════════════════════════════════
# Autonomous Worker — multi-turn agentic loop with tool access
# ═══════════════════════════════════════════════════════════════════════════════


class AutonomousWorker:
    """
    A worker that runs its own agentic loop: it can call tools (search),
    reason across multiple turns, and write compressed findings to shared
    memory. Each worker has its own independent context window.

    This is NOT a single LLM call. It loops up to max_steps times,
    processing tool results and refining its output.
    """

    def __init__(
        self,
        worker_id: int,
        subtask: dict,
        client: CopilotClient,
        model_name: str,
        oracle_search_fn,
        shared_memory: SharedMemory,
        max_steps: int = 5,
    ):
        self.worker_id = worker_id
        self.subtask = subtask
        self.client = client
        self.model_name = model_name
        self.oracle_search = oracle_search_fn
        self.memory = shared_memory
        self.max_steps = max_steps
        # Each worker has its OWN context — independent from all others
        self.context: list[dict] = []
        self.search_results: list[str] = []
        self.recommendations: list[str] = []

    def _log(self, msg: str):
        print(f"  [Worker {self.worker_id} ({self.subtask.get('role', '?')})] {msg}")

    async def run(self, observation: str) -> WorkerFinding:
        """
        Execute the worker's agentic loop. Returns a WorkerFinding written
        to shared memory.
        """
        # Build the worker's unique system prompt from its subtask
        system_prompt = WORKER_PROMPT_TEMPLATE.format(
            role=self.subtask.get("role", "general"),
            objective=self.subtask.get("objective", "Complete the assigned task"),
            instructions=self.subtask.get("instructions", "Follow your best judgement"),
            observation=observation,
        )

        # Seed the worker's own context with the observation
        self.context = [
            {"role": "user", "content": observation},
        ]

        steps_taken = 0
        final_output = ""
        status = "partial"

        for step in range(self.max_steps):
            steps_taken += 1

            try:
                response = await call_copilot_with_retry(
                    self.client,
                    self.model_name,
                    self.context,
                    system_prompt,
                    temperature=0.3,
                )
            except Exception as e:
                self._log(f"Step {step} failed: {e}")
                status = "failed"
                final_output = f"Worker error: {e}"
                break

            self._log(f"Step {step}: {response[:120]}...")
            self.context.append({"role": "model", "content": response})

            # ── Tool dispatch: search ──────────────────────────────────────
            if "search:" in response.lower():
                search_match = re.search(r"search:\s*(\S+)", response)
                if search_match:
                    target = search_match.group(1).strip().lower()
                    search_result = self.oracle_search(f"search: {target}")
                    self.search_results.append(search_result)
                    self._log(f"  🔍 search: {target}")
                    self.context.append({"role": "user", "content": search_result})
                    continue  # next step in the agentic loop

            # ── Tool dispatch: recommend (terminal) ────────────────────────
            if "recommend:" in response.lower():
                rec_match = re.search(
                    r"recommend:\s*(.+)", response, re.IGNORECASE | re.DOTALL
                )
                if rec_match:
                    recommendation = rec_match.group(1).strip()
                    self.recommendations.append(recommendation)
                    final_output = recommendation
                    status = "success"
                    self._log(f"  ✅ recommend: {recommendation}")
                    break

            # ── Tool dispatch: analyse (non-terminal, adds to context) ─────
            if "analyse:" in response.lower() or "analyze:" in response.lower():
                # The analysis is already in the response and context.
                # Prompt the worker to continue toward a recommendation.
                self.context.append(
                    {
                        "role": "user",
                        "content": (
                            "Good analysis. Now, based on your findings, either "
                            "use `search: <item>` to look up a recipe, or "
                            "use `recommend: <action>` to propose your final action."
                        ),
                    }
                )
                continue

            # ── No tool matched — treat as implicit analysis, nudge worker ─
            # If the worker just produced free text, push it toward a
            # recommendation so it doesn't loop forever.
            if step < self.max_steps - 1:
                self.context.append(
                    {
                        "role": "user",
                        "content": (
                            "Please output a concrete recommendation using "
                            "`recommend: <action>` based on your analysis so far."
                        ),
                    }
                )
            else:
                # Last step — take whatever we have
                final_output = response
                status = "partial"

        # Write finding to shared memory (not conversation history)
        finding = WorkerFinding(
            worker_id=self.worker_id,
            role=self.subtask.get("role", "general"),
            objective=self.subtask.get("objective", ""),
            status=status,
            finding=final_output,
            recommendations=list(self.recommendations),
            search_results=list(self.search_results),
            steps_taken=steps_taken,
        )
        self.memory.store_finding(finding)
        return finding


# ═══════════════════════════════════════════════════════════════════════════════
# CopilotHybridAgent — the full orchestrator-worker system
# ═══════════════════════════════════════════════════════════════════════════════

COMPLEXITY_TO_WORKERS = {
    "simple": 1,
    "moderate": 2,
    "complex": 3,
}


class CopilotHybridAgent(CopilotBaseAgent):
    """
    Anthropic-style Orchestrator-Worker Hybrid Agent.

    Implements the full pattern:
      1. Orchestrator analyses → decomposes into specialised subtasks
      2. Autonomous workers execute subtasks in parallel (multi-turn loops)
      3. Workers write compressed findings to shared memory
      4. Orchestrator reviews findings, optionally re-plans with more workers
      5. Orchestrator synthesises a single action from all findings

    Architecture:
        A = {a_orch, a_1, ..., a_n}   (n chosen dynamically)
        C = star via shared memory     (no game of telephone)
        Ω = hierarchical + iterative re-planning

    Complexity per step:
        O(k_orch) + O(n * max_worker_steps * k) + O(k_synth)
        where k = cost per LLM call, n = dynamic worker count
    """

    def __init__(
        self,
        model_name: str,
        client: CopilotClient,
        max_workers: int = 3,
        max_worker_steps: int = 5,
        max_replan_rounds: int = 1,
    ):
        super().__init__(model_name, client)
        self.max_workers = max_workers
        self.max_worker_steps = max_worker_steps
        self.max_replan_rounds = max_replan_rounds
        self.shared_memory = SharedMemory()

    def reset(self, example_id: str, target: str):
        """Reset agent state and shared memory for a new episode."""
        super().reset(example_id, target)
        self.shared_memory.clear()

    # ── Phase 1: Orchestrator plans and decomposes ─────────────────────────

    async def _orchestrator_plan(self, context: list[dict]) -> dict:
        """
        Orchestrator analyses the current state and produces a structured
        plan with subtasks. Returns the parsed plan dict.
        """
        raw = await call_copilot_with_retry(
            self.client,
            self.model_name,
            context,
            ORCHESTRATOR_PLAN_PROMPT,
            temperature=0.0,
        )
        self.log(f"Orchestrator plan (raw): {raw[:200]}...")

        # Parse JSON — be resilient to markdown fences
        plan = self._parse_json(raw)

        if plan is None:
            # Fallback: single worker with generic instructions
            self.log("Failed to parse orchestrator plan, using fallback")
            plan = {
                "complexity": "simple",
                "analysis": "Could not parse plan. Falling back to single worker.",
                "subtasks": [
                    {
                        "id": 1,
                        "role": "general_crafter",
                        "objective": "Determine the correct action",
                        "instructions": (
                            "Look up the recipe if needed, then recommend the "
                            "correct move/smelt/impossible action."
                        ),
                        "tools_hint": "search",
                    }
                ],
            }

        self.shared_memory.store_plan(plan)
        return plan

    # ── Phase 2: Spawn autonomous workers ──────────────────────────────────

    async def _spawn_workers(
        self, subtasks: list[dict], observation: str, start_id: int = 1
    ) -> list[WorkerFinding]:
        """
        Spawn autonomous workers in parallel, one per subtask. Each worker
        runs its own multi-turn agentic loop and writes findings to shared
        memory.
        """
        workers = []
        for i, subtask in enumerate(subtasks):
            worker = AutonomousWorker(
                worker_id=start_id + i,
                subtask=subtask,
                client=self.client,
                model_name=self.model_name,
                oracle_search_fn=self._oracle_search,
                shared_memory=self.shared_memory,
                max_steps=self.max_worker_steps,
            )
            workers.append(worker)

        self.log(f"Spawning {len(workers)} autonomous worker(s)")

        # Run all workers in parallel — each has its own context window
        results = await asyncio.gather(
            *[w.run(observation) for w in workers],
            return_exceptions=True,
        )

        findings = []
        for r in results:
            if isinstance(r, Exception):
                self.log(f"Worker failed with exception: {r}")
            elif isinstance(r, WorkerFinding):
                findings.append(r)

        return findings

    # ── Phase 3: Orchestrator reviews and optionally re-plans ──────────────

    async def _orchestrator_replan(self, context: list[dict]) -> dict | None:
        """
        Orchestrator reviews worker findings and decides whether to spawn
        more workers or proceed to synthesis.
        """
        findings_text = self.shared_memory.get_findings_summary()
        plan_text = self.shared_memory.get_plan_summary()

        prompt = ORCHESTRATOR_REPLAN_PROMPT.format(
            findings=findings_text,
            plan=plan_text,
        )

        raw = await call_copilot_with_retry(
            self.client,
            self.model_name,
            context,
            prompt,
            temperature=0.0,
        )
        self.log(f"Orchestrator replan (raw): {raw[:200]}...")

        result = self._parse_json(raw)
        return result

    # ── Phase 4: Orchestrator synthesises final action ─────────────────────

    async def _orchestrator_synthesize(self, context: list[dict]) -> str:
        """
        Orchestrator reads all findings from shared memory and produces
        the single best action.
        """
        findings_text = self.shared_memory.get_findings_summary()

        prompt = ORCHESTRATOR_SYNTHESIZE_PROMPT.format(
            findings=findings_text,
        )

        action = await call_copilot_with_retry(
            self.client,
            self.model_name,
            context,
            prompt,
            temperature=0.0,
        )
        self.log(f"Orchestrator synthesised action: {action}")
        return action

    # ── Main act() — ties all phases together ──────────────────────────────

    async def act(self, observation_text: str | None) -> str:
        if observation_text:
            self.conversation.append({"role": "user", "content": observation_text})

        # Clear per-step memory (findings from previous step are stale)
        self.shared_memory.clear()

        context = list(self.conversation)
        observation = observation_text or (
            self.conversation[-1]["content"] if self.conversation else ""
        )

        # ── Phase 1: Plan ──────────────────────────────────────────────────
        plan = await self._orchestrator_plan(context)
        subtasks = plan.get("subtasks", [])
        complexity = plan.get("complexity", "simple")

        # Dynamic scaling: cap workers based on complexity
        max_for_complexity = COMPLEXITY_TO_WORKERS.get(complexity, 1)
        max_allowed = min(max_for_complexity, self.max_workers)
        subtasks = subtasks[:max_allowed]

        if not subtasks:
            subtasks = [
                {
                    "id": 1,
                    "role": "general_crafter",
                    "objective": "Determine the correct action",
                    "instructions": "Analyse the state and recommend an action.",
                    "tools_hint": "search",
                }
            ]

        self.log(
            f"Plan: complexity={complexity}, "
            f"workers={len(subtasks)}, "
            f"analysis={plan.get('analysis', 'N/A')[:100]}"
        )

        # ── Phase 2: Spawn workers ─────────────────────────────────────────
        findings = await self._spawn_workers(subtasks, observation)

        if not findings:
            self.log("All workers failed — returning impossible")
            action = "impossible: All workers failed to produce findings"
            self.conversation.append({"role": "model", "content": action})
            return action

        # ── Phase 3: Re-plan if needed ─────────────────────────────────────
        next_worker_id = len(subtasks) + 1
        for replan_round in range(self.max_replan_rounds):
            replan_result = await self._orchestrator_replan(context)

            if replan_result is None:
                self.log(f"Replan round {replan_round}: failed to parse, proceeding")
                break

            if replan_result.get("ready", True):
                self.log(
                    f"Replan round {replan_round}: ready to synthesise. "
                    f"Reason: {replan_result.get('reasoning', 'N/A')[:100]}"
                )
                break

            additional = replan_result.get("additional_subtasks", [])
            if not additional:
                self.log(f"Replan round {replan_round}: not ready but no new subtasks")
                break

            # Cap additional workers
            remaining_budget = self.max_workers - len(self.shared_memory.findings)
            additional = additional[: max(0, remaining_budget)]

            if additional:
                self.log(
                    f"Replan round {replan_round}: spawning {len(additional)} "
                    f"additional worker(s)"
                )
                new_findings = await self._spawn_workers(
                    additional, observation, start_id=next_worker_id
                )
                next_worker_id += len(additional)
                findings.extend(new_findings)
            else:
                self.log(f"Replan round {replan_round}: worker budget exhausted")
                break

        # ── Phase 4: Synthesise ────────────────────────────────────────────
        action = await self._orchestrator_synthesize(context)

        # Handle search action from synthesis
        if "search:" in action.lower():
            search_result = self._oracle_search(action)
            if search_result:
                self.log(f"🔍 Orchestrator searched: {action}")
                self.conversation.append({"role": "model", "content": action})
                self.conversation.append({"role": "user", "content": search_result})
                return await self.act(None)

        self.conversation.append({"role": "model", "content": action})
        return action

    # ── Utilities ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """
        Attempt to parse JSON from LLM output. Handles markdown code fences
        and other common LLM formatting quirks.
        """
        # Try direct parse first
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try extracting from markdown code fences
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1).strip())
            except (json.JSONDecodeError, TypeError):
                pass

        # Try finding first { ... } block
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except (json.JSONDecodeError, TypeError):
                pass

        return None
