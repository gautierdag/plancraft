"""
Multi-Agent System (MAS) architectures for Plancraft, implemented according to:

  "Towards a Science of Scaling Agent Systems"
  Kim et al., 2025 (arXiv:2512.08296), Section 3.1 / Table 2.

Five canonical architectures are defined in the paper:
  - Single-Agent System (SAS)       → copilot_single.py
  - Independent MAS                 → CopilotIndependentAgent
  - Centralized MAS                 → CopilotCentralizedAgent
  - Decentralized MAS               → CopilotDecentralizedAgent
  - Hybrid MAS                      → CopilotHybridAgent

Communication topology (C) and orchestration policy (Ω) for each:

  Independent:   C = {(ai, aagg)},              Ω = synthesis_only
  Centralized:   C = {(aorch, ai) : ∀i},        Ω = hierarchical
  Decentralized: C = {(ai, aj) : ∀i,j, i≠j},   Ω = consensus
  Hybrid:        C = star + peer edges,          Ω = hierarchical + lateral
"""

import asyncio
import logging
from collections import Counter
from copilot import CopilotClient
from .copilot_base import CopilotBaseAgent
from .copilot_llm import call_copilot_with_retry

SYSTEM_PROMPT = """\
You are a Minecraft crafting agent. Your goal is to craft a target item by \
manipulating items in your inventory using a 3×3 crafting grid.

## Actions
Respond with EXACTLY ONE action per turn (no extra text):

1. **move** – move items between slots
   `move: from [Source] to [Target] with quantity N`

2. **smelt** – smelt an item (e.g. ores → ingots)
   `smelt: from [Source] to [Target] with quantity N`

3. **impossible** – declare the task impossible with current inventory
   `impossible: <reason>`

4. **search** – look up how to craft an item  
   `search: <item_name>`

## Slot names
- Crafting grid (3×3):
    [A1] [A2] [A3]
    [B1] [B2] [B3]
    [C1] [C2] [C3]
- Crafting output: [0]  (crafted items appear here)
- Inventory: [I1] through [I36]

## Strategy
- First, if you don't know the recipe, use `search: <target_item>` to look it up.
- Then place the required items in the crafting grid.
- Finally, move the crafted item from [0] to an inventory slot.
- If the inventory lacks the required materials, declare `impossible: <reason>`.
- You may need multi-step crafting (e.g. logs → planks → sticks).

## Important
- Respond with ONLY the action, nothing else.
- Use exact slot names like [I1], [A1], [0], etc.
- Quantities must be between 1 and 64.
"""


class CopilotIndependentAgent(CopilotBaseAgent):
    """
    Independent MAS — Paper §3.1:
        A = {a1, ..., an}
        C = {(ai, aagg)}           (agent-to-aggregator only, no peer communication)
        Ω = synthesis_only

    The synthesis_only policy concatenates all n sub-agent outputs into a single
    context, then makes ONE final LLM call to synthesise a concrete action.
    There is NO cross-validation or majority voting — differences in performance
    arise purely from parallel exploration (ensemble-style reasoning).

    Complexity: O(nk) + O(1) aggregation call.
    """

    def __init__(self, model_name: str, client: CopilotClient, num_agents: int = 3):
        super().__init__(model_name, client)
        self.num_agents = num_agents

    async def act(self, observation_text: str) -> str:
        if observation_text:
            self.conversation.append({"role": "user", "content": observation_text})

        # ── Phase 1: n agents explore in parallel (no communication) ──────────
        async def _call_agent(i: int) -> str | None:
            try:
                return await call_copilot_with_retry(
                    self.client,
                    self.model_name,
                    self.conversation,
                    SYSTEM_PROMPT,
                    temperature=0.7,
                )
            except Exception as e:
                self.log(f"Agent {i} failed: {e}")
                return None

        results = await asyncio.gather(*[_call_agent(i) for i in range(self.num_agents)])
        proposals = [r for r in results if r is not None]

        if not proposals:
            return "impossible: All agents failed"

        # ── Phase 2: synthesis_only aggregation — concatenate, then one LLM call ─
        # The aggregator receives ALL proposals as context and produces a single
        # synthesised action.  No voting, no comparison, no error-correction.
        synthesis_context = "\n\n".join(
            [f"[Agent {i+1} proposal]\n{p}" for i, p in enumerate(proposals)]
        )
        aggregation_prompt = (
            SYSTEM_PROMPT
            + "\n\n[Aggregator]\nThe following proposals were made by parallel agents "
            "exploring the same state. Synthesise them into a single concrete action. "
            "Output ONLY the final action.\n\n"
            + synthesis_context
        )

        synthesis_msgs = self.conversation + [
            {"role": "user", "content": "Synthesise the proposals above into one action."}
        ]

        action = await call_copilot_with_retry(
            self.client,
            self.model_name,
            synthesis_msgs,
            aggregation_prompt,
            temperature=0.0,
        )

        if "search:" in action.lower():
            search_result = self._oracle_search(action)
            if search_result:
                self.conversation.append({"role": "model", "content": action})
                self.conversation.append({"role": "user", "content": search_result})
                return await self.act(None)

        self.conversation.append({"role": "model", "content": action})
        return action


class CopilotCentralizedAgent(CopilotBaseAgent):
    """
    Centralized MAS — Paper §3.1:
        A = {aorch, a1, ..., an}
        C = {(aorch, ai) : ∀i}    (orchestrator-to-all-agents, no peer communication)
        Ω = hierarchical

    A single orchestrator coordinates r rounds.  In each round:
      1. The orchestrator generates a directive (planning step).
      2. n workers receive the directive in parallel and propose actions.
      3. The orchestrator synthesises the worker outputs.
    The final synthesis after round r is the returned action.

    Complexity: O(rnk) + O(r) orchestrator calls.
    """

    def __init__(
        self,
        model_name: str,
        client: CopilotClient,
        num_agents: int = 3,
        rounds: int = 1,
    ):
        super().__init__(model_name, client)
        self.num_agents = num_agents
        self.rounds = rounds

    async def act(self, observation_text: str) -> str:
        if observation_text:
            self.conversation.append({"role": "user", "content": observation_text})

        context = list(self.conversation)  # working copy for this step

        final_action = "impossible: Centralized agent produced no output"

        for r in range(self.rounds):
            # ── Orchestrator directive ─────────────────────────────────────────
            orch_prompt = (
                SYSTEM_PROMPT
                + "\n\n[Orchestrator] Analyse the current state and issue a precise "
                "directive to your worker agents. Describe WHAT to do and WHY. "
                "Do NOT output the final action yet."
            )
            directive = await call_copilot_with_retry(
                self.client, self.model_name, context, orch_prompt
            )
            self.log(f"Round {r} | Orchestrator directive: {directive}")

            # ── n workers receive directive in parallel ────────────────────────
            worker_msgs = context + [
                {"role": "model", "content": directive},
                {"role": "user", "content": "Worker: execute the directive. Output ONE action."},
            ]

            worker_tasks = [
                call_copilot_with_retry(
                    self.client, self.model_name, worker_msgs, SYSTEM_PROMPT, temperature=0.7
                )
                for _ in range(self.num_agents)
            ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)
            worker_outputs = [o for o in results if not isinstance(o, Exception)]

            if not worker_outputs:
                self.log(f"Round {r} | All workers failed")
                break

            self.log(f"Round {r} | Worker outputs: {worker_outputs}")

            # ── Orchestrator synthesises worker outputs ────────────────────────
            worker_summary = "\n".join(
                [f"- Worker {i+1}: {o}" for i, o in enumerate(worker_outputs)]
            )
            synthesis_msgs = worker_msgs + [
                {"role": "model", "content": worker_summary},
                {
                    "role": "user",
                    "content": (
                        "Orchestrator: review the worker proposals above and output "
                        "the single best action."
                    ),
                },
            ]
            final_action = await call_copilot_with_retry(
                self.client, self.model_name, synthesis_msgs, SYSTEM_PROMPT
            )
            self.log(f"Round {r} | Orchestrator final: {final_action}")

            # Update context for next round
            context = synthesis_msgs + [{"role": "model", "content": final_action}]

        if "search:" in final_action.lower():
            search_result = self._oracle_search(final_action)
            if search_result:
                self.conversation.append({"role": "model", "content": final_action})
                self.conversation.append({"role": "user", "content": search_result})
                return await self.act(None)

        self.conversation.append({"role": "model", "content": final_action})
        return final_action


class CopilotDecentralizedAgent(CopilotBaseAgent):
    """
    Decentralized MAS — Paper §3.1:
        A = {a1, ..., an}
        C = {(ai, aj) : ∀i,j, i≠j}   (all-to-all peer communication)
        Ω = consensus

    Agents communicate in d sequential debate rounds.  Each agent sees all
    peers' proposals and updates its own position.  After d rounds a majority
    vote produces the consensus action.

    Complexity: O(dnk) + O(1).  Memory: O(dnk) per agent.
    """

    def __init__(
        self,
        model_name: str,
        client: CopilotClient,
        num_agents: int = 3,
        rounds: int = 2,
    ):
        super().__init__(model_name, client)
        self.num_agents = num_agents
        self.rounds = rounds

    async def act(self, observation_text: str) -> str:
        if observation_text:
            self.conversation.append({"role": "user", "content": observation_text})

        # ── Initial proposals: all agents in parallel ─────────────────────────
        initial_tasks = [
            call_copilot_with_retry(
                self.client, self.model_name, self.conversation, SYSTEM_PROMPT, temperature=0.7
            )
            for _ in range(self.num_agents)
        ]
        results = await asyncio.gather(*initial_tasks, return_exceptions=True)
        proposals = [p for p in results if not isinstance(p, Exception)]

        if not proposals:
            return "impossible: All agents failed"

        # ── d debate rounds: all-to-all peer exchange ─────────────────────────
        # Each agent receives ALL peers' proposals, then updates its own position.
        # All agents within each round run in parallel.
        for d in range(self.rounds):
            self.log(f"Debate round {d} | Proposals: {proposals}")

            peer_summary = "Other agents proposed:\n" + "\n".join(
                [f"- Agent {j+1}: {p}" for j, p in enumerate(proposals)]
            )
            debate_question = peer_summary + "\nGiven these proposals, output your updated action."

            debate_tasks = [
                call_copilot_with_retry(
                    self.client,
                    self.model_name,
                    self.conversation + [{"role": "user", "content": debate_question}],
                    SYSTEM_PROMPT,
                    temperature=0.7,
                )
                for _ in range(self.num_agents)
            ]
            results = await asyncio.gather(*debate_tasks, return_exceptions=True)
            updated = [p for p in results if not isinstance(p, Exception)]
            if updated:
                proposals = updated

        # ── Consensus: majority vote after d rounds ───────────────────────────
        if not proposals:
            return "impossible: All agents failed during debate"

        counts = Counter(proposals)
        best_action, _ = counts.most_common(1)[0]
        self.log(f"Consensus: {dict(counts)} -> {best_action}")

        if "search:" in best_action.lower():
            search_result = self._oracle_search(best_action)
            if search_result:
                self.conversation.append({"role": "model", "content": best_action})
                self.conversation.append({"role": "user", "content": search_result})
                return await self.act(None)

        self.conversation.append({"role": "model", "content": best_action})
        return best_action


class CopilotHybridAgent(CopilotBaseAgent):
    """
    Hybrid MAS — Paper §3.1:
        A = {aorch, a1, ..., an}
        C = star + peer edges      (orchestrator→workers AND worker↔worker)
        Ω = hierarchical + lateral

    Combines orchestrated hierarchy (r orchestrator rounds) with limited
    peer communication (p lateral rounds between workers).

    Per round:
      1. Orchestrator issues a directive.
      2. Workers propose in parallel (hierarchical, star topology).
      3. Workers share proposals with each other for p peer rounds (lateral).
      4. Orchestrator synthesises the final peer-refined proposals.

    Complexity: O(rnk + pn) per step.
    """

    def __init__(
        self,
        model_name: str,
        client: CopilotClient,
        num_agents: int = 3,
        rounds: int = 1,
        peer_rounds: int = 1,
    ):
        super().__init__(model_name, client)
        self.num_agents = num_agents
        self.rounds = rounds
        self.peer_rounds = peer_rounds

    async def act(self, observation_text: str) -> str:
        if observation_text:
            self.conversation.append({"role": "user", "content": observation_text})

        context = list(self.conversation)
        final_action = "impossible: Hybrid agent produced no output"

        for r in range(self.rounds):
            # ── Step 1: Orchestrator directive (hierarchical control) ──────────
            orch_prompt = (
                SYSTEM_PROMPT
                + "\n\n[Orchestrator] Analyse the situation and issue a directive "
                "to your workers. Do NOT output the final action yet."
            )
            directive = await call_copilot_with_retry(
                self.client, self.model_name, context, orch_prompt
            )
            self.log(f"Round {r} | Orchestrator directive: {directive}")

            # ── Step 2: Workers propose in parallel (star edges) ──────────────
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
            proposals = [a for a in results if not isinstance(a, Exception)]

            if not proposals:
                self.log(f"Round {r} | All workers failed")
                break

            self.log(f"Round {r} | Worker proposals: {proposals}")

            # ── Step 3: p lateral peer rounds (peer edges) ────────────────────
            # Workers share and refine proposals with each other before
            # the orchestrator makes its final synthesis.
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
                        self.client,
                        self.model_name,
                        worker_msgs + [{"role": "user", "content": peer_question}],
                        SYSTEM_PROMPT,
                        temperature=0.7,
                    )
                    for _ in range(self.num_agents)
                ]
                results = await asyncio.gather(*peer_tasks, return_exceptions=True)
                refined = [a for a in results if not isinstance(a, Exception)]
                if refined:
                    proposals = refined
                self.log(f"Round {r} | Peer round {p} refined: {proposals}")

            # ── Step 4: Orchestrator synthesises peer-refined proposals ────────
            final_summary = "\n".join(
                [f"- Worker {i+1}: {prop}" for i, prop in enumerate(proposals)]
            )
            synthesis_msgs = worker_msgs + [
                {"role": "model", "content": final_summary},
                {
                    "role": "user",
                    "content": (
                        "Orchestrator: workers have exchanged proposals. "
                        "Select the single best action."
                    ),
                },
            ]
            final_action = await call_copilot_with_retry(
                self.client, self.model_name, synthesis_msgs, SYSTEM_PROMPT
            )
            self.log(f"Round {r} | Orchestrator final: {final_action}")

            context = synthesis_msgs + [{"role": "model", "content": final_action}]

        if "search:" in final_action.lower():
            search_result = self._oracle_search(final_action)
            if search_result:
                self.conversation.append({"role": "model", "content": final_action})
                self.conversation.append({"role": "user", "content": search_result})
                return await self.act(None)

        self.conversation.append({"role": "model", "content": final_action})
        return final_action
