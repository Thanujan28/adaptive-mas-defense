# Remediation Design (Containment Layer)

**Status:** design only — no implementation code yet. This document is shown
before any code is written, per the STOP-AND-ASK rule.

**Scope:** a NEW containment/remediation layer that *acts* on a
judge-confirmed contradiction. Everything built so far (Tier 1 chunked
semantic, Tier 2 NLI, Tier 3 LLM judge / `security/observer.py`,
`security/contradiction_checker.py`, `security/llm_judge.py`) only
detects and flags. This is the first change that **mutates pipeline
execution**, so it is gated behind its own feature flag and fails closed.

**Terminology (fixed by the task, do not deviate):**
- **detective agent** — the component that inspects a confirmed
  contradicting chunk and decides whether its problem is attributable to
  a specific upstream artifact.
- **root-cause tracer** — the component that walks backward one hop at a
  time to find the artifact that is actually corrupt.
- We deliberately do NOT call the tracer a "back-prop rerunner": that
  name collides with ML backpropagation. The tracer has nothing to do
  with gradients.

---

## 1. Where this layer sits

```
agents run (LangGraph)  ->  publish_agent_result()
                              |
                              v
                     _observe_agent_response()
                              |
                              v
             SecurityObserver.observe_tiered()      <-- EXISTING (unchanged)
                              |
                    Tier1 + Tier2 (+ gated Tier3)
                              |
             per-chunk ChunkDecision.tier3_verdict
                              |
        contradictions_evidence == True  (ONLY possible when Tier 3 ran)
                              |
                              v
        =================== NEW LAYER (this doc) ===================
                              |
        [1] detective agent  (security/detective_agent.py)
                              |
                 attributes to an upstream artifact?
                        /                 \
                       no                  yes
                       |                    |
              stop, log                  [2] tracer?  (detective decides)
              "not attributable"          /        \
                                        no          yes
                                        |            |
                                   stop, log    [3] root-cause tracer
                                               (security/root_cause_tracer.py)
                                                    |
                                        root_found? /        \  no (hop limit /
                                                   /          \  trusted root)
                                                  yes           |
                                                   |            |
                                       [4] remediation executor |
                                       (security/remediation.py)|
                                                   |            |
                                        RemediationPlan          stop, log
                                        (agents_to_rerun in      reason
                                         dependency order)
                                                   |
                                        [5] environment performs
                                            the re-run (discard +
                                            re-invoke affected
                                            nodes only)
        =============================================================
```

Two independent flags:
- `MAS_TIER3_JUDGE` (existing, default **off**): gates Tier 3 itself.
- `MAS_REMEDIATION_ENABLED` (new, default **off**): gates THIS layer.

The layer only ever activates on a chunk where Tier 3 actually ran AND
returned `contradicts_evidence=True`. **If `MAS_TIER3_JUDGE` is off, this
layer never activates** — it depends entirely on judge confirmation, not
on Tier 1/2 alone.

---

## 2. Flow, in words

1. **Activation.** For each chunk of an agent response, after
   `observe_tiered()` returns, look at `ChunkDecision.tier3_verdict`. If
   `tier3_verdict is not None and tier3_verdict.contradicts_evidence is
   True`, the chunk is *judge-confirmed*. If `MAS_REMEDIATION_ENABLED`
   is off, skip everything below.

2. **Budget proposal.** The observer proposes a **token ceiling** for
   investigating this chunk. This is a *named constant*
   (`DETECTIVE_TOKEN_BUDGET`) and it is a **proposal, not a spend** —
   nothing consumes it until the detective actually runs, and the
   detective is hard-capped at it.

3. **Detective agent.** Spawned with the proposed budget. It inspects
   the confirmed chunk and its linked evidence (reusing
   `security/evidence_linker.py`) and produces a structured
   `DetectiveFinding`:
   - `attributable: bool`
   - `source_artifact_id: str | None` (a tool-result `request_id`, or an
     `agent:<name>` tag for a prior agent's output)
   - `requests_tracer: bool`
   - `reasoning: str`
   - `tokens_used: int`
   The detective enforces its own budget: if its own LLM usage would
   exceed the ceiling, it stops and reports **partial findings**
   (e.g. `attributable=False` with a "budget exhausted" reason) rather
   than overspending.

4. **Tracer decision.** The tracer is **not automatic**. If the
   detective attributes the problem to a specific upstream source, it
   then decides — based on its own findings — whether tracing further
   back is warranted, and records that decision plus a stated reason.
   (A clear attribution to a raw tool result usually does NOT need
   tracing; an attribution to an intermediate agent output whose own
   evidence is not yet checked usually DOES.)

5. **Root-cause tracer.** If invoked, the tracer walks backward through
   the evidence chain **one hop at a time**, starting from the artifact
   the detective identified. At each hop it checks whether THAT
   artifact/prompt is itself contradictory or corrupted (reusing the
   contradiction checker on that artifact against its own upstream
   evidence). If yes → go back one more hop. If no → stop and report
   the current hop as the root cause.

6. **Hop limit — max 3 hops back from the point of detection.** A named,
   hard constant (`MAX_TRACE_HOPS = 3`). If reached without a clean
   source, stop and report `"root cause not identified within hop
   limit"`. Never treat the pipeline's original task or the
   coordinator's initial plan as suspect.

7. **Trusted root.** The original human task (`episode_state.task`) and
   the coordinator's initial plan are **NEVER** traced into and
   **NEVER** subject to removal/rerun. If tracing would go past the
   coordinator's plan, stop at the plan and report
   `"no identifiable source before trusted root"`.

8. **Remediation, once a root cause is found.** The identified artifact
   (a specific tool result keyed by `request_id`, or a specific prior
   agent's output) is discarded/excluded from what the affected
   agent(s) see. Only the agent(s) that directly consumed that artifact
   re-run — **not the whole episode**. If a downstream agent already ran
   on the now-discarded output, that agent **also** re-runs afterward:
   remediation **cascades forward** from the fix point, in dependency
   order. Total re-run cost (tokens, LLM calls) is tracked and
   reported.

9. **Attempt cap.** If remediation is triggered but the newly re-run
   agent's output STILL contains a contradiction after cleanup, do not
   retry forever. `MAX_REMEDIATION_ATTEMPTS = 2` per chunk, tracked
   externally (passed in and returned, never global state). If exceeded,
   report the chunk as **unresolved**.

---

## 3. New classes / functions and their signatures

### 3.1 `security/detective_agent.py`

```python
# Named constants (no inline magic numbers)
DETECTIVE_TOKEN_BUDGET = 1500          # per-chunk ceiling PROPOSED for investigation
DETECTIVE_LINK_TOP_K = 3               # how many linked evidence items to inspect

@dataclass
class DetectiveFinding:
    attributable: bool
    source_artifact_id: Optional[str]   # "req:<request_id>" | "agent:<name>" | None
    requests_tracer: bool
    reasoning: str
    tokens_used: int
    budget_exhausted: bool = False

class DetectiveAgent:
    def __init__(
        self,
        assessor: SemanticAssessor,          # shared encoder (reuses EvidenceLinker)
        llm: Optional[Any] = None,           # injectable; stub in tests
        token_budget: int = DETECTIVE_TOKEN_BUDGET,
        linker: Optional[EvidenceLinker] = None,
    ) -> None: ...

    def investigate(
        self,
        *,
        chunk_text: str,
        evidence_chunks: Sequence[str],
        evidence_artifacts: Sequence[Mapping[str, Any]],  # artifacts delivered to the agent
        upstream_responses: Mapping[str, str],            # {agent_name: response}
        task: str,
    ) -> DetectiveFinding: ...
```

`investigate()` builds evidence candidates with `EvidenceLinker`,
ranks them, asks the (injectable) LLM to attribute the chunk to a
candidate, and returns a `DetectiveFinding`. `tokens_used` is counted
from the LLM prompt/response; if it would exceed `token_budget`, the
method returns early with `budget_exhausted=True` and
`attributable=False`.

### 3.2 `security/root_cause_tracer.py`

```python
# HARD constants -- constructor defaults, never configurable per call.
MAX_TRACE_HOPS = 3        # max 3 hops back from the point of detection
TRUSTED_ROOT_LABEL = "trusted_root"   # task + coordinator plan; never traced into

@dataclass
class TraceResult:
    root_found: bool
    root_artifact_id: Optional[str]
    hops_taken: int
    stopped_reason: str   # "root_found" | "hop_limit_exceeded" |
                          # "reached_trusted_root" | "no_upstream_evidence"

class RootCauseTracer:
    def __init__(
        self,
        contradiction_checker: ContradictionChecker,
        evidence_lookup: Callable[[str], "ArtifactView | None"],
        max_hops: int = MAX_TRACE_HOPS,          # default only; caller may
        trusted_roots: frozenset[str] = ...,     # override by constructing,
    ) -> None: ...                               # NOT per call

    def trace(self, start_artifact_id: str) -> TraceResult: ...
```

`evidence_lookup(artifact_id)` returns an `ArtifactView` describing one
artifact and its own upstream evidence + provenance; the tracer calls it
one hop at a time. `trace()` keeps a **visited set** so a
branching/shared artifact that two paths converge on is never visited
twice (no infinite loop). It stops on the first hop whose artifact is
NOT itself contradictory (that hop is the root), on the hop limit, or
when the next hop is a trusted root.

### 3.3 `security/remediation.py`

```python
# Named constant
MAX_REMEDIATION_ATTEMPTS = 2   # per chunk/artifact; tracked EXTERNALLY

@dataclass
class RemediationPlan:
    agents_to_rerun: list[str]        # dependency order (upstream -> downstream)
    discarded_artifact_id: str
    attempt: int
    over_attempt_cap: bool = False

class RemediationExecutor:
    def __init__(
        self,
        consumer_graph: Mapping[str, Sequence[str]],  # agent -> agents it feeds
        consumer_index: Mapping[str, Sequence[str]],  # artifact_id -> agents that consumed it
    ) -> None: ...

    def plan(
        self,
        *,
        root_artifact_id: str,
        attempts: Mapping[str, int],   # external counter, passed IN
    ) -> tuple[RemediationPlan, dict[str, int]]: ...  # returns updated counter
```

`plan()` does NOT touch `MASEnvironment`. It:
1. Looks up every agent that consumed `root_artifact_id` (directly).
2. Adds every downstream agent reachable via `consumer_graph` (the
   cascade), in dependency order.
3. Reads the attempt counter for this artifact/chunk. If it is already
   `>= MAX_REMEDIATION_ATTEMPTS`, returns `over_attempt_cap=True` and an
   EMPTY `agents_to_rerun` (stop, report unresolved). Otherwise returns
   the plan and an incremented counter (the counter is *returned*, never
   mutated globally).

### 3.4 Environment wiring (`environment/mas_environment.py`) — gated

New named constant + comment in the `__init__` area, mirroring the
existing `MAS_TIER3_JUDGE` comment:

```python
# Remediation/containment is UNVALIDATED and MUTATES execution. It stays
# off unless MAS_REMEDIATION_ENABLED=1. It depends on Tier 3 CONFIRMING a
# contradiction, so it can only fire when MAS_TIER3_JUDGE=1 as well. Off
# by default because the re-run cost/behaviour has not been validated.
self.remediation_enabled = os.getenv("MAS_REMEDIATION_ENABLED", "0") == "1"
```

A new method (additive) on `MASEnvironment`:

```python
def _maybe_remediate(
    self,
    *,
    tiered: TieredObservation,
    agent_id: str,
) -> Optional[dict]:
    """Runs the detective -> tracer -> remediation flow for judge-confirmed
    chunks, returns a summary dict (or None). No-op when the flag is off."""
```

It is called from `_observe_agent_response` right after `observe_tiered`
returns. The re-run itself is a separate method (see §4).

### 3.5 Feature flag

| Flag                | Default | Gates                                            |
|---------------------|---------|--------------------------------------------------|
| `MAS_TIER3_JUDGE`   | off     | Tier 3 (existing)                                 |
| `MAS_REMEDIATION_ENABLED` | off | this whole layer (new, independent)          |

Both default OFF. Remediation requires Tier 3 *confirmation*, so
enabling remediation with Tier 3 off is a no-op (and is logged as such).

---

## 4. Re-run mechanism — pseudocode (the risky part)

### 4.1 Current LangGraph structure (verified, not assumed)

`_build_graph()` (`environment/mas_environment.py` ~line 3263) builds a
**linear** `StateGraph`:

```
START -> coordinator -> outline -> researcher -> executor -> final -> report_writer -> END
```

Each node is a bound method (`self.coordinator_node`, `self.outline_node`,
`self.research_node`, `self.execution_node`, `self.final_node`,
`self.report_writer_node`). The `analyst` node is added but NOT connected.

**Key finding (the biggest technical risk):** agent-to-agent content does
NOT travel through LangGraph state. It travels through the **mailboxes**
(`self.agent_mailboxes` / `episode_state.mailboxes`) via
`publish_agent_result` -> `send_message`, and each node **pulls** its
input with `receive_agent_message(receiver=..., expected_sender=...)`.
LangGraph state (`MASState`) carries only `task`, `plan`, `outline`,
`final_result`, `report_file`, `error`.

Implications for re-running ONE node:
- Re-running is possible **without restructuring the graph**: a node is
  a plain Python method taking a `MASState`-shaped dict, and its real
  input comes from a mailbox. So a re-run = "put the upstream output
  back in the mailbox (minus the discarded artifact), then call the
  method again."
- BUT the nodes are **not side-effect-free**: each node re-reads memory,
  re-logs events, and re-publishes downstream. Re-invoking the researcher
  re-runs its tool calls; re-invoking the executor re-publishes to the
  coordinator mailbox. The re-run must therefore be **scoped** (only the
  affected node(s), in dependency order) and must **drain the mailbox
  duplicates** it creates, or `final_node` will read a stale message.
- `receive_agent_message` pops a message from the mailbox; re-running a
  producer appends a NEW message. So before re-running a consumer we must
  discard the *stale* queued message for that consumer, otherwise it
  consumes the poisoned input again. This is exactly "discard the
  artifact": drop the queued message/artifact keyed by
  `request_id`/`message_id`.

**Conclusion (to confirm with you before implementing Task 5):**
re-invoking a single node is **straightforward with the existing graph
structure** — no restructuring required — provided we (a) call the node
method directly with a reconstructed `MASState`, and (b) manage the
mailbox (drain stale message, inject the cleaned upstream output) around
the call. We do NOT need to recompile or re-enter the whole graph.

### 4.2 Pseudocode

```
function remediate(tiered, agent_id, state):
    if not remediation_enabled:
        return None
    if not tier3_ran_on_any_chunk(tiered):
        log("remediation_skipped: no judge-confirmed chunk")
        return None

    summary = {detections: [], traces: [], plans: [], reruns: [], cost: {}}

    for chunk in judge_confirmed_chunks(tiered):        # tier3_verdict.contradicts_evidence
        artifact = propose_budget_and_run_detective(chunk)   # DetectiveFinding
        if not artifact.attributable:
            log("detective_not_attributable", chunk.index, artifact.reasoning)
            continue
        if not artifact.requests_tracer:
            log("detective_declined_tracing", artifact.reasoning)
            root = artifact.source_artifact_id
        else:
            trace = tracer.trace(artifact.source_artifact_id)   # TraceResult
            if not trace.root_found:
                log("tracer_no_root", trace.stopped_reason, trace.hops_taken)
                continue
            root = trace.root_artifact_id

        if root in already_remediated:                # <-- see 4.3
            log("skip_duplicate_root", root)
            continue

        plan, attempts = executor.plan(root_artifact_id=root,
                                       attempts=self._remediation_attempts)
        self._remediation_attempts = attempts
        if plan.over_attempt_cap:
            log("remediation_unresolved", root, "attempt cap reached")
            continue

        cost_before = snapshot_cost()                 # tokens, llm calls
        rerun_agents(plan.agents_to_rerun, discard=root)   # <-- see 4.4
        cost_after = snapshot_cost()
        already_remediated.add(root)
        summary.plans.append(plan)
        summary.cost = cost_after - cost_before

        # attempt 9: did the re-run still contradict?
        if still_contradicts(plan.agents_to_rerun, root):
            log("remediation_still_unresolved", root)
            # attempt counter already advanced; next detection will hit the cap

    return summary


function rerun_agents(agents_in_dependency_order, discard):
    for agent in agents_in_dependency_order:                # upstream -> downstream
        upstream_text = latest_upstream_output(agent)       # from mailbox/pool
        cleaned = remove_artifact(upstream_text, discard)   # drop request_id/message_id
        drain_mailbox(agent)                                # drop STALE queued input
        enqueue_input(agent, cleaned)                       # inject cleaned input
        state = reconstruct_mas_state()                     # task/plan/outline/...
        call_node_method(agent)(state)                       # e.g. self.research_node(state)
        record_rerun_cost(agent)
```

### 4.3 Edge case: two chunks / two agents implicating the same artifact

Per the task: **do not remediate/re-run twice for the same artifact.**

- `already_remediated` is a set of root artifact ids for the CURRENT
  episode (reset per `execute_task`). Before planning a remediation, if
  the root is already in the set, **skip** and log `skip_duplicate_root`.
- Within one `_maybe_remediate` call, the set is updated immediately
  after a successful plan, so a second chunk that traces to the same root
  is skipped.
- Across calls (different agents whose responses both trace to the same
  upstream tool result), the same episode-scoped set persists, so only
  the first one triggers the re-run.

### 4.4 Edge case: cascade ordering

`agents_to_rerun` is computed from `consumer_graph` in **dependency
order** (topological order over the pipeline
`outline -> researcher -> executor -> final`). A downstream agent is only
re-run AFTER its upstream producer has been re-run, so it consumes the
cleaned input, not the discarded artifact.

### 4.5 Edge case: re-run produces no change

If `remove_artifact` finds nothing to remove (the artifact was the
agent's only input), we still re-run (the point is to exclude the
poisoned input) and log the fact. If the re-run output is byte-identical,
we log it but do NOT loop (the attempt cap governs retries).

---

## 5. Tests planned per task (written with each task)

- **Task 2 (detective):** clear source found + tracing requested; clear
  source found + tracing declined; nothing attributable; budget
  exhaustion returns partial findings.
- **Task 3 (tracer):** root found within limit; hop limit hit;
  trusted-root hit first; shared/branching artifact does not loop
  (visited-set).
- **Task 4 (remediation):** single-agent; cascading two-agent;
  attempt-cap reached and reported.
- **Task 6 (end-to-end):** full `execute_task()` with
  `MAS_REMEDIATION_ENABLED=1`, stub judge confirming a specific chunk,
  stub tracer finding a fake root within 2 hops → correct agent re-ran,
  final output changed, remediation cost non-zero. Plus "not
  attributable" and "hop limit exceeded" → no re-run, clear log.
- **Task 7 (eval):** optional remediation-comparison mode in
  `experiments/eval_detector.py` reporting added cost + a
  semantic-deviation before/after proxy (with stated limitation).

---

## 6. What this design does NOT do (guardrails)

- Does not modify `observe_tiered`, `contradiction_checker.py`,
  `llm_judge.py`. This layer only *consumes* their output.
- Does not touch `attacks/*.py`.
- Does not implement or reference PPO.
- Does not turn on either flag by default.
- Does not commit test-scratch files or mutate
  `configs/mock_calendar.json`.

## 7. Open question for you (Task 5 gate)

Everything above is my proposal. The one thing I want explicit
confirmation on before writing any `mas_environment.py` change is the
re-run mechanism in §4: **calling the existing node methods directly
(with mailbox management around them), no graph restructuring.** If you
agree, Task 5 will do exactly that; if you would rather I restructure the
graph into a resumable subgraph, say so and I will stop and describe the
options instead of restructuring unprompted.