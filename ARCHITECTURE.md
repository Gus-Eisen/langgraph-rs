# Architecture

langgraph-rs is an independent Rust port of [LangGraph](https://github.com/langchain-ai/langgraph) semantics (MIT, Copyright (c) 2024 LangChain, Inc.). It is not affiliated with LangChain. The port keeps the execution model and drops Python idioms: no TypedDict, no `Annotated` reducers, no LangChain runnables, no callback manager.

Milestone 0 (this repository's first PR) contains this document, [ROADMAP.md](ROADMAP.md), and a compiling workspace. It does not implement the graph API. Later milestones must follow the decisions in this file.

Upstream behavior this design is taken from, read at the time of milestone 0:

- `langgraph/graph/state.py` — `StateGraph`, edges, `compile` validation
- `langgraph/channels/last_value.py`, `langgraph/channels/binop.py` — overwrite and binary reducers
- `langgraph/graph/message.py` — `add_messages`
- `langgraph/pregel/_loop.py`, `langgraph/pregel/_algo.py` — supersteps, `apply_writes`, task scheduling
- `langgraph/types.py` — `Command`, `Send`, `Interrupt`, stream modes
- `langgraph/checkpoint/base/__init__.py` — saver protocol, checkpoint shape, thread ids
- `langgraph/prebuilt/chat_agent_executor.py` — ReAct tool loop

## Non-goals

v0.1 does not include:

- LangGraph Platform, the API server, Studio, or any hosted deploy path
- a JavaScript or Python SDK, and a CLI
- publishing to crates.io (`publish = false` stays on every crate)
- `unsafe` code (`#![forbid(unsafe_code)]` in every crate root)
- a cross-thread store (long-term memory across thread ids)
- delta channels, a general managed-value system, or the deprecated `MessageGraph`
- the functional API (`entrypoint` / `task`)
- token-level `messages` streaming, a custom stream writer, or LangSmith tracing
- Postgres (or any saver other than memory and SQLite)
- node caches, retry policies, timeout policies, and durability modes other than a synchronous checkpoint at each superstep boundary
- reading or writing Python LangGraph checkpoint blobs
- a `#[derive(GraphState)]` proc-macro (hand-written trait impls; a derive can come after v0.1)
- multimodal message content and `add_messages(format="langchain-openai")`
- a hidden tokio runtime inside `invoke` (callers are already on a runtime)

Stable Rust, edition 2021. The MSRV is the toolchain in `rust-toolchain.toml` (stable 1.99.1 at milestone 0). Async work uses tokio.

## Crate layout

```
crates/langgraph             public facade; re-exports only, no logic
crates/langgraph-core        state, reducers, builder, Pregel runtime, streaming
crates/langgraph-checkpoint  Checkpointer trait, in-memory saver, SQLite saver
crates/langgraph-llm         ChatModel trait, mock model, OpenAI feature
crates/langgraph-prebuilt    ReAct agent
```

Dependency direction, which milestone 0 already wires:

- `langgraph-checkpoint` and `langgraph-llm` depend on no workspace crate
- `langgraph-core` depends on `langgraph-checkpoint`
- `langgraph-prebuilt` depends on `langgraph-core` and `langgraph-llm`
- `langgraph` depends on all four and is the only crate downstream code should name

Checkpoint values are serde JSON. `langgraph-checkpoint` does not depend on core types. Core converts state to and from `serde_json::Value` at the saver boundary.

Planned dependencies, added by the milestone that first uses them:

- tokio on the core runtime (`rt-multi-thread`, `macros`, `sync`). Milestone 0 already depends on it from `langgraph-core` so the workspace runtime is fixed.
- `serde`, `serde_json` for state and checkpoints
- `thiserror` for library errors
- `uuid` (v7 for checkpoint ids, v4 for message ids)
- `rusqlite` with `bundled`, called through `spawn_blocking`, behind the checkpoint crate's default `sqlite` feature
- `reqwest` with rustls, behind `langgraph-llm`'s `openai` feature (default off)

No `async-trait`. Object-safe traits store `Pin<Box<dyn Future<Output = ...> + Send + 'a>>` so a compiled graph can hold `Arc<dyn Checkpointer>` and `Arc<dyn ChatModel>`. Native `async fn` in traits is not object-safe, and subgraphs need type erasure.

Public functions return `Result`. Panics are reserved for broken internal invariants. Do not add a public item that returns `unimplemented!` or otherwise pretends a later milestone already works. Crate versions stay `0.0.0` until the last v0.1 milestone, which bumps the workspace to `0.1.0`. `publish` stays false.

## State and reducers

A graph is generic over `S: GraphState`. `GraphState` provides:

- a static field list (channel name plus reducer)
- `Update`, the partial update a node returns
- conversion of `S` and `Update` to per-field `serde_json::Value` writes
- reconstruction of `S` from committed channel values

A node receives a shared reference and returns an update. It does not replace the whole state unless its update writes every field. Fields omitted from the update are not writes. The runtime may pass the same committed snapshot to every task in a superstep; nodes must not rely on mutating that snapshot.

Reducer kinds, fixed when the schema is defined:

| Kind | Superstep rule |
| --- | --- |
| `Overwrite` | At most one write. Zero writes leave the channel unchanged. Two or more writes in the same superstep are `Error::InvalidUpdate`. |
| `Append` | Both the current value and each write are JSON arrays. Fold concatenation in task order. A non-array is `Error::InvalidUpdate`. |
| `Custom` | `fn(current: Option<&Value>, update: &Value) -> Result<Value, Error>`, folded left to right in task order. The current value is the channel at the start of the fold; concurrent tasks do not see each other. |
| `AddMessages` | The message reducer below. Each write is one message or an array of messages. |

`Append` writes are the items to add, not the full list. A node that wants to append one string writes a one-element array.

Task order for a fold is the ascending sort of `(task_path, task_id, write_index)`. Completion order must not change reducer results. Tasks still *run* concurrently.

Reserved names, rejected for nodes and for state fields: `__start__`, `__end__`, `__interrupt__`, `__resume__`, `__error__`, `__pregel_tasks`. `START` and `END` in the builder are those first two sentinels.

Channel versions are `u64`, starting at 0 for an unseen channel and increasing by 1 when a superstep accepts a write to that channel. Versions are integers on purpose; they are not Python version strings.

## Messages and `add_messages`

v0.1 messages are JSON objects:

- `human`, `ai`, `system`: `type`, `id`, `content` (string)
- `ai` may include `tool_calls`: `{id, name, arguments}` where `arguments` is a JSON object
- `tool`: `type`, `id`, `content`, `tool_call_id`
- `remove`: `type`, `id`

`add_messages(left, right)`:

- Coerce a single message on either side into a one-element list.
- Assign a UUID v4 `id` when `id` is missing or empty.
- A `remove` whose id is `__remove_all__` drops `left` and every message in `right` up to and including that sentinel. The result is the messages after the last such sentinel in `right`.
- Otherwise merge `right` into a copy of `left`. The same id replaces in place. A `remove` of an existing id deletes it. A `remove` of an unknown id is `Error::InvalidUpdate`.
- Several writes to the same channel in one superstep fold this function in task order, so a later remove-all clears earlier writes from that step.

The Rust type used by the LLM crate and the ReAct agent is this same shape. Core owns the reducer. The LLM crate owns the typed `Message` newtype that serializes to it.

## Graph builder

`GraphBuilder<S>` records structure and `compile`s into `CompiledGraph<S>`.

- `add_node` accepts a sync function `fn(&S) -> Result<S::Update, Error>` or an async function returning `Future<Output = Result<S::Update, Error>> + Send`. Sync nodes run on `spawn_blocking`. Async nodes are awaited on the current runtime. Both are stored type-erased on the compiled graph. Node names are unique.
- `add_edge(from, to)` draws a normal edge. `from` may be a single node or a list of nodes (a waiting edge / named barrier). `START` and `END` are legal endpoints with the restrictions below.
- `add_conditional_edges(from, router)` runs `router` inside the source task, after the source node returns. The router sees `reduce(snapshot_at_step_start, this_node_update)` and does not see other tasks from this superstep. It returns one node name, several node names, `END`, or (from the Send milestone onward) one or more `Send`s. An optional static path map is checked at `compile`. Without a path map, an unknown name fails when the router returns it, at runtime.
- Cycles are ordinary edges back to an earlier node. There is no special loop API. A run ends by routing to `END`, by an interrupt, or by the recursion limit.
- A waiting edge `(A, B) -> C` schedules `C` only after both `A` and `B` have a completion newer than the one `C` last observed. The predecessors need not have finished in the same superstep.

`compile` is the structural check. It returns `Result` and rejects:

- no outgoing edge from `START`
- an edge whose source is not `START` and not a known node
- an edge whose target is not `END` and not a known node
- any edge that leaves `END` or enters `START`
- a static path-map target that is neither `END` nor a known node
- a duplicate node name, or a node or field that uses a reserved name
- two registrations of the same field with different reducers

Unreachable nodes are allowed. Upstream does not reject them.

`compile` does not run the graph. `interrupt_before` / `interrupt_after` are parameters of `compile` added by the interrupt milestone, not before, because a parameter that is stored and ignored would be a false API.

Until the Send milestone, every node input is `&S`. That milestone is allowed to widen a node's input type. The builder milestone should keep node invocation behind one internal trait so the widening does not rewrite call sites.

## Pregel supersteps

Execution is Bulk Synchronous Parallel, the same model as Pregel.

1. Apply the run input as writes attributed to `START`, through the normal reducers. On a fresh thread this is the whole state. On a resumed thread the input is reduced into the checkpoint (an `add_messages` field appends; an overwrite field replaces). This application is not a superstep.
2. Schedule tasks. A node runs when any trigger channel's version is greater than the version recorded in `versions_seen` for that node. Normal edges trigger the successor when the predecessor completes. `START` triggers the entry nodes.
3. A superstep runs every scheduled task concurrently on tokio. Each task reads the committed snapshot from the start of the step.
4. Join the step. A task `Err` fails the superstep: wait for the other tasks, drop every write from the step, return the error. No partial commit.
5. At the barrier, fold writes with the reducers in task order, bump versions, update `versions_seen`, and emit stream output.
6. If a checkpointer is configured, `put` a checkpoint with source `Loop`, then `put_writes` as described under interrupts. Checkpoint I/O is synchronous from the run's point of view: the next superstep does not start until `put` finishes.
7. Repeat until no tasks are scheduled (`END`), an interrupt pauses the run, or the recursion limit is hit.

The recursion limit counts supersteps. A run executes at most `recursion_limit` supersteps and refuses superstep `recursion_limit + 1` with `Error::RecursionLimit` before running it. The default is 25. The minimum is 1. The value lives on the run config, not in an environment variable. Upstream's process default is currently 10007 via `LANGGRAPH_DEFAULT_RECURSION_LIMIT`; this port uses 25 so a cycle bug fails quickly. Routers are part of the source task, not their own superstep. A subgraph uses its own counter (see Subgraphs).

`invoke` and `stream` are async and require `Handle::current()`. Calling them outside tokio returns `Error::NoRuntime`.

`RunOutput<S>` is `Finished { state }` or `Interrupted { state, interrupts }`. An interrupt is a successful `Result` (`Ok(Interrupted { .. })`). It is not an `Err`.

## Checkpointer

```rust
trait Checkpointer: Send + Sync {
    fn get_tuple<'a>(&'a self, config: &'a CheckpointConfig)
        -> BoxFuture<'a, Result<Option<CheckpointTuple>, Error>>;
    fn list<'a>(&'a self, config: &'a CheckpointConfig, before: Option<&'a str>, limit: Option<usize>)
        -> BoxFuture<'a, Result<Vec<CheckpointTuple>, Error>>;
    fn put<'a>(&'a self, config: &'a CheckpointConfig, checkpoint: &'a Checkpoint,
               metadata: &'a CheckpointMetadata, new_versions: &'a BTreeMap<String, u64>)
        -> BoxFuture<'a, Result<CheckpointConfig, Error>>;
    fn put_writes<'a>(&'a self, config: &'a CheckpointConfig, writes: &'a [PendingWrite], task_id: &'a str)
        -> BoxFuture<'a, Result<(), Error>>;
    fn delete_thread<'a>(&'a self, thread_id: &'a str)
        -> BoxFuture<'a, Result<(), Error>>;
}
```

`CheckpointConfig` carries `thread_id`, `checkpoint_ns` (empty at the root), and optional `checkpoint_id`. A configured saver with no `thread_id` is `Error::MissingThreadId`. `get_tuple` with no `checkpoint_id` loads the latest checkpoint in that thread and namespace. `list` returns newest first.

`Checkpoint` fields: `v` (format 1), time-ordered UUID v7 `id`, RFC3339 `ts`, `channel_values`, `channel_versions`, `versions_seen` (node to channel to version), `updated_channels`. Metadata `source` is `Input`, `Loop`, `Update`, or `Fork`. `step` is `-1` for the input checkpoint and `0` for the first loop checkpoint. `parents` maps checkpoint namespace to parent checkpoint id.

In-memory saver: a mutex map, per saver instance, not process-global. SQLite saver: one `rusqlite` connection behind a mutex, `spawn_blocking` for every call, `bundled` so builds do not need a system libsqlite. Same trait, same tests, temporary file in the SQLite tests. The `sqlite` feature is on by default and CI tests it.

With no saver, state exists only for that call. `thread_id` is ignored.

`get_state` returns channel values, the next node names, config, parent config, pending tasks, and interrupts. `get_state_history` walks `list`. `update_state` writes a new checkpoint with source `Update`, attributed to `as_node`, so that node's successors are next.

Time travel: a run whose config sets `checkpoint_id` loads that snapshot. The following `put` creates a new checkpoint whose parent is the selected id. Older checkpoints stay as they were.

## Interrupts and `Command` resume

Two pause mechanisms, both requiring a checkpointer:

- `interrupt_before` / `interrupt_after` on `compile`, naming nodes (or all nodes). The runtime pauses without a resume value.
- `interrupt(value)` inside a node. The first call for that task records an `Interrupt { id, value }` and pauses. `id` is stable for `(checkpoint_ns, task_path, call_index)`. On resume the node function runs again from the start. Each `interrupt` call returns the next stored resume value for that task, in order. A further unanswered call pauses again.

Resume input is `Resume::Value(Value)` for the next unanswered interrupt, or `Resume::Map` from interrupt id to value. Until the Command milestone the public carrier is `Resume`. That milestone adds `Command` and resume goes through `Command.resume` of the same type.

When a task interrupts, sibling tasks in the superstep are allowed to finish. Their writes are stored with `put_writes` and are not applied to channel values yet. On resume, tasks that already finished are not executed again; their pending writes are replayed. The interrupted task runs again. Channel values commit only when every task in the superstep has finished without a new interrupt.

`Interrupt` is data: `id` and JSON `value`. The graph outcome is `RunOutput::Interrupted`.

## `Command`

`Command` is the single control value a node may return in place of a plain update. Fields:

- `update`: an `S::Update` merged like a normal node return
- `resume`: a `Resume`, meaningful on the input of `invoke` / `stream`, not as a node return
- `goto`: node names and `Send`s that replace the node's static outgoing edges for this step
- `graph`: `Current` (default) or `Parent`

`goto` is applied by the source task the same way a conditional router is. Static edges from that node do not also fire. `Parent` is how a subgraph node writes or routes in the parent; it is implemented with subgraphs.

## `Send`

`Send { node, value }` schedules `node` on the next superstep with `value` as that task's input. A conditional router or a `Command.goto` may return many `Send`s to the same node. Each is a separate concurrent task. Outputs still fold into the shared channels through the reducers.

`value` deserializes into the node input. From the Send milestone on, a node may declare an input type other than `S`. The default remains `S`, and a partial object is merged onto a clone of the committed snapshot to form the task view. Map-reduce is the intended use: fan out with `Send`, fan in with an `Append` or `AddMessages` field, optionally behind a waiting edge.

## Subgraphs

A compiled graph may be added as a node.

- The subgraph's `invoke` occupies one parent task and therefore one parent superstep. Supersteps inside the subgraph increment the subgraph's own recursion counter and do not consume the parent's.
- Same `GraphState` type: the subgraph reads and writes the parent's channels under those field names.
- A different input or output shape: the parent supplies `map_in` and `map_out`. `map_out` produces an `S::Update` on the parent.
- Checkpointer `None` inherits the parent saver. An explicit saver is private to the subgraph. An explicit disabled saver does not inherit.
- The subgraph's `checkpoint_ns` is the parent namespace plus the node name, so its checkpoints are a separate list on the same thread.
- Interrupts inside the subgraph surface as the parent run's `Interrupted` output.
- Stream events from the subgraph are forwarded with that namespace (the streaming milestone's event type must already carry a namespace path, even when the only path is the root).

## Streaming

`stream` yields events as the run goes. Modes, which can be combined:

- `values`: the full committed state after each superstep barrier, and the state at an interrupt
- `updates`: one item per task that wrote, carrying the node name and that task's update, emitted at the barrier in task order
- `events`: `TaskStarted`, `TaskFinished` (with the update or the error), `Checkpoint`, and `Interrupted`

`events` is this port's task and checkpoint stream. It is not LangChain `astream_events`. Token streaming and a user stream writer are out of v0.1.

## LLM trait and ReAct

`langgraph-llm`:

```rust
trait ChatModel: Send + Sync {
    fn complete<'a>(&'a self, request: &'a ChatRequest)
        -> BoxFuture<'a, Result<ChatResponse, Error>>;
}
```

`ChatRequest` is messages plus `ToolSpec` (`name`, `description`, JSON Schema `parameters`). `ChatResponse` is one `ai` message, with `tool_calls` when the model wants tools.

`MockModel` is public. Tests script a queue of responses. It is the stand-in for a provider in every ReAct test.

The `openai` feature (default off) talks to the OpenAI Chat Completions HTTP API with `reqwest` and rustls. The key is `OPENAI_API_KEY`. The feature builds the request JSON and parses `tool_calls`. CI covers request construction with no network. A live call, if any, is `#[ignore]`.

`langgraph-prebuilt` builds a graph with two nodes:

1. `agent` calls `ChatModel::complete` with the message channel and the tool specs.
2. If the assistant message has no tool calls, the conditional edge routes to `END`. Otherwise it routes to `tools`.
3. `tools` runs each tool call. `Tool::call` returns `Result<String, Error>`. `Err` becomes a `tool` message containing the error text, and the loop continues. `Ok` becomes a `tool` message with that `tool_call_id`.
4. `tools` routes back to `agent`.

Messages use the `AddMessages` reducer. If the model still requests tools and fewer than two supersteps remain in the recursion limit, `agent` writes a final assistant message `Sorry, need more steps to process this request.` and routes to `END` instead of raising `RecursionLimit`. That matches the prebuilt agent's step budget.

Tools are async, `Send + Sync`, named, and described by a JSON Schema. The prebuilt graph accepts an optional checkpointer and the same interrupt lists, limited to the `agent` and `tools` nodes.

## Errors

One `Error` enum in `langgraph-core`, re-exported by the facade when it exists. Checkpoint and LLM crates may define their own error types and core maps them at the boundary. Variants the milestones need:

- `InvalidUpdate` — concurrent overwrite, bad append, unknown message remove
- `GraphValidation` — `compile` failures
- `UnknownNode` — a router or `Command` named a node that does not exist
- `RecursionLimit`
- `MissingThreadId`
- `Checkpoint` — saver failure, unknown checkpoint id
- `NoRuntime`
- `Node` — a node returned `Err`
- `Llm` — provider or mock failure

## Decisions later milestones follow

1. Facade crate only for downstream users. Logic stays in the crate that owns it.
2. `GraphState` plus JSON channels. No proc-macro in v0.1. Reducers are overwrite, append, custom, and `add_messages`, with the fold rules above.
3. Message JSON shape is fixed as in this document. Content is a string.
4. `compile` validates structure and returns `Result`. It does not require reachability.
5. Pregel: concurrent tasks, reducer fold in stable task order at the barrier, default recursion limit 25 supersteps, input writes are not a superstep.
6. `invoke` / `stream` use the current tokio runtime. Sync nodes use `spawn_blocking`.
7. Interrupts are `Ok(RunOutput::Interrupted)`, not `Err`.
8. `Checkpointer` is object-safe with boxed futures and lives in `langgraph-checkpoint`. Integer channel versions. UUID v7 checkpoint ids. Memory saver is not process-global. SQLite uses `rusqlite` `bundled` and `spawn_blocking`.
9. Time travel forks a new child checkpoint. It does not rewrite history.
10. `Resume` is the resume payload. `Command` gains `update`, `goto`, and `Parent` in its own milestone, after `Send` exists.
11. Stream modes for v0.1 are `values`, `updates`, and `events`, and event paths have a namespace from the start.
12. One provider feature, `openai`, default off. Tests use `MockModel`. No network in CI.
13. No `unsafe`, no crates.io publish, edition 2021, MSRV = `rust-toolchain.toml`.
14. Do not land a public API ahead of its behavior.
