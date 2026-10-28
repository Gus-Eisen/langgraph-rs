# Roadmap

v0.1 is the ordered list below. Each milestone is one pull request onto `main`, and each one leaves `cargo test --workspace --all-targets` green. A later milestone can assume the earlier ones are already on `main`. Do not start a milestone by reverting or rewriting the public API of an earlier one, except where this file explicitly says the type widens (`Send` inputs, `Command` as the resume carrier).

Crate versions stay `0.0.0` through milestone 12 and become `0.1.0` in milestone 13. `publish` stays false. Behavior and types that are not in [ARCHITECTURE.md](ARCHITECTURE.md) are out of v0.1.

The looping-graph demo (`examples/looping_graph.rs`: a node updates a counter, a conditional edge loops until a limit, then `END`) lands in milestone 4. Until then, `examples/README.md` is only a pointer.

## Status

The owner paused this project after streaming. Milestones 0–10 are on `main`. The next thread should start at subgraphs. Milestones 11–13 are not started.

| # | Milestone | Status |
| --- | --- | --- |
| 0 | Design docs, workspace scaffold, and CI | Merged in PR #1 |
| 1 | State channels and reducers | Merged in PR #2 |
| 2 | Message types and add_messages | Merged in PR #3 |
| 3 | Graph builder and compile-time validation | Merged in PR #4 |
| 4 | Pregel superstep runtime | Merged in PR #5 |
| 5 | In-memory checkpointer | Merged in PR #6 |
| 6 | SQLite checkpointer | Merged in PR #7 |
| 7 | Interrupts and human-in-the-loop | Merged in PR #8 |
| 8 | Send fan-out | Merged in PR #9 |
| 9 | Command updates and dynamic goto | Merged in PR #10 |
| 10 | Streaming | Merged in PR #11 |
| 11 | Subgraphs | Not started |
| 12 | LLM provider trait, mock model, and OpenAI feature | Not started |
| 13 | Prebuilt ReAct tool-calling loop (also bumps the workspace to 0.1.0) | Not started |

## Milestones

| # | Title | PR delivers |
| --- | --- | --- |
| 0 | Design docs, workspace scaffold, and CI | This PR. Docs, crates, smoke tests, GitHub Actions. No graph API. |
| 1 | State channels and reducers | `GraphState`, overwrite, append, and custom reducers. |
| 2 | Message types and the add_messages reducer | Message JSON, ids, replace, remove, remove-all. |
| 3 | Graph builder and compile-time validation | Nodes, edges, conditional edges, waiting edges, `START` / `END`, `compile`. |
| 4 | Pregel superstep runtime | Concurrent sync and async nodes, barrier merges, recursion limit, cycles, the looping-graph example. |
| 5 | In-memory checkpointer, thread ids, resume, and time travel | Saver trait, memory saver, `get_state`, history, fork, `update_state`. |
| 6 | SQLite checkpointer | Same trait on `rusqlite`, default `sqlite` feature. |
| 7 | Interrupts and human-in-the-loop | `interrupt`, breakpoints, `Resume`, pending writes. |
| 8 | Send fan-out | `Send` from conditional edges, per-task input, reducer fan-in. |
| 9 | Command updates and dynamic goto | `Command.update`, `Command.goto`, `Command` as the resume carrier. |
| 10 | Streaming | `values`, `updates`, and `events`, with a namespace path on every event. |
| 11 | Subgraphs | Compiled graph as a node, state mapping, checkpoint namespace, parent routing. |
| 12 | LLM provider trait, mock model, and OpenAI feature | `ChatModel`, `MockModel`, `openai` feature flag. |
| 13 | Prebuilt ReAct tool-calling loop | Agent / tools cycle, tool errors as tool messages, step budget. Bump to `0.1.0`. |

## 0. Design docs, workspace scaffold, and CI

Docs, the five-crate workspace, `#![forbid(unsafe_code)]`, tokio on `langgraph-core`, edition 2021, `publish = false`, a smoke test, and GitHub Actions for rustfmt, `clippy -D warnings`, and tests. The only public items are version constants that record the crate graph.

Done when `cargo test --workspace --all-targets` passes and CI is green on the merge.

## 1. State channels and reducers

In `langgraph-core`: the `GraphState` trait, JSON channel map, versions, and overwrite / append / custom folds in stable task order. Unit tests cover a single write, an untouched field, two overwrites in one step (`InvalidUpdate`), append concatenation, a custom fold that is order-sensitive, and an empty step that does not bump a version.

No graph, no nodes, no `add_messages`.

## 2. Message types and the add_messages reducer

Message JSON in core, plus the `AddMessages` reducer kind. Tests cover append, id replacement in place, UUID assignment when `id` is empty, remove, remove of an unknown id, remove-all, and two writes in one step folded in task order.

The typed `Message` newtype used by the LLM crate can wait until milestone 12 if core's reducer tests use JSON values directly. Do not add chat-model types here.

## 3. Graph builder and compile-time validation

`GraphBuilder<S>` with sync and async node registration (stored, not executed), normal edges, conditional edges, waiting edges, `START` and `END`. `compile` returns `Result` and enforces the validation list in [ARCHITECTURE.md](ARCHITECTURE.md). Tests build graphs and assert `compile` accepts a cycle and rejects a missing entry edge, an edge out of `END`, an edge into `START`, an unknown target, a duplicate name, and a reserved field name.

No superstep execution and no checkpointer parameter that is ignored.

## 4. Pregel superstep runtime

`CompiledGraph::invoke` on the current tokio runtime. Parallel tasks, barrier reducer merges, `START` input writes, conditional routers seeing only their own update, waiting-edge barriers, and the recursion limit (default 25, minimum 1). A cycle that reaches `END` returns `Finished`. A cycle that never does returns `RecursionLimit`. Two overwrite writers in one step fail the step with no commit. Sync nodes run via `spawn_blocking`; async nodes are awaited together.

Tests use only in-memory state (no saver yet): a branch, a parallel pair whose append result is independent of completion order, and the recursion limit. Add `examples/looping_graph.rs` and drop the "not yet" wording in `examples/README.md`.

No checkpointing, interrupts, `Send`, or `Command`.

## 5. In-memory checkpointer, thread ids, resume, and time travel

`Checkpointer` and the checkpoint types in `langgraph-checkpoint`. `InMemorySaver` is a mutex map owned by the saver value. Core calls it from the runtime: input checkpoint, loop checkpoint after each committed superstep, `thread_id` required, missing id is `MissingThreadId`.

A second `invoke` on the same thread reduces the new input into the stored channels. `get_state` and `get_state_history` follow [ARCHITECTURE.md](ARCHITECTURE.md). Invoking with `checkpoint_id` forks a child and leaves the previous tip in history. `update_state` writes source `Update` and moves `next` to that node's successors.

No SQLite, no `interrupt`.

## 6. SQLite checkpointer

`SqliteSaver` behind the default `sqlite` feature, `rusqlite` `bundled`, `spawn_blocking`, one connection behind a mutex. The same tests as the memory saver, pointed at a temporary file, including a new saver opened on that file after the first is dropped. `delete_thread` removes that thread's rows.

Facade features do not need to forward `sqlite` if the checkpoint crate's default already turns it on for workspace tests. CI runs those tests.

## 7. Interrupts and human-in-the-loop

`interrupt(value)` and `interrupt_before` / `interrupt_after`. Outcome is `Ok(Interrupted)`. Resume with `Resume::Value` or `Resume::Map` re-enters the interrupted node from the top; finished siblings are not re-run; their pending writes commit with the step. A node with two `interrupt` calls accepts two resume values in order. Breakpoints need no resume value. No checkpointer is `MissingThreadId` (or a dedicated validation error if `compile` is given breakpoints without a saver — pick one and test it).

Do not add `Command.goto` here. `Resume` is the public resume input.

## 8. Send fan-out

Conditional routers may return `Send`. Each `Send` is a concurrent task on the next superstep with its own input value, defaulting to a partial merge onto the committed snapshot. Outputs fold through the target field's reducer. Tests: two sends to one node, append fan-in, and an unknown target node name.

`Command.goto = Send` waits for milestone 9. Widen node inputs in this PR if a test needs an input other than `S`.

## 9. Command updates and dynamic goto

A node may return `Command` with `update`, `goto` (names and `Send`s), or both. `goto` replaces static outgoing edges for that step. `invoke` / `stream` accept `Command` whose `resume` field is the milestone 7 `Resume`, and that becomes the public resume carrier. `Resume` remains the field type. `Command::parent()` is parsed and rejected with a clear error until milestone 11, unless implementing it here is free of subgraph work — prefer rejecting it so this PR does not pretend parent routing works. A documented `Error` variant for "parent routing requires a subgraph" is the honest behavior; do not silently ignore `Parent`.

## 10. Streaming

`stream` yields `values`, `updates`, and `events` as specified in [ARCHITECTURE.md](ARCHITECTURE.md). Every event carries a namespace path. Root runs use the empty path. Tests cover a two-node graph: value snapshots match `invoke`, updates are in task order, and `TaskStarted` / `TaskFinished` / `Checkpoint` appear when a saver is set. An interrupt emits `Interrupted` and stops the stream.

No token stream.

## 11. Subgraphs

`add_node` accepts a compiled graph. Same state type shares channels. Different shapes use `map_in` / `map_out`. Checkpoint namespace is parent ns plus node name. Saver inheritance follows [ARCHITECTURE.md](ARCHITECTURE.md). Nested supersteps use the child recursion limit. Child interrupts surface on the parent result. Child stream events keep the child namespace. `Command.graph = Parent` from a subgraph node updates or routes the parent.

Tests: shared state, mapped state, namespace isolation on one thread, and a parent `goto`.

## 12. LLM provider trait, mock model, and OpenAI feature

`ChatModel`, `ChatRequest`, `ChatResponse`, `ToolSpec`, and public `MockModel` in `langgraph-llm`. Message types serialize to the milestone 2 JSON. `MockModel` pops a scripted queue and errors when the queue is empty.

Feature `openai` (default off) builds a Chat Completions request and parses a response, including tool calls. Unit test the JSON with no network. Document `OPENAI_API_KEY` and the model name argument. An ignored live test is optional. Do not enable the feature in default CI.

No ReAct graph in this PR.

## 13. Prebuilt ReAct tool-calling loop

`langgraph-prebuilt` exposes a builder that returns a compiled graph: `agent` calls `ChatModel`, `tools` executes tool calls, the conditional edge loops until the assistant message has no tool calls. Tool `Err` becomes a tool message and the loop continues. The step-budget message from [ARCHITECTURE.md](ARCHITECTURE.md) is tested with `MockModel` and a low recursion limit. Optional checkpointer and interrupts on `agent` and `tools` only.

Tests use `MockModel` only. Bump the workspace version to `0.1.0`. Keep `publish = false`.

Done when a scripted tool call is answered, the following assistant message ends the graph, and the message channel contains the human message, the tool call, the tool result, and the final assistant message.
