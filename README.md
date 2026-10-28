# langgraph-rs

This repository is an experiment in fully AI-delegated engineering: all code was written, reviewed, and merged by AI agents, not by the repo owner.

langgraph-rs is an independent Rust port of the [LangGraph](https://github.com/langchain-ai/langgraph) execution model (MIT). It is not affiliated with, endorsed by, or maintained by LangChain, Inc. See [NOTICE](NOTICE) for the upstream copyright and [LICENSE](LICENSE) for this tree.

The design is in [ARCHITECTURE.md](ARCHITECTURE.md). The v0.1 pull-request sequence is in [ROADMAP.md](ROADMAP.md).

## Quickstart

State channels, reducers, the graph builder, and the Pregel runtime are in the crates. `GraphState` is a JSON channel map with `u64` versions and overwrite, append, custom, and `add_messages` folds. `GraphBuilder` records sync and async nodes, ordinary edges, waiting edges, and conditional routers. `compile` checks that structure. `CompiledGraph::invoke` runs supersteps on the current tokio runtime. `InMemorySaver` checkpoints a thread: a later `invoke` with the same `thread_id` merges its input into the stored channels and resumes. SQLite, interrupts, and the ReAct agent arrive in the later milestones in [ROADMAP.md](ROADMAP.md). The looping graph is `examples/looping_graph.rs`:

```bash
cargo run --example looping_graph
```

Rust is pinned to stable 1.99.1 in `rust-toolchain.toml`. From a checkout:

```bash
cargo test --workspace --all-targets
```

That builds the workspace and runs the state-channel tests, the message-reducer tests, the graph-builder tests, the Pregel runtime tests, the in-memory checkpointer tests, and the scaffold smoke tests.

GitHub Actions runs `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace --all-targets` on every pull request and on pushes to `main`.

## Layout

| Path | Role |
| --- | --- |
| `crates/langgraph` | Facade. Downstream code should depend on this crate. |
| `crates/langgraph-core` | State channels, reducers, the graph builder, and the Pregel runtime. |
| `crates/langgraph-checkpoint` | `Checkpointer`, in-memory saver, SQLite saver. |
| `crates/langgraph-llm` | `ChatModel`, mock model, optional OpenAI feature. |
| `crates/langgraph-prebuilt` | ReAct tool-calling loop. |
| `examples/looping_graph.rs` | Counter loop. Run with `cargo run --example looping_graph`. |

Crates are edition 2021, `#![forbid(unsafe_code)]`, and `publish = false`. Nothing is published to crates.io.
