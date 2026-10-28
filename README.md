# langgraph-rs

This repository is an experiment in fully AI-delegated engineering: all code was written, reviewed, and merged by AI agents, not by the repo owner.

langgraph-rs is an independent Rust port of the [LangGraph](https://github.com/langchain-ai/langgraph) execution model (MIT). It is not affiliated with, endorsed by, or maintained by LangChain, Inc. See [NOTICE](NOTICE) for the upstream copyright and [LICENSE](LICENSE) for this tree.

The design is in [ARCHITECTURE.md](ARCHITECTURE.md). The v0.1 pull-request sequence is in [ROADMAP.md](ROADMAP.md).

## Quickstart

State channels and reducers are in the crates: `GraphState`, a JSON channel map, `u64` channel versions, and overwrite, append, custom, and `add_messages` folds. The `langgraph` facade re-exports those types. The graph builder, Pregel execution, checkpointers, and the ReAct agent arrive in the later milestones in [ROADMAP.md](ROADMAP.md). This tree has no runnable graph example yet.

Rust is pinned to stable 1.99.1 in `rust-toolchain.toml`. From a checkout:

```bash
cargo test --workspace --all-targets
```

That builds the workspace and runs the state-channel tests, the message-reducer tests, and the scaffold smoke tests.

GitHub Actions runs `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace --all-targets` on every pull request and on pushes to `main`.

## Layout

| Path | Role |
| --- | --- |
| `crates/langgraph` | Facade. Downstream code should depend on this crate once the API exists. |
| `crates/langgraph-core` | State channels and reducers. The builder and Pregel runtime arrive later. |
| `crates/langgraph-checkpoint` | `Checkpointer`, in-memory saver, SQLite saver. |
| `crates/langgraph-llm` | `ChatModel`, mock model, optional OpenAI feature. |
| `crates/langgraph-prebuilt` | ReAct tool-calling loop. |
| `examples/README.md` | Pointer at the looping-graph demo, which lands with the runtime milestone. |

Crates are edition 2021, `#![forbid(unsafe_code)]`, and `publish = false`. Nothing is published to crates.io.
