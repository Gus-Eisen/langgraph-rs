# Examples

The looping-graph demo is not in the tree yet. The Pregel runtime milestone in [ROADMAP.md](../ROADMAP.md) adds `examples/looping_graph.rs`: one node updates a counter, a conditional edge routes back to that node until the counter reaches a limit, then routes to `END`.

No example crate target is checked in before that milestone, so `cargo test` has nothing here to compile.
