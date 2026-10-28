# Examples

`looping_graph` is one node that updates a counter. A conditional edge routes back to that node until the counter reaches a limit, then routes to `END`.

```bash
cargo run --example looping_graph
```

The example is in-memory. It does not call a model or the network.
