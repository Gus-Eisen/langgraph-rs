//! `Send` fan-out from a conditional router.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use langgraph::{
    CheckpointSend, Checkpointer, Error, Field, GraphBuilder, GraphState, InMemorySaver,
    InterruptNodes, Reducer, RunConfig, RunOutput, Send, END, START,
};
use serde_json::{json, Value};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Bag {
    n: i64,
    note: String,
    items: Vec<String>,
}

#[derive(Clone, Debug, Default)]
struct BagUpdate {
    n: Option<i64>,
    note: Option<String>,
    items: Option<Vec<String>>,
}

impl GraphState for Bag {
    type Update = BagUpdate;

    fn schema() -> &'static [Field] {
        static SCHEMA: &[Field] = &[
            Field {
                name: "n",
                reducer: Reducer::Overwrite,
            },
            Field {
                name: "note",
                reducer: Reducer::Overwrite,
            },
            Field {
                name: "items",
                reducer: Reducer::Append,
            },
        ];
        SCHEMA
    }

    fn to_channel_values(&self) -> Result<std::collections::BTreeMap<String, Value>, Error> {
        Ok(std::collections::BTreeMap::from([
            ("n".to_string(), json!(self.n)),
            ("note".to_string(), Value::String(self.note.clone())),
            ("items".to_string(), string_array(&self.items)),
        ]))
    }

    fn update_to_channel_values(
        update: &Self::Update,
    ) -> Result<std::collections::BTreeMap<String, Value>, Error> {
        let mut values = std::collections::BTreeMap::new();
        if let Some(n) = update.n {
            values.insert("n".to_string(), json!(n));
        }
        if let Some(note) = &update.note {
            values.insert("note".to_string(), Value::String(note.clone()));
        }
        if let Some(items) = &update.items {
            values.insert("items".to_string(), string_array(items));
        }
        Ok(values)
    }

    fn from_channel_values(
        values: &std::collections::BTreeMap<String, Value>,
    ) -> Result<Self, Error> {
        let n = values
            .get("n")
            .and_then(Value::as_i64)
            .ok_or_else(|| Error::InvalidUpdate("channel 'n' is not an integer".into()))?;
        let note = values
            .get("note")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let items = values
            .get("items")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        Ok(Self { n, note, items })
    }
}

fn string_array(items: &[String]) -> Value {
    Value::Array(items.iter().cloned().map(Value::String).collect())
}

fn bag(n: i64, items: &[&str]) -> Bag {
    Bag {
        n,
        note: String::new(),
        items: items.iter().map(|item| (*item).to_string()).collect(),
    }
}

fn append(item: impl Into<String>) -> BagUpdate {
    BagUpdate {
        items: Some(vec![item.into()]),
        ..BagUpdate::default()
    }
}

fn finished(output: RunOutput<Bag>) -> Bag {
    match output {
        RunOutput::Finished { state } => state,
        RunOutput::Interrupted { .. } => panic!("expected Finished"),
    }
}

fn memory_config(thread_id: &str) -> (Arc<dyn Checkpointer>, RunConfig) {
    let saver: Arc<dyn Checkpointer> = Arc::new(InMemorySaver::new());
    let config = RunConfig::new()
        .checkpointer(Arc::clone(&saver))
        .thread_id(thread_id);
    (saver, config)
}

#[tokio::test]
async fn two_sends_to_one_node_each_see_their_own_value() {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let calls_node = Arc::clone(&calls);
    let seen_node = Arc::clone(&seen);
    let mut builder = GraphBuilder::<Bag>::new();
    builder
        .add_node("pick", |_state: &Bag| Ok(BagUpdate::default()))
        .add_node("map", move |state: &Bag| {
            calls_node.fetch_add(1, Ordering::SeqCst);
            seen_node.lock().expect("seen").push((
                state.note.clone(),
                state.items.clone(),
                state.n,
            ));
            Ok(append(state.note.clone()))
        })
        .add_edge(START, "pick")
        .add_conditional_edges("pick", |_state: &Bag| {
            Ok(vec![
                Send::new("map", json!({"note": "first"})),
                Send::new("map", json!({"note": "second"})),
            ])
        });
    let graph = builder.compile().expect("compile");
    let result = finished(
        graph
            .invoke(&bag(4, &["seed"]), &RunConfig::default())
            .await
            .expect("two sends"),
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let seen = seen.lock().expect("seen");
    assert!(seen.contains(&("first".to_string(), vec!["seed".to_string()], 4)));
    assert!(seen.contains(&("second".to_string(), vec!["seed".to_string()], 4)));
    assert_eq!(
        result.items,
        [
            "seed".to_string(),
            "first".to_string(),
            "second".to_string()
        ]
    );
    assert_eq!(result.note, "");
    assert_eq!(result.n, 4);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn append_fan_in_follows_send_index_not_completion() {
    let order = Arc::new(Mutex::new(Vec::new()));
    let release = Arc::new(AtomicBool::new(false));
    let order_node = Arc::clone(&order);
    let release_node = Arc::clone(&release);
    let mut builder = GraphBuilder::<Bag>::new();
    builder
        .add_node("pick", |_state: &Bag| Ok(BagUpdate::default()))
        .add_node("map", move |state: &Bag| {
            let note = state.note.clone();
            let order_node = Arc::clone(&order_node);
            let release_node = Arc::clone(&release_node);
            async move {
                if note == "later" {
                    order_node.lock().expect("order").push(note.clone());
                    release_node.store(true, Ordering::SeqCst);
                } else {
                    let started = Instant::now();
                    while !release_node.load(Ordering::SeqCst) {
                        assert!(
                            started.elapsed() < Duration::from_secs(2),
                            "first send was not concurrent with the later send"
                        );
                        tokio::task::yield_now().await;
                    }
                    order_node.lock().expect("order").push(note.clone());
                }
                Ok(append(note))
            }
        })
        .add_edge(START, "pick")
        .add_conditional_edges("pick", |_state: &Bag| {
            Ok(vec![
                Send::new("map", json!({"note": "first"})),
                Send::new("map", json!({"note": "later"})),
            ])
        });
    let graph = builder.compile().expect("compile");
    let result = finished(
        graph
            .invoke(&bag(1, &["seed"]), &RunConfig::default())
            .await
            .expect("fan in"),
    );
    assert_eq!(
        order.lock().expect("order").as_slice(),
        ["later".to_string(), "first".to_string()]
    );
    assert_eq!(
        result.items,
        ["seed".to_string(), "first".to_string(), "later".to_string()]
    );
}

#[tokio::test]
async fn unknown_send_target_does_not_commit() {
    let mut builder = GraphBuilder::<Bag>::new();
    builder
        .add_node("pick", |_state: &Bag| {
            Ok(BagUpdate {
                note: Some("wrote".to_string()),
                ..BagUpdate::default()
            })
        })
        .add_edge(START, "pick")
        .add_conditional_edges("pick", |_state: &Bag| Ok(Send::new("missing", json!({}))));
    let graph = builder.compile().expect("compile");
    let (_saver, config) = memory_config("unknown-send");
    let error = graph
        .invoke(&bag(3, &["kept"]), &config)
        .await
        .expect_err("unknown target");
    match error {
        Error::UnknownNode(name) => assert_eq!(name, "missing"),
        other => panic!("expected UnknownNode, got {other:?}"),
    }
    let snapshot = graph.get_state(&config).await.expect("checkpoint");
    assert_eq!(snapshot.values.note, "");
    assert_eq!(snapshot.values.n, 3);
    assert_eq!(snapshot.values.items, ["kept".to_string()]);
    assert_eq!(snapshot.next, ["pick".to_string()]);
}

#[tokio::test]
async fn repeated_plain_names_still_schedule_one_task() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_node = Arc::clone(&calls);
    let mut builder = GraphBuilder::<Bag>::new();
    builder
        .add_node("pick", |_state: &Bag| Ok(BagUpdate::default()))
        .add_node("map", move |_state: &Bag| {
            calls_node.fetch_add(1, Ordering::SeqCst);
            Ok(append("once"))
        })
        .add_edge(START, "pick")
        .add_conditional_edges("pick", |_state: &Bag| Ok(vec!["map", "map"]));
    let graph = builder.compile().expect("compile");
    let result = finished(
        graph
            .invoke(&bag(0, &[]), &RunConfig::default())
            .await
            .expect("deduped"),
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(result.items, ["once".to_string()]);
}

#[tokio::test]
async fn sends_survive_a_breakpoint_and_stay_distinct() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_node = Arc::clone(&calls);
    let mut builder = GraphBuilder::<Bag>::new();
    builder
        .add_node("pick", |_state: &Bag| Ok(BagUpdate::default()))
        .add_node("map", move |state: &Bag| {
            calls_node.fetch_add(1, Ordering::SeqCst);
            Ok(append(state.note.clone()))
        })
        .add_edge(START, "pick")
        .add_conditional_edges("pick", |_state: &Bag| {
            Ok([
                Send::new("map", json!({"note": "one"})),
                Send::new("map", json!({"note": "two"})),
            ])
        });
    let graph = builder
        .compile_with(["map"], InterruptNodes::none())
        .expect("compile");
    let (saver, config) = memory_config("send-pause");
    let paused = graph
        .invoke(&bag(4, &["seed"]), &config)
        .await
        .expect("pause");
    let RunOutput::Interrupted { state, interrupts } = paused else {
        panic!("expected Interrupted");
    };
    assert!(interrupts.is_empty());
    assert_eq!(state.items, ["seed".to_string()]);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let snapshot = graph.get_state(&config).await.expect("paused state");
    assert_eq!(snapshot.next, ["map".to_string(), "map".to_string()]);
    let stored = saver
        .get_tuple(&config.checkpoint_config().expect("thread"))
        .await
        .expect("tuple")
        .expect("checkpoint");
    assert_eq!(
        stored.checkpoint.sends,
        vec![
            CheckpointSend {
                edge_index: 0,
                node: "map".to_string(),
                value: json!({"note": "one"}),
            },
            CheckpointSend {
                edge_index: 0,
                node: "map".to_string(),
                value: json!({"note": "two"}),
            },
        ]
    );

    let resumed = finished(
        graph
            .invoke(&bag(99, &["ignored"]), &config)
            .await
            .expect("resume"),
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(resumed.n, 4);
    assert_eq!(
        resumed.items,
        ["seed".to_string(), "one".to_string(), "two".to_string()]
    );
}

#[tokio::test]
async fn router_can_mix_a_send_with_end() {
    let mut builder = GraphBuilder::<Bag>::new();
    builder
        .add_node("pick", |_state: &Bag| Ok(BagUpdate::default()))
        .add_node("map", |state: &Bag| Ok(append(state.note.clone())))
        .add_edge(START, "pick")
        .add_conditional_edges("pick", |_state: &Bag| {
            Ok(vec![
                langgraph::RouteTarget::Send(Send::new("map", json!({"note": "only"}))),
                langgraph::RouteTarget::Node(END.to_string()),
            ])
        });
    let graph = builder.compile().expect("compile");
    let result = finished(
        graph
            .invoke(&bag(0, &[]), &RunConfig::default())
            .await
            .expect("mixed"),
    );
    assert_eq!(result.items, ["only".to_string()]);
    assert_eq!(result.note, "");
}
