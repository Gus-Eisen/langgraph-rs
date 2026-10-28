//! `Command` updates, dynamic `goto`, and resume.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use langgraph::{
    interrupt, CheckpointSource, Checkpointer, Command, Error, Field, GraphBuilder, GraphState,
    InMemorySaver, Reducer, Resume, RunConfig, RunOutput, Send, END, START,
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
async fn command_update_and_goto_skips_static_edges_and_the_router() {
    let static_ran = Arc::new(AtomicBool::new(false));
    let router_ran = Arc::new(AtomicBool::new(false));
    let chosen_calls = Arc::new(AtomicUsize::new(0));
    let static_flag = Arc::clone(&static_ran);
    let router_flag = Arc::clone(&router_ran);
    let chosen_flag = Arc::clone(&chosen_calls);
    let mut builder = GraphBuilder::<Bag>::new();
    builder
        .add_node("choose", |_state: &Bag| {
            Ok(Command::new()
                .update(BagUpdate {
                    n: Some(7),
                    ..BagUpdate::default()
                })
                .goto(vec!["chosen", "chosen"]))
        })
        .add_node("static_next", move |_state: &Bag| {
            static_flag.store(true, Ordering::SeqCst);
            Ok(append("static"))
        })
        .add_node("router_next", |_state: &Bag| Ok(append("router")))
        .add_node("chosen", move |_state: &Bag| {
            chosen_flag.fetch_add(1, Ordering::SeqCst);
            Ok(append("chosen"))
        })
        .add_edge(START, "choose")
        .add_edge("choose", "static_next")
        .add_edge("static_next", END)
        .add_edge("router_next", END)
        .add_edge("chosen", END)
        .add_conditional_edges("choose", move |_state: &Bag| {
            router_flag.store(true, Ordering::SeqCst);
            Ok("router_next")
        });
    let graph = builder.compile().expect("compile");
    let result = finished(
        graph
            .invoke(&bag(1, &["seed"]), &RunConfig::default())
            .await
            .expect("goto"),
    );
    assert!(!static_ran.load(Ordering::SeqCst));
    assert!(!router_ran.load(Ordering::SeqCst));
    assert_eq!(chosen_calls.load(Ordering::SeqCst), 1);
    assert_eq!(result.n, 7);
    assert_eq!(result.items, ["seed".to_string(), "chosen".to_string()]);
}

#[tokio::test]
async fn command_goto_send_fans_out_and_skips_the_static_edge() {
    let static_ran = Arc::new(AtomicBool::new(false));
    let static_flag = Arc::clone(&static_ran);
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_node = Arc::clone(&calls);
    let mut builder = GraphBuilder::<Bag>::new();
    builder
        .add_node("fan", |_state: &Bag| {
            Ok(Command::new().goto(vec![
                Send::new("map", json!({"note": "one"})),
                Send::new("map", json!({"note": "two"})),
            ]))
        })
        .add_node("map", move |state: &Bag| {
            calls_node.fetch_add(1, Ordering::SeqCst);
            Ok(append(state.note.clone()))
        })
        .add_node("static_next", move |_state: &Bag| {
            static_flag.store(true, Ordering::SeqCst);
            Ok(append("static"))
        })
        .add_edge(START, "fan")
        .add_edge("fan", "static_next")
        .add_edge("map", END)
        .add_edge("static_next", END);
    let graph = builder.compile().expect("compile");
    let result = finished(
        graph
            .invoke(&bag(4, &["seed"]), &RunConfig::default())
            .await
            .expect("send goto"),
    );
    assert!(!static_ran.load(Ordering::SeqCst));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        result.items,
        ["seed".to_string(), "one".to_string(), "two".to_string()]
    );
    assert_eq!(result.note, "");
    assert_eq!(result.n, 4);
}

#[tokio::test]
async fn invoke_resumes_an_interrupt_through_command_resume() {
    let mut builder = GraphBuilder::<Bag>::new();
    builder
        .add_node("ask", |_state: &Bag| {
            let answer = interrupt(json!("need"))?;
            Ok(BagUpdate {
                note: Some(answer.as_str().unwrap_or("").to_string()),
                ..BagUpdate::default()
            })
        })
        .add_edge(START, "ask")
        .add_edge("ask", END);
    let graph = builder.compile().expect("compile");
    let (_saver, config) = memory_config("command-resume");
    let paused = graph
        .invoke(&bag(4, &["seed"]), &config)
        .await
        .expect("pause");
    let RunOutput::Interrupted { state, interrupts } = paused else {
        panic!("expected Interrupted");
    };
    assert_eq!(interrupts.len(), 1);
    assert_eq!(interrupts[0].value, json!("need"));
    assert_eq!(state.n, 4);
    assert_eq!(state.items, ["seed".to_string()]);

    let resumed = finished(
        graph
            .invoke(Command::new().resume(Resume::Value(json!("yes"))), &config)
            .await
            .expect("resume"),
    );
    assert_eq!(resumed.note, "yes");
    assert_eq!(resumed.n, 4);
    assert_eq!(resumed.items, ["seed".to_string()]);
}

#[tokio::test]
async fn command_resume_wins_over_run_config_resume() {
    let mut builder = GraphBuilder::<Bag>::new();
    builder
        .add_node("ask", |_state: &Bag| {
            let answer = interrupt(json!("need"))?;
            Ok(BagUpdate {
                note: Some(answer.as_str().unwrap_or("").to_string()),
                ..BagUpdate::default()
            })
        })
        .add_edge(START, "ask")
        .add_edge("ask", END);
    let graph = builder.compile().expect("compile");
    let (_saver, config) = memory_config("command-resume-wins");
    graph.invoke(&bag(1, &[]), &config).await.expect("pause");
    let resumed = finished(
        graph
            .invoke(
                Command::new().resume(Resume::Value(json!("from-command"))),
                &config.resume(Resume::Value(json!("from-config"))),
            )
            .await
            .expect("resume"),
    );
    assert_eq!(resumed.note, "from-command");
}

#[tokio::test]
async fn parent_command_from_a_node_does_not_commit() {
    let next_ran = Arc::new(AtomicBool::new(false));
    let next_flag = Arc::clone(&next_ran);
    let mut builder = GraphBuilder::<Bag>::new();
    builder
        .add_node("actor", |_state: &Bag| {
            Ok(Command::parent()
                .update(BagUpdate {
                    n: Some(99),
                    note: Some("nope".to_string()),
                    ..BagUpdate::default()
                })
                .goto("next"))
        })
        .add_node("next", move |_state: &Bag| {
            next_flag.store(true, Ordering::SeqCst);
            Ok(append("next"))
        })
        .add_edge(START, "actor")
        .add_edge("actor", "next")
        .add_edge("next", END);
    let graph = builder.compile().expect("compile");
    let (_saver, config) = memory_config("parent-node");
    let error = graph
        .invoke(&bag(1, &["seed"]), &config)
        .await
        .expect_err("parent");
    assert_eq!(error, Error::ParentGraph);
    assert_eq!(error.to_string(), "parent routing requires a subgraph");
    assert!(!next_ran.load(Ordering::SeqCst));

    let snapshot = graph.get_state(&config).await.expect("checkpoint");
    assert_eq!(snapshot.values.n, 1);
    assert_eq!(snapshot.values.note, "");
    assert_eq!(snapshot.values.items, ["seed".to_string()]);
    assert_eq!(snapshot.metadata.step, -1);
    assert_eq!(snapshot.metadata.source, CheckpointSource::Input);
    assert_eq!(snapshot.next, ["actor".to_string()]);
}

#[tokio::test]
async fn parent_command_input_writes_no_checkpoint() {
    let ran = Arc::new(AtomicBool::new(false));
    let ran_flag = Arc::clone(&ran);
    let mut builder = GraphBuilder::<Bag>::new();
    builder
        .add_node("actor", move |_state: &Bag| {
            ran_flag.store(true, Ordering::SeqCst);
            Ok(append("ran"))
        })
        .add_edge(START, "actor")
        .add_edge("actor", END);
    let graph = builder.compile().expect("compile");
    let (_saver, config) = memory_config("parent-input");
    let error = graph
        .invoke(
            Command::parent().update(BagUpdate {
                n: Some(99),
                note: Some("nope".to_string()),
                ..BagUpdate::default()
            }),
            &config,
        )
        .await
        .expect_err("parent input");
    assert_eq!(error, Error::ParentGraph);
    assert_eq!(error.to_string(), "parent routing requires a subgraph");
    assert!(!ran.load(Ordering::SeqCst));
    let missing = graph.get_state(&config).await.expect_err("no checkpoint");
    assert!(matches!(missing, Error::Checkpoint(_)), "{missing:?}");
}

#[tokio::test]
async fn invoke_command_goto_replaces_start_edges() {
    let skipped = Arc::new(AtomicBool::new(false));
    let skipped_flag = Arc::clone(&skipped);
    let mut builder = GraphBuilder::<Bag>::new();
    builder
        .add_node("skipped", move |_state: &Bag| {
            skipped_flag.store(true, Ordering::SeqCst);
            Ok(append("skipped"))
        })
        .add_node("taken", |_state: &Bag| Ok(append("taken")))
        .add_edge(START, "skipped")
        .add_edge("skipped", END)
        .add_edge("taken", END);
    let graph = builder.compile().expect("compile");
    let result = finished(
        graph
            .invoke(
                Command::new()
                    .update(BagUpdate {
                        n: Some(5),
                        items: Some(vec!["from-input".to_string()]),
                        ..BagUpdate::default()
                    })
                    .goto("taken"),
                &RunConfig::default(),
            )
            .await
            .expect("start goto"),
    );
    assert!(!skipped.load(Ordering::SeqCst));
    assert_eq!(result.n, 5);
    assert_eq!(
        result.items,
        ["from-input".to_string(), "taken".to_string()]
    );
}

async fn step(state: &Bag) -> Result<Command<BagUpdate>, Error> {
    Ok(Command::new().update(BagUpdate {
        n: Some(state.n + 1),
        ..BagUpdate::default()
    }))
}

#[tokio::test]
async fn command_without_goto_still_runs_the_router() {
    let mut builder = GraphBuilder::<Bag>::new();
    builder
        .add_node("step", step)
        .add_edge(START, "step")
        .add_conditional_edges(
            "step",
            |state: &Bag| {
                if state.n >= 2 {
                    Ok(END)
                } else {
                    Ok("step")
                }
            },
        );
    let graph = builder.compile().expect("compile");
    let result = finished(
        graph
            .invoke(&bag(0, &[]), &RunConfig::default())
            .await
            .expect("loop"),
    );
    assert_eq!(result.n, 2);
}

#[tokio::test]
async fn unknown_command_goto_does_not_commit() {
    let mut builder = GraphBuilder::<Bag>::new();
    builder
        .add_node("choose", |_state: &Bag| {
            Ok(Command::new()
                .update(BagUpdate {
                    note: Some("wrote".to_string()),
                    ..BagUpdate::default()
                })
                .goto("missing"))
        })
        .add_edge(START, "choose")
        .add_edge("choose", END);
    let graph = builder.compile().expect("compile");
    let (_saver, config) = memory_config("unknown-goto");
    let error = graph
        .invoke(&bag(3, &["kept"]), &config)
        .await
        .expect_err("unknown");
    match error {
        Error::UnknownNode(name) => assert_eq!(name, "missing"),
        other => panic!("expected UnknownNode, got {other:?}"),
    }
    let snapshot = graph.get_state(&config).await.expect("checkpoint");
    assert_eq!(snapshot.values.note, "");
    assert_eq!(snapshot.values.n, 3);
    assert_eq!(snapshot.values.items, ["kept".to_string()]);
    assert_eq!(snapshot.next, ["choose".to_string()]);
}

#[tokio::test]
async fn command_goto_survives_a_sibling_interrupt() {
    let static_ran = Arc::new(AtomicBool::new(false));
    let static_flag = Arc::clone(&static_ran);
    let mut builder = GraphBuilder::<Bag>::new();
    builder
        .add_node("choose", |_state: &Bag| {
            Ok(Command::new()
                .update(BagUpdate {
                    n: Some(3),
                    ..BagUpdate::default()
                })
                .goto("later"))
        })
        .add_node("ask", |_state: &Bag| {
            let answer = interrupt(json!("wait"))?;
            Ok(BagUpdate {
                note: Some(answer.as_str().unwrap_or("").to_string()),
                ..BagUpdate::default()
            })
        })
        .add_node("later", |_state: &Bag| Ok(append("later")))
        .add_node("static_next", move |_state: &Bag| {
            static_flag.store(true, Ordering::SeqCst);
            Ok(append("static"))
        })
        .add_edge(START, "choose")
        .add_edge(START, "ask")
        .add_edge("choose", "static_next")
        .add_edge("static_next", END)
        .add_edge("ask", END)
        .add_edge("later", END);
    let graph = builder.compile().expect("compile");
    let (_saver, config) = memory_config("goto-replay");
    let paused = graph
        .invoke(&bag(1, &["seed"]), &config)
        .await
        .expect("pause");
    assert!(matches!(paused, RunOutput::Interrupted { .. }));
    assert!(!static_ran.load(Ordering::SeqCst));

    let resumed = finished(
        graph
            .invoke(Command::new().resume(Resume::Value(json!("go"))), &config)
            .await
            .expect("resume"),
    );
    assert!(!static_ran.load(Ordering::SeqCst));
    assert_eq!(resumed.n, 3);
    assert_eq!(resumed.note, "go");
    assert_eq!(resumed.items, ["seed".to_string(), "later".to_string()]);
}
