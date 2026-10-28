//! In-memory checkpointer: resume, history, fork, and `update_state`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use langgraph::{
    CheckpointConfig, CheckpointSource, Checkpointer, CompiledGraph, Error, Field, GraphBuilder,
    GraphState, InMemorySaver, Reducer, RunConfig, END, START,
};
use serde_json::{json, Value};

#[derive(Clone, Debug, PartialEq)]
struct Bundle {
    n: i64,
    tag: String,
    notes: Vec<String>,
    messages: Vec<Value>,
}

#[derive(Clone, Debug, Default)]
struct BundleUpdate {
    n: Option<i64>,
    tag: Option<String>,
    notes: Option<Vec<String>>,
    messages: Option<Vec<Value>>,
}

impl GraphState for Bundle {
    type Update = BundleUpdate;

    fn schema() -> &'static [Field] {
        static SCHEMA: &[Field] = &[
            Field {
                name: "n",
                reducer: Reducer::Overwrite,
            },
            Field {
                name: "tag",
                reducer: Reducer::Overwrite,
            },
            Field {
                name: "notes",
                reducer: Reducer::Append,
            },
            Field {
                name: "messages",
                reducer: Reducer::AddMessages,
            },
        ];
        SCHEMA
    }

    fn to_channel_values(&self) -> Result<std::collections::BTreeMap<String, Value>, Error> {
        Ok(std::collections::BTreeMap::from([
            ("n".to_string(), json!(self.n)),
            ("tag".to_string(), Value::String(self.tag.clone())),
            ("notes".to_string(), string_array(&self.notes)),
            ("messages".to_string(), Value::Array(self.messages.clone())),
        ]))
    }

    fn update_to_channel_values(
        update: &Self::Update,
    ) -> Result<std::collections::BTreeMap<String, Value>, Error> {
        let mut values = std::collections::BTreeMap::new();
        if let Some(n) = update.n {
            values.insert("n".to_string(), json!(n));
        }
        if let Some(tag) = &update.tag {
            values.insert("tag".to_string(), Value::String(tag.clone()));
        }
        if let Some(notes) = &update.notes {
            values.insert("notes".to_string(), string_array(notes));
        }
        if let Some(messages) = &update.messages {
            values.insert("messages".to_string(), Value::Array(messages.clone()));
        }
        Ok(values)
    }

    fn from_channel_values(
        values: &std::collections::BTreeMap<String, Value>,
    ) -> Result<Self, Error> {
        Ok(Bundle {
            n: match values.get("n") {
                Some(value) => value
                    .as_i64()
                    .ok_or_else(|| Error::InvalidUpdate("channel 'n' is not an integer".into()))?,
                None => 0,
            },
            tag: match values.get("tag") {
                Some(value) => value.as_str().unwrap_or_default().to_string(),
                None => String::new(),
            },
            notes: match values.get("notes") {
                Some(value) => string_vec(value)?,
                None => Vec::new(),
            },
            messages: match values.get("messages") {
                Some(Value::Array(items)) => items.clone(),
                Some(_) => {
                    return Err(Error::InvalidUpdate(
                        "channel 'messages' is not an array".into(),
                    ))
                }
                None => Vec::new(),
            },
        })
    }
}

fn string_array(items: &[String]) -> Value {
    Value::Array(items.iter().cloned().map(Value::String).collect())
}

fn string_vec(value: &Value) -> Result<Vec<String>, Error> {
    let array = value
        .as_array()
        .ok_or_else(|| Error::InvalidUpdate("channel 'notes' is not an array".into()))?;
    array
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_string)
                .ok_or_else(|| Error::InvalidUpdate("notes entry is not a string".into()))
        })
        .collect()
}

fn message(id: &str, content: &str) -> Value {
    json!({"type": "human", "id": id, "content": content})
}

fn message_ids(messages: &[Value]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|message| {
            message
                .get("id")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .collect()
}

fn bundle(n: i64, tag: &str, notes: &[&str], messages: Vec<Value>) -> Bundle {
    Bundle {
        n,
        tag: tag.to_string(),
        notes: notes.iter().map(|note| (*note).to_string()).collect(),
        messages,
    }
}

fn compile(build: impl FnOnce(&mut GraphBuilder<Bundle>)) -> CompiledGraph<Bundle> {
    let mut builder = GraphBuilder::new();
    build(&mut builder);
    builder.compile().expect("compile")
}

fn take_finished(output: langgraph::RunOutput<Bundle>) -> Bundle {
    match output {
        langgraph::RunOutput::Finished { state } => state,
        langgraph::RunOutput::Interrupted { .. } => panic!("expected Finished"),
    }
}

fn saver_config(thread_id: &str) -> (Arc<dyn Checkpointer>, RunConfig) {
    let saver: Arc<dyn Checkpointer> = Arc::new(InMemorySaver::new());
    let config = RunConfig::new()
        .checkpointer(Arc::clone(&saver))
        .thread_id(thread_id);
    (saver, config)
}

fn root(thread_id: &str) -> CheckpointConfig {
    CheckpointConfig {
        thread_id: thread_id.to_string(),
        checkpoint_ns: String::new(),
        checkpoint_id: None,
    }
}

#[tokio::test]
async fn second_invoke_merges_input_into_stored_channels() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_node = Arc::clone(&calls);
    let graph = compile(|builder| {
        builder
            .add_node("bump", move |state: &Bundle| {
                calls_node.fetch_add(1, Ordering::SeqCst);
                Ok(BundleUpdate {
                    n: Some(state.n + 1),
                    ..BundleUpdate::default()
                })
            })
            .add_edge(START, "bump")
            .add_edge("bump", END);
    });
    let (saver, config) = saver_config("merge");
    let first = take_finished(
        graph
            .invoke(
                &bundle(1, "kept", &["a"], vec![message("m1", "hi")]),
                &config,
            )
            .await
            .expect("first"),
    );
    assert_eq!(first.n, 2);
    assert_eq!(first.tag, "kept");
    assert_eq!(first.notes, ["a".to_string()]);
    assert_eq!(message_ids(&first.messages), ["m1".to_string()]);

    let second = take_finished(
        graph
            .invoke(
                &bundle(9, "replaced", &["b"], vec![message("m2", "there")]),
                &config,
            )
            .await
            .expect("second"),
    );
    assert_eq!(second.n, 10);
    assert_eq!(second.tag, "replaced");
    assert_eq!(second.notes, ["a".to_string(), "b".to_string()]);
    assert_eq!(
        message_ids(&second.messages),
        ["m1".to_string(), "m2".to_string()]
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);

    let state = graph.get_state(&config).await.expect("state");
    assert_eq!(state.values, second);
    assert!(state.next.is_empty());
    assert!(state.tasks.is_empty());
    assert!(state.interrupts.is_empty());
    assert_eq!(state.metadata.source, CheckpointSource::Loop);

    let stored = saver
        .get_tuple(&root("merge"))
        .await
        .expect("tuple")
        .expect("latest");
    assert_eq!(stored.checkpoint.v, 1);
    assert_eq!(stored.checkpoint.channel_versions.get("n"), Some(&4));
    assert_eq!(stored.checkpoint.channel_versions.get("notes"), Some(&2));
    assert_eq!(stored.checkpoint.channel_versions.get("messages"), Some(&2));
    assert_eq!(stored.checkpoint.done.get("__start__"), Some(&2));
    assert_eq!(stored.checkpoint.done.get("bump"), Some(&2));
    assert_eq!(
        stored
            .checkpoint
            .versions_seen
            .get("bump")
            .and_then(|seen| seen.get("done:__start__")),
        Some(&2)
    );
}

#[tokio::test]
async fn missing_thread_id_does_not_run_nodes() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_node = Arc::clone(&calls);
    let graph = compile(|builder| {
        builder
            .add_node("bump", move |_state: &Bundle| {
                calls_node.fetch_add(1, Ordering::SeqCst);
                Ok(BundleUpdate::default())
            })
            .add_edge(START, "bump")
            .add_edge("bump", END);
    });
    let saver: Arc<dyn Checkpointer> = Arc::new(InMemorySaver::new());
    let missing = RunConfig::new().checkpointer(Arc::clone(&saver));
    let error = graph
        .invoke(&bundle(0, "", &[], vec![]), &missing)
        .await
        .expect_err("missing thread");
    assert_eq!(error, Error::MissingThreadId);

    let empty = RunConfig::new().checkpointer(saver).thread_id("");
    let error = graph
        .invoke(&bundle(0, "", &[], vec![]), &empty)
        .await
        .expect_err("empty thread");
    assert_eq!(error, Error::MissingThreadId);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn get_state_history_is_newest_first_and_links_parents() {
    let graph = compile(|builder| {
        builder
            .add_node("bump", |state: &Bundle| {
                Ok(BundleUpdate {
                    n: Some(state.n + 1),
                    ..BundleUpdate::default()
                })
            })
            .add_edge(START, "bump")
            .add_edge("bump", END);
    });
    let (_saver, config) = saver_config("history");
    graph
        .invoke(&bundle(1, "t", &[], vec![]), &config)
        .await
        .expect("run");

    let history = graph
        .get_state_history(&config, None, None)
        .await
        .expect("history");
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].metadata.source, CheckpointSource::Loop);
    assert_eq!(history[0].metadata.step, 0);
    assert!(history[0].next.is_empty());
    assert_eq!(history[0].values.n, 2);
    assert_eq!(history[1].metadata.source, CheckpointSource::Input);
    assert_eq!(history[1].metadata.step, -1);
    assert_eq!(history[1].next, ["bump".to_string()]);
    assert_eq!(history[1].tasks.len(), 1);
    assert_eq!(history[1].tasks[0].name, "bump");
    assert_eq!(history[1].values.n, 1);
    assert!(history[1].interrupts.is_empty());
    assert!(history[1].parent_config.is_none());
    assert_eq!(
        history[0]
            .parent_config
            .as_ref()
            .and_then(|parent| parent.get_checkpoint_id()),
        history[1].config.get_checkpoint_id()
    );

    let input_id = history[1].config.get_checkpoint_id().unwrap().to_string();
    let older = graph
        .get_state_history(&config, Some(&input_id), None)
        .await
        .expect("before");
    assert!(older.is_empty());
    let limited = graph
        .get_state_history(&config, None, Some(1))
        .await
        .expect("limit");
    assert_eq!(limited.len(), 1);
    assert_eq!(limited[0].metadata.source, CheckpointSource::Loop);

    let at_input = config.clone().checkpoint_id(&input_id);
    let snapshot = graph.get_state(&at_input).await.expect("input snapshot");
    assert_eq!(snapshot.values.n, 1);
    assert_eq!(snapshot.next, ["bump".to_string()]);

    let other_ns = config.checkpoint_ns("child");
    let missing = graph.get_state(&other_ns).await.expect_err("other ns");
    assert!(matches!(missing, Error::Checkpoint(ref message) if message.contains("no checkpoint")));
}

#[tokio::test]
async fn checkpoint_id_forks_a_child_and_keeps_the_previous_tip() {
    let graph = compile(|builder| {
        builder
            .add_node("a", |state: &Bundle| {
                Ok(BundleUpdate {
                    n: Some(state.n + 1),
                    tag: Some("a".to_string()),
                    ..BundleUpdate::default()
                })
            })
            .add_node("b", |_state: &Bundle| {
                Ok(BundleUpdate {
                    tag: Some("b".to_string()),
                    ..BundleUpdate::default()
                })
            })
            .add_edge(START, "a")
            .add_edge("a", "b")
            .add_edge("b", END);
    });
    let (_saver, config) = saver_config("fork");
    let finished = take_finished(
        graph
            .invoke(&bundle(1, "", &[], vec![]), &config)
            .await
            .expect("first run"),
    );
    assert_eq!(finished.n, 2);
    assert_eq!(finished.tag, "b");

    let history = graph
        .get_state_history(&config, None, None)
        .await
        .expect("history");
    let tip = history[0].clone();
    let tip_id = tip.config.get_checkpoint_id().unwrap().to_string();
    let tip_parent = tip
        .parent_config
        .as_ref()
        .and_then(|parent| parent.get_checkpoint_id())
        .unwrap()
        .to_string();
    let input = history
        .iter()
        .find(|snapshot| snapshot.metadata.source == CheckpointSource::Input)
        .expect("input");
    let input_id = input.config.get_checkpoint_id().unwrap().to_string();

    let forked = take_finished(
        graph
            .invoke(
                &bundle(10, "x", &[], vec![]),
                &config.clone().checkpoint_id(&input_id),
            )
            .await
            .expect("fork"),
    );
    assert_eq!(forked.n, 11);
    assert_eq!(forked.tag, "b");

    let after = graph
        .get_state_history(&config, None, None)
        .await
        .expect("after");
    assert_ne!(after[0].config.get_checkpoint_id(), Some(tip_id.as_str()));
    let old_tip = graph
        .get_state(&config.clone().checkpoint_id(&tip_id))
        .await
        .expect("old tip");
    assert_eq!(old_tip.values.n, tip.values.n);
    assert_eq!(old_tip.values.tag, tip.values.tag);
    assert_eq!(old_tip.metadata, tip.metadata);
    assert_eq!(
        old_tip
            .parent_config
            .as_ref()
            .and_then(|parent| parent.get_checkpoint_id()),
        Some(tip_parent.as_str())
    );

    let fork_input = after
        .iter()
        .find(|snapshot| {
            snapshot.metadata.source == CheckpointSource::Input
                && snapshot
                    .parent_config
                    .as_ref()
                    .and_then(|parent| parent.get_checkpoint_id())
                    == Some(input_id.as_str())
        })
        .expect("fork input");
    assert_ne!(
        fork_input.config.get_checkpoint_id(),
        Some(input_id.as_str())
    );
    assert_eq!(fork_input.values.n, 10);
    assert_eq!(fork_input.values.tag, "x");

    let latest = graph.get_state(&config).await.expect("latest");
    assert_eq!(latest.values, forked);
}

#[tokio::test]
async fn update_state_moves_next_to_that_nodes_successors() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_b = Arc::clone(&calls);
    let graph = compile(|builder| {
        builder
            .add_node("a", |_state: &Bundle| Ok(BundleUpdate::default()))
            .add_node("b", move |_state: &Bundle| {
                calls_b.fetch_add(1, Ordering::SeqCst);
                Ok(BundleUpdate {
                    tag: Some("ran".to_string()),
                    ..BundleUpdate::default()
                })
            })
            .add_edge(START, "a")
            .add_edge("a", "b")
            .add_edge("b", END);
    });
    let (_saver, config) = saver_config("update");
    let updated = graph
        .update_state(
            &config,
            &BundleUpdate {
                tag: Some("patched".to_string()),
                ..BundleUpdate::default()
            },
            "a",
        )
        .await
        .expect("update");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(updated.get_checkpoint_id().is_some());

    let state = graph.get_state(&config).await.expect("state");
    assert_eq!(state.next, ["b".to_string()]);
    assert_eq!(state.tasks.len(), 1);
    assert_eq!(state.values.tag, "patched");
    assert_eq!(state.metadata.source, CheckpointSource::Update);
    assert!(state.interrupts.is_empty());
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let unknown = graph
        .update_state(&config, &BundleUpdate::default(), "missing")
        .await
        .expect_err("unknown node");
    assert!(matches!(unknown, Error::UnknownNode(ref name) if name == "missing"));
}

#[tokio::test]
async fn update_state_follows_a_conditional_router_without_running_the_node() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_node = Arc::clone(&calls);
    let graph = compile(|builder| {
        builder
            .add_node("tick", move |_state: &Bundle| {
                calls_node.fetch_add(1, Ordering::SeqCst);
                Ok(BundleUpdate::default())
            })
            .add_edge(START, "tick")
            .add_conditional_edges(
                "tick",
                |state: &Bundle| {
                    if state.n >= 3 {
                        Ok(END)
                    } else {
                        Ok("tick")
                    }
                },
            );
    });
    let (_saver, config) = saver_config("router");
    graph
        .update_state(
            &config,
            &BundleUpdate {
                n: Some(1),
                ..BundleUpdate::default()
            },
            "tick",
        )
        .await
        .expect("continue");
    let looping = graph.get_state(&config).await.expect("looping");
    assert_eq!(looping.next, ["tick".to_string()]);

    graph
        .update_state(
            &config,
            &BundleUpdate {
                n: Some(3),
                ..BundleUpdate::default()
            },
            "tick",
        )
        .await
        .expect("stop");
    let done = graph.get_state(&config).await.expect("done");
    assert!(done.next.is_empty());
    assert_eq!(done.values.n, 3);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn resume_continues_scheduled_nodes_instead_of_restarting() {
    let a_calls = Arc::new(AtomicUsize::new(0));
    let b_calls = Arc::new(AtomicUsize::new(0));
    let a_node = Arc::clone(&a_calls);
    let b_node = Arc::clone(&b_calls);
    let graph = compile(|builder| {
        builder
            .add_node("a", move |state: &Bundle| {
                a_node.fetch_add(1, Ordering::SeqCst);
                Ok(BundleUpdate {
                    tag: Some(format!("a{}", state.n)),
                    ..BundleUpdate::default()
                })
            })
            .add_node("b", move |state: &Bundle| {
                b_node.fetch_add(1, Ordering::SeqCst);
                Ok(BundleUpdate {
                    tag: Some(format!("b{}", state.n)),
                    ..BundleUpdate::default()
                })
            })
            .add_edge(START, "a")
            .add_edge("a", "b")
            .add_edge("b", END);
    });
    let (_saver, config) = saver_config("partial");
    let error = graph
        .invoke(
            &bundle(1, "", &[], vec![]),
            &config.clone().recursion_limit(1),
        )
        .await
        .expect_err("limit");
    assert_eq!(error, Error::RecursionLimit(1));
    assert_eq!(a_calls.load(Ordering::SeqCst), 1);
    assert_eq!(b_calls.load(Ordering::SeqCst), 0);

    let paused = graph.get_state(&config).await.expect("paused");
    assert_eq!(paused.next, ["b".to_string()]);
    assert_eq!(paused.values.tag, "a1");
    assert_eq!(paused.metadata.step, 0);
    assert!(paused.interrupts.is_empty());

    let resumed = take_finished(
        graph
            .invoke(&bundle(4, "input", &["later"], vec![]), &config)
            .await
            .expect("resume"),
    );
    assert_eq!(a_calls.load(Ordering::SeqCst), 1);
    assert_eq!(b_calls.load(Ordering::SeqCst), 1);
    assert_eq!(resumed.n, 4);
    assert_eq!(resumed.tag, "b4");
    assert_eq!(resumed.notes, ["later".to_string()]);

    let history = graph
        .get_state_history(&config, None, None)
        .await
        .expect("history");
    assert_eq!(history[0].metadata.source, CheckpointSource::Loop);
    assert_eq!(history[0].metadata.step, 1);
    assert_eq!(history[1].metadata.source, CheckpointSource::Input);
    assert_eq!(history[1].metadata.step, 0);
}

#[tokio::test]
async fn unknown_checkpoint_id_is_checkpoint_error_before_nodes_run() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_node = Arc::clone(&calls);
    let graph = compile(|builder| {
        builder
            .add_node("bump", move |_state: &Bundle| {
                calls_node.fetch_add(1, Ordering::SeqCst);
                Ok(BundleUpdate::default())
            })
            .add_edge(START, "bump")
            .add_edge("bump", END);
    });
    let (_saver, config) = saver_config("missing-id");
    let error = graph
        .invoke(
            &bundle(0, "", &[], vec![]),
            &config.checkpoint_id("does-not-exist"),
        )
        .await
        .expect_err("missing checkpoint");
    match error {
        Error::Checkpoint(message) => assert!(message.contains("does-not-exist"), "{message}"),
        other => panic!("expected Checkpoint, got {other:?}"),
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let no_saver = graph
        .get_state(&RunConfig::new().thread_id("t"))
        .await
        .expect_err("no saver");
    assert!(
        matches!(no_saver, Error::Checkpoint(ref message) if message.contains("no checkpointer"))
    );
}
