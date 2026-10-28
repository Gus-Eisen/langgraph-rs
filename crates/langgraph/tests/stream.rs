//! `stream` yields values, updates, and events for a root run.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use langgraph::{
    interrupt, Error, Field, GraphBuilder, GraphState, GraphStream, InMemorySaver, Namespace,
    Reducer, RunConfig, RunOutput, StreamEvent, StreamModes, TaskResult, END, START,
};
use serde_json::{json, Value};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Row {
    step: String,
    items: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct RowUpdate {
    step: Option<String>,
    items: Option<Vec<String>>,
}

impl GraphState for Row {
    type Update = RowUpdate;

    fn schema() -> &'static [Field] {
        static SCHEMA: &[Field] = &[
            Field {
                name: "step",
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
            ("step".to_string(), Value::String(self.step.clone())),
            ("items".to_string(), string_array(&self.items)),
        ]))
    }

    fn update_to_channel_values(
        update: &Self::Update,
    ) -> Result<std::collections::BTreeMap<String, Value>, Error> {
        let mut values = std::collections::BTreeMap::new();
        if let Some(step) = &update.step {
            values.insert("step".to_string(), Value::String(step.clone()));
        }
        if let Some(items) = &update.items {
            values.insert("items".to_string(), string_array(items));
        }
        Ok(values)
    }

    fn from_channel_values(
        values: &std::collections::BTreeMap<String, Value>,
    ) -> Result<Self, Error> {
        let step = values
            .get("step")
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
        Ok(Self { step, items })
    }
}

fn string_array(items: &[String]) -> Value {
    Value::Array(items.iter().cloned().map(Value::String).collect())
}

fn row() -> Row {
    Row {
        step: String::new(),
        items: Vec::new(),
    }
}

fn append(label: &str) -> RowUpdate {
    RowUpdate {
        step: Some(label.to_string()),
        items: Some(vec![label.to_string()]),
    }
}

fn namespace(event: &StreamEvent<Row>) -> &Namespace {
    match event {
        StreamEvent::Values { namespace, .. }
        | StreamEvent::Updates { namespace, .. }
        | StreamEvent::TaskStarted { namespace, .. }
        | StreamEvent::TaskFinished { namespace, .. }
        | StreamEvent::Checkpoint { namespace, .. }
        | StreamEvent::Interrupted { namespace, .. } => namespace,
    }
}

async fn drain(stream: &mut GraphStream<Row>) -> Vec<StreamEvent<Row>> {
    let mut events = Vec::new();
    while let Some(event) = stream.recv().await {
        events.push(event);
    }
    events
}

fn finished(output: RunOutput<Row>) -> Row {
    match output {
        RunOutput::Finished { state } => state,
        RunOutput::Interrupted { .. } => panic!("expected Finished"),
    }
}

#[tokio::test]
async fn two_node_values_match_invoke_and_events_include_checkpoints() {
    let mut builder = GraphBuilder::<Row>::new();
    builder
        .add_node("alpha", |_state: &Row| Ok(append("alpha")))
        .add_node("beta", |_state: &Row| Ok(append("beta")))
        .add_edge(START, "alpha")
        .add_edge("alpha", "beta")
        .add_edge("beta", END);
    let graph = builder.compile().expect("compile");
    let saver = Arc::new(InMemorySaver::new());
    let config = RunConfig::new()
        .checkpointer(saver)
        .thread_id("stream-two-node");

    let invoked = finished(graph.invoke(&row(), &config).await.expect("invoke"));

    let mut stream = graph
        .stream(
            &row(),
            &RunConfig::new()
                .checkpointer(Arc::new(InMemorySaver::new()))
                .thread_id("stream-two-node-live"),
            StreamModes::all(),
        )
        .await
        .expect("stream");
    let events = drain(&mut stream).await;
    let output = finished(stream.output().await.expect("stream output"));

    assert!(events.iter().all(|event| namespace(event).is_root()));
    assert_eq!(output, invoked);
    assert_eq!(output.step, "beta");
    assert_eq!(output.items, ["alpha".to_string(), "beta".to_string()]);

    let snapshots: Vec<&Row> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Values { state, .. } => Some(state),
            _ => None,
        })
        .collect();
    assert_eq!(snapshots.len(), 2);
    assert_eq!(snapshots[0].step, "alpha");
    assert_eq!(snapshots[0].items, ["alpha".to_string()]);
    assert_eq!(snapshots[1], &invoked);

    let updates: Vec<(&str, &RowUpdate)> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Updates { node, update, .. } => Some((node.as_str(), update)),
            _ => None,
        })
        .collect();
    assert_eq!(
        updates,
        [("alpha", &append("alpha")), ("beta", &append("beta")),]
    );

    let started: Vec<&str> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::TaskStarted { node, .. } => Some(node.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(started, ["alpha", "beta"]);

    let finished_nodes: Vec<(&str, bool)> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::TaskFinished { node, result, .. } => {
                Some((node.as_str(), matches!(result, TaskResult::Update(Some(_)))))
            }
            _ => None,
        })
        .collect();
    assert_eq!(finished_nodes, [("alpha", true), ("beta", true)]);
    assert!(events
        .iter()
        .any(|event| matches!(event, StreamEvent::Checkpoint { .. })));
}

#[tokio::test]
async fn updates_follow_task_order_not_completion_order() {
    let released = Arc::new(AtomicBool::new(false));
    let wait_for_second = Arc::clone(&released);
    let signal = Arc::clone(&released);
    let finished_bodies = Arc::new(std::sync::Mutex::new(Vec::new()));
    let first_log = Arc::clone(&finished_bodies);
    let second_log = Arc::clone(&finished_bodies);
    let mut builder = GraphBuilder::<Row>::new();
    builder
        .add_node("first", move |_state: &Row| {
            while !wait_for_second.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(1));
            }
            first_log.lock().expect("log").push("first");
            Ok(RowUpdate {
                items: Some(vec!["first".to_string()]),
                ..RowUpdate::default()
            })
        })
        .add_node("second", move |_state: &Row| {
            second_log.lock().expect("log").push("second");
            signal.store(true, Ordering::SeqCst);
            Ok(RowUpdate {
                items: Some(vec!["second".to_string()]),
                ..RowUpdate::default()
            })
        })
        .add_edge(START, "first")
        .add_edge(START, "second")
        .add_edge("first", END)
        .add_edge("second", END);
    let graph = builder.compile().expect("compile");
    let config = RunConfig::default();
    let invoked = finished(graph.invoke(&row(), &config).await.expect("invoke"));
    released.store(false, Ordering::SeqCst);
    finished_bodies.lock().expect("log").clear();

    let mut stream = graph
        .stream(
            &row(),
            &config,
            StreamModes::values() | StreamModes::updates() | StreamModes::events(),
        )
        .await
        .expect("stream");
    let events = drain(&mut stream).await;
    let output = finished(stream.output().await.expect("stream output"));

    assert_eq!(output, invoked);
    assert_eq!(output.items, ["first".to_string(), "second".to_string()]);
    let snapshots: Vec<&Row> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Values { state, .. } => Some(state),
            _ => None,
        })
        .collect();
    assert_eq!(snapshots, [&output]);

    let update_nodes: Vec<&str> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Updates { node, .. } => Some(node.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(update_nodes, ["first", "second"]);
    assert_eq!(
        finished_bodies.lock().expect("log").as_slice(),
        ["second", "first"]
    );
}

#[tokio::test]
async fn task_started_arrives_before_the_node_returns() {
    let seen = Arc::new(AtomicBool::new(false));
    let gate = Arc::clone(&seen);
    let mut builder = GraphBuilder::<Row>::new();
    builder
        .add_node("hold", move |_state: &Row| {
            while !gate.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(1));
            }
            Ok(append("hold"))
        })
        .add_edge(START, "hold")
        .add_edge("hold", END);
    let graph = builder.compile().expect("compile");
    let mut stream = graph
        .stream(&row(), &RunConfig::default(), StreamModes::events())
        .await
        .expect("stream");
    let started = stream.recv().await.expect("task started");
    assert!(matches!(
        started,
        StreamEvent::TaskStarted { ref node, .. } if node == "hold" && namespace(&started).is_root()
    ));
    seen.store(true, Ordering::SeqCst);
    let task_finished = stream.recv().await.expect("task finished");
    assert!(matches!(
        task_finished,
        StreamEvent::TaskFinished { ref node, result: TaskResult::Update(Some(_)), .. } if node == "hold"
    ));
    let output = finished(stream.output().await.expect("output"));
    assert_eq!(output.step, "hold");
}

#[tokio::test]
async fn interrupt_emits_interrupted_and_stops_the_stream() {
    let mut builder = GraphBuilder::<Row>::new();
    builder
        .add_node("ask", |_state: &Row| {
            let _answer = interrupt(json!("approve"))?;
            Ok(append("ask"))
        })
        .add_node("after", |_state: &Row| Ok(append("after")))
        .add_edge(START, "ask")
        .add_edge("ask", "after")
        .add_edge("after", END);
    let graph = builder.compile().expect("compile");
    let config = RunConfig::new()
        .checkpointer(Arc::new(InMemorySaver::new()))
        .thread_id("stream-interrupt");
    let invoked = graph.invoke(&row(), &config).await.expect("invoke");
    let RunOutput::Interrupted {
        state: invoked_state,
        interrupts: invoked_interrupts,
    } = &invoked
    else {
        panic!("expected invoke to pause");
    };

    let mut stream = graph
        .stream(
            &row(),
            &RunConfig::new()
                .checkpointer(Arc::new(InMemorySaver::new()))
                .thread_id("stream-interrupt-live"),
            StreamModes::all(),
        )
        .await
        .expect("stream");
    let events = drain(&mut stream).await;
    let streamed = stream.output().await.expect("stream output");

    assert!(events.iter().all(|event| namespace(event).is_root()));
    assert!(matches!(
        events.last(),
        Some(StreamEvent::Interrupted { .. })
    ));
    assert!(!events.iter().any(|event| matches!(
        event,
        StreamEvent::TaskStarted { node, .. } if node == "after"
    )));
    match &streamed {
        RunOutput::Interrupted { state, interrupts } => {
            assert_eq!(state, invoked_state);
            assert_eq!(interrupts, invoked_interrupts);
            assert_eq!(state.step, "");
            assert!(state.items.is_empty());
            assert_eq!(interrupts.len(), 1);
            assert_eq!(interrupts[0].value, json!("approve"));
        }
        RunOutput::Finished { .. } => panic!("expected Interrupted"),
    }
    match events.last() {
        Some(StreamEvent::Interrupted {
            state, interrupts, ..
        }) => {
            assert_eq!(state, invoked_state);
            assert_eq!(interrupts, invoked_interrupts);
        }
        other => panic!("expected Interrupted last, got {other:?}"),
    }
}
