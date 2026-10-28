//! The facade exposes `invoke` and the recursion-limit config.

use std::collections::BTreeMap;

use serde_json::Value;

struct Counter {
    n: i64,
}

struct CounterUpdate {
    n: Option<i64>,
}

impl langgraph::GraphState for Counter {
    type Update = CounterUpdate;

    fn schema() -> &'static [langgraph::Field] {
        static SCHEMA: &[langgraph::Field] = &[langgraph::Field {
            name: "n",
            reducer: langgraph::Reducer::Overwrite,
        }];
        SCHEMA
    }

    fn to_channel_values(&self) -> Result<BTreeMap<String, Value>, langgraph::Error> {
        Ok(BTreeMap::from([("n".to_string(), Value::from(self.n))]))
    }

    fn update_to_channel_values(
        update: &Self::Update,
    ) -> Result<BTreeMap<String, Value>, langgraph::Error> {
        let mut values = BTreeMap::new();
        if let Some(n) = update.n {
            values.insert("n".to_string(), Value::from(n));
        }
        Ok(values)
    }

    fn from_channel_values(values: &BTreeMap<String, Value>) -> Result<Self, langgraph::Error> {
        let n = values.get("n").and_then(Value::as_i64).ok_or_else(|| {
            langgraph::Error::InvalidUpdate("channel 'n' is not an integer".into())
        })?;
        Ok(Counter { n })
    }
}

#[tokio::test]
async fn facade_invoke_runs_a_finished_graph() {
    assert_eq!(langgraph::DEFAULT_RECURSION_LIMIT, 25);
    assert_eq!(langgraph::RunConfig::default(), langgraph::RunConfig::new());

    let mut builder = langgraph::GraphBuilder::<Counter>::new();
    builder
        .add_node("step", |state: &Counter| {
            Ok(CounterUpdate {
                n: Some(state.n + 1),
            })
        })
        .add_edge(langgraph::START, "step")
        .add_conditional_edges("step", |state: &Counter| {
            if state.n >= 2 {
                Ok(langgraph::END)
            } else {
                Ok("step")
            }
        });
    let graph = builder.compile().expect("compile");
    let result = graph
        .invoke(&Counter { n: 0 }, &langgraph::RunConfig::default())
        .await
        .expect("invoke");
    assert_eq!(result.n, 2);
}
