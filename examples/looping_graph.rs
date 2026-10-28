//! Loop a counter until it reaches a limit, then stop.
//!
//! One node increments `n`. A conditional edge routes back to that node until
//! `n` reaches `limit`, then routes to `END`.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use langgraph::{Error, Field, GraphBuilder, GraphState, Reducer, RunConfig, END, START};
use serde_json::Value;

struct Counter {
    n: i64,
    limit: i64,
}

struct CounterUpdate {
    n: Option<i64>,
}

impl GraphState for Counter {
    type Update = CounterUpdate;

    fn schema() -> &'static [Field] {
        static SCHEMA: &[Field] = &[
            Field {
                name: "n",
                reducer: Reducer::Overwrite,
            },
            Field {
                name: "limit",
                reducer: Reducer::Overwrite,
            },
        ];
        SCHEMA
    }

    fn to_channel_values(&self) -> Result<BTreeMap<String, Value>, Error> {
        Ok(BTreeMap::from([
            ("n".to_string(), Value::from(self.n)),
            ("limit".to_string(), Value::from(self.limit)),
        ]))
    }

    fn update_to_channel_values(update: &Self::Update) -> Result<BTreeMap<String, Value>, Error> {
        let mut values = BTreeMap::new();
        if let Some(n) = update.n {
            values.insert("n".to_string(), Value::from(n));
        }
        Ok(values)
    }

    fn from_channel_values(values: &BTreeMap<String, Value>) -> Result<Self, Error> {
        Ok(Counter {
            n: read_i64(values, "n")?,
            limit: read_i64(values, "limit")?,
        })
    }
}

fn read_i64(values: &BTreeMap<String, Value>, key: &str) -> Result<i64, Error> {
    values
        .get(key)
        .and_then(Value::as_i64)
        .ok_or_else(|| Error::InvalidUpdate(format!("channel '{key}' is not an integer")))
}

fn step(state: &Counter) -> Result<CounterUpdate, Error> {
    Ok(CounterUpdate {
        n: Some(state.n + 1),
    })
}

fn route(state: &Counter) -> Result<&'static str, Error> {
    if state.n >= state.limit {
        Ok(END)
    } else {
        Ok("step")
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let mut builder = GraphBuilder::<Counter>::new();
    builder
        .add_node("step", step)
        .add_edge(START, "step")
        .add_conditional_edges("step", route);
    let graph = builder.compile()?;
    let result = graph
        .invoke(&Counter { n: 0, limit: 3 }, &RunConfig::default())
        .await?;
    println!("counter reached {}", result.n);
    assert_eq!(result.n, 3);
    assert_eq!(result.limit, 3);
    Ok(())
}
