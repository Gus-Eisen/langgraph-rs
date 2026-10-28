//! The facade re-exports the builder, and an outside crate can register nodes.

use std::collections::BTreeMap;
use std::future::Future;

use serde_json::Value;

struct Count;

impl langgraph::GraphState for Count {
    type Update = ();

    fn schema() -> &'static [langgraph::Field] {
        static SCHEMA: &[langgraph::Field] = &[langgraph::Field {
            name: "n",
            reducer: langgraph::Reducer::Overwrite,
        }];
        SCHEMA
    }

    fn to_channel_values(&self) -> Result<BTreeMap<String, Value>, langgraph::Error> {
        Ok(BTreeMap::from([("n".to_string(), Value::from(0))]))
    }

    fn update_to_channel_values(
        _update: &Self::Update,
    ) -> Result<BTreeMap<String, Value>, langgraph::Error> {
        Ok(BTreeMap::new())
    }

    fn from_channel_values(_values: &BTreeMap<String, Value>) -> Result<Self, langgraph::Error> {
        Ok(Count)
    }
}

fn panic_node(_state: &Count) -> Result<(), langgraph::Error> {
    panic!("compile must not run nodes");
}

async fn panic_async(_state: &Count) -> Result<(), langgraph::Error> {
    panic!("compile must not run async nodes");
}

fn deferred(state: &Count) -> impl Future<Output = Result<(), langgraph::Error>> + Send + '_ {
    let _ = state;
    async { panic!("compile must not run deferred nodes") }
}

#[test]
fn facade_compiles_a_cycle_without_running_nodes() {
    let mut builder = langgraph::GraphBuilder::<Count>::new();
    builder
        .add_node("a", panic_node)
        .add_node("b", panic_async)
        .add_node("c", deferred)
        .add_edge(langgraph::START, "a")
        .add_edge("a", "b")
        .add_edge(["b", "c"], "a")
        .add_conditional_edges("a", |_state: &Count| Ok(langgraph::END))
        .add_conditional_edges_with_path_map(
            "b",
            |_state: &Count| Ok::<_, langgraph::Error>(langgraph::Route::end()),
            [("stop", langgraph::END)],
        );

    let compiled = builder.compile().expect("cycle");
    let _typed: langgraph::CompiledGraph<Count> = compiled;

    let mut broken = langgraph::GraphBuilder::<Count>::new();
    broken
        .add_node("a", panic_node)
        .add_edge("a", langgraph::END);
    let error = broken.compile().expect_err("missing entry");
    assert!(matches!(error, langgraph::Error::GraphValidation(_)));
}

#[test]
fn facade_route_and_edge_traits_convert() {
    fn destinations<T: langgraph::IntoRoute>(value: T) -> Vec<String> {
        value.into_destinations()
    }
    fn source<T: langgraph::IntoEdgeSource>(value: T) -> langgraph::EdgeSource {
        value.into_edge_source()
    }

    assert_eq!(
        destinations(langgraph::END),
        vec![langgraph::END.to_string()]
    );
    assert_eq!(
        source(langgraph::START),
        langgraph::EdgeSource::One(langgraph::START.to_string())
    );
    assert_eq!(
        source(["a", "b"]),
        langgraph::EdgeSource::Wait(vec!["a".to_string(), "b".to_string()])
    );
}
