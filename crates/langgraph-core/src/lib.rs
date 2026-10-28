//! Graph state, reducers, the builder, and the Pregel runtime.
//!
//! Channels are a JSON map with `u64` versions. [`GraphState`] supplies the
//! field list, partial updates, and conversion to and from those values.
//! Overwrite, append, custom, and `add_messages` reducers fold a step in
//! `(task_path, task_id, write_index)` order. [`GraphBuilder`] records nodes
//! and edges. [`GraphBuilder::compile`] checks that structure and does not
//! run the graph. The Pregel runtime is a later milestone. See the repository
//! `ARCHITECTURE.md`.

#![forbid(unsafe_code)]

mod builder;
mod error;
mod messages;
mod reducer;
mod state;

pub use builder::{
    AsyncNodeMarker, CompiledGraph, EdgeSource, GraphBuilder, IntoEdgeSource, IntoRoute, Route,
    SyncNodeMarker, END, START,
};
pub use error::Error;
pub use messages::{add_messages, REMOVE_ALL_MESSAGES};
pub use reducer::{CustomReducer, Field, Reducer};
pub use state::{ChannelMap, GraphState, Write, RESERVED_NAMES};

/// Version of `langgraph-core`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Version of the checkpoint crate the runtime will call.
///
/// The checkpointer API is not re-exported here.
pub const CHECKPOINT_VERSION: &str = langgraph_checkpoint::VERSION;

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::{json, Value};

    use crate::{
        ChannelMap, Error, Field, GraphState, Reducer, Write, REMOVE_ALL_MESSAGES, RESERVED_NAMES,
    };

    fn write(channel: &str, value: Value, path: &[&str], task_id: &str, write_index: u64) -> Write {
        Write {
            channel: channel.to_string(),
            value,
            task_path: path.iter().map(|segment| (*segment).to_string()).collect(),
            task_id: task_id.to_string(),
            write_index,
        }
    }

    fn overwrite_fields(names: &[&'static str]) -> Vec<Field> {
        names
            .iter()
            .map(|name| Field {
                name,
                reducer: Reducer::Overwrite,
            })
            .collect()
    }

    #[test]
    fn version_is_set() {
        assert!(!crate::VERSION.is_empty());
        assert_eq!(crate::CHECKPOINT_VERSION, crate::VERSION);
    }

    #[tokio::test]
    async fn tokio_runtime_executes_a_task() {
        let value = tokio::spawn(async { 7 }).await.expect("join task");
        assert_eq!(value, 7);
    }

    #[test]
    fn one_write_updates_the_channel_and_bumps_its_version() {
        let mut map = ChannelMap::new(&overwrite_fields(&["count"])).expect("schema");
        assert_eq!(map.version("count"), Some(0));
        assert_eq!(map.get("count"), None);

        map.apply_writes(&[write("count", json!(7), &[], "task", 0)])
            .expect("one overwrite");

        assert_eq!(map.get("count"), Some(&json!(7)));
        assert_eq!(map.version("count"), Some(1));
        assert_eq!(map.versions().get("count"), Some(&1));
    }

    #[test]
    fn untouched_field_keeps_its_value_and_version() {
        let mut map = ChannelMap::new(&overwrite_fields(&["kept", "changed"])).expect("schema");
        map.apply_writes(&[
            write("kept", json!("stay"), &[], "setup", 0),
            write("changed", json!(1), &[], "setup", 1),
        ])
        .expect("seed both fields");
        assert_eq!(map.version("kept"), Some(1));
        assert_eq!(map.version("changed"), Some(1));

        map.apply_writes(&[write("changed", json!(2), &[], "next", 0)])
            .expect("write one field");

        assert_eq!(map.get("kept"), Some(&json!("stay")));
        assert_eq!(map.version("kept"), Some(1));
        assert_eq!(map.get("changed"), Some(&json!(2)));
        assert_eq!(map.version("changed"), Some(2));
    }

    #[test]
    fn two_overwrites_in_one_step_are_invalid_update() {
        let mut map = ChannelMap::new(&[
            Field {
                name: "count",
                reducer: Reducer::Overwrite,
            },
            Field {
                name: "items",
                reducer: Reducer::Append,
            },
        ])
        .expect("schema");
        map.apply_writes(&[write("count", json!(1), &[], "setup", 0)])
            .expect("seed");
        let values_before = map.values();
        let versions_before = map.versions();

        let error = map
            .apply_writes(&[
                write("count", json!(2), &[], "a", 0),
                write("count", json!(3), &[], "b", 0),
                write("items", json!([1]), &[], "a", 1),
            ])
            .expect_err("two overwrites");

        match error {
            Error::InvalidUpdate(message) => {
                assert!(message.contains("count"), "{message}");
                assert!(message.contains("overwrite"), "{message}");
            }
            other => panic!("expected InvalidUpdate, got {other:?}"),
        }
        assert_eq!(map.values(), values_before);
        assert_eq!(map.versions(), versions_before);
        assert_eq!(map.get("items"), None);
        assert_eq!(map.version("items"), Some(0));
    }

    #[test]
    fn append_concatenates_in_task_order() {
        let mut map = ChannelMap::new(&[Field {
            name: "items",
            reducer: Reducer::Append,
        }])
        .expect("schema");

        map.apply_writes(&[
            write("items", json!([2]), &[], "b", 0),
            write("items", json!([1]), &[], "a", 0),
        ])
        .expect("first append step");
        assert_eq!(map.get("items"), Some(&json!([1, 2])));
        assert_eq!(map.version("items"), Some(1));

        map.apply_writes(&[
            write("items", json!([4]), &["n"], "b", 0),
            write("items", json!([3]), &["n"], "a", 0),
        ])
        .expect("append onto the committed array");

        assert_eq!(map.get("items"), Some(&json!([1, 2, 3, 4])));
        assert_eq!(map.version("items"), Some(2));
    }

    #[test]
    fn append_rejects_a_non_array_without_committing() {
        let mut map = ChannelMap::new(&[Field {
            name: "items",
            reducer: Reducer::Append,
        }])
        .expect("schema");
        let values_before = map.values();
        let versions_before = map.versions();

        let error = map
            .apply_writes(&[write("items", json!("nope"), &[], "task", 0)])
            .expect_err("non-array append");

        assert!(matches!(error, Error::InvalidUpdate(_)));
        assert_eq!(map.values(), values_before);
        assert_eq!(map.versions(), versions_before);
        assert_eq!(map.version("items"), Some(0));
    }

    #[test]
    fn custom_fold_follows_task_order_not_completion_order() {
        fn concat(current: Option<&Value>, update: &Value) -> Result<Value, Error> {
            let next = match update {
                Value::String(text) => text.as_str(),
                _ => return Err(Error::InvalidUpdate("custom fold expected a string".into())),
            };
            let previous = match current {
                None => "",
                Some(Value::String(text)) => text.as_str(),
                Some(_) => {
                    return Err(Error::InvalidUpdate(
                        "custom fold expected a string channel".into(),
                    ))
                }
            };
            Ok(Value::String(format!("{previous}{next}")))
        }

        let mut map = ChannelMap::new(&[Field {
            name: "acc",
            reducer: Reducer::Custom(concat),
        }])
        .expect("schema");

        // Submitted in completion order. The fold key is
        // (task_path, task_id, write_index), which yields D, C, A, B.
        map.apply_writes(&[
            write("acc", json!("B"), &["b"], "2", 0),
            write("acc", json!("A"), &["a"], "9", 0),
            write("acc", json!("C"), &["a"], "1", 1),
            write("acc", json!("D"), &["a"], "1", 0),
        ])
        .expect("custom fold");

        assert_eq!(map.get("acc"), Some(&json!("DCAB")));
        assert_eq!(map.version("acc"), Some(1));
    }

    #[test]
    fn empty_step_does_not_bump_a_version() {
        let mut map = ChannelMap::new(&overwrite_fields(&["count", "idle"])).expect("schema");
        map.apply_writes(&[write("count", json!(4), &[], "task", 0)])
            .expect("write");
        assert_eq!(map.version("count"), Some(1));
        assert_eq!(map.version("idle"), Some(0));

        map.apply_writes(&[]).expect("empty step");

        assert_eq!(map.version("count"), Some(1));
        assert_eq!(map.get("count"), Some(&json!(4)));
        assert_eq!(map.version("idle"), Some(0));
        assert_eq!(map.get("idle"), None);
    }

    #[test]
    fn graph_state_round_trips_and_omits_untouched_fields() {
        let state = Sample {
            count: 1,
            label: "a".to_string(),
        };
        let mut map = ChannelMap::from_state(&state).expect("from state");
        assert_eq!(map.version("count"), Some(0));
        assert_eq!(map.version("label"), Some(0));
        assert_eq!(map.to_state::<Sample>().expect("to state"), state);

        let partial = SampleUpdate {
            count: Some(2),
            label: None,
        };
        let writes = Sample::update_to_writes(&partial, &[], "node").expect("writes");
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].channel, "count");
        assert_eq!(writes[0].write_index, 0);
        map.apply_writes(&writes).expect("apply partial");

        assert_eq!(map.version("count"), Some(1));
        assert_eq!(map.version("label"), Some(0));
        assert_eq!(
            map.to_state::<Sample>().expect("read back"),
            Sample {
                count: 2,
                label: "a".to_string(),
            }
        );

        let both = SampleUpdate {
            count: Some(3),
            label: Some("b".to_string()),
        };
        let path = vec!["child".to_string()];
        let writes = Sample::update_to_writes(&both, &path, "node").expect("both fields");
        assert_eq!(writes[0].channel, "count");
        assert_eq!(writes[0].write_index, 0);
        assert_eq!(writes[1].channel, "label");
        assert_eq!(writes[1].write_index, 1);
        assert_eq!(writes[0].task_path, path);
        assert_eq!(writes[0].task_id, "node");
    }

    #[test]
    fn reserved_and_duplicate_field_names_are_rejected() {
        for name in RESERVED_NAMES {
            let error = ChannelMap::new(&[Field {
                name,
                reducer: Reducer::Overwrite,
            }])
            .expect_err("reserved name");
            assert!(matches!(error, Error::GraphValidation(_)), "{error:?}");
        }

        let duplicate =
            ChannelMap::new(&overwrite_fields(&["count", "count"])).expect_err("duplicate");
        assert!(matches!(duplicate, Error::GraphValidation(_)));

        let empty = ChannelMap::new(&[Field {
            name: "",
            reducer: Reducer::Overwrite,
        }])
        .expect_err("empty name");
        assert!(matches!(empty, Error::GraphValidation(_)));
    }

    fn human(id: &str, content: &str) -> Value {
        json!({"type": "human", "id": id, "content": content})
    }

    fn messages_field() -> Field {
        Field {
            name: "messages",
            reducer: Reducer::AddMessages,
        }
    }

    #[test]
    fn add_messages_appends() {
        let mut map = ChannelMap::new(&[messages_field()]).expect("schema");

        map.apply_writes(&[write("messages", json!([human("1", "hi")]), &[], "task", 0)])
            .expect("first message");
        assert_eq!(map.get("messages"), Some(&json!([human("1", "hi")])));
        assert_eq!(map.version("messages"), Some(1));

        map.apply_writes(&[write("messages", human("2", "yo"), &[], "task", 0)])
            .expect("coerce a single message");
        assert_eq!(
            map.get("messages"),
            Some(&json!([human("1", "hi"), human("2", "yo")]))
        );
        assert_eq!(map.version("messages"), Some(2));

        map.apply_writes(&[]).expect("empty step");
        assert_eq!(map.version("messages"), Some(2));
        assert_eq!(
            map.get("messages"),
            Some(&json!([human("1", "hi"), human("2", "yo")]))
        );
    }

    #[test]
    fn add_messages_replaces_same_id_in_place() {
        let mut map = ChannelMap::new(&[messages_field()]).expect("schema");
        map.apply_writes(&[write(
            "messages",
            json!([human("1", "a"), human("2", "b"), human("3", "c")]),
            &[],
            "seed",
            0,
        )])
        .expect("seed");

        map.apply_writes(&[write(
            "messages",
            json!({"type": "ai", "id": "2", "content": "replaced"}),
            &[],
            "task",
            0,
        )])
        .expect("replace id 2");

        assert_eq!(
            map.get("messages"),
            Some(&json!([
                human("1", "a"),
                {"type": "ai", "id": "2", "content": "replaced"},
                human("3", "c")
            ]))
        );
        assert_eq!(map.version("messages"), Some(2));
    }

    #[test]
    fn add_messages_assigns_uuid_v4_when_id_is_empty() {
        let mut map = ChannelMap::new(&[messages_field()]).expect("schema");
        map.apply_writes(&[write(
            "messages",
            json!([
                {"type": "human", "id": "", "content": "empty"},
                {"type": "system", "content": "missing"}
            ]),
            &[],
            "task",
            0,
        )])
        .expect("assign ids");

        let messages = map
            .get("messages")
            .and_then(Value::as_array)
            .expect("message list");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["type"], json!("human"));
        assert_eq!(messages[0]["content"], json!("empty"));
        assert_eq!(messages[1]["type"], json!("system"));
        assert_eq!(messages[1]["content"], json!("missing"));

        let first = messages[0]["id"].as_str().expect("first id");
        let second = messages[1]["id"].as_str().expect("second id");
        assert_ne!(first, second);
        for id in [first, second] {
            let parsed = uuid::Uuid::parse_str(id).expect("uuid");
            assert_eq!(parsed.get_version_num(), 4);
            assert_eq!(parsed.hyphenated().to_string(), id);
        }
        assert_eq!(map.version("messages"), Some(1));
    }

    #[test]
    fn add_messages_removes_existing_id() {
        let mut map = ChannelMap::new(&[messages_field()]).expect("schema");
        map.apply_writes(&[write(
            "messages",
            json!([human("1", "a"), human("2", "b"), human("3", "c")]),
            &[],
            "seed",
            0,
        )])
        .expect("seed");

        map.apply_writes(&[write(
            "messages",
            json!({"type": "remove", "id": "2"}),
            &[],
            "task",
            0,
        )])
        .expect("remove id 2");

        assert_eq!(
            map.get("messages"),
            Some(&json!([human("1", "a"), human("3", "c")]))
        );
        assert_eq!(map.version("messages"), Some(2));
    }

    #[test]
    fn add_messages_rejects_unknown_remove() {
        let mut map = ChannelMap::new(&[
            messages_field(),
            Field {
                name: "note",
                reducer: Reducer::Overwrite,
            },
        ])
        .expect("schema");
        map.apply_writes(&[
            write("messages", json!([human("1", "a")]), &[], "seed", 0),
            write("note", json!("stay"), &[], "seed", 1),
        ])
        .expect("seed");
        let values_before = map.values();
        let versions_before = map.versions();

        let error = map
            .apply_writes(&[
                write(
                    "messages",
                    json!({"type": "remove", "id": "missing"}),
                    &[],
                    "task",
                    0,
                ),
                write("note", json!("changed"), &[], "task", 1),
            ])
            .expect_err("unknown id");

        match error {
            Error::InvalidUpdate(message) => {
                assert!(message.contains("missing"), "{message}");
                assert!(message.contains("unknown"), "{message}");
            }
            other => panic!("expected InvalidUpdate, got {other:?}"),
        }
        assert_eq!(map.values(), values_before);
        assert_eq!(map.versions(), versions_before);
        assert_eq!(map.version("messages"), Some(1));
        assert_eq!(map.version("note"), Some(1));
    }

    #[test]
    fn add_messages_remove_all() {
        let mut map = ChannelMap::new(&[messages_field()]).expect("schema");
        map.apply_writes(&[write(
            "messages",
            json!([human("1", "old"), human("2", "older")]),
            &[],
            "seed",
            0,
        )])
        .expect("seed");

        map.apply_writes(&[write(
            "messages",
            json!([
                {"type": "remove", "id": REMOVE_ALL_MESSAGES},
                human("3", "kept")
            ]),
            &[],
            "task",
            0,
        )])
        .expect("remove all");

        assert_eq!(map.get("messages"), Some(&json!([human("3", "kept")])));
        assert_eq!(map.version("messages"), Some(2));
    }

    #[test]
    fn add_messages_two_writes_follow_task_order() {
        let mut map = ChannelMap::new(&[messages_field()]).expect("schema");
        map.apply_writes(&[write(
            "messages",
            json!([human("old", "committed")]),
            &[],
            "seed",
            0,
        )])
        .expect("seed");

        // Submitted as task "b" then task "a". The fold key orders "a" first,
        // so the remove-all clears the committed message and task "b" appends
        // after the message task "a" kept.
        map.apply_writes(&[
            write("messages", json!([human("b", "from-b")]), &[], "b", 0),
            write(
                "messages",
                json!([
                    {"type": "remove", "id": REMOVE_ALL_MESSAGES},
                    human("a", "from-a")
                ]),
                &[],
                "a",
                0,
            ),
        ])
        .expect("fold in task order");

        assert_eq!(
            map.get("messages"),
            Some(&json!([human("a", "from-a"), human("b", "from-b")]))
        );
        assert_eq!(map.version("messages"), Some(2));
    }

    #[test]
    fn add_messages_state_round_trips_an_array_and_rejects_other_json() {
        let state = MessageState {
            messages: json!([human("1", "hi")]),
        };
        let map = ChannelMap::from_state(&state).expect("from state");
        assert_eq!(map.version("messages"), Some(0));
        assert_eq!(map.to_state::<MessageState>().expect("to state"), state);

        let invalid = MessageState {
            messages: json!("nope"),
        };
        let error = ChannelMap::from_state(&invalid).expect_err("non-array");
        assert!(matches!(error, Error::InvalidUpdate(_)), "{error:?}");
    }

    #[derive(Debug, PartialEq)]
    struct MessageState {
        messages: Value,
    }

    struct MessageUpdate;

    impl GraphState for MessageState {
        type Update = MessageUpdate;

        fn schema() -> &'static [Field] {
            static SCHEMA: &[Field] = &[Field {
                name: "messages",
                reducer: Reducer::AddMessages,
            }];
            SCHEMA
        }

        fn to_channel_values(&self) -> Result<BTreeMap<String, Value>, Error> {
            Ok(BTreeMap::from([(
                "messages".to_string(),
                self.messages.clone(),
            )]))
        }

        fn update_to_channel_values(
            _update: &Self::Update,
        ) -> Result<BTreeMap<String, Value>, Error> {
            Ok(BTreeMap::new())
        }

        fn from_channel_values(values: &BTreeMap<String, Value>) -> Result<Self, Error> {
            let messages = values
                .get("messages")
                .cloned()
                .ok_or_else(|| Error::InvalidUpdate("channel 'messages' is missing".into()))?;
            Ok(Self { messages })
        }
    }

    #[derive(Debug, PartialEq)]
    struct Sample {
        count: i64,
        label: String,
    }

    struct SampleUpdate {
        count: Option<i64>,
        label: Option<String>,
    }

    impl GraphState for Sample {
        type Update = SampleUpdate;

        fn schema() -> &'static [Field] {
            static SCHEMA: &[Field] = &[
                Field {
                    name: "count",
                    reducer: Reducer::Overwrite,
                },
                Field {
                    name: "label",
                    reducer: Reducer::Overwrite,
                },
            ];
            SCHEMA
        }

        fn to_channel_values(&self) -> Result<BTreeMap<String, Value>, Error> {
            Ok(BTreeMap::from([
                ("count".to_string(), json!(self.count)),
                ("label".to_string(), Value::String(self.label.clone())),
            ]))
        }

        fn update_to_channel_values(
            update: &Self::Update,
        ) -> Result<BTreeMap<String, Value>, Error> {
            let mut values = BTreeMap::new();
            if let Some(count) = update.count {
                values.insert("count".to_string(), json!(count));
            }
            if let Some(label) = &update.label {
                values.insert("label".to_string(), Value::String(label.clone()));
            }
            Ok(values)
        }

        fn from_channel_values(values: &BTreeMap<String, Value>) -> Result<Self, Error> {
            let count = values
                .get("count")
                .and_then(Value::as_i64)
                .ok_or_else(|| Error::InvalidUpdate("channel 'count' is not an integer".into()))?;
            let label = values
                .get("label")
                .and_then(Value::as_str)
                .ok_or_else(|| Error::InvalidUpdate("channel 'label' is not a string".into()))?
                .to_string();
            Ok(Self { count, label })
        }
    }
}
