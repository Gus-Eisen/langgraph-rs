//! Message JSON and the `add_messages` merge.
//!
//! Core stores messages as JSON values. The typed `Message` newtype belongs
//! to the LLM crate in a later milestone.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};
use uuid::Uuid;

use crate::error::Error;

/// Id of a `remove` message that clears every message before it.
///
/// See [`add_messages`].
pub const REMOVE_ALL_MESSAGES: &str = "__remove_all__";

/// Merge message JSON in `right` into `left`.
///
/// Each side is one message object or an array of message objects. A single
/// object is a one-element list. The result is a JSON array.
///
/// Message objects:
///
/// - `human`, `ai`, and `system` have `type`, `id`, and string `content`.
///   An `ai` message may include `tool_calls`, an array of
///   `{id, name, arguments}` objects. `arguments` is a JSON object.
/// - `tool` has `type`, `id`, string `content`, and `tool_call_id`.
/// - `remove` has `type` and `id`.
///
/// An `id` that is missing or `""` becomes a new UUID v4. A message whose
/// `id` is already in the list replaces that message in place. A `remove` of
/// an existing id deletes it. A `remove` of an unknown id returns
/// [`Error::InvalidUpdate`](Error::InvalidUpdate).
///
/// A `remove` whose id is [`REMOVE_ALL_MESSAGES`] drops `left` and every
/// message in `right` up to and including the last such sentinel. Those
/// messages are discarded, not applied, so a `remove` of an unknown id in
/// that prefix does not fail the merge. The result is the merge of the
/// messages after that sentinel, starting from an empty list.
pub fn add_messages(left: &Value, right: &Value) -> Result<Value, Error> {
    let right_messages = coerce(right, "right")?;
    let left_messages = if right_messages
        .iter()
        .any(|message| message.kind == MessageKind::RemoveAll)
    {
        Vec::new()
    } else {
        coerce(left, "left")?
    };
    Ok(Value::Array(merge_messages(left_messages, right_messages)?))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MessageKind {
    Message,
    Remove,
    RemoveAll,
}

struct PreparedMessage {
    id: String,
    kind: MessageKind,
    value: Value,
}

fn coerce(value: &Value, side: &str) -> Result<Vec<PreparedMessage>, Error> {
    match value {
        Value::Array(items) => items
            .iter()
            .map(prepare_message)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| with_side(side, error)),
        Value::Object(_) => prepare_message(value)
            .map(|message| vec![message])
            .map_err(|error| with_side(side, error)),
        _ => Err(Error::InvalidUpdate(format!(
            "{side} must be a message object or an array of messages"
        ))),
    }
}

fn with_side(side: &str, error: Error) -> Error {
    match error {
        Error::InvalidUpdate(message) => Error::InvalidUpdate(format!("{side}: {message}")),
        other => other,
    }
}

fn prepare_message(value: &Value) -> Result<PreparedMessage, Error> {
    let mut object = match value {
        Value::Object(map) => map.clone(),
        _ => return Err(Error::InvalidUpdate("message must be a JSON object".into())),
    };
    let type_name = require_string(&object, "type")?.to_string();
    validate_shape(&object, &type_name)?;
    let id = assign_id(&mut object)?;
    let kind = message_kind(&type_name, &id);
    Ok(PreparedMessage {
        id,
        kind,
        value: Value::Object(object),
    })
}

fn message_kind(type_name: &str, id: &str) -> MessageKind {
    if type_name == "remove" {
        if id == REMOVE_ALL_MESSAGES {
            MessageKind::RemoveAll
        } else {
            MessageKind::Remove
        }
    } else {
        MessageKind::Message
    }
}

fn assign_id(object: &mut Map<String, Value>) -> Result<String, Error> {
    let existing = match object.get("id") {
        None => None,
        Some(Value::String(id)) if id.is_empty() => None,
        Some(Value::String(id)) => Some(id.clone()),
        Some(_) => {
            return Err(Error::InvalidUpdate(
                "message field 'id' must be a string".into(),
            ))
        }
    };
    let id = existing.unwrap_or_else(|| Uuid::new_v4().to_string());
    object.insert("id".to_string(), Value::String(id.clone()));
    Ok(id)
}

fn validate_shape(object: &Map<String, Value>, type_name: &str) -> Result<(), Error> {
    match type_name {
        "human" | "system" => {
            require_string(object, "content")?;
            reject_field(
                object,
                "tool_calls",
                "only an ai message may include 'tool_calls'",
            )?;
            reject_field(
                object,
                "tool_call_id",
                "only a tool message may include 'tool_call_id'",
            )?;
        }
        "ai" => {
            require_string(object, "content")?;
            if let Some(tool_calls) = object.get("tool_calls") {
                validate_tool_calls(tool_calls)?;
            }
            reject_field(
                object,
                "tool_call_id",
                "only a tool message may include 'tool_call_id'",
            )?;
        }
        "tool" => {
            require_string(object, "content")?;
            require_string(object, "tool_call_id")?;
            reject_field(
                object,
                "tool_calls",
                "only an ai message may include 'tool_calls'",
            )?;
        }
        "remove" => {}
        other => {
            return Err(Error::InvalidUpdate(format!(
                "message type '{other}' is not supported"
            )));
        }
    }
    Ok(())
}

fn validate_tool_calls(value: &Value) -> Result<(), Error> {
    let calls = value.as_array().ok_or_else(|| {
        Error::InvalidUpdate("message field 'tool_calls' must be an array".into())
    })?;
    for call in calls {
        let object = call
            .as_object()
            .ok_or_else(|| Error::InvalidUpdate("tool call must be a JSON object".into()))?;
        require_string(object, "id")?;
        require_string(object, "name")?;
        match object.get("arguments") {
            Some(Value::Object(_)) => {}
            Some(_) => {
                return Err(Error::InvalidUpdate(
                    "tool call field 'arguments' must be a JSON object".into(),
                ))
            }
            None => {
                return Err(Error::InvalidUpdate(
                    "tool call is missing 'arguments'".into(),
                ))
            }
        }
    }
    Ok(())
}

fn require_string<'a>(object: &'a Map<String, Value>, field: &str) -> Result<&'a str, Error> {
    match object.get(field) {
        Some(Value::String(text)) => Ok(text.as_str()),
        Some(_) => Err(Error::InvalidUpdate(format!(
            "message field '{field}' must be a string"
        ))),
        None => Err(Error::InvalidUpdate(format!(
            "message is missing '{field}'"
        ))),
    }
}

fn reject_field(object: &Map<String, Value>, field: &str, reason: &str) -> Result<(), Error> {
    if object.contains_key(field) {
        Err(Error::InvalidUpdate(reason.into()))
    } else {
        Ok(())
    }
}

fn merge_messages(
    left: Vec<PreparedMessage>,
    right: Vec<PreparedMessage>,
) -> Result<Vec<Value>, Error> {
    let cut = right
        .iter()
        .rposition(|message| message.kind == MessageKind::RemoveAll);
    let (base, ops) = if let Some(index) = cut {
        let tail = right.into_iter().skip(index + 1).collect();
        (Vec::new(), tail)
    } else {
        (left, right)
    };
    apply_updates(base, ops)
}

fn apply_updates(
    mut base: Vec<PreparedMessage>,
    ops: Vec<PreparedMessage>,
) -> Result<Vec<Value>, Error> {
    let mut index: BTreeMap<String, usize> = BTreeMap::new();
    for (position, message) in base.iter().enumerate() {
        index.insert(message.id.clone(), position);
    }
    let mut removed: BTreeSet<String> = BTreeSet::new();

    for message in ops {
        match message.kind {
            MessageKind::RemoveAll => {
                base.clear();
                index.clear();
                removed.clear();
            }
            MessageKind::Remove => {
                if !index.contains_key(&message.id) {
                    return Err(Error::InvalidUpdate(format!(
                        "remove targets unknown message id '{}'",
                        message.id
                    )));
                }
                removed.insert(message.id);
            }
            MessageKind::Message => {
                if let Some(&position) = index.get(&message.id) {
                    removed.remove(&message.id);
                    base[position] = message;
                } else {
                    index.insert(message.id.clone(), base.len());
                    base.push(message);
                }
            }
        }
    }

    base.retain(|message| !removed.contains(&message.id));
    Ok(base.into_iter().map(|message| message.value).collect())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{add_messages, REMOVE_ALL_MESSAGES};
    use crate::Error;

    fn human(id: &str, content: &str) -> serde_json::Value {
        json!({"type": "human", "id": id, "content": content})
    }

    #[test]
    fn coerces_a_single_message_on_either_side() {
        let left = human("1", "hi");
        let right = json!({"type": "ai", "id": "2", "content": "hello"});
        let merged = add_messages(&left, &right).expect("coerce both sides");
        assert_eq!(merged, json!([left, right]));
    }

    #[test]
    fn remove_all_keeps_only_the_tail_after_the_last_sentinel() {
        let left = json!("not a message list");
        let right = json!([
            {"type": "remove", "id": "missing"},
            human("1", "dropped"),
            {"type": "remove", "id": REMOVE_ALL_MESSAGES},
            human("2", "also dropped"),
            {"type": "remove", "id": REMOVE_ALL_MESSAGES},
            human("3", "kept"),
            {"type": "ai", "id": "3", "content": "replaced"}
        ]);

        let merged = add_messages(&left, &right).expect("last sentinel wins");

        assert_eq!(
            merged,
            json!([{"type": "ai", "id": "3", "content": "replaced"}])
        );
    }

    #[test]
    fn message_json_accepts_documented_shapes_and_rejects_bad_ones() {
        let right = json!([
            {"type": "system", "id": "s", "content": "rules"},
            {
                "type": "ai",
                "id": "a",
                "content": "",
                "tool_calls": [{
                    "id": "call-1",
                    "name": "search",
                    "arguments": {"query": "rust"}
                }]
            },
            {
                "type": "tool",
                "id": "t",
                "content": "found",
                "tool_call_id": "call-1"
            }
        ]);
        let merged = add_messages(&json!([]), &right).expect("documented shapes");
        assert_eq!(merged, right);

        let bad_arguments = json!({
            "type": "ai",
            "id": "a",
            "content": "",
            "tool_calls": [{
                "id": "call-1",
                "name": "search",
                "arguments": "{\"query\":\"rust\"}"
            }]
        });
        let error = add_messages(&json!([]), &bad_arguments).expect_err("arguments object");
        assert!(
            matches!(error, Error::InvalidUpdate(ref message) if message.contains("arguments")),
            "{error:?}"
        );

        let missing_content = json!({"type": "human", "id": "1"});
        let error = add_messages(&json!([]), &missing_content).expect_err("content");
        assert!(matches!(error, Error::InvalidUpdate(_)));

        let unknown_type = json!({"type": "function", "id": "1", "content": "nope"});
        let error = add_messages(&json!([]), &unknown_type).expect_err("type");
        assert!(
            matches!(error, Error::InvalidUpdate(ref message) if message.contains("function")),
            "{error:?}"
        );
    }
}
