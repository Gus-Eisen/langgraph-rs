//! JSON channel map, versions, and [`GraphState`].

use std::collections::BTreeMap;

use serde_json::Value;

use crate::error::Error;
use crate::reducer::{CustomReducer, Field, Reducer};

/// Names that cannot be state fields.
///
/// The builder rejects the same strings as node names. Comparison is exact.
pub const RESERVED_NAMES: &[&str] = &[
    "__start__",
    "__end__",
    "__interrupt__",
    "__resume__",
    "__error__",
    "__pregel_tasks",
];

/// Typed state stored as one JSON value per channel.
///
/// A node returns [`GraphState::Update`]. Fields omitted from that update are
/// not writes. The channel map reconstructs `Self` from the committed values.
pub trait GraphState: Sized {
    /// Partial update a node returns. Absent fields are not writes.
    type Update;

    /// Channel name and reducer for every field, in field order.
    fn schema() -> &'static [Field];

    /// Every field of this value, keyed by channel name.
    fn to_channel_values(&self) -> Result<BTreeMap<String, Value>, Error>;

    /// Fields this update writes, keyed by channel name.
    ///
    /// Omitting a key means the update does not write that channel.
    fn update_to_channel_values(update: &Self::Update) -> Result<BTreeMap<String, Value>, Error>;

    /// Build a value from committed channel values.
    ///
    /// Channels that have never accepted a value are absent from the map.
    fn from_channel_values(values: &BTreeMap<String, Value>) -> Result<Self, Error>;

    /// Package one task's update as channel writes.
    ///
    /// `write_index` counts from 0 in schema order, only among fields the
    /// update actually sets. The same `task_path` and `task_id` are stored
    /// on every write. A key that is not in [`GraphState::schema`] is
    /// [`Error::InvalidUpdate`](Error::InvalidUpdate).
    fn update_to_writes(
        update: &Self::Update,
        task_path: &[String],
        task_id: &str,
    ) -> Result<Vec<Write>, Error> {
        let values = Self::update_to_channel_values(update)?;
        let schema = Self::schema();
        for key in values.keys() {
            if !schema.iter().any(|field| field.name == key) {
                return Err(Error::InvalidUpdate(format!(
                    "update writes unknown channel '{key}'"
                )));
            }
        }
        let mut writes = Vec::new();
        for field in schema {
            if let Some(value) = values.get(field.name) {
                writes.push(Write {
                    channel: field.name.to_string(),
                    value: value.clone(),
                    task_path: task_path.to_vec(),
                    task_id: task_id.to_string(),
                    write_index: u64::try_from(writes.len()).map_err(|_| {
                        Error::InvalidUpdate(format!(
                            "channel '{}' write index overflow",
                            field.name
                        ))
                    })?,
                });
            }
        }
        Ok(writes)
    }
}

/// One channel write, attributed to the task that produced it.
///
/// [`ChannelMap::apply_writes`] sorts a step's writes by
/// `(task_path, task_id, write_index)` ascending before folding. That order
/// is independent of the order writes are submitted.
#[derive(Debug, Clone, PartialEq)]
pub struct Write {
    /// Channel this write updates.
    pub channel: String,
    /// JSON value the reducer folds in.
    pub value: Value,
    /// Task location. An empty path is the root. Compared element by element.
    pub task_path: Vec<String>,
    /// Identity of the task inside `task_path`.
    pub task_id: String,
    /// Index of this write among the task's writes. Lower indexes fold first.
    pub write_index: u64,
}

#[derive(Debug, Clone)]
struct Channel {
    reducer: Reducer,
    value: Option<Value>,
    version: u64,
}

/// Committed channel values, reducers, and `u64` versions.
///
/// Versions start at 0. [`ChannelMap::apply_writes`] adds 1 to a channel when
/// a step accepts at least one write to it. A step with no writes does not
/// change any version. A failed step leaves the map unchanged.
#[derive(Debug, Clone)]
pub struct ChannelMap {
    channels: BTreeMap<String, Channel>,
}

impl ChannelMap {
    /// Build an empty map from a field list.
    ///
    /// Every channel is unseen: no value, version 0. Reserved names, empty
    /// names, and duplicates are [`Error::GraphValidation`].
    pub fn new(fields: &[Field]) -> Result<Self, Error> {
        let mut channels = BTreeMap::new();
        for field in fields {
            if field.name.is_empty() {
                return Err(Error::GraphValidation("channel name is empty".into()));
            }
            if RESERVED_NAMES.contains(&field.name) {
                return Err(Error::GraphValidation(format!(
                    "channel name '{}' is reserved",
                    field.name
                )));
            }
            if channels.contains_key(field.name) {
                return Err(Error::GraphValidation(format!(
                    "duplicate channel '{}'",
                    field.name
                )));
            }
            channels.insert(
                field.name.to_string(),
                Channel {
                    reducer: field.reducer,
                    value: None,
                    version: 0,
                },
            );
        }
        Ok(Self { channels })
    }

    /// Install `state` as the committed snapshot.
    ///
    /// Versions stay 0. This is not a step: reducers are not folded and no
    /// version increases. Every schema field must be present. An append
    /// field must already be a JSON array.
    pub fn from_state<S: GraphState>(state: &S) -> Result<Self, Error> {
        let mut map = Self::new(S::schema())?;
        map.install_values(&state.to_channel_values()?)?;
        Ok(map)
    }

    /// Reconstruct state from committed values.
    ///
    /// Channel names and reducer kinds must match `S::schema`. Custom folds
    /// match as a kind; their function pointers are not compared.
    pub fn to_state<S: GraphState>(&self) -> Result<S, Error> {
        if !self.schema_matches(S::schema()) {
            return Err(Error::GraphValidation(
                "channel map does not match GraphState::schema".into(),
            ));
        }
        S::from_channel_values(&self.values())
    }

    /// Committed value for `channel`.
    ///
    /// `None` means the channel is not in the schema, or it is in the schema
    /// and has never accepted a value. Use [`ChannelMap::version`] to tell
    /// those cases apart.
    pub fn get(&self, channel: &str) -> Option<&Value> {
        self.channels
            .get(channel)
            .and_then(|channel| channel.value.as_ref())
    }

    /// Version of a schema channel.
    ///
    /// `Some(0)` is an unseen channel. `None` means the name is not in the
    /// schema.
    pub fn version(&self, channel: &str) -> Option<u64> {
        self.channels.get(channel).map(|channel| channel.version)
    }

    /// Committed values. Unseen channels are omitted.
    pub fn values(&self) -> BTreeMap<String, Value> {
        self.channels
            .iter()
            .filter_map(|(name, channel)| {
                channel
                    .value
                    .as_ref()
                    .map(|value| (name.clone(), value.clone()))
            })
            .collect()
    }

    /// Version of every schema channel, including unseen channels at 0.
    pub fn versions(&self) -> BTreeMap<String, u64> {
        self.channels
            .iter()
            .map(|(name, channel)| (name.clone(), channel.version))
            .collect()
    }

    /// Fold `writes` into the committed channels.
    ///
    /// Writes are grouped by channel and sorted by
    /// `(task_path, task_id, write_index)`. Equal keys keep their submitted
    /// order. Each channel that receives at least one write is folded with
    /// its schema reducer and its version increases by 1. Channels with no
    /// writes are left alone, including their versions.
    ///
    /// On [`Error::InvalidUpdate`] the map is unchanged: a failed overwrite
    /// or append does not commit any channel in the step.
    pub fn apply_writes(&mut self, writes: &[Write]) -> Result<(), Error> {
        let grouped = group_writes(writes)?;
        for name in grouped.keys() {
            if !self.channels.contains_key(*name) {
                return Err(Error::InvalidUpdate(format!(
                    "write to unknown channel '{name}'"
                )));
            }
        }

        let mut folded = BTreeMap::new();
        for (name, group) in &grouped {
            let channel = &self.channels[*name];
            let value = fold_writes(name, channel.reducer, channel.value.as_ref(), group)?;
            folded.insert((*name).to_string(), value);
        }

        let mut planned = Vec::with_capacity(folded.len());
        for (name, value) in folded {
            let version = self.channels[&name].version.checked_add(1).ok_or_else(|| {
                Error::InvalidUpdate(format!("channel '{name}' version overflow"))
            })?;
            planned.push((name, value, version));
        }
        for (name, value, version) in planned {
            let channel = self.channels.get_mut(&name).expect("channel exists");
            channel.value = Some(value);
            channel.version = version;
        }
        Ok(())
    }

    fn install_values(&mut self, values: &BTreeMap<String, Value>) -> Result<(), Error> {
        for key in values.keys() {
            if !self.channels.contains_key(key) {
                return Err(Error::InvalidUpdate(format!(
                    "state value for unknown channel '{key}'"
                )));
            }
        }
        for name in self.channels.keys() {
            if !values.contains_key(name) {
                return Err(Error::InvalidUpdate(format!(
                    "state is missing channel '{name}'"
                )));
            }
        }
        for (name, value) in values {
            let reducer = self.channels[name].reducer;
            if matches!(reducer, Reducer::Append) && !value.is_array() {
                return Err(Error::InvalidUpdate(format!(
                    "channel '{name}' append reducer expected a JSON array"
                )));
            }
        }
        for (name, value) in values {
            self.channels.get_mut(name).expect("channel exists").value = Some(value.clone());
        }
        Ok(())
    }

    fn schema_matches(&self, schema: &[Field]) -> bool {
        if self.channels.len() != schema.len() {
            return false;
        }
        schema.iter().all(|field| {
            self.channels
                .get(field.name)
                .is_some_and(|channel| same_reducer(channel.reducer, field.reducer))
        })
    }
}

/// Reducer kinds match. Custom folds match as a kind; function pointers are
/// not compared.
fn same_reducer(left: Reducer, right: Reducer) -> bool {
    matches!(
        (left, right),
        (Reducer::Overwrite, Reducer::Overwrite)
            | (Reducer::Append, Reducer::Append)
            | (Reducer::Custom(_), Reducer::Custom(_))
    )
}

fn group_writes(writes: &[Write]) -> Result<BTreeMap<&str, Vec<&Write>>, Error> {
    let mut grouped: BTreeMap<&str, Vec<&Write>> = BTreeMap::new();
    for write in writes {
        if write.channel.is_empty() {
            return Err(Error::InvalidUpdate("write channel name is empty".into()));
        }
        grouped
            .entry(write.channel.as_str())
            .or_default()
            .push(write);
    }
    for group in grouped.values_mut() {
        group.sort_by(|left, right| write_order(left, right));
    }
    Ok(grouped)
}

fn write_order(left: &Write, right: &Write) -> std::cmp::Ordering {
    (&left.task_path, left.task_id.as_str(), left.write_index).cmp(&(
        &right.task_path,
        right.task_id.as_str(),
        right.write_index,
    ))
}

fn fold_writes(
    channel: &str,
    reducer: Reducer,
    current: Option<&Value>,
    writes: &[&Write],
) -> Result<Value, Error> {
    debug_assert!(!writes.is_empty());
    match reducer {
        Reducer::Overwrite => fold_overwrite(channel, writes),
        Reducer::Append => fold_append(channel, current, writes),
        Reducer::Custom(reduce) => fold_custom(reduce, current, writes),
    }
}

fn fold_overwrite(channel: &str, writes: &[&Write]) -> Result<Value, Error> {
    if writes.len() != 1 {
        return Err(Error::InvalidUpdate(format!(
            "channel '{channel}' received {} overwrite writes in one step",
            writes.len()
        )));
    }
    Ok(writes[0].value.clone())
}

fn fold_append(channel: &str, current: Option<&Value>, writes: &[&Write]) -> Result<Value, Error> {
    let mut combined = Vec::new();
    match current {
        None => {}
        Some(Value::Array(items)) => combined.extend(items.iter().cloned()),
        Some(_) => {
            return Err(Error::InvalidUpdate(format!(
                "channel '{channel}' append reducer expected a JSON array"
            )));
        }
    }
    for write in writes {
        match &write.value {
            Value::Array(items) => combined.extend(items.iter().cloned()),
            _ => {
                return Err(Error::InvalidUpdate(format!(
                    "channel '{channel}' append write is not a JSON array"
                )));
            }
        }
    }
    Ok(Value::Array(combined))
}

fn fold_custom(
    reduce: CustomReducer,
    current: Option<&Value>,
    writes: &[&Write],
) -> Result<Value, Error> {
    let mut accumulated = reduce(current, &writes[0].value)?;
    for write in &writes[1..] {
        accumulated = reduce(Some(&accumulated), &write.value)?;
    }
    Ok(accumulated)
}
