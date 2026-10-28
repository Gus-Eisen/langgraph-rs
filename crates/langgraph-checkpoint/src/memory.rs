//! Process-local saver. The map lives on the value, not in a global.

use std::collections::BTreeMap;
use std::future::ready;
use std::sync::{Mutex, MutexGuard};

use crate::{
    Checkpoint, CheckpointConfig, CheckpointMetadata, CheckpointTuple, Checkpointer, Error,
    PendingWrite,
};

type Threads = BTreeMap<(String, String), Vec<Stored>>;

#[derive(Debug, Clone)]
struct Stored {
    checkpoint: Checkpoint,
    metadata: CheckpointMetadata,
    pending_writes: Vec<PendingWrite>,
}

/// In-memory [`Checkpointer`].
///
/// Each value owns its mutex map. Two savers do not share threads. Share one
/// run's history by cloning an `Arc<InMemorySaver>` (or `Arc<dyn Checkpointer>`).
#[derive(Debug)]
pub struct InMemorySaver {
    inner: Mutex<Threads>,
}

impl InMemorySaver {
    /// Empty saver. No threads are stored.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(BTreeMap::new()),
        }
    }

    fn lock(&self) -> Result<MutexGuard<'_, Threads>, Error> {
        self.inner
            .lock()
            .map_err(|_| Error::Storage("checkpointer lock poisoned".into()))
    }
}

impl Default for InMemorySaver {
    fn default() -> Self {
        Self::new()
    }
}

impl Checkpointer for InMemorySaver {
    fn get_tuple<'a>(
        &'a self,
        config: &'a CheckpointConfig,
    ) -> crate::BoxFuture<'a, Result<Option<CheckpointTuple>, Error>> {
        Box::pin(ready(self.get_tuple_sync(config)))
    }

    fn list<'a>(
        &'a self,
        config: &'a CheckpointConfig,
        before: Option<&'a str>,
        limit: Option<usize>,
    ) -> crate::BoxFuture<'a, Result<Vec<CheckpointTuple>, Error>> {
        Box::pin(ready(self.list_sync(config, before, limit)))
    }

    fn put<'a>(
        &'a self,
        config: &'a CheckpointConfig,
        checkpoint: &'a Checkpoint,
        metadata: &'a CheckpointMetadata,
        new_versions: &'a BTreeMap<String, u64>,
    ) -> crate::BoxFuture<'a, Result<CheckpointConfig, Error>> {
        Box::pin(ready(self.put_sync(
            config,
            checkpoint,
            metadata,
            new_versions,
        )))
    }

    fn put_writes<'a>(
        &'a self,
        config: &'a CheckpointConfig,
        writes: &'a [PendingWrite],
        task_id: &'a str,
    ) -> crate::BoxFuture<'a, Result<(), Error>> {
        Box::pin(ready(self.put_writes_sync(config, writes, task_id)))
    }

    fn delete_thread<'a>(&'a self, thread_id: &'a str) -> crate::BoxFuture<'a, Result<(), Error>> {
        Box::pin(ready(self.delete_thread_sync(thread_id)))
    }
}

impl InMemorySaver {
    fn get_tuple_sync(&self, config: &CheckpointConfig) -> Result<Option<CheckpointTuple>, Error> {
        let guard = self.lock()?;
        let Some(slots) = guard.get(&key(config)) else {
            return missing_id(config);
        };
        let stored = match &config.checkpoint_id {
            Some(id) => find(slots, id).ok_or_else(|| Error::UnknownCheckpoint(id.clone()))?,
            None => match slots.last() {
                Some(stored) => stored,
                None => return Ok(None),
            },
        };
        Ok(Some(to_tuple(config, stored)))
    }

    fn list_sync(
        &self,
        config: &CheckpointConfig,
        before: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<CheckpointTuple>, Error> {
        let guard = self.lock()?;
        let Some(slots) = guard.get(&key(config)) else {
            if let Some(id) = before {
                return Err(Error::UnknownCheckpoint(id.to_string()));
            }
            return Ok(Vec::new());
        };
        let end = if let Some(id) = before {
            find_index(slots, id).ok_or_else(|| Error::UnknownCheckpoint(id.to_string()))?
        } else {
            slots.len()
        };
        let mut tuples: Vec<CheckpointTuple> = slots[..end]
            .iter()
            .rev()
            .map(|stored| to_tuple(config, stored))
            .collect();
        if let Some(limit) = limit {
            tuples.truncate(limit);
        }
        Ok(tuples)
    }

    fn put_sync(
        &self,
        config: &CheckpointConfig,
        checkpoint: &Checkpoint,
        metadata: &CheckpointMetadata,
        new_versions: &BTreeMap<String, u64>,
    ) -> Result<CheckpointConfig, Error> {
        if checkpoint.v != 1 {
            return Err(Error::Storage(format!(
                "unsupported checkpoint version {}",
                checkpoint.v
            )));
        }
        for (channel, version) in new_versions {
            match checkpoint.channel_versions.get(channel) {
                Some(stored) if stored == version => {}
                _ => {
                    return Err(Error::Storage(format!(
                        "new version for channel '{channel}' does not match the checkpoint"
                    )));
                }
            }
        }

        let mut checkpoint = checkpoint.clone();
        if checkpoint.id.is_empty() {
            checkpoint.id = crate::new_checkpoint_id();
        }
        if checkpoint.ts.is_empty() {
            checkpoint.ts = crate::checkpoint_timestamp();
        }

        let mut guard = self.lock()?;
        let slots = guard.entry(key(config)).or_default();
        if slots
            .iter()
            .any(|stored| stored.checkpoint.id == checkpoint.id)
        {
            return Err(Error::Storage(format!(
                "checkpoint id '{}' already exists",
                checkpoint.id
            )));
        }

        let parent = if let Some(id) = &config.checkpoint_id {
            Some(
                find(slots, id)
                    .ok_or_else(|| Error::UnknownCheckpoint(id.clone()))?
                    .clone(),
            )
        } else {
            slots.last().cloned()
        };
        let mut metadata = metadata.clone();
        let mut parents = parent
            .as_ref()
            .map(|stored| stored.metadata.parents.clone())
            .unwrap_or_default();
        if let Some(stored) = &parent {
            parents.insert(config.checkpoint_ns.clone(), stored.checkpoint.id.clone());
        }
        metadata.parents = parents;

        let id = checkpoint.id.clone();
        slots.push(Stored {
            checkpoint,
            metadata,
            pending_writes: Vec::new(),
        });
        Ok(CheckpointConfig {
            thread_id: config.thread_id.clone(),
            checkpoint_ns: config.checkpoint_ns.clone(),
            checkpoint_id: Some(id),
        })
    }

    fn put_writes_sync(
        &self,
        config: &CheckpointConfig,
        writes: &[PendingWrite],
        task_id: &str,
    ) -> Result<(), Error> {
        let Some(id) = &config.checkpoint_id else {
            return Err(Error::Storage("put_writes requires a checkpoint id".into()));
        };
        let mut guard = self.lock()?;
        let slots = guard
            .get_mut(&key(config))
            .ok_or_else(|| Error::UnknownCheckpoint(id.clone()))?;
        let stored = find_mut(slots, id).ok_or_else(|| Error::UnknownCheckpoint(id.clone()))?;
        for write in writes {
            stored.pending_writes.push(PendingWrite {
                task_id: task_id.to_string(),
                channel: write.channel.clone(),
                value: write.value.clone(),
            });
        }
        Ok(())
    }

    fn delete_thread_sync(&self, thread_id: &str) -> Result<(), Error> {
        let mut guard = self.lock()?;
        guard.retain(|(stored_thread, _), _| stored_thread != thread_id);
        Ok(())
    }
}

fn key(config: &CheckpointConfig) -> (String, String) {
    (config.thread_id.clone(), config.checkpoint_ns.clone())
}

fn missing_id(config: &CheckpointConfig) -> Result<Option<CheckpointTuple>, Error> {
    if let Some(id) = &config.checkpoint_id {
        Err(Error::UnknownCheckpoint(id.clone()))
    } else {
        Ok(None)
    }
}

fn find<'a>(slots: &'a [Stored], id: &str) -> Option<&'a Stored> {
    slots.iter().find(|stored| stored.checkpoint.id == id)
}

fn find_mut<'a>(slots: &'a mut [Stored], id: &str) -> Option<&'a mut Stored> {
    slots.iter_mut().find(|stored| stored.checkpoint.id == id)
}

fn find_index(slots: &[Stored], id: &str) -> Option<usize> {
    slots.iter().position(|stored| stored.checkpoint.id == id)
}

fn to_tuple(config: &CheckpointConfig, stored: &Stored) -> CheckpointTuple {
    let parent_id = stored.metadata.parents.get(&config.checkpoint_ns).cloned();
    CheckpointTuple {
        config: CheckpointConfig {
            thread_id: config.thread_id.clone(),
            checkpoint_ns: config.checkpoint_ns.clone(),
            checkpoint_id: Some(stored.checkpoint.id.clone()),
        },
        checkpoint: stored.checkpoint.clone(),
        metadata: stored.metadata.clone(),
        parent_config: parent_id.map(|id| CheckpointConfig {
            thread_id: config.thread_id.clone(),
            checkpoint_ns: config.checkpoint_ns.clone(),
            checkpoint_id: Some(id),
        }),
        pending_writes: stored.pending_writes.clone(),
    }
}
