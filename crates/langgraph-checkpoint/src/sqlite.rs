//! File-backed saver. One SQLite connection lives on the value.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use rusqlite::{params, Connection, OptionalExtension};

use crate::{
    Checkpoint, CheckpointConfig, CheckpointMetadata, CheckpointTuple, Checkpointer, Error,
    PendingWrite,
};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS checkpoints (
    row_id INTEGER PRIMARY KEY,
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    checkpoint_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    UNIQUE (thread_id, checkpoint_ns, checkpoint_id)
);
CREATE INDEX IF NOT EXISTS checkpoints_thread_ns
    ON checkpoints (thread_id, checkpoint_ns, row_id);
CREATE TABLE IF NOT EXISTS pending_writes (
    row_id INTEGER PRIMARY KEY,
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    value_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS pending_writes_checkpoint
    ON pending_writes (thread_id, checkpoint_ns, checkpoint_id, row_id);
";

struct Stored {
    checkpoint: Checkpoint,
    metadata: CheckpointMetadata,
}

/// File-backed [`Checkpointer`](crate::Checkpointer).
///
/// Each value owns one SQLite connection behind a mutex. Every trait method
/// runs that connection on `spawn_blocking` and must be polled on a tokio
/// runtime. The database is created on [`SqliteSaver::open`] if it does not
/// exist. Drop this value and open the same path to read what was written.
///
/// Two savers opened on different paths do not share threads. Share one file
/// across tasks with `Arc<SqliteSaver>` or `Arc<dyn Checkpointer>`.
pub struct SqliteSaver {
    conn: Arc<Mutex<Connection>>,
    path: PathBuf,
}

impl SqliteSaver {
    /// Open or create the database at `path`.
    ///
    /// `path` is a filesystem path. The parent directory must already exist.
    /// The connection uses the bundled SQLite library, so the host does not
    /// need a system libsqlite.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Storage`](crate::Error::Storage) when the file cannot
    /// be opened or the checkpoint tables cannot be created.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let conn = Connection::open(path)
            .map_err(|err| Error::Storage(format!("failed to open {}: {err}", path.display())))?;
        conn.busy_timeout(Duration::from_secs(5)).map_err(storage)?;
        conn.execute_batch(SCHEMA).map_err(storage)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            path: path.to_path_buf(),
        })
    }

    fn blocking<'a, T, F>(&'a self, work: F) -> crate::BoxFuture<'a, Result<T, Error>>
    where
        T: Send + 'static,
        F: FnOnce(&mut Connection) -> Result<T, Error> + Send + 'static,
    {
        let conn = Arc::clone(&self.conn);
        Box::pin(async move {
            tokio::task::spawn_blocking(move || {
                let mut guard = lock(&conn)?;
                work(&mut guard)
            })
            .await
            .map_err(storage)?
        })
    }
}

impl std::fmt::Debug for SqliteSaver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteSaver")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl Checkpointer for SqliteSaver {
    fn get_tuple<'a>(
        &'a self,
        config: &'a CheckpointConfig,
    ) -> crate::BoxFuture<'a, Result<Option<CheckpointTuple>, Error>> {
        let config = config.clone();
        self.blocking(move |conn| get_tuple_sync(conn, &config))
    }

    fn list<'a>(
        &'a self,
        config: &'a CheckpointConfig,
        before: Option<&'a str>,
        limit: Option<usize>,
    ) -> crate::BoxFuture<'a, Result<Vec<CheckpointTuple>, Error>> {
        let config = config.clone();
        let before = before.map(str::to_string);
        self.blocking(move |conn| list_sync(conn, &config, before.as_deref(), limit))
    }

    fn put<'a>(
        &'a self,
        config: &'a CheckpointConfig,
        checkpoint: &'a Checkpoint,
        metadata: &'a CheckpointMetadata,
        new_versions: &'a BTreeMap<String, u64>,
    ) -> crate::BoxFuture<'a, Result<CheckpointConfig, Error>> {
        let config = config.clone();
        let checkpoint = checkpoint.clone();
        let metadata = metadata.clone();
        let new_versions = new_versions.clone();
        self.blocking(move |conn| put_sync(conn, &config, &checkpoint, &metadata, &new_versions))
    }

    fn put_writes<'a>(
        &'a self,
        config: &'a CheckpointConfig,
        writes: &'a [PendingWrite],
        task_id: &'a str,
    ) -> crate::BoxFuture<'a, Result<(), Error>> {
        let config = config.clone();
        let writes = writes.to_vec();
        let task_id = task_id.to_string();
        self.blocking(move |conn| put_writes_sync(conn, &config, &writes, &task_id))
    }

    fn delete_thread<'a>(&'a self, thread_id: &'a str) -> crate::BoxFuture<'a, Result<(), Error>> {
        let thread_id = thread_id.to_string();
        self.blocking(move |conn| delete_thread_sync(conn, &thread_id))
    }
}

fn lock(conn: &Mutex<Connection>) -> Result<MutexGuard<'_, Connection>, Error> {
    conn.lock()
        .map_err(|_| Error::Storage("checkpointer lock poisoned".into()))
}

fn storage(err: impl std::fmt::Display) -> Error {
    Error::Storage(err.to_string())
}

fn get_tuple_sync(
    conn: &Connection,
    config: &CheckpointConfig,
) -> Result<Option<CheckpointTuple>, Error> {
    let stored = match &config.checkpoint_id {
        Some(id) => match load_by_id(conn, config, id)? {
            Some(stored) => stored,
            None => return Err(Error::UnknownCheckpoint(id.clone())),
        },
        None => match load_latest(conn, config)? {
            Some(stored) => stored,
            None => return Ok(None),
        },
    };
    let writes = load_writes(conn, config, &stored.checkpoint.id)?;
    Ok(Some(to_tuple(config, &stored, writes)))
}

fn list_sync(
    conn: &Connection,
    config: &CheckpointConfig,
    before: Option<&str>,
    limit: Option<usize>,
) -> Result<Vec<CheckpointTuple>, Error> {
    let stored = load_namespace(conn, config)?;
    if stored.is_empty() {
        if let Some(id) = before {
            return Err(Error::UnknownCheckpoint(id.to_string()));
        }
        return Ok(Vec::new());
    }
    let end = if let Some(id) = before {
        stored
            .iter()
            .position(|row| row.checkpoint.id == id)
            .ok_or_else(|| Error::UnknownCheckpoint(id.to_string()))?
    } else {
        stored.len()
    };
    let mut tuples = Vec::new();
    for row in stored[..end].iter().rev() {
        let writes = load_writes(conn, config, &row.checkpoint.id)?;
        tuples.push(to_tuple(config, row, writes));
        if limit.is_some_and(|limit| tuples.len() == limit) {
            break;
        }
    }
    Ok(tuples)
}

fn put_sync(
    conn: &mut Connection,
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

    let tx = conn.transaction().map_err(storage)?;
    if checkpoint_exists(&tx, config, &checkpoint.id)? {
        return Err(Error::Storage(format!(
            "checkpoint id '{}' already exists",
            checkpoint.id
        )));
    }
    let parent = if let Some(id) = &config.checkpoint_id {
        Some(load_by_id(&tx, config, id)?.ok_or_else(|| Error::UnknownCheckpoint(id.clone()))?)
    } else {
        load_latest(&tx, config)?
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

    let checkpoint_json = serde_json::to_string(&checkpoint).map_err(storage)?;
    let metadata_json = serde_json::to_string(&metadata).map_err(storage)?;
    let id = checkpoint.id.clone();
    tx.execute(
        "INSERT INTO checkpoints (
            thread_id, checkpoint_ns, checkpoint_id, checkpoint_json, metadata_json
        ) VALUES (?1, ?2, ?3, ?4, ?5)",
        params![
            config.thread_id,
            config.checkpoint_ns,
            id,
            checkpoint_json,
            metadata_json,
        ],
    )
    .map_err(storage)?;
    tx.commit().map_err(storage)?;
    Ok(CheckpointConfig {
        thread_id: config.thread_id.clone(),
        checkpoint_ns: config.checkpoint_ns.clone(),
        checkpoint_id: Some(id),
    })
}

fn put_writes_sync(
    conn: &mut Connection,
    config: &CheckpointConfig,
    writes: &[PendingWrite],
    task_id: &str,
) -> Result<(), Error> {
    let Some(id) = &config.checkpoint_id else {
        return Err(Error::Storage("put_writes requires a checkpoint id".into()));
    };
    let tx = conn.transaction().map_err(storage)?;
    if !checkpoint_exists(&tx, config, id)? {
        return Err(Error::UnknownCheckpoint(id.clone()));
    }
    for write in writes {
        let value_json = serde_json::to_string(&write.value).map_err(storage)?;
        tx.execute(
            "INSERT INTO pending_writes (
                thread_id, checkpoint_ns, checkpoint_id, task_id, channel, value_json
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                config.thread_id,
                config.checkpoint_ns,
                id,
                task_id,
                write.channel,
                value_json,
            ],
        )
        .map_err(storage)?;
    }
    tx.commit().map_err(storage)?;
    Ok(())
}

fn delete_thread_sync(conn: &mut Connection, thread_id: &str) -> Result<(), Error> {
    let tx = conn.transaction().map_err(storage)?;
    tx.execute(
        "DELETE FROM pending_writes WHERE thread_id = ?1",
        params![thread_id],
    )
    .map_err(storage)?;
    tx.execute(
        "DELETE FROM checkpoints WHERE thread_id = ?1",
        params![thread_id],
    )
    .map_err(storage)?;
    tx.commit().map_err(storage)?;
    Ok(())
}

fn checkpoint_exists(
    conn: &Connection,
    config: &CheckpointConfig,
    id: &str,
) -> Result<bool, Error> {
    let found: Option<i64> = conn
        .query_row(
            "SELECT row_id FROM checkpoints
             WHERE thread_id = ?1 AND checkpoint_ns = ?2 AND checkpoint_id = ?3",
            params![config.thread_id, config.checkpoint_ns, id],
            |row| row.get(0),
        )
        .optional()
        .map_err(storage)?;
    Ok(found.is_some())
}

fn load_by_id(
    conn: &Connection,
    config: &CheckpointConfig,
    id: &str,
) -> Result<Option<Stored>, Error> {
    let row = conn
        .query_row(
            "SELECT checkpoint_json, metadata_json FROM checkpoints
             WHERE thread_id = ?1 AND checkpoint_ns = ?2 AND checkpoint_id = ?3",
            params![config.thread_id, config.checkpoint_ns, id],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(storage)?;
    row.map(|(checkpoint_json, metadata_json)| decode(&checkpoint_json, &metadata_json))
        .transpose()
}

fn load_latest(conn: &Connection, config: &CheckpointConfig) -> Result<Option<Stored>, Error> {
    let row = conn
        .query_row(
            "SELECT checkpoint_json, metadata_json FROM checkpoints
             WHERE thread_id = ?1 AND checkpoint_ns = ?2
             ORDER BY row_id DESC
             LIMIT 1",
            params![config.thread_id, config.checkpoint_ns],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(storage)?;
    row.map(|(checkpoint_json, metadata_json)| decode(&checkpoint_json, &metadata_json))
        .transpose()
}

fn load_namespace(conn: &Connection, config: &CheckpointConfig) -> Result<Vec<Stored>, Error> {
    let mut stmt = conn
        .prepare(
            "SELECT checkpoint_json, metadata_json FROM checkpoints
             WHERE thread_id = ?1 AND checkpoint_ns = ?2
             ORDER BY row_id ASC",
        )
        .map_err(storage)?;
    let rows = stmt
        .query_map(params![config.thread_id, config.checkpoint_ns], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(storage)?;
    let mut stored = Vec::new();
    for row in rows {
        let (checkpoint_json, metadata_json) = row.map_err(storage)?;
        stored.push(decode(&checkpoint_json, &metadata_json)?);
    }
    Ok(stored)
}

fn load_writes(
    conn: &Connection,
    config: &CheckpointConfig,
    checkpoint_id: &str,
) -> Result<Vec<PendingWrite>, Error> {
    let mut stmt = conn
        .prepare(
            "SELECT task_id, channel, value_json FROM pending_writes
             WHERE thread_id = ?1 AND checkpoint_ns = ?2 AND checkpoint_id = ?3
             ORDER BY row_id ASC",
        )
        .map_err(storage)?;
    let rows = stmt
        .query_map(
            params![config.thread_id, config.checkpoint_ns, checkpoint_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .map_err(storage)?;
    let mut writes = Vec::new();
    for row in rows {
        let (task_id, channel, value_json) = row.map_err(storage)?;
        writes.push(PendingWrite {
            task_id,
            channel,
            value: serde_json::from_str(&value_json).map_err(storage)?,
        });
    }
    Ok(writes)
}

fn decode(checkpoint_json: &str, metadata_json: &str) -> Result<Stored, Error> {
    Ok(Stored {
        checkpoint: serde_json::from_str(checkpoint_json).map_err(storage)?,
        metadata: serde_json::from_str(metadata_json).map_err(storage)?,
    })
}

fn to_tuple(
    config: &CheckpointConfig,
    stored: &Stored,
    pending_writes: Vec<PendingWrite>,
) -> CheckpointTuple {
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
        pending_writes,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    use serde_json::json;

    use super::SqliteSaver;
    use crate::{
        Checkpoint, CheckpointConfig, CheckpointMetadata, CheckpointSource, Checkpointer,
        PendingWrite,
    };

    struct TempDb {
        path: PathBuf,
    }

    impl TempDb {
        fn new() -> Self {
            let name = format!(
                "langgraph-sqlite-{}-{}.db",
                std::process::id(),
                crate::new_checkpoint_id()
            );
            Self {
                path: std::env::temp_dir().join(name),
            }
        }
    }

    impl Drop for TempDb {
        fn drop(&mut self) {
            remove_sqlite_files(&self.path);
        }
    }

    fn remove_sqlite_files(path: &Path) {
        let _ = std::fs::remove_file(path);
        let mut wal = path.as_os_str().to_owned();
        wal.push("-wal");
        let _ = std::fs::remove_file(wal);
        let mut shm = path.as_os_str().to_owned();
        shm.push("-shm");
        let _ = std::fs::remove_file(shm);
    }

    fn config(thread_id: &str, checkpoint_ns: &str) -> CheckpointConfig {
        CheckpointConfig {
            thread_id: thread_id.to_string(),
            checkpoint_ns: checkpoint_ns.to_string(),
            checkpoint_id: None,
        }
    }

    fn sample_checkpoint() -> Checkpoint {
        let mut checkpoint = Checkpoint::new();
        checkpoint
            .channel_values
            .insert("n".to_string(), json!({"k": [1, "a"]}));
        checkpoint.channel_versions.insert("n".to_string(), 1);
        checkpoint.channel_versions.insert("other".to_string(), 0);
        checkpoint.updated_channels.push("n".to_string());
        checkpoint.done.insert("__start__".to_string(), 1);
        checkpoint.branch.insert("branch:0:__end__".to_string(), 1);
        checkpoint.versions_seen.insert(
            "node".to_string(),
            BTreeMap::from([("done:__start__".to_string(), 1)]),
        );
        checkpoint
    }

    fn versions(checkpoint: &Checkpoint) -> BTreeMap<String, u64> {
        checkpoint.channel_versions.clone()
    }

    fn metadata(source: CheckpointSource, step: i64) -> CheckpointMetadata {
        CheckpointMetadata {
            source,
            step,
            parents: BTreeMap::new(),
        }
    }

    fn count_rows(path: &Path, table: &str, thread_id: &str) -> i64 {
        let conn = rusqlite::Connection::open(path).expect("count open");
        let sql = format!("SELECT COUNT(*) FROM {table} WHERE thread_id = ?1");
        conn.query_row(&sql, [thread_id], |row| row.get(0))
            .expect("count")
    }

    #[tokio::test]
    async fn sqlite_saver_matches_the_checkpointer_contract() {
        let file = TempDb::new();
        let created = SqliteSaver::open(&file.path).expect("create");
        drop(created);
        let saver = SqliteSaver::open(&file.path).expect("reopen empty");
        crate::suite::assert_checkpointer(&saver).await;
    }

    #[tokio::test]
    async fn sqlite_file_survives_reopen_and_delete_thread_removes_rows() {
        let file = TempDb::new();
        let root = config("thread", "");
        let child = config("thread", "child");
        let other = config("other", "");
        let first_checkpoint = sample_checkpoint();
        let first_id = first_checkpoint.id.clone();

        let (second_id, child_id) = {
            let saver = SqliteSaver::open(&file.path).expect("create");
            let saved = saver
                .put(
                    &root,
                    &first_checkpoint,
                    &metadata(CheckpointSource::Input, -1),
                    &versions(&first_checkpoint),
                )
                .await
                .expect("put");
            assert_eq!(saved.checkpoint_id.as_deref(), Some(first_id.as_str()));
            saver
                .put_writes(
                    &saved,
                    &[PendingWrite {
                        task_id: String::new(),
                        channel: "n".to_string(),
                        value: json!(8),
                    }],
                    "node-a",
                )
                .await
                .expect("writes");

            let mut second = sample_checkpoint();
            second.channel_values.insert("n".to_string(), json!(2));
            second.channel_versions.insert("n".to_string(), 2);
            let second_id = second.id.clone();
            saver
                .put(
                    &saved,
                    &second,
                    &metadata(CheckpointSource::Loop, 0),
                    &versions(&second),
                )
                .await
                .expect("second");

            let namespaced = sample_checkpoint();
            let child_id = namespaced.id.clone();
            saver
                .put(
                    &child,
                    &namespaced,
                    &metadata(CheckpointSource::Input, -1),
                    &versions(&namespaced),
                )
                .await
                .expect("child");

            let elsewhere = sample_checkpoint();
            saver
                .put(
                    &other,
                    &elsewhere,
                    &metadata(CheckpointSource::Update, 0),
                    &versions(&elsewhere),
                )
                .await
                .expect("other thread");
            drop(saver);
            (second_id, child_id)
        };

        let separate = TempDb::new();
        let separate_saver = SqliteSaver::open(&separate.path).expect("other file");
        assert!(separate_saver
            .get_tuple(&root)
            .await
            .expect("other file get")
            .is_none());
        drop(separate_saver);

        {
            let saver = SqliteSaver::open(&file.path).expect("reopen");
            let latest = saver.get_tuple(&root).await.expect("latest").expect("some");
            assert_eq!(latest.checkpoint.id, second_id);
            assert_eq!(latest.checkpoint.channel_values.get("n"), Some(&json!(2)));
            assert_eq!(latest.metadata.source, CheckpointSource::Loop);
            assert_eq!(latest.metadata.step, 0);
            assert_eq!(
                latest
                    .parent_config
                    .as_ref()
                    .and_then(|parent| parent.checkpoint_id.as_deref()),
                Some(first_id.as_str())
            );

            let original = saver
                .get_tuple(&CheckpointConfig {
                    checkpoint_id: Some(first_id.clone()),
                    ..root.clone()
                })
                .await
                .expect("by id")
                .expect("found");
            assert_eq!(original.checkpoint, first_checkpoint);
            assert_eq!(original.pending_writes.len(), 1);
            assert_eq!(original.pending_writes[0].task_id, "node-a");
            assert_eq!(original.pending_writes[0].value, json!(8));
            assert!(original.parent_config.is_none());

            let namespaced = saver
                .get_tuple(&child)
                .await
                .expect("child")
                .expect("found");
            assert_eq!(namespaced.checkpoint.id, child_id);
            assert_eq!(saver.list(&root, None, None).await.expect("list").len(), 2);

            saver.delete_thread("thread").await.expect("delete");
            assert!(saver.get_tuple(&root).await.expect("deleted").is_none());
            assert!(saver.get_tuple(&child).await.expect("deleted ns").is_none());
            assert!(saver.get_tuple(&other).await.expect("kept").is_some());
            drop(saver);
        }

        assert_eq!(count_rows(&file.path, "checkpoints", "thread"), 0);
        assert_eq!(count_rows(&file.path, "pending_writes", "thread"), 0);
        assert_eq!(count_rows(&file.path, "checkpoints", "other"), 1);

        let saver = SqliteSaver::open(&file.path).expect("reopen after delete");
        assert!(saver.get_tuple(&root).await.expect("still gone").is_none());
        assert!(saver.get_tuple(&other).await.expect("still kept").is_some());
    }
}
