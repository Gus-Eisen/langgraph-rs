//! Saver behavior shared by the in-memory implementation.
//!
//! A later SQLite saver should call [`assert_checkpointer`] with a temporary
//! file. The assertions use the trait only.

use std::collections::BTreeMap;

use serde_json::json;

use crate::{
    Checkpoint, CheckpointConfig, CheckpointMetadata, CheckpointSource, Checkpointer, Error,
    PendingWrite,
};

fn config(thread_id: &str, checkpoint_ns: &str, checkpoint_id: Option<&str>) -> CheckpointConfig {
    CheckpointConfig {
        thread_id: thread_id.to_string(),
        checkpoint_ns: checkpoint_ns.to_string(),
        checkpoint_id: checkpoint_id.map(str::to_string),
    }
}

fn metadata(source: CheckpointSource, step: i64) -> CheckpointMetadata {
    CheckpointMetadata {
        source,
        step,
        parents: BTreeMap::new(),
    }
}

fn checkpoint(channel: &str, value: serde_json::Value, version: u64) -> Checkpoint {
    let mut checkpoint = Checkpoint::new();
    checkpoint.channel_values.insert(channel.to_string(), value);
    checkpoint
        .channel_versions
        .insert(channel.to_string(), version);
    checkpoint.updated_channels.push(channel.to_string());
    checkpoint.done.insert("__start__".to_string(), 1);
    checkpoint
}

/// Round-trip, order, fork, writes, and deletion contract.
pub async fn assert_checkpointer(saver: &dyn Checkpointer) {
    let root = config("thread", "", None);
    assert!(saver.get_tuple(&root).await.expect("empty get").is_none());
    assert!(saver
        .list(&root, None, None)
        .await
        .expect("empty list")
        .is_empty());

    let first = checkpoint("n", json!(1), 1);
    let first_id = first.id.clone();
    let saved = saver
        .put(
            &root,
            &first,
            &metadata(CheckpointSource::Input, -1),
            &BTreeMap::from([("n".to_string(), 1)]),
        )
        .await
        .expect("put input");
    assert_eq!(saved.checkpoint_id.as_deref(), Some(first_id.as_str()));

    let second = checkpoint("n", json!(2), 2);
    let second_id = second.id.clone();
    let saved = saver
        .put(
            &saved,
            &second,
            &metadata(CheckpointSource::Loop, 0),
            &BTreeMap::from([("n".to_string(), 2)]),
        )
        .await
        .expect("put loop");
    assert_eq!(saved.checkpoint_id.as_deref(), Some(second_id.as_str()));

    let third = checkpoint("n", json!(3), 3);
    third_put(saver, &saved, third).await;

    let latest = saver.get_tuple(&root).await.expect("latest").expect("some");
    assert_eq!(latest.checkpoint.channel_values.get("n"), Some(&json!(3)));
    assert_eq!(latest.metadata.source, CheckpointSource::Loop);
    assert_eq!(latest.metadata.step, 1);
    assert_eq!(
        latest
            .parent_config
            .as_ref()
            .and_then(|parent| parent.checkpoint_id.as_deref()),
        Some(second_id.as_str())
    );
    assert_eq!(latest.checkpoint.v, 1);
    let parsed = uuid::Uuid::parse_str(&latest.checkpoint.id).expect("uuid");
    assert_eq!(parsed.get_version(), Some(uuid::Version::SortRand));
    assert!(
        latest.checkpoint.ts.ends_with('Z') && latest.checkpoint.ts.contains('T'),
        "{}",
        latest.checkpoint.ts
    );

    let history = saver.list(&root, None, None).await.expect("history");
    assert_eq!(history.len(), 3);
    assert_eq!(
        history[0].checkpoint.channel_values.get("n"),
        Some(&json!(3))
    );
    assert_eq!(
        history[2].checkpoint.channel_values.get("n"),
        Some(&json!(1))
    );
    assert_eq!(history[2].metadata.step, -1);
    assert!(history[2].parent_config.is_none());
    assert_eq!(
        history[2].metadata.parents.get("").map(String::as_str),
        None
    );

    let limited = saver.list(&root, None, Some(1)).await.expect("limit");
    assert_eq!(limited.len(), 1);
    assert_eq!(limited[0].checkpoint.id, history[0].checkpoint.id);

    let older = saver
        .list(&root, Some(&history[0].checkpoint.id), None)
        .await
        .expect("before");
    assert_eq!(older.len(), 2);
    assert_eq!(older[0].checkpoint.id, history[1].checkpoint.id);

    let tip_before_fork = history[0].clone();
    let forked = checkpoint("n", json!(9), 4);
    let forked_id = forked.id.clone();
    let forked_saved = saver
        .put(
            &config("thread", "", Some(&first_id)),
            &forked,
            &metadata(CheckpointSource::Fork, -1),
            &BTreeMap::from([("n".to_string(), 4)]),
        )
        .await
        .expect("fork");
    assert_eq!(
        forked_saved.checkpoint_id.as_deref(),
        Some(forked_id.as_str())
    );

    let after = saver.list(&root, None, None).await.expect("after fork");
    assert_eq!(after[0].checkpoint.id, forked_id);
    assert_eq!(
        after[0]
            .parent_config
            .as_ref()
            .and_then(|parent| parent.checkpoint_id.as_deref()),
        Some(first_id.as_str())
    );
    assert_eq!(after[0].metadata.source, CheckpointSource::Fork);
    let unchanged = after
        .iter()
        .find(|tuple| tuple.checkpoint.id == tip_before_fork.checkpoint.id)
        .expect("previous tip");
    assert_eq!(
        unchanged.checkpoint.channel_values,
        tip_before_fork.checkpoint.channel_values
    );
    assert_eq!(unchanged.metadata, tip_before_fork.metadata);
    assert_eq!(unchanged.parent_config, tip_before_fork.parent_config);

    let specific = saver
        .get_tuple(&config("thread", "", Some(&first_id)))
        .await
        .expect("by id")
        .expect("found");
    assert_eq!(specific.checkpoint.channel_values.get("n"), Some(&json!(1)));

    let missing = saver
        .get_tuple(&config("thread", "", Some("missing")))
        .await
        .expect_err("unknown id");
    assert!(matches!(missing, Error::UnknownCheckpoint(ref id) if id == "missing"));

    saver
        .put_writes(
            &config("thread", "", Some(&forked_id)),
            &[PendingWrite {
                task_id: String::new(),
                channel: "n".to_string(),
                value: json!(8),
            }],
            "node-a",
        )
        .await
        .expect("writes");
    let with_writes = saver
        .get_tuple(&config("thread", "", Some(&forked_id)))
        .await
        .expect("writes get")
        .expect("found");
    assert_eq!(with_writes.pending_writes.len(), 1);
    assert_eq!(with_writes.pending_writes[0].task_id, "node-a");
    assert_eq!(with_writes.pending_writes[0].value, json!(8));
    assert_eq!(
        with_writes.checkpoint.channel_values.get("n"),
        Some(&json!(9))
    );

    let other_ns = config("thread", "child", None);
    assert!(saver.get_tuple(&other_ns).await.expect("ns").is_none());
    let namespaced = checkpoint("n", json!(5), 1);
    saver
        .put(
            &other_ns,
            &namespaced,
            &metadata(CheckpointSource::Input, -1),
            &BTreeMap::from([("n".to_string(), 1)]),
        )
        .await
        .expect("ns put");
    assert!(saver
        .get_tuple(&other_ns)
        .await
        .expect("ns latest")
        .is_some());
    assert_eq!(
        saver
            .list(&root, None, None)
            .await
            .expect("root list")
            .len(),
        4
    );

    let other_thread = config("other", "", None);
    let other_checkpoint = checkpoint("n", json!(7), 1);
    saver
        .put(
            &other_thread,
            &other_checkpoint,
            &metadata(CheckpointSource::Update, 0),
            &BTreeMap::from([("n".to_string(), 1)]),
        )
        .await
        .expect("other thread");

    saver.delete_thread("thread").await.expect("delete");
    assert!(saver.get_tuple(&root).await.expect("deleted").is_none());
    assert!(saver
        .get_tuple(&other_ns)
        .await
        .expect("deleted ns")
        .is_none());
    assert!(saver
        .get_tuple(&other_thread)
        .await
        .expect("other remains")
        .is_some());
    saver.delete_thread("thread").await.expect("delete again");

    let bad_version = {
        let mut checkpoint = checkpoint("n", json!(1), 1);
        checkpoint.v = 2;
        checkpoint
    };
    let error = saver
        .put(
            &other_thread,
            &bad_version,
            &metadata(CheckpointSource::Input, -1),
            &BTreeMap::new(),
        )
        .await
        .expect_err("version");
    assert!(matches!(error, Error::Storage(ref message) if message.contains("version")));

    let mismatched = checkpoint("n", json!(1), 1);
    let error = saver
        .put(
            &other_thread,
            &mismatched,
            &metadata(CheckpointSource::Input, -1),
            &BTreeMap::from([("n".to_string(), 9)]),
        )
        .await
        .expect_err("delta");
    assert!(matches!(error, Error::Storage(ref message) if message.contains("new version")));
}

async fn third_put(saver: &dyn Checkpointer, parent: &CheckpointConfig, checkpoint: Checkpoint) {
    saver
        .put(
            parent,
            &checkpoint,
            &metadata(CheckpointSource::Loop, 1),
            &BTreeMap::from([("n".to_string(), 3)]),
        )
        .await
        .expect("put third");
}

#[tokio::test]
async fn memory_saver_matches_the_checkpointer_contract() {
    assert_checkpointer(&crate::InMemorySaver::new()).await;
}

#[tokio::test]
async fn two_memory_savers_do_not_share_a_map() {
    let first = crate::InMemorySaver::new();
    let second = crate::InMemorySaver::new();
    let checkpoint = checkpoint("n", json!(1), 1);
    let saved = first
        .put(
            &config("thread", "", None),
            &checkpoint,
            &metadata(CheckpointSource::Input, -1),
            &BTreeMap::from([("n".to_string(), 1)]),
        )
        .await
        .expect("put");
    assert!(saved.checkpoint_id.is_some());
    assert!(second
        .get_tuple(&config("thread", "", None))
        .await
        .expect("other saver")
        .is_none());
}
