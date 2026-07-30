//! Purge of root containers whose owning tree node was deleted.
//!
//! Holon models a block's rich text as a *mergeable* root container hanging off the
//! block's tree-node meta map. Deleting the block deletes the tree node, which makes the
//! mergeable child unreachable — but before this fix its content stayed in doc state and
//! in every export, because `delete_root_container` could not emit the emptying ops on an
//! already-dead container (it failed with `ContainerDeleted` and only `eprintln!`d).
//!
//! What a purge can and cannot reach:
//! - **Shallow snapshots** drop the purged bytes — that is the deliverable.
//! - **Full `ExportMode::Snapshot`** keeps them, because it carries the complete op
//!   history and the original insert ops are part of it. That is true of *every* deletion
//!   in Loro, purge or not; removing it needs history redaction, not a purge.

use loro::{ContainerID, ContainerTrait, ContainerType, ExportMode, LoroDoc, TreeParentId};
use serial_test::parallel;

const SECRET_TEXT: &str = "PURGE-PROBE-TEXT-c0ffee";
const SECRET_MARK: &str = "PURGE-PROBE-MARK-deadbeef";

fn contains(haystack: &[u8], needle: &str) -> bool {
    haystack
        .windows(needle.len())
        .any(|w| w == needle.as_bytes())
}

fn shallow(doc: &LoroDoc) -> Vec<u8> {
    doc.export(ExportMode::shallow_snapshot(&doc.oplog_frontiers()))
        .unwrap()
}

fn sync(a: &LoroDoc, b: &LoroDoc) {
    a.import(&b.export(ExportMode::updates(&a.oplog_vv())).unwrap())
        .unwrap();
    b.import(&a.export(ExportMode::updates(&b.oplog_vv())).unwrap())
        .unwrap();
}

/// A doc with a tree node owning a mergeable text child carrying both secrets. The node
/// is deleted, so the child is unreachable but its content is still resident.
fn doc_with_deleted_owner(peer: u64) -> (LoroDoc, ContainerID) {
    let doc = LoroDoc::new();
    doc.set_peer_id(peer).unwrap();

    let tree = doc.get_tree("blocks");
    let node = tree.create(TreeParentId::Root).unwrap();
    let meta = tree.get_meta(node).unwrap();
    let text = meta.ensure_mergeable_text("body").unwrap();
    let cid = text.id();

    text.insert(0, SECRET_TEXT).unwrap();
    text.mark(0..SECRET_TEXT.len(), "comment", SECRET_MARK)
        .unwrap();
    doc.commit();

    tree.delete(node).unwrap();
    doc.commit();

    (doc, cid)
}

/// The core fix: purging a container whose owner is gone succeeds and its bytes leave the
/// shallow snapshot.
#[test]
#[parallel]
fn purge_of_deleted_owner_container_clears_shallow_snapshot() {
    let (doc, cid) = doc_with_deleted_owner(1);

    assert!(
        contains(&shallow(&doc), SECRET_TEXT),
        "precondition: content is stranded in the shallow snapshot before the purge"
    );

    doc.delete_root_container(cid).unwrap();
    doc.commit();

    let after = shallow(&doc);
    assert!(
        !contains(&after, SECRET_TEXT),
        "text secret survives shallow-snapshot export after purge"
    );
    assert!(
        !contains(&after, SECRET_MARK),
        "mark secret survives shallow-snapshot export after purge"
    );

    // A peer bootstrapping from that shallow snapshot never sees the content either.
    let fresh = LoroDoc::new();
    fresh.import(&after).unwrap();
    assert!(!contains(&shallow(&fresh), SECRET_TEXT));
    assert!(!contains(
        &fresh.export(ExportMode::Snapshot).unwrap(),
        SECRET_TEXT
    ));
}

/// Convergence: A purges, B never does. After a full sync both agree on the document
/// value, and B's own shallow export no longer carries the secret either — the purge ops
/// are ordinary CRDT ops, so B applies them like any other deletion.
#[test]
#[parallel]
fn purging_and_non_purging_peers_converge() {
    let (a, cid) = doc_with_deleted_owner(1);
    let b = LoroDoc::new();
    b.set_peer_id(2).unwrap();
    b.import(&a.export(ExportMode::Snapshot).unwrap()).unwrap();

    a.delete_root_container(cid).unwrap();
    a.commit();
    sync(&a, &b);

    assert_eq!(
        a.get_deep_value(),
        b.get_deep_value(),
        "purging and non-purging peers diverged on document value"
    );
    assert!(!contains(&shallow(&a), SECRET_TEXT));
    assert!(
        !contains(&shallow(&b), SECRET_TEXT),
        "peer that never purged still leaks the secret through its own shallow snapshot"
    );
}

/// Convergence under concurrency: B resurrects the block (re-creates the node and
/// re-marks the mergeable child) while A concurrently purges. Both peers must land on the
/// same value — a purged span stays deleted, and B's post-purge writes survive.
#[test]
#[parallel]
fn concurrent_purge_and_resurrect_converge() {
    let (a, cid) = doc_with_deleted_owner(1);
    let b = LoroDoc::new();
    b.set_peer_id(2).unwrap();
    b.import(&a.export(ExportMode::Snapshot).unwrap()).unwrap();

    // A purges.
    a.delete_root_container(cid).unwrap();
    a.commit();

    // B concurrently re-attaches a mergeable child under a fresh node and writes to it.
    let b_tree = b.get_tree("blocks");
    let b_node = b_tree.create(TreeParentId::Root).unwrap();
    let b_text = b_tree
        .get_meta(b_node)
        .unwrap()
        .ensure_mergeable_text("body")
        .unwrap();
    b_text.insert(0, "REVIVED").unwrap();
    b.commit();

    sync(&a, &b);

    assert_eq!(
        a.get_deep_value(),
        b.get_deep_value(),
        "concurrent purge and resurrect diverged"
    );
    let a_json = format!("{:?}", a.get_deep_value());
    assert!(
        !a_json.contains(SECRET_TEXT),
        "purged span reappeared after resurrect: {a_json}"
    );
    assert!(
        a_json.contains("REVIVED"),
        "post-purge write lost: {a_json}"
    );
}

/// Failures must propagate, not print. Before this fix, every one of these was a silent
/// no-op or an `eprintln!`.
#[test]
#[parallel]
fn purge_failures_are_returned_not_printed() {
    let doc = LoroDoc::new();
    doc.set_peer_id(1).unwrap();
    doc.get_text("present");

    // Plain roots are lazily created, so any name "exists"; a mergeable root does not.
    let absent_mergeable =
        ContainerID::new_mergeable(&doc.get_map("nothing").id(), "body", ContainerType::Text);
    assert!(
        doc.delete_root_container(absent_mergeable).is_err(),
        "purging an unknown mergeable container must error"
    );

    let map = doc.get_map("m");
    let nested = map
        .insert_container("inner", loro::LoroText::new())
        .unwrap();
    assert!(
        doc.delete_root_container(nested.id()).is_err(),
        "purging a non-root container must error"
    );

    doc.delete_root_container(ContainerID::new_root("present", ContainerType::Text))
        .unwrap();
}

/// Purging a live root container keeps working exactly as before.
#[test]
#[parallel]
fn purge_of_live_root_container_still_works() {
    let doc = LoroDoc::new();
    doc.set_peer_id(1).unwrap();
    let text = doc.get_text("body");
    text.insert(0, SECRET_TEXT).unwrap();
    doc.commit();

    doc.delete_root_container(text.id()).unwrap();
    doc.commit();

    assert!(!contains(&shallow(&doc), SECRET_TEXT));
}

/// NOT YET IMPLEMENTED — see the module docs and the report accompanying this branch.
///
/// A full `ExportMode::Snapshot` embeds the whole op history, so the purged text is still
/// recoverable from it. Making this green requires redacting the *state/oplog* encoding
/// the way `loro_internal::json::redact` (PR #504) redacts the JSON-updates encoding:
/// replace the content of ops whose target container is provably dead, keeping op ids and
/// lengths so the causal structure and every peer's version vector stay intact.
#[test]
#[parallel]
#[ignore = "requires history redaction in the snapshot encoding; purge alone cannot reach it"]
fn purge_removes_secrets_from_full_snapshot() {
    let (doc, cid) = doc_with_deleted_owner(1);
    doc.delete_root_container(cid).unwrap();
    doc.commit();

    let snapshot = doc.export(ExportMode::Snapshot).unwrap();
    assert!(!contains(&snapshot, SECRET_TEXT));
    assert!(!contains(&snapshot, SECRET_MARK));
}
