use loro::{ExportMode, FractionalIndex, LoroDoc, LoroError, LoroTree, LoroTreeError, TreeID};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};

fn fi(hex: &str) -> FractionalIndex {
    FractionalIndex::from_hex_string(hex)
}

fn new_doc(peer: u64) -> LoroDoc {
    let doc = LoroDoc::new();
    doc.set_peer_id(peer).unwrap();
    doc.get_tree("tree").enable_fractional_index(0);
    doc
}

/// Key shapes an external order owner mints with the same algorithm:
/// appends, a long run of inserts at the head (bytes reach 0x00), a long run
/// of inserts between two neighbours (long keys), and an even spread.
fn minted_key_shapes() -> Vec<Vec<FractionalIndex>> {
    let mut append = vec![FractionalIndex::default()];
    for _ in 0..300 {
        let next = FractionalIndex::new(append.last(), None).unwrap();
        append.push(next);
    }

    let mut head = vec![FractionalIndex::default()];
    for _ in 0..300 {
        let first = FractionalIndex::new(None, head.first()).unwrap();
        head.insert(0, first);
    }

    let mut dense = vec![
        FractionalIndex::default(),
        FractionalIndex::new(Some(&FractionalIndex::default()), None).unwrap(),
    ];
    for i in 0..300 {
        let (lo, hi) = if i % 2 == 0 {
            (0, 1)
        } else {
            (dense.len() - 2, dense.len() - 1)
        };
        let mid = FractionalIndex::new(Some(&dense[lo]), Some(&dense[hi])).unwrap();
        dense.insert(lo + 1, mid);
    }

    let even = FractionalIndex::generate_n_evenly(None, None, 1000).unwrap();
    vec![append, head, dense, even]
}

fn adopt_shuffled(
    tree: &LoroTree,
    parent: Option<TreeID>,
    keys: &[FractionalIndex],
    seed: u64,
) -> Vec<TreeID> {
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.shuffle(&mut StdRng::seed_from_u64(seed));
    let mut ids = vec![None; keys.len()];
    for i in order {
        ids[i] = Some(
            tree.create_at_fractional_index(parent, keys[i].clone())
                .unwrap(),
        );
    }
    ids.into_iter().map(Option::unwrap).collect()
}

#[test]
fn sibling_order_follows_the_given_keys_and_keys_read_back_unchanged() {
    for (shape, keys) in minted_key_shapes().into_iter().enumerate() {
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted, "shape {shape}: generator emits keys in order");

        let doc = new_doc(1);
        let tree = doc.get_tree("tree");
        let parent = tree.create(None).unwrap();
        let ids = adopt_shuffled(&tree, Some(parent), &keys, shape as u64);
        doc.commit();

        assert_eq!(tree.children(parent).unwrap(), ids, "shape {shape}");
        for (id, key) in ids.iter().zip(&keys) {
            assert_eq!(tree.fractional_index(*id).unwrap(), key.to_string());
        }
        doc.check_state_correctness_slow();
    }
}

#[test]
fn a_generated_insert_between_adopted_nodes_lands_between_them() {
    for keys in minted_key_shapes() {
        let doc = new_doc(1);
        let tree = doc.get_tree("tree");
        let ids = adopt_shuffled(&tree, None, &keys, 7);
        let mut rng = StdRng::seed_from_u64(11);
        let mut expected = ids.clone();
        for _ in 0..200 {
            let index = rng.gen_range(0..=expected.len());
            let new = tree.create_at(None, index).unwrap();
            expected.insert(index, new);
        }
        // Moves re-generate positions between adopted neighbours too.
        for _ in 0..200 {
            let from = rng.gen_range(0..expected.len());
            let node = expected.remove(from);
            let to = rng.gen_range(0..=expected.len());
            tree.mov_to(node, None, to).unwrap();
            expected.insert(to, node);
        }
        doc.commit();
        assert_eq!(tree.roots(), expected);
        let positions: Vec<String> = expected
            .iter()
            .map(|id| tree.fractional_index(*id).unwrap())
            .collect();
        for pair in positions.windows(2) {
            assert!(pair[0] < pair[1], "{pair:?}");
        }
        doc.check_state_correctness_slow();
    }
}

#[test]
fn adopted_positions_survive_snapshot_and_update_import() {
    let keys = minted_key_shapes().remove(2);
    let doc = new_doc(1);
    let tree = doc.get_tree("tree");
    let ids = adopt_shuffled(&tree, None, &keys, 3);
    doc.commit();

    for mode in [ExportMode::Snapshot, ExportMode::all_updates()] {
        let other = LoroDoc::new();
        other.import(&doc.export(mode).unwrap()).unwrap();
        let other_tree = other.get_tree("tree");
        assert_eq!(other_tree.roots(), ids);
        for (id, key) in ids.iter().zip(&keys) {
            assert_eq!(other_tree.fractional_index(*id).unwrap(), key.to_string());
        }
        other.check_state_correctness_slow();
    }
}

#[test]
fn concurrent_inserts_next_to_adopted_nodes_converge() {
    let keys = FractionalIndex::generate_n_evenly(None, None, 20).unwrap();
    let base = new_doc(1);
    let base_tree = base.get_tree("tree");
    let parent = base_tree.create(None).unwrap();
    adopt_shuffled(&base_tree, Some(parent), &keys, 5);
    base.commit();
    let snapshot = base.export(ExportMode::Snapshot).unwrap();

    let mut rng = StdRng::seed_from_u64(99);
    for round in 0..50 {
        let a = new_doc(10);
        let b = new_doc(20);
        a.import(&snapshot).unwrap();
        b.import(&snapshot).unwrap();
        for doc in [&a, &b] {
            let tree = doc.get_tree("tree");
            for _ in 0..rng.gen_range(1..5) {
                let len = tree.children_num(parent).unwrap();
                // Same index on both peers in half the rounds: equal
                // generated positions, resolved by the op id.
                let index = if round % 2 == 0 {
                    3
                } else {
                    rng.gen_range(0..=len)
                };
                tree.create_at(parent, index).unwrap();
            }
            doc.commit();
        }
        a.import(&b.export(ExportMode::all_updates()).unwrap())
            .unwrap();
        b.import(&a.export(ExportMode::all_updates()).unwrap())
            .unwrap();
        let (ta, tb) = (a.get_tree("tree"), b.get_tree("tree"));
        assert_eq!(ta.children(parent), tb.children(parent), "round {round}");
        assert_eq!(a.get_deep_value(), b.get_deep_value(), "round {round}");
        // A generated insert next to tied positions still works afterwards.
        ta.create_at(parent, 4).unwrap();
        a.commit();
        a.check_state_correctness_slow();
        b.check_state_correctness_slow();
    }
}

#[test]
fn a_position_a_sibling_holds_is_refused() {
    let doc = new_doc(1);
    let tree = doc.get_tree("tree");
    let parent = tree.create(None).unwrap();
    tree.create_at_fractional_index(parent, fi("80")).unwrap();
    tree.create_at_fractional_index(parent, fi("8180")).unwrap();
    for taken in ["80", "8180"] {
        let err = tree
            .create_at_fractional_index(parent, fi(taken))
            .unwrap_err();
        assert_eq!(
            err,
            LoroError::TreeError(LoroTreeError::DuplicateFractionalIndex {
                parent: Some(parent),
                position: taken.into(),
            })
        );
    }
    // Uniqueness is per parent: `parent` itself holds the generated root
    // position "80", and "8180" is free at the root level.
    assert!(matches!(
        tree.create_at_fractional_index(None, fi("80")).unwrap_err(),
        LoroError::TreeError(LoroTreeError::DuplicateFractionalIndex { parent: None, .. })
    ));
    tree.create_at_fractional_index(None, fi("8180")).unwrap();
    assert_eq!(tree.children_num(parent), Some(2));
    assert_eq!(tree.roots().len(), 2);
}

#[test]
fn malformed_input_is_refused() {
    let doc = new_doc(1);
    let tree = doc.get_tree("tree");
    for bad in ["", "A0", "8081", "41303030303031"] {
        let err = tree.create_at_fractional_index(None, fi(bad)).unwrap_err();
        assert!(
            matches!(
                err,
                LoroError::TreeError(LoroTreeError::InvalidFractionalIndex(_))
            ),
            "{bad:?}: {err:?}"
        );
    }
    let missing = TreeID::new(9, 9);
    assert_eq!(
        tree.create_at_fractional_index(missing, fi("80"))
            .unwrap_err(),
        LoroError::TreeError(LoroTreeError::TreeNodeParentNotFound(missing))
    );
    tree.disable_fractional_index();
    assert_eq!(
        tree.create_at_fractional_index(None, fi("80")).unwrap_err(),
        LoroError::TreeError(LoroTreeError::FractionalIndexNotEnabled)
    );
    assert!(tree.roots().is_empty());
}
