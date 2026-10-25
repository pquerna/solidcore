//! Capacity-visible ordered lookup storage for retained tessellation indexes.
//!
//! The AVL arena keeps arbitrary arrival orders logarithmic without hiding tree
//! allocations. Parent links also let consuming iteration unlink the minimum
//! in place: sorted output needs neither recursion nor another backing buffer.

use std::mem::size_of;

use crate::budget::QueryBudget;
use solidcore::Result;

#[derive(Debug, Clone)]
struct Node<K, V> {
    entry: (K, V),
    parent: Option<usize>,
    children: [Option<usize>; 2],
    height: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct FlatIndex<K, V> {
    nodes: Vec<Node<K, V>>,
    root: Option<usize>,
}

impl<K, V> Default for FlatIndex<K, V> {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            root: None,
        }
    }
}

struct SearchPosition {
    next: Option<usize>,
    found: Option<usize>,
    parent: Option<usize>,
    side: usize,
}

impl<K: Ord, V> FlatIndex<K, V> {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn clear(&mut self) {
        self.nodes.clear();
        self.root = None;
    }

    pub(crate) fn get(&self, key: &K) -> Option<&V> {
        self.search(key)
            .found
            .map(|index| &self.nodes[index].entry.1)
    }

    pub(crate) fn get_with_budget(&self, key: &K, budget: &QueryBudget<'_>) -> Result<Option<&V>> {
        Ok(self
            .search_with_budget(key, budget)?
            .found
            .map(|index| &self.nodes[index].entry.1))
    }

    pub(crate) fn insert(&mut self, key: K, value: V) -> Option<V> {
        let position = self.search(&key);
        self.insert_at(position, key, value)
    }

    pub(crate) fn insert_with_budget(
        &mut self,
        key: K,
        value: V,
        budget: &QueryBudget<'_>,
    ) -> Result<Option<V>> {
        let position = self.search_with_budget(&key, budget)?;
        if position.found.is_some() {
            budget.charge(1)?;
        } else {
            // Admit the complete mutation before linking the new node. A refusal
            // cannot leave an incompletely rebalanced tree behind. Each ancestor
            // needs bounded height/link work, including at most two rotations.
            budget.charge(1 + 32 * u64::from(self.height(self.root)))?;
            if self.nodes.len() == self.nodes.capacity() {
                budget.charge(self.nodes.len() as u64)?;
            }
            // QueryBudget admits old/new coexistence and any actual excess
            // capacity before this insertion can publish the new node.
            budget.reserve_vec(&mut self.nodes, 1)?;
        }
        Ok(self.insert_at(position, key, value))
    }

    pub(crate) fn retained_buffer_bytes(&self) -> usize {
        vector_bytes(&self.nodes)
    }

    fn search_start(&self) -> SearchPosition {
        SearchPosition {
            next: self.root,
            found: None,
            parent: None,
            side: 0,
        }
    }

    fn search_step(&self, key: &K, position: &mut SearchPosition) {
        let Some(index) = position.next else {
            return;
        };
        match key.cmp(&self.nodes[index].entry.0) {
            std::cmp::Ordering::Equal => {
                position.found = Some(index);
                position.next = None;
            }
            ordering => {
                position.parent = Some(index);
                position.side = usize::from(ordering == std::cmp::Ordering::Greater);
                position.next = self.nodes[index].children[position.side];
            }
        }
    }

    fn search(&self, key: &K) -> SearchPosition {
        let mut position = self.search_start();
        while position.next.is_some() {
            self.search_step(key, &mut position);
        }
        position
    }

    fn search_with_budget(&self, key: &K, budget: &QueryBudget<'_>) -> Result<SearchPosition> {
        budget.checkpoint()?;
        let mut position = self.search_start();
        while position.next.is_some() {
            budget.charge(1)?;
            self.search_step(key, &mut position);
        }
        Ok(position)
    }

    // Both entrypoints use this allocation/mutation core. The bounded caller
    // has already reserved the arena and admitted all remaining mutation work.
    fn insert_at(&mut self, position: SearchPosition, key: K, value: V) -> Option<V> {
        if let Some(index) = position.found {
            return Some(std::mem::replace(&mut self.nodes[index].entry.1, value));
        }
        let index = self.nodes.len();
        self.nodes.push(Node {
            entry: (key, value),
            parent: position.parent,
            children: [None, None],
            height: 1,
        });
        if let Some(parent) = position.parent {
            self.nodes[parent].children[position.side] = Some(index);
        } else {
            self.root = Some(index);
        }
        self.rebalance_from(position.parent);
        None
    }

    fn height(&self, index: Option<usize>) -> u32 {
        index.map_or(0, |index| self.nodes[index].height)
    }

    fn update_height(&mut self, index: usize) {
        let [left, right] = self.nodes[index].children;
        self.nodes[index].height = 1 + self.height(left).max(self.height(right));
    }

    // Promote child `side` (left child = right rotation, and vice versa).
    fn rotate(&mut self, index: usize, side: usize) -> usize {
        let Some(pivot) = self.nodes[index].children[side] else {
            return index;
        };
        let parent = self.nodes[index].parent;
        let middle = self.nodes[pivot].children[1 - side];
        self.nodes[index].children[side] = middle;
        if let Some(middle) = middle {
            self.nodes[middle].parent = Some(index);
        }
        self.nodes[pivot].parent = parent;
        if let Some(parent) = parent {
            let old_side = usize::from(self.nodes[parent].children[1] == Some(index));
            self.nodes[parent].children[old_side] = Some(pivot);
        } else {
            self.root = Some(pivot);
        }
        self.nodes[pivot].children[1 - side] = Some(index);
        self.nodes[index].parent = Some(pivot);
        self.update_height(index);
        self.update_height(pivot);
        pivot
    }

    fn rebalance_from(&mut self, mut next: Option<usize>) {
        while let Some(index) = next {
            self.update_height(index);
            let [left, right] = self.nodes[index].children;
            let left_height = self.height(left);
            let right_height = self.height(right);
            let mut top = index;
            if left_height > right_height.saturating_add(1) {
                if let Some(left) = left {
                    let [outer, inner] = self.nodes[left].children;
                    if self.height(inner) > self.height(outer) {
                        self.rotate(left, 1);
                    }
                    top = self.rotate(index, 0);
                }
            } else if right_height > left_height.saturating_add(1) {
                if let Some(right) = right {
                    let [inner, outer] = self.nodes[right].children;
                    if self.height(inner) > self.height(outer) {
                        self.rotate(right, 0);
                    }
                    top = self.rotate(index, 1);
                }
            }
            next = self.nodes[top].parent;
        }
    }
}

pub(crate) fn vector_bytes<T>(values: &Vec<T>) -> usize {
    values.capacity().saturating_mul(size_of::<T>())
}

pub(crate) struct FlatIndexIntoIter<K, V> {
    nodes: Vec<Node<K, V>>,
    next: Option<usize>,
}

impl<K, V> FlatIndexIntoIter<K, V> {
    pub(crate) fn next_with_budget(&mut self, budget: &QueryBudget<'_>) -> Result<Option<(K, V)>> {
        budget.checkpoint()?;
        let Some(mut index) = self.next else {
            return Ok(None);
        };
        // Removal/relinking has constant work. Charge before any mutation;
        // descent can then stop without changing the iterator.
        budget.charge(16)?;
        while let Some(left) = self.nodes[index].children[0] {
            budget.charge(1)?;
            index = left;
        }
        Ok(Some(self.remove_minimum(index)))
    }

    fn remove_minimum(&mut self, index: usize) -> (K, V) {
        let parent = self.nodes[index].parent;
        let right = self.nodes[index].children[1];
        let mut next = right.or(parent);
        if let Some(parent) = parent {
            let side = usize::from(self.nodes[parent].children[1] == Some(index));
            self.nodes[parent].children[side] = right;
        }
        if let Some(right) = right {
            self.nodes[right].parent = parent;
        }

        let last = self.nodes.len() - 1;
        let removed = self.nodes.swap_remove(index);
        if index != last {
            // Only the last arena slot moved. Repair its parent and children;
            // no keys, payloads or additional buffers need to be copied.
            let moved_parent = self.nodes[index].parent;
            if let Some(parent) = moved_parent {
                let side = usize::from(self.nodes[parent].children[1] == Some(last));
                self.nodes[parent].children[side] = Some(index);
            }
            for child in self.nodes[index].children.into_iter().flatten() {
                self.nodes[child].parent = Some(index);
            }
            if next == Some(last) {
                next = Some(index);
            }
        }
        self.next = next;
        removed.entry
    }
}

impl<K, V> Iterator for FlatIndexIntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        let mut index = self.next?;
        while let Some(left) = self.nodes[index].children[0] {
            index = left;
        }
        Some(self.remove_minimum(index))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.nodes.len(), Some(self.nodes.len()))
    }
}

impl<K, V> ExactSizeIterator for FlatIndexIntoIter<K, V> {}
impl<K, V> std::iter::FusedIterator for FlatIndexIntoIter<K, V> {}

impl<K, V> IntoIterator for FlatIndex<K, V> {
    type Item = (K, V);
    type IntoIter = FlatIndexIntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        FlatIndexIntoIter {
            nodes: self.nodes,
            next: self.root,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FlatSet<T> {
    entries: FlatIndex<T, ()>,
}

impl<T> Default for FlatSet<T> {
    fn default() -> Self {
        Self {
            entries: FlatIndex::default(),
        }
    }
}

impl<T: Ord> FlatSet<T> {
    pub(crate) fn new() -> Self {
        Self::default()
    }
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }
    pub(crate) fn contains(&self, value: &T) -> bool {
        self.entries.get(value).is_some()
    }

    pub(crate) fn contains_with_budget(&self, value: &T, budget: &QueryBudget<'_>) -> Result<bool> {
        Ok(self.entries.get_with_budget(value, budget)?.is_some())
    }

    pub(crate) fn insert_with_budget(
        &mut self,
        value: T,
        budget: &QueryBudget<'_>,
    ) -> Result<bool> {
        Ok(self
            .entries
            .insert_with_budget(value, (), budget)?
            .is_none())
    }

    pub(crate) fn retained_buffer_bytes(&self) -> usize {
        self.entries.retained_buffer_bytes()
    }
}

pub(crate) struct FlatSetIntoIter<T> {
    entries: FlatIndexIntoIter<T, ()>,
}

impl<T> FlatSetIntoIter<T> {
    pub(crate) fn next_with_budget(&mut self, budget: &QueryBudget<'_>) -> Result<Option<T>> {
        Ok(self.entries.next_with_budget(budget)?.map(|entry| entry.0))
    }
}

impl<T> Iterator for FlatSetIntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.entries.next().map(|entry| entry.0)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.entries.size_hint()
    }
}

impl<T> ExactSizeIterator for FlatSetIntoIter<T> {}
impl<T> std::iter::FusedIterator for FlatSetIntoIter<T> {}

impl<T> IntoIterator for FlatSet<T> {
    type Item = T;
    type IntoIter = FlatSetIntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        FlatSetIntoIter {
            entries: self.entries.into_iter(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::QueryLimit;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn assert_avl(index: &FlatIndex<usize, usize>) {
        assert_eq!(index.root.is_none(), index.nodes.is_empty());
        for (id, node) in index.nodes.iter().enumerate() {
            let [left, right] = node.children;
            let left_height = index.height(left);
            let right_height = index.height(right);
            assert!(left_height.abs_diff(right_height) <= 1);
            assert_eq!(node.height, 1 + left_height.max(right_height));
            for (side, child) in node.children.into_iter().enumerate() {
                if let Some(child) = child {
                    assert_eq!(index.nodes[child].parent, Some(id));
                    if side == 0 {
                        assert!(index.nodes[child].entry.0 < node.entry.0);
                    } else {
                        assert!(index.nodes[child].entry.0 > node.entry.0);
                    }
                }
            }
            if let Some(parent) = node.parent {
                assert!(index.nodes[parent].children.contains(&Some(id)));
            } else {
                assert_eq!(index.root, Some(id));
            }
        }
    }

    fn descending_work(count: usize) -> u64 {
        let cancel = AtomicBool::new(false);
        let budget = QueryBudget::new(u64::MAX, usize::MAX, &cancel);
        let mut index = FlatIndex::new();
        for key in (0..count).rev() {
            assert_eq!(
                index.insert_with_budget(key, key + 1, &budget).unwrap(),
                None
            );
        }
        let work = budget.consumed();
        assert_avl(&index);
        assert_eq!(
            index.retained_buffer_bytes(),
            index.nodes.capacity() * size_of::<Node<usize, usize>>()
        );
        for key in 0..count {
            assert_eq!(
                index.get_with_budget(&key, &budget).unwrap(),
                Some(&(key + 1))
            );
        }
        let mut entries = index.into_iter();
        let bytes = budget.admitted_bytes();
        for key in 0..count {
            assert_eq!(entries.len(), count - key);
            assert_eq!(
                entries.next_with_budget(&budget).unwrap(),
                Some((key, key + 1))
            );
        }
        assert_eq!(entries.next_with_budget(&budget).unwrap(), None);
        assert_eq!(budget.admitted_bytes(), bytes);
        work
    }

    #[test]
    fn descending_arrival_has_logarithmic_insertion_work_and_sorted_output() {
        let small = descending_work(512);
        let large = descending_work(4096);
        // Eight times as many descending arrivals must stay well below the
        // 64-fold shift work of sorted-Vec insertion.
        assert!(large <= small * 12);
    }

    #[test]
    fn rotations_replacement_and_legacy_consumption_keep_order() {
        for arrivals in [[3, 2, 1, 0], [0, 1, 2, 3], [3, 1, 2, 0], [0, 2, 1, 3]] {
            let mut index = FlatIndex::new();
            for key in arrivals {
                assert_eq!(index.insert(key, key + 1), None);
                assert_avl(&index);
            }
            assert_eq!(index.insert(2, 99), Some(3));
            assert_eq!(index.get(&2), Some(&99));
            assert_avl(&index);
            assert_eq!(
                index.into_iter().collect::<Vec<_>>(),
                vec![(0, 1), (1, 2), (2, 99), (3, 4)]
            );
        }
    }

    #[test]
    fn clear_retains_arena_and_duplicate_sets_keep_boolean_contract() {
        let budget = QueryBudget::unbounded();
        let mut set = FlatSet::new();
        assert!(set.insert_with_budget(7, &budget).unwrap());
        assert!(!set.insert_with_budget(7, &budget).unwrap());
        assert!(set.contains_with_budget(&7, &budget).unwrap());
        let bytes = set.retained_buffer_bytes();
        set.clear();
        assert_eq!(set.retained_buffer_bytes(), bytes);
        assert!(!set.contains(&7));
        assert!(set.insert_with_budget(2, &budget).unwrap());
        assert!(set.insert_with_budget(1, &budget).unwrap());
        assert_eq!(set.into_iter().collect::<Vec<_>>(), vec![1, 2]);
    }

    #[test]
    fn refusals_preserve_tree_and_precede_arena_growth() {
        let cancel = AtomicBool::new(false);
        let mut index = FlatIndex::new();
        let bytes = QueryBudget::new(u64::MAX, 0, &cancel);
        assert!(index.insert_with_budget(1, 2, &bytes).is_err());
        assert_eq!(bytes.failure(), Some(QueryLimit::Bytes));
        assert_eq!(index.nodes.capacity(), 0);
        assert_avl(&index);
        for key in [2, 0, 3, 1] {
            index.insert(key, key);
        }
        let before = index.clone().into_iter().collect::<Vec<_>>();
        let work = QueryBudget::new(2, usize::MAX, &cancel);
        assert!(index.insert_with_budget(4, 4, &work).is_err());
        assert_eq!(work.failure(), Some(QueryLimit::Work));
        assert_avl(&index);
        assert_eq!(index.clone().into_iter().collect::<Vec<_>>(), before);
        let cancelled = QueryBudget::new(u64::MAX, usize::MAX, &cancel);
        cancel.store(true, Ordering::Relaxed);
        assert!(index.get_with_budget(&0, &cancelled).is_err());
        let mut entries = index.into_iter();
        assert!(entries.next_with_budget(&cancelled).is_err());
        assert_eq!(entries.collect::<Vec<_>>(), before);
    }

    #[test]
    fn mixed_arrivals_keep_first_encounter_values_when_consumed_in_key_order() {
        let budget = QueryBudget::unbounded();
        let mut index = FlatIndex::new();
        // An odd stride permutes this power-of-two key range. Values model
        // snapshot vertex IDs assigned by first triangle encounter, not key.
        for encounter in 0..512 {
            let key = (encounter * 73) % 512;
            assert_eq!(
                index.insert_with_budget(key, encounter, &budget).unwrap(),
                None
            );
        }
        assert_avl(&index);
        let retained = index.retained_buffer_bytes();
        let mut entries = index.into_iter();
        let capacity = entries.nodes.capacity();
        for key in 0..512 {
            let (actual_key, encounter) = entries.next_with_budget(&budget).unwrap().unwrap();
            assert_eq!(actual_key, key);
            assert_eq!((encounter * 73) % 512, key);
            assert_eq!(entries.nodes.capacity(), capacity);
            assert_eq!(vector_bytes(&entries.nodes), retained);
        }
        assert_eq!(entries.next_with_budget(&budget).unwrap(), None);
    }
}
