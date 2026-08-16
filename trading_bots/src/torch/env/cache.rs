//! Bounded, insertion-ordered caches for the per-symbol indicator grids.
//!
//! The trading universe is the packed-bar corpus itself, thousands of symbols deep, and PPO
//! draws a fresh ticker on every episode reset. A cache keyed by symbol that never evicts
//! therefore grows without bound: one symbol's momentum, earnings and macro grids run to
//! tens of megabytes at corpus depth, so a long run would eventually pin the whole corpus in
//! indicator form.
//!
//! Every entry is consumed by the episodes currently in flight, so holding a small multiple
//! of the vectorized environment width is enough. Eviction only drops the cache's handle:
//! any environment still using an entry owns its own `Arc` and is unaffected.

use std::borrow::Borrow;
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

/// Entries kept per indicator cache. `PPO_NPROCS` defaults to 16 and each environment holds
/// one symbol at a time, so this leaves several episodes of slack before anything recomputes.
pub(super) const INDICATOR_CACHE_CAPACITY: usize = 64;

pub(super) struct BoundedCache<K, V> {
    capacity: usize,
    order: VecDeque<K>,
    entries: HashMap<K, V>,
}

impl<K: Eq + Hash + Clone, V> BoundedCache<K, V> {
    pub(super) fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "indicator cache capacity must be positive");
        Self {
            capacity,
            order: VecDeque::with_capacity(capacity),
            entries: HashMap::with_capacity(capacity),
        }
    }

    pub(super) fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.entries.get(key)
    }

    /// Insert, evicting the oldest entry once the cache is full. Re-inserting an existing
    /// key refreshes its value and moves it to the back of the eviction queue, so the symbol
    /// an environment just recomputed is the last one to be dropped.
    pub(super) fn insert(&mut self, key: K, value: V) {
        if self.entries.insert(key.clone(), value).is_some() {
            if let Some(position) = self.order.iter().position(|existing| *existing == key) {
                self.order.remove(position);
            }
        }
        self.order.push_back(key);
        while self.order.len() > self.capacity {
            if let Some(evicted) = self.order.pop_front() {
                self.entries.remove(&evicted);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BoundedCache;

    #[test]
    fn eviction_drops_the_oldest_entry_and_keeps_the_rest() {
        let mut cache = BoundedCache::new(2);
        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3);

        assert_eq!(cache.get(&"a"), None, "the oldest entry is evicted");
        assert_eq!(cache.get(&"b"), Some(&2));
        assert_eq!(cache.get(&"c"), Some(&3));
    }

    #[test]
    fn reinserting_refreshes_the_value_and_defers_eviction() {
        let mut cache = BoundedCache::new(2);
        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("a", 10);
        cache.insert("c", 3);

        assert_eq!(cache.get(&"a"), Some(&10), "refreshed entry survives");
        assert_eq!(cache.get(&"b"), None, "the stale entry is evicted instead");
        assert_eq!(cache.get(&"c"), Some(&3));
    }
}
