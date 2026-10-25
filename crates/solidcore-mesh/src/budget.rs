//! Caller-owned cumulative work, cancellation and allocation admission.
//!
//! Byte admission is a conservative backing-storage bound, not RSS: a query
//! keeps actual admissions until it returns, including old/new growth buffers
//! and temporary buffers already dropped. A distinguished provisional dependency
//! reservation is replaced by its actual returned capacity. The next query resets
//! the local ledger; retained caller owners must be excluded from its allowance.
//!
//! The allowance covers capacity-visible mesh/adaptor buffers. Spade's private
//! triangulation storage and native geometry evaluator/cache allocations remain
//! opaque under native hard candidate limits; they are not represented as exact
//! owned bytes here and require a separate host-resource limit/measurement.
//! A Spade output vector is opaque during its dependency call: a provisional
//! minimum reservation is not a capacity upper bound. Its actual capacity is
//! reconciled immediately on return, before processing or adapter growth; it is
//! fully visible/charged from that boundary onward.

use solidcore::{Result, SolidError};
use std::{
    cell::Cell,
    fmt,
    mem::size_of,
    sync::atomic::{AtomicBool, Ordering},
};

/// Why a bounded mesh query stopped before publishing its candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryLimit {
    /// The caller requested cancellation.
    Cancelled,
    /// The cumulative work allowance was exhausted.
    Work,
    /// The current query's backing-storage allowance was exhausted.
    Bytes,
    /// A fallible backing-storage allocation failed.
    Allocation,
}

/// A geometry failure or an explicit caller resource/cancellation refusal.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryError {
    /// Native geometry or topology could not be meshed.
    Geometry(SolidError),
    /// No partial candidate is supplied after a query limit is reached.
    Limit(QueryLimit),
}
impl fmt::Display for QueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Geometry(error) => fmt::Display::fmt(error, f),
            Self::Limit(limit) => write!(f, "bounded native mesh query: {limit:?}"),
        }
    }
}
impl std::error::Error for QueryError {}

/// One caller-owned cumulative allowance, shared across edge and face queries.
///
/// Work and cancellation are sticky. Set the next call's available bytes with
/// [`Self::set_byte_limit`], excluding all already-retained caller/source owners.
/// Each bounded entrypoint starts a new local byte ledger; work never resets.
/// This type is deliberately not shared across concurrently running queries.
#[derive(Debug)]
pub struct QueryBudget<'a> {
    work_limit: u64,
    consumed: Cell<u64>,
    byte_limit: Cell<usize>,
    admitted: Cell<usize>,
    cancel: Option<&'a AtomicBool>,
    failure: Cell<Option<QueryLimit>>,
}

pub(crate) struct DependencyReservation<'b, 'a> {
    budget: &'b QueryBudget<'a>,
    bytes: usize,
}
impl DependencyReservation<'_, '_> {
    pub(crate) fn reconcile(self, actual: usize) -> Result<()> {
        self.budget.reconcile_allocation(self.bytes, actual)
    }
}
impl<'a> QueryBudget<'a> {
    /// Constructs a bounded session. Zero allowances are valid and refuse work
    /// or allocation before it starts.
    #[must_use]
    pub fn new(work_limit: u64, byte_limit: usize, cancel: &'a AtomicBool) -> Self {
        Self {
            work_limit,
            consumed: Cell::new(0),
            byte_limit: Cell::new(byte_limit),
            admitted: Cell::new(0),
            cancel: Some(cancel),
            failure: Cell::new(None),
        }
    }
    /// Work admitted cumulatively, including completed work before a refusal.
    #[must_use]
    pub fn consumed(&self) -> u64 {
        self.consumed.get()
    }
    /// Work still available in this session.
    #[must_use]
    pub fn remaining(&self) -> u64 {
        self.work_limit.saturating_sub(self.consumed.get())
    }
    /// Sets the available bytes for the next bounded entrypoint. Already
    /// retained owners remain the caller's responsibility and are not erased.
    pub fn set_byte_limit(&self, bytes: usize) {
        self.byte_limit.set(bytes);
    }
    /// Conservative cumulative buffer admission in the current/last call.
    #[must_use]
    pub fn admitted_bytes(&self) -> usize {
        self.admitted.get()
    }
    /// The sticky resource refusal, if any. Geometry failures do not set it.
    #[must_use]
    pub fn failure(&self) -> Option<QueryLimit> {
        self.failure.get()
    }

    pub(crate) fn unbounded() -> Self {
        Self {
            work_limit: u64::MAX,
            consumed: Cell::new(0),
            byte_limit: Cell::new(usize::MAX),
            admitted: Cell::new(0),
            cancel: None,
            failure: Cell::new(None),
        }
    }
    pub(crate) fn begin_query(&self) -> Result<()> {
        self.checkpoint()?;
        self.admitted.set(0);
        Ok(())
    }
    /// Checks sticky limits and caller cancellation before native work.
    pub fn checkpoint(&self) -> Result<()> {
        if let Some(limit) = self.failure.get() {
            return Err(Self::native_limit(limit));
        }
        if self
            .cancel
            .is_some_and(|cancel| cancel.load(Ordering::Relaxed))
        {
            return self.stop(QueryLimit::Cancelled);
        }
        Ok(())
    }
    /// Admits work before performing it; cumulative work never resets.
    pub fn charge(&self, amount: u64) -> Result<()> {
        self.checkpoint()?;
        let Some(next) = self.consumed.get().checked_add(amount) else {
            return self.stop(QueryLimit::Work);
        };
        if next > self.work_limit {
            return self.stop(QueryLimit::Work);
        }
        self.consumed.set(next);
        Ok(())
    }
    /// Admits capacity-visible owned backing storage before allocation.
    pub fn admit(&self, bytes: usize) -> Result<()> {
        self.checkpoint()?;
        let Some(next) = self.admitted.get().checked_add(bytes) else {
            return self.stop(QueryLimit::Bytes);
        };
        if next > self.byte_limit.get() {
            return self.stop(QueryLimit::Bytes);
        }
        self.admitted.set(next);
        Ok(())
    }
    /// Fallibly grows a visible buffer, admitting the complete new allocation
    /// while its previous backing storage coexists.
    pub fn reserve_vec<T>(&self, values: &mut Vec<T>, additional: usize) -> Result<()> {
        self.checkpoint()?;
        let Some(needed) = values.len().checked_add(additional) else {
            return self.stop(QueryLimit::Bytes);
        };
        if needed <= values.capacity() || size_of::<T>() == 0 {
            return Ok(());
        }
        let minimum = if size_of::<T>() == 1 {
            8
        } else if size_of::<T>() <= 1024 {
            4
        } else {
            1
        };
        let target = needed.max(values.capacity().saturating_mul(2)).max(minimum);
        let Some(bytes) = target.checked_mul(size_of::<T>()) else {
            return self.stop(QueryLimit::Bytes);
        };
        // Admit the entire new buffer while the old allocation still coexists.
        // No inferred upper bound on allocator-returned capacity is required.
        self.admit(bytes)?;
        if values
            .try_reserve_exact(target.saturating_sub(values.len()))
            .is_err()
        {
            return self.stop(QueryLimit::Allocation);
        }
        let actual = values.capacity().saturating_mul(size_of::<T>());
        if actual > bytes {
            self.admit(actual - bytes)?;
        }
        Ok(())
    }
    /// Pushes one value after fallible, capacity-accounted growth if required.
    pub fn push<T>(&self, values: &mut Vec<T>, value: T) -> Result<()> {
        self.reserve_vec(values, 1)?;
        values.push(value);
        Ok(())
    }
    #[cfg(feature = "nurbs-grid-mesh")]
    pub(crate) fn allocation_failed(&self) -> Result<()> {
        self.stop(QueryLimit::Allocation)
    }

    pub(crate) fn reserve_dependency_output(
        &self,
        bytes: usize,
    ) -> Result<DependencyReservation<'_, 'a>> {
        self.admit(bytes)?;
        Ok(DependencyReservation {
            budget: self,
            bytes,
        })
    }

    /// Only a distinguished provisional reservation can release unused bytes;
    /// actual prior admissions and dropped-buffer admissions are never refunded.
    fn reconcile_allocation(&self, reserved: usize, actual: usize) -> Result<()> {
        self.checkpoint()?;
        let Some(base) = self.admitted.get().checked_sub(reserved) else {
            return Err(SolidError::invalid_input(
                "solidcore_mesh::bounded_query",
                "dependency reservation is absent from the current query ledger",
            ));
        };
        let Some(next) = base.checked_add(actual) else {
            return self.stop(QueryLimit::Bytes);
        };
        if next > self.byte_limit.get() {
            return self.stop(QueryLimit::Bytes);
        }
        self.admitted.set(next);
        Ok(())
    }
    pub(crate) fn finish<T>(&self, result: Result<T>) -> Result<T, QueryError> {
        if let Some(limit) = self.failure.get() {
            return Err(QueryError::Limit(limit));
        }
        result.map_err(QueryError::Geometry)
    }
    fn stop<T>(&self, limit: QueryLimit) -> Result<T> {
        self.failure.set(Some(limit));
        Err(Self::native_limit(limit))
    }
    fn native_limit(limit: QueryLimit) -> SolidError {
        let reason = match limit {
            QueryLimit::Cancelled => "caller cancelled native mesh query",
            QueryLimit::Work => "caller native mesh work allowance exhausted",
            QueryLimit::Bytes => "caller native mesh byte allowance exhausted",
            QueryLimit::Allocation => "native mesh backing-storage allocation failed",
        };
        SolidError::invalid_input("solidcore_mesh::bounded_query", reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dependency_reconciliation_releases_only_its_provisional_bytes() {
        let cancel = AtomicBool::new(false);
        let budget = QueryBudget::new(100, 100, &cancel);
        budget.admit(20).unwrap();
        let reservation = budget.reserve_dependency_output(16).unwrap();
        budget.admit(7).unwrap();
        reservation.reconcile(4).unwrap();
        assert_eq!(budget.admitted_bytes(), 31);
        let reservation = budget.reserve_dependency_output(4).unwrap();
        reservation.reconcile(12).unwrap();
        assert_eq!(budget.admitted_bytes(), 43);
        let reservation = budget.reserve_dependency_output(4).unwrap();
        assert!(reservation.reconcile(101).is_err());
        assert_eq!(budget.failure(), Some(QueryLimit::Bytes));
        assert_eq!(budget.admitted_bytes(), 47);
    }

    #[test]
    fn work_and_cancellation_are_sticky_across_byte_ledgers() {
        let cancel = AtomicBool::new(false);
        let budget = QueryBudget::new(2, 1024, &cancel);
        budget.begin_query().unwrap();
        budget.charge(1).unwrap();
        budget.begin_query().unwrap();
        budget.charge(1).unwrap();
        assert!(budget.charge(1).is_err());
        assert_eq!(budget.failure(), Some(QueryLimit::Work));
        assert_eq!(budget.consumed(), 2);
        let other = QueryBudget::new(2, 1024, &cancel);
        cancel.store(true, Ordering::Relaxed);
        assert!(other.charge(1).is_err());
        assert_eq!(other.failure(), Some(QueryLimit::Cancelled));
    }
    #[test]
    fn allocation_refuses_before_growing_and_counts_retained_empty_capacity() {
        let cancel = AtomicBool::new(false);
        let budget = QueryBudget::new(100, 0, &cancel);
        let mut values = Vec::<u64>::new();
        assert!(budget.reserve_vec(&mut values, 1).is_err());
        assert_eq!(values.capacity(), 0);
        let other = QueryBudget::new(100, 1024, &cancel);
        other.push(&mut values, 1).unwrap();
        let bytes = other.admitted_bytes();
        values.clear();
        assert!(values.capacity() > 0);
        assert_eq!(other.admitted_bytes(), bytes);
    }
}
