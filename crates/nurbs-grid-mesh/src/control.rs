//! Caller control for the existing extraction, grid and deviation algorithms.
//!
//! Controlled entrypoints never replace the geometric algorithm. Every callback
//! refusal is terminal. Admissions count visible owned buffer capacity, including
//! old and new buffers coexisting during growth; they are not an RSS estimate.

use crate::MeshError;

/// Work/cancellation and backing-storage admission supplied by the caller.
///
/// Implementations should keep refusals sticky across callbacks. Inputs moved
/// into a controlled surface constructor remain the caller's existing owners;
/// only newly allocated buffers are admitted by the constructor.
pub trait MeshControl {
    /// Checks cancellation and admits the next amount of algorithmic work.
    fn charge(&self, work: u64) -> Result<(), MeshError>;
    /// Admits additional bytes before allocation and any actual capacity excess
    /// immediately afterwards. Temporary admissions are not reclaimed.
    fn admit(&self, bytes: usize) -> Result<(), MeshError>;
    /// Records a failed or unrepresentable backing-storage allocation.
    fn allocation_failed(&self) -> MeshError;
}

pub(crate) struct Unlimited;
impl MeshControl for Unlimited {
    fn charge(&self, _work: u64) -> Result<(), MeshError> {
        Ok(())
    }
    fn admit(&self, _bytes: usize) -> Result<(), MeshError> {
        Ok(())
    }
    fn allocation_failed(&self) -> MeshError {
        MeshError::TooLarge
    }
}

pub(crate) fn reserve<T>(
    values: &mut Vec<T>,
    additional: usize,
    control: &impl MeshControl,
) -> Result<(), MeshError> {
    control.charge(0)?;
    let needed = values
        .len()
        .checked_add(additional)
        .ok_or_else(|| control.allocation_failed())?;
    if needed <= values.capacity() || size_of::<T>() == 0 {
        return Ok(());
    }
    let target = needed.max(values.capacity().saturating_mul(2)).max(4);
    let bytes = target
        .checked_mul(size_of::<T>())
        .ok_or_else(|| control.allocation_failed())?;
    control.admit(bytes)?;
    values
        .try_reserve_exact(target - values.len())
        .map_err(|_| control.allocation_failed())?;
    let actual = values
        .capacity()
        .checked_mul(size_of::<T>())
        .ok_or_else(|| control.allocation_failed())?;
    if actual > bytes {
        control.admit(actual - bytes)?;
    }
    Ok(())
}

pub(crate) fn zeros(len: usize, control: &impl MeshControl) -> Result<Vec<f64>, MeshError> {
    let mut out = Vec::new();
    reserve(&mut out, len, control)?;
    for _ in 0..len {
        control.charge(1)?;
        out.push(0.0);
    }
    Ok(out)
}
