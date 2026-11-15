//! The caller's say over a plan's work and memory ([`crate::plan_with`]): every loop of the
//! planner counts its work and asks before it keeps what it makes, so a caller can cancel a plan
//! mid-way or hold it to a budget. A stop refuses the whole plan ([`crate::TessError::Stopped`]);
//! nothing planned before it is returned.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// Why a [`Control`] stopped a plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    /// The caller cancelled it.
    Cancelled,
    /// It counted more work than the caller allows.
    Work,
    /// It would keep more bytes than the caller allows.
    Bytes,
}

/// The caller's control of a plan: [`crate::plan_with`] and [`crate::TessPlan::evaluate_with`]
/// call it from their loops (from the planner's worker threads too, hence `Sync`).
///
/// Work is counted in the planner's own units (an edge node, a grid line, a refinement step, a
/// face, a triangle kept), the same for the same body and tolerance on every run, however the
/// faces are spread over threads. Bytes are the plan's retained output (edge nodes and their
/// parameters, face nodes and triangles), admitted before they are kept; a face's scratch while
/// it is planned is bounded by the planner's own limits ([`crate::PlanStats`] names any it met).
pub trait Control: Sync {
    /// Count `work` more units of work.
    ///
    /// # Errors
    ///
    /// Why the plan must stop: it is refused, never returned partly made.
    fn charge(&self, work: u64) -> Result<(), Stop>;

    /// Admit `bytes` more bytes of retained output, before they are kept.
    ///
    /// # Errors
    ///
    /// Why the plan must stop: it is refused, never returned partly made.
    fn admit(&self, bytes: usize) -> Result<(), Stop>;
}

/// No control: [`crate::plan`]'s, which counts nothing and never stops.
#[derive(Clone, Copy, Debug, Default)]
pub struct Unlimited;

impl Control for Unlimited {
    fn charge(&self, _work: u64) -> Result<(), Stop> {
        Ok(())
    }

    fn admit(&self, _bytes: usize) -> Result<(), Stop> {
        Ok(())
    }
}

/// A plan's control and its first stop: once it has stopped, every loop is told to stop without
/// asking the control again.
pub(crate) struct Gauge<'a> {
    control: &'a dyn Control,
    /// Set once the plan has stopped, read by every loop before it asks the control.
    stopped: AtomicBool,
    /// The first stop.
    stop: Mutex<Option<Stop>>,
}

impl<'a> Gauge<'a> {
    pub(crate) fn new(control: &'a dyn Control) -> Self {
        Gauge {
            control,
            stopped: AtomicBool::new(false),
            stop: Mutex::new(None),
        }
    }

    /// The meter the planner's loops hold.
    pub(crate) fn meter(&self) -> Meter<'_> {
        Meter { gauge: Some(self) }
    }

    /// The first stop, if the plan has stopped.
    pub(crate) fn stopped(&self) -> Option<Stop> {
        *self
            .stop
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn ask(&self, answer: impl FnOnce(&dyn Control) -> Result<(), Stop>) -> bool {
        if self.stopped.load(Ordering::Acquire) {
            return false;
        }
        match answer(self.control) {
            Ok(()) => true,
            Err(stop) => {
                let mut slot = self
                    .stop
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                // The first stop is kept; a worker that asked meanwhile only confirms it.
                slot.get_or_insert(stop);
                self.stopped.store(true, Ordering::Release);
                false
            }
        }
    }
}

/// What the planner's loops hold of the caller's control: `true` from [`Meter::charge`] and
/// [`Meter::admit`] to go on, `false` to stop at once (the plan is then refused whole).
#[derive(Clone, Copy, Default)]
pub(crate) struct Meter<'a> {
    gauge: Option<&'a Gauge<'a>>,
}

impl Meter<'_> {
    /// A meter that counts nothing (the mesh path's, and tests').
    pub(crate) const NONE: Meter<'static> = Meter { gauge: None };

    /// Count `work` units; `false` to stop.
    pub(crate) fn charge(self, work: u64) -> bool {
        self.gauge
            .is_none_or(|g| g.ask(|control| control.charge(work)))
    }

    /// Admit `bytes` of retained output before keeping it; `false` to stop.
    pub(crate) fn admit(self, bytes: usize) -> bool {
        self.gauge
            .is_none_or(|g| g.ask(|control| control.admit(bytes)))
    }

    /// The plan's first stop, if it has stopped.
    pub(crate) fn stop(self) -> Option<Stop> {
        self.gauge.and_then(Gauge::stopped)
    }

    /// Whether the plan has stopped.
    pub(crate) fn stopped(self) -> bool {
        self.gauge
            .is_some_and(|g| g.stopped.load(Ordering::Acquire))
    }
}

impl std::fmt::Debug for Meter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Meter")
            .field("controlled", &self.gauge.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::{Control, Stop};
    use crate::fixtures::{fixture, size_of};
    use crate::{MeshTolerance, TessError, plan, plan_with};
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    /// Counts work and bytes; stops past `work` units or `bytes` bytes.
    struct Counter {
        work: AtomicU64,
        bytes: AtomicUsize,
        calls: AtomicU64,
        max_work: u64,
        max_bytes: usize,
    }

    impl Counter {
        fn new(max_work: u64, max_bytes: usize) -> Counter {
            Counter {
                work: AtomicU64::new(0),
                bytes: AtomicUsize::new(0),
                calls: AtomicU64::new(0),
                max_work,
                max_bytes,
            }
        }
    }

    impl Control for Counter {
        fn charge(&self, work: u64) -> Result<(), Stop> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if self.work.fetch_add(work, Ordering::Relaxed) + work > self.max_work {
                return Err(Stop::Cancelled);
            }
            Ok(())
        }

        fn admit(&self, bytes: usize) -> Result<(), Stop> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if self.bytes.fetch_add(bytes, Ordering::Relaxed) + bytes > self.max_bytes {
                return Err(Stop::Bytes);
            }
            Ok(())
        }
    }

    const BODY: &str = "smlib_thread_run_out";

    fn tolerance() -> MeshTolerance {
        MeshTolerance::display_of(size_of(BODY))
    }

    #[test]
    fn a_controlled_plan_is_the_plan_and_counts_the_same_work_every_run() {
        let b = fixture(BODY);
        let (expected, expected_stats) = plan(&b, &tolerance(), size_of(BODY)).unwrap();
        let counts: Vec<(u64, usize)> = (0..3)
            .map(|_| {
                let counter = Counter::new(u64::MAX, usize::MAX);
                let (planned, stats) =
                    plan_with(&b, &tolerance(), size_of(BODY), &counter).unwrap();
                assert_eq!(planned, expected);
                assert_eq!(stats, expected_stats);
                (counter.work.into_inner(), counter.bytes.into_inner())
            })
            .collect();
        assert!(counts[0].0 > 0 && counts[0].1 > 0, "{counts:?}");
        assert!(counts.iter().all(|c| *c == counts[0]), "{counts:?}");
        // Evaluating the plan counts its nodes and triangles, the same every run, and gives the
        // uncontrolled mesh.
        let evaluated: Vec<u64> = (0..2)
            .map(|_| {
                let counter = Counter::new(u64::MAX, usize::MAX);
                let (mesh, refused) = expected.evaluate_with(&counter).unwrap();
                let (uncontrolled, uncontrolled_refused) = expected.evaluate();
                assert_eq!(refused, uncontrolled_refused);
                assert_eq!(
                    (mesh.positions, mesh.triangles),
                    (uncontrolled.positions, uncontrolled.triangles)
                );
                counter.work.into_inner()
            })
            .collect();
        assert!(
            evaluated[0] > 0 && evaluated[0] == evaluated[1],
            "{evaluated:?}"
        );
    }

    #[test]
    fn a_stop_refuses_a_large_plan_whole_and_its_loops_stop_asking() {
        let b = fixture(BODY);
        let full = Counter::new(u64::MAX, usize::MAX);
        plan_with(&b, &tolerance(), size_of(BODY), &full).unwrap();
        let (total, calls) = (full.work.into_inner(), full.calls.into_inner());
        // Stopped a tenth of the way in: the plan is refused, and the control is asked about a
        // tenth as often (the worker threads in flight may each ask once more).
        let stop = Counter::new(total / 10, usize::MAX);
        let refused = plan_with(&b, &tolerance(), size_of(BODY), &stop);
        assert_eq!(refused, Err(TessError::Stopped(Stop::Cancelled)));
        let asked = stop.calls.into_inner();
        assert!(asked < calls / 4, "asked {asked} of {calls}");
        // Bytes past what the caller allows stop it the same way.
        let bytes = Counter::new(u64::MAX, 1024);
        assert_eq!(
            plan_with(&b, &tolerance(), size_of(BODY), &bytes),
            Err(TessError::Stopped(Stop::Bytes))
        );
        let evaluated = Counter::new(1, usize::MAX);
        let (p, _) = plan(&b, &tolerance(), size_of(BODY)).unwrap();
        assert!(matches!(
            p.evaluate_with(&evaluated),
            Err(TessError::Stopped(Stop::Cancelled))
        ));
    }
}
