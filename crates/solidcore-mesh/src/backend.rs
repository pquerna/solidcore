//! Face backends: other code that places a face's interior points.
//!
//! The B-rep layer divides the edges, maps each face's loops into UV (seams, poles and
//! all) and triangulates each face with a constrained Delaunay triangulation whose
//! constraints are the shared edge polylines, so the mesh is watertight by construction.
//! A [`FaceBackend`] may supply the face's interior grid lines (where the Steiner points
//! go) in place of the built-in sampled grid, and may measure the finished face. It never
//! triangulates: the triangulation stays the only one, and refinement still checks every
//! triangle against the surface.
//!
//! With the `nurbs-grid-mesh` feature, [`NurbsGridBackend`] places the lines of dense
//! B-spline faces per knot span with the nurbs-grid-mesh crate.

use std::fmt;

use solidcore::geom::extent2d::Extent2d;
use solidcore::kernel::Kernel;
use solidcore::math::{Point2, Point3};
use solidcore::surface::surface::Surface;
use solidcore::{FaceId, Result};

use crate::budget::QueryBudget;

/// A face offered to a backend.
#[derive(Debug, Clone, Copy)]
pub struct BackendFace<'a> {
    /// The face.
    pub face: FaceId,
    /// Its surface.
    pub surface: &'a Surface,
    /// The UV box of its boundary loops.
    pub bbox: Extent2d,
    /// Chord tolerance (model units).
    pub chord: f64,
    /// Angle tolerance (radians); the built-in grid's lines stay where they are denser.
    pub angle: f64,
}

/// A finished face mesh, as a backend measures it: per-vertex UVs and positions and
/// triangles over them.
#[derive(Debug, Clone, Copy)]
pub struct FaceMeshView<'a> {
    /// Each vertex's (u, v).
    pub uvs: &'a [Point2],
    /// Each vertex's position.
    pub positions: &'a [Point3],
    /// Triangles over the vertices.
    pub triangles: &'a [[u32; 3]],
}

/// Supplies a face's interior grid lines and, optionally, measures its mesh.
pub trait FaceBackend: Send + Sync {
    /// A short name for reports.
    fn name(&self) -> &'static str;

    /// The face's interior grid lines `(us, vs)`, increasing and strictly inside
    /// `face.bbox`; `None` to decline the face (the built-in grid then). Whatever it
    /// returns is bounded like the built-in grid: sorted, clipped to the box and thinned
    /// to [`crate::MAX_GRID_LINES`] per direction and the face's triangle budget.
    /// Implementations must use the supplied caller budget for real work and
    /// visible owned buffers; resource/cancellation errors are terminal, not a
    /// reason to decline. The returned buffers remain admitted by this call.
    ///
    /// # Errors
    ///
    /// What stopped the backend on a face it accepted.
    fn grid_lines(
        &self,
        k: &Kernel,
        face: &BackendFace<'_>,
        budget: &QueryBudget<'_>,
    ) -> Option<Result<(Vec<f64>, Vec<f64>)>>;

    /// The largest distance of the mesh from the face's surface by the backend's own
    /// measure, `None` when it has none for this face.
    /// A diagnostic error, including caller refusal, is propagated rather than
    /// silently discarded.
    fn deviation(
        &self,
        _k: &Kernel,
        _face: &BackendFace<'_>,
        _mesh: &FaceMeshView<'_>,
        _budget: &QueryBudget<'_>,
    ) -> Option<Result<f64>> {
        None
    }
}

impl fmt::Debug for dyn FaceBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FaceBackend({})", self.name())
    }
}

#[cfg(feature = "nurbs-grid-mesh")]
pub use grid::NurbsGridBackend;

#[cfg(feature = "nurbs-grid-mesh")]
mod grid {
    use nurbs_grid_mesh::{
        deviation_with_control, grid_lines_with_control, Dir, MeshControl, MeshError, NurbsSurface,
        TrimmedMesh,
    };
    use solidcore::kernel::Kernel;
    use solidcore::math::Point3;
    use solidcore::{Result, SolidError};

    use super::{BackendFace, FaceBackend, FaceMeshView};
    use crate::budget::QueryBudget;
    use crate::grid::MAX_GRID_LINES;
    use crate::nurbs::canonical_arrays_with_budget;

    const OP: &str = "solidcore_mesh::NurbsGridBackend";

    /// The dependency executes the real algorithm with the caller's sticky
    /// budget; `Stopped` is only a transport marker, never a declined face.
    struct Control<'a, 'b>(&'a QueryBudget<'b>);

    impl MeshControl for Control<'_, '_> {
        fn charge(&self, work: u64) -> std::result::Result<(), MeshError> {
            self.0.charge(work).map_err(|_| MeshError::Stopped)
        }

        fn admit(&self, bytes: usize) -> std::result::Result<(), MeshError> {
            self.0.admit(bytes).map_err(|_| MeshError::Stopped)
        }

        fn allocation_failed(&self) -> MeshError {
            let _ = self.0.allocation_failed();
            MeshError::Stopped
        }
    }

    fn controlled<T>(
        result: std::result::Result<T, MeshError>,
        budget: &QueryBudget<'_>,
    ) -> Result<T> {
        budget.checkpoint()?;
        result.map_err(|e| err(&e))
    }

    /// Snapshot buffers coexist with the source mesh and all extracted surface
    /// coefficients. Reserve before copying and retain every capacity admission.
    fn snapshot(mesh: &FaceMeshView<'_>, budget: &QueryBudget<'_>) -> Result<TrimmedMesh> {
        let mut positions = Vec::new();
        let mut uvs = Vec::new();
        let mut triangles = Vec::new();
        budget.reserve_vec(&mut positions, mesh.positions.len())?;
        budget.reserve_vec(&mut uvs, mesh.uvs.len())?;
        budget.reserve_vec(&mut triangles, mesh.triangles.len())?;
        for p in mesh.positions {
            budget.charge(1)?;
            positions.push([p.x, p.y, p.z]);
        }
        for uv in mesh.uvs {
            budget.charge(1)?;
            uvs.push([uv.x, uv.y]);
        }
        for ids in mesh.triangles {
            budget.charge(1)?;
            triangles.push(*ids);
        }
        Ok(TrimmedMesh {
            positions,
            uvs,
            triangles,
            grid: [0, 0],
            capped_spans: 0,
            refined_lines: 0,
            dropped_degenerate: 0,
        })
    }

    /// Grid lines of dense B-spline faces from nurbs-grid-mesh (per knot span, from the
    /// second-derivative control-point bound), and its Newton-foot deviation: faces whose
    /// surface evaluates through a B-spline with at least [`NurbsGridBackend::min_poles`]
    /// control points.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct NurbsGridBackend {
        /// Fewest control points (u count times v count) of a face it takes.
        pub min_poles: usize,
        /// Most grid lines per knot span (fewer when the spans times this would pass
        /// [`crate::MAX_GRID_LINES`]).
        pub max_lines_per_span: usize,
        /// The lines are placed for this fraction of the chord tolerance (the
        /// triangulation's diagonals sag like the cells, so a margin saves refinement).
        pub chord_fraction: f64,
    }

    impl Default for NurbsGridBackend {
        /// Faces of 1 000 control points or more, 64 lines per span, 0.7 of the chord.
        fn default() -> Self {
            Self {
                min_poles: 1000,
                max_lines_per_span: 64,
                chord_fraction: 0.7,
            }
        }
    }

    fn err(e: &MeshError) -> SolidError {
        let reason = match e {
            MeshError::BadSurface(r) | MeshError::BadOptions(r) => r,
            MeshError::NoLoops => "no boundary loop",
            MeshError::LoopInsideCell => "a loop inside one grid cell",
            MeshError::OpenCell { .. } => "loop pieces in a grid cell did not close",
            MeshError::TooLarge => "too many vertices",
            MeshError::Internal => "nurbs-grid-mesh internal error",
            MeshError::Stopped => "nurbs-grid-mesh control stopped",
        };
        SolidError::Unsupported {
            operation: OP,
            reason,
        }
    }

    impl NurbsGridBackend {
        /// The face's surface as nurbs-grid-mesh's, when it is dense enough and the
        /// arrays evaluate like the kernel's surface, with its knot span counts in u
        /// and v.
        fn surface(
            &self,
            k: &Kernel,
            face: &BackendFace<'_>,
            budget: &QueryBudget<'_>,
        ) -> Result<Option<(NurbsSurface, [usize; 2])>> {
            let Some(c) = canonical_arrays_with_budget(k, face.surface, budget)? else {
                return Ok(None);
            };
            let spans = |kv: &[f64]| -> Result<usize> {
                let mut count = 0;
                for pair in kv.windows(2) {
                    budget.charge(1)?;
                    if pair[1] > pair[0] {
                        count += 1;
                    }
                }
                Ok(count)
            };
            let spans = [spans(&c.uk)?, spans(&c.vk)?];
            if c.nu.saturating_mul(c.nv) < self.min_poles {
                return Ok(None);
            }
            let result = NurbsSurface::new_with_control(
                c.p,
                c.q,
                c.uk,
                c.vk,
                c.nu,
                c.nv,
                c.poles,
                c.weights.as_deref(),
                &Control(budget),
            );
            let ns = match controlled(result, budget) {
                Ok(ns) => ns,
                Err(error) => {
                    budget.checkpoint()?;
                    let _ = error;
                    return Ok(None);
                }
            };
            let b = face.bbox;
            for j in 0..=2 {
                for i in 0..=2 {
                    budget.charge(1)?;
                    let uv = solidcore::math::Point2::new(
                        b.u_min() + (b.u_max() - b.u_min()) * f64::from(i) / 2.0,
                        b.v_min() + (b.v_max() - b.v_min()) * f64::from(j) / 2.0,
                    );
                    let result = ns.evaluate_with_control(uv.x, uv.y, &Control(budget));
                    let evaluated = match controlled(result, budget) {
                        Ok(point) => point,
                        Err(error) => {
                            budget.checkpoint()?;
                            let _ = error;
                            return Ok(None);
                        }
                    };
                    let mine = evaluated.point;
                    let theirs = match face.surface.evaluate_point(k, uv) {
                        Ok(point) => point,
                        Err(_) => {
                            budget.checkpoint()?;
                            return Ok(None);
                        }
                    };
                    budget.checkpoint()?;
                    let d = Point3::new(mine[0], mine[1], mine[2]).distance(theirs);
                    if d > 1e-9 * (1.0 + theirs.length()) {
                        return Ok(None);
                    }
                }
            }
            Ok(Some((ns, spans)))
        }
    }

    impl FaceBackend for NurbsGridBackend {
        fn name(&self) -> &'static str {
            "nurbs-grid-mesh"
        }

        fn grid_lines(
            &self,
            k: &Kernel,
            face: &BackendFace<'_>,
            budget: &QueryBudget<'_>,
        ) -> Option<Result<(Vec<f64>, Vec<f64>)>> {
            let (ns, spans) = match self.surface(k, face, budget) {
                Ok(Some(surface)) => surface,
                Ok(None) => return None,
                Err(error) => return Some(Err(error)),
            };
            let tol = self.chord_fraction * face.chord;
            let b = face.bbox;
            let lines = |dir: Dir, spans: usize, lo: f64, hi: f64| -> Result<Vec<f64>> {
                // lines per span times spans stays within MAX_GRID_LINES
                let per_span = self
                    .max_lines_per_span
                    .min(MAX_GRID_LINES / spans.max(1))
                    .max(1);
                let mut g = controlled(
                    grid_lines_with_control(&ns, dir, tol, hi - lo, per_span, &Control(budget)),
                    budget,
                )?;
                // Filter in place: no second output buffer or hidden collect.
                let mut write = 0;
                for read in 0..g.lines.len() {
                    budget.charge(1)?;
                    let t = g.lines[read];
                    if t > lo && t < hi {
                        g.lines[write] = t;
                        write += 1;
                    }
                }
                g.lines.truncate(write);
                Ok(g.lines)
            };
            Some(
                lines(Dir::U, spans[0], b.u_min(), b.u_max())
                    .and_then(|us| Ok((us, lines(Dir::V, spans[1], b.v_min(), b.v_max())?))),
            )
        }

        fn deviation(
            &self,
            k: &Kernel,
            face: &BackendFace<'_>,
            mesh: &FaceMeshView<'_>,
            budget: &QueryBudget<'_>,
        ) -> Option<Result<f64>> {
            let (ns, _) = match self.surface(k, face, budget) {
                Ok(Some(surface)) => surface,
                Ok(None) => return None,
                Err(error) => return Some(Err(error)),
            };
            Some(snapshot(mesh, budget).and_then(|mesh| {
                controlled(
                    deviation_with_control(&ns, &mesh, face.chord, &Control(budget)),
                    budget,
                )
                .map(|d| d.max)
            }))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::budget::QueryLimit;
        use solidcore::math::Point2;
        use std::sync::atomic::{AtomicBool, Ordering};

        struct CancelDuring<'a, 'b> {
            budget: &'a QueryBudget<'b>,
            cancel: &'a AtomicBool,
            after: u64,
        }

        impl MeshControl for CancelDuring<'_, '_> {
            fn charge(&self, work: u64) -> std::result::Result<(), MeshError> {
                self.budget.charge(work).map_err(|_| MeshError::Stopped)?;
                if self.budget.consumed() >= self.after {
                    self.cancel.store(true, Ordering::Relaxed);
                }
                Ok(())
            }
            fn admit(&self, bytes: usize) -> std::result::Result<(), MeshError> {
                Control(self.budget).admit(bytes)
            }
            fn allocation_failed(&self) -> MeshError {
                Control(self.budget).allocation_failed()
            }
        }

        fn surface() -> NurbsSurface {
            let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
            let poles = (0..3)
                .flat_map(|i| (0..3).map(move |j| [i as f64, j as f64, 0.1 * (i * j) as f64]))
                .collect();
            NurbsSurface::new(2, 2, knots.clone(), knots, 3, 3, poles, None).unwrap()
        }

        #[test]
        fn constructor_admits_new_coefficients_while_inputs_coexist() {
            let cancel = AtomicBool::new(false);
            let u = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
            let v = u.clone();
            let poles = vec![[0.0; 3]; 9];
            let inputs = crate::storage::vector_bytes(&u)
                + crate::storage::vector_bytes(&v)
                + crate::storage::vector_bytes(&poles);
            let budget = QueryBudget::new(u64::MAX, inputs + 287, &cancel);
            budget.admit(inputs).unwrap();
            let result =
                NurbsSurface::new_with_control(2, 2, u, v, 3, 3, poles, None, &Control(&budget));
            assert!(controlled(result, &budget).is_err());
            assert_eq!(budget.failure(), Some(QueryLimit::Bytes));
            assert!(budget.consumed() > 1);
        }

        #[test]
        fn grid_exhaustion_happens_inside_the_real_grid_loop() {
            let s = surface();
            let cancel = AtomicBool::new(false);
            let budget = QueryBudget::new(3, usize::MAX, &cancel);
            let result = grid_lines_with_control(&s, Dir::U, 0.01, 0.1, 64, &Control(&budget));
            assert!(controlled(result, &budget).is_err());
            assert_eq!(budget.failure(), Some(QueryLimit::Work));
            assert_eq!(budget.consumed(), 3);
            assert!(budget.admitted_bytes() > 0);
        }

        #[test]
        fn deviation_stops_inside_newton_basis_work_and_keeps_refusal() {
            let s = surface();
            let mesh = TrimmedMesh {
                positions: vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                triangles: vec![[0, 1, 2]],
                grid: [0, 0],
                capped_spans: 0,
                refined_lines: 0,
                dropped_degenerate: 0,
            };
            let cancel = AtomicBool::new(false);
            let budget = QueryBudget::new(150, usize::MAX, &cancel);
            let result = deviation_with_control(&s, &mesh, 0.01, &Control(&budget));
            assert!(controlled(result, &budget).is_err());
            assert_eq!(budget.failure(), Some(QueryLimit::Work));
            assert_eq!(budget.consumed(), 150);
            assert!(budget.admitted_bytes() > 0);
            assert!(budget.checkpoint().is_err());

            let bytes = QueryBudget::new(u64::MAX, 0, &cancel);
            let result = deviation_with_control(&s, &mesh, 0.01, &Control(&bytes));
            assert!(controlled(result, &bytes).is_err());
            assert_eq!(bytes.failure(), Some(QueryLimit::Bytes));
            // Cancellation is checked in Newton, not only backend dispatch.
            let cancelled = QueryBudget::new(u64::MAX, usize::MAX, &cancel);
            let control = CancelDuring {
                budget: &cancelled,
                cancel: &cancel,
                after: 145,
            };
            let result = deviation_with_control(&s, &mesh, 0.01, &control);
            assert!(controlled(result, &cancelled).is_err());
            assert_eq!(cancelled.consumed(), 145);
            assert!(cancelled.admitted_bytes() > 0);
            assert_eq!(cancelled.failure(), Some(QueryLimit::Cancelled));
        }

        #[test]
        fn snapshot_admission_coexists_with_all_source_capacities() {
            let positions = vec![Point3::new(0.0, 0.0, 0.0); 3];
            let uvs = vec![Point2::new(0.0, 0.0); 3];
            let triangles = vec![[0, 1, 2]];
            let view = FaceMeshView {
                positions: &positions,
                uvs: &uvs,
                triangles: &triangles,
            };
            let source = crate::storage::vector_bytes(&positions)
                + crate::storage::vector_bytes(&uvs)
                + crate::storage::vector_bytes(&triangles);
            let cancel = AtomicBool::new(false);
            let unlimited = QueryBudget::new(u64::MAX, usize::MAX, &cancel);
            unlimited.admit(source).unwrap();
            let copied = snapshot(&view, &unlimited).unwrap();
            let copy_bytes = crate::storage::vector_bytes(&copied.positions)
                + crate::storage::vector_bytes(&copied.uvs)
                + crate::storage::vector_bytes(&copied.triangles);
            assert_eq!(unlimited.admitted_bytes(), source + copy_bytes);
            let bounded = QueryBudget::new(u64::MAX, source + copy_bytes - 1, &cancel);
            bounded.admit(source).unwrap();
            assert!(snapshot(&view, &bounded).is_err());
            assert_eq!(bounded.failure(), Some(QueryLimit::Bytes));
            assert_eq!(positions.len(), 3);
            assert_eq!(triangles, vec![[0, 1, 2]]);
        }
    }
}
