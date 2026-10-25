//! One face: its boundary loops in UV from the shared edge polylines, a constrained
//! Delaunay triangulation in scaled UV, grid points and refinement, kept between levels
//! of detail so each level only inserts points.
//!
//! Bounded queries admit adapter-owned Vec/flat-index backing storage before growth.
//! Spade 2.15.1 keeps private DCEL Vecs and transient legalization/constraint buffers
//! (`delaunay_core/{dcel,triangulation_ext,dcel_operations}.rs`, `cdt.rs`), and the
//! SolidCore kernel may allocate during topology traversal, canonical extraction and
//! evaluation. Those opaque dependency allocations remain host-bounded, not measured
//! by the adapter byte limit. Cancellation/work are checked at every native loop and
//! dependency call boundary; a dependency operation itself is not interruptible.

use crate::budget::QueryBudget;
use crate::storage::{FlatIndex, FlatSet};

use spade::handles::{FixedFaceHandle, FixedUndirectedEdgeHandle, FixedVertexHandle, InnerTag};
use spade::{
    ConstrainedDelaunayTriangulation, HasPosition, PositionInTriangulation, Triangulation,
};

use solidcore::curve::curve::Curve;
use solidcore::geom::extent1d::Extent1d;
use solidcore::geom::extent2d::Extent2d;
use solidcore::kernel::Kernel;
use solidcore::math::{Point2, Point3, Vec3};
use solidcore::nurbs::b_spline_surface_n_lib::p2::OrientType;
use solidcore::surface::surface::Surface;
use solidcore::topology::edgeuse::p1 as eu1;
use solidcore::topology::{Brep, EdgeuseId, FaceuseId};
use solidcore::{FaceId, Result};

use crate::backend::{BackendFace, FaceBackend, FaceMeshView};
use crate::edges::{EdgeSet, SegmentSplit, MAX_SPLIT_PIECES};
use crate::geom::{
    angle, closes_with_budget, foot_with_budget, invalid, invert_point_with_budget, power_of_two,
    segment_distance, surface_normal, surface_point, surface_scales_with_budget,
};
use crate::grid::{bounded_with_budget, grid_lines_with_budget, MAX_CELL_ASPECT, SAMPLE_GRID};
use crate::mesh::EdgePolyline;
use crate::nurbs::Nurbs;
use crate::options::FaceParams;

/// A boundary point's trim-curve UV is polished when its surface point is farther than
/// this fraction of the chord tolerance from the edge point.
const POLISH_FRACTION: f64 = 0.1;
/// A refinement point keeps this fraction of its triangle's (or edge's) size from the
/// boundary.
const CLEARANCE_FRACTION: f64 = 0.1;
/// Triangles smaller than this fraction of the chord tolerance are not split.
const MIN_SPLIT_FRACTION: f64 = 1e-3;
/// A triangle whose doubled area is below this fraction of its longest edge squared is
/// a sliver: its normal is not compared with the surface's.
const SLIVER: f64 = 0.02;
/// The B-spline fast path is used when it agrees with the surface's own evaluation
/// within this fraction of the face's size.
const FAST_PATH_AGREEMENT: f64 = 1e-12;
/// A finer level inserts its grid when it has this many times the cells of the last
/// grid inserted.
const REGRID_FACTOR: usize = 4;
/// Refinement points closer than this fraction of the grid step to a point already
/// there are dropped.
const MIN_POINT_GAP: f64 = 1e-6;
/// A triangle across a fold of its surface is split while larger than this many
/// chord tolerances.
const FOLD_SIZE: f64 = 4.0;
/// Most UV samples of one degenerate (pole) segment.
const MAX_POLE_SEGMENTS: u32 = 1024;
/// Most generations of refinement points below a boundary or grid point: a criterion
/// still failing there (a singular point, a kink in the surface) stops refining.
pub const MAX_LOCAL_DEPTH: u8 = 48;
/// Largest loop a face with no area in UV is ear-clipped from, in points.
const MAX_EAR_LOOP: usize = 256;

/// One point of a face's boundary loop in UV.
#[derive(Debug, Clone, Copy)]
struct LoopPoint {
    uv: Point2,
    xyz: Point3,
    vertex: u32,
    /// The segment to the next point collapses to this point's vertex (a pole).
    pole_next: bool,
}

/// A boundary point as inserted: UV, position and mesh vertex.
type BoundaryPoint = (Point2, Point3, u32);

/// A vertex of a face's triangulation.
#[derive(Debug, Clone, Copy)]
struct FaceVertex {
    uv: Point2,
    xyz: Point3,
    /// The shared mesh vertex: a boundary point's from the start, an interior point's
    /// once it is output.
    vertex: Option<u32>,
    /// Unit outward normal, when the surface has one there.
    normal: Option<Vec3>,
    /// On a boundary loop (never removed by refinement).
    boundary: bool,
    /// Still in the triangulation.
    alive: bool,
    /// How far the point is off the surface at its UV: the B-rep's own edge/face gap
    /// for a boundary point, `0` for an interior one. A triangle may be off the surface
    /// by its corners' gap on top of the chord tolerance.
    gap: f64,
    /// Refinement generation: `0` for boundary and grid points, one more than the
    /// deepest corner of the triangle (or edge) a refinement point splits.
    depth: u8,
}

/// The triangulation's vertex: its position in scaled UV and its [`FaceVertex`] index.
#[derive(Debug, Clone, Copy)]
struct CdtVertex {
    pos: spade::Point2<f64>,
    id: usize,
}

impl HasPosition for CdtVertex {
    type Scalar = f64;

    fn position(&self) -> spade::Point2<f64> {
        self.pos
    }
}

/// The native constrained triangulation. Last-used hints avoid hidden hierarchy copies.
/// Spade owns opaque DCEL and scratch buffers: host-bounded, not adapter-byte-accounted.
type Cdt =
    ConstrainedDelaunayTriangulation<CdtVertex, (), (), (), spade::LastUsedVertexHintGenerator>;

/// The mesh of one face, ready to join a [`crate::Mesh`] ([`crate::assemble`]).
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::struct_excessive_bools)] // independent flags
pub struct FaceMesh {
    /// The face.
    pub face: FaceId,
    /// Each local vertex's (u, v).
    pub uvs: Vec<Point2>,
    /// Each local vertex's position (a boundary vertex's is the edge polyline's point).
    pub positions: Vec<Point3>,
    /// Each local vertex's unit outward normal (zero where the surface has none).
    pub normals: Vec<Vec3>,
    /// Each local vertex's shared mesh vertex: boundary vertices have one, interior
    /// vertices get one when assembled.
    pub shared: Vec<Option<u32>>,
    /// Triangles over local vertices, counter-clockwise seen from outside.
    pub triangles: Vec<[u32; 3]>,
    /// Each local vertex's distance from the surface at its UV: the B-rep's own
    /// edge/face gap at a boundary vertex, `0` inside.
    pub gaps: Vec<f64>,
    /// Edge uses mapped into UV by point inversion.
    pub inverted_edgeuses: Vec<EdgeuseId>,
    /// Refinement stopped at a bound with work left.
    pub unconverged: bool,
    /// The boundary crossed itself in UV and was split there.
    pub split_boundary: bool,
    /// The face had no area in UV and was ear-clipped in 3D.
    pub ear_clipped: bool,
    /// Largest measured deviation of the triangles from the surface (when measured).
    pub deviation: Option<f64>,
    /// Triangles whose measured deviation exceeds the chord tolerance plus the largest
    /// gap of their corners.
    pub triangles_over: usize,
    /// The backend that placed the face's grid lines (`None`: the built-in grid).
    pub backend: Option<&'static str>,
    /// A backend took the face and failed; the built-in grid was used.
    pub backend_failed: bool,
    /// The backend's own measure of the face's deviation (when measured).
    pub backend_deviation: Option<f64>,
}

impl FaceMesh {
    /// Bytes of this owner and all output buffers, using actual capacities.
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        use crate::storage::vector_bytes;
        size_of::<Self>()
            .saturating_add(vector_bytes(&self.uvs))
            .saturating_add(vector_bytes(&self.positions))
            .saturating_add(vector_bytes(&self.normals))
            .saturating_add(vector_bytes(&self.shared))
            .saturating_add(vector_bytes(&self.triangles))
            .saturating_add(vector_bytes(&self.gaps))
            .saturating_add(vector_bytes(&self.inverted_edgeuses))
    }
}

/// A face being meshed, kept between levels.
#[allow(clippy::struct_excessive_bools)] // independent state flags
pub(crate) struct FaceState {
    face: FaceId,
    /// The UV the triangulation is centred on.
    origin: Point2,
    /// UV scale factors of the triangulation (powers of two): the speed scaling with
    /// the metric of the grid's cells (see [`FaceState::refine`]).
    scale: (f64, f64),
    /// Powers of two near the mean `|Su|`, `|Sv|`.
    speed: (f64, f64),
    /// `+1` when the outward side is the surface normal's.
    sense: f64,
    /// Triangles counter-clockwise in UV face inward and are wound the other way.
    flip: bool,
    /// The surface's natural domain (UV searches stay in it).
    domain: Extent2d,
    /// Periods of a surface closed in u or v (over its natural domain).
    period: (Option<f64>, Option<f64>),
    nurbs: Option<Nurbs>,
    /// The model size (bounds the fast path's disagreement).
    size: f64,
    /// With [`FaceState::nurbs`] checked against the surface: the sign turning its
    /// `Su x Sv` outward. Points and normals then come from the B-spline directly.
    fast: Option<f64>,
    cdt: Cdt,
    vertices: Vec<FaceVertex>,
    /// The boundary loops as inserted (pole runs expanded).
    boundary: Vec<Vec<(Point2, u32)>>,
    /// Samples of each pole run, by its first point (vertex, u bits, v bits).
    pole_segments: FlatIndex<(u32, u64, u64), u32>,
    inverted: Vec<EdgeuseId>,
    unconverged: bool,
    split_boundary: bool,
    /// Ear-clipped triangles of a face with no area in UV.
    ears: Vec<[usize; 3]>,
    /// Triangles found within the tolerances at the current level, by sorted vertex ids.
    good: FlatSet<[usize; 3]>,
    /// Interior edges found within the tolerances at the current level.
    good_edges: FlatSet<[usize; 2]>,
    /// The tolerances [`FaceState::good`] and [`FaceState::good_edges`] were found
    /// for (a vertex id keeps its position, so they hold until those change).
    good_for: Option<FaceParams>,
    /// Boundary segments too long for a bad triangle on or next to them to be refined
    /// from inside: the edge polylines should be split there.
    requests: Vec<SegmentSplit>,
    /// The grid lines `(us, vs)` of the current level.
    lines: (Vec<f64>, Vec<f64>),
    /// The edges of the face's loops.
    edges_used: FlatSet<solidcore::EdgeId>,
    /// A refinement point was refused at [`MAX_LOCAL_DEPTH`] this level.
    depth_capped: bool,
    /// Cells of the last grid inserted.
    grid_cells: usize,
    /// The backend that placed the face's grid lines at this level.
    backend: Option<&'static str>,
    /// The UV box of the loops at the last refinement.
    bbox: Extent2d,
    /// A backend took the face and failed; the triangulation meshed it instead.
    backend_failed: bool,
}

/// The outward side of face `f`: `+1` when it is the surface normal's side. A solid's
/// outward side faces its void region: the face use whose region is void looks out
/// along its orientation. A sheet (no or both sides void) takes its upward face use's.
pub(crate) fn outward_sense(b: &Brep, f: FaceId) -> Result<f64> {
    let up = b
        .face(f)?
        .faceuse
        .ok_or_else(|| invalid("face without face use"))?;
    let mate = b.faceuse(up)?.mate;
    let sign = |fu: FaceuseId| -> Result<f64> {
        Ok(if b.faceuse(fu)?.orientation == OrientType::Opposite {
            -1.0
        } else {
            1.0
        })
    };
    let void = |fu: FaceuseId| -> Result<Option<bool>> {
        let Some(shell) = b.faceuse(fu)?.shell else {
            return Ok(None);
        };
        let Some(region) = b.shell(shell)?.region else {
            return Ok(None);
        };
        Ok(Some(b.region(region)?.is_void() == 1))
    };
    if let Some(mate) = mate {
        match (void(up)?, void(mate)?) {
            (Some(false), Some(true)) => return sign(mate),
            (Some(true), Some(false)) => return sign(up),
            _ => {}
        }
    }
    sign(up)
}

fn surface_of(b: &Brep, f: FaceId) -> Result<&Surface> {
    b.face(f)?
        .surface
        .as_ref()
        .ok_or_else(|| invalid("face without surface"))
}

fn same_uv(a: Point2, b: Point2) -> bool {
    a.x.to_bits() == b.x.to_bits() && a.y.to_bits() == b.y.to_bits()
}

impl FaceState {
    /// A face not meshed yet.
    pub(crate) fn new(k: &Kernel, b: &Brep, f: FaceId, model_size: f64) -> Result<Self> {
        Self::new_with_budget(k, b, f, model_size, &QueryBudget::unbounded())
    }

    pub(crate) fn refine(
        &mut self,
        k: &Kernel,
        b: &Brep,
        edges: &EdgeSet,
        p: &FaceParams,
        backend: Option<&dyn FaceBackend>,
    ) -> Result<()> {
        self.refine_with_budget(k, b, edges, p, backend, &QueryBudget::unbounded())
    }

    pub(crate) fn snapshot(
        &self,
        k: &Kernel,
        b: &Brep,
        chord: Option<f64>,
        backend: Option<&dyn FaceBackend>,
    ) -> Result<FaceMesh> {
        self.snapshot_with_budget(k, b, chord, backend, &QueryBudget::unbounded())
    }

    pub(crate) fn new_with_budget(
        k: &Kernel,
        b: &Brep,
        f: FaceId,
        model_size: f64,
        budget: &QueryBudget<'_>,
    ) -> Result<Self> {
        budget.checkpoint()?;
        let surface = surface_of(b, f)?;
        let sense = outward_sense(b, f)?;
        let natural = surface.natural_uv_domain();
        let mut domain = b.face(f)?.uv_domain;
        if !(domain.u_min() < domain.u_max() && domain.v_min() < domain.v_max()) {
            domain = natural;
        }
        let scale = surface_scales_with_budget(k, surface, &domain, budget)?;
        let tol = 1e-9 * model_size.max(1e-300);
        let period = (
            closes_with_budget(k, surface, &natural, true, tol, budget)?
                .then(|| natural.u_max() - natural.u_min()),
            closes_with_budget(k, surface, &natural, false, tol, budget)?
                .then(|| natural.v_max() - natural.v_min()),
        );
        Ok(Self {
            face: f,
            origin: domain.mid(),
            scale,
            speed: scale,
            sense,
            flip: false,
            domain: natural,
            period,
            nurbs: Nurbs::from_surface_with_budget(k, surface, budget)?,
            size: model_size,
            fast: None,
            cdt: Cdt::new(),
            vertices: Vec::new(),
            boundary: Vec::new(),
            pole_segments: FlatIndex::new(),
            inverted: Vec::new(),
            unconverged: false,
            split_boundary: false,
            ears: Vec::new(),
            good: FlatSet::new(),
            good_edges: FlatSet::new(),
            good_for: None,
            requests: Vec::new(),
            lines: (Vec::new(), Vec::new()),
            edges_used: FlatSet::new(),
            depth_capped: false,
            grid_cells: 0,
            backend: None,
            backend_failed: false,
            bbox: Extent2d::new(),
        })
    }

    fn admit_retained(&self, budget: &QueryBudget<'_>) -> Result<()> {
        use crate::storage::vector_bytes;
        let mut bytes = vector_bytes(&self.vertices)
            .saturating_add(vector_bytes(&self.boundary))
            .saturating_add(vector_bytes(&self.inverted))
            .saturating_add(vector_bytes(&self.ears))
            .saturating_add(vector_bytes(&self.requests))
            .saturating_add(vector_bytes(&self.lines.0))
            .saturating_add(vector_bytes(&self.lines.1))
            .saturating_add(self.pole_segments.retained_buffer_bytes())
            .saturating_add(self.good.retained_buffer_bytes())
            .saturating_add(self.good_edges.retained_buffer_bytes())
            .saturating_add(self.edges_used.retained_buffer_bytes());
        for lp in &self.boundary {
            budget.charge(1)?;
            bytes = bytes.saturating_add(vector_bytes(lp));
        }
        if let Some(nb) = &self.nurbs {
            bytes = bytes.saturating_add(nb.retained_buffer_bytes());
        }
        budget.admit(bytes)
    }

    fn scaled(&self, uv: Point2) -> spade::Point2<f64> {
        spade::mitigate_underflow(spade::Point2::new(
            (uv.x - self.origin.x) * self.scale.0,
            (uv.y - self.origin.y) * self.scale.1,
        ))
    }

    fn unscaled(&self, p: spade::Point2<f64>) -> Point2 {
        Point2::new(
            self.origin.x + p.x / self.scale.0,
            self.origin.y + p.y / self.scale.1,
        )
    }

    /// The surface point at `uv` (from the B-spline directly when checked).
    fn point(
        &self,
        k: &Kernel,
        s: &Surface,
        uv: Point2,
        budget: &QueryBudget<'_>,
    ) -> Result<Point3> {
        budget.charge(1)?;
        if let (Some(nb), Some(_)) = (&self.nurbs, self.fast) {
            if let Some((p, _, _)) = nb
                .eval_with_budget(uv.x, uv.y, budget)?
                .filter(|e| crate::geom::finite3(e.0))
            {
                return Ok(p);
            }
        }
        surface_point(k, s, uv)
    }

    /// Checks the B-spline fast path against the surface's own evaluation at a sample
    /// grid of `bbox` and turns it on when they agree (and `Su x Sv` gives the normal's
    /// side).
    fn check_fast_path(
        &mut self,
        k: &Kernel,
        s: &Surface,
        bbox: &Extent2d,
        size: f64,
        budget: &QueryBudget<'_>,
    ) -> Result<()> {
        self.fast = None;
        let Some(nb) = &self.nurbs else {
            return Ok(());
        };
        for j in 0..=4 {
            budget.charge(1)?;
            for i in 0..=4 {
                budget.charge(1)?;
                let uv = Point2::new(
                    bbox.u_min() + (bbox.u_max() - bbox.u_min()) * f64::from(i) / 4.0,
                    bbox.v_min() + (bbox.v_max() - bbox.v_min()) * f64::from(j) / 4.0,
                );
                let (Some((p, _, _)), Ok(q)) = (
                    nb.eval_with_budget(uv.x, uv.y, budget)?,
                    s.evaluate_point(k, uv),
                ) else {
                    return Ok(());
                };
                let d = p.distance(q);
                if d > FAST_PATH_AGREEMENT * size || d.is_nan() {
                    return Ok(());
                }
            }
        }
        self.fast = self.cross_sign(k, s, budget)?.map(|c| c * self.sense);
        Ok(())
    }

    fn outward(
        &self,
        k: &Kernel,
        s: &Surface,
        uv: Point2,
        budget: &QueryBudget<'_>,
    ) -> Result<Option<Vec3>> {
        budget.charge(1)?;
        if let (Some(nb), Some(sign)) = (&self.nurbs, self.fast) {
            if let Some((_, su, sv)) = nb.eval_with_budget(uv.x, uv.y, budget)? {
                if let Some(n) = unit_cross(su, sv) {
                    return Ok(Some(n * sign));
                }
            }
        }
        Ok(surface_normal(k, s, uv).map(|n| n * self.sense))
    }

    fn vertex(&self, h: FixedVertexHandle) -> Result<&FaceVertex> {
        let id = self.cdt.vertex(h).data().id;
        self.vertices
            .get(id)
            .ok_or_else(|| invalid("triangulation vertex out of range"))
    }

    fn id(&self, h: FixedVertexHandle) -> usize {
        self.cdt.vertex(h).data().id
    }

    /// Adds `v` to the vertex table and the triangulation (its position must be free).
    fn insert(&mut self, v: FaceVertex, budget: &QueryBudget<'_>) -> Result<FixedVertexHandle> {
        let id = self.vertices.len();
        budget.push(&mut self.vertices, v)?;
        self.insert_existing(id, budget)
    }

    /// Puts vertex `id` of the table into the triangulation.
    fn insert_existing(
        &mut self,
        id: usize,
        budget: &QueryBudget<'_>,
    ) -> Result<FixedVertexHandle> {
        budget.charge(1)?;
        let uv = self
            .vertices
            .get(id)
            .ok_or_else(|| invalid("vertex out of range"))?
            .uv;
        let pos = self.scaled(uv);
        let h = self
            .cdt
            .insert(CdtVertex { pos, id })
            .map_err(|_| invalid("UV point cannot be triangulated"))?;
        if let Some(v) = self.vertices.get_mut(id) {
            v.alive = true;
        }
        Ok(h)
    }

    /// Inserts an interior point at `uv` with its surface point and normal.
    fn insert_interior(
        &mut self,
        k: &Kernel,
        s: &Surface,
        uv: Point2,
        xyz: Option<(Point3, Option<Vec3>)>,
        depth: u8,
        budget: &QueryBudget<'_>,
    ) -> Result<FixedVertexHandle> {
        let (xyz, normal) = match xyz {
            Some(x) => x,
            None => (
                self.point(k, s, uv, budget)?,
                self.outward(k, s, uv, budget)?,
            ),
        };
        self.insert(
            FaceVertex {
                uv,
                xyz,
                vertex: None,
                normal,
                boundary: false,
                alive: true,
                gap: 0.0,
                depth,
            },
            budget,
        )
    }

    /// The handle of the boundary point at `uv` on mesh vertex `vertex`, inserting it
    /// when it is new.
    fn boundary_handle(
        &mut self,
        k: &Kernel,
        s: &Surface,
        uv: Point2,
        xyz: Point3,
        vertex: u32,
        budget: &QueryBudget<'_>,
    ) -> Result<FixedVertexHandle> {
        if let PositionInTriangulation::OnVertex(h) = self.cdt.locate(self.scaled(uv)) {
            let v = self.vertex(h)?;
            return if v.vertex == Some(vertex) && v.boundary {
                Ok(h)
            } else {
                Err(invalid("two boundary points at one UV position"))
            };
        }
        let normal = self.outward(k, s, uv, budget)?;
        let gap = surface_gap(k, s, uv, xyz, budget)?;
        self.insert(
            FaceVertex {
                uv,
                xyz,
                vertex: Some(vertex),
                normal,
                boundary: true,
                alive: true,
                gap,
                depth: 0,
            },
            budget,
        )
    }

    /// Joins two boundary vertices by a constraint; where it would cross a constraint
    /// already there, both are split at the crossing (the face then is closed there only
    /// geometrically).
    fn constrain(
        &mut self,
        k: &Kernel,
        s: &Surface,
        a: FixedVertexHandle,
        c: FixedVertexHandle,
        budget: &QueryBudget<'_>,
    ) -> Result<()> {
        budget.charge(1)?;
        if a == c || self.cdt.exists_constraint(a, c) {
            return Ok(());
        }
        let added = constraint_edges_with_budget(&mut self.cdt, budget, |cdt| {
            cdt.try_add_constraint(a, c)
        })?;
        if !added.is_empty() || self.cdt.exists_constraint(a, c) {
            return Ok(());
        }
        drop(added);
        let edges = constraint_edges_with_budget(&mut self.cdt, budget, |cdt| {
            cdt.add_constraint_and_split(a, c, |pos| CdtVertex {
                pos,
                id: usize::MAX,
            })
        })?;
        if edges.is_empty() {
            return Err(invalid("boundary constraint cannot be added"));
        }
        self.split_boundary = true;
        // the split points become boundary vertices on the surface
        for e in edges {
            budget.charge(1)?;
            let to = self.cdt.directed_edge(e).to().fix();
            if self.cdt.vertex(to).data().id == usize::MAX {
                let pos = self.cdt.vertex(to).position();
                let uv = self.unscaled(pos);
                let xyz = surface_point(k, s, uv)?;
                let normal = self.outward(k, s, uv, budget)?;
                let id = self.vertices.len();
                budget.push(
                    &mut self.vertices,
                    FaceVertex {
                        uv,
                        xyz,
                        vertex: None,
                        normal,
                        boundary: true,
                        alive: true,
                        gap: 0.0,
                        depth: 0,
                    },
                )?;
                self.cdt.vertex_data_mut(to).id = id;
            }
        }
        Ok(())
    }

    /// Meshes (or, after an earlier level, refines) the face to `p` against the edge
    /// polylines of `edges`. Only inserts points into what an earlier level made, except
    /// interior points the refined boundary leaves outside or against it.
    pub(crate) fn refine_with_budget(
        &mut self,
        k: &Kernel,
        b: &Brep,
        edges: &EdgeSet,
        p: &FaceParams,
        backend: Option<&dyn FaceBackend>,
        budget: &QueryBudget<'_>,
    ) -> Result<()> {
        self.admit_retained(budget)?;
        let s = surface_of(b, self.face)?;
        if self.good_for != Some(*p) {
            self.good.clear();
            self.good_edges.clear();
            self.good_for = Some(*p);
        }
        self.unconverged = false;
        self.depth_capped = false;
        self.backend_failed = false;
        self.requests.clear();
        budget.checkpoint()?;
        let loops = self.face_loops(k, b, s, edges, p, budget)?;
        if loops.is_empty() {
            return Err(invalid("face without boundary loops"));
        }
        let mut bbox = Extent2d::new();
        for q in loops.iter().flatten() {
            budget.charge(1)?;
            bbox.add_point(q.uv);
        }
        if !(bbox.u_min() <= bbox.u_max() && bbox.v_min() <= bbox.v_max()) {
            return Err(invalid("face loops have no UV points"));
        }
        let first = self.boundary.is_empty();
        let (mut us, mut vs) = if bbox.u_min() < bbox.u_max() && bbox.v_min() < bbox.v_max() {
            if first {
                // the B-spline fast path, before anything is evaluated
                self.check_fast_path(k, s, &bbox, self.size, budget)?;
            }
            // a B-spline (on the fast path) is sampled once per knot span at least
            budget.charge(self.nurbs.as_ref().map_or(0, |nb| nb.knot_count()) as u64)?;
            let samples =
                self.nurbs
                    .as_ref()
                    .filter(|_| self.fast.is_some())
                    .map_or(SAMPLE_GRID, |nb| {
                        nb.spans_in(true, bbox.u_min(), bbox.u_max())
                            .max(nb.spans_in(false, bbox.v_min(), bbox.v_max()))
                    });
            let eval = |uv: Point2| {
                Ok((
                    self.point(k, s, uv, budget)?,
                    self.outward(k, s, uv, budget)?,
                ))
            };
            grid_lines_with_budget(&eval, &bbox, self.speed, p, samples, budget)?
        } else {
            (Vec::new(), Vec::new())
        };
        self.bbox = bbox;
        // a backend's lines, where denser than the sampled ones (those keep the angle
        // tolerance)
        self.backend = None;
        if let Some(be) =
            backend.filter(|_| bbox.u_min() < bbox.u_max() && bbox.v_min() < bbox.v_max())
        {
            let face = BackendFace {
                face: self.face,
                surface: s,
                bbox,
                chord: p.chord,
                angle: p.angle,
            };
            budget.charge(1)?;
            match be.grid_lines(k, &face, budget) {
                Some(Ok((bu, bv))) => {
                    budget.checkpoint()?;
                    self.backend = Some(be.name());
                    if bu.len() > us.len() {
                        us = bu;
                    }
                    if bv.len() > vs.len() {
                        vs = bv;
                    }
                }
                // the sampled grid for a face the backend failed on
                Some(Err(_)) => {
                    budget.checkpoint()?;
                    self.backend_failed = true;
                }
                None => {}
            }
        }
        // whichever source they came from, the lines are bounded like the built-in ones
        let (us, vs) = bounded_with_budget(us, vs, &bbox, p.max_triangles, budget)?;
        let step =
            |lines: &[f64], lo: f64, hi: f64, sc: f64| (hi - lo) * sc / (lines.len() + 1) as f64;
        let has_pole = loops.iter().flatten().any(|q| q.pole_next);
        let mut metric = self.speed;
        if !has_pole {
            // the triangulation's metric: UV scaled so the grid cells the tolerances
            // ask for are about square (at most MAX_CELL_ASPECT to one), so Delaunay
            // triangles are long where the surface is straight (along a ruling) and
            // refinement splits across it. A face with a pole keeps the speed scaling
            // (near a pole the grid's cells say nothing about the surface).
            let hu = step(&us, bbox.u_min(), bbox.u_max(), self.speed.0);
            let hv = step(&vs, bbox.v_min(), bbox.v_max(), self.speed.1);
            let h = hu.min(hv);
            if h > 0.0 && h.is_finite() {
                let (hu, hv) = (hu.min(MAX_CELL_ASPECT * h), hv.min(MAX_CELL_ASPECT * h));
                metric = (
                    power_of_two(self.speed.0 * h / hu),
                    power_of_two(self.speed.1 * h / hv),
                );
            }
        }
        // a finer level whose cells have another shape triangulates its points (all
        // kept) again in the new metric
        let ratio = |m: (f64, f64)| m.0 / m.1;
        let remetric = !first && {
            let r = ratio(metric) / ratio(self.scale);
            !(0.5..=2.0).contains(&r)
        };
        if first || remetric {
            self.scale = metric;
        }
        let scaled_step = step(&us, bbox.u_min(), bbox.u_max(), self.scale.0).min(step(
            &vs,
            bbox.v_min(),
            bbox.v_max(),
            self.scale.1,
        ));

        // the boundary, pole runs expanded
        let expanded = self.expand_poles(&loops, scaled_step, budget)?;

        if first || remetric || !self.update_boundary(k, s, &loops, &expanded, budget)? {
            self.rebuild(k, s, &loops, &expanded, budget)?;
        }
        let mut boundary = Vec::new();
        budget.reserve_vec(&mut boundary, expanded.len())?;
        for lp in &expanded {
            budget.push(
                &mut boundary,
                collect_with_budget(lp.iter().map(|q| (q.0, q.2)), budget)?,
            )?;
        }
        self.boundary = boundary;
        self.flip = self.uv_ccw_faces_inward(k, s, &bbox, budget)?;
        if !first {
            self.remove_stale_interior(budget)?;
        }

        // grid points, away from the boundary and the points already there; a finer
        // level adds its grid only when it is much denser than the last one inserted
        // (else it refines the earlier triangles: a grid on top of a similar one would
        // only crowd it)
        let cells = (us.len() + 1) * (vs.len() + 1);
        if first || cells >= REGRID_FACTOR * self.grid_cells {
            self.insert_grid(k, s, &us, &vs, scaled_step, p.max_triangles, budget)?;
            self.grid_cells = cells;
        }
        self.lines = (us, vs);

        // refinement
        let mut converged = false;
        let mut changed: Option<Vec<FixedVertexHandle>> = None;
        for _ in 0..p.max_passes {
            budget.charge(1)?;
            if self.cdt.num_inner_faces() >= p.max_triangles {
                break;
            }
            let candidates = self.refine_candidates(k, s, p, changed.as_deref(), budget)?;
            if candidates.is_empty() {
                converged = true;
                break;
            }
            let mut inserted = Vec::new();
            budget.reserve_vec(&mut inserted, candidates.len())?;
            for (uv, depth) in candidates {
                budget.charge(1)?;
                if self.cdt.num_inner_faces() >= p.max_triangles {
                    break;
                }
                let pos = self.scaled(uv);
                let near = match self.cdt.locate(pos) {
                    PositionInTriangulation::OnFace(f) => self.cdt.face(f).vertices().map(Some),
                    PositionInTriangulation::OnEdge(e)
                        if !self.cdt.is_constraint_edge(e.as_undirected()) =>
                    {
                        let [a, b] = self.cdt.directed_edge(e).vertices();
                        [Some(a), Some(b), None]
                    }
                    _ => continue,
                };
                // a point (all but) on one already there (two candidates of one pass
                // can meet) would make a sliver
                let gap = MIN_POINT_GAP * scaled_step;
                if near.iter().flatten().any(|v| {
                    let q = v.position();
                    (q.x - pos.x).abs() <= gap && (q.y - pos.y).abs() <= gap
                }) {
                    continue;
                }
                budget.push(
                    &mut inserted,
                    self.insert_interior(k, s, uv, None, depth, budget)?,
                )?;
            }
            let added = inserted.len();
            changed = Some(inserted);
            if added == 0 {
                converged = true;
                break;
            }
        }
        self.unconverged = !converged || self.depth_capped;

        // a face with no area in UV: clip ears off its one loop in 3D
        self.ears.clear();
        let inside = inside_faces(&self.cdt, budget)?;
        budget.charge(self.cdt.num_inner_faces() as u64)?;
        if !self.cdt.inner_faces().any(|f| is_inside(&inside, f.fix())) {
            self.ear_clip(k, s, budget)?;
        }
        Ok(())
    }

    /// The boundary loops of the face in UV, outer first, each closed implicitly.
    fn face_loops(
        &mut self,
        k: &Kernel,
        b: &Brep,
        s: &Surface,
        edges: &EdgeSet,
        p: &FaceParams,
        budget: &QueryBudget<'_>,
    ) -> Result<Vec<Vec<LoopPoint>>> {
        let domain = self.domain;
        let scale = self.scale;
        let span = ((domain.u_max() - domain.u_min()) * scale.0)
            .max((domain.v_max() - domain.v_min()) * scale.1);
        let polish_above = POLISH_FRACTION * p.chord;
        let fu = b
            .face(self.face)?
            .faceuse
            .ok_or_else(|| invalid("face without face use"))?;
        let mut loops = Vec::new();
        let mut inverted = Vec::new();
        let loopuses = eu1::faceuse_loopuses(b, fu, false, None)?;
        budget.admit(crate::storage::vector_bytes(&loopuses))?;
        budget.reserve_vec(&mut loops, loopuses.len())?;
        for lu in loopuses {
            budget.charge(1)?;
            let mut pts: Vec<LoopPoint> = Vec::new();
            let eus = eu1::loopuse_edgeuses(b, lu)?;
            budget.admit(crate::storage::vector_bytes(&eus))?;
            // start at an edge use with a UV trim curve, so edge uses mapped by point
            // inversion continue from a known side of any seam
            let mut start = 0;
            for (i, eu) in eus.iter().enumerate() {
                budget.charge(1)?;
                if eu1::uv_trim_curve(b, *eu)?.0.is_some() {
                    start = i;
                    break;
                }
            }
            for eu in eus.iter().cycle().skip(start).take(eus.len()) {
                budget.charge(1)?;
                let eu = *eu;
                let e = eu1::edge(b, eu)?.ok_or_else(|| invalid("edge use without edge"))?;
                self.edges_used.insert_with_budget(e, budget)?;
                let poly = edges
                    .polyline_with_budget(e, budget)?
                    .ok_or_else(|| invalid("face uses an edge that failed"))?;
                let ivl = b.edge(e)?.interval;
                let trim = eu1::uv_trim_curve(b, eu)?.0;
                let reverse = eu1::orientation(b, eu)? == OrientType::Opposite;
                let seed = pts.last().map(|q| q.uv);
                let ctx = MapContext {
                    budget,
                    k,
                    s,
                    domain: &domain,
                    edges,
                    polish_above,
                    period: self.period,
                };
                let mapped = match trim {
                    Some(c) => ctx.edgeuse_points(poly, &ivl, Some(c), reverse, seed).ok(),
                    None => None,
                };
                budget.checkpoint()?;
                let seq = if let Some(seq) = mapped {
                    seq
                } else {
                    budget.push(&mut inverted, eu)?;
                    ctx.edgeuse_points(poly, &ivl, None, reverse, seed)?
                };
                join(&mut pts, seq, k, s, p.chord, span, scale, budget)?;
            }
            // close the loop
            if pts.len() > 1 {
                if let (Some(first), Some(last)) = (pts.first().copied(), pts.last().copied()) {
                    if first.vertex != last.vertex {
                        return Err(invalid("loop does not close"));
                    }
                    if junction_is_pole(k, s, p.chord, last, first, span, scale, budget)? {
                        if let Some(l) = pts.last_mut() {
                            l.pole_next = true;
                        }
                    } else {
                        pts.pop();
                    }
                }
            }
            if pts.len() >= 2 {
                budget.push(&mut loops, pts)?;
            }
        }
        for eu in inverted {
            budget.charge(1)?;
            budget.charge(self.inverted.len() as u64)?;
            if !self.inverted.contains(&eu) {
                budget.push(&mut self.inverted, eu)?;
            }
        }
        Ok(loops)
    }

    /// The loops with every pole run sampled at about the grid spacing (a power of two
    /// of samples, never fewer than an earlier level's, so levels nest): `(uv, xyz,
    /// vertex)` per point.
    fn expand_poles(
        &mut self,
        loops: &[Vec<LoopPoint>],
        scaled_step: f64,
        budget: &QueryBudget<'_>,
    ) -> Result<Vec<Vec<(Point2, Point3, u32)>>> {
        let mut out = Vec::new();
        budget.reserve_vec(&mut out, loops.len())?;
        for lp in loops {
            budget.charge(1)?;
            let count = lp.len();
            let mut run = Vec::new();
            budget.reserve_vec(&mut run, count)?;
            for (i, q) in lp.iter().enumerate() {
                budget.charge(1)?;
                budget.push(&mut run, (q.uv, q.xyz, q.vertex))?;
                if !q.pole_next {
                    continue;
                }
                let Some(next) = lp.get((i + 1) % count) else {
                    continue;
                };
                let (a, c) = (self.scaled(q.uv), self.scaled(next.uv));
                let len = ((c.x - a.x).powi(2) + (c.y - a.y).powi(2)).sqrt();
                let need = if scaled_step > 0.0 && scaled_step.is_finite() {
                    ((len / scaled_step).ceil() as u32).clamp(1, MAX_POLE_SEGMENTS)
                } else {
                    1
                };
                let key = (q.vertex, q.uv.x.to_bits(), q.uv.y.to_bits());
                let segs = need.next_power_of_two().max(
                    self.pole_segments
                        .get_with_budget(&key, budget)?
                        .copied()
                        .unwrap_or(1),
                );
                self.pole_segments.insert_with_budget(key, segs, budget)?;
                for j in 1..segs {
                    budget.charge(1)?;
                    let w = f64::from(j) / f64::from(segs);
                    budget.push(
                        &mut run,
                        (
                            Point2::new(
                                q.uv.x + (next.uv.x - q.uv.x) * w,
                                q.uv.y + (next.uv.y - q.uv.y) * w,
                            ),
                            q.xyz,
                            q.vertex,
                        ),
                    )?;
                }
            }
            budget.push(&mut out, run)?;
        }
        Ok(out)
    }

    /// A new triangulation of the boundary; interior points of earlier levels that are
    /// still clear of it are put back.
    fn rebuild(
        &mut self,
        k: &Kernel,
        s: &Surface,
        _loops: &[Vec<LoopPoint>],
        expanded: &[Vec<(Point2, Point3, u32)>],
        budget: &QueryBudget<'_>,
    ) -> Result<()> {
        let old: Vec<usize> = collect_with_budget(
            (0..self.vertices.len()).filter(|i| {
                self.vertices
                    .get(*i)
                    .is_some_and(|v| v.alive && !v.boundary)
            }),
            budget,
        )?;
        self.cdt = Cdt::new();
        for v in &mut self.vertices {
            budget.charge(1)?;
            v.alive = false;
        }
        self.split_boundary = false;
        // every boundary point once (one UV position holds one mesh vertex), and the
        // loop segments between them; bulk loading is fast and stable
        let mut index: FlatIndex<(u64, u64), usize> = FlatIndex::new();
        let mut points: Vec<CdtVertex> = Vec::new();
        let mut segments: Vec<[usize; 2]> = Vec::new();
        for lp in expanded {
            budget.charge(1)?;
            let mut ids: Vec<usize> = Vec::new();
            budget.reserve_vec(&mut ids, lp.len())?;
            for (uv, xyz, vertex) in lp {
                budget.charge(1)?;
                let key = (uv.x.to_bits(), uv.y.to_bits());
                let i = if let Some(i) = index.get_with_budget(&key, budget)? {
                    let id = points.get(*i).map_or(usize::MAX, |p| p.id);
                    if self.vertices.get(id).and_then(|v| v.vertex) != Some(*vertex) {
                        return Err(invalid("two boundary points at one UV position"));
                    }
                    *i
                } else {
                    let normal = self.outward(k, s, *uv, budget)?;
                    let id = self.vertices.len();
                    budget.push(
                        &mut self.vertices,
                        FaceVertex {
                            uv: *uv,
                            xyz: *xyz,
                            vertex: Some(*vertex),
                            normal,
                            boundary: true,
                            alive: true,
                            gap: surface_gap(k, s, *uv, *xyz, budget)?,
                            depth: 0,
                        },
                    )?;
                    let pos = self.scaled(*uv);
                    spade::validate_vertex(&CdtVertex { pos, id })
                        .map_err(|_| invalid("UV point cannot be triangulated"))?;
                    index.insert_with_budget(key, points.len(), budget)?;
                    budget.push(&mut points, CdtVertex { pos, id })?;
                    points.len() - 1
                };
                if ids.last() != Some(&i) {
                    budget.push(&mut ids, i)?;
                }
            }
            if ids.len() > 1 && ids.first() == ids.last() {
                ids.pop();
            }
            let n = ids.len();
            for i in 0..n {
                budget.charge(1)?;
                if let (Some(a), Some(c)) = (ids.get(i), ids.get((i + 1) % n)) {
                    if a != c {
                        budget.push(&mut segments, [*a, *c])?;
                    }
                }
            }
        }
        let ids: Vec<usize> = collect_with_budget(points.iter().map(|p| p.id), budget)?;
        let mut conflicts = Vec::new();
        // Spade 2.15.1 bulk_load.rs:261-275 pops each notified input segment;
        // skipped segments are deduplicated and visited once at 364-368.
        // Thus the callback cannot append more than the input segment count.
        budget.reserve_vec(&mut conflicts, segments.len())?;
        budget.charge(segments.len() as u64)?;
        self.cdt = Cdt::try_bulk_load_cdt(points, segments, |e| conflicts.push(e))
            .map_err(|_| invalid("UV boundary cannot be triangulated"))?;
        // a segment crossing another is added with splits
        for [a, c] in conflicts {
            budget.charge(1)?;
            let (Some(ia), Some(ic)) = (ids.get(a), ids.get(c)) else {
                continue;
            };
            let (Some(ha), Some(hc)) = (self.handle_of(*ia), self.handle_of(*ic)) else {
                continue;
            };
            self.constrain(k, s, ha, hc, budget)?;
        }
        if !old.is_empty() {
            // which points are inside, all decided on the boundary's triangulation (an
            // insertion makes faces the flood fill has not seen)
            let inside = inside_faces(&self.cdt, budget)?;
            let keep: Vec<usize> = collect_with_budget(
                old.into_iter().filter(|id| {
                    self.vertices
                        .get(*id)
                        .is_some_and(|v| self.clear_of_boundary(&inside, v.uv, 0.0))
                }),
                budget,
            )?;
            for id in keep {
                budget.charge(1)?;
                self.insert_existing(id, budget)?;
            }
        }
        Ok(())
    }

    /// Inserts the points the edges gained since the last level into the boundary,
    /// replacing each constraint they split. `false` when the boundary changed other
    /// than by insertion (the caller rebuilds).
    fn update_boundary(
        &mut self,
        k: &Kernel,
        s: &Surface,
        _loops: &[Vec<LoopPoint>],
        expanded: &[Vec<(Point2, Point3, u32)>],
        budget: &QueryBudget<'_>,
    ) -> Result<bool> {
        if expanded.len() != self.boundary.len() {
            return Ok(false);
        }
        // per loop: for every old segment, the new points inside it
        let mut plans: Vec<Vec<(usize, Vec<BoundaryPoint>)>> = Vec::new();
        for (old, new) in self.boundary.iter().zip(expanded) {
            budget.charge(1)?;
            let mut plan = Vec::new();
            let mut j = 0;
            let mut pending: Vec<BoundaryPoint> = Vec::new();
            for (i, (ouv, ov)) in old.iter().enumerate() {
                budget.charge(1)?;
                loop {
                    budget.charge(1)?;
                    let Some(q) = new.get(j) else {
                        return Ok(false);
                    };
                    j += 1;
                    if q.2 == *ov && same_uv(q.0, *ouv) {
                        break;
                    }
                    if i == 0 {
                        return Ok(false);
                    }
                    budget.push(&mut pending, *q)?;
                }
                if i > 0 && !pending.is_empty() {
                    budget.push(&mut plan, (i - 1, std::mem::take(&mut pending)))?;
                }
            }
            // after the last old point: inside the closing segment
            budget.reserve_vec(&mut pending, new.len().saturating_sub(j))?;
            pending.extend(new.iter().skip(j).copied());
            if !pending.is_empty() {
                budget.push(&mut plan, (old.len().saturating_sub(1), pending))?;
            }
            budget.push(&mut plans, plan)?;
        }
        for (li, plan) in plans.into_iter().enumerate() {
            budget.charge(1)?;
            for (si, points) in plan {
                budget.charge(1)?;
                let Some(old) = self.boundary.get(li) else {
                    return Ok(false);
                };
                let n = old.len();
                let (Some(&(auv, av)), Some(&(cuv, cv))) = (old.get(si), old.get((si + 1) % n))
                else {
                    return Ok(false);
                };
                let ha = self.find_boundary(auv, av)?;
                let hc = self.find_boundary(cuv, cv)?;
                let (Some(ha), Some(hc)) = (ha, hc) else {
                    return Ok(false);
                };
                let Some(e) = self
                    .cdt
                    .get_edge_from_neighbors(ha, hc)
                    .map(|e| e.fix().as_undirected())
                else {
                    return Ok(false);
                };
                if !self.cdt.is_constraint_edge(e) {
                    return Ok(false);
                }
                self.cdt.remove_constraint_edge(e);
                let mut chain = Vec::new();
                budget.reserve_vec(&mut chain, points.len().saturating_add(2))?;
                chain.push((auv, av));
                for (uv, xyz, v) in &points {
                    budget.charge(1)?;
                    self.boundary_handle(k, s, *uv, *xyz, *v, budget)?;
                    budget.push(&mut chain, (*uv, *v))?;
                }
                budget.push(&mut chain, (cuv, cv))?;
                for w in chain.windows(2) {
                    budget.charge(1)?;
                    let [(xa, va), (xc, vc)] = w else {
                        continue;
                    };
                    let (Some(h0), Some(h1)) =
                        (self.find_boundary(*xa, *va)?, self.find_boundary(*xc, *vc)?)
                    else {
                        return Ok(false);
                    };
                    self.constrain(k, s, h0, h1, budget)?;
                }
            }
        }
        Ok(true)
    }

    /// The handle of vertex `id` of the table, by its position.
    fn handle_of(&self, id: usize) -> Option<FixedVertexHandle> {
        let uv = self.vertices.get(id)?.uv;
        match self.cdt.locate(self.scaled(uv)) {
            PositionInTriangulation::OnVertex(h) => Some(h),
            _ => None,
        }
    }

    fn find_boundary(&self, uv: Point2, vertex: u32) -> Result<Option<FixedVertexHandle>> {
        match self.cdt.locate(self.scaled(uv)) {
            PositionInTriangulation::OnVertex(h) if self.vertex(h)?.vertex == Some(vertex) => {
                Ok(Some(h))
            }
            _ => Ok(None),
        }
    }

    /// Removes interior points the refined boundary leaves outside, or closer to it
    /// than a tenth of their shortest edge.
    fn remove_stale_interior(&mut self, budget: &QueryBudget<'_>) -> Result<()> {
        let inside = inside_faces(&self.cdt, budget)?;
        let mut stale = Vec::new();
        budget.reserve_vec(&mut stale, self.cdt.num_vertices())?;
        for v in self.cdt.vertices() {
            budget.charge(1)?;
            let id = v.data().id;
            if self.vertices.get(id).is_none_or(|x| x.boundary) {
                continue;
            }
            let mut out = false;
            let mut shortest = f64::INFINITY;
            let mut near = f64::INFINITY;
            let pos = v.position();
            for e in v.out_edges() {
                budget.charge(1)?;
                match e.face().as_inner() {
                    Some(f) if is_inside(&inside, f.fix()) => {
                        near = near.min(constraint_clearance(&self.cdt, f.fix(), pos));
                    }
                    _ => out = true,
                }
                let q = e.to().position();
                shortest = shortest.min(((q.x - pos.x).powi(2) + (q.y - pos.y).powi(2)).sqrt());
            }
            if out || near < CLEARANCE_FRACTION * shortest {
                budget.push(&mut stale, pos)?;
            }
        }
        for pos in stale {
            budget.charge(1)?;
            if let PositionInTriangulation::OnVertex(h) = self.cdt.locate(pos) {
                let id = self.id(h);
                self.cdt.remove(h);
                if let Some(v) = self.vertices.get_mut(id) {
                    v.alive = false;
                }
            }
        }
        Ok(())
    }

    /// Inserts the grid points inside the loops that are clear of the boundary (0.3 of
    /// the step) and of the points already there (half the step), until the triangulation has
    /// `max_triangles` triangles.
    fn insert_grid(
        &mut self,
        k: &Kernel,
        s: &Surface,
        us: &[f64],
        vs: &[f64],
        scaled_step: f64,
        max_triangles: usize,
        budget: &QueryBudget<'_>,
    ) -> Result<()> {
        if us.is_empty() && vs.is_empty() {
            return Ok(());
        }
        let inside = inside_faces(&self.cdt, budget)?;
        let clearance = 0.3 * scaled_step;
        let spacing = 0.5 * scaled_step;
        let mut wanted = Vec::new();
        budget.reserve_vec(&mut wanted, us.len().saturating_mul(vs.len()))?;
        for (j, v) in vs.iter().enumerate() {
            budget.charge(1)?;
            for (i, u) in us.iter().enumerate() {
                budget.charge(1)?;
                let uv = Point2::new(*u, *v);
                let pos = self.scaled(uv);
                let PositionInTriangulation::OnFace(fh) = self.cdt.locate(pos) else {
                    continue;
                };
                if !is_inside(&inside, fh) || constraint_clearance(&self.cdt, fh, pos) < clearance {
                    continue;
                }
                let near = self.cdt.face(fh).vertices().iter().any(|w| {
                    let q = w.position();
                    ((q.x - pos.x).powi(2) + (q.y - pos.y).powi(2)).sqrt() < spacing
                });
                if !near {
                    budget.push(&mut wanted, (i, j))?;
                }
            }
        }
        for (i, j) in wanted {
            budget.charge(1)?;
            // the face's triangle budget holds whatever the lines asked for
            if self.cdt.num_inner_faces() >= max_triangles {
                break;
            }
            let (Some(u), Some(v)) = (us.get(i), vs.get(j)) else {
                continue;
            };
            self.insert_interior(k, s, Point2::new(*u, *v), None, 0, budget)?;
        }
        Ok(())
    }

    /// The sign of `(Su x Sv) . EvaluateNormal` at the face's origin, so a fast-path
    /// normal `Su x Sv` can be turned like the surface normal.
    fn cross_sign(&self, k: &Kernel, s: &Surface, budget: &QueryBudget<'_>) -> Result<Option<f64>> {
        budget.charge(1)?;
        let uv = self.origin;
        let Ok((_, su, sv)) = s.evaluate_1st_derivatives(k, uv, false, false) else {
            return Ok(None);
        };
        let Some(n) = surface_normal(k, s, uv) else {
            return Ok(None);
        };
        let d = su.cross(sv).dot(n);
        Ok((d != 0.0 && d.is_finite()).then(|| d.signum()))
    }

    /// Whether a triangle counter-clockwise in UV faces inward: the sign of
    /// `(Su x Sv) . outward` where `Su x Sv` is largest on a sample grid of `bbox`.
    fn uv_ccw_faces_inward(
        &self,
        k: &Kernel,
        s: &Surface,
        bbox: &Extent2d,
        budget: &QueryBudget<'_>,
    ) -> Result<bool> {
        let mut best: Option<(f64, f64)> = None;
        for j in 0..=4 {
            budget.charge(1)?;
            for i in 0..=4 {
                budget.charge(1)?;
                let uv = Point2::new(
                    bbox.u_min() + (bbox.u_max() - bbox.u_min()) * (0.1 + 0.8 * f64::from(i) / 4.0),
                    bbox.v_min() + (bbox.v_max() - bbox.v_min()) * (0.1 + 0.8 * f64::from(j) / 4.0),
                );
                let Ok((_, su, sv)) = s.evaluate_1st_derivatives(k, uv, false, false) else {
                    continue;
                };
                let Some(n) = surface_normal(k, s, uv) else {
                    continue;
                };
                let c = su.cross(sv);
                let l = c.length();
                if l.is_finite() && best.is_none_or(|(bl, _)| l > bl) {
                    best = Some((l, c.dot(n) * self.sense));
                }
            }
        }
        match best {
            Some((l, d)) if l > 0.0 => Ok(d < 0.0),
            _ => Err(invalid("surface has no normal on the face")),
        }
    }

    /// Whether `uv` lies inside the loops, off every vertex and boundary edge, and at
    /// least `need` (scaled UV) from the boundary edges near it.
    fn clear_of_boundary(&self, inside: &[bool], uv: Point2, need: f64) -> bool {
        let pos = self.scaled(uv);
        let faces = match self.cdt.locate(pos) {
            PositionInTriangulation::OnFace(f) => [Some(f), None],
            PositionInTriangulation::OnEdge(e)
                if !self.cdt.is_constraint_edge(e.as_undirected()) =>
            {
                let d = self.cdt.directed_edge(e);
                [
                    d.face().as_inner().map(|f| f.fix()),
                    d.rev().face().as_inner().map(|f| f.fix()),
                ]
            }
            _ => return false,
        };
        faces.iter().any(Option::is_some)
            && faces
                .into_iter()
                .flatten()
                .all(|f| is_inside(inside, f) && constraint_clearance(&self.cdt, f, pos) >= need)
    }

    /// Asks for boundary segment `r.segment` to be split (into the most pieces asked).
    fn request(&mut self, r: SegmentSplit, budget: &QueryBudget<'_>) -> Result<()> {
        budget.charge(self.requests.len() as u64)?;
        match self.requests.iter_mut().find(|q| q.segment == r.segment) {
            Some(q) => q.pieces = q.pieces.max(r.pieces),
            None => budget.push(&mut self.requests, r)?,
        }
        Ok(())
    }

    /// The request to split boundary segment `e` (between two mesh vertices) into one
    /// piece more than the grid lines it crosses, at least two.
    fn split_request(&self, e: FixedUndirectedEdgeHandle) -> Option<SegmentSplit> {
        let [va, vb] = self.cdt.undirected_edge(e).vertices();
        let a = self.vertices.get(va.data().id)?;
        let b = self.vertices.get(vb.data().id)?;
        let (x, y) = (a.vertex?, b.vertex?);
        if x == y {
            return None;
        }
        let crossed = |lines: &[f64], p: f64, q: f64| {
            let (lo, hi) = if p < q { (p, q) } else { (q, p) };
            lines.iter().filter(|t| **t > lo && **t < hi).count()
        };
        let n = crossed(&self.lines.0, a.uv.x, b.uv.x)
            .max(crossed(&self.lines.1, a.uv.y, b.uv.y))
            .saturating_add(1);
        let pieces = u32::try_from(n)
            .unwrap_or(MAX_SPLIT_PIECES)
            .clamp(2, MAX_SPLIT_PIECES);
        Some(SegmentSplit {
            segment: if x < y { [x, y] } else { [y, x] },
            pieces,
        })
    }

    /// The split request for the boundary segment nearest `uv` (inside the loops) when
    /// it is longer than `len` (scaled UV).
    fn blocking_segment(&self, inside: &[bool], uv: Point2, len: f64) -> Option<SegmentSplit> {
        let pos = self.scaled(uv);
        let f = match self.cdt.locate(pos) {
            PositionInTriangulation::OnFace(f) => f,
            PositionInTriangulation::OnEdge(e) => {
                let d = self.cdt.directed_edge(e);
                [d.face().as_inner(), d.rev().face().as_inner()]
                    .into_iter()
                    .flatten()
                    .next()?
                    .fix()
            }
            _ => return None,
        };
        if !is_inside(inside, f) {
            return None;
        }
        let (_, e) = nearest_constraint(&self.cdt, f, pos);
        let e = e?;
        let [x, y] = self.cdt.undirected_edge(e).positions();
        let seg = ((y.x - x.x).powi(2) + (y.y - x.y).powi(2)).sqrt();
        if seg > len {
            self.split_request(e)
        } else {
            None
        }
    }

    /// A point a quarter of directed edge `e`'s length beyond its midpoint, on the side
    /// away from `e`'s face, when it falls inside the loops clear of the boundary.
    fn beyond_chord(
        &self,
        inside: &[bool],
        e: spade::handles::DirectedEdgeHandle<'_, CdtVertex, (), spade::CdtEdge<()>, ()>,
    ) -> Option<Point2> {
        let [p, q] = e.positions();
        let (dx, dy) = (q.x - p.x, q.y - p.y);
        let len = (dx * dx + dy * dy).sqrt();
        if len <= 0.0 || len.is_nan() {
            return None;
        }
        // the face is on the edge's left: go right
        let pos = spade::Point2::new((p.x + q.x) / 2.0 + dy * 0.25, (p.y + q.y) / 2.0 - dx * 0.25);
        let PositionInTriangulation::OnFace(f) = self.cdt.locate(pos) else {
            return None;
        };
        if !is_inside(inside, f) || constraint_clearance(&self.cdt, f, pos) < 0.05 * len {
            return None;
        }
        Some(self.unscaled(pos))
    }

    /// The UV points the next refinement pass inserts: the midpoint of every interior
    /// edge off the chord (or too long), and for every triangle off the surface,
    /// flipped or turned too far from it, the midpoint of its longest edge in scaled UV
    /// (its centroid when that edge is on the boundary). Every point keeps clear of the
    /// boundary, so no sliver along it is made.
    fn refine_candidates(
        &mut self,
        k: &Kernel,
        s: &Surface,
        p: &FaceParams,
        changed: Option<&[FixedVertexHandle]>,
        budget: &QueryBudget<'_>,
    ) -> Result<Vec<(Point2, u8)>> {
        let inside = inside_faces(&self.cdt, budget)?;
        // every triangle and edge, or (after a pass that only inserted points) those
        // around the new points: every other one was checked with the same neighbours
        let (faces, edges): (
            Vec<FixedFaceHandle<InnerTag>>,
            Vec<FixedUndirectedEdgeHandle>,
        ) = match changed {
            None => (
                collect_with_budget(self.cdt.inner_faces().map(|f| f.fix()), budget)?,
                collect_with_budget(self.cdt.undirected_edges().map(|e| e.fix()), budget)?,
            ),
            Some(vs) => {
                let mut fs = FlatSet::new();
                let mut es = FlatSet::new();
                for v in vs {
                    budget.charge(1)?;
                    for e in self.cdt.vertex(*v).out_edges() {
                        budget.charge(1)?;
                        es.insert_with_budget(e.fix().as_undirected(), budget)?;
                        if let Some(f) = e.face().as_inner() {
                            fs.insert_with_budget(f.fix(), budget)?;
                        }
                    }
                }
                let mut faces = Vec::new();
                let mut face_iter = fs.into_iter();
                while let Some(face) = face_iter.next_with_budget(budget)? {
                    budget.push(&mut faces, face)?;
                }
                let mut edges = Vec::new();
                let mut edge_iter = es.into_iter();
                while let Some(edge) = edge_iter.next_with_budget(budget)? {
                    budget.push(&mut edges, edge)?;
                }
                (faces, edges)
            }
        };
        let tol = p.chord;
        let mut split_edges: Vec<FixedUndirectedEdgeHandle> = Vec::new();
        budget.reserve_vec(&mut split_edges, self.cdt.num_undirected_edges())?;
        let mut marked: FlatSet<usize> = FlatSet::new();
        let mut centroids: Vec<(Point2, f64, Option<SegmentSplit>, u8)> = Vec::new();
        budget.reserve_vec(&mut centroids, faces.len())?;
        let mut newly_good = Vec::new();
        budget.reserve_vec(&mut newly_good, faces.len())?;
        for fh in faces {
            budget.charge(self.lines.0.len().saturating_add(self.lines.1.len()) as u64)?;
            budget.charge(1)?;
            if !is_inside(&inside, fh) {
                continue;
            }
            let face = self.cdt.face(fh);
            let [va, vb, vc] = face.vertices();
            let ids = {
                let mut x = [va.data().id, vb.data().id, vc.data().id];
                x.sort_unstable();
                x
            };
            if self.good.contains_with_budget(&ids, budget)? {
                continue;
            }
            let a = self.vertex(va.fix())?;
            let b = self.vertex(vb.fix())?;
            let c = self.vertex(vc.fix())?;
            // a triangle collapsed onto a pole is dropped on output
            let same = |x: &FaceVertex, y: &FaceVertex| x.vertex.is_some() && x.vertex == y.vertex;
            if same(a, b) || same(b, c) || same(c, a) {
                continue;
            }
            let uv = Point2::new(
                (a.uv.x + b.uv.x + c.uv.x) / 3.0,
                (a.uv.y + b.uv.y + c.uv.y) / 3.0,
            );
            let mut n = (b.xyz - a.xyz).cross(c.xyz - a.xyz);
            if self.flip {
                n = -n;
            }
            let nl = n.length();
            let size = a
                .xyz
                .distance(b.xyz)
                .max(b.xyz.distance(c.xyz))
                .max(c.xyz.distance(a.xyz));
            let mut bad = p.max_len.is_some_and(|l| size > l);
            if !bad && nl > 0.0 && nl.is_finite() {
                // the surface at the centroid and between it and each corner (where
                // a twisted patch leaves a flat triangle) against the triangle's plane
                for w in [
                    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                    [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
                    [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                    [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
                ] {
                    let q = Point2::new(
                        a.uv.x * w[0] + b.uv.x * w[1] + c.uv.x * w[2],
                        a.uv.y * w[0] + b.uv.y * w[1] + c.uv.y * w[2],
                    );
                    let pt = self.point(k, s, q, budget)?;
                    if ((pt - a.xyz).dot(n) / nl).abs() > tol + a.gap.max(b.gap).max(c.gap) {
                        bad = true;
                        break;
                    }
                }
                // the surface folds over inside the triangle (its normal reverses
                // between corners): no triangle there can follow it, so no flip or
                // angle test
                let normals = [a.normal, b.normal, c.normal];
                let folded = normals.iter().flatten().enumerate().any(|(i, x)| {
                    normals
                        .iter()
                        .flatten()
                        .skip(i + 1)
                        .any(|y| x.dot(*y) < 0.0)
                });
                if folded {
                    // ... and its size: a fold is resolved down to a band of small
                    // triangles along it
                    bad = bad || size > FOLD_SIZE * tol;
                } else if bad {
                } else if let Some(o) = self.outward(k, s, uv, budget)? {
                    // flipped, or turned from the surface by more than the angle
                    // tolerance at the centroid or an interior corner (a boundary
                    // corner's normal may be a singular point's)
                    let facet = n / nl;
                    let interior = [a, b, c]
                        .into_iter()
                        .filter(|x| !x.boundary)
                        .filter_map(|x| x.normal);
                    // a sliver's facet normal is no measure of anything (a fan at a
                    // pole is all slivers): only the flip counts there
                    let sliver = nl < SLIVER * size * size;
                    bad = o.dot(n) < 0.0
                        || (!sliver
                            && core::iter::once(o)
                                .chain(interior)
                                .any(|q| angle(facet, q) > p.angle));
                }
            }
            // a triangle far below the tolerance or the length floor is left alone
            if !bad || size < MIN_SPLIT_FRACTION * tol || size < 2.0 * p.min_len {
                if !bad {
                    budget.push(&mut newly_good, ids)?;
                }
                continue;
            }
            let mut longest = None;
            let mut longest_len = -1.0;
            for e in face.adjacent_edges() {
                budget.charge(1)?;
                let [x, y] = e.positions();
                let l = (y.x - x.x).powi(2) + (y.y - x.y).powi(2);
                if l > longest_len {
                    longest_len = l;
                    longest = Some(e);
                }
            }
            let Some(e) = longest else {
                continue;
            };
            let ue = e.fix().as_undirected();
            let depth = a.depth.max(b.depth).max(c.depth).saturating_add(1);
            if self.cdt.is_constraint_edge(ue) {
                // too close to that boundary segment, the segment itself is split
                let request = self.split_request(ue);
                budget.push(
                    &mut centroids,
                    (uv, CLEARANCE_FRACTION * longest_len.sqrt(), request, depth),
                )?;
            } else if [a, b, c].iter().all(|x| x.boundary) {
                // all corners on the boundary (an ear between boundary points):
                // splitting the chord would put a point next to the boundary, so a
                // point goes beyond it, into the face across it
                if let Some(q) = self.beyond_chord(&inside, e) {
                    budget.push(&mut centroids, (q, 0.0, None, depth))?;
                }
            } else if marked.insert_with_budget(ue.index(), budget)? {
                budget.push(&mut split_edges, ue)?;
            }
        }
        let mut good_edges = Vec::new();
        budget.reserve_vec(&mut good_edges, edges.len())?;
        for eh in edges {
            budget.charge(1)?;
            let edge = self.cdt.undirected_edge(eh);
            if self.cdt.is_constraint_edge(eh)
                || marked.contains_with_budget(&eh.index(), budget)?
            {
                continue;
            }
            let [va, vb] = edge.vertices();
            let key = {
                let (x, y) = (va.data().id, vb.data().id);
                if x < y {
                    [x, y]
                } else {
                    [y, x]
                }
            };
            if self.good_edges.contains_with_budget(&key, budget)? {
                continue;
            }
            let d = edge.as_directed();
            let in_face =
                |f: Option<FixedFaceHandle<InnerTag>>| f.is_some_and(|f| is_inside(&inside, f));
            if !(in_face(d.face().as_inner().map(|f| f.fix()))
                || in_face(d.rev().face().as_inner().map(|f| f.fix())))
            {
                continue;
            }
            let a = self.vertex(va.fix())?;
            let b = self.vertex(vb.fix())?;
            if a.vertex.is_some() && a.vertex == b.vertex {
                continue;
            }
            let len = a.xyz.distance(b.xyz);
            if len < 2.0 * p.min_len {
                continue;
            }
            let uv = Point2::new((a.uv.x + b.uv.x) / 2.0, (a.uv.y + b.uv.y) / 2.0);
            let mut split = p.max_len.is_some_and(|l| len > l);
            if !split {
                let pt = self.point(k, s, uv, budget)?;
                split = segment_distance(pt, a.xyz, b.xyz) > tol + a.gap.max(b.gap);
            }
            if split && marked.insert_with_budget(edge.fix().index(), budget)? {
                budget.push(&mut split_edges, edge.fix())?;
            } else if !split {
                budget.push(&mut good_edges, key)?;
            }
        }
        for ids in newly_good {
            self.good.insert_with_budget(ids, budget)?;
        }
        for key in good_edges {
            self.good_edges.insert_with_budget(key, budget)?;
        }
        let mut out = Vec::new();
        budget.reserve_vec(&mut out, split_edges.len() + centroids.len())?;
        for e in split_edges {
            budget.charge(1)?;
            let edge = self.cdt.undirected_edge(e);
            let [x, y] = edge.positions();
            let len = ((y.x - x.x).powi(2) + (y.y - x.y).powi(2)).sqrt();
            let [va, vb] = edge.vertices();
            let a = self.vertex(va.fix())?;
            let b = self.vertex(vb.fix())?;
            let uv = Point2::new((a.uv.x + b.uv.x) / 2.0, (a.uv.y + b.uv.y) / 2.0);
            let depth = a.depth.max(b.depth).saturating_add(1);
            if depth > MAX_LOCAL_DEPTH {
                self.depth_capped = true;
            } else if self.clear_of_boundary(&inside, uv, CLEARANCE_FRACTION * len) {
                budget.push(&mut out, (uv, depth))?;
            } else if let Some(r) = self.blocking_segment(&inside, uv, len) {
                // a boundary segment longer than the edge keeps the split off it (a fan
                // from a long segment's ends): that segment is split
                self.request(r, budget)?;
            }
        }
        for (uv, need, request, depth) in centroids {
            budget.charge(1)?;
            if depth > MAX_LOCAL_DEPTH {
                self.depth_capped = true;
            } else if self.clear_of_boundary(&inside, uv, need) {
                budget.push(&mut out, (uv, depth))?;
            } else if let Some(r) = request {
                self.request(r, budget)?;
            }
        }
        Ok(out)
    }

    /// Fills a face with no triangle in UV from its one loop, by clipping ears in the
    /// loop's best-fit plane (no point added, so its neighbours still meet it).
    fn ear_clip(&mut self, k: &Kernel, s: &Surface, budget: &QueryBudget<'_>) -> Result<()> {
        let [lp] = self.boundary.as_slice() else {
            return Ok(());
        };
        if lp.len() < 3 || lp.len() > MAX_EAR_LOOP {
            return Ok(());
        }
        let mut ids = Vec::new();
        budget.reserve_vec(&mut ids, lp.len())?;
        for (uv, v) in lp {
            budget.charge(1)?;
            let Some(h) = self.find_boundary(*uv, *v)? else {
                return Ok(());
            };
            let id = self.id(h);
            if ids.last() != Some(&id) {
                budget.push(&mut ids, id)?;
            }
        }
        let pts: Vec<Point3> = collect_with_budget(
            ids.iter()
                .filter_map(|i| self.vertices.get(*i).map(|v| v.xyz)),
            budget,
        )?;
        // Newell normal of the loop
        let mut nrm = Vec3::new(0.0, 0.0, 0.0);
        for (i, a) in pts.iter().enumerate() {
            budget.charge(1)?;
            let Some(b) = pts.get((i + 1) % pts.len()) else {
                continue;
            };
            nrm += Vec3::new(
                (a.y - b.y) * (a.z + b.z),
                (a.z - b.z) * (a.x + b.x),
                (a.x - b.x) * (a.y + b.y),
            );
        }
        let Ok(nrm) = nrm.unitize() else {
            return Ok(());
        };
        let tris = ear_clip_polygon(&pts, nrm, budget)?;
        if tris.is_empty() {
            return Ok(());
        }
        // wind the triangles outward
        let outward = match ids.first().and_then(|i| self.vertices.get(*i)) {
            Some(v) => self.outward(k, s, v.uv, budget)?,
            None => None,
        };
        let reverse = outward.is_some_and(|o| o.dot(nrm) < 0.0);
        self.ears = collect_with_budget(
            tris.into_iter().filter_map(|[a, b, c]| {
                let (a, b, c) = (*ids.get(a)?, *ids.get(b)?, *ids.get(c)?);
                Some(if reverse { [a, c, b] } else { [a, b, c] })
            }),
            budget,
        )?;
        Ok(())
    }

    /// The face's current triangles over its vertex table, wound outward (corners on
    /// one pole vertex dropped).
    fn triangles(&self, budget: &QueryBudget<'_>) -> Result<Vec<[usize; 3]>> {
        if !self.ears.is_empty() {
            let mut out = Vec::new();
            budget.reserve_vec(&mut out, self.ears.len())?;
            out.extend_from_slice(&self.ears);
            return Ok(out);
        }
        let inside = inside_faces(&self.cdt, budget)?;
        let mut out = Vec::new();
        budget.reserve_vec(&mut out, self.cdt.num_inner_faces())?;
        for face in self.cdt.inner_faces() {
            budget.charge(1)?;
            if !is_inside(&inside, face.fix()) {
                continue;
            }
            let [a, b, c] = face.vertices().map(|v| v.data().id);
            let shared = |i: usize| self.vertices.get(i).and_then(|v| v.vertex);
            let same = |x: usize, y: usize| shared(x).is_some() && shared(x) == shared(y);
            if same(a, b) || same(b, c) || same(c, a) {
                continue;
            }
            budget.push(&mut out, if self.flip { [a, c, b] } else { [a, b, c] })?;
        }
        Ok(out)
    }

    /// The face's mesh now, over the vertices its triangles use.
    pub(crate) fn snapshot_with_budget(
        &self,
        k: &Kernel,
        b: &Brep,
        chord: Option<f64>,
        backend: Option<&dyn FaceBackend>,
        budget: &QueryBudget<'_>,
    ) -> Result<FaceMesh> {
        self.admit_retained(budget)?;
        let tris = self.triangles(budget)?;
        let mut local: FlatIndex<usize, u32> = FlatIndex::new();
        let mut m = FaceMesh {
            face: self.face,
            inverted_edgeuses: collect_with_budget(self.inverted.iter().copied(), budget)?,
            unconverged: self.unconverged,
            split_boundary: self.split_boundary,
            ear_clipped: !self.ears.is_empty(),
            uvs: Vec::new(),
            positions: Vec::new(),
            normals: Vec::new(),
            shared: Vec::new(),
            triangles: Vec::new(),
            gaps: Vec::new(),
            deviation: None,
            triangles_over: 0,
            backend: self.backend,
            backend_failed: self.backend_failed,
            backend_deviation: None,
        };
        for t in &tris {
            budget.charge(1)?;
            let mut corners = [0u32; 3];
            for (slot, id) in corners.iter_mut().zip(t) {
                budget.charge(1)?;
                if let Some(l) = local.get_with_budget(id, budget)? {
                    *slot = *l;
                    continue;
                }
                let v = self
                    .vertices
                    .get(*id)
                    .ok_or_else(|| invalid("vertex out of range"))?;
                let l = u32::try_from(m.uvs.len())
                    .map_err(|_| invalid("face has too many vertices"))?;
                budget.push(&mut m.uvs, v.uv)?;
                budget.push(&mut m.positions, v.xyz)?;
                budget.push(&mut m.normals, v.normal.unwrap_or(Vec3::new(0.0, 0.0, 0.0)))?;
                budget.push(&mut m.shared, v.vertex)?;
                budget.push(&mut m.gaps, v.gap)?;
                local.insert_with_budget(*id, l, budget)?;
                *slot = l;
            }
            budget.push(&mut m.triangles, corners)?;
        }
        if let Some(chord) = chord {
            let s = surface_of(b, self.face)?;
            let (dev, over) = crate::deviation::face_mesh_deviation_with_budget(
                k,
                s,
                &self.domain,
                &m,
                chord,
                budget,
            )?;
            m.deviation = Some(dev);
            m.triangles_over = over;
            // the backend's own measure of a face whose grid it placed
            if let (Some(be), Some(_)) = (backend, self.backend) {
                let face = BackendFace {
                    face: self.face,
                    surface: s,
                    bbox: self.bbox,
                    chord,
                    angle: 0.0,
                };
                let view = FaceMeshView {
                    uvs: &m.uvs,
                    positions: &m.positions,
                    triangles: &m.triangles,
                };
                budget.charge(1)?;
                m.backend_deviation = be.deviation(k, &face, &view, budget).transpose()?;
                budget.checkpoint()?;
            }
        }
        Ok(m)
    }

    /// The face.
    pub(crate) fn face(&self) -> FaceId {
        self.face
    }

    /// Boundary segments (sorted mesh vertex pairs, with the pieces each should become)
    /// that the last refinement could not resolve from inside the face: splitting them in the edge polylines
    /// ([`EdgeSet::split_segments`]) and refining again lets it.
    pub(crate) fn requests(&self) -> &[SegmentSplit] {
        &self.requests
    }

    /// Whether edge `e` bounds the face.
    pub(crate) fn uses_edge(&self, e: solidcore::EdgeId) -> bool {
        self.edges_used.contains(&e)
    }

    /// Records the shared mesh vertices an assembly gave the snapshot's interior
    /// vertices (matched by UV), so later levels keep them.
    pub(crate) fn adopt(&mut self, m: &FaceMesh) {
        let mut by_uv: FlatIndex<(u64, u64), u32> = FlatIndex::new();
        for (uv, sh) in m.uvs.iter().zip(&m.shared) {
            if let Some(v) = sh {
                by_uv.insert((uv.x.to_bits(), uv.y.to_bits()), *v);
            }
        }
        for v in &mut self.vertices {
            if v.vertex.is_none() {
                if let Some(id) = by_uv.get(&(v.uv.x.to_bits(), v.uv.y.to_bits())) {
                    v.vertex = Some(*id);
                }
            }
        }
    }
}

/// Fallible adapter collection; no inferred capacity from an opaque owner.
fn collect_with_budget<T>(
    items: impl IntoIterator<Item = T>,
    budget: &QueryBudget<'_>,
) -> Result<Vec<T>> {
    let mut out = Vec::new();
    let iter = items.into_iter();
    budget.reserve_vec(&mut out, iter.size_hint().0)?;
    for item in iter {
        budget.charge(1)?;
        budget.push(&mut out, item)?;
    }
    Ok(out)
}

/// What mapping an edge use into UV needs.
struct MapContext<'a, 'q> {
    budget: &'a QueryBudget<'q>,
    k: &'a Kernel,
    s: &'a Surface,
    domain: &'a Extent2d,
    edges: &'a EdgeSet,
    polish_above: f64,
    period: (Option<f64>, Option<f64>),
}

impl MapContext<'_, '_> {
    /// `uv` moved by whole periods to within half a period of `near`.
    fn unwrap(&self, uv: Point2, near: Point2) -> Point2 {
        let shift = |x: f64, c: f64, p: Option<f64>| match p {
            Some(p) if p > 0.0 => x - ((x - c) / p).round() * p,
            _ => x,
        };
        Point2::new(
            shift(uv.x, near.x, self.period.0),
            shift(uv.y, near.y, self.period.1),
        )
    }

    /// The points of one edge use in loop order (`reverse`: against the edge) mapped
    /// into UV: by the UV trim curve at the same normalised edge parameter when given
    /// (polished towards the edge point only when its surface point is off it), else
    /// by point inversion continuing from the previous point (`seed` for the first),
    /// kept within half a period of it on a closed surface.
    fn edgeuse_points(
        &self,
        poly: &EdgePolyline,
        ivl: &Extent1d,
        trim: Option<&Curve>,
        reverse: bool,
        seed: Option<Point2>,
    ) -> Result<Vec<LoopPoint>> {
        let budget = self.budget;
        let (k, s) = (self.k, self.s);
        let (t0, t1) = (ivl.min(), ivl.max());
        let len = t1 - t0;
        let n = poly.params.len();
        let mut out = Vec::new();
        budget.reserve_vec(&mut out, n)?;
        let mut prev = seed;
        for pos in 0..n {
            budget.charge(1)?;
            let i = if reverse { n - 1 - pos } else { pos };
            let (Some(t), Some(v)) = (poly.params.get(i), poly.vertices.get(i)) else {
                continue;
            };
            let xyz = self.edges.position(*v)?;
            let sn = if len > 0.0 {
                ((t - t0) / len).clamp(0.0, 1.0)
            } else if i == 0 {
                0.0
            } else {
                1.0
            };
            let uv = match trim {
                Some(c) => {
                    let ti = c.natural_interval();
                    let q = c.evaluate_point(k, ti.evaluate(sn))?;
                    if !(q.x.is_finite() && q.y.is_finite()) {
                        return Err(invalid("UV trim curve is not finite"));
                    }
                    let uv = Point2::new(q.x, q.y);
                    if poly.degenerate
                        || surface_point(k, s, uv)?.distance(xyz) <= self.polish_above
                    {
                        uv
                    } else {
                        foot_with_budget(k, s, uv, xyz, self.domain, budget)?.0
                    }
                }
                None => match prev {
                    Some(seed) => {
                        let uv = foot_with_budget(k, s, seed, xyz, self.domain, budget)?.0;
                        self.unwrap(uv, seed)
                    }
                    None => invert_point_with_budget(k, s, xyz, self.domain, budget)?,
                },
            };
            prev = Some(uv);
            budget.push(
                &mut out,
                LoopPoint {
                    uv,
                    xyz,
                    vertex: *v,
                    pole_next: poly.degenerate && pos == 0,
                },
            )?;
        }
        Ok(out)
    }
}

/// `su x sv` as a unit vector, `None` where it is too small against the tangents to
/// give a direction (a pole).
fn unit_cross(su: Vec3, sv: Vec3) -> Option<Vec3> {
    let c = su.cross(sv);
    let l = c.length();
    let m = su.length_squared().max(sv.length_squared());
    (l.is_finite() && l > 1e-10 * m).then(|| c / l)
}

/// How far `xyz` is from the surface point at `uv` (`0` when it does not evaluate).
fn surface_gap(
    k: &Kernel,
    s: &Surface,
    uv: Point2,
    xyz: Point3,
    budget: &QueryBudget<'_>,
) -> Result<f64> {
    budget.charge(1)?;
    Ok(s.evaluate_point(k, uv).map_or(0.0, |p| p.distance(xyz)))
}

/// Distance in scaled UV between `a` and `c`.
fn scaled_gap(a: Point2, c: Point2, scale: (f64, f64)) -> f64 {
    let (du, dv) = ((a.x - c.x) * scale.0, (a.y - c.y) * scale.1);
    (du * du + dv * dv).sqrt()
}

/// Whether the UV jump from `a` to `b` (one mesh vertex) is a pole: far apart in UV but
/// the surface stays at the vertex between them. A jump across most of the domain where
/// the surface does not stay is a loop that only closes across a seam.
fn junction_is_pole(
    k: &Kernel,
    s: &Surface,
    chord: f64,
    a: LoopPoint,
    b: LoopPoint,
    span: f64,
    scale: (f64, f64),
    budget: &QueryBudget<'_>,
) -> Result<bool> {
    let gap = scaled_gap(a.uv, b.uv, scale);
    if gap <= chord.max(1e-9 * span) {
        return Ok(false);
    }
    let mid = Point2::new((a.uv.x + b.uv.x) / 2.0, (a.uv.y + b.uv.y) / 2.0);
    budget.charge(1)?;
    if surface_point(k, s, mid)?.distance(a.xyz) <= chord {
        return Ok(true);
    }
    if gap > 0.1 * span {
        return Err(invalid(
            "loop does not close in UV (periodic face without seam edge)",
        ));
    }
    Ok(false)
}

/// Appends an edge use's points to a loop, merging the shared junction point.
fn join(
    pts: &mut Vec<LoopPoint>,
    mut seq: Vec<LoopPoint>,
    k: &Kernel,
    s: &Surface,
    chord: f64,
    span: f64,
    scale: (f64, f64),
    budget: &QueryBudget<'_>,
) -> Result<()> {
    if let (Some(last), Some(first)) = (pts.last().copied(), seq.first().copied()) {
        if last.vertex != first.vertex {
            return Err(invalid("loop edge uses do not connect"));
        }
        if junction_is_pole(k, s, chord, last, first, span, scale, budget)? {
            if let Some(l) = pts.last_mut() {
                l.pole_next = true;
            }
        } else {
            pts.pop();
            if let Some(f) = seq.first_mut() {
                f.pole_next |= last.pole_next;
            }
        }
    }
    budget.reserve_vec(pts, seq.len())?;
    pts.extend(seq);
    Ok(())
}

/// Flood fill from the outer face, flipping at every constraint edge: `true` for faces
/// inside the loops, indexed by face handle.
fn inside_faces(cdt: &Cdt, budget: &QueryBudget<'_>) -> Result<Vec<bool>> {
    let n = cdt.num_all_faces();
    let mut label = Vec::new();
    budget.reserve_vec(&mut label, n)?;
    label.resize(n, None);
    let outer = cdt.outer_face().fix();
    if let Some(l) = label.get_mut(outer.index()) {
        *l = Some(false);
    }
    let mut queue = Vec::new();
    budget.reserve_vec(&mut queue, n)?;
    queue.push(outer);
    let mut cursor = 0;
    let limit = cdt.num_directed_edges();
    while let Some(&fh) = queue.get(cursor) {
        budget.charge(1)?;
        cursor += 1;
        let inside = label.get(fh.index()).copied().flatten().unwrap_or(false);
        let Some(start) = cdt.face(fh).adjacent_edge() else {
            continue;
        };
        let mut e = start;
        for _ in 0..limit {
            budget.charge(1)?;
            let across = e.rev().face().fix();
            let constraint = cdt.is_constraint_edge(e.fix().as_undirected());
            if let Some(slot) = label.get_mut(across.index()) {
                if slot.is_none() {
                    *slot = Some(inside ^ constraint);
                    budget.push(&mut queue, across)?;
                }
            }
            e = e.next();
            if e == start {
                break;
            }
        }
    }
    collect_with_budget(label.into_iter().map(|l| l.unwrap_or(false)), budget)
}

fn is_inside(inside: &[bool], f: FixedFaceHandle<InnerTag>) -> bool {
    inside.get(f.index()).copied().unwrap_or(false)
}

/// Distance in the plane from `p` to segment `a`-`b`.
fn segment_distance_2d(p: spade::Point2<f64>, a: spade::Point2<f64>, b: spade::Point2<f64>) -> f64 {
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    let l2 = dx * dx + dy * dy;
    let s = if l2 > 0.0 {
        (((p.x - a.x) * dx + (p.y - a.y) * dy) / l2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (qx, qy) = (a.x + dx * s - p.x, a.y + dy * s - p.y);
    (qx * qx + qy * qy).sqrt()
}

/// The distance from `p` (in face `f`) to the nearest constraint edge of `f` or of its
/// neighbours.
fn constraint_clearance(cdt: &Cdt, f: FixedFaceHandle<InnerTag>, p: spade::Point2<f64>) -> f64 {
    nearest_constraint(cdt, f, p).0
}

/// The nearest constraint edge to `p` of face `f` and its neighbours, and its distance
/// (infinite and `None` when there is none).
fn nearest_constraint(
    cdt: &Cdt,
    f: FixedFaceHandle<InnerTag>,
    p: spade::Point2<f64>,
) -> (f64, Option<FixedUndirectedEdgeHandle>) {
    let mut best = (f64::INFINITY, None);
    for e in cdt.face(f).adjacent_edges() {
        let mut check =
            |x: spade::handles::DirectedEdgeHandle<'_, CdtVertex, (), spade::CdtEdge<()>, ()>| {
                let u = x.fix().as_undirected();
                if cdt.is_constraint_edge(u) {
                    let [a, b] = x.positions();
                    let d = segment_distance_2d(p, a, b);
                    if d < best.0 {
                        best = (d, Some(u));
                    }
                }
            };
        check(e);
        if let Some(other) = e.rev().face().as_inner() {
            for x in other.adjacent_edges() {
                check(x);
            }
        }
    }
    best
}

/// Ear clipping of the polygon `pts` (no repeated closing point) projected on the plane
/// of unit normal `n`: triangles counter-clockwise about `n`, empty when it fails.
fn ear_clip_polygon(pts: &[Point3], n: Vec3, budget: &QueryBudget<'_>) -> Result<Vec<[usize; 3]>> {
    // a frame of the plane
    // (`n` itself is the first axis it returns)
    let Ok([_, x, y]) = n.make_unit_ortho_vectors(None) else {
        return Ok(Vec::new());
    };
    let (x, y) = if x.cross(y).dot(n) < 0.0 {
        (y, x)
    } else {
        (x, y)
    };
    let p2: Vec<(f64, f64)> =
        collect_with_budget(pts.iter().map(|p| (p.dot(x), p.dot(y))), budget)?;
    let at = |i: usize| p2.get(i).copied().unwrap_or((0.0, 0.0));
    let cross = |a: (f64, f64), b: (f64, f64), c: (f64, f64)| {
        (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
    };
    let mut idx: Vec<usize> = collect_with_budget(0..pts.len(), budget)?;
    let mut out = Vec::new();
    budget.reserve_vec(&mut out, pts.len().saturating_sub(2))?;
    let mut guard = 0;
    while idx.len() > 3 && guard < pts.len() * pts.len() {
        budget.charge(1)?;
        guard += 1;
        let m = idx.len();
        let mut clipped = false;
        for i in 0..m {
            budget.charge(1)?;
            let (Some(&ia), Some(&ib), Some(&ic)) =
                (idx.get((i + m - 1) % m), idx.get(i), idx.get((i + 1) % m))
            else {
                continue;
            };
            let (a, b, c) = (at(ia), at(ib), at(ic));
            if cross(a, b, c) <= 0.0 {
                continue;
            }
            budget.charge(idx.len() as u64)?;
            let blocked = idx.iter().any(|&j| {
                j != ia && j != ib && j != ic && {
                    let q = at(j);
                    cross(a, b, q) >= 0.0 && cross(b, c, q) >= 0.0 && cross(c, a, q) >= 0.0
                }
            });
            if blocked {
                continue;
            }
            budget.push(&mut out, [ia, ib, ic])?;
            idx.remove(i);
            clipped = true;
            break;
        }
        if !clipped {
            return Ok(Vec::new());
        }
    }
    if let [a, b, c] = idx.as_slice() {
        if cross(at(*a), at(*b), at(*c)) > 0.0 {
            budget.push(&mut out, [*a, *b, *c])?;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dependency_constraint_outputs_are_admitted_before_adapter_growth() {
        use crate::budget::QueryLimit;
        use std::sync::atomic::AtomicBool;
        let make = || {
            let mut cdt = Cdt::new();
            for x in 0..8 {
                let bottom = cdt
                    .insert(CdtVertex {
                        pos: spade::Point2::new(f64::from(x), -1.0),
                        id: usize::MAX,
                    })
                    .unwrap();
                let top = cdt
                    .insert(CdtVertex {
                        pos: spade::Point2::new(f64::from(x), 1.0),
                        id: usize::MAX,
                    })
                    .unwrap();
                cdt.add_constraint(bottom, top);
            }
            let from = cdt
                .insert(CdtVertex {
                    pos: spade::Point2::new(-1.0, 0.0),
                    id: usize::MAX,
                })
                .unwrap();
            let to = cdt
                .insert(CdtVertex {
                    pos: spade::Point2::new(8.0, 0.0),
                    id: usize::MAX,
                })
                .unwrap();
            (cdt, from, to)
        };
        let cancel = AtomicBool::new(false);
        let (mut cdt, from, to) = make();
        let count = cdt.num_vertices();
        let denied = QueryBudget::new(u64::MAX, 0, &cancel);
        assert!(constraint_edges_with_budget(&mut cdt, &denied, |cdt| {
            cdt.try_add_constraint(from, to)
        })
        .is_err());
        assert_eq!(denied.failure(), Some(QueryLimit::Bytes));
        assert_eq!(cdt.num_vertices(), count);

        let limited = QueryBudget::new(
            u64::MAX,
            4 * size_of::<spade::handles::FixedDirectedEdgeHandle>(),
            &cancel,
        );
        assert!(constraint_edges_with_budget(&mut cdt, &limited, |cdt| {
            cdt.add_constraint_and_split(from, to, |pos| CdtVertex {
                pos,
                id: usize::MAX,
            })
        })
        .is_err());
        assert_eq!(limited.failure(), Some(QueryLimit::Bytes));

        let (mut cdt, from, to) = make();
        let budget = QueryBudget::new(u64::MAX, usize::MAX, &cancel);
        let returned = constraint_edges_with_budget(&mut cdt, &budget, |cdt| {
            cdt.add_constraint_and_split(from, to, |pos| CdtVertex {
                pos,
                id: usize::MAX,
            })
        })
        .unwrap();
        assert!(returned.len() >= 9);
        let returned_bytes = crate::storage::vector_bytes(&returned);
        assert_eq!(budget.admitted_bytes(), returned_bytes);
        let mut vertices = Vec::<FaceVertex>::new();
        budget
            .reserve_vec(&mut vertices, cdt.num_vertices() - count)
            .unwrap();
        assert!(
            budget.admitted_bytes() >= returned_bytes + crate::storage::vector_bytes(&vertices)
        );
    }

    #[test]
    fn flood_fill_and_ear_clipping_obey_query_limits() {
        use crate::budget::QueryLimit;
        use std::sync::atomic::AtomicBool;
        let cancel = AtomicBool::new(false);
        let cdt = Cdt::new();
        let bytes = QueryBudget::new(u64::MAX, 0, &cancel);
        assert!(inside_faces(&cdt, &bytes).is_err());
        assert_eq!(bytes.failure(), Some(QueryLimit::Bytes));

        let pts = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let work = QueryBudget::new(0, usize::MAX, &cancel);
        assert!(ear_clip_polygon(&pts, Vec3::new(0.0, 0.0, 1.0), &work).is_err());
        assert_eq!(work.failure(), Some(QueryLimit::Work));
        let cancelled = AtomicBool::new(true);
        let budget = QueryBudget::new(u64::MAX, usize::MAX, &cancelled);
        assert!(inside_faces(&cdt, &budget).is_err());
        assert_eq!(budget.failure(), Some(QueryLimit::Cancelled));
    }

    fn area(pts: &[Point3], tris: &[[usize; 3]]) -> f64 {
        tris.iter()
            .map(|t| {
                (pts[t[1]] - pts[t[0]])
                    .cross(pts[t[2]] - pts[t[0]])
                    .length()
                    / 2.0
            })
            .sum()
    }

    #[test]
    fn ear_clipping_covers_the_polygon() {
        let z = Vec3::new(0.0, 0.0, 1.0);
        // an L of area 3, counter-clockwise about +z
        let l = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(1.0, 2.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
        ];
        let t = ear_clip_polygon(&l, z, &QueryBudget::unbounded()).unwrap();
        assert_eq!(t.len(), 4);
        assert!((area(&l, &t) - 3.0).abs() < 1e-12);
        // every triangle counter-clockwise about the normal
        for [a, b, c] in &t {
            assert!((l[*b] - l[*a]).cross(l[*c] - l[*a]).dot(z) > 0.0);
        }
        // seen from the other side the polygon is clockwise: no ear, nothing clipped
        assert!(ear_clip_polygon(&l, -z, &QueryBudget::unbounded())
            .unwrap()
            .is_empty());
    }

    #[test]
    fn flood_fill_marks_the_inside_of_a_square_with_a_hole() {
        let mut cdt = Cdt::new();
        let ring = |pts: &[(f64, f64)], base: usize, cdt: &mut Cdt| {
            let hs: Vec<_> = pts
                .iter()
                .enumerate()
                .map(|(i, (x, y))| {
                    cdt.insert(CdtVertex {
                        pos: spade::Point2::new(*x, *y),
                        id: base + i,
                    })
                    .unwrap()
                })
                .collect();
            for i in 0..hs.len() {
                cdt.add_constraint(hs[i], hs[(i + 1) % hs.len()]);
            }
        };
        ring(
            &[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
            0,
            &mut cdt,
        );
        ring(
            &[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)],
            4,
            &mut cdt,
        );
        let inside = inside_faces(&cdt, &QueryBudget::unbounded()).unwrap();
        let mut area_in = 0.0;
        for f in cdt.inner_faces() {
            if is_inside(&inside, f.fix()) {
                area_in += f.area();
            }
        }
        assert!((area_in - 12.0).abs() < 1e-12, "{area_in}");
    }
}

/// Spade owns the allocation while the operation runs. Its splitting path may
/// restart on an offset segment (cdt.rs:938-987), so source vertex/edge counts do
/// not establish an output-capacity upper bound. Admit the pinned Rust Vec
/// minimum provisionally before entering, then account its actual returned
/// capacity before processing any handles or adding face vertices.
///
/// Internal DCEL/scratch/growth peaks remain the explicit opaque host boundary;
/// this reservation is not represented as an upper bound on those allocations.
fn constraint_edges_with_budget(
    cdt: &mut Cdt,
    budget: &QueryBudget<'_>,
    operation: impl FnOnce(&mut Cdt) -> Vec<spade::handles::FixedDirectedEdgeHandle>,
) -> Result<Vec<spade::handles::FixedDirectedEdgeHandle>> {
    use std::mem::size_of;
    type Handle = spade::handles::FixedDirectedEdgeHandle;
    // Rust 1.98 alloc/raw_vec/mod.rs::min_non_zero_cap: elements of this size
    // start with four slots. Actual allocator-returned capacity is checked below.
    let element = size_of::<Handle>();
    let slots = if element == 1 {
        8
    } else if element <= 1024 {
        4
    } else {
        1
    };
    let reserved = slots * element;
    budget.charge(1)?;
    let reservation = budget.reserve_dependency_output(reserved)?;
    let edges = operation(cdt);
    reservation.reconcile(crate::storage::vector_bytes(&edges))?;
    Ok(edges)
}
