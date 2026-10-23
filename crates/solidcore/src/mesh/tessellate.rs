//! B-rep tessellation: a watertight triangle mesh of a [`Brep`] within a chord tolerance.
//!
//! Upstream: none. SMLib has no face tessellator to a triangle mesh (only boundary
//! tessellation, `SmApiTessellateBoundaries`), so this module is new code, not a port.
//! The design follows the usual trimmed-surface scheme (as in truck's `truck-meshalgo`,
//! Apache-2.0; no code copied): sample trim boundaries in UV, add interior points,
//! triangulate with a constrained Delaunay triangulation in UV (the [`spade`] crate,
//! MIT or Apache-2.0), map to 3D and refine until within tolerance.
//!
//! | Rust                    | Upstream |
//! |-------------------------|----------|
//! | [`tessellate`]          | none     |
//! | [`TessellateOptions`]   | none     |
//! | [`Mesh`]                | none     |
//!
//! # Algorithm
//!
//! 1. **Edges.** Every edge is tessellated once, adaptively in its curve parameter
//!    (chord deviation at the quarter points, tangent turning angle, optional length),
//!    from its start vertex to its end vertex. Every face using the edge reuses that
//!    polyline, so adjacent faces share their boundary vertices exactly and the mesh is
//!    watertight by construction. B-rep vertices become one mesh vertex each.
//! 2. **Loops in UV.** Each edge use of a face's loops maps the shared polyline into the
//!    face's UV with the edge use's UV trim curve at the same normalised edge parameter
//!    (as upstream `SmEdgeuse::NormalizedEvaluate`), then polishes the UV with a few
//!    Gauss-Newton steps towards the 3D polyline point. An edge use without a UV trim
//!    curve (or whose curve fails to evaluate) falls back to point inversion on the
//!    surface; those edge uses are listed in [`Mesh::inverted_edgeuses`].
//! 3. **Seams and poles.** A periodic surface's seam is an ordinary pair of loop edges
//!    in UV (one use at each side of the period), so it needs no special case. A
//!    degenerate edge (a pole: a closed edge whose curve stays at its vertex), or a jump
//!    in UV between consecutive edge uses at a point where the surface is singular, is
//!    sampled in UV at the interior grid's spacing and every sample maps to the one
//!    pole vertex; the triangles that collapse there are dropped.
//! 4. **Interior.** A uniform UV grid sized from the surface's chord deviation and
//!    normal turning over a sample grid adds Steiner points inside the loops (points too
//!    close to the boundary are skipped). The constrained triangulation runs in UV
//!    scaled by the surface's mean `|Su|` and `|Sv|`, so its triangles are close to
//!    isotropic in 3D. Triangles inside the loops are found by flooding from the outer
//!    face, flipping at every constraint edge.
//! 5. **Refinement.** Each pass inserts the UV centroid of every triangle whose surface
//!    point leaves the triangle's plane by more than the chord tolerance, whose corner
//!    normals turn more than the angle tolerance or whose 3D normal opposes the surface
//!    normal, and the UV midpoint of every interior edge whose surface midpoint leaves
//!    the chord, or that is longer than the maximum length. Points are never inserted
//!    on a boundary (constraint) edge, so the boundary stays the shared polyline.
//!
//! Positions on a boundary are the edge curve's points (vertices: the B-rep vertex's
//! point); interior positions are surface points. Normals are the exact surface normal
//! at each triangle corner's UV, turned to the face's outward side (the side facing the
//! void region of a solid; the upward face use's side for a sheet).
//!
//! Deviations: not a port. Bounds (no recursion anywhere): an edge segment is split at
//! most [`MAX_EDGE_DEPTH`] times and an edge has at most [`MAX_EDGE_POINTS`] points
//! (more is an error); the interior grid has at most [`MAX_GRID`] cells per direction;
//! refinement runs at most [`TessellateOptions::max_refine_passes`] passes and stops
//! at [`TessellateOptions::max_face_vertices`] vertices per face (such a face is listed
//! in [`Mesh::unconverged_faces`]). Output is deterministic: every collection is
//! visited in id or insertion order and the triangulation is deterministic.

use std::collections::{BTreeMap, VecDeque};

use spade::handles::{FixedFaceHandle, FixedVertexHandle, InnerTag};
use spade::{ConstrainedDelaunayTriangulation, HasPosition, PositionInTriangulation, Triangulation};

use crate::curve::curve::Curve;
use crate::error::{EntityRef, Result, SolidError};
use crate::geom::extent2d::Extent2d;
use crate::kernel::Kernel;
use crate::math::{Point2, Point3, Vec3};
use crate::nurbs::b_spline_surface_n_lib::p2::OrientType;
use crate::nurbs::knots::Side;
use crate::surface::surface::Surface;
use crate::topology::brep::p1 as bl;
use crate::topology::edge as ed;
use crate::topology::edgeuse::p1 as eu1;
use crate::topology::{Brep, EdgeId, EdgeuseId, FaceId, FaceuseId, VertexId};

const OP: &str = "mesh::tessellate";

/// Most times an edge segment is halved.
pub const MAX_EDGE_DEPTH: u32 = 30;
/// Most points of one edge polyline; more is an error.
pub const MAX_EDGE_POINTS: usize = 1 << 20;
/// Most interior grid cells per UV direction of one face.
pub const MAX_GRID: usize = 512;
/// Segments every edge starts from before adaptive splitting.
const EDGE_INITIAL_SEGMENTS: usize = 4;
/// Cells per direction of the sample grid that sizes a face's interior grid.
const SAMPLE_GRID: usize = 8;
/// Most UV samples of one degenerate (pole) segment.
const MAX_DEGENERATE_SEGMENTS: usize = 1024;
/// Gauss-Newton steps of a UV polish.
const POLISH_STEPS: usize = 8;
/// Cells per direction of the coarse search that seeds a point inversion.
const INVERT_GRID: usize = 16;

/// What [`tessellate`] aims for.
#[derive(Debug, Clone, PartialEq)]
pub struct TessellateOptions {
    /// Largest distance between the mesh and the surface or curve it approximates
    /// (absolute, model units).
    pub chord_tolerance: f64,
    /// Largest angle (radians) between the normals at a triangle's corners and between
    /// the tangents at a polyline segment's ends.
    pub angle_tolerance: f64,
    /// Longest edge segment or interior triangle edge (model units), if bounded.
    pub max_edge_length: Option<f64>,
    /// Most refinement passes per face.
    pub max_refine_passes: u32,
    /// Most triangulation vertices per face; refinement stops there.
    pub max_face_vertices: usize,
}

impl Default for TessellateOptions {
    /// Chord `0.01`, angle 20 degrees, no length bound, 24 passes, 200 000 vertices.
    fn default() -> Self {
        Self {
            chord_tolerance: 0.01,
            angle_tolerance: 20.0_f64.to_radians(),
            max_edge_length: None,
            max_refine_passes: 24,
            max_face_vertices: 200_000,
        }
    }
}

/// The shared polyline of one B-rep edge.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgePolyline {
    /// The edge.
    pub edge: EdgeId,
    /// Curve parameters of the points, increasing over the edge's interval.
    pub params: Vec<f64>,
    /// Mesh vertex of each point (the first and last are the edge's vertices).
    pub vertices: Vec<u32>,
    /// The edge collapses to its vertex (a pole): two points on one mesh vertex.
    pub degenerate: bool,
}

/// An entity [`tessellate`] could not mesh, and why.
#[derive(Debug, Clone, PartialEq)]
pub struct Failure {
    /// The edge or face.
    pub entity: EntityRef,
    /// What stopped it.
    pub error: SolidError,
}

/// A triangle mesh of a B-rep.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mesh {
    /// Vertex positions.
    pub positions: Vec<Point3>,
    /// Triangles as vertex indices, counter-clockwise seen from the outward side.
    pub triangles: Vec<[u32; 3]>,
    /// Unit outward surface normal at each triangle's corners (parallel to
    /// [`Mesh::triangles`]); a corner's normal comes from its own face, so a vertex on
    /// a sharp edge has one normal per face.
    pub normals: Vec<[Vec3; 3]>,
    /// The B-rep face of each triangle (parallel to [`Mesh::triangles`]).
    pub triangle_faces: Vec<FaceId>,
    /// The polyline of every tessellated edge, in the B-rep's edge order.
    pub edges: Vec<EdgePolyline>,
    /// Edges and faces left out, in the order met.
    pub failures: Vec<Failure>,
    /// Edge uses mapped into UV by point inversion (no usable UV trim curve).
    pub inverted_edgeuses: Vec<EdgeuseId>,
    /// Faces whose refinement stopped at a bound before meeting the tolerances.
    pub unconverged_faces: Vec<FaceId>,
}

/// How the triangle edges of a [`Mesh`] are shared ([`Mesh::edge_report`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EdgeReport {
    /// Distinct undirected triangle edges.
    pub edges: usize,
    /// Edges of exactly one triangle (the boundary of an open mesh).
    pub open: Vec<[u32; 2]>,
    /// Edges of more than two triangles.
    pub non_manifold: Vec<[u32; 2]>,
    /// Edges of two triangles that run the same way along them (inconsistent winding).
    pub inconsistent: Vec<[u32; 2]>,
}

impl EdgeReport {
    /// Every edge has two triangles running opposite ways: a closed, consistently
    /// oriented mesh.
    #[must_use]
    pub fn is_watertight(&self) -> bool {
        self.open.is_empty() && self.non_manifold.is_empty() && self.inconsistent.is_empty()
    }
}

impl Mesh {
    /// The triangle's corner positions, `None` for an index out of range.
    #[must_use]
    pub fn triangle_points(&self, t: [u32; 3]) -> Option<[Point3; 3]> {
        Some([
            *self.positions.get(t[0] as usize)?,
            *self.positions.get(t[1] as usize)?,
            *self.positions.get(t[2] as usize)?,
        ])
    }

    /// Total triangle area.
    #[must_use]
    pub fn area(&self) -> f64 {
        self.triangles
            .iter()
            .filter_map(|t| self.triangle_points(*t))
            .map(|[a, b, c]| (b - a).cross(c - a).length() / 2.0)
            .sum()
    }

    /// Signed enclosed volume (divergence theorem; positive for a closed mesh wound
    /// outward).
    #[must_use]
    pub fn volume(&self) -> f64 {
        self.triangles
            .iter()
            .filter_map(|t| self.triangle_points(*t))
            .map(|[a, b, c]| a.dot(b.cross(c)) / 6.0)
            .sum()
    }

    /// How the triangle edges are shared: open, non-manifold and inconsistently wound
    /// edges, each as its sorted vertex pair in ascending order.
    #[must_use]
    pub fn edge_report(&self) -> EdgeReport {
        // (count, sum of directions: +1 for low -> high, -1 for high -> low)
        let mut uses: BTreeMap<[u32; 2], (usize, i64)> = BTreeMap::new();
        for t in &self.triangles {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let (key, dir) = if a < b { ([a, b], 1) } else { ([b, a], -1) };
                let e = uses.entry(key).or_insert((0, 0));
                e.0 += 1;
                e.1 += dir;
            }
        }
        let mut r = EdgeReport {
            edges: uses.len(),
            ..EdgeReport::default()
        };
        for (key, (count, dir)) in uses {
            match count {
                1 => r.open.push(key),
                2 if dir != 0 => r.inconsistent.push(key),
                2 => {}
                _ => r.non_manifold.push(key),
            }
        }
        r
    }
}

/// Tessellates every face of `b` into one mesh within `opts`.
///
/// Edges and faces that cannot be meshed are skipped and listed in
/// [`Mesh::failures`] (the mesh then has holes there); everything else is meshed.
///
/// # Errors
///
/// [`SolidError::InvalidInput`] for options out of range (a chord tolerance or maximum
/// edge length that is not finite and positive, an angle tolerance outside
/// `(0, pi]`); [`SolidError::NotFound`] for a B-rep whose lists name missing entities.
pub fn tessellate(k: &Kernel, b: &Brep, opts: &TessellateOptions) -> Result<Mesh> {
    check_options(opts)?;
    let mut mesh = Mesh::default();
    let mut vertex_index: BTreeMap<VertexId, u32> = BTreeMap::new();
    let mut edge_index: BTreeMap<EdgeId, usize> = BTreeMap::new();
    for e in bl::edges(b, None)? {
        match tessellate_edge(k, b, e, opts, &mut mesh.positions, &mut vertex_index) {
            Ok(poly) => {
                edge_index.insert(e, mesh.edges.len());
                mesh.edges.push(poly);
            }
            Err(error) => mesh.failures.push(Failure {
                entity: EntityRef::Edge(e),
                error,
            }),
        }
    }
    for f in bl::faces(b, None)? {
        let shared = Shared {
            edges: &mesh.edges,
            edge_index: &edge_index,
            positions: &mesh.positions,
        };
        match mesh_face(k, b, f, opts, &shared) {
            Ok(out) => append_face(&mut mesh, f, out),
            Err(error) => mesh.failures.push(Failure {
                entity: EntityRef::Face(f),
                error,
            }),
        }
    }
    Ok(mesh)
}

fn check_options(opts: &TessellateOptions) -> Result<()> {
    if !(opts.chord_tolerance.is_finite() && opts.chord_tolerance > 0.0) {
        return Err(SolidError::invalid_input(
            OP,
            "chord tolerance must be finite and positive",
        ));
    }
    if !(opts.angle_tolerance > 0.0 && opts.angle_tolerance <= core::f64::consts::PI) {
        return Err(SolidError::invalid_input(
            OP,
            "angle tolerance must be in (0, pi]",
        ));
    }
    if let Some(l) = opts.max_edge_length {
        if !(l.is_finite() && l > 0.0) {
            return Err(SolidError::invalid_input(
                OP,
                "maximum edge length must be finite and positive",
            ));
        }
    }
    Ok(())
}

// ---- edges ----------------------------------------------------------------------------

/// The mesh vertex of B-rep vertex `v`, added at its point on first use.
fn vertex_slot(
    b: &Brep,
    v: VertexId,
    positions: &mut Vec<Point3>,
    vertex_index: &mut BTreeMap<VertexId, u32>,
) -> Result<u32> {
    if let Some(i) = vertex_index.get(&v) {
        return Ok(*i);
    }
    let i = push_position(positions, b.vertex(v)?.point)?;
    vertex_index.insert(v, i);
    Ok(i)
}

fn push_position(positions: &mut Vec<Point3>, p: Point3) -> Result<u32> {
    let i = u32::try_from(positions.len())
        .map_err(|_| SolidError::invalid_input(OP, "more than u32::MAX mesh vertices"))?;
    positions.push(p);
    Ok(i)
}

/// Point and first derivative of `curve` at `t` (from the left at the interval end).
fn curve_eval(k: &Kernel, curve: &Curve, t: f64, from_left: bool) -> Result<(Point3, Vec3)> {
    let d = curve.evaluate(k, t, 1, from_left, true)?;
    let p = *d
        .first()
        .ok_or_else(|| SolidError::invalid_input(OP, "curve evaluation without point"))?;
    let t1 = d.get(1).copied().unwrap_or(Vec3::new(0.0, 0.0, 0.0));
    if !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite()) {
        return Err(SolidError::invalid_input(OP, "curve evaluates to a non-finite point"));
    }
    Ok((p, t1))
}

/// Distance from `p` to the segment `a`-`b`.
fn segment_distance(p: Point3, a: Point3, b: Point3) -> f64 {
    let d = b - a;
    let l2 = d.length_squared();
    if l2 <= 0.0 {
        return p.distance(a);
    }
    let s = ((p - a).dot(d) / l2).clamp(0.0, 1.0);
    p.distance(a + d * s)
}

/// Angle between two vectors, `0` when either is zero.
fn angle(a: Vec3, b: Vec3) -> f64 {
    let la = a.length();
    let lb = b.length();
    if la <= 0.0 || lb <= 0.0 || !(la.is_finite() && lb.is_finite()) {
        return 0.0;
    }
    (a.dot(b) / (la * lb)).clamp(-1.0, 1.0).acos()
}

/// One pending piece of an edge's adaptive subdivision.
#[derive(Debug, Clone, Copy)]
struct Segment {
    t0: f64,
    t1: f64,
    p0: Point3,
    p1: Point3,
    d0: Vec3,
    d1: Vec3,
    depth: u32,
}

fn tessellate_edge(
    k: &Kernel,
    b: &Brep,
    e: EdgeId,
    opts: &TessellateOptions,
    positions: &mut Vec<Point3>,
    vertex_index: &mut BTreeMap<VertexId, u32>,
) -> Result<EdgePolyline> {
    let (v0, v1) = ed::start_end_vertices(b, e)?;
    let (Some(v0), Some(v1)) = (v0, v1) else {
        return Err(SolidError::invalid_input(OP, "edge without vertices"));
    };
    let i0 = vertex_slot(b, v0, positions, vertex_index)?;
    let i1 = vertex_slot(b, v1, positions, vertex_index)?;
    let rec = b.edge(e)?;
    let ivl = rec.interval;
    let (t0, t1) = (ivl.min(), ivl.max());
    let degenerate = |params: Vec<f64>| EdgePolyline {
        edge: e,
        params,
        vertices: vec![i0, i1],
        degenerate: true,
    };
    let Some(curve) = rec.curve.as_ref() else {
        if v0 == v1 {
            return Ok(degenerate(vec![t0, t1]));
        }
        return Err(SolidError::invalid_input(OP, "edge without curve"));
    };
    if !(t0.is_finite() && t1.is_finite() && t0 < t1) {
        if v0 == v1 {
            return Ok(degenerate(vec![t0, t1]));
        }
        return Err(SolidError::invalid_input(OP, "edge with an empty interval"));
    }

    // the initial uniform samples
    let mut samples = Vec::with_capacity(EDGE_INITIAL_SEGMENTS + 1);
    for i in 0..=EDGE_INITIAL_SEGMENTS {
        let t = ivl.evaluate(i as f64 / EDGE_INITIAL_SEGMENTS as f64);
        let (p, d) = curve_eval(k, curve, t, i == EDGE_INITIAL_SEGMENTS)?;
        samples.push((t, p, d));
    }

    // a closed edge whose curve stays at its vertex is a pole
    if v0 == v1 {
        let pv = b.vertex(v0)?.point;
        let tol = opts.chord_tolerance.min(rec.tolerance.max(0.0) * 10.0).max(1e-12);
        let mut collapsed = true;
        for i in 0..=16 {
            let t = ivl.evaluate(f64::from(i) / 16.0);
            if curve_eval(k, curve, t, i == 16)?.0.distance(pv) > tol {
                collapsed = false;
                break;
            }
        }
        if collapsed {
            return Ok(degenerate(vec![t0, t1]));
        }
    }

    let mut stack: Vec<Segment> = Vec::new();
    for w in samples.windows(2).rev() {
        if let [(ta, pa, da), (tb, pb, db)] = w {
            stack.push(Segment {
                t0: *ta,
                t1: *tb,
                p0: *pa,
                p1: *pb,
                d0: *da,
                d1: *db,
                depth: 0,
            });
        }
    }
    let mut params = vec![t0];
    let mut vertices = vec![i0];
    while let Some(s) = stack.pop() {
        let split = s.depth < MAX_EDGE_DEPTH && needs_split(k, curve, &s, opts)?;
        if split {
            let tm = (s.t0 + s.t1) / 2.0;
            let (pm, dm) = curve_eval(k, curve, tm, false)?;
            stack.push(Segment {
                t0: tm,
                t1: s.t1,
                p0: pm,
                p1: s.p1,
                d0: dm,
                d1: s.d1,
                depth: s.depth + 1,
            });
            stack.push(Segment {
                t0: s.t0,
                t1: tm,
                p0: s.p0,
                p1: pm,
                d0: s.d0,
                d1: dm,
                depth: s.depth + 1,
            });
            continue;
        }
        if s.t1 >= t1 {
            params.push(t1);
            vertices.push(i1);
        } else {
            params.push(s.t1);
            vertices.push(push_position(positions, s.p1)?);
        }
        if params.len() > MAX_EDGE_POINTS {
            return Err(SolidError::invalid_input(OP, "edge needs too many points"));
        }
    }
    Ok(EdgePolyline {
        edge: e,
        params,
        vertices,
        degenerate: false,
    })
}

/// Whether segment `s` of `curve` is outside the tolerances.
fn needs_split(k: &Kernel, curve: &Curve, s: &Segment, opts: &TessellateOptions) -> Result<bool> {
    let chord = s.p1 - s.p0;
    if let Some(l) = opts.max_edge_length {
        if chord.length() > l {
            return Ok(true);
        }
    }
    if angle(s.d0, s.d1) > opts.angle_tolerance {
        return Ok(true);
    }
    for f in [0.25, 0.5, 0.75] {
        let t = s.t0 + (s.t1 - s.t0) * f;
        let (p, _) = curve_eval(k, curve, t, false)?;
        if segment_distance(p, s.p0, s.p1) > opts.chord_tolerance {
            return Ok(true);
        }
    }
    Ok(false)
}

// ---- faces ----------------------------------------------------------------------------

/// What the faces share: the edge polylines and the positions so far.
struct Shared<'a> {
    edges: &'a [EdgePolyline],
    edge_index: &'a BTreeMap<EdgeId, usize>,
    positions: &'a [Point3],
}

impl Shared<'_> {
    fn polyline(&self, e: EdgeId) -> Result<&EdgePolyline> {
        self.edge_index
            .get(&e)
            .and_then(|i| self.edges.get(*i))
            .ok_or_else(|| SolidError::invalid_input(OP, "face uses an edge that failed"))
    }

    fn position(&self, i: u32) -> Result<Point3> {
        self.positions
            .get(i as usize)
            .copied()
            .ok_or_else(|| SolidError::invalid_input(OP, "mesh vertex out of range"))
    }
}

/// One point of a face's boundary loop in UV.
#[derive(Debug, Clone, Copy)]
struct LoopPoint {
    uv: Point2,
    xyz: Point3,
    vertex: u32,
    /// The segment to the next point collapses to this point's vertex (a pole).
    degenerate_next: bool,
}

/// A vertex of a face's triangulation.
#[derive(Debug, Clone, Copy)]
struct FaceVertex {
    uv: Point2,
    xyz: Point3,
    /// The shared mesh vertex of a boundary point; interior points get theirs on output.
    vertex: Option<u32>,
    /// Unit outward normal, when the surface has one there.
    normal: Option<Vec3>,
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

type Cdt = ConstrainedDelaunayTriangulation<CdtVertex>;

/// One face's triangles before they join the mesh.
struct FaceOutput {
    vertices: Vec<FaceVertex>,
    triangles: Vec<[usize; 3]>,
    inverted: Vec<EdgeuseId>,
    converged: bool,
}

fn append_face(mesh: &mut Mesh, f: FaceId, out: FaceOutput) {
    let mut ids: Vec<Option<u32>> = out.vertices.iter().map(|v| v.vertex).collect();
    for t in &out.triangles {
        let mut corners = [0u32; 3];
        let mut normals = [Vec3::new(0.0, 0.0, 0.0); 3];
        let mut ok = true;
        for (c, (slot, n)) in t.iter().zip(corners.iter_mut().zip(normals.iter_mut())) {
            let (Some(v), Some(id)) = (out.vertices.get(*c), ids.get_mut(*c)) else {
                ok = false;
                break;
            };
            let index = match *id {
                Some(i) => i,
                None => match push_position(&mut mesh.positions, v.xyz) {
                    Ok(i) => {
                        *id = Some(i);
                        i
                    }
                    Err(_) => {
                        ok = false;
                        break;
                    }
                },
            };
            *slot = index;
            *n = v.normal.unwrap_or(Vec3::new(0.0, 0.0, 0.0));
        }
        if !ok || corners[0] == corners[1] || corners[1] == corners[2] || corners[2] == corners[0]
        {
            continue;
        }
        // a corner without surface normal takes the triangle's own
        if normals.iter().any(|n| n.length_squared() == 0.0) {
            if let Some([a, b, c]) = mesh.triangle_points(corners) {
                let g = (b - a).cross(c - a).unitize_or_z();
                for n in &mut normals {
                    if n.length_squared() == 0.0 {
                        *n = g;
                    }
                }
            }
        }
        mesh.triangles.push(corners);
        mesh.normals.push(normals);
        mesh.triangle_faces.push(f);
    }
    mesh.inverted_edgeuses.extend(out.inverted);
    if !out.converged {
        mesh.unconverged_faces.push(f);
    }
}

/// The outward side of face `f`: `+1` when it is the surface normal's side, `-1`
/// otherwise. A solid's outward side faces its void region: the face use whose region
/// is void looks out along its orientation. A sheet (no or both sides void) takes its
/// upward face use's side.
fn outward_sense(b: &Brep, f: FaceId) -> Result<f64> {
    let up = b
        .face(f)?
        .faceuse
        .ok_or_else(|| SolidError::invalid_input(OP, "face without face use"))?;
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

/// Unit surface normal at `uv` (`EvaluateNormal`, else `Su x Sv`), `None` where the
/// surface has none.
fn surface_normal(k: &Kernel, s: &Surface, uv: Point2) -> Option<Vec3> {
    if let Ok(n) = s.evaluate_normal(k, uv, Side::Right, Side::Right) {
        let l = n.length();
        if l.is_finite() && l > 0.5 {
            return Some(n / l);
        }
    }
    let (_, su, sv) = s.evaluate_1st_derivatives(k, uv, false, false).ok()?;
    let c = su.cross(sv);
    let l = c.length();
    (l.is_finite() && l > 0.0).then(|| c / l)
}

fn finite3(p: Point3) -> bool {
    p.x.is_finite() && p.y.is_finite() && p.z.is_finite()
}

fn surface_point(k: &Kernel, s: &Surface, uv: Point2) -> Result<Point3> {
    let p = s.evaluate_point(k, uv)?;
    if finite3(p) {
        Ok(p)
    } else {
        Err(SolidError::invalid_input(OP, "surface evaluates to a non-finite point"))
    }
}

fn clamp_uv(uv: Point2, d: &Extent2d) -> Point2 {
    if d.u_min() <= d.u_max() && d.v_min() <= d.v_max() {
        Point2::new(uv.x.clamp(d.u_min(), d.u_max()), uv.y.clamp(d.v_min(), d.v_max()))
    } else {
        uv
    }
}

/// `uv` moved by Gauss-Newton steps towards the surface point nearest `target`, kept
/// inside `domain`; each step must bring the surface point closer.
fn polish_uv(k: &Kernel, s: &Surface, uv: Point2, target: Point3, domain: &Extent2d) -> Point2 {
    let Ok(p0) = s.evaluate_point(k, uv) else {
        return uv;
    };
    let mut best = p0.distance(target);
    let mut uv = uv;
    for _ in 0..POLISH_STEPS {
        if best == 0.0 {
            break;
        }
        let Ok((p, su, sv)) = s.evaluate_1st_derivatives(k, uv, false, false) else {
            break;
        };
        let r = target - p;
        let (a11, a12, a22) = (su.dot(su), su.dot(sv), sv.dot(sv));
        let (b1, b2) = (su.dot(r), sv.dot(r));
        let det = a11 * a22 - a12 * a12;
        if !(det.is_finite() && det > 1e-24 * a11 * a22) {
            break;
        }
        let du = (b1 * a22 - b2 * a12) / det;
        let dv = (a11 * b2 - a12 * b1) / det;
        let next = clamp_uv(Point2::new(uv.x + du, uv.y + dv), domain);
        let Ok(q) = s.evaluate_point(k, next) else {
            break;
        };
        let d = q.distance(target);
        if !(d < best) {
            break;
        }
        best = d;
        uv = next;
    }
    uv
}

/// The UV of `target` by a coarse search over `domain` and a polish.
fn invert_point(k: &Kernel, s: &Surface, target: Point3, domain: &Extent2d) -> Result<Point2> {
    let mut best: Option<(f64, Point2)> = None;
    for j in 0..=INVERT_GRID {
        for i in 0..=INVERT_GRID {
            let uv = Point2::new(
                domain.u_min() + (domain.u_max() - domain.u_min()) * i as f64 / INVERT_GRID as f64,
                domain.v_min() + (domain.v_max() - domain.v_min()) * j as f64 / INVERT_GRID as f64,
            );
            if let Ok(p) = s.evaluate_point(k, uv) {
                let d = p.distance(target);
                if best.is_none_or(|(bd, _)| d < bd) {
                    best = Some((d, uv));
                }
            }
        }
    }
    let (_, uv) = best.ok_or_else(|| SolidError::invalid_input(OP, "surface does not evaluate"))?;
    Ok(polish_uv(k, s, uv, target, domain))
}

/// Mean `|Su|` and `|Sv|` over a sample grid of `d` (each at least a tiny positive
/// value), so `(u * su, v * sv)` is roughly arc length.
fn surface_scales(k: &Kernel, s: &Surface, d: &Extent2d) -> (f64, f64) {
    let (mut su_sum, mut sv_sum, mut n) = (0.0, 0.0, 0.0);
    for j in 0..=4 {
        for i in 0..=4 {
            let uv = Point2::new(
                d.u_min() + (d.u_max() - d.u_min()) * f64::from(i) / 4.0,
                d.v_min() + (d.v_max() - d.v_min()) * f64::from(j) / 4.0,
            );
            if let Ok((_, su, sv)) = s.evaluate_1st_derivatives(k, uv, false, false) {
                let (a, b) = (su.length(), sv.length());
                if a.is_finite() && b.is_finite() {
                    su_sum += a;
                    sv_sum += b;
                    n += 1.0;
                }
            }
        }
    }
    let (mut su, mut sv) = if n > 0.0 {
        (su_sum / n, sv_sum / n)
    } else {
        (1.0, 1.0)
    };
    if !(su > 1e-300) {
        su = if sv > 1e-300 { sv } else { 1.0 };
    }
    if !(sv > 1e-300) {
        sv = su;
    }
    (su, sv)
}

/// A face being meshed: its surface, UV frame and triangulation.
struct FaceMesher<'a> {
    k: &'a Kernel,
    surface: &'a Surface,
    opts: &'a TessellateOptions,
    /// The UV the triangulation is centred on.
    origin: Point2,
    /// UV scale factors (`|Su|`, `|Sv|` means).
    scale: (f64, f64),
    /// `+1` when the outward side is the surface normal's.
    sense: f64,
    cdt: Cdt,
    vertices: Vec<FaceVertex>,
}

impl FaceMesher<'_> {
    fn scaled(&self, uv: Point2) -> spade::Point2<f64> {
        spade::mitigate_underflow(spade::Point2::new(
            (uv.x - self.origin.x) * self.scale.0,
            (uv.y - self.origin.y) * self.scale.1,
        ))
    }

    fn outward(&self, uv: Point2) -> Option<Vec3> {
        surface_normal(self.k, self.surface, uv).map(|n| n * self.sense)
    }

    fn vertex(&self, h: FixedVertexHandle) -> Result<&FaceVertex> {
        let id = self.cdt.vertex(h).data().id;
        self.vertices
            .get(id)
            .ok_or_else(|| SolidError::invalid_input(OP, "triangulation vertex out of range"))
    }

    /// Inserts a vertex not yet in the triangulation.
    fn insert(&mut self, v: FaceVertex) -> Result<FixedVertexHandle> {
        let pos = self.scaled(v.uv);
        let id = self.vertices.len();
        let h = self
            .cdt
            .insert(CdtVertex { pos, id })
            .map_err(|_| SolidError::invalid_input(OP, "UV point cannot be triangulated"))?;
        self.vertices.push(v);
        Ok(h)
    }

    /// Inserts an interior point at `uv` (its surface point and normal).
    fn insert_interior(&mut self, uv: Point2) -> Result<FixedVertexHandle> {
        let xyz = surface_point(self.k, self.surface, uv)?;
        let normal = self.outward(uv);
        self.insert(FaceVertex {
            uv,
            xyz,
            vertex: None,
            normal,
        })
    }
}

/// Flood fill from the outer face, flipping at every constraint edge: `true` for faces
/// inside the loops, indexed by face handle.
fn inside_faces(cdt: &Cdt) -> Vec<bool> {
    let n = cdt.num_all_faces();
    let mut label: Vec<Option<bool>> = vec![None; n];
    let outer = cdt.outer_face().fix();
    if let Some(l) = label.get_mut(outer.index()) {
        *l = Some(false);
    }
    let mut queue = VecDeque::from([outer]);
    let limit = cdt.num_directed_edges();
    while let Some(fh) = queue.pop_front() {
        let inside = label.get(fh.index()).copied().flatten().unwrap_or(false);
        let Some(start) = cdt.face(fh).adjacent_edge() else {
            continue;
        };
        let mut e = start;
        for _ in 0..limit {
            let across = e.rev().face().fix();
            let constraint = cdt.is_constraint_edge(e.fix().as_undirected());
            if let Some(slot) = label.get_mut(across.index()) {
                if slot.is_none() {
                    *slot = Some(inside ^ constraint);
                    queue.push_back(across);
                }
            }
            e = e.next();
            if e == start {
                break;
            }
        }
    }
    label.into_iter().map(|l| l.unwrap_or(false)).collect()
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
    let mut best = f64::INFINITY;
    let face = cdt.face(f);
    for e in face.adjacent_edges() {
        let mut edges = vec![e];
        if let Some(other) = e.rev().face().as_inner() {
            edges.extend(other.adjacent_edges());
        }
        for x in edges {
            if cdt.is_constraint_edge(x.fix().as_undirected()) {
                let [a, b] = x.positions();
                best = best.min(segment_distance_2d(p, a, b));
            }
        }
    }
    best
}

/// The boundary loops of face `f` in UV, outer first, each closed implicitly.
fn face_loops(
    k: &Kernel,
    b: &Brep,
    f: FaceId,
    s: &Surface,
    shared: &Shared<'_>,
    opts: &TessellateOptions,
    scale: (f64, f64),
    inverted: &mut Vec<EdgeuseId>,
) -> Result<Vec<Vec<LoopPoint>>> {
    let domain = s.natural_uv_domain();
    let span = Point2::new(
        (domain.u_max() - domain.u_min()) * scale.0,
        (domain.v_max() - domain.v_min()) * scale.1,
    );
    let span = span.x.max(span.y);
    let scaled_gap = |a: Point2, c: Point2| {
        let (du, dv) = ((a.x - c.x) * scale.0, (a.y - c.y) * scale.1);
        (du * du + dv * dv).sqrt()
    };
    let fu = b
        .face(f)?
        .faceuse
        .ok_or_else(|| SolidError::invalid_input(OP, "face without face use"))?;
    let mut loops = Vec::new();
    for lu in eu1::faceuse_loopuses(b, fu, false, None)? {
        let mut pts: Vec<LoopPoint> = Vec::new();
        for eu in eu1::loopuse_edgeuses(b, lu)? {
            let e = eu1::edge(b, eu)?
                .ok_or_else(|| SolidError::invalid_input(OP, "edge use without edge"))?;
            let poly = shared.polyline(e)?;
            let ivl = b.edge(e)?.interval;
            let trim = eu1::uv_trim_curve(b, eu)?.0;
            let seq = edgeuse_points(k, s, poly, &ivl, trim, &domain, shared)
                .or_else(|_| {
                    inverted.push(eu);
                    edgeuse_points(k, s, poly, &ivl, None, &domain, shared)
                })?;
            let mut seq = seq;
            if eu1::orientation(b, eu)? == OrientType::Opposite {
                seq.reverse();
                // a degenerate run is flagged on its first point
                if poly.degenerate {
                    for p in &mut seq {
                        p.degenerate_next = false;
                    }
                    if let Some(first) = seq.first_mut() {
                        first.degenerate_next = true;
                    }
                }
            }
            join(&mut pts, seq, k, s, opts, span, &scaled_gap)?;
        }
        // close the loop
        if pts.len() > 1 {
            let (Some(first), Some(last)) = (pts.first().copied(), pts.last().copied()) else {
                continue;
            };
            if first.vertex != last.vertex {
                return Err(SolidError::invalid_input(OP, "loop does not close"));
            }
            if junction_is_pole(k, s, opts, last, first, span, &scaled_gap)? {
                if let Some(l) = pts.last_mut() {
                    l.degenerate_next = true;
                }
            } else {
                pts.pop();
            }
        }
        if pts.len() >= 2 {
            loops.push(pts);
        }
    }
    Ok(loops)
}

/// The points of one edge use in edge direction: the shared polyline mapped into UV.
fn edgeuse_points(
    k: &Kernel,
    s: &Surface,
    poly: &EdgePolyline,
    ivl: &crate::geom::extent1d::Extent1d,
    trim: Option<&Curve>,
    domain: &Extent2d,
    shared: &Shared<'_>,
) -> Result<Vec<LoopPoint>> {
    let (t0, t1) = (ivl.min(), ivl.max());
    let len = t1 - t0;
    let mut out = Vec::with_capacity(poly.params.len());
    let mut prev: Option<Point2> = None;
    for (i, (t, v)) in poly.params.iter().zip(&poly.vertices).enumerate() {
        let xyz = shared.position(*v)?;
        let sn = if len > 0.0 { ((t - t0) / len).clamp(0.0, 1.0) } else if i == 0 { 0.0 } else { 1.0 };
        let uv = match trim {
            Some(c) => {
                let ti = c.natural_interval();
                let p = c.evaluate_point(k, ti.evaluate(sn))?;
                if !(p.x.is_finite() && p.y.is_finite()) {
                    return Err(SolidError::invalid_input(OP, "UV trim curve is not finite"));
                }
                let uv = Point2::new(p.x, p.y);
                if poly.degenerate {
                    uv
                } else {
                    polish_uv(k, s, uv, xyz, domain)
                }
            }
            None => match prev {
                Some(seed) => polish_uv(k, s, seed, xyz, domain),
                None => invert_point(k, s, xyz, domain)?,
            },
        };
        prev = Some(uv);
        out.push(LoopPoint {
            uv,
            xyz,
            vertex: *v,
            degenerate_next: poly.degenerate && i == 0,
        });
    }
    Ok(out)
}

/// Whether the UV jump from `a` to `b` (one mesh vertex) is a pole: far apart in UV but
/// the surface stays at the vertex between them.
fn junction_is_pole(
    k: &Kernel,
    s: &Surface,
    opts: &TessellateOptions,
    a: LoopPoint,
    b: LoopPoint,
    span: f64,
    scaled_gap: &dyn Fn(Point2, Point2) -> f64,
) -> Result<bool> {
    let gap = scaled_gap(a.uv, b.uv);
    if gap <= opts.chord_tolerance.max(1e-9 * span) {
        return Ok(false);
    }
    let mid = Point2::new((a.uv.x + b.uv.x) / 2.0, (a.uv.y + b.uv.y) / 2.0);
    let pm = surface_point(k, s, mid)?;
    if pm.distance(a.xyz) <= opts.chord_tolerance {
        return Ok(true);
    }
    if gap > 0.1 * span {
        return Err(SolidError::invalid_input(
            OP,
            "loop does not close in UV (periodic face without seam edge)",
        ));
    }
    Ok(false)
}

/// Appends an edge use's points to a loop, merging the shared junction point.
fn join(
    pts: &mut Vec<LoopPoint>,
    seq: Vec<LoopPoint>,
    k: &Kernel,
    s: &Surface,
    opts: &TessellateOptions,
    span: f64,
    scaled_gap: &dyn Fn(Point2, Point2) -> f64,
) -> Result<()> {
    if let (Some(last), Some(first)) = (pts.last().copied(), seq.first().copied()) {
        if last.vertex != first.vertex {
            return Err(SolidError::invalid_input(OP, "loop edge uses do not connect"));
        }
        if junction_is_pole(k, s, opts, last, first, span, scaled_gap)? {
            if let Some(l) = pts.last_mut() {
                l.degenerate_next = true;
            }
        } else {
            let keep = last.degenerate_next;
            pts.pop();
            let mut seq = seq;
            if let Some(f) = seq.first_mut() {
                f.degenerate_next |= keep;
            }
            pts.extend(seq);
            return Ok(());
        }
    }
    pts.extend(seq);
    Ok(())
}

/// Interior grid cell counts `(nu, nv)` for `bbox`: the chord deviation and normal
/// turning between neighbouring samples of a sample grid, scaled to the tolerances.
fn grid_counts(
    k: &Kernel,
    s: &Surface,
    bbox: &Extent2d,
    opts: &TessellateOptions,
) -> Result<(usize, usize)> {
    let n = SAMPLE_GRID;
    let at = |i: usize, j: usize| {
        Point2::new(
            bbox.u_min() + (bbox.u_max() - bbox.u_min()) * i as f64 / n as f64,
            bbox.v_min() + (bbox.v_max() - bbox.v_min()) * j as f64 / n as f64,
        )
    };
    let need = |a: Point2, b: Point2| -> Result<f64> {
        let pa = surface_point(k, s, a)?;
        let pb = surface_point(k, s, b)?;
        let pm = surface_point(k, s, Point2::new((a.x + b.x) / 2.0, (a.y + b.y) / 2.0))?;
        let dev = segment_distance(pm, pa, pb);
        let mut m = (dev / opts.chord_tolerance).sqrt();
        if let (Some(na), Some(nb)) = (surface_normal(k, s, a), surface_normal(k, s, b)) {
            m = m.max(angle(na, nb) / opts.angle_tolerance);
        }
        if let Some(l) = opts.max_edge_length {
            m = m.max(pa.distance(pb) / l);
        }
        Ok(m)
    };
    let (mut nu, mut nv) = (0.0_f64, 0.0_f64);
    for j in 0..=n {
        let mut row = 0.0;
        for i in 0..n {
            row += need(at(i, j), at(i + 1, j))?;
        }
        nu = nu.max(row);
    }
    for i in 0..=n {
        let mut col = 0.0;
        for j in 0..n {
            col += need(at(i, j), at(i, j + 1))?;
        }
        nv = nv.max(col);
    }
    Ok((
        (nu.ceil() as usize).clamp(1, MAX_GRID),
        (nv.ceil() as usize).clamp(1, MAX_GRID),
    ))
}

/// Meshes face `f` against the shared edge polylines.
fn mesh_face(
    k: &Kernel,
    b: &Brep,
    f: FaceId,
    opts: &TessellateOptions,
    shared: &Shared<'_>,
) -> Result<FaceOutput> {
    let surface = b
        .face(f)?
        .surface
        .as_ref()
        .ok_or_else(|| SolidError::invalid_input(OP, "face without surface"))?;
    let sense = outward_sense(b, f)?;
    let mut domain = b.face(f)?.uv_domain;
    if !(domain.u_min() < domain.u_max() && domain.v_min() < domain.v_max()) {
        domain = surface.natural_uv_domain();
    }
    let scale = surface_scales(k, surface, &domain);
    let mut inverted = Vec::new();
    let loops = face_loops(k, b, f, surface, shared, opts, scale, &mut inverted)?;
    if loops.is_empty() {
        return Err(SolidError::invalid_input(OP, "face without boundary loops"));
    }

    // the loops' UV box and the interior grid
    let mut bbox = Extent2d::new();
    for p in loops.iter().flatten() {
        bbox.add_point(p.uv);
    }
    if !(bbox.u_min() < bbox.u_max() && bbox.v_min() < bbox.v_max()) {
        return Err(SolidError::invalid_input(OP, "face loops have no UV area"));
    }
    let (nu, nv) = grid_counts(k, surface, &bbox, opts)?;
    let step = (
        (bbox.u_max() - bbox.u_min()) / nu as f64,
        (bbox.v_max() - bbox.v_min()) / nv as f64,
    );
    let scaled_step = (step.0 * scale.0).min(step.1 * scale.1);

    let mut m = FaceMesher {
        k,
        surface,
        opts,
        origin: bbox.mid(),
        scale,
        sense,
        cdt: Cdt::new(),
        vertices: Vec::new(),
    };

    // boundary vertices (pole runs sampled at the grid spacing) and constraints
    for lp in &loops {
        let mut handles: Vec<FixedVertexHandle> = Vec::new();
        let count = lp.len();
        for (i, p) in lp.iter().enumerate() {
            let mut run = vec![p.uv];
            if p.degenerate_next {
                if let Some(next) = lp.get((i + 1) % count) {
                    let a = m.scaled(p.uv);
                    let c = m.scaled(next.uv);
                    let len = ((c.x - a.x).powi(2) + (c.y - a.y).powi(2)).sqrt();
                    let segs = if scaled_step > 0.0 {
                        ((len / scaled_step).ceil() as usize).clamp(1, MAX_DEGENERATE_SEGMENTS)
                    } else {
                        1
                    };
                    for j in 1..segs {
                        let w = j as f64 / segs as f64;
                        run.push(Point2::new(
                            p.uv.x + (next.uv.x - p.uv.x) * w,
                            p.uv.y + (next.uv.y - p.uv.y) * w,
                        ));
                    }
                }
            }
            for uv in run {
                let h = match m.cdt.locate(m.scaled(uv)) {
                    PositionInTriangulation::OnVertex(h) => {
                        if m.vertex(h)?.vertex != Some(p.vertex) {
                            return Err(SolidError::invalid_input(
                                OP,
                                "two boundary points at one UV position",
                            ));
                        }
                        h
                    }
                    _ => m.insert(FaceVertex {
                        uv,
                        xyz: p.xyz,
                        vertex: Some(p.vertex),
                        normal: m.outward(uv),
                    })?,
                };
                if handles.last() != Some(&h) {
                    handles.push(h);
                }
            }
        }
        if handles.first() == handles.last() && handles.len() > 1 {
            handles.pop();
        }
        let n = handles.len();
        for i in 0..n {
            let (Some(a), Some(c)) = (handles.get(i), handles.get((i + 1) % n)) else {
                continue;
            };
            if a == c || m.cdt.exists_constraint(*a, *c) {
                continue;
            }
            if m.cdt.try_add_constraint(*a, *c).is_empty() {
                return Err(SolidError::invalid_input(
                    OP,
                    "boundary crosses itself in UV",
                ));
            }
        }
    }

    // which way UV-counter-clockwise triangles face
    let flip = uv_ccw_faces_inward(k, surface, &bbox, sense)?;

    // interior grid points, away from the boundary
    let inside = inside_faces(&m.cdt);
    let clearance = 0.3 * scaled_step;
    let mut grid = Vec::new();
    for j in 1..nv {
        for i in 1..nu {
            let uv = Point2::new(
                bbox.u_min() + step.0 * i as f64,
                bbox.v_min() + step.1 * j as f64,
            );
            let pos = m.scaled(uv);
            if let PositionInTriangulation::OnFace(fh) = m.cdt.locate(pos) {
                if is_inside(&inside, fh) && constraint_clearance(&m.cdt, fh, pos) >= clearance {
                    grid.push(uv);
                }
            }
        }
    }
    for uv in grid {
        m.insert_interior(uv)?;
    }

    // refinement
    let mut converged = false;
    for _ in 0..opts.max_refine_passes {
        let candidates = refine_candidates(&m, flip)?;
        if candidates.is_empty() {
            converged = true;
            break;
        }
        let mut added = 0;
        for uv in candidates {
            if m.vertices.len() >= opts.max_face_vertices {
                break;
            }
            match m.cdt.locate(m.scaled(uv)) {
                PositionInTriangulation::OnFace(_) => {}
                PositionInTriangulation::OnEdge(e) if !m.cdt.is_constraint_edge(e.as_undirected()) => {}
                _ => continue,
            }
            m.insert_interior(uv)?;
            added += 1;
        }
        if added == 0 {
            break;
        }
    }

    // the inside triangles, wound outward
    let inside = inside_faces(&m.cdt);
    let mut triangles = Vec::new();
    for face in m.cdt.inner_faces() {
        if !is_inside(&inside, face.fix()) {
            continue;
        }
        let [a, c, d] = face.vertices().map(|v| v.data().id);
        triangles.push(if flip { [a, d, c] } else { [a, c, d] });
    }
    Ok(FaceOutput {
        vertices: m.vertices,
        triangles,
        inverted,
        converged,
    })
}

/// Whether a triangle counter-clockwise in UV faces against the outward side: the
/// sign of `(Su x Sv) . outward` where `Su x Sv` is largest on a sample grid of `bbox`.
fn uv_ccw_faces_inward(k: &Kernel, s: &Surface, bbox: &Extent2d, sense: f64) -> Result<bool> {
    let mut best: Option<(f64, f64)> = None;
    for j in 0..=4 {
        for i in 0..=4 {
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
                best = Some((l, c.dot(n) * sense));
            }
        }
    }
    match best {
        Some((l, d)) if l > 0.0 => Ok(d < 0.0),
        _ => Err(SolidError::invalid_input(OP, "surface has no normal on the face")),
    }
}

/// The UV points the next refinement pass inserts.
fn refine_candidates(m: &FaceMesher<'_>, flip: bool) -> Result<Vec<Point2>> {
    let inside = inside_faces(&m.cdt);
    let tol = m.opts.chord_tolerance;
    let mut out = Vec::new();
    for face in m.cdt.inner_faces() {
        if !is_inside(&inside, face.fix()) {
            continue;
        }
        let [va, vb, vc] = face.vertices();
        let a = m.vertex(va.fix())?;
        let b = m.vertex(vb.fix())?;
        let c = m.vertex(vc.fix())?;
        // a triangle collapsed onto a pole is dropped on output
        let same = |x: &FaceVertex, y: &FaceVertex| x.vertex.is_some() && x.vertex == y.vertex;
        if same(a, b) || same(b, c) || same(c, a) {
            continue;
        }
        let uv = Point2::new((a.uv.x + b.uv.x + c.uv.x) / 3.0, (a.uv.y + b.uv.y + c.uv.y) / 3.0);
        let mut n = (b.xyz - a.xyz).cross(c.xyz - a.xyz);
        if flip {
            n = -n;
        }
        let nl = n.length();
        let mut split = false;
        if nl > 0.0 && nl.is_finite() {
            let p = surface_point(m.k, m.surface, uv)?;
            if ((p - a.xyz).dot(n) / nl).abs() > tol {
                split = true;
            } else if let Some(o) = m.outward(uv) {
                if o.dot(n) < 0.0 {
                    split = true;
                }
            }
        }
        if !split {
            let normals = [a.normal, b.normal, c.normal];
            for (x, y) in [(0, 1), (1, 2), (2, 0)] {
                if let (Some(Some(p)), Some(Some(q))) = (normals.get(x), normals.get(y)) {
                    if angle(*p, *q) > m.opts.angle_tolerance {
                        split = true;
                    }
                }
            }
        }
        if split {
            out.push(uv);
        }
    }
    for edge in m.cdt.undirected_edges() {
        if m.cdt.is_constraint_edge(edge.fix()) {
            continue;
        }
        let d = edge.as_directed();
        let in_face = |f: Option<FixedFaceHandle<InnerTag>>| f.is_some_and(|f| is_inside(&inside, f));
        if !(in_face(d.face().as_inner().map(|f| f.fix()))
            || in_face(d.rev().face().as_inner().map(|f| f.fix())))
        {
            continue;
        }
        let [va, vb] = edge.vertices();
        let a = m.vertex(va.fix())?;
        let b = m.vertex(vb.fix())?;
        if a.vertex.is_some() && a.vertex == b.vertex {
            continue;
        }
        let uv = Point2::new((a.uv.x + b.uv.x) / 2.0, (a.uv.y + b.uv.y) / 2.0);
        let mut split = m
            .opts
            .max_edge_length
            .is_some_and(|l| a.xyz.distance(b.xyz) > l);
        if !split {
            let p = surface_point(m.k, m.surface, uv)?;
            split = segment_distance(p, a.xyz, b.xyz) > tol;
        }
        if split {
            out.push(uv);
        }
    }
    Ok(out)
}
