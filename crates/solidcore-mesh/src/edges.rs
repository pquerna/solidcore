//! Edge polylines: every B-rep edge divided once, adaptively, and shared by its faces.

use solidcore::nurbs::knots::Side;

use solidcore::curve::curve::Curve;
use solidcore::kernel::Kernel;
use solidcore::math::{Point2, Point3, Vec3};
use solidcore::surface::surface::Surface;
use solidcore::topology::edge as ed;
use solidcore::topology::edgeuse::p1 as eu1;
use solidcore::topology::Brep;
use solidcore::{EdgeId, EntityRef, FaceId, Result, VertexId};

use crate::budget::QueryBudget;
use crate::geom::{angle, curve_eval, invalid, segment_distance};
use crate::mesh::{EdgePolyline, Failure};
use crate::options::{EdgeParams, FaceParams, TessellateOptions};
use crate::storage::{vector_bytes, FlatIndex};

/// Most times one segment is halved.
pub const MAX_EDGE_DEPTH: u32 = 30;
/// Most points of one edge polyline; more is an error.
pub const MAX_EDGE_POINTS: usize = 1 << 20;
/// Edge segments smaller than this fraction of the chord tolerance are not split.
const MIN_SPLIT_FRACTION: f64 = 1e-3;
/// Most pieces one [`SegmentSplit`] divides a segment into.
pub const MAX_SPLIT_PIECES: u32 = 64;

/// A boundary segment a face asks to have divided ([`EdgeSet::split_segments`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SegmentSplit {
    /// The mesh vertices of two neighbouring points of one edge polyline (sorted).
    pub segment: [u32; 2],
    /// The equal parameter pieces it should become (at most [`MAX_SPLIT_PIECES`]).
    pub pieces: u32,
}

/// The edge polylines of a B-rep and the mesh vertices they hold: the shared input of
/// every face ([`crate::tessellate_face`]). Its positions are the mesh's first vertices:
/// the B-rep vertices, then the edges' interior points.
#[derive(Debug, Clone, Default)]
pub struct EdgeSet {
    pub(crate) positions: Vec<Point3>,
    pub(crate) vertex_index: FlatIndex<VertexId, u32>,
    pub(crate) edges: Vec<EdgePolyline>,
    pub(crate) index: FlatIndex<EdgeId, usize>,
    pub(crate) failures: Vec<Failure>,
    pub(crate) model_size: f64,
    /// Each edge's segment limit at the last division.
    pub(crate) max_segments: FlatIndex<EdgeId, u32>,
}

impl EdgeSet {
    /// Bytes of this owner and its retained backing storage, using actual vector
    /// capacities (including buffers whose lengths have been cleared).
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        let flat = vector_bytes(&self.positions)
            .saturating_add(vector_bytes(&self.edges))
            .saturating_add(vector_bytes(&self.failures))
            .saturating_add(self.vertex_index.retained_buffer_bytes())
            .saturating_add(self.index.retained_buffer_bytes())
            .saturating_add(self.max_segments.retained_buffer_bytes());
        self.edges
            .iter()
            .fold(size_of::<Self>().saturating_add(flat), |sum, edge| {
                sum.saturating_add(vector_bytes(&edge.params))
                    .saturating_add(vector_bytes(&edge.vertices))
            })
    }

    /// The mesh vertex positions so far.
    #[must_use]
    pub fn positions(&self) -> &[Point3] {
        &self.positions
    }

    /// The edge polylines, in the B-rep's edge order.
    #[must_use]
    pub fn polylines(&self) -> &[EdgePolyline] {
        &self.edges
    }

    /// The polyline of edge `e`, if it was tessellated.
    #[must_use]
    pub fn polyline(&self, e: EdgeId) -> Option<&EdgePolyline> {
        self.index.get(&e).and_then(|i| self.edges.get(*i))
    }

    pub(crate) fn polyline_with_budget(
        &self,
        e: EdgeId,
        budget: &QueryBudget<'_>,
    ) -> Result<Option<&EdgePolyline>> {
        Ok(self
            .index
            .get_with_budget(&e, budget)?
            .and_then(|i| self.edges.get(*i)))
    }

    /// Edges that could not be divided.
    #[must_use]
    pub fn failures(&self) -> &[Failure] {
        &self.failures
    }

    /// The model size relative tolerances resolve against (bounding-box diagonal).
    #[must_use]
    pub fn model_size(&self) -> f64 {
        self.model_size
    }

    pub(crate) fn position(&self, i: u32) -> Result<Point3> {
        self.positions
            .get(i as usize)
            .copied()
            .ok_or_else(|| invalid("mesh vertex out of range"))
    }

    /// Divides (or, after an earlier level, further divides) every edge of `b` by
    /// `opts`. A finer level only adds points: every earlier point stays, with its id.
    pub(crate) fn refine(&mut self, k: &Kernel, b: &Brep, opts: &TessellateOptions) -> Result<()> {
        self.refine_with_budget(k, b, opts, &QueryBudget::unbounded())
    }

    /// The caller-bounded route, including topology, evaluation and buffer growth.
    pub(crate) fn refine_with_budget(
        &mut self,
        k: &Kernel,
        b: &Brep,
        opts: &TessellateOptions,
        budget: &QueryBudget<'_>,
    ) -> Result<()> {
        let size = self.model_size;
        // `bl::edges` copies the entire source list; this borrowed order is identical.
        for &e in b.edge_list() {
            budget.charge(1)?;
            let faces = edge_faces_with_budget(b, e, budget)?;
            let mut params: Vec<FaceParams> = Vec::new();
            budget.reserve_vec(&mut params, faces.len())?;
            for f in &faces {
                budget.charge(1)?;
                budget.push(&mut params, opts.face_params(f.face, size))?;
            }
            budget.charge(1_u64.saturating_add(params.len() as u64))?;
            let p = opts.edge_params(e, size, &params);
            budget.checkpoint()?;
            self.max_segments
                .insert_with_budget(e, p.max_segments, budget)?;
            let slot = self.index.get_with_budget(&e, budget)?.copied();
            let poly = slot
                .and_then(|i| self.edges.get(i))
                .map(|poly| clone_polyline(poly, budget))
                .transpose()?;
            let result = match poly {
                Some(poly) => self.refine_edge(k, b, &faces, &p, poly, budget),
                None => self.first_edge(k, b, e, &faces, &p, budget),
            };
            // Resource/cancellation errors are sticky, never per-edge failures.
            budget.checkpoint()?;
            match (result, slot) {
                (Ok(poly), Some(i)) => {
                    if let Some(s) = self.edges.get_mut(i) {
                        *s = poly;
                    }
                }
                (Ok(poly), None) => {
                    self.index.insert_with_budget(e, self.edges.len(), budget)?;
                    budget.push(&mut self.edges, poly)?;
                }
                (Err(error), _) => {
                    let entity = EntityRef::Edge(e);
                    let mut found = false;
                    for failure in &self.failures {
                        budget.charge(1)?;
                        if failure.entity == entity {
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        budget.push(&mut self.failures, Failure { entity, error })?;
                    }
                }
            }
        }
        budget.checkpoint()
    }

    /// Splits each polyline segment of `segments` (a pair of neighbouring points of one
    /// edge polyline, in either order; the most pieces asked when one is listed twice)
    /// into equal parameter pieces, keeping every other point; returns the edges
    /// changed, in id order. Pairs that are no polyline segment are left alone, and an
    /// edge stops gaining points at its segment limit.
    ///
    /// # Errors
    ///
    /// The curve evaluation's errors.
    pub fn split_segments(
        &mut self,
        k: &Kernel,
        b: &Brep,
        segments: &[SegmentSplit],
    ) -> Result<Vec<EdgeId>> {
        self.split_segments_with_budget(k, b, segments, &QueryBudget::unbounded())
    }

    /// Caller-bounded segment splitting, with the same ordering and limits.
    ///
    /// # Errors
    ///
    /// The curve evaluation's errors, or exhaustion/cancellation of `budget`.
    pub(crate) fn split_segments_with_budget(
        &mut self,
        k: &Kernel,
        b: &Brep,
        segments: &[SegmentSplit],
        budget: &QueryBudget<'_>,
    ) -> Result<Vec<EdgeId>> {
        let mut wanted: FlatIndex<[u32; 2], u32> = FlatIndex::default();
        for r in segments {
            budget.charge(1)?;
            let [x, y] = r.segment;
            let key = if x < y { [x, y] } else { [y, x] };
            let n = wanted
                .get_with_budget(&key, budget)?
                .copied()
                .unwrap_or(0)
                .max(r.pieces.min(MAX_SPLIT_PIECES));
            wanted.insert_with_budget(key, n, budget)?;
        }
        let mut changed = Vec::new();
        for ei in 0..self.edges.len() {
            budget.charge(1)?;
            let Some(poly) = self.edges.get(ei) else {
                continue;
            };
            if poly.degenerate {
                continue;
            }
            let mut segs = Vec::new();
            for (si, w) in poly.vertices.windows(2).enumerate() {
                budget.charge(1)?;
                if let [x, y] = w {
                    let key = if x < y { [*x, *y] } else { [*y, *x] };
                    if let Some(n) = wanted.get_with_budget(&key, budget)? {
                        budget.push(&mut segs, (si, *n))?;
                    }
                }
            }
            if segs.is_empty() {
                continue;
            }
            let poly = clone_polyline(poly, budget)?;
            let rec = b.edge(poly.edge)?;
            let curve = rec
                .curve
                .as_ref()
                .ok_or_else(|| invalid("edge without curve"))?;
            let mut poly = poly;
            let limit = self
                .max_segments
                .get_with_budget(&poly.edge, budget)?
                .copied()
                .unwrap_or(u32::MAX) as usize;
            // Back to front, so earlier indices stay valid.
            let before = poly.params.len();
            for (si, n) in segs.into_iter().rev() {
                budget.charge(1)?;
                let (Some(&t0), Some(&t1)) = (poly.params.get(si), poly.params.get(si + 1)) else {
                    continue;
                };
                // the new points in order after point si
                for i in 1..n {
                    budget.charge(1)?;
                    let tm = t0 + (t1 - t0) * f64::from(i) / f64::from(n);
                    let prev = poly.params.get(si + i as usize - 1).copied().unwrap_or(t0);
                    if !(tm > prev && tm < t1)
                        || poly.params.len() >= MAX_EDGE_POINTS
                        || poly.params.len() > limit
                    {
                        break;
                    }
                    let (p, _) = curve_eval_with_budget(k, curve, tm, false, budget)?;
                    let v = push_position_with_budget(&mut self.positions, p, budget)?;
                    budget.reserve_vec(&mut poly.params, 1)?;
                    budget.reserve_vec(&mut poly.vertices, 1)?;
                    budget.charge(poly.params.len().saturating_sub(si + i as usize) as u64)?;
                    poly.params.insert(si + i as usize, tm);
                    poly.vertices.insert(si + i as usize, v);
                }
            }
            if poly.params.len() == before {
                continue;
            }
            budget.push(&mut changed, poly.edge)?;
            if let Some(slot) = self.edges.get_mut(ei) {
                *slot = poly;
            }
        }
        budget.charge(
            (changed.len() as u64)
                .saturating_mul(u64::from(usize::BITS - changed.len().leading_zeros())),
        )?;
        changed.sort_unstable();
        Ok(changed)
    }

    fn vertex_slot(&mut self, b: &Brep, v: VertexId, budget: &QueryBudget<'_>) -> Result<u32> {
        budget.charge(1)?;
        if let Some(i) = self.vertex_index.get_with_budget(&v, budget)? {
            return Ok(*i);
        }
        let i = push_position_with_budget(&mut self.positions, b.vertex(v)?.point, budget)?;
        self.vertex_index.insert_with_budget(v, i, budget)?;
        Ok(i)
    }

    /// The first division of edge `e`.
    fn first_edge(
        &mut self,
        k: &Kernel,
        b: &Brep,
        e: EdgeId,
        faces: &[EdgeFace<'_>],
        p: &EdgeParams,
        budget: &QueryBudget<'_>,
    ) -> Result<EdgePolyline> {
        budget.charge(1)?;
        let (v0, v1) = ed::start_end_vertices(b, e)?;
        let (Some(v0), Some(v1)) = (v0, v1) else {
            return Err(invalid("edge without vertices"));
        };
        let i0 = self.vertex_slot(b, v0, budget)?;
        let i1 = self.vertex_slot(b, v1, budget)?;
        let rec = b.edge(e)?;
        let ivl = rec.interval;
        let (t0, t1) = (ivl.min(), ivl.max());
        let Some(curve) = rec.curve.as_ref() else {
            return if v0 == v1 {
                endpoint_polyline(e, [t0, t1], [i0, i1], budget)
            } else {
                Err(invalid("edge without curve"))
            };
        };
        if !(t0.is_finite() && t1.is_finite() && t0 < t1) {
            return if v0 == v1 {
                endpoint_polyline(e, [t0, t1], [i0, i1], budget)
            } else {
                Err(invalid("edge with an empty interval"))
            };
        }
        // a closed edge whose curve stays at its vertex is a pole
        if v0 == v1 {
            let pv = b.vertex(v0)?.point;
            let tol = p.chord.min(rec.tolerance.max(0.0) * 10.0).max(1e-12);
            let mut collapsed = true;
            for i in 0..=16 {
                budget.charge(1)?;
                let t = ivl.evaluate(f64::from(i) / 16.0);
                if curve_eval_with_budget(k, curve, t, i == 16, budget)?
                    .0
                    .distance(pv)
                    > tol
                {
                    collapsed = false;
                    break;
                }
            }
            if collapsed {
                return endpoint_polyline(e, [t0, t1], [i0, i1], budget);
            }
        }
        let n0 = p.min_segments.max(if v0 == v1 { 2 } else { 1 });
        let mut params = Vec::new();
        let mut vertices = Vec::new();
        for i in 0..=n0 {
            budget.charge(1)?;
            budget.push(&mut params, ivl.evaluate(f64::from(i) / f64::from(n0)))?;
            budget.push(
                &mut vertices,
                if i == 0 {
                    i0
                } else if i == n0 {
                    i1
                } else {
                    u32::MAX
                },
            )?;
        }
        // the inner starting points become vertices as they are kept
        for (t, v) in params.iter().zip(vertices.iter_mut()) {
            budget.charge(1)?;
            if *v == u32::MAX {
                let (pt, _) = curve_eval_with_budget(k, curve, *t, false, budget)?;
                *v = push_position_with_budget(&mut self.positions, pt, budget)?;
            }
        }
        let poly = EdgePolyline {
            edge: e,
            params,
            vertices,
            degenerate: false,
        };
        self.split(k, curve, (t0, t1), faces, p, &poly, budget)
    }

    /// A further division of an edge already divided.
    fn refine_edge(
        &mut self,
        k: &Kernel,
        b: &Brep,
        faces: &[EdgeFace<'_>],
        p: &EdgeParams,
        poly: EdgePolyline,
        budget: &QueryBudget<'_>,
    ) -> Result<EdgePolyline> {
        budget.charge(1)?;
        if poly.degenerate {
            return Ok(poly);
        }
        let rec = b.edge(poly.edge)?;
        let curve = rec
            .curve
            .as_ref()
            .ok_or_else(|| invalid("edge without curve"))?;
        let ivl = rec.interval;
        self.split(k, curve, (ivl.min(), ivl.max()), faces, p, &poly, budget)
    }

    /// Splits every segment of `poly` until it meets `p`, keeping every point.
    fn split(
        &mut self,
        k: &Kernel,
        curve: &Curve,
        interval: (f64, f64),
        faces: &[EdgeFace<'_>],
        p: &EdgeParams,
        poly: &EdgePolyline,
        budget: &QueryBudget<'_>,
    ) -> Result<EdgePolyline> {
        let n = poly.params.len();
        let mut params = Vec::new();
        let mut vertices = Vec::new();
        budget.reserve_vec(&mut params, n)?;
        budget.reserve_vec(&mut vertices, n)?;
        let mut segments = n.saturating_sub(1);
        let last = n.saturating_sub(1);
        let Some((&t_first, &v_first)) = poly.params.first().zip(poly.vertices.first()) else {
            return Err(invalid("edge polyline without points"));
        };
        let (_, mut da) = curve_eval_with_budget(k, curve, t_first, last == 0, budget)?;
        let mut ta = t_first;
        let mut pa = self.position(v_first)?;
        budget.push(&mut params, t_first)?;
        budget.push(&mut vertices, v_first)?;
        let mut stack: Vec<Segment> = Vec::new();
        // Keep just the preceding endpoint, not a source-sized tangent buffer.
        for (i, (&tb, &vb)) in poly.params.iter().zip(&poly.vertices).enumerate().skip(1) {
            budget.charge(1)?;
            let (_, db) = curve_eval_with_budget(k, curve, tb, i == last, budget)?;
            let pb = self.position(vb)?;
            budget.push(
                &mut stack,
                Segment {
                    t0: ta,
                    t1: tb,
                    p0: pa,
                    p1: pb,
                    d0: da,
                    d1: db,
                    v1: Some(vb),
                    mid: None,
                    depth: 0,
                },
            )?;
            (ta, pa, da) = (tb, pb, db);
            // process this original segment fully before the next, in curve order
            while let Some(s) = stack.pop() {
                budget.charge(1)?;
                let (tm, mid) = if let Some(m) = s.mid {
                    (m.0, (m.1, m.2))
                } else {
                    let tm = (s.t0 + s.t1) / 2.0;
                    let (pm, dm) = curve_eval_with_budget(k, curve, tm, false, budget)?;
                    (tm, (pm, dm))
                };
                // a segment far below the chord tolerance is left alone (a tangent or
                // normal still turning there is a singular point's, as at a fillet's
                // collapsed corner)
                let size =
                    s.p0.distance(s.p1)
                        .max(s.p0.distance(mid.0))
                        .max(s.p1.distance(mid.0));
                let can = s.depth < MAX_EDGE_DEPTH
                    && segments < p.max_segments as usize
                    && s.p0.distance(s.p1) >= 2.0 * p.min_len
                    && size >= MIN_SPLIT_FRACTION * p.chord;
                let quarters = if can {
                    Some(Self::needs_split(
                        k, curve, &s, mid.0, interval, faces, p, budget,
                    )?)
                } else {
                    None
                };
                if let Some(Some((q1, q3))) = quarters {
                    segments += 1;
                    budget.push(
                        &mut stack,
                        Segment {
                            t0: tm,
                            t1: s.t1,
                            p0: mid.0,
                            p1: s.p1,
                            d0: mid.1,
                            d1: s.d1,
                            v1: s.v1,
                            mid: Some(q3),
                            depth: s.depth + 1,
                        },
                    )?;
                    budget.push(
                        &mut stack,
                        Segment {
                            t0: s.t0,
                            t1: tm,
                            p0: s.p0,
                            p1: mid.0,
                            d0: s.d0,
                            d1: mid.1,
                            v1: None,
                            mid: Some(q1),
                            depth: s.depth + 1,
                        },
                    )?;
                    continue;
                }
                let v = match s.v1 {
                    Some(v) => v,
                    None => push_position_with_budget(&mut self.positions, s.p1, budget)?,
                };
                budget.push(&mut params, s.t1)?;
                budget.push(&mut vertices, v)?;
                if params.len() > MAX_EDGE_POINTS {
                    return Err(invalid("edge needs too many points"));
                }
            }
        }
        Ok(EdgePolyline {
            edge: poly.edge,
            params,
            vertices,
            degenerate: false,
        })
    }

    /// When segment `s` (midpoint `pm`) is outside the tolerances, its two quarter
    /// points (the midpoints of its halves); `None` when it is within them. The chord
    /// is checked at the quarter points and the midpoint; the curve's tangent and every
    /// face's normal must not turn more than the angle tolerance.
    #[allow(clippy::type_complexity)]
    fn needs_split(
        k: &Kernel,
        curve: &Curve,
        s: &Segment,
        pm: Point3,
        interval: (f64, f64),
        faces: &[EdgeFace<'_>],
        p: &EdgeParams,
        budget: &QueryBudget<'_>,
    ) -> Result<Option<((f64, Point3, Vec3), (f64, Point3, Vec3))>> {
        let t1q = s.t0 + (s.t1 - s.t0) * 0.25;
        let t3q = s.t0 + (s.t1 - s.t0) * 0.75;
        let (p1q, d1q) = curve_eval_with_budget(k, curve, t1q, false, budget)?;
        let (p3q, d3q) = curve_eval_with_budget(k, curve, t3q, false, budget)?;
        let quarters = Some(((t1q, p1q, d1q), (t3q, p3q, d3q)));
        if let Some(l) = p.max_len {
            if s.p0.distance(s.p1) > l {
                return Ok(quarters);
            }
        }
        if angle(s.d0, s.d1) > p.angle {
            return Ok(quarters);
        }
        for q in [p1q, pm, p3q] {
            budget.charge(1)?;
            if segment_distance(q, s.p0, s.p1) > p.chord {
                return Ok(quarters);
            }
        }
        let len = interval.1 - interval.0;
        for face in faces {
            budget.charge(1)?;
            let normal = |t: f64| -> Result<Option<Vec3>> {
                budget.charge(1)?;
                let ti = face.trim.natural_interval();
                let q = face
                    .trim
                    .evaluate_point(k, ti.evaluate((t - interval.0) / len));
                budget.checkpoint()?;
                let Ok(q) = q else {
                    return Ok(None);
                };
                surface_normal_with_budget(k, face.surface, Point2::new(q.x, q.y), budget)
            };
            if let (Some(a), Some(c)) = (normal(s.t0)?, normal(s.t1)?) {
                if angle(a, c) > p.angle {
                    return Ok(quarters);
                }
            }
        }
        Ok(None)
    }
}

pub(crate) fn push_position(positions: &mut Vec<Point3>, p: Point3) -> Result<u32> {
    push_position_with_budget(positions, p, &QueryBudget::unbounded())
}

pub(crate) fn push_position_with_budget(
    positions: &mut Vec<Point3>,
    p: Point3,
    budget: &QueryBudget<'_>,
) -> Result<u32> {
    budget.charge(1)?;
    let i =
        u32::try_from(positions.len()).map_err(|_| invalid("more than u32::MAX mesh vertices"))?;
    budget.push(positions, p)?;
    Ok(i)
}

/// One pending piece of an edge's division; its midpoint is cached when a parent
/// evaluated it as a quarter point.
#[derive(Debug, Clone, Copy)]
struct Segment {
    t0: f64,
    t1: f64,
    p0: Point3,
    p1: Point3,
    d0: Vec3,
    d1: Vec3,
    /// The mesh vertex at `t1`, when it exists already.
    v1: Option<u32>,
    mid: Option<(f64, Point3, Vec3)>,
    depth: u32,
}

/// A face along an edge: its surface and the edge use's UV trim curve.
pub(crate) struct EdgeFace<'a> {
    pub face: FaceId,
    pub surface: &'a Surface,
    pub trim: &'a Curve,
}

/// The faces along edge `e` with a surface and a UV trim curve for it, once each.
pub(crate) fn edge_faces_with_budget<'a>(
    b: &'a Brep,
    e: EdgeId,
    budget: &QueryBudget<'_>,
) -> Result<Vec<EdgeFace<'a>>> {
    let mut out: Vec<EdgeFace<'a>> = Vec::new();
    for eu in &b.edge(e)?.edgeuses {
        budget.charge(1)?;
        let Some(f) = eu1::face(b, *eu)? else {
            continue;
        };
        // The mate lookup inside uv_trim_curve linearly searches its edge's list.
        // Preadmit its entire possible scan before entering that kernel operation.
        if b.edgeuse(*eu)?.uv_trim_curve.is_none() {
            if let Some(edge) = b.edgeuse(*eu)?.edge {
                budget.charge(b.edge(edge)?.edgeuses.len() as u64)?;
            }
        }
        let (Some(trim), _) = eu1::uv_trim_curve(b, *eu)? else {
            continue;
        };
        let Some(surface) = b.face(f)?.surface.as_ref() else {
            continue;
        };
        let mut seen = false;
        for prior in &out {
            budget.charge(1)?;
            if prior.face == f {
                seen = true;
                break;
            }
        }
        if !seen {
            budget.push(
                &mut out,
                EdgeFace {
                    face: f,
                    surface,
                    trim,
                },
            )?;
        }
    }
    budget.checkpoint()?;
    Ok(out)
}

fn endpoint_polyline(
    edge: EdgeId,
    params: [f64; 2],
    vertices: [u32; 2],
    budget: &QueryBudget<'_>,
) -> Result<EdgePolyline> {
    let mut poly = EdgePolyline {
        edge,
        params: Vec::new(),
        vertices: Vec::new(),
        degenerate: true,
    };
    for (t, v) in params.into_iter().zip(vertices) {
        budget.charge(1)?;
        budget.push(&mut poly.params, t)?;
        budget.push(&mut poly.vertices, v)?;
    }
    Ok(poly)
}

fn clone_polyline(poly: &EdgePolyline, budget: &QueryBudget<'_>) -> Result<EdgePolyline> {
    let mut params = Vec::new();
    let mut vertices = Vec::new();
    budget.reserve_vec(&mut params, poly.params.len())?;
    budget.reserve_vec(&mut vertices, poly.vertices.len())?;
    for &t in &poly.params {
        budget.charge(1)?;
        budget.push(&mut params, t)?;
    }
    for &v in &poly.vertices {
        budget.charge(1)?;
        budget.push(&mut vertices, v)?;
    }
    Ok(EdgePolyline {
        edge: poly.edge,
        params,
        vertices,
        degenerate: poly.degenerate,
    })
}

/// Kernel evaluators own opaque scratch/derivative vectors; their allocations
/// cannot be admitted through this crate's visible-buffer budget. Each call is
/// charged and checked on both sides, not mistaken for an interruptible kernel.
fn curve_eval_with_budget(
    k: &Kernel,
    curve: &Curve,
    t: f64,
    from_left: bool,
    budget: &QueryBudget<'_>,
) -> Result<(Point3, Vec3)> {
    budget.charge(1)?;
    let result = curve_eval(k, curve, t, from_left);
    budget.checkpoint()?;
    result
}

fn surface_normal_with_budget(
    k: &Kernel,
    s: &Surface,
    uv: Point2,
    budget: &QueryBudget<'_>,
) -> Result<Option<Vec3>> {
    budget.charge(1)?;
    let normal = s.evaluate_normal(k, uv, Side::Right, Side::Right);
    budget.checkpoint()?;
    if let Ok(n) = normal {
        let l = n.length();
        if l.is_finite() && l > 0.5 {
            return Ok(Some(n / l));
        }
    }
    budget.charge(1)?;
    let derivatives = s.evaluate_1st_derivatives(k, uv, false, false);
    budget.checkpoint()?;
    let Ok((_, su, sv)) = derivatives else {
        return Ok(None);
    };
    let c = su.cross(sv);
    let l = c.length();
    Ok((l.is_finite() && l > 0.0).then(|| c / l))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::QueryLimit;
    use solidcore::primitive::api_primitives;
    use std::sync::atomic::AtomicBool;

    fn box_brep(k: &mut Kernel) -> Brep {
        api_primitives::create_box(k, Point3::new(0.0, 0.0, 0.0), 5.0, 10.0, 20.0).unwrap()
    }

    #[test]
    fn bounded_refine_propagates_limits_instead_of_recording_edge_failures() {
        let mut k = Kernel::default();
        let b = box_brep(&mut k);
        let cancel = AtomicBool::new(false);
        let budget = QueryBudget::new(100, usize::MAX, &cancel);
        let mut opts = TessellateOptions::default();
        opts.edge.min_segments = 1000;
        let mut edges = EdgeSet {
            model_size: 25.0,
            ..EdgeSet::default()
        };
        assert!(edges.refine_with_budget(&k, &b, &opts, &budget).is_err());
        assert_eq!(budget.failure(), Some(QueryLimit::Work));
        assert!(edges.failures().is_empty());
        assert!(edges.polylines().is_empty());
    }

    #[test]
    fn bounded_refine_refuses_cancelled_and_unfunded_queries() {
        let mut k = Kernel::default();
        let b = box_brep(&mut k);
        let cancel = AtomicBool::new(true);
        let budget = QueryBudget::new(u64::MAX, usize::MAX, &cancel);
        let mut edges = EdgeSet::default();
        let empty_bytes = edges.retained_bytes();
        assert!(edges
            .refine_with_budget(&k, &b, &TessellateOptions::default(), &budget)
            .is_err());
        assert_eq!(budget.failure(), Some(QueryLimit::Cancelled));
        assert_eq!(edges.retained_bytes(), empty_bytes);

        let cancel = AtomicBool::new(false);
        let budget = QueryBudget::new(u64::MAX, 0, &cancel);
        assert!(edges
            .refine_with_budget(&k, &b, &TessellateOptions::default(), &budget)
            .is_err());
        assert_eq!(budget.failure(), Some(QueryLimit::Bytes));
        assert_eq!(edges.retained_bytes(), empty_bytes);
        assert!(edges.failures().is_empty());
    }

    #[test]
    fn bounded_edges_match_legacy_and_admit_retained_buffers() {
        let mut k = Kernel::default();
        let b = box_brep(&mut k);
        let opts = TessellateOptions::default();
        let mut legacy = EdgeSet {
            model_size: 25.0,
            ..EdgeSet::default()
        };
        legacy.refine(&k, &b, &opts).unwrap();
        let mut bounded = EdgeSet {
            model_size: 25.0,
            ..EdgeSet::default()
        };
        let cancel = AtomicBool::new(false);
        let budget = QueryBudget::new(u64::MAX, usize::MAX, &cancel);
        bounded.refine_with_budget(&k, &b, &opts, &budget).unwrap();
        assert_eq!(bounded.positions(), legacy.positions());
        assert_eq!(bounded.polylines().len(), legacy.polylines().len());
        for (a, b) in bounded.polylines().iter().zip(legacy.polylines()) {
            assert_eq!(a.edge, b.edge);
            assert_eq!(a.params, b.params);
            assert_eq!(a.vertices, b.vertices);
            assert_eq!(a.degenerate, b.degenerate);
        }
        assert!(budget.consumed() > 0);
        assert!(
            budget.admitted_bytes() >= bounded.retained_bytes() - std::mem::size_of::<EdgeSet>()
        );
        let retained = bounded.retained_bytes();
        assert!(bounded.vertex_index.retained_buffer_bytes() > 0);
        assert!(bounded.index.retained_buffer_bytes() > 0);
        assert!(bounded.max_segments.retained_buffer_bytes() > 0);
        bounded.positions.clear();
        bounded.vertex_index.clear();
        bounded.index.clear();
        bounded.max_segments.clear();
        for poly in &mut bounded.edges {
            poly.params.clear();
            poly.vertices.clear();
        }
        assert_eq!(bounded.retained_bytes(), retained);
    }

    #[test]
    fn bounded_segment_splitting_matches_legacy_duplicate_requests() {
        let mut k = Kernel::default();
        let b = box_brep(&mut k);
        let mut legacy = EdgeSet {
            model_size: 25.0,
            ..EdgeSet::default()
        };
        legacy
            .refine(&k, &b, &TessellateOptions::default())
            .unwrap();
        let mut bounded = legacy.clone();
        let poly = legacy.polylines().first().unwrap();
        let segment = [poly.vertices[0], poly.vertices[1]];
        let requests = [
            SegmentSplit { segment, pieces: 2 },
            SegmentSplit {
                segment: [segment[1], segment[0]],
                pieces: 4,
            },
        ];
        let expected = legacy.split_segments(&k, &b, &requests).unwrap();
        let cancel = AtomicBool::new(false);
        let budget = QueryBudget::new(u64::MAX, usize::MAX, &cancel);
        let changed = bounded
            .split_segments_with_budget(&k, &b, &requests, &budget)
            .unwrap();
        assert_eq!(changed, expected);
        assert_eq!(bounded.positions(), legacy.positions());
        for (a, b) in bounded.polylines().iter().zip(legacy.polylines()) {
            assert_eq!(a.params, b.params);
            assert_eq!(a.vertices, b.vertices);
        }
    }
}
