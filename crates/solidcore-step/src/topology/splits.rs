//! One face split in two along an edge that runs off both. Some exporters cut a face on a
//! periodic surface into two faces on the same surface record along an edge that joins two of
//! its corners the wrong way round the axis: a thread's narrow helical band, cut along the
//! straight ruling between a corner of one turn and a corner of the next. Each half is then
//! bounded by a loop that winds once around an open surface (no region of the surface lies
//! inside it, and it has no pole to close on), while the two together are a band whose loop winds
//! nowhere. The material is that band; only the cut is wrong. The cut is laid again inside the
//! band, from the same vertex to the same vertex, on the surface: from each end into the band
//! along its corner's bisector, halfway across, and straight in (u, v) between those two points
//! (a thin band's middle), so neither face comes to a needle point. Both faces keep their loops,
//! edges and vertices; the solid's notes count the cuts laid again.
//!
//! Origin: SolidCraft's STEP reader as vendored by Spiling (`crates/step/src/topology/splits.rs`,
//! MIT OR Apache-2.0), seen on an Inventor modelled thread whose 0.027 mm root band is written so.

use crate::vec::Vec3Geom as _;

use crate::kernel::{CurveEval, SurfaceEval, interpolate};
use solidcore_record::Vec3;
use solidcore_record::brep::Curve as BCurve;

use super::{Builder, LoopB, R};

/// Samples per edge when following a loop around the axis.
const SAMPLES: usize = 16;
/// Pieces a cut starts with along its length, and the most one piece of it may take.
const CUT_PIECES: usize = 64;
const MAX_CUT_PIECES: usize = 512;

/// `x` wrapped into (−period/2, period/2].
fn wrap_by(x: f64, period: f64) -> f64 {
    x - (x / period).round() * period
}

/// A path's points in (u, v), unwrapped along it from `from` (or wherever its first point
/// projects), with their points in space.
struct Lifted {
    uv: Vec<[f64; 2]>,
    xyz: Vec<Vec3>,
    /// The last point's (u, v) as projected, before unwrapping.
    raw: Option<[f64; 2]>,
}

/// Where a use of an edge is: (face, loop, position) and the use's direction.
type UseAt = (usize, usize, usize, bool);

impl Builder<'_, '_> {
    /// Lay again the cut that splits face `fi`'s loop `li` (winding once around its open
    /// surface) from the face across it, if one edge of that loop is shared with exactly one
    /// other face on the same surface record and the two loops together wind nowhere. Returns
    /// whether it did.
    pub(super) fn relay_split(&mut self, fi: usize, li: usize, eval: &SurfaceEval, periods: [Option<f64>; 2]) -> R<bool> {
        let Some(face) = self.faces.get(fi) else { return Err("face reference".into()) };
        let Some(LoopB::Edges(own)) = face.loops.get(li) else { return Ok(false) };
        let own = own.clone();
        for (k, &(e, forward)) in own.iter().enumerate() {
            // Every use of the edge: here once, and once in one other face on the same surface.
            let mut others: Vec<UseAt> = Vec::new();
            for (gi, g) in self.faces.iter().enumerate() {
                for (gl, l) in g.loops.iter().enumerate() {
                    let LoopB::Edges(es) = l else { continue };
                    for (gk, u) in es.iter().enumerate() {
                        if u.0 == e && (gi, gl, gk) != (fi, li, k) {
                            others.push((gi, gl, gk, u.1));
                        }
                    }
                }
            }
            let [(gi, gl, gk, back)] = others.as_slice() else { continue };
            let (Some(face), Some(other)) = (self.faces.get(fi), self.faces.get(*gi)) else { continue };
            if *gi == fi || *back == forward || other.surface != face.surface {
                continue;
            }
            let Some(LoopB::Edges(theirs)) = other.loops.get(*gl) else { continue };
            // The band's loop: this face's loop after the cut, then the other's after it.
            let after = |es: &[(u32, bool)], at: usize| -> Vec<(u32, bool)> { es.iter().skip(at + 1).chain(es.iter().take(at)).copied().collect() };
            let mine = after(&own, k);
            let yours = after(theirs, *gk);
            let edge = self.edges.get(e as usize).ok_or("edge reference")?;
            let [v0, v1] = edge.vertices;
            // The cut runs from `p` to `q` in this face's loop.
            let (p, q) = if forward { (v0, v1) } else { (v1, v0) };
            if p == q {
                continue;
            }
            let first = self.lift(eval, &mine, periods, None)?;
            let Some(&p_uv) = first.uv.last() else { continue };
            let second = self.lift(eval, &yours, periods, first.raw.map(|r| (r, p_uv)))?;
            let (Some(&q_uv), Some(&back_uv)) = (first.uv.first(), second.uv.last()) else { continue };
            // The two loops together must wind nowhere: the band closes in (u, v).
            let winds = periods.iter().zip(back_uv.iter().zip(&q_uv)).any(|(per, (b, q))| per.is_some_and(|per| ((b - q) / per).round() != 0.0));
            if winds {
                continue;
            }
            let band = Lifted {
                uv: first.uv.iter().chain(&second.uv).copied().collect(),
                xyz: first.xyz.iter().chain(&second.xyz).copied().collect(),
                raw: None,
            };
            let Some(from_p) = self.cut(eval, &band, first.uv.len().saturating_sub(1), q_uv)? else { continue };
            // Keep the edge's own direction: its curve runs from its first vertex to its second.
            let curve = if v0 == p { from_p } else { reverse_spline(from_p) };
            let c = CurveEval::new(&curve).map_err(|m| m.to_string())?;
            for (t, v) in [(0.0, v0), (1.0, v1)] {
                let gap = c.point(t).ok_or("cut point")?.dist(self.vpoint(v)?);
                if let Some(x) = self.vertices.get_mut(v as usize) {
                    x.tolerance = x.tolerance.max(1.5 * gap);
                }
            }
            let edge = self.edges.get_mut(e as usize).ok_or("edge reference")?;
            edge.curve = curve;
            edge.interval = [0.0, 1.0];
            edge.form = None;
            return Ok(true);
        }
        Ok(false)
    }

    /// The uses' points in (u, v), unwrapped along them from `from` (the point before them, as
    /// projected and as unwrapped).
    fn lift(&self, eval: &SurfaceEval, uses: &[(u32, bool)], periods: [Option<f64>; 2], from: Option<([f64; 2], [f64; 2])>) -> R<Lifted> {
        let mut out = Lifted { uv: Vec::new(), xyz: Vec::new(), raw: None };
        let mut last: Option<([f64; 2], [f64; 2])> = from;
        for &(e, forward) in uses {
            let edge = self.edges.get(e as usize).ok_or("edge reference")?;
            let c = CurveEval::new(&edge.curve).map_err(|m| m.to_string())?;
            let [lo, hi] = edge.interval;
            for k in 0..=SAMPLES {
                let s = k as f64 / SAMPLES as f64;
                let t = if forward { lo + (hi - lo) * s } else { hi - (hi - lo) * s };
                let p = c.point(t).ok_or("edge point")?;
                let Some(raw) = eval.project(p, last.map(|l| l.0), f64::INFINITY) else { continue };
                let uv = match last {
                    Some((r, l)) => {
                        let step = |d: usize| {
                            let (x, rd, ld) =
                                (raw.get(d).copied().unwrap_or(0.0), r.get(d).copied().unwrap_or(0.0), l.get(d).copied().unwrap_or(0.0));
                            ld + periods.get(d).copied().flatten().map_or(x - rd, |per| wrap_by(x - rd, per))
                        };
                        [step(0), step(1)]
                    }
                    None => raw,
                };
                last = Some((raw, uv));
                out.raw = Some(raw);
                out.uv.push(uv);
                out.xyz.push(p);
            }
        }
        Ok(out)
    }

    /// The cut from the band's corner `at` to its first point `b`, on the surface: into the band
    /// from each end along its corner's bisector, halfway to where that leaves the band, and
    /// straight in (u, v) between; cubics through points of each piece, within half the
    /// tolerance of it, joined at the turns. Inside `band` (or within the tolerance of its
    /// boundary) all the way, or nothing. Returns the curve from the corner on [0, 1].
    fn cut(&self, eval: &SurfaceEval, band: &Lifted, at: usize, b: [f64; 2]) -> R<Option<BCurve>> {
        let ring = &band.uv;
        let n = ring.len();
        let Some(&a) = ring.get(at) else { return Ok(None) };
        // (u, v) scaled to lengths on the surface at the corner.
        let scale = match eval.derivs(a[0], a[1]) {
            Some((_, su, sv)) if su.len() > 0.0 && sv.len() > 0.0 => [su.len(), sv.len()],
            _ => return Ok(None),
        };
        let inward = |corner: usize, before: [f64; 2], after: [f64; 2]| -> Option<[f64; 2]> {
            let c = *ring.get(corner)?;
            let unit = |d: [f64; 2]| {
                let d = [d[0] * scale[0], d[1] * scale[1]];
                let l = d[0].hypot(d[1]);
                (l > 0.0).then(|| [d[0] / l, d[1] / l])
            };
            let (i, o) = (unit([before[0] - c[0], before[1] - c[1]])?, unit([after[0] - c[0], after[1] - c[1]])?);
            let mut d = [i[0] + o[0], i[1] + o[1]];
            if d[0].hypot(d[1]) < 1e-9 {
                d = [-o[1], o[0]];
            }
            let l = d[0].hypot(d[1]);
            let mut d = [d[0] / l / scale[0], d[1] / l / scale[1]];
            // Into the band: the far side of a reflex corner's bisector.
            let probe = |d: [f64; 2], t: f64| [c[0] + d[0] * t, c[1] + d[1] * t];
            let reach = exit(ring, c, d, corner)?;
            if !inside(ring, probe(d, 0.5 * reach)) {
                d = [-d[0], -d[1]];
            }
            let reach = exit(ring, c, d, corner)?;
            Some(probe(d, 0.5 * reach))
        };
        // The corner's neighbours along the band, past its repeated point.
        let (Some(&before_a), Some(&after_a)) = (at.checked_sub(1).and_then(|i| ring.get(i)), ring.get(at + 2)) else { return Ok(None) };
        let (Some(&before_b), Some(&after_b)) = (n.checked_sub(2).and_then(|i| ring.get(i)), ring.get(1)) else { return Ok(None) };
        let (Some(ma), Some(mb)) = (inward(at, before_a, after_a), inward(0, before_b, after_b)) else { return Ok(None) };
        let path = [a, ma, mb, b];
        let length = |p: [f64; 2], q: [f64; 2]| ((q[0] - p[0]) * scale[0]).hypot((q[1] - p[1]) * scale[1]);
        let legs: Vec<([f64; 2], [f64; 2])> = path
            .windows(2)
            .filter_map(|w| match w {
                [p, q] => Some((*p, *q)),
                _ => None,
            })
            .collect();
        let lengths: Vec<f64> = legs.iter().map(|&(p, q)| length(p, q)).collect();
        let total: f64 = lengths.iter().sum();
        if !(total > 0.0) {
            return Ok(None);
        }
        let mut pieces: Vec<(Vec<f64>, Vec<[f64; 3]>)> = Vec::new();
        for (&(p, q), l) in legs.iter().zip(&lengths) {
            let uv = |t: f64| [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t];
            let mut count = ((CUT_PIECES as f64 * l / total).ceil() as usize).max(4);
            let piece = loop {
                let params: Vec<f64> = (0..=count).map(|i| i as f64 / count as f64).collect();
                let mut points = Vec::with_capacity(count + 1);
                for &t in &params {
                    let s = eval.point(uv(t)).ok_or("cut point")?;
                    if t > 0.0 && t < 1.0 && !inside(ring, uv(t)) && !near(&band.xyz, s, self.tol) {
                        return Ok(None);
                    }
                    points.push([s.x, s.y, s.z]);
                }
                let curve = interpolate(&points, &params, 3)?;
                let BCurve::Spline { knots, points: ctrl } = &curve else { return Err("a cut piece the kernel did not make a spline".into()) };
                let ce = CurveEval::new(&curve).map_err(|m| m.to_string())?;
                let mut worst: f64 = 0.0;
                for w in params.windows(2) {
                    let [t0, t1] = w else { continue };
                    let t = 0.5 * (t0 + t1);
                    let on = eval.point(uv(t)).ok_or("cut point")?;
                    worst = worst.max(ce.point(t).ok_or("cut point")?.dist(on));
                    if !inside(ring, uv(t)) && !near(&band.xyz, on, self.tol) {
                        return Ok(None);
                    }
                }
                if worst <= 0.5 * self.tol {
                    break (knots.clone(), ctrl.clone());
                }
                if count >= MAX_CUT_PIECES {
                    return Err(format!("a split face's cut not within {} mm in {MAX_CUT_PIECES} pieces", 0.5 * self.tol));
                }
                count = (2 * count).min(MAX_CUT_PIECES);
            };
            pieces.push(piece);
        }
        // One cubic on [0, 1]: each piece on its share of the length, joined where they meet.
        let (mut knots, mut points): (Vec<f64>, Vec<[f64; 3]>) = (Vec::new(), Vec::new());
        let mut from = 0.0;
        for (k, ((pk, pc), l)) in pieces.iter().zip(&lengths).enumerate() {
            let to = if k + 1 == pieces.len() { 1.0 } else { from + l / total };
            let map = |t: f64| from + (to - from) * t;
            // A clamped cubic on [0, 1]: four end knots each side.
            let Some(inner) = pk.get(4..pk.len().saturating_sub(4)) else { return Err("a cut piece that is not a clamped cubic".into()) };
            if k == 0 {
                knots.extend([0.0; 4]);
            } else {
                knots.extend([from; 3]);
            }
            knots.extend(inner.iter().map(|&t| map(t)));
            points.extend(pc.iter().skip(usize::from(k > 0)).copied());
            from = to;
        }
        knots.extend([1.0; 4]);
        Ok(Some(BCurve::Spline { knots, points }))
    }
}

/// How far from `c` along `d` the ray first crosses `ring` away from its corner `corner`.
fn exit(ring: &[[f64; 2]], c: [f64; 2], d: [f64; 2], corner: usize) -> Option<f64> {
    let n = ring.len();
    let mut best = f64::INFINITY;
    for (i, &p) in ring.iter().enumerate() {
        let j = (i + 1) % n;
        // The corner's own edges (it is repeated where the two loops meet).
        if [i, j].iter().any(|&k| k + 1 >= corner && k <= corner + 2) || (corner == 0 && (i + 2 >= n || j + 2 >= n)) {
            continue;
        }
        let Some(&q) = ring.get(j) else { continue };
        let e = [q[0] - p[0], q[1] - p[1]];
        let den = d[0] * e[1] - d[1] * e[0];
        if den == 0.0 {
            continue;
        }
        let w = [p[0] - c[0], p[1] - c[1]];
        let t = (w[0] * e[1] - w[1] * e[0]) / den;
        let s = (w[0] * d[1] - w[1] * d[0]) / den;
        if t > 0.0 && (0.0..=1.0).contains(&s) {
            best = best.min(t);
        }
    }
    best.is_finite().then_some(best)
}

/// The spline run the other way on [0, 1].
fn reverse_spline(c: BCurve) -> BCurve {
    match c {
        BCurve::Spline { knots, mut points } => {
            points.reverse();
            BCurve::Spline { knots: knots.iter().rev().map(|k| 1.0 - k).collect(), points }
        }
        other => other,
    }
}

/// The closed polygon's sides: each point and the next, the last back to the first.
fn sides<T: Copy>(ring: &[T]) -> impl Iterator<Item = (T, T)> + '_ {
    ring.iter().zip(ring.iter().cycle().skip(1)).map(|(a, b)| (*a, *b))
}

/// Whether `p` is inside the closed polygon `ring` (non-zero winding).
fn inside(ring: &[[f64; 2]], p: [f64; 2]) -> bool {
    let mut winding = 0i64;
    for (a, b) in sides(ring) {
        let cross = (b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1]);
        if a[1] <= p[1] {
            if b[1] > p[1] && cross > 0.0 {
                winding += 1;
            }
        } else if b[1] <= p[1] && cross < 0.0 {
            winding -= 1;
        }
    }
    winding != 0
}

/// Whether `p` lies within `tol` of the polyline through `ring` (closed).
fn near(ring: &[Vec3], p: Vec3, tol: f64) -> bool {
    sides(ring).any(|(a, b)| {
        let ab = b - a;
        let l = ab.dot(ab);
        let s = if l > 0.0 { ((p - a).dot(ab) / l).clamp(0.0, 1.0) } else { 0.0 };
        (a + ab * s).dist(p) <= tol
    })
}
