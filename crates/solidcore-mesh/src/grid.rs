//! Interior grid lines of a face: where its Steiner points go before refinement.

use solidcore::geom::extent2d::Extent2d;
use solidcore::kernel::Kernel;
use solidcore::math::Point2;
use solidcore::surface::surface::Surface;
use solidcore::Result;

use crate::geom::{angle, segment_distance, surface_normal, surface_point};
use crate::nurbs::Nurbs;
use crate::options::FaceParams;

/// Most interior grid lines per UV direction of one face.
pub const MAX_GRID_LINES: usize = 2048;
/// Cells per direction of the sample grid that sizes a face's grid.
const SAMPLE_GRID: usize = 8;
/// Largest aspect ratio, in scaled UV, of a curved face's grid cells.
pub(crate) const MAX_CELL_ASPECT: f64 = 6.0;

/// The interior grid lines `(us, vs)` over `bbox` (strictly inside it, increasing):
/// from the chord deviation, normal turning and length between neighbouring points of
/// a sample grid, refined per knot span for a B-spline (`nurbs`), with cells of bounded
/// aspect in scaled UV on a curved face. A flat face gets none.
pub(crate) fn grid_lines(
    k: &Kernel,
    s: &Surface,
    bbox: &Extent2d,
    scale: (f64, f64),
    p: &FaceParams,
    nurbs: Option<&Nurbs>,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = SAMPLE_GRID;
    let at = |i: usize, j: usize| {
        Point2::new(
            bbox.u_min() + (bbox.u_max() - bbox.u_min()) * i as f64 / n as f64,
            bbox.v_min() + (bbox.v_max() - bbox.v_min()) * j as f64 / n as f64,
        )
    };
    // sample points and normals, evaluated once
    let mut pts = Vec::with_capacity((n + 1) * (n + 1));
    for j in 0..=n {
        for i in 0..=n {
            let uv = at(i, j);
            pts.push((surface_point(k, s, uv)?, surface_normal(k, s, uv)));
        }
    }
    let sample = |i: usize, j: usize| pts.get(j * (n + 1) + i);
    let need = |a: (usize, usize), b: (usize, usize)| -> Result<f64> {
        let (Some((pa, na)), Some((pb, nb))) = (sample(a.0, a.1), sample(b.0, b.1)) else {
            return Ok(0.0);
        };
        let (ua, ub) = (at(a.0, a.1), at(b.0, b.1));
        let pm = surface_point(k, s, Point2::new((ua.x + ub.x) / 2.0, (ua.y + ub.y) / 2.0))?;
        let mut m = (segment_distance(pm, *pa, *pb) / p.chord).sqrt();
        if let (Some(na), Some(nb)) = (na, nb) {
            m = m.max(angle(*na, *nb) / p.angle);
        }
        if let Some(l) = p.max_len {
            m = m.max(pa.distance(*pb) / l);
        }
        Ok(m)
    };
    // cells needed per sample interval, the worst over the rows (columns)
    let mut cu = vec![0.0_f64; n];
    let mut cv = vec![0.0_f64; n];
    for j in 0..=n {
        for (i, c) in cu.iter_mut().enumerate() {
            *c = c.max(need((i, j), (i + 1, j))?);
        }
    }
    for i in 0..=n {
        for (j, c) in cv.iter_mut().enumerate() {
            *c = c.max(need((i, j), (i, j + 1))?);
        }
    }
    let mut us = lines_from(&cu, bbox.u_min(), bbox.u_max());
    let mut vs = lines_from(&cv, bbox.v_min(), bbox.v_max());
    if let Some(nb) = nurbs {
        let curved = |c: &[f64]| c.iter().any(|x| *x > 0.0);
        if curved(&cu) || curved(&cv) {
            us.extend(nb.lines(
                true,
                bbox.u_min(),
                bbox.u_max(),
                (bbox.v_min(), bbox.v_max()),
                p.chord,
            ));
            vs.extend(nb.lines(
                false,
                bbox.v_min(),
                bbox.v_max(),
                (bbox.u_min(), bbox.u_max()),
                p.chord,
            ));
        }
    }
    let lu = (bbox.u_max() - bbox.u_min()) * scale.0;
    let lv = (bbox.v_max() - bbox.v_min()) * scale.1;
    let mut us = tidy(us, bbox.u_min(), bbox.u_max());
    let mut vs = tidy(vs, bbox.v_min(), bbox.v_max());
    // cells of bounded aspect: a direction left coarse gets uniform lines at a spacing
    // comparable to the other's mean spacing
    if !us.is_empty() || !vs.is_empty() {
        let hu = lu / (us.len() + 1) as f64;
        let hv = lv / (vs.len() + 1) as f64;
        let h = hu.min(hv);
        if h > 0.0 {
            us = with_uniform(
                us,
                bbox.u_min(),
                bbox.u_max(),
                (lu / (MAX_CELL_ASPECT * h)).ceil(),
            );
            vs = with_uniform(
                vs,
                bbox.v_min(),
                bbox.v_max(),
                (lv / (MAX_CELL_ASPECT * h)).ceil(),
            );
        }
    }
    // bounded work: at most MAX_GRID_LINES per direction, and about the face's
    // triangle budget in total
    let budget = (p.max_triangles / 2).max(1);
    while us.len() > MAX_GRID_LINES
        || vs.len() > MAX_GRID_LINES
        || (us.len() + 1) * (vs.len() + 1) > budget
    {
        if us.len() >= vs.len() {
            us = us.into_iter().step_by(2).collect();
        } else {
            vs = vs.into_iter().step_by(2).collect();
        }
        if us.is_empty() && vs.is_empty() {
            break;
        }
    }
    Ok((us, vs))
}

/// Lines over `[lo, hi]` for sample intervals needing `cells[i]` cells each (a flat
/// direction, all below one, gets none).
fn lines_from(cells: &[f64], lo: f64, hi: f64) -> Vec<f64> {
    if cells.iter().all(|c| *c < 1.0) {
        return Vec::new();
    }
    let n = cells.len() as f64;
    let mut out = Vec::new();
    for (i, c) in cells.iter().enumerate() {
        let a = lo + (hi - lo) * i as f64 / n;
        let b = lo + (hi - lo) * (i + 1) as f64 / n;
        if i > 0 {
            out.push(a);
        }
        let m = c.ceil().clamp(1.0, MAX_GRID_LINES as f64) as usize;
        for kk in 1..m {
            out.push(a + (b - a) * kk as f64 / m as f64);
        }
    }
    out
}

/// `lines` sorted, strictly inside `(lo, hi)`, without near duplicates.
fn tidy(mut lines: Vec<f64>, lo: f64, hi: f64) -> Vec<f64> {
    let eps = (hi - lo) * 1e-9;
    lines.retain(|t| t.is_finite() && *t > lo + eps && *t < hi - eps);
    lines.sort_by(f64::total_cmp);
    let mut out: Vec<f64> = Vec::with_capacity(lines.len());
    for t in lines {
        if out.last().is_none_or(|l| t - l > eps) {
            out.push(t);
        }
    }
    out
}

/// `lines` plus, where two neighbours (or the ends) are farther apart than
/// `(hi - lo) / cells`, uniform lines between them.
fn with_uniform(lines: Vec<f64>, lo: f64, hi: f64, cells: f64) -> Vec<f64> {
    if cells <= 1.0 || cells.is_nan() {
        return lines;
    }
    let h = (hi - lo) / cells.min(MAX_GRID_LINES as f64);
    let mut out = Vec::with_capacity(lines.len());
    let mut prev = lo;
    for t in lines.iter().copied().chain(core::iter::once(hi)) {
        let gap = t - prev;
        if gap > h {
            let m = (gap / h).ceil() as usize;
            for kk in 1..m {
                out.push(prev + gap * kk as f64 / m as f64);
            }
        }
        if t < hi {
            out.push(t);
        }
        prev = t;
    }
    tidy(out, lo, hi)
}

/// The typical grid spacing (model units, roughly arc length) of face surface `s` over
/// `domain` at tolerances `p`: the smaller mean cell size of its grid, `None` for a flat
/// face (no grid lines).
pub(crate) fn face_spacing(
    k: &Kernel,
    s: &Surface,
    domain: &Extent2d,
    p: &FaceParams,
) -> Option<f64> {
    if !(domain.u_min() < domain.u_max() && domain.v_min() < domain.v_max()) {
        return None;
    }
    let scale = crate::geom::surface_scales(k, s, domain);
    // a direction the surface does not move in (a collapsed face) sets no spacing
    let mid = domain.mid();
    let (_, su, sv) = s.evaluate_1st_derivatives(k, mid, false, false).ok()?;
    if !(su.length() > 1e-9 * p.chord.max(1e-300) && sv.length() > 1e-9 * p.chord.max(1e-300)) {
        return None;
    }
    let (us, vs) = grid_lines(k, s, domain, scale, p, None).ok()?;
    if us.is_empty() && vs.is_empty() {
        return None;
    }
    let lu = (domain.u_max() - domain.u_min()) * scale.0 / (us.len() + 1) as f64;
    let lv = (domain.v_max() - domain.v_min()) * scale.1 / (vs.len() + 1) as f64;
    // never below the chord tolerance
    let h = lu.min(lv).max(p.chord);
    (h.is_finite() && h > 0.0).then_some(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_are_sorted_inside_and_bounded() {
        assert!(lines_from(&[0.2, 0.5, 0.9], 0.0, 3.0).is_empty());
        let l = tidy(lines_from(&[2.0, 1.0, 3.0], 0.0, 3.0), 0.0, 3.0);
        assert_eq!(l.len(), 5);
        assert!(l.windows(2).all(|w| w[0] < w[1]) && l.iter().all(|t| *t > 0.0 && *t < 3.0));
        let u = with_uniform(vec![0.1], 0.0, 1.0, 4.0);
        assert!(u.windows(2).all(|w| w[1] - w[0] <= 0.25 + 1e-12));
        assert!(lines_from(&[1e300], 0.0, 1.0).len() <= MAX_GRID_LINES);
    }
}
