//! Small geometric helpers on solidcore curves and surfaces.

use solidcore::curve::curve::Curve;
use solidcore::geom::extent2d::Extent2d;
use solidcore::kernel::Kernel;
use solidcore::math::{Point2, Point3, Vec3};
use solidcore::nurbs::knots::Side;
use solidcore::surface::surface::Surface;
use solidcore::{Result, SolidError};

pub(crate) const OP: &str = "solidcore_mesh::tessellate";

/// Times a Gauss-Newton step is halved before the search stops.
const DAMPING_HALVINGS: usize = 6;
/// Gauss-Newton steps of a UV polish.
const POLISH_STEPS: usize = 12;
/// Cells per direction of the coarse search that seeds a point inversion.
const INVERT_GRID: usize = 16;

pub(crate) fn invalid(reason: &'static str) -> SolidError {
    SolidError::invalid_input(OP, reason)
}

pub(crate) fn finite3(p: Point3) -> bool {
    p.x.is_finite() && p.y.is_finite() && p.z.is_finite()
}

/// Point and first derivative of `curve` at `t` (from the left when `from_left`).
pub(crate) fn curve_eval(
    k: &Kernel,
    curve: &Curve,
    t: f64,
    from_left: bool,
) -> Result<(Point3, Vec3)> {
    let d = curve.evaluate(k, t, 1, from_left, true)?;
    let p = *d
        .first()
        .ok_or_else(|| invalid("curve evaluation without point"))?;
    let t1 = d.get(1).copied().unwrap_or(Vec3::new(0.0, 0.0, 0.0));
    if !finite3(p) {
        return Err(invalid("curve evaluates to a non-finite point"));
    }
    Ok((p, t1))
}

/// Distance from `p` to the segment `a`-`b`.
pub(crate) fn segment_distance(p: Point3, a: Point3, b: Point3) -> f64 {
    let d = b - a;
    let l2 = d.length_squared();
    if l2 <= 0.0 {
        return p.distance(a);
    }
    let s = ((p - a).dot(d) / l2).clamp(0.0, 1.0);
    p.distance(a + d * s)
}

/// Angle between two vectors, `0` when either is zero or not finite.
pub(crate) fn angle(a: Vec3, b: Vec3) -> f64 {
    let la = a.length();
    let lb = b.length();
    if !(la > 0.0 && lb > 0.0 && la.is_finite() && lb.is_finite()) {
        return 0.0;
    }
    (a.dot(b) / (la * lb)).clamp(-1.0, 1.0).acos()
}

/// Unit surface normal at `uv` (`EvaluateNormal`, else `Su x Sv`), `None` where the
/// surface has none.
pub(crate) fn surface_normal(k: &Kernel, s: &Surface, uv: Point2) -> Option<Vec3> {
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

/// The surface point at `uv`, which must be finite.
pub(crate) fn surface_point(k: &Kernel, s: &Surface, uv: Point2) -> Result<Point3> {
    let p = s.evaluate_point(k, uv)?;
    if finite3(p) {
        Ok(p)
    } else {
        Err(invalid("surface evaluates to a non-finite point"))
    }
}

/// A surface's natural UV domain and its period in each direction where it closes on
/// itself. A face's UV may wind past the domain along a closed direction (a thread
/// flank whose trims run just over one turn); its UV is then kept unwrapped and each
/// evaluation takes the same point one or more whole periods back in the domain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Chart {
    pub(crate) domain: Extent2d,
    pub(crate) period: (Option<f64>, Option<f64>),
}

impl Chart {
    /// `s`'s chart: its natural domain, closed in a direction when its two opposite
    /// sides agree within `tol`.
    pub(crate) fn of_with_budget(
        k: &Kernel,
        s: &Surface,
        tol: f64,
        budget: &crate::budget::QueryBudget<'_>,
    ) -> Result<Self> {
        let domain = s.natural_uv_domain_with_nurb(k)?;
        let period = (
            closes_with_budget(k, s, &domain, true, tol, budget)?
                .then(|| domain.u_max() - domain.u_min()),
            closes_with_budget(k, s, &domain, false, tol, budget)?
                .then(|| domain.v_max() - domain.v_min()),
        );
        Ok(Self { domain, period })
    }

    /// `uv` kept inside the domain along open directions; closed directions are not
    /// bounded.
    pub(crate) fn clamp(&self, uv: Point2) -> Point2 {
        let d = &self.domain;
        if !(d.u_min() <= d.u_max() && d.v_min() <= d.v_max()) {
            return uv;
        }
        let keep = |x: f64, lo: f64, hi: f64, p: Option<f64>| match p {
            Some(p) if p > 0.0 => x,
            _ => x.clamp(lo, hi),
        };
        Point2::new(
            keep(uv.x, d.u_min(), d.u_max(), self.period.0),
            keep(uv.y, d.v_min(), d.v_max(), self.period.1),
        )
    }

    /// `uv` moved by whole periods into the domain along closed directions, where the
    /// surface evaluates the same point; unchanged (bit for bit) when already inside.
    pub(crate) fn reduce(&self, uv: Point2) -> Point2 {
        let d = &self.domain;
        let wrap = |x: f64, lo: f64, hi: f64, p: Option<f64>| match p {
            Some(p) if p > 0.0 && x.is_finite() && !(lo..=hi).contains(&x) => {
                (x - ((x - lo) / p).floor() * p).clamp(lo, hi)
            }
            _ => x,
        };
        Point2::new(
            wrap(uv.x, d.u_min(), d.u_max(), self.period.0),
            wrap(uv.y, d.v_min(), d.v_max(), self.period.1),
        )
    }
}

/// `uv` moved by damped Gauss-Newton steps towards the surface point nearest
/// `target`, kept inside `chart` along its open directions and unwrapped along closed
/// ones, with caller checks before each native evaluation. Kernel-owned evaluation
/// scratch remains an opaque host allocation.
pub(crate) fn foot_with_budget(
    k: &Kernel,
    s: &Surface,
    uv: Point2,
    target: Point3,
    chart: &Chart,
    budget: &crate::budget::QueryBudget<'_>,
) -> Result<(Point2, Option<f64>)> {
    budget.charge(1)?;
    let Ok(p0) = s.evaluate_point(k, chart.reduce(uv)) else {
        return Ok((uv, None));
    };
    let mut best = p0.distance(target);
    let mut uv = uv;
    for _ in 0..POLISH_STEPS {
        budget.charge(1)?;
        if best == 0.0 {
            break;
        }
        let Ok((p, su, sv)) = s.evaluate_1st_derivatives(k, chart.reduce(uv), false, false) else {
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
        // damped: halve a step that does not bring the point closer (a kink in the
        // parameterisation, a far start)
        let mut improved = None;
        let mut f = 1.0;
        for _ in 0..DAMPING_HALVINGS {
            budget.charge(1)?;
            let next = chart.clamp(Point2::new(uv.x + du * f, uv.y + dv * f));
            if let Ok(q) = s.evaluate_point(k, chart.reduce(next)) {
                let d = q.distance(target);
                if d < best {
                    improved = Some((d, next));
                    break;
                }
            }
            f *= 0.5;
        }
        let Some((d, next)) = improved else {
            break;
        };
        best = d;
        uv = next;
    }
    Ok((uv, Some(best)))
}

/// The UV of `target` by a coarse search over the chart's domain and a Newton polish.
pub(crate) fn invert_point_with_budget(
    k: &Kernel,
    s: &Surface,
    target: Point3,
    chart: &Chart,
    budget: &crate::budget::QueryBudget<'_>,
) -> Result<Point2> {
    let domain = &chart.domain;
    let mut best: Option<(f64, Point2)> = None;
    for j in 0..=INVERT_GRID {
        for i in 0..=INVERT_GRID {
            budget.charge(1)?;
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
    let (_, uv) = best.ok_or_else(|| invalid("surface does not evaluate"))?;
    Ok(foot_with_budget(k, s, uv, target, chart, budget)?.0)
}

/// The power of two nearest `x` (`x` positive and finite), so scaling by it is exact.
pub(crate) fn power_of_two(x: f64) -> f64 {
    if !(x.is_finite() && x > 0.0) {
        return 1.0;
    }
    let e = x.log2().round().clamp(-1000.0, 1000.0);
    2.0_f64.powi(e as i32)
}

/// Mean `|Su|` and `|Sv|` over a sample grid of `d` (in `chart`), each rounded to a power
/// of two (so scaling is exact both ways), so `(u * su, v * sv)` is roughly arc length.
pub(crate) fn surface_scales_with_budget(
    k: &Kernel,
    s: &Surface,
    chart: &Chart,
    d: &Extent2d,
    budget: &crate::budget::QueryBudget<'_>,
) -> Result<(f64, f64)> {
    let (mut su_sum, mut sv_sum, mut n) = (0.0, 0.0, 0.0);
    for j in 0..=4 {
        for i in 0..=4 {
            budget.charge(1)?;
            let uv = Point2::new(
                d.u_min() + (d.u_max() - d.u_min()) * f64::from(i) / 4.0,
                d.v_min() + (d.v_max() - d.v_min()) * f64::from(j) / 4.0,
            );
            if let Ok((_, su, sv)) = s.evaluate_1st_derivatives(k, chart.reduce(uv), false, false) {
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
    if su <= 1e-300 || su.is_nan() {
        su = if sv > 1e-300 { sv } else { 1.0 };
    }
    if sv <= 1e-300 || sv.is_nan() {
        sv = su;
    }
    Ok((power_of_two(su), power_of_two(sv)))
}

/// Whether the surface closes on itself across `domain` in u (`along_u`) or v: its
/// points on the two opposite domain sides agree within `tol` at five places.
pub(crate) fn closes_with_budget(
    k: &Kernel,
    s: &Surface,
    domain: &Extent2d,
    along_u: bool,
    tol: f64,
    budget: &crate::budget::QueryBudget<'_>,
) -> Result<bool> {
    for i in 0..=4 {
        budget.charge(2)?;
        let w = f64::from(i) / 4.0;
        let (a, b) = if along_u {
            let v = domain.v_min() + (domain.v_max() - domain.v_min()) * w;
            (
                Point2::new(domain.u_min(), v),
                Point2::new(domain.u_max(), v),
            )
        } else {
            let u = domain.u_min() + (domain.u_max() - domain.u_min()) * w;
            (
                Point2::new(u, domain.v_min()),
                Point2::new(u, domain.v_max()),
            )
        };
        match (s.evaluate_point(k, a), s.evaluate_point(k, b)) {
            (Ok(p), Ok(q)) if p.distance(q) <= tol => {}
            _ => return Ok(false),
        }
    }
    Ok(true)
}
