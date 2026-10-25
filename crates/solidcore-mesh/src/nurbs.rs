//! A B-spline surface evaluated directly (one point with its first derivatives, the
//! basis on the stack): the evaluation fast path for faces whose surface evaluates
//! through its B-spline. It does no meshing: dense NURBS faces can go to the
//! nurbs-grid-mesh backend ([`crate::backend`]).

use crate::budget::QueryBudget;
use solidcore::kernel::Kernel;
use solidcore::math::{Point3, Vec3};
use solidcore::nurbs::b_spline_surface::BSplineSurface;
use solidcore::surface::surface::{Surface, SurfaceKind};
use solidcore::Result;

/// A B-spline surface as plain arrays: degrees, full knot vectors and homogeneous poles.
#[derive(Debug, Clone)]
pub(crate) struct Nurbs {
    p: usize,
    q: usize,
    uk: Vec<f64>,
    vk: Vec<f64>,
    nu: usize,
    nv: usize,
    /// `w*x, w*y, w*z, w` of pole `(i, j)` at `i * nv + j`.
    pw: Vec<[f64; 4]>,
}

fn expand(knots: &[f64], mult: &[usize], budget: &QueryBudget<'_>) -> Result<Vec<f64>> {
    let mut out = Vec::new();
    for (k, m) in knots.iter().zip(mult) {
        budget.charge(1)?;
        budget.reserve_vec(&mut out, *m)?;
        for _ in 0..*m {
            budget.charge(1)?;
            out.push(*k);
        }
    }
    Ok(out)
}

/// The B-spline a surface evaluates through, for the kinds whose point evaluation is
/// exactly their B-spline's.
fn b_spline(s: &Surface) -> Option<&BSplineSurface> {
    match s.kind() {
        SurfaceKind::BSpline(b) => Some(b),
        SurfaceKind::SurfOfRevolution(r) => Some(r.base()),
        // its own B-spline when it has one (else `canonical` fails and the face takes
        // the general path)
        SurfaceKind::SurfOfExtrusion(e) => Some(e.base()),
        SurfaceKind::Sphere(r) => Some(r.base().base()),
        SurfaceKind::Torus(r) => Some(r.base().base()),
        _ => None,
    }
}

/// A surface's B-spline as plain arrays: degrees, full knot vectors, pole counts,
/// Cartesian poles (`(i, j)` at `i * nv + j`) and weights when rational.
#[derive(Debug, Clone)]
pub(crate) struct Canonical {
    pub p: usize,
    pub q: usize,
    pub uk: Vec<f64>,
    pub vk: Vec<f64>,
    pub nu: usize,
    pub nv: usize,
    pub poles: Vec<[f64; 3]>,
    pub weights: Option<Vec<f64>>,
}

/// The B-spline arrays of `s`, `None` for a surface evaluated otherwise or with
/// inconsistent arrays. A surface of extrusion without its B-spline yet (a lazy
/// generator, whose every evaluation makes one) gets it made once here.
pub(crate) fn canonical_arrays_with_budget(
    k: &Kernel,
    s: &Surface,
    budget: &QueryBudget<'_>,
) -> Result<Option<Canonical>> {
    let Some(b) = b_spline(s) else {
        return Ok(None);
    };
    if let Some(c) = arrays(b, budget)? {
        return Ok(Some(c));
    }
    match s.kind() {
        SurfaceKind::SurfOfExtrusion(e) => {
            // Kernel-owned cloning/generation is opaque and host-bounded. Do not
            // represent it as an exact adapter allocation or a cancellable inner loop.
            budget.charge(1)?;
            let mut e = (**e).clone();
            if e.make_nurb(k).is_err() {
                budget.checkpoint()?;
                return Ok(None);
            }
            budget.checkpoint()?;
            arrays(e.base(), budget)
        }
        _ => Ok(None),
    }
}

/// Kernel canonical extraction owns opaque temporary allocations. Admit the known
/// output cardinalities before crossing that boundary, then reconcile visible
/// returned capacities before any adapter copying (with both owners coexisting).
fn arrays(b: &BSplineSurface, budget: &QueryBudget<'_>) -> Result<Option<Canonical>> {
    use crate::storage::vector_bytes;
    use solidcore::nurbs::b_spline_surface::SurfParamType;
    let knots = b
        .number_natural_knots(SurfParamType::U)
        .saturating_add(b.number_natural_knots(SurfParamType::V));
    let poles = b
        .number_control_points(SurfParamType::U)
        .saturating_mul(b.number_control_points(SurfParamType::V));
    budget.charge(
        (knots as u64)
            .saturating_add(poles as u64)
            .saturating_add(1),
    )?;
    let expected = knots
        .saturating_mul(size_of::<f64>() + size_of::<usize>())
        .saturating_add(poles.saturating_mul(size_of::<Point3>() + size_of::<f64>()));
    budget.admit(expected)?;
    let Ok(c) = b.canonical() else {
        budget.checkpoint()?;
        return Ok(None);
    };
    budget.checkpoint()?;
    let actual = vector_bytes(&c.u_knots)
        .saturating_add(vector_bytes(&c.v_knots))
        .saturating_add(vector_bytes(&c.u_multiplicities))
        .saturating_add(vector_bytes(&c.v_multiplicities))
        .saturating_add(vector_bytes(&c.control_points))
        .saturating_add(vector_bytes(&c.weights));
    if actual > expected {
        budget.admit(actual - expected)?;
    }
    let uk = expand(&c.u_knots, &c.u_multiplicities, budget)?;
    let vk = expand(&c.v_knots, &c.v_multiplicities, budget)?;
    let (p, q) = (c.u_degree, c.v_degree);
    let Some(nu) = p.checked_add(1).and_then(|n| uk.len().checked_sub(n)) else {
        return Ok(None);
    };
    let Some(nv) = q.checked_add(1).and_then(|n| vk.len().checked_sub(n)) else {
        return Ok(None);
    };
    if nu == 0 || nv == 0 || Some(c.control_points.len()) != nu.checked_mul(nv) {
        return Ok(None);
    }
    let rational = c.weights.len() == c.control_points.len();
    let mut poles = Vec::new();
    budget.reserve_vec(&mut poles, c.control_points.len())?;
    for pt in &c.control_points {
        budget.charge(1)?;
        poles.push([pt.x, pt.y, pt.z]);
    }
    // Transfer the existing buffer, rather than allocating a second weights copy.
    Ok(Some(Canonical {
        p,
        q,
        uk,
        vk,
        nu,
        nv,
        poles,
        weights: rational.then_some(c.weights),
    }))
}

impl Nurbs {
    /// The B-spline data of `s`, `None` for a surface evaluated otherwise or with
    /// inconsistent arrays.
    pub(crate) fn from_surface_with_budget(
        k: &Kernel,
        s: &Surface,
        budget: &QueryBudget<'_>,
    ) -> Result<Option<Self>> {
        let Some(c) = canonical_arrays_with_budget(k, s, budget)? else {
            return Ok(None);
        };
        let mut pw = Vec::new();
        budget.reserve_vec(&mut pw, c.poles.len())?;
        for (i, [x, y, z]) in c.poles.iter().enumerate() {
            budget.charge(1)?;
            let w = c
                .weights
                .as_ref()
                .and_then(|w| w.get(i))
                .copied()
                .unwrap_or(1.0);
            pw.push([x * w, y * w, z * w, w]);
        }
        Ok(Some(Self {
            p: c.p,
            q: c.q,
            uk: c.uk,
            vk: c.vk,
            nu: c.nu,
            nv: c.nv,
            pw,
        }))
    }

    pub(crate) fn retained_buffer_bytes(&self) -> usize {
        crate::storage::vector_bytes(&self.uk)
            .saturating_add(crate::storage::vector_bytes(&self.vk))
            .saturating_add(crate::storage::vector_bytes(&self.pw))
    }

    pub(crate) fn knot_count(&self) -> usize {
        self.uk.len().saturating_add(self.vk.len())
    }

    /// Basis evaluation is stack-only. Charge its source cardinalities before the
    /// degree-limited recurrences and the (p+1)*(q+1) pole accumulation.
    pub(crate) fn eval_with_budget(
        &self,
        u: f64,
        v: f64,
        budget: &QueryBudget<'_>,
    ) -> Result<Option<(Point3, Vec3, Vec3)>> {
        let p = self.p.min(MAX_BASIS) as u64;
        let q = self.q.min(MAX_BASIS) as u64;
        let work = p
            .saturating_mul(p + 1)
            .saturating_add(q.saturating_mul(q + 1))
            .saturating_add((p + 1).saturating_mul(q + 1))
            .saturating_add(u64::from(usize::BITS - self.uk.len().leading_zeros()))
            .saturating_add(u64::from(usize::BITS - self.vk.len().leading_zeros()));
        budget.charge(work)?;
        Ok(self.eval(u, v))
    }

    /// Knot spans (of nonzero length) meeting `[lo, hi]` in u (`along_u`) or v.
    pub(crate) fn spans_in(&self, along_u: bool, lo: f64, hi: f64) -> usize {
        let k = if along_u { &self.uk } else { &self.vk };
        k.windows(2)
            .filter(|w| matches!(w, [a, b] if b > a && *b > lo && *a < hi))
            .count()
    }

    fn pole(&self, i: usize, j: usize) -> Option<[f64; 4]> {
        self.pw
            .get(i.checked_mul(self.nv)?.checked_add(j)?)
            .copied()
    }

    /// Point, `Su` and `Sv` at `(u, v)`; `None` where it does not evaluate.
    pub(crate) fn eval(&self, u: f64, v: f64) -> Option<(Point3, Vec3, Vec3)> {
        let bu = basis(&self.uk, self.p, self.nu, u)?;
        let bv = basis(&self.vk, self.q, self.nv, v)?;
        let mut a = [0.0; 4];
        let mut au = [0.0; 4];
        let mut av = [0.0; 4];
        for (i, (nu, dnu)) in bu.n().iter().zip(bu.dn()).enumerate() {
            for (j, (nv, dnv)) in bv.n().iter().zip(bv.dn()).enumerate() {
                let pw = self.pole(bu.first + i, bv.first + j)?;
                axpy(&mut a, nu * nv, &pw);
                axpy(&mut au, dnu * nv, &pw);
                axpy(&mut av, nu * dnv, &pw);
            }
        }
        if a[3] == 0.0 {
            return None;
        }
        let p = Vec3::new(a[0] / a[3], a[1] / a[3], a[2] / a[3]);
        let su = (Vec3::new(au[0], au[1], au[2]) - p * au[3]) / a[3];
        let sv = (Vec3::new(av[0], av[1], av[2]) - p * av[3]) / a[3];
        Some((p, su, sv))
    }
}

/// `acc += s * x`.
fn axpy(acc: &mut [f64; 4], s: f64, x: &[f64; 4]) {
    for (a, x) in acc.iter_mut().zip(x) {
        *a += s * x;
    }
}

/// Most nonzero basis functions at a parameter (degree + 1); higher degrees take the
/// general path.
const MAX_BASIS: usize = 32;

/// Nonzero basis functions and their first derivatives at one parameter, on the stack.
#[derive(Debug, Clone)]
struct Basis {
    /// Index of the pole the first function belongs to.
    first: usize,
    len: usize,
    values: [f64; MAX_BASIS],
    derivs: [f64; MAX_BASIS],
}

impl Basis {
    fn n(&self) -> &[f64] {
        self.values.get(..self.len).unwrap_or(&[])
    }

    fn dn(&self) -> &[f64] {
        self.derivs.get(..self.len).unwrap_or(&[])
    }
}

/// The knot span holding `t` (clamped to the valid range `deg..n`).
fn span(knots: &[f64], deg: usize, n: usize, t: f64) -> usize {
    let hi = n.saturating_sub(1).max(deg);
    let (mut lo_i, mut hi_i) = (deg, hi);
    if knots.get(hi + 1).is_some_and(|k| t >= *k) {
        return hi;
    }
    if knots.get(deg).is_some_and(|k| t <= *k) {
        return deg;
    }
    // binary search for knots[s] <= t < knots[s + 1]
    while lo_i < hi_i {
        let mid = (lo_i + hi_i).div_ceil(2);
        if knots.get(mid).is_some_and(|k| *k <= t) {
            lo_i = mid;
        } else {
            hi_i = mid - 1;
        }
    }
    lo_i
}

/// The basis functions of degree `deg` and their derivatives at `t` (the Cox-de Boor
/// triangle with the derivative from the degree `deg - 1` functions); `None` past
/// [`MAX_BASIS`] functions or for knots too short.
fn basis(knots: &[f64], deg: usize, n: usize, t: f64) -> Option<Basis> {
    if deg + 1 > MAX_BASIS {
        return None;
    }
    let s = span(knots, deg, n, t);
    let k = |i: usize| knots.get(i).copied();
    // row[r] holds the degree-r functions N_{s-r..s}; prev the row before
    let mut row = [0.0; MAX_BASIS];
    let mut prev = [0.0; MAX_BASIS];
    if let Some(r0) = row.get_mut(0) {
        *r0 = 1.0;
    }
    for r in 1..=deg {
        // the r functions of degree r - 1 (copying the whole row costs more than
        // the recurrence at low degree)
        for (p, x) in prev.iter_mut().zip(row.iter()).take(r) {
            *p = *x;
        }
        for jj in 0..=r {
            // N_{s-r+jj, r}
            let i = (s + jj).checked_sub(r)?;
            let mut v = 0.0;
            if jj >= 1 {
                let left = prev.get(jj - 1).copied().unwrap_or(0.0);
                let d = k(i + r)? - k(i)?;
                if d > 0.0 {
                    v += (t - k(i)?) / d * left;
                }
            }
            if jj < r {
                let right = prev.get(jj).copied().unwrap_or(0.0);
                let d = k(i + r + 1)? - k(i + 1)?;
                if d > 0.0 {
                    v += (k(i + r + 1)? - t) / d * right;
                }
            }
            if let Some(slot) = row.get_mut(jj) {
                *slot = v;
            }
        }
    }
    let first = s.checked_sub(deg)?;
    let mut derivs = [0.0; MAX_BASIS];
    if deg > 0 {
        // from the degree deg - 1 functions (prev)
        let pf = deg as f64;
        for jj in 0..=deg {
            let i = first + jj;
            let mut v = 0.0;
            if jj >= 1 {
                let d = k(i + deg)? - k(i)?;
                if d > 0.0 {
                    v += pf / d * prev.get(jj - 1).copied().unwrap_or(0.0);
                }
            }
            if jj < deg {
                let d = k(i + deg + 1)? - k(i + 1)?;
                if d > 0.0 {
                    v -= pf / d * prev.get(jj).copied().unwrap_or(0.0);
                }
            }
            if let Some(slot) = derivs.get_mut(jj) {
                *slot = v;
            }
        }
    }
    Some(Basis {
        first,
        len: deg + 1,
        values: row,
        derivs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coefficient_fixture() -> BSplineSurface {
        use solidcore::nurbs::math::{CPoint, NOW};
        let poles = vec![
            vec![
                CPoint::new(0.0, 0.0, 0.0, NOW),
                CPoint::new(0.0, 1.0, 0.0, NOW),
            ],
            vec![
                CPoint::new(1.0, 0.0, 0.0, NOW),
                CPoint::new(1.0, 1.0, 0.0, NOW),
            ],
        ];
        let knots = [0.0, 0.0, 1.0, 1.0];
        BSplineSurface::from_data(1, 1, 1, 1, 3, 3, &poles, &knots, &knots).unwrap()
    }

    #[test]
    fn extraction_limits_are_terminal_inside_knot_expansion() {
        use crate::budget::QueryLimit;
        use std::sync::atomic::AtomicBool;
        let surface = coefficient_fixture();
        let cancel = AtomicBool::new(false);
        // The boundary charge is 13; this permits it and the first expansion
        // charge, but refuses before the first expanded knot is copied.
        let budget = QueryBudget::new(14, usize::MAX, &cancel);
        assert!(arrays(&surface, &budget).is_err());
        assert_eq!(budget.failure(), Some(QueryLimit::Work));
        assert_eq!(budget.consumed(), 14);
        assert!(budget.admitted_bytes() > 0);
        let no_bytes = QueryBudget::new(u64::MAX, 0, &cancel);
        assert!(arrays(&surface, &no_bytes).is_err());
        assert_eq!(no_bytes.failure(), Some(QueryLimit::Bytes));
        assert_eq!(no_bytes.admitted_bytes(), 0);
    }

    #[test]
    fn basis_is_a_partition_of_unity_with_zero_sum_derivative() {
        let knots = [0.0, 0.0, 0.0, 0.0, 0.3, 0.5, 0.5, 0.8, 1.0, 1.0, 1.0, 1.0];
        let n = knots.len() - 4; // degree 3
        for i in 0..=100 {
            let t = f64::from(i) / 100.0;
            let b = basis(&knots, 3, n, t).unwrap();
            assert!((b.n().iter().sum::<f64>() - 1.0).abs() < 1e-14, "t {t}");
            assert!(b.dn().iter().sum::<f64>().abs() < 1e-11, "t {t}");
            assert!(b.first + 3 < n);
        }
    }

    #[test]
    fn spans_hold_their_parameter() {
        let knots = [0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 1.0, 1.0, 1.0];
        let n = knots.len() - 3;
        for (t, s) in [
            (0.0, 2),
            (0.1, 2),
            (0.25, 3),
            (0.49, 3),
            (0.5, 5),
            (0.9, 5),
            (1.0, 5),
        ] {
            assert_eq!(span(&knots, 2, n, t), s, "t {t}");
        }
    }
}
