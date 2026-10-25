//! The NURBS surface: validation, evaluation with first derivatives, and projection back onto it
//! (The NURBS Book, A2.1 to A2.3).

use crate::control::{reserve, zeros, Unlimited};
use crate::MeshControl;
use crate::MeshError;

/// A NURBS surface. Poles are indexed `i * nv + j`, `i` along u and `j` along v.
#[derive(Debug, Clone)]
pub struct NurbsSurface {
    pub(crate) u_degree: usize,
    pub(crate) v_degree: usize,
    pub(crate) u_knots: Vec<f64>,
    pub(crate) v_knots: Vec<f64>,
    pub(crate) nu: usize,
    pub(crate) nv: usize,
    /// Cartesian poles, for the curvature bound and for seeding.
    pub(crate) poles: Vec<[f64; 3]>,
    /// The poles in homogeneous form, `[x w, y w, z w, w]`, for evaluation.
    pub(crate) hpoles: Vec<[f64; 4]>,
    pub(crate) rational: bool,
    domain: [[f64; 2]; 2],
}

/// A point on the surface with its first partial derivatives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfacePoint {
    /// S(u, v).
    pub point: [f64; 3],
    /// dS/du.
    pub du: [f64; 3],
    /// dS/dv.
    pub dv: [f64; 3],
}

impl NurbsSurface {
    /// A surface from flat knot vectors (lengths `nu + u_degree + 1` and `nv + v_degree + 1`),
    /// `nu * nv` poles and, for a rational surface, one positive weight per pole. Every size and
    /// value is checked here, so evaluation never meets a short array.
    pub fn new(
        u_degree: usize,
        v_degree: usize,
        u_knots: Vec<f64>,
        v_knots: Vec<f64>,
        nu: usize,
        nv: usize,
        poles: Vec<[f64; 3]>,
        weights: Option<&[f64]>,
    ) -> Result<Self, MeshError> {
        Self::new_with_control(
            u_degree, v_degree, u_knots, v_knots, nu, nv, poles, weights, &Unlimited,
        )
    }

    /// Constructs the same surface with caller control over validation and new
    /// homogeneous coefficient storage. Moved input buffers are already owned
    /// and admitted by the caller; this method does not count them a second time.
    pub fn new_with_control(
        u_degree: usize,
        v_degree: usize,
        u_knots: Vec<f64>,
        v_knots: Vec<f64>,
        nu: usize,
        nv: usize,
        poles: Vec<[f64; 3]>,
        weights: Option<&[f64]>,
        control: &impl MeshControl,
    ) -> Result<Self, MeshError> {
        control.charge(1)?;
        if u_degree == 0 || v_degree == 0 {
            return Err(MeshError::BadSurface("degree 0"));
        }
        if nu <= u_degree || nv <= v_degree {
            return Err(MeshError::BadSurface("fewer poles than degree + 1"));
        }
        if u_knots.len() != nu + u_degree + 1 || v_knots.len() != nv + v_degree + 1 {
            return Err(MeshError::BadSurface(
                "knot count is not poles + degree + 1",
            ));
        }
        if poles.len() != nu * nv {
            return Err(MeshError::BadSurface("pole count is not nu * nv"));
        }
        for k in [&u_knots, &v_knots] {
            for (i, value) in k.iter().enumerate() {
                control.charge(1)?;
                if !value.is_finite() || (i > 0 && *value < k[i - 1]) {
                    return Err(MeshError::BadSurface("knots not finite and nondecreasing"));
                }
            }
        }
        for pole in &poles {
            control.charge(1)?;
            if pole.iter().any(|x| !x.is_finite()) {
                return Err(MeshError::BadSurface("pole not finite"));
            }
        }
        if let Some(w) = weights {
            if w.len() != poles.len() {
                return Err(MeshError::BadSurface("weight count is not nu * nv"));
            }
            for weight in w {
                control.charge(1)?;
                if !(weight.is_finite() && *weight > 0.0) {
                    return Err(MeshError::BadSurface("weight not finite and positive"));
                }
            }
        }
        let mut hpoles = Vec::new();
        reserve(&mut hpoles, poles.len(), control)?;
        for (i, &[x, y, z]) in poles.iter().enumerate() {
            control.charge(1)?;
            let w = weights.map_or(1.0, |weights| weights[i]);
            hpoles.push([x * w, y * w, z * w, w]);
        }
        let end = |k: &[f64], i: usize| k.get(i).copied().ok_or(MeshError::Internal);
        let domain = [
            [end(&u_knots, u_degree)?, end(&u_knots, nu)?],
            [end(&v_knots, v_degree)?, end(&v_knots, nv)?],
        ];
        let [[du0, du1], [dv0, dv1]] = domain;
        if !(du0 < du1 && dv0 < dv1) {
            return Err(MeshError::BadSurface("empty parameter domain"));
        }
        Ok(Self {
            u_degree,
            v_degree,
            u_knots,
            v_knots,
            nu,
            nv,
            poles,
            hpoles,
            rational: weights.is_some(),
            domain,
        })
    }

    /// The parameter domain, `[[u0, u1], [v0, v1]]`.
    pub fn domain(&self) -> [[f64; 2]; 2] {
        self.domain
    }

    /// Degrees in u and v.
    pub fn degrees(&self) -> [usize; 2] {
        [self.u_degree, self.v_degree]
    }

    /// Pole counts in u and v.
    pub fn pole_counts(&self) -> [usize; 2] {
        [self.nu, self.nv]
    }

    /// True when the surface carries weights.
    pub fn is_rational(&self) -> bool {
        self.rational
    }

    /// The surface and its first partial derivatives at (u, v). Parameters outside the domain
    /// evaluate the end spans' polynomials.
    pub fn evaluate(&self, u: f64, v: f64) -> Result<SurfacePoint, MeshError> {
        self.evaluate_with_control(u, v, &Unlimited)
    }

    /// Evaluates the original basis/derivative algorithm with caller control
    /// through its recurrences, accumulation and capacity-visible scratch.
    pub fn evaluate_with_control(
        &self,
        u: f64,
        v: f64,
        control: &impl MeshControl,
    ) -> Result<SurfacePoint, MeshError> {
        control.charge(1)?;
        let (p, q) = (self.u_degree, self.v_degree);
        // Bisection has a fixed 64-iteration bound in each direction.
        control.charge(128)?;
        let iu = find_span(self.nu, p, u, &self.u_knots).ok_or(MeshError::Internal)?;
        let iv = find_span(self.nv, q, v, &self.v_knots).ok_or(MeshError::Internal)?;
        let (bu, du) = basis_with_control(iu, u, p, &self.u_knots, control)?;
        let (bv, dv) = basis_with_control(iv, v, q, &self.v_knots, control)?;
        // Homogeneous sums A = sum N h, with their u and v derivatives.
        let mut a = [0.0; 4];
        let mut au = [0.0; 4];
        let mut av = [0.0; 4];
        for (r, (&nu, &nu1)) in bu.iter().zip(&du).enumerate() {
            control.charge(1)?;
            let row = (iu + r - p) * self.nv + iv - q;
            let poles = self.hpoles.get(row..=row + q).ok_or(MeshError::Internal)?;
            for ((h, &nv), &nv1) in poles.iter().zip(&bv).zip(&dv) {
                control.charge(1)?;
                axpy(&mut a, nu * nv, h);
                axpy(&mut au, nu1 * nv, h);
                axpy(&mut av, nu * nv1, h);
            }
        }
        let [ax, ay, az, w] = a;
        let point = [ax / w, ay / w, az / w];
        let deriv = |[x, y, z, dw]: [f64; 4]| {
            let [px, py, pz] = point;
            [(x - dw * px) / w, (y - dw * py) / w, (z - dw * pz) / w]
        };
        Ok(SurfacePoint {
            point,
            du: deriv(au),
            dv: deriv(av),
        })
    }

    /// The (u, v) that Gauss-Newton reaches from `seed` toward the foot of `q`, clamped to the
    /// domain, and the distance from `q` to the surface point there. A local search: the distance
    /// is to a real surface point, so it bounds the distance from `q` to the surface from above.
    pub fn foot_near(&self, q: [f64; 3], seed: [f64; 2]) -> Result<([f64; 2], f64), MeshError> {
        self.foot_near_with_control(q, seed, &Unlimited)
    }

    /// Runs the same Gauss-Newton search, checking the caller on each iteration
    /// and in every derivative evaluation, including the final distance.
    pub fn foot_near_with_control(
        &self,
        q: [f64; 3],
        seed: [f64; 2],
        control: &impl MeshControl,
    ) -> Result<([f64; 2], f64), MeshError> {
        control.charge(1)?;
        let [[u0, u1], [v0, v1]] = self.domain;
        let [mut u, mut v] = seed;
        for _ in 0..30 {
            control.charge(1)?;
            let SurfacePoint {
                point,
                du: su,
                dv: sv,
            } = self.evaluate_with_control(u, v, control)?;
            let r = sub(q, point);
            let (a, b, c) = (dot(su, su), dot(su, sv), dot(sv, sv));
            let (e, f) = (dot(su, r), dot(sv, r));
            let det = a * c - b * b;
            if det == 0.0 || !det.is_finite() {
                break;
            }
            let (mut nu, mut nv) = (u + (c * e - b * f) / det, v + (a * f - b * e) / det);
            // A joint step that leaves the domain in one parameter: hold that one at the edge and
            // solve for the other alone. Clamping the joint step instead keeps a v (or u) that is
            // right only if the clamped parameter had moved, and settles on the wrong point.
            let (out_u, out_v) = (!(u0..=u1).contains(&nu), !(v0..=v1).contains(&nv));
            if out_u && !out_v && c > 0.0 {
                nu = nu.clamp(u0, u1);
                nv = v + f / c;
            } else if out_v && !out_u && a > 0.0 {
                nv = nv.clamp(v0, v1);
                nu = u + e / a;
            }
            let (nu, nv) = (nu.clamp(u0, u1), nv.clamp(v0, v1));
            let (step_u, step_v) = (nu - u, nv - v);
            u = nu;
            v = nv;
            if step_u.abs() < 1e-12 * (u1 - u0) && step_v.abs() < 1e-12 * (v1 - v0) {
                break;
            }
        }
        let point = self.evaluate_with_control(u, v, control)?.point;
        Ok(([u, v], norm(sub(q, point))))
    }

    /// The (u, v) of a point on the surface (within `tol`), searched from `seed`. `None` when the
    /// search does not get within `tol`: a point that is not on the surface has no business on
    /// one of its faces' boundaries.
    pub fn project(
        &self,
        q: [f64; 3],
        seed: [f64; 2],
        tol: f64,
    ) -> Result<Option<[f64; 2]>, MeshError> {
        let (uv, d) = self.foot_near(q, seed)?;
        Ok((d <= tol).then_some(uv))
    }

    /// A seed for [`project`](Self::project): the Greville abscissae of the pole nearest `q`.
    pub fn seed_near(&self, q: [f64; 3]) -> Result<[f64; 2], MeshError> {
        let best = self
            .poles
            .iter()
            .map(|&p| norm2(sub(p, q)))
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map_or(0, |(k, _)| k);
        let greville = |k: &[f64], p: usize, i: usize| {
            k.get(i + 1..=i + p)
                .map(|s| s.iter().sum::<f64>() / p as f64)
                .ok_or(MeshError::Internal)
        };
        Ok([
            greville(&self.u_knots, self.u_degree, best / self.nv)?,
            greville(&self.v_knots, self.v_degree, best % self.nv)?,
        ])
    }
}

/// The knot span holding `t`: the `i` with `k[i] <= t < k[i + 1]`, clamped to `p..n`.
pub(crate) fn find_span(n: usize, p: usize, t: f64, k: &[f64]) -> Option<usize> {
    if !t.is_finite() {
        return None;
    }
    if t >= *k.get(n)? {
        return Some(n - 1);
    }
    if t <= *k.get(p)? {
        return Some(p);
    }
    let (mut lo, mut hi) = (p, n);
    // Bisection; 64 halvings cover any usize range.
    for _ in 0..64 {
        let mid = (lo + hi) / 2;
        if t < *k.get(mid)? {
            hi = mid;
        } else if t >= *k.get(mid + 1)? {
            lo = mid;
        } else {
            return Some(mid);
        }
    }
    None
}

/// The `p + 1` basis functions that are nonzero on span `i` at `t`, and their first derivatives.
pub(crate) fn basis(i: usize, t: f64, p: usize, k: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    basis_with_control(i, t, p, k, &Unlimited).ok()
}

fn basis_with_control(
    i: usize,
    t: f64,
    p: usize,
    k: &[f64],
    control: &impl MeshControl,
) -> Result<(Vec<f64>, Vec<f64>), MeshError> {
    control.charge(1)?;
    let w = p
        .checked_add(1)
        .ok_or_else(|| control.allocation_failed())?;
    let count = w
        .checked_mul(w)
        .ok_or_else(|| control.allocation_failed())?;
    let at = |xs: &[f64], index: usize| xs.get(index).copied().ok_or(MeshError::Internal);
    // ndu[r][c] at r * w + c: basis values above the diagonal, knot differences below it.
    let mut ndu = zeros(count, control)?;
    let mut left = zeros(w, control)?;
    let mut right = zeros(w, control)?;
    *ndu.get_mut(0).ok_or(MeshError::Internal)? = 1.0;
    for j in 1..=p {
        control.charge(1)?;
        *left.get_mut(j).ok_or(MeshError::Internal)? =
            t - at(k, (i + 1).checked_sub(j).ok_or(MeshError::Internal)?)?;
        *right.get_mut(j).ok_or(MeshError::Internal)? = at(k, i + j)? - t;
        let mut saved = 0.0;
        for r in 0..j {
            control.charge(1)?;
            let lower = at(&right, r + 1)? + at(&left, j - r)?;
            *ndu.get_mut(j * w + r).ok_or(MeshError::Internal)? = lower;
            let tmp = at(&ndu, r * w + j - 1)? / lower;
            *ndu.get_mut(r * w + j).ok_or(MeshError::Internal)? = saved + at(&right, r + 1)? * tmp;
            saved = at(&left, j - r)? * tmp;
        }
        *ndu.get_mut(j * w + j).ok_or(MeshError::Internal)? = saved;
    }
    let mut n = Vec::new();
    reserve(&mut n, w, control)?;
    for r in 0..w {
        control.charge(1)?;
        n.push(at(&ndu, r * w + p)?);
    }
    let mut d = zeros(w, control)?;
    for (r, dr) in d.iter_mut().enumerate() {
        control.charge(1)?;
        let mut x = 0.0;
        if r >= 1 {
            x += at(&ndu, (r - 1) * w + p - 1)? / at(&ndu, p * w + r - 1)?;
        }
        if r < p {
            x -= at(&ndu, r * w + p - 1)? / at(&ndu, p * w + r)?;
        }
        *dr = x * p as f64;
    }
    Ok((n, d))
}

fn axpy(acc: &mut [f64; 4], s: f64, h: &[f64; 4]) {
    for (a, x) in acc.iter_mut().zip(h) {
        *a += s * x;
    }
}

pub(crate) fn sub([ax, ay, az]: [f64; 3], [bx, by, bz]: [f64; 3]) -> [f64; 3] {
    [ax - bx, ay - by, az - bz]
}

pub(crate) fn dot([ax, ay, az]: [f64; 3], [bx, by, bz]: [f64; 3]) -> f64 {
    ax * bx + ay * by + az * bz
}

pub(crate) fn norm2(a: [f64; 3]) -> f64 {
    dot(a, a)
}

pub(crate) fn norm(a: [f64; 3]) -> f64 {
    norm2(a).sqrt()
}

pub(crate) fn lerp([ax, ay, az]: [f64; 3], [bx, by, bz]: [f64; 3], s: f64) -> [f64; 3] {
    [ax + s * (bx - ax), ay + s * (by - ay), az + s * (bz - az)]
}
