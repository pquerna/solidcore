//! The B-spline fast path: grid line spacing per knot span from a bound on the second
//! derivative, and grid evaluation one row at a time.
//!
//! On one knot span a curve's second derivative is a B-spline whose control points
//! bound it, and every iso-curve of a surface is a convex combination of its pole rows,
//! so the largest second-derivative control point over the rows bounds the sag of a
//! chord of parameter length `h` by `h^2 * bound / 8` on every iso-curve at once (for
//! rational surfaces the Cartesian poles give an estimate, not a bound). Grid
//! evaluation computes each grid line's basis functions once, contracts the poles
//! along it to one row, and evaluates every point of the line as a short dot product.
//! (Idea from the nurbs-grid-mesh design note, Apache-2.0; own code.)

use solidcore::math::{Point3, Vec3};
use solidcore::nurbs::b_spline_surface::BSplineSurface;
use solidcore::surface::surface::{Surface, SurfaceKind};

/// Most grid cells one knot span gets from the bound.
const MAX_CELLS_PER_SPAN: f64 = 64.0;

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

fn expand(knots: &[f64], mult: &[usize]) -> Vec<f64> {
    let mut out = Vec::new();
    for (k, m) in knots.iter().zip(mult) {
        for _ in 0..*m {
            out.push(*k);
        }
    }
    out
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

impl Nurbs {
    /// The B-spline data of `s`, `None` for a surface evaluated otherwise or with
    /// inconsistent arrays.
    pub(crate) fn from_surface(s: &Surface) -> Option<Self> {
        let c = b_spline(s)?.canonical().ok()?;
        let uk = expand(&c.u_knots, &c.u_multiplicities);
        let vk = expand(&c.v_knots, &c.v_multiplicities);
        let (p, q) = (c.u_degree, c.v_degree);
        let nu = uk.len().checked_sub(p + 1)?;
        let nv = vk.len().checked_sub(q + 1)?;
        if nu == 0 || nv == 0 || c.control_points.len() != nu.checked_mul(nv)? {
            return None;
        }
        let rational = c.weights.len() == c.control_points.len();
        let pw = c
            .control_points
            .iter()
            .enumerate()
            .map(|(i, pt)| {
                let w = if rational {
                    c.weights.get(i).copied().unwrap_or(1.0)
                } else {
                    1.0
                };
                [pt.x * w, pt.y * w, pt.z * w, w]
            })
            .collect();
        Some(Self {
            p,
            q,
            uk,
            vk,
            nu,
            nv,
            pw,
        })
    }

    fn pole(&self, i: usize, j: usize) -> Option<[f64; 4]> {
        self.pw
            .get(i.checked_mul(self.nv)?.checked_add(j)?)
            .copied()
    }

    /// Grid lines in u (`along_u`) or v over `[lo, hi]`: spaced so that the cells the
    /// second-derivative bound of each knot span asks for at chord tolerance `chord`
    /// (over the pole rows acting on `other`) add up to one per cell, with a line at
    /// every knot where the surface may kink (multiplicity at least the degree).
    pub(crate) fn lines(
        &self,
        along_u: bool,
        lo: f64,
        hi: f64,
        other: (f64, f64),
        chord: f64,
    ) -> Vec<f64> {
        let (deg, knots, n) = if along_u {
            (self.p, &self.uk, self.nu)
        } else {
            (self.q, &self.vk, self.nv)
        };
        let (odeg, oknots, on) = if along_u {
            (self.q, &self.vk, self.nv)
        } else {
            (self.p, &self.uk, self.nu)
        };
        // the pole rows that act on the other direction's range
        let r0 = span(oknots, odeg, on, other.0).saturating_sub(odeg);
        let r1 = span(oknots, odeg, on, other.1).min(on.saturating_sub(1));
        let multiplicity = |t: f64| knots.iter().filter(|k| **k == t).count();
        let mut out = Vec::new();
        // cells accumulated since the last line
        let mut acc = 0.0_f64;
        for s in deg..n {
            let (Some(&a), Some(&b)) = (knots.get(s), knots.get(s + 1)) else {
                continue;
            };
            if b <= a || b.is_nan() || b <= lo || a >= hi {
                continue;
            }
            if a > lo && multiplicity(a) >= deg.max(1) {
                out.push(a);
                acc = 0.0;
            }
            let (a, b) = (a.max(lo), b.min(hi));
            let bound = self.second_derivative_bound(along_u, s, (r0, r1));
            let total = if bound > 0.0 && chord > 0.0 {
                ((b - a) * (bound / (8.0 * chord)).sqrt()).min(MAX_CELLS_PER_SPAN)
            } else {
                0.0
            };
            if total <= 0.0 || total.is_nan() {
                continue;
            }
            let per_param = total / (b - a);
            let mut need = 1.0 - acc;
            let mut pos = a;
            let mut remaining = total;
            while remaining >= need {
                pos += need / per_param;
                if pos < hi {
                    out.push(pos);
                }
                remaining -= need;
                need = 1.0;
            }
            acc = 1.0 - need + remaining;
        }
        out
    }

    /// The largest second-derivative control point (Cartesian) of knot span `s` in u
    /// (`along_u`) or v over pole rows `rows` of the other direction.
    fn second_derivative_bound(&self, along_u: bool, s: usize, rows: (usize, usize)) -> f64 {
        let (deg, knots) = if along_u {
            (self.p, &self.uk)
        } else {
            (self.q, &self.vk)
        };
        if deg < 2 {
            return 0.0;
        }
        let k = |i: usize| knots.get(i).copied().unwrap_or(0.0);
        let point = |i: usize, r: usize| -> Option<Vec3> {
            let w = if along_u {
                self.pole(i, r)?
            } else {
                self.pole(r, i)?
            };
            if w[3] == 0.0 {
                return None;
            }
            Some(Vec3::new(w[0] / w[3], w[1] / w[3], w[2] / w[3]))
        };
        let pf = deg as f64;
        let mut bound = 0.0_f64;
        for r in rows.0..=rows.1 {
            for i in s.saturating_sub(deg)..=s.saturating_sub(2) {
                let (Some(p0), Some(p1), Some(p2)) =
                    (point(i, r), point(i + 1, r), point(i + 2, r))
                else {
                    continue;
                };
                let d1 = k(i + deg + 1) - k(i + 1);
                let d2 = k(i + deg + 2) - k(i + 2);
                let d = k(i + deg + 1) - k(i + 2);
                if !(d1 > 0.0 && d2 > 0.0 && d > 0.0) {
                    continue;
                }
                let a = (p2 - p1) * (pf / d2);
                let c = (p1 - p0) * (pf / d1);
                let q2 = (a - c) * ((pf - 1.0) / d);
                bound = bound.max(q2.length());
            }
        }
        bound
    }

    /// Point, `Su` and `Sv` at the grid points `(us[i], vs[j])` of `wanted`, in its
    /// order, one u line at a time: each line's basis is computed once and the poles
    /// contracted to one row along v. `None` for a point that does not evaluate.
    pub(crate) fn eval_grid(
        &self,
        us: &[f64],
        vs: &[f64],
        wanted: &[(usize, usize)],
    ) -> Vec<Option<(Point3, Vec3, Vec3)>> {
        let mut out = vec![None; wanted.len()];
        let vb: Vec<Option<Basis>> = vs
            .iter()
            .map(|v| basis(&self.vk, self.q, self.nv, *v))
            .collect();
        // wanted points by u line
        let mut order: Vec<usize> = (0..wanted.len()).collect();
        order.sort_by_key(|w| wanted.get(*w).map(|x| x.0));
        let mut row = vec![[0.0; 4]; self.nv];
        let mut drow = vec![[0.0; 4]; self.nv];
        let mut current: Option<usize> = None;
        let mut ub: Option<Basis> = None;
        for w in order {
            let Some(&(i, j)) = wanted.get(w) else {
                continue;
            };
            if current != Some(i) {
                current = Some(i);
                ub = us.get(i).and_then(|u| basis(&self.uk, self.p, self.nu, *u));
                if let Some(b) = &ub {
                    for (jj, (r, dr)) in row.iter_mut().zip(drow.iter_mut()).enumerate() {
                        *r = [0.0; 4];
                        *dr = [0.0; 4];
                        for (l, (n, dn)) in b.n.iter().zip(&b.dn).enumerate() {
                            if let Some(pw) = self.pole(b.first + l, jj) {
                                axpy(r, *n, &pw);
                                axpy(dr, *dn, &pw);
                            }
                        }
                    }
                }
            }
            let (Some(_), Some(Some(bv))) = (&ub, vb.get(j)) else {
                continue;
            };
            let mut a = [0.0; 4];
            let mut au = [0.0; 4];
            let mut av = [0.0; 4];
            for (l, (n, dn)) in bv.n.iter().zip(&bv.dn).enumerate() {
                let (Some(r), Some(dr)) = (row.get(bv.first + l), drow.get(bv.first + l)) else {
                    continue;
                };
                axpy(&mut a, *n, r);
                axpy(&mut au, *n, dr);
                axpy(&mut av, *dn, r);
            }
            if a[3] == 0.0 {
                continue;
            }
            let p = Vec3::new(a[0] / a[3], a[1] / a[3], a[2] / a[3]);
            let su = (Vec3::new(au[0], au[1], au[2]) - p * au[3]) / a[3];
            let sv = (Vec3::new(av[0], av[1], av[2]) - p * av[3]) / a[3];
            if let Some(o) = out.get_mut(w) {
                *o = Some((p, su, sv));
            }
        }
        out
    }
}

/// `acc += s * x`.
fn axpy(acc: &mut [f64; 4], s: f64, x: &[f64; 4]) {
    for (a, x) in acc.iter_mut().zip(x) {
        *a += s * x;
    }
}

/// Nonzero basis functions and their first derivatives at one parameter.
#[derive(Debug, Clone)]
struct Basis {
    /// Index of the pole the first function belongs to.
    first: usize,
    n: Vec<f64>,
    dn: Vec<f64>,
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
/// triangle with the derivative from the degree `deg - 1` functions).
fn basis(knots: &[f64], deg: usize, n: usize, t: f64) -> Option<Basis> {
    let s = span(knots, deg, n, t);
    let k = |i: usize| knots.get(i).copied();
    // ndu[r] are the degree-r functions N_{s-r..s}
    let mut ndu = vec![1.0];
    let mut prev = vec![1.0];
    for r in 1..=deg {
        let mut next = vec![0.0; r + 1];
        for (jj, slot) in next.iter_mut().enumerate() {
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
            *slot = v;
        }
        if r == deg {
            ndu = next;
        } else {
            prev = next;
        }
    }
    if deg == 0 {
        return Some(Basis {
            first: s,
            n: ndu,
            dn: vec![0.0],
        });
    }
    // derivatives from the degree deg - 1 functions (prev)
    let first = s.checked_sub(deg)?;
    let pf = deg as f64;
    let mut dn = vec![0.0; deg + 1];
    for (jj, slot) in dn.iter_mut().enumerate() {
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
        *slot = v;
    }
    Some(Basis { first, n: ndu, dn })
}
