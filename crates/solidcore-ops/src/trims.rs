//! The kernel's own pcurves for a face's uses that have no closed form (an approximated
//! intersection curve on a cone, a file's spline edge on a cylinder or spline surface), as
//! [`crate::pcurve::Trims`] for `solidcore-step`'s `read_solids_with`. The face goes to the
//! kernel alone, an open sheet, with the pcurves it has and without the others
//! ([`to_kernel_with_bare_uses`]), and the kernel makes those as it makes every trim it needs
//! (`SmEdgeuse::GetOrCreateUVTrimCurve`, asked for by [`to_record`]). They are the kernel's,
//! relayed as it holds them on the record's parameters; nothing here fits a curve.
//!
//! Origin: SolidCraft's `crates/smlib/src/pcurves.rs` (`SmlibTrims`, MIT OR Apache-2.0), which
//! relays SMLib's pcurves the same way; this one asks the Rust kernel through the record
//! conversion, so a reader needs no application crate to make them.

use solidcore_convert::brep::{to_kernel_with_bare_uses, to_record, Unsupported};
use solidcore_record::brep::{
    self as b, Curve2, EdgeUseRef, Loop, OrientedEdge, OrientedFace, Region, Shell, Vertex,
};

use crate::pcurve::{TrimFace, Trims};
use crate::session::run;

/// [`Trims`] by the kernel.
#[derive(Clone, Copy, Debug, Default)]
pub struct KernelTrims;

impl Trims for KernelTrims {
    fn trims(&self, face: &TrimFace<'_>) -> Result<Vec<Vec<Curve2>>, String> {
        trims(face).map_err(|e| format!("kernel pcurves: {e}"))
    }
}

fn trims(face: &TrimFace<'_>) -> Result<Vec<Vec<Curve2>>, String> {
    let (record, bare) = sheet(face)?;
    let mut made = None;
    let mut refusal = None;
    run("brep::kernel_trims", |k| {
        let built =
            to_kernel_with_bare_uses(k, &record, true, &bare).and_then(|(mut body, given)| {
                to_record(k, &mut body).map(|(out, order)| (out, order, given))
            });
        match built {
            Ok(m) => made = Some(m),
            Err(Unsupported::Kernel(e)) => return Err(e),
            Err(e) => refusal = Some(e.to_string()),
        }
        Ok(())
    })
    .map_err(|e| e.to_string())?;
    if let Some(r) = refusal {
        return Err(r);
    }
    let (out, order, given) = made.ok_or("the kernel made no body")?;
    let [made] = out.faces.as_slice() else {
        return Err(format!("the kernel made {} faces of one", out.faces.len()));
    };
    let made_uses: Vec<&OrientedEdge> = made
        .loops
        .iter()
        .flat_map(|l| match l {
            Loop::Edges { edges } => edges.as_slice(),
            Loop::Vertex { .. } => &[],
        })
        .collect();
    // The sheet's record edge `i` is the kernel edge `given.edges[i]`, which the record the
    // kernel wrote back names by its own index.
    let back = order.record_edges();
    let edge_of = |u: u32| -> Result<u32, String> {
        let kernel = given
            .edges
            .get(u as usize)
            .ok_or("an edge the kernel lost")?;
        back.get(kernel)
            .copied()
            .ok_or_else(|| "an edge the kernel lost".into())
    };
    // A sheet's loops come back the way the kernel runs them, which may be against ours: a use
    // is the same pcurve either way (on its edge's interval in its curve's direction), and the
    // loop's way round is read from an edge the face uses once.
    let ours = face.loops.iter().flatten();
    let once = ours
        .clone()
        .find(|u| ours.clone().filter(|v| v.edge == u.edge).count() == 1)
        .ok_or("a face of seam edges only")?;
    let at = edge_of(once.edge)?;
    let reversed = made_uses
        .iter()
        .find(|m| m.edge == at)
        .ok_or("a use the kernel did not keep")?
        .forward
        != once.forward;
    let mut result = Vec::with_capacity(face.loops.len());
    for l in &face.loops {
        let mut pcs = Vec::with_capacity(l.len());
        for u in l {
            let e = edge_of(u.edge)?;
            let theirs = out.edges.get(e as usize).ok_or("an edge the kernel lost")?;
            let ours = face.edges.get(u.edge as usize).ok_or("edge reference")?;
            let found = made_uses
                .iter()
                .find(|m| m.edge == e && m.forward == (u.forward != reversed))
                .ok_or("a use the kernel did not keep")?;
            // Each edge's tolerance as the kernel held it (raised where it lies off the
            // surface), never below ours.
            let tolerance = theirs.tolerance.max(ours.tolerance);
            let pc = crate::pcurve::on_interval(
                &found.pcurve,
                ours.curve,
                ours.interval,
                theirs.interval,
                tolerance,
            )
            .map_err(|e| e.to_string())?;
            pcs.push(pc);
        }
        result.push(pcs);
    }
    Ok(result)
}

/// A use of a face by its place: `(face, loop, position)`.
type UsePlace = (u32, u32, u32);

/// The face as a one-face open sheet, and its uses that go without pcurves, sorted.
fn sheet(face: &TrimFace<'_>) -> Result<(b::Brep, Vec<UsePlace>), String> {
    let mut edges: Vec<b::Edge> = face
        .edges
        .iter()
        .map(|e| b::Edge {
            curve: e.curve.clone(),
            nurbs: e.nurbs.cloned(),
            interval: e.interval,
            vertices: e.vertices,
            tolerance: e.tolerance,
            uses: Vec::new(),
        })
        .collect();
    let mut bare = Vec::new();
    let mut loops = Vec::with_capacity(face.loops.len());
    for (li, l) in face.loops.iter().enumerate() {
        let li = u32::try_from(li).map_err(|_| "too many loops")?;
        let mut uses = Vec::with_capacity(l.len());
        for (p, u) in l.iter().enumerate() {
            let p = u32::try_from(p).map_err(|_| "too many uses")?;
            let edge = edges.get_mut(u.edge as usize).ok_or("edge reference")?;
            edge.uses.push(EdgeUseRef {
                face: 0,
                r#loop: li,
                position: p,
                upper: !u.forward,
            });
            let pcurve = if let Some(pc) = &u.pcurve {
                pc.clone()
            } else {
                bare.push((0, li, p));
                // Not read: the use goes without a trim (`to_kernel_with_bare_uses`).
                let [a, z] = edge.interval;
                Curve2::Spline {
                    knots: vec![a, a, z, z],
                    points: vec![[0.0; 2]; 2],
                }
            };
            uses.push(OrientedEdge {
                edge: u.edge,
                forward: u.forward,
                pcurve,
            });
        }
        loops.push(Loop::Edges { edges: uses });
    }
    let vertices = face
        .vertices
        .iter()
        .map(|&point| Vertex {
            point,
            tolerance: face.tolerance,
        })
        .collect();
    let record = b::Brep {
        version: b::VERSION,
        tolerance: face.tolerance,
        regions: vec![Region {
            shells: vec![0],
            void: false,
        }],
        shells: vec![Shell::Faces {
            faces: vec![OrientedFace {
                face: 0,
                forward: true,
            }],
            wires: Vec::new(),
        }],
        faces: vec![b::Face {
            surface: face.surface.clone(),
            nurbs: face.nurbs.cloned(),
            domain: face.domain,
            loops,
        }],
        edges,
        vertices,
    };
    Ok((record, bare))
}
