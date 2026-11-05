//! Optical-flow fallback for steep head angles.
//!
//! MediaPipe's landmark model is trained on fairly frontal faces: past ~60 degrees (a far
//! turn, looking down at the keyboard) it loses confidence or its points drift. Textured spots
//! on the face (brows, nostrils, eye corners, skin texture) still move rigidly with the head,
//! so they are tracked frame to frame with pyramidal Lucas-Kanade and the head pose is fitted
//! to them against the same canonical face mesh: each spot knows where it sits on the mesh
//! from the last frame MediaPipe was sure about. That carries the pose through angles the
//! landmark model can't see, and tells whether its unsure landmarks are still right.

use crate::canonical_mesh::{TRIANGLES, VERTICES};
use crate::compose::{LEFT_EYE, Projection, RIGHT_EYE};
use crate::tracker::{Frame, Head, fit_pose};
use bevy::math::{Mat3, Vec2, Vec3};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Pyramid levels (half-res base, /2, /4): follows ~50 px of motion per frame at full res.
const LEVELS: usize = 3;
/// Lucas-Kanade window radius, base pixels.
const RADIUS: i32 = 6;
/// Points on skin that turns this far from the camera (cosine) stop counting: the window's
/// texture squashes from frame to frame there and a translation-only tracker misreads it.
const MIN_FACING: f32 = 0.3;
/// Below this the window has no texture to lock onto (smallest structure-tensor eigenvalue
/// per pixel, grey 0..1).
const MIN_EIGEN: f32 = 4e-6;
/// Forward-backward check: tracking a point back must land within this (base pixels).
const MAX_ROUND_TRIP: f32 = 0.7;
const MAX_POINTS: usize = 70;
const MIN_GAP: f32 = 5.0;
const MIN_POINTS: usize = 10;
const RESEED_BELOW: usize = 40;
/// Reprojection error (full-res px) where a point starts losing weight, and the RMS that
/// still counts as a good fit. Never less than the same in cm on the face: leaning in close
/// makes everything (perspective error included) bigger in pixels, and fixed pixel limits
/// threw the fit out exactly then.
const HUBER_PX: f32 = 1.5;
const MAX_RMS_PX: f32 = 3.0;
const HUBER_CM: f32 = 0.08;
const MAX_RMS_CM: f32 = 0.16;
/// The flow alone (MediaPipe never confirming it) drifts slowly: give up after this.
const MAX_UNVERIFIED: Duration = Duration::from_secs(5);
/// Starting from an older trusted frame would be a jump too far to track.
const MAX_ANCHOR_AGE: Duration = Duration::from_millis(500);

/// Grey image, 0..1. Only a window of it is computed (the face, not the room behind it):
/// `w` x `h` pixels from (`ox`, `oy`) of a `full`-sized image. Coordinates are always those of
/// the full image.
pub struct Gray {
    w: usize,
    h: usize,
    ox: usize,
    oy: usize,
    full: (usize, usize),
    px: Vec<f32>,
}

impl Gray {
    #[cfg(test)]
    fn new(w: usize, h: usize, px: Vec<f32>) -> Self {
        Self { w, h, ox: 0, oy: 0, full: (w, h), px }
    }

    /// Luma at half resolution: a face is still ~150 px tall there, and everything is 4x cheaper.
    /// Only inside `region` (frame pixels), or all of it.
    pub fn half(frame: &Frame, region: Option<(Vec2, Vec2)>) -> Self {
        let full = (frame.w as usize / 2, frame.h as usize / 2);
        // Aligned to the coarsest level, so every level's window starts on a whole pixel.
        let align = 1 << (LEVELS - 1);
        let (ox, oy, ex, ey) = match region {
            Some((lo, hi)) => {
                let lo = (lo * 0.5).max(Vec2::ZERO);
                let (ox, oy) = (lo.x as usize / align * align, lo.y as usize / align * align);
                let (ex, ey) = ((hi.x * 0.5).ceil().max(0.0) as usize, (hi.y * 0.5).ceil().max(0.0) as usize);
                (ox.min(full.0), oy.min(full.1), ex.clamp(ox.min(full.0), full.0), ey.clamp(oy.min(full.1), full.1))
            }
            None => (0, 0, full.0, full.1),
        };
        // Face (nearly) out of the picture: the whole frame, as before.
        let (ox, oy, ex, ey) = if ex < ox + 16 || ey < oy + 16 { (0, 0, full.0, full.1) } else { (ox, oy, ex, ey) };
        let (w, h, stride) = (ex - ox, ey - oy, frame.w as usize);
        let luma = |x: usize, y: usize| {
            let c = &frame.rgb[(y * stride + x) * 3..][..3];
            0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32
        };
        let mut px = Vec::with_capacity(w * h);
        for y in oy..ey {
            for x in ox..ex {
                let sum = luma(2 * x, 2 * y) + luma(2 * x + 1, 2 * y) + luma(2 * x, 2 * y + 1) + luma(2 * x + 1, 2 * y + 1);
                px.push(sum / (4.0 * 255.0));
            }
        }
        Self { w, h, ox, oy, full, px }
    }

    fn down(&self) -> Self {
        let (w, h, ox, oy) = (self.w / 2, self.h / 2, self.ox / 2, self.oy / 2);
        let mut px = Vec::with_capacity(w * h);
        for y in oy..oy + h {
            for x in ox..ox + w {
                px.push((self.get(2 * x, 2 * y) + self.get(2 * x + 1, 2 * y) + self.get(2 * x, 2 * y + 1) + self.get(2 * x + 1, 2 * y + 1)) * 0.25);
            }
        }
        Self { w, h, ox, oy, full: (self.full.0 / 2, self.full.1 / 2), px }
    }

    fn get(&self, x: usize, y: usize) -> f32 {
        self.px[(y - self.oy) * self.w + (x - self.ox)]
    }

    /// Bilinear, clamped to the edge of the window.
    fn at(&self, p: Vec2) -> f32 {
        let x = p.x.clamp(self.ox as f32, (self.ox + self.w - 1) as f32);
        let y = p.y.clamp(self.oy as f32, (self.oy + self.h - 1) as f32);
        let (x0, y0) = (x as usize, y as usize);
        let (x1, y1) = ((x0 + 1).min(self.ox + self.w - 1), (y0 + 1).min(self.oy + self.h - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let top = self.get(x0, y0) * (1.0 - fx) + self.get(x1, y0) * fx;
        let bottom = self.get(x0, y1) * (1.0 - fx) + self.get(x1, y1) * fx;
        top * (1.0 - fy) + bottom * fy
    }
}

pub struct Pyramid(Vec<Gray>);

impl Pyramid {
    pub fn new(base: Gray) -> Self {
        let mut levels = vec![base];
        while levels.len() < LEVELS {
            let next = levels.last().unwrap().down();
            levels.push(next);
        }
        Self(levels)
    }
}

/// Where the point `p` (base pixels of `prev`) is in `next`, if it can be followed both ways.
/// `guess` is the expected motion: the search starts there.
pub fn track(prev: &Pyramid, next: &Pyramid, p: Vec2, guess: Vec2) -> Option<Vec2> {
    let q = lucas_kanade(prev, next, p, guess)?;
    let back = lucas_kanade(next, prev, q, p - q)?;
    ((back - p).length() < MAX_ROUND_TRIP).then_some(q)
}

/// Pyramidal Lucas-Kanade (Bouguet): coarse to fine, Gauss-Newton on the window at each level.
fn lucas_kanade(a: &Pyramid, b: &Pyramid, p: Vec2, guess: Vec2) -> Option<Vec2> {
    let mut flow = guess / (1 << (LEVELS - 1)) as f32;
    for level in (0..LEVELS).rev() {
        let (ia, ib) = (&a.0[level], &b.0[level]);
        let c = p / (1 << level) as f32;
        let mut template = Vec::with_capacity(((2 * RADIUS + 1) * (2 * RADIUS + 1)) as usize);
        let (mut xx, mut xy, mut yy) = (0.0f32, 0.0f32, 0.0f32);
        for dy in -RADIUS..=RADIUS {
            for dx in -RADIUS..=RADIUS {
                let q = c + Vec2::new(dx as f32, dy as f32);
                let g = Vec2::new(ia.at(q + Vec2::X) - ia.at(q - Vec2::X), ia.at(q + Vec2::Y) - ia.at(q - Vec2::Y)) * 0.5;
                template.push((ia.at(q), g));
                xx += g.x * g.x;
                xy += g.x * g.y;
                yy += g.y * g.y;
            }
        }
        let n = template.len() as f32;
        let min_eigen = (xx + yy - ((xx - yy).powi(2) + 4.0 * xy * xy).sqrt()) * 0.5 / n;
        let det = xx * yy - xy * xy;
        let mut v = Vec2::ZERO;
        if min_eigen >= MIN_EIGEN * 0.25 && det > 0.0 {
            for _ in 0..20 {
                let (mut e, mut k) = (Vec2::ZERO, 0);
                for dy in -RADIUS..=RADIUS {
                    for dx in -RADIUS..=RADIUS {
                        let (value, g) = template[k];
                        e += g * (value - ib.at(c + flow + v + Vec2::new(dx as f32, dy as f32)));
                        k += 1;
                    }
                }
                let step = Vec2::new(yy * e.x - xy * e.y, xx * e.y - xy * e.x) / det;
                v += step;
                if step.length() < 0.01 {
                    break;
                }
            }
        } else if level == 0 {
            return None; // nothing to lock onto at full detail
        }
        flow = if level == 0 { flow + v } else { (flow + v) * 2.0 };
    }
    let q = p + flow;
    let base = &b.0[0];
    let inside = q.x >= base.ox as f32 && q.y >= base.oy as f32 && q.x < (base.ox + base.w) as f32 && q.y < (base.oy + base.h) as f32;
    (q.is_finite() && inside).then_some(q)
}

/// Frame pixels (pixel centres at +0.5) -> base pixels of the half-res grey image, whose
/// pixel u averages frame pixels 2u and 2u+1, so it sits at frame coordinate 2u+1.
fn to_gray(p: Vec2) -> Vec2 {
    p * 0.5 - Vec2::splat(0.5)
}
fn to_frame(g: Vec2) -> Vec2 {
    (g + Vec2::splat(0.5)) * 2.0
}

/// A followed spot: where it is (base pixels), where it sits on the canonical mesh, and the
/// mesh normal there.
#[derive(Clone, Copy)]
struct Spot {
    at: Vec2,
    canonical: Vec3,
    normal: Vec3,
}

/// Well-textured spots on the visible face (Shi-Tomasi corners, base pixels of `gray`), each
/// with where it sits on the canonical mesh: found through the front-most camera-facing
/// mesh triangle under it, at the pose `head` with the mesh where `landmarks` say.
///
/// Your face is not the canonical one. Turned far, image x is mostly depth, and a centimetre
/// of difference there put the spots ~20 px off any pose: the flow could not even start. So
/// each spot is moved sideways (in the image plane of this pose) until `head` projects it
/// exactly where it is; the flow then measures motion relative to here, where errors in
/// depth only matter by how much the head turns from now on.
fn seed(gray: &Gray, landmarks: &[Vec3], head: &Head) -> Vec<Spot> {
    let m = Mat3::from_quat(head.rotation);
    let proj = Projection::new(head);
    let img = |i: u16| to_gray(landmarks[i as usize].truncate());
    let canon = |i: u16| Vec3::from(VERTICES[i as usize]);
    let (lo, hi) = landmarks[..VERTICES.len()].iter().fold((Vec2::MAX, Vec2::MIN), |(lo, hi), p| (lo.min(to_gray(p.truncate())), hi.max(to_gray(p.truncate()))));
    let x0 = (lo.x.floor().max((gray.ox + 4) as f32)) as usize;
    let y0 = (lo.y.floor().max((gray.oy + 4) as f32)) as usize;
    let x1 = (hi.x.ceil().max(0.0) as usize).min((gray.ox + gray.w).saturating_sub(5));
    let y1 = (hi.y.ceil().max(0.0) as usize).min((gray.oy + gray.h).saturating_sub(5));
    if x1 <= x0 || y1 <= y0 {
        return Vec::new();
    }
    let (w, h) = (x1 - x0, y1 - y0);
    // Front-most visible triangle per pixel.
    let mut owner = vec![u16::MAX; w * h];
    let mut depth = vec![f32::NEG_INFINITY; w * h];
    for (k, t) in TRIANGLES.iter().enumerate() {
        let x = t.map(canon);
        if (m * (x[1] - x[0]).cross(x[2] - x[0]).normalize()).z < MIN_FACING + 0.1 {
            continue;
        }
        let p = t.map(img);
        let area = (p[1] - p[0]).perp_dot(p[2] - p[0]);
        if area.abs() < 1e-6 {
            continue;
        }
        let z = x.map(|v| m.row(2).dot(v));
        let (tl, br) = (p[0].min(p[1]).min(p[2]), p[0].max(p[1]).max(p[2]));
        for y in (tl.y.floor().max(y0 as f32) as usize)..(br.y.ceil().min(y1 as f32) as usize) {
            for x in (tl.x.floor().max(x0 as f32) as usize)..(br.x.ceil().min(x1 as f32) as usize) {
                let q = Vec2::new(x as f32, y as f32);
                let wa = (p[1] - q).perp_dot(p[2] - q) / area;
                let wb = (p[2] - q).perp_dot(p[0] - q) / area;
                let wc = 1.0 - wa - wb;
                if wa < 0.0 || wb < 0.0 || wc < 0.0 {
                    continue;
                }
                let i = (y - y0) * w + (x - x0);
                let d = z[0] * wa + z[1] * wb + z[2] * wc;
                if d > depth[i] {
                    depth[i] = d;
                    owner[i] = k as u16;
                }
            }
        }
    }
    let mut candidates = Vec::new();
    for y in (y0..y1).step_by(2) {
        for x in (x0..x1).step_by(2) {
            let k = owner[(y - y0) * w + (x - x0)];
            if k == u16::MAX {
                continue;
            }
            let (mut xx, mut xy, mut yy) = (0.0f32, 0.0f32, 0.0f32);
            for v in y - 3..=y + 3 {
                for u in x - 3..=x + 3 {
                    let gx = (gray.get(u + 1, v) - gray.get(u - 1, v)) * 0.5;
                    let gy = (gray.get(u, v + 1) - gray.get(u, v - 1)) * 0.5;
                    xx += gx * gx;
                    xy += gx * gy;
                    yy += gy * gy;
                }
            }
            let score = (xx + yy - ((xx - yy).powi(2) + 4.0 * xy * xy).sqrt()) * 0.5 / 49.0;
            if score >= MIN_EIGEN * 2.0 {
                candidates.push((score, Vec2::new(x as f32, y as f32), k));
            }
        }
    }
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut chosen: Vec<Spot> = Vec::new();
    for (_, q, k) in candidates {
        if chosen.len() >= MAX_POINTS {
            break;
        }
        if chosen.iter().any(|c| c.at.distance_squared(q) < MIN_GAP * MIN_GAP) {
            continue;
        }
        let t = TRIANGLES[k as usize];
        let (p, x) = (t.map(img), t.map(canon));
        let area = (p[1] - p[0]).perp_dot(p[2] - p[0]);
        let wa = (p[1] - q).perp_dot(p[2] - q) / area;
        let wb = (p[2] - q).perp_dot(p[0] - q) / area;
        let canonical = x[0] * wa + x[1] * wb + x[2] * (1.0 - wa - wb);
        let off = (to_frame(q) - proj.to_image(canonical)) / head.scale;
        let canonical = canonical + m.transpose() * Vec3::new(off.x, -off.y, 0.0);
        chosen.push(Spot { at: q, canonical, normal: (x[1] - x[0]).cross(x[2] - x[0]).normalize() });
    }
    chosen
}

/// The last frame MediaPipe's landmarks were trusted on: where the flow starts from.
pub struct Anchor {
    pub frame: Arc<Frame>,
    pub landmarks: Vec<Vec3>,
    pub head: Head,
    /// The head one frame earlier, if it was tracked: how fast it was moving.
    pub before: Option<Head>,
    pub at: Instant,
}

/// Spots being followed, and the pose they gave in the previous frame.
pub struct Flow {
    prev: Pyramid,
    points: Vec<Spot>,
    head: Head,
    before: Option<Head>,
    since: Instant,
}

impl Flow {
    /// MediaPipe confirmed the pose: the drift limit starts over.
    pub fn verified(&mut self, now: Instant) {
        self.since = now;
    }
}

/// Head pose in `frame` by following the face from the last frame (or from `anchor` when the
/// flow starts). None when too few points survive; the flow then ends.
pub fn step(flow: &mut Option<Flow>, anchor: Option<&Anchor>, frame: &Frame, now: Instant) -> Option<Head> {
    if flow.as_ref().is_some_and(|f| now - f.since > MAX_UNVERIFIED) {
        *flow = None;
        return None;
    }
    let start = match flow {
        Some(_) => None,
        None => Some(anchor.filter(|a| now - a.at < MAX_ANCHOR_AGE)?),
    };
    let (last, before) = match (&start, &*flow) {
        (Some(a), _) => (a.head, a.before),
        (None, Some(f)) => (f.head, f.before),
        (None, None) => return None,
    };
    // Search each spot where the head would be if it kept its speed: a fast but steady move
    // is then no jump at all. Searching from standstill, most spots locked onto the wrong
    // detail past ~25 px per frame and the flow gave up.
    let expected = before.map_or(last, |b| predict(&b, &last));
    // Only the face is turned grey, where it was and where it should be now.
    let region = face_region(&[last, expected]);
    let next = Pyramid::new(Gray::half(frame, Some(region)));
    if let Some(a) = start {
        let prev = Pyramid::new(Gray::half(&a.frame, Some(region)));
        let points = seed(&prev.0[0], &a.landmarks, &a.head);
        *flow = Some(Flow { prev, points, head: a.head, before: a.before, since: now });
    }
    let f = flow.as_mut()?;
    let (from, to) = (Projection::new(&f.head), Projection::new(&expected));
    let guess = |s: &Spot| to_gray(to.to_image(s.canonical)) - to_gray(from.to_image(s.canonical));
    // Spots turning edge-on are dropped before they mislead the fit.
    let facing = |s: &Spot, head: &Head| (head.rotation * s.normal).z;
    let moved: Vec<(Spot, Vec2)> = f
        .points
        .iter()
        .filter(|s| facing(s, &f.head) >= MIN_FACING && facing(s, &expected) >= MIN_FACING)
        .filter_map(|s| track(&f.prev, &next, s.at, guess(s)).map(|q| (*s, q)))
        .collect();
    let Some((head, inliers)) = robust_pose(&moved, &expected) else {
        *flow = None;
        return None;
    };
    f.points = inliers;
    if f.points.len() < RESEED_BELOW {
        // Spots die off as the face turns away; new ones on what is visible now, placed on
        // the mesh through the pose just fitted.
        for s in seed(&next.0[0], &synthetic(&head), &head) {
            if f.points.len() >= MAX_POINTS {
                break;
            }
            if f.points.iter().all(|p| p.at.distance_squared(s.at) >= MIN_GAP * MIN_GAP) {
                f.points.push(s);
            }
        }
    }
    f.prev = next;
    f.before = Some(f.head);
    f.head = head;
    Some(head)
}

/// Frame pixels around the face at these poses, with room for the tracking windows and
/// whatever the prediction got wrong.
fn face_region(heads: &[Head]) -> (Vec2, Vec2) {
    let (mut lo, mut hi) = (Vec2::MAX, Vec2::MIN);
    for h in heads {
        for p in &synthetic(h)[..VERTICES.len()] {
            lo = lo.min(p.truncate());
            hi = hi.max(p.truncate());
        }
    }
    // A window at the coarsest level spans RADIUS << (LEVELS - 1) half-res pixels.
    let pad = (hi - lo).max_element() * 0.5 + (RADIUS << LEVELS) as f32;
    (lo - Vec2::splat(pad), hi + Vec2::splat(pad))
}

/// The head one frame after `now` if it keeps moving as it did from `before` (one frame
/// earlier): same turn, same shift, same change in size.
pub fn predict(before: &Head, now: &Head) -> Head {
    let turn = now.rotation * before.rotation.inverse();
    Head { rotation: (turn * now.rotation).normalize(), pivot: now.pivot * 2.0 - before.pivot, scale: now.scale * (now.scale / before.scale).clamp(0.8, 1.25) }
}

/// Pose fitted to the followed spots (Huber-reweighted: a few slide along the face edge or get
/// covered; and weighted by how squarely they face the camera), and the spots that agree.
fn robust_pose(moved: &[(Spot, Vec2)], previous: &Head) -> Option<(Head, Vec<Spot>)> {
    if moved.len() < MIN_POINTS {
        return None;
    }
    let target = |s: &Spot, q: Vec2| (s.canonical, to_frame(q));
    let view = |s: &Spot, head: &Head| (head.rotation * s.normal).z.max(0.0);
    let mut weights: Vec<f32> = moved.iter().map(|(s, _)| view(s, previous)).collect();
    let huber = HUBER_PX.max(HUBER_CM * previous.scale);
    let mut head = None;
    for _ in 0..4 {
        let weighted: Vec<(Vec3, Vec2, f32)> = moved.iter().zip(&weights).map(|((s, q), &w)| {
            let (x, p) = target(s, *q);
            (x, p, w)
        }).collect();
        let h = fit_pose(&weighted)?;
        let proj = Projection::new(&h);
        for (w, (s, q)) in weights.iter_mut().zip(moved) {
            let (x, p) = target(s, *q);
            *w = (huber / (proj.to_image(x) - p).length().max(1e-6)).min(1.0) * view(s, &h);
        }
        head = Some(h);
    }
    let head = head?;
    let proj = Projection::new(&head);
    let error = |s: &Spot, q: Vec2| {
        let (x, p) = target(s, q);
        (proj.to_image(x) - p).length()
    };
    let inliers: Vec<(Spot, f32)> = moved.iter().map(|(s, q)| (Spot { at: *q, ..*s }, error(s, *q))).filter(|(_, e)| *e < 3.0 * huber).collect();
    let rms = (inliers.iter().map(|(_, e)| e * e).sum::<f32>() / inliers.len().max(1) as f32).sqrt();
    (inliers.len() >= MIN_POINTS && rms < MAX_RMS_PX.max(MAX_RMS_CM * head.scale)).then(|| (head, inliers.into_iter().map(|(s, _)| s).collect()))
}

/// Landmarks of the rigid canonical face at this pose, so everything downstream (crop region,
/// filters, cutouts) runs unchanged while the flow drives. No expression: lids and lips rest.
pub fn synthetic(head: &Head) -> Vec<Vec3> {
    let proj = Projection::new(head);
    let mut lm: Vec<Vec3> = VERTICES.iter().map(|&v| proj.to_image(Vec3::from(v)).extend(0.0)).collect();
    let centre = |lm: &[Vec3], ring: &[usize]| ring.iter().map(|&i| lm[i]).sum::<Vec3>() / ring.len() as f32;
    let (right, left) = (centre(&lm, RIGHT_EYE), centre(&lm, LEFT_EYE));
    lm.extend([right; 5]); // irises 468..473, 473..478
    lm.extend([left; 5]);
    lm
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::Quat;

    /// Gaussian blobs at random places: smooth, not periodic (one right answer), and exact at
    /// any sub-pixel shift (no resampling blur in the test itself).
    fn blobs(shift: Vec2) -> Gray {
        let mut seed = 777u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        let blobs: Vec<(Vec2, f32, f32)> = (0..120).map(|_| (Vec2::new(rnd() * 200.0, rnd() * 150.0), 2.5 + rnd() * 5.0, rnd() - 0.5)).collect();
        let f = |p: Vec2| 0.5 + blobs.iter().map(|&(c, s, a)| a * (-(p - c).length_squared() / (2.0 * s * s)).exp()).sum::<f32>();
        Gray::new(200, 150, (0..200 * 150).map(|i| f(Vec2::new((i % 200) as f32, (i / 200) as f32) - shift)).collect())
    }

    #[test]
    fn lucas_kanade_follows_a_shift() {
        // Sub-pixel, and larger than the window (needs the pyramid).
        for shift in [Vec2::new(0.4, -0.3), Vec2::new(6.3, -4.6), Vec2::new(-17.2, 11.5)] {
            let (a, b) = (Pyramid::new(blobs(Vec2::ZERO)), Pyramid::new(blobs(shift)));
            for p in [Vec2::new(70.0, 60.0), Vec2::new(120.0, 85.0)] {
                let q = track(&a, &b, p, Vec2::ZERO).unwrap();
                assert!((q - p - shift).length() < 0.06, "shift {shift:?}: found {:?}", q - p);
            }
        }
    }

    /// Skin-like detail at several scales, varying over the whole surface (also the sides).
    fn skin(x: Vec3) -> [f32; 3] {
        let v = 0.5
            + 0.16 * (3.1 * x.x).sin() * (2.7 * x.y).sin()
            + 0.12 * (7.3 * x.x + 1.1 * x.y + 2.3 * x.z).sin()
            + 0.12 * (5.9 * x.y - 4.1 * x.z).sin()
            + 0.08 * (6.7 * x.z + 3.3 * x.x).cos();
        [v, v * 0.85, v * 0.75]
    }

    #[test]
    fn flow_carries_the_pose_past_the_landmark_model() {
        // MediaPipe last sure at 50 degrees (and looking a bit down), then the head keeps
        // turning to 86 in 4 degree steps per frame, and back.
        let pose = |deg: f32| Quat::from_rotation_y(deg.to_radians()) * Quat::from_rotation_x(0.25);
        let (frame, landmarks, head) = crate::testface::render(pose(50.0), skin);
        let t0 = Instant::now();
        let anchor = Anchor { frame: Arc::new(frame), landmarks, head, before: None, at: t0 };
        let mut flow = None;
        let path: Vec<i32> = (54..=86).step_by(4).chain((58..=82).rev().step_by(4)).collect();
        for (k, deg) in path.into_iter().enumerate() {
            let (frame, _, truth) = crate::testface::render(pose(deg as f32), skin);
            let at = t0 + Duration::from_millis(40 * (k as u64 + 1));
            let got = step(&mut flow, Some(&anchor), &frame, at).unwrap_or_else(|| panic!("lost the head at {deg} degrees"));
            let err = got.rotation.angle_between(truth.rotation).to_degrees();
            assert!(err < 3.0, "{deg} degrees: off by {err:.1}");
            assert!((got.scale / truth.scale - 1.0).abs() < 0.05, "{deg} degrees: scale {}", got.scale);
        }
    }

    #[test]
    fn untextured_points_are_refused() {
        let flat = Pyramid::new(Gray::new(100, 100, vec![0.5; 100 * 100]));
        assert!(track(&flat, &flat, Vec2::new(50.0, 50.0), Vec2::ZERO).is_none());
    }

    #[test]
    fn flow_keeps_up_with_fast_motion() {
        // MediaPipe loses the face mid-move: a quick sweep sideways while turning, already
        // under way when the flow takes over (it knows the last two trusted poses).
        let (dx, deg) = (45.0, 8.0);
        let at = |k: i32| Head { rotation: Quat::from_rotation_y((deg * k as f32).to_radians()), pivot: Vec2::new(150.0 + dx * k as f32, 300.0), scale: 19.0 };
        let (frame, landmarks, head) = crate::testface::render_at(at(0), skin);
        let t0 = Instant::now();
        let anchor = Anchor { frame: Arc::new(frame), landmarks, head, before: Some(at(-1)), at: t0 };
        let mut flow = None;
        for k in 1..=6 {
            let (frame, _, truth) = crate::testface::render_at(at(k), skin);
            let got = step(&mut flow, Some(&anchor), &frame, t0 + Duration::from_millis(33 * k as u64)).unwrap_or_else(|| panic!("lost the head at frame {k}"));
            let err = got.rotation.angle_between(truth.rotation).to_degrees();
            // ~240 degrees per second drifts a little (the window turns too); losing it is the bug.
            assert!(err < 4.0 && (got.pivot - truth.pivot).length() < 0.75 * truth.scale, "frame {k}: off by {err:.1} degrees, {:?}", got.pivot - truth.pivot);
        }
    }
}
