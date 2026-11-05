//! Layer 3: the real eyes and mouth, cut out of the webcam frame. Each one becomes a texture
//! plus a size and position on the model's face; the avatar puts it on a mesh patch that hugs
//! the model surface, so Bevy renders it as part of the body.
//!
//! The cut is a 3D frontalization against MediaPipe's canonical face mesh: every live
//! landmark is traced back through the head pose to where it sits on a face seen straight on
//! (its depth taken from the canonical mesh), and the camera image is resampled triangle by
//! triangle into that frontal view. Blinks and the open mouth survive (the live points move),
//! while turning, tilting and the face's own curvature (outward-facing eyes, lips wrapping
//! round) are undone exactly. Eye triangles the camera sees edge-on or folded fade out
//! instead of smearing their few pixels; the mouth is kept whole (see `mouth_depth`).

use crate::canonical_mesh::{TRIANGLES, VERTICES};
use crate::config::Calibration;
use crate::tracker::{Frame, Head, NECK};
use bevy::math::{Mat3, Vec2, Vec3};
use tiny_skia::{FillRule, Mask, PathBuilder, Transform};

/// Contour rings (MediaPipe indices) used as cutout masks: just the eye and just the lips,
/// no brows. When you blink the eye ring closes to a slit and the cutout follows.
pub const RIGHT_EYE: &[usize] = &[33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7];
pub const LEFT_EYE: &[usize] = &[263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249];
pub const LIPS: &[usize] = &[61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146];
/// Inner lip line: the opening of the mouth.
const INNER_LIPS: &[usize] = &[78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];

/// Grid cells per side of a feature patch and its relief: fine enough for an eyeball's dome
/// and the lips' roll.
pub const RELIEF_CELLS: usize = 24;

/// The fixed scene camera, Three.js style: at (0,0,4.2) looking down -z, 35 degree vertical fov.
pub const CAMERA_Z: f32 = 4.2;
pub const FOV_DEG: f32 = 35.0;

/// Texture side of each feature. The crop box is stretched into it; the patch on the model
/// carries the real aspect, so this only sets resolution.
pub const TEX: u32 = 256;

/// Model units per eye-centre distance at face_scale 1: outer corners span 0.42, and the
/// canonical face's corner span is 1.41 eye-centre distances.
const UNIT: f32 = 0.42 / 1.41;
/// Eye-centre distance of the canonical face, cm.
const EYE_CENTERS_CM: f32 = 6.30;
/// Fixed crop boxes on the frontal face, cm, before dilation/margin: an open eye is about
/// 3 x 1.1 cm, lips 4.9 cm wide and up to ~4.5 cm tall when open. The box does not follow
/// the live contour (that jittered the patch size); only the mask inside it moves.
const EYE_BOX: Vec2 = Vec2::new(2.96, 1.51);
const MOUTH_BOX: Vec2 = Vec2::new(4.91, 4.54);

/// Outward direction of each feature on the canonical face, for the whole-feature fade:
/// real eyes face ~13 degrees outward (canonical corners 33/133 and 362/263).
const RIGHT_EYE_NORMAL: Vec3 = Vec3::new(-0.22, 0.0, 0.975);
const LEFT_EYE_NORMAL: Vec3 = Vec3::new(0.22, 0.0, 0.975);

/// How open each eye usually is (lid gap over width), learned over ~10 s of frames: the
/// expressive eyes grow when they open wider than *your* normal, not a model face's.
/// ponytail: process-global, fine for one tracked face.
static EYE_REST: std::sync::Mutex<[f32; 2]> = std::sync::Mutex::new([0.0; 2]);

/// One eye or the mouth, ready to put on the model.
pub struct Feature {
    /// TEX x TEX straight-alpha RGBA, already mirrored and colour-filtered: the real pixels,
    /// alpha covering the cut.
    pub rgba: Vec<u8>,
    /// Patch centre and half size on the model's face plane (head-local x right, y up).
    pub center: Vec2,
    pub half: Vec2,
    /// How visible the real feature is to the camera: the far eye fades out as the nose
    /// hides it instead of smearing.
    pub alpha: f32,
    /// Height over the patch plane (model units, toward the viewer), (RELIEF_CELLS + 1)^2
    /// in texture order (row by row, top down, mirrored like the texture): lids, eyeball,
    /// lips, the open mouth's depth.
    pub relief: Vec<f32>,
}

/// Head pose as a projection: canonical point (cm) -> frame pixel, and back for a point of
/// known canonical depth.
pub struct Projection {
    rows: [Vec3; 2],
    pivot: Vec2, // camera space, y up
    scale: f32,
}

impl Projection {
    pub fn new(head: &Head) -> Self {
        let m = Mat3::from_quat(head.rotation);
        Self { rows: [m.row(0), m.row(1)], pivot: Vec2::new(head.pivot.x, -head.pivot.y), scale: head.scale }
    }

    pub fn to_image(&self, x: Vec3) -> Vec2 {
        let y = x - NECK;
        let p = self.pivot + Vec2::new(self.rows[0].dot(y), self.rows[1].dot(y)) * self.scale;
        Vec2::new(p.x, -p.y)
    }

    /// Frontal (canonical x, y in cm, y up) of an image point whose canonical depth is `z`.
    /// None when the face is seen too edge-on to invert.
    fn to_frontal(&self, p: Vec2, z: f32) -> Option<Vec2> {
        let q = (Vec2::new(p.x, -p.y) - self.pivot) / self.scale;
        let [r0, r1] = self.rows;
        let w = z - NECK.z;
        let (bx, by) = (q.x - r0.z * w, q.y - r1.z * w);
        let det = r0.x * r1.y - r0.y * r1.x;
        // Only truly edge-on faces fail; steep views are handled per triangle (they fade).
        (det > 0.08).then(|| Vec2::new((bx * r1.y - by * r0.y) / det, (r0.x * by - r1.x * bx) / det) + NECK.truncate())
    }
}

/// The three features for this frame, or None without a head pose or when the face is seen
/// too edge-on.
pub fn features(frame: &Frame, landmarks: &[Vec3], head: Option<Head>, cfg: &Calibration) -> Option<[Feature; 3]> {
    let head = head?;
    let proj = Projection::new(&head);
    // Every live landmark on the frontal face (cm). This is where the expression lives.
    let frontal: Vec<Vec2> = (0..VERTICES.len())
        .map(|i| proj.to_frontal(landmarks[i].truncate(), VERTICES[i][2]))
        .collect::<Option<_>>()?;
    let centroid = |ring: &[usize]| ring.iter().map(|&i| frontal[i]).sum::<Vec2>() / ring.len() as f32;
    let mid = (centroid(RIGHT_EYE) + centroid(LEFT_EYE)) * 0.5;
    let model_per_cm = cfg.face_scale * UNIT / EYE_CENTERS_CM;
    let side = if cfg.mirror { -1.0 } else { 1.0 };
    // Self-calibrating exposure: the cutouts are scaled so your skin lands near mid grey. A
    // dim room otherwise turns eyes and mouth into dark smudges on a brightly lit model.
    let gain = if cfg.auto_exposure { (0.5 / skin_luma(frame, landmarks)).clamp(1.0, 2.2) } else { 1.0 };

    let flat: Vec<Vec2> = (0..VERTICES.len()).map(|i| proj.to_frontal(landmarks[i].truncate(), MOUTH_Z[i])).collect::<Option<_>>()?;
    let one = |ring: &[usize], shape: &[Vec2], normal: Vec3, box_cm: Vec2, dilation: f32, margin: f32, softness: f32, size: f32, offset: Vec2| {
        // Whole-feature fade from the real pose: the far eye goes behind the nose.
        let facing = (head.rotation * normal.normalize()).z;
        // The eyes fade early (the nose covers the far one long before it turns edge-on);
        // the mouth has nothing in front of it, its triangles fade on their own.
        let alpha = if normal == Vec3::Z { (facing - 0.1) / 0.15 } else { (facing - 0.3) / 0.2 }.clamp(0.0, 1.0);
        if alpha <= 0.0 {
            return Feature { rgba: Vec::new(), center: Vec2::ZERO, half: Vec2::ONE, alpha, relief: Vec::new() };
        }
        // The skin border is as wide as `dilation` used to add top and bottom on the canonical
        // feature, but now even all round.
        let canonical: Vec<Vec2> = ring.iter().map(|&i| Vec3::from(VERTICES[i]).truncate()).collect();
        let border = (dilation - 1.0) * 0.5 * (bounds(&canonical).1.y - bounds(&canonical).0.y);
        let contour = grow(&ring.iter().map(|&i| shape[i]).collect::<Vec<_>>(), border);
        let (lo, hi) = bounds(&contour);
        let centre = (lo + hi) * 0.5;
        let placed = {
            let (lo, hi) = bounds(&grow(&ring.iter().map(|&i| frontal[i]).collect::<Vec<_>>(), border));
            (lo + hi) * 0.5
        };
        // Fixed box; grows only if the contour would otherwise be clipped (a huge yawn).
        let box_size = (box_cm * dilation).max(hi - lo) * margin;
        let at = (placed - mid) * model_per_cm * cfg.spacing;
        // The soft edge follows the feature as it is now, not the box sized for a wide-open
        // mouth: on closed lips (~1 cm tall) that edge ate half the lip and blurred it. Even
        // all round, from the height: one from the width blurred the corners away.
        let slot = if normal == Vec3::Z { 2 } else { (normal.x > 0.0) as usize };
        let feather = Vec2::splat((hi - lo).min(box_size).y * softness * 0.18);
        let mut rgba = texture(frame, landmarks, shape, &head, (centre, box_size), (&contour, feather), normal != Vec3::Z, slot, cfg, gain);
        let eye = normal != Vec3::Z;
        let depth: &[f32] = if eye { &CANONICAL_Z } else { &MOUTH_Z };
        let relief = relief(shape, depth, ring, eye, &contour, (centre, box_size), cfg.mirror)
            .into_iter()
            .map(|h| h * cfg.relief * model_per_cm * size)
            .collect();
        let mut stretch = Vec2::ONE;
        if normal != Vec3::Z {
            // Ring order: corners at 0 and 8, upper and lower lid at 4 and 12.
            let width = frontal[ring[0]].distance(frontal[ring[8]]);
            let open = frontal[ring[4]].distance(frontal[ring[12]]) / width.max(1e-3);
            let rest = {
                let mut rest = EYE_REST.lock().unwrap();
                let r = &mut rest[(normal.x > 0.0) as usize];
                *r = if *r == 0.0 { open } else { *r + (open - *r) * 0.003 };
                *r
            };
            // Wide open grows the eye, squinting shrinks it; mostly in height.
            // Small changes are landmark jitter, not expression: they would make the eye pump.
            let ratio = (open / rest).clamp(0.3, 2.5);
            let ratio = 1.0 + (ratio - 1.0).signum() * ((ratio - 1.0).abs() - 0.08).max(0.0);
            stretch = Vec2::new(ratio.powf(0.3), (1.0 + cfg.eye_round) * ratio);
            if cfg.eye_bulge > 0.0 {
                // Magnify around the iris (whichever iris landmark sits in this eye).
                let ring_centre = ring.iter().map(|&i| landmarks[i].truncate()).sum::<Vec2>() / ring.len() as f32;
                let iris = [468, 473].map(|i| landmarks[i].truncate()).into_iter().min_by(|a, b| a.distance(ring_centre).total_cmp(&b.distance(ring_centre))).unwrap();
                let z = ring.iter().map(|&i| VERTICES[i][2]).sum::<f32>() / ring.len() as f32;
                if let Some(f) = proj.to_frontal(iris, z) {
                    let t = TEX as f32;
                    let mut c = Vec2::new((f.x - centre.x) / box_size.x + 0.5, (centre.y - f.y) / box_size.y + 0.5) * t;
                    if cfg.mirror {
                        c.x = t - c.x;
                    }
                    bulge(&mut rgba, c, Vec2::splat(0.55 * width) / box_size * t, cfg.eye_bulge * 0.5);
                }
            }
        }
        Feature {
            rgba,
            center: Vec2::new(cfg.face_x + at.x * side, cfg.face_y + at.y) + offset,
            half: box_size * 0.5 * model_per_cm * size * stretch,
            alpha,
            relief,
        }
    };
    // Eye x offset is symmetric: positive pushes both apart. Mirror decides which is which.
    let eye = |subject_side: f32| Vec2::new(cfg.eye_x * subject_side * side * -1.0, cfg.eye_y);
    // One thread each: ~7 ms per feature would otherwise stall the render loop.
    let (one, frontal, flat) = (&one, &frontal, &flat);
    Some(std::thread::scope(|s| {
        [
            s.spawn(move || one(RIGHT_EYE, frontal, RIGHT_EYE_NORMAL, EYE_BOX, cfg.eye_dilation, cfg.eye_margin, cfg.eye_softness, cfg.eye_size, eye(-1.0))),
            s.spawn(move || one(LEFT_EYE, frontal, LEFT_EYE_NORMAL, EYE_BOX, cfg.eye_dilation, cfg.eye_margin, cfg.eye_softness, cfg.eye_size, eye(1.0))),
            s.spawn(move || one(LIPS, flat, Vec3::Z, MOUTH_BOX, cfg.mouth_dilation, cfg.mouth_margin, cfg.mouth_softness, cfg.mouth_size, Vec2::new(cfg.mouth_x, cfg.mouth_y))),
        ]
        .map(|h| h.join().unwrap())
    }))
}

/// Your skin's brightness this frame, from forehead, cheeks and chin (rarely covered by
/// glasses, brows or the eye sockets' shadow). Median, so a stray hair doesn't move it.
fn skin_luma(frame: &Frame, landmarks: &[Vec3]) -> f32 {
    let mut y = Vec::new();
    for i in [10, 50, 280, 199] {
        let c = landmarks[i].truncate();
        for dy in -3..=3 {
            for dx in -3..=3 {
                let [r, g, b] = sample(frame, c + Vec2::new(dx as f32, dy as f32) * 2.0);
                y.push((0.299 * r + 0.587 * g + 0.114 * b).max(0.02));
            }
        }
    }
    y.sort_by(f32::total_cmp);
    y[y.len() / 2]
}

/// Grow a closed contour outward by `d`, evenly all round, so it keeps the feature's shape.
/// (Scaling from the centroid stretched the corners into long wings of skin.)
fn grow(points: &[Vec2], d: f32) -> Vec<Vec2> {
    let n = points.len();
    let area: f32 = (0..n).map(|i| points[i].perp_dot(points[(i + 1) % n])).sum();
    let out = |a: Vec2, b: Vec2| -(b - a).perp().normalize_or_zero() * area.signum();
    (0..n)
        .map(|i| {
            let (a, p, b) = (points[(i + n - 1) % n], points[i], points[(i + 1) % n]);
            // No mitre: it shot the pointed eye and mouth corners out.
            p + (out(a, p) + out(p, b)).normalize_or_zero() * d
        })
        .collect()
}

/// Depth the mouth is cut at: the true canonical depth on and outside the lip contour,
/// inside it a smooth sheet spanned between the upper and lower contour. With each
/// landmark's own depth the lips folded over themselves whenever yours don't match the
/// canonical pout (pressed, curled in, a wide smile): the folded triangles were dropped and
/// the model showed through. The sheet can't fold; the outline and its skin stay exact.
fn mouth_depth(i: usize) -> f32 {
    let v = Vec3::from(VERTICES[i]);
    let ring: Vec<Vec3> = LIPS.iter().map(|&i| Vec3::from(VERTICES[i])).collect();
    if LIPS.contains(&i) || !inside(&ring.iter().map(|p| p.truncate()).collect::<Vec<_>>(), v.truncate()) {
        return v.z;
    }
    // Upper contour runs left to right (corner 61 to corner 291), the lower one back.
    let upper = &ring[..=10];
    let mut lower = ring[10..].to_vec();
    lower.push(ring[0]);
    lower.reverse();
    let along = |chain: &[Vec3]| {
        let k = chain.windows(2).position(|w| v.x <= w[1].x).unwrap_or(chain.len() - 2);
        let (a, b) = (chain[k], chain[k + 1]);
        a.lerp(b, ((v.x - a.x) / (b.x - a.x)).clamp(0.0, 1.0))
    };
    let (u, l) = (along(upper), along(&lower));
    u.z + (l.z - u.z) * ((v.y - u.y) / (l.y - u.y)).clamp(0.0, 1.0)
}

static CANONICAL_Z: std::sync::LazyLock<Vec<f32>> = std::sync::LazyLock::new(|| VERTICES.iter().map(|v| v[2]).collect());
static MOUTH_Z: std::sync::LazyLock<Vec<f32>> = std::sync::LazyLock::new(|| (0..VERTICES.len()).map(mouth_depth).collect());

/// The feature's own relief (cm, toward the viewer) at the patch grid over the crop box. The
/// canonical face's depth laid through the live points (so lids and lips move with you),
/// minus the plane of the feature's rim, so only its shape stands out, not the face's
/// curvature or the eyes' outward turn. On top: the eyeball's dome in the opening, and the
/// open mouth falling back into the head. Zero outside the cut, smooth into it.
fn relief(shape: &[Vec2], depth: &[f32], ring: &[usize], eye: bool, contour: &[Vec2], (centre, size): (Vec2, Vec2), mirror: bool) -> Vec<f32> {
    let n = RELIEF_CELLS;
    let g = n + 1;
    // Grid point (i, j) -> frontal cm; i runs along the (possibly mirrored) texture.
    let u = |i: usize| if mirror { 1.0 - i as f32 / n as f32 } else { i as f32 / n as f32 };
    let at = |i: usize, j: usize| Vec2::new(centre.x + (u(i) - 0.5) * size.x, centre.y + (0.5 - j as f32 / n as f32) * size.y);
    let to_grid = |p: Vec2| {
        let x = (p.x - centre.x) / size.x + 0.5;
        Vec2::new(if mirror { 1.0 - x } else { x }, 0.5 - (p.y - centre.y) / size.y) * n as f32
    };
    let mut h = vec![f32::NEG_INFINITY; g * g];
    for tri in &TRIANGLES {
        let [a, b, c] = tri.map(|i| i as usize);
        let (pa, pb, pc) = (shape[a], shape[b], shape[c]);
        let area = (pb - pa).perp_dot(pc - pa);
        if area.abs() < 1e-9 {
            continue;
        }
        let (ga, gb, gc) = (to_grid(pa), to_grid(pb), to_grid(pc));
        let (lo, hi) = (ga.min(gb).min(gc), ga.max(gb).max(gc));
        if hi.x < 0.0 || hi.y < 0.0 || lo.x > n as f32 || lo.y > n as f32 {
            continue;
        }
        for j in lo.y.ceil().max(0.0) as usize..=(hi.y.floor().min(n as f32)) as usize {
            for i in lo.x.ceil().max(0.0) as usize..=(hi.x.floor().min(n as f32)) as usize {
                let q = at(i, j);
                let (wa, wb) = ((pb - q).perp_dot(pc - q) / area, (pc - q).perp_dot(pa - q) / area);
                let wc = 1.0 - wa - wb;
                if wa >= -1e-4 && wb >= -1e-4 && wc >= -1e-4 {
                    let k = j * g + i;
                    h[k] = h[k].max(depth[a] * wa + depth[b] * wb + depth[c] * wc);
                }
            }
        }
    }
    // Least-squares plane z = ax + by + c through the rim: what is left is the feature itself.
    let rim: Vec<Vec3> = ring.iter().map(|&i| shape[i].extend(depth[i])).collect();
    let (mut m, mut r) = (Mat3::ZERO, Vec3::ZERO);
    for p in &rim {
        let v = Vec3::new(p.x, p.y, 1.0);
        m += Mat3::from_cols(v * v.x, v * v.y, v * v.z);
        r += v * p.z;
    }
    let plane = if m.determinant().abs() > 1e-9 { m.inverse() * r } else { Vec3::new(0.0, 0.0, rim.iter().map(|p| p.z).sum::<f32>() / rim.len() as f32) };
    let opening: Vec<Vec2> = if eye { ring.iter().map(|&i| shape[i]).collect() } else { INNER_LIPS.iter().map(|&i| shape[i]).collect() };
    let (olo, ohi) = bounds(&opening);
    let (oc, or) = ((olo + ohi) * 0.5, ((ohi - olo) * 0.5).max(Vec2::splat(1e-3)));
    for j in 0..g {
        for i in 0..g {
            let (k, q) = (j * g + i, at(i, j));
            if !inside(contour, q) {
                h[k] = 0.0;
                continue;
            }
            let base = if h[k].is_finite() { h[k] - (plane.x * q.x + plane.y * q.y + plane.z) } else { 0.0 };
            let d = (q - oc) / or;
            let r2 = d.length_squared();
            h[k] = base
                + if !inside(&opening, q) {
                    0.0
                } else if eye {
                    // Eyeball: a dome, as high as half the opening (a blink flattens it).
                    0.5 * or.y * (1.0 - r2).max(0.0).sqrt()
                } else {
                    // Open mouth: back into the head, as deep as half the opening is tall.
                    -0.5 * or.y * (1.0 - r2).max(0.0).sqrt()
                };
        }
    }
    // One box pass: the canonical mesh's facets and the cut's step become smooth slopes.
    let mut out = vec![0.0; g * g];
    for j in 0..g {
        for i in 0..g {
            let (mut sum, mut w) = (0.0, 0.0);
            for (dj, di) in [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)] {
                let (y, x) = (j as i32 + dj, i as i32 + di);
                if (0..g as i32).contains(&x) && (0..g as i32).contains(&y) {
                    sum += h[y as usize * g + x as usize];
                    w += 1.0;
                }
            }
            out[j * g + i] = sum / w;
        }
    }
    out
}

/// Point in polygon (even-odd).
fn inside(ring: &[Vec2], p: Vec2) -> bool {
    let mut odd = false;
    for (i, &a) in ring.iter().enumerate() {
        let b = ring[(i + 1) % ring.len()];
        if (a.y > p.y) != (b.y > p.y) && p.x < a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x) {
            odd = !odd;
        }
    }
    odd
}

fn bounds(points: &[Vec2]) -> (Vec2, Vec2) {
    points.iter().fold((Vec2::MAX, Vec2::MIN), |(lo, hi), p| (lo.min(*p), hi.max(*p)))
}

/// The feature texture: the frontal face inside the box, resampled from the camera triangle
/// by triangle (piecewise affine through the live landmarks), times a feathered mask from the
/// contour, times each triangle's visibility. Mirrored with the display.
fn texture(
    frame: &Frame,
    landmarks: &[Vec3],
    frontal: &[Vec2],
    head: &Head,
    (centre, size): (Vec2, Vec2),
    (contour, feather): (&[Vec2], Vec2),
    fade: bool,
    slot: usize,
    cfg: &Calibration,
    gain: f32,
) -> Vec<u8> {
    let t = TEX as f32;
    let (w, n) = (TEX as usize, (TEX * TEX) as usize);
    let top_left = Vec2::new(centre.x - size.x * 0.5, centre.y + size.y * 0.5);
    // Frontal cm (y up) -> texel coords (y down), unmirrored.
    let to_tex = |f: Vec2| Vec2::new((f.x - top_left.x) / size.x, (top_left.y - f.y) / size.y) * t;
    let texels_per_px = t / (size.x * head.scale).max(1.0);

    // Closed Catmull-Rom curve through every landmark: corners and the lip's bow stay where
    // they are (a curve through the midpoints rounded them off), without facets.
    let pts: Vec<Vec2> = contour.iter().map(|&p| to_tex(p)).collect();
    let k = pts.len();
    let mut pb = PathBuilder::new();
    pb.move_to(pts[0].x, pts[0].y);
    for i in 0..k {
        let (p0, p1, p2, p3) = (pts[(i + k - 1) % k], pts[i], pts[(i + 1) % k], pts[(i + 2) % k]);
        let (c1, c2) = (p1 + (p2 - p0) / 6.0, p2 - (p3 - p1) / 6.0);
        pb.cubic_to(c1.x, c1.y, c2.x, c2.y, p2.x, p2.y);
    }
    pb.close();
    let mut mask = Mask::new(TEX, TEX).unwrap();
    if let Some(path) = pb.finish() {
        mask.fill_path(&path, FillRule::Winding, true, Transform::identity());
    }
    let shape: Vec<f32> = mask.data().iter().map(|&a| a as f32 / 255.0).collect();
    // The cut: feathered inward only (the blurred edge sits at 50% on the contour; remapping
    // 50% to 0 keeps everything outside the contour out of it).
    let mut cut = shape;
    blur_xy(&mut cut, w, w, (feather / size * t).max(Vec2::splat(0.5)));

    let mut rgb = [vec![0f32; n], vec![0f32; n], vec![0f32; n]];
    let mut seen = vec![0f32; n];
    for tri in &TRIANGLES {
        let [a, b, c] = tri.map(|i| i as usize);
        let (ta, tb, tc) = (to_tex(frontal[a]), to_tex(frontal[b]), to_tex(frontal[c]));
        let (lo, hi) = (ta.min(tb).min(tc), ta.max(tb).max(tc));
        if hi.x < 0.0 || hi.y < 0.0 || lo.x > t || lo.y > t {
            continue;
        }
        let area = (tb - ta).perp_dot(tc - ta);
        if area.abs() < 1e-6 {
            continue;
        }
        let (pa, pb, pc) = (landmarks[a].truncate(), landmarks[b].truncate(), landmarks[c].truncate());
        // Visibility: image area over frontal area (texel y is down, image y is down, so
        // the signs agree). ~cos of the angle to the camera; negative = folded behind
        // something (the nose over the far eye).
        let image_area = (pb - pa).perp_dot(pc - pa);
        let frontal_area = area / (t * t) * size.x * size.y * head.scale * head.scale;
        let ratio = image_area / frontal_area;
        // Eyes: steep triangles fade (the far eye behind the nose). The mouth has nothing in
        // front of it, and fading or dropping its steep and folded parts left see-through
        // patches: folded ones only fill what nothing else covers (a squeezed, but real, bit).
        let visible = if fade { ((ratio - 0.15) / 0.25).clamp(0.0, 1.0) } else if ratio > 0.0 { 1.0 } else { 0.99 };
        if visible <= 0.0 {
            continue;
        }
        let (x0, x1) = (lo.x.floor().max(0.0) as u32, (hi.x.ceil() as u32).min(TEX));
        let (y0, y1) = (lo.y.floor().max(0.0) as u32, (hi.y.ceil() as u32).min(TEX));
        for y in y0..y1 {
            for x in x0..x1 {
                let q = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let wa = (tb - q).perp_dot(tc - q) / area;
                let wb = (tc - q).perp_dot(ta - q) / area;
                let wc = 1.0 - wa - wb;
                if wa < -1e-4 || wb < -1e-4 || wc < -1e-4 {
                    continue;
                }
                let i = (y * TEX + x) as usize;
                // Only texels that end up visible are sampled.
                if seen[i] >= visible || cut[i] <= 0.5 {
                    continue;
                }
                let c = sample(frame, pa * wa + pb * wb + pc * wc);
                for ch in 0..3 {
                    rgb[ch][i] = c[ch];
                }
                seen[i] = visible;
            }
        }
    }
    // Denoised on the camera's own values (before brightening multiplies the noise), then
    // coloured.
    if cfg.denoise > 0.0 {
        denoise(&mut HISTORY[slot].lock().unwrap(), &mut rgb, &seen, cfg.denoise);
    }
    for i in (0..n).filter(|&i| seen[i] > 0.0) {
        let c = filter([rgb[0][i], rgb[1][i], rgb[2][i]], cfg, gain);
        for ch in 0..3 {
            rgb[ch][i] = c[ch];
        }
    }
    // Unsharp mask at ~1.2 camera pixels: the crop is upscaled several times and the webcam
    // image is soft to begin with.
    if cfg.sharpen > 0.0 {
        for ch in &mut rgb {
            let mut soft = ch.clone();
            blur_f32(&mut soft, w, w, 1.2 * texels_per_px);
            for (c, b) in ch.iter_mut().zip(soft) {
                *c += cfg.sharpen * (*c - b);
            }
        }
    }

    let mut rgba = vec![0u8; n * 4];
    for i in 0..n {
        let a = ((cut[i] - 0.5) * 2.0).clamp(0.0, 1.0) * seen[i];
        if a < 1.0 / 255.0 {
            continue;
        }
        let (x, y) = (i % w, i / w);
        let out = y * w + if cfg.mirror { w - 1 - x } else { x };
        let px = |ch: usize| (rgb[ch][i].clamp(0.0, 1.0) * 255.0) as u8;
        rgba[out * 4..out * 4 + 4].copy_from_slice(&[px(0), px(1), px(2), (a * 255.0) as u8]);
    }
    rgba
}

/// Last denoised texture per feature (right eye, left eye, mouth).
/// ponytail: process-global like EYE_REST, fine for one tracked face.
static HISTORY: [std::sync::Mutex<Vec<[f32; 3]>>; 3] = [const { std::sync::Mutex::new(Vec::new()) }; 3];

/// Temporal denoising in the frontal view. The texture holds still while the head moves, so
/// webcam noise (new every frame) averages out over a few frames, while a real change (a
/// blink, the mouth moving) differs by more than the noise and comes through at once.
fn denoise(history: &mut Vec<[f32; 3]>, rgb: &mut [Vec<f32>; 3], seen: &[f32], strength: f32) {
    let n = seen.len();
    if history.len() != n {
        *history = vec![[f32::NAN; 3]; n];
    }
    // How much of the past to keep, and the difference above which it is motion, not noise.
    let (keep, noise) = (0.6 + 0.3 * strength.min(1.0), 0.03 + 0.1 * strength);
    for i in 0..n {
        let now = [rgb[0][i], rgb[1][i], rgb[2][i]];
        let past = history[i];
        let d = (0..3).map(|c| (now[c] - past[c]).abs()).fold(0.0, f32::max);
        // NaN: nothing there last frame, taken as is.
        let w = if seen[i] > 0.0 && !past[0].is_nan() && d < noise { keep * (1.0 - (d / noise).powi(2)) } else { 0.0 };
        if w > 0.0 {
            for c in 0..3 {
                rgb[c][i] = now[c] + (past[c] - now[c]) * w;
            }
        }
        history[i] = if seen[i] > 0.0 { [rgb[0][i], rgb[1][i], rgb[2][i]] } else { [f32::NAN; 3] };
    }
}

/// Big-eye lens: magnifies a TEX x TEX straight-alpha texture around `centre` (texels),
/// 1 / (1 - strength) times in the middle, fading to no change at `radius` (texels per axis).
fn bulge(rgba: &mut [u8], centre: Vec2, radius: Vec2, strength: f32) {
    let t = TEX as usize;
    let src = rgba.to_vec();
    // Premultiplied while blending, so transparent texels don't darken the edge.
    let texel = |x: i32, y: i32| {
        let i = (y.clamp(0, t as i32 - 1) as usize * t + x.clamp(0, t as i32 - 1) as usize) * 4;
        let a = src[i + 3] as f32 / 255.0;
        [src[i] as f32 * a, src[i + 1] as f32 * a, src[i + 2] as f32 * a, a]
    };
    let (lo, hi) = ((centre - radius).max(Vec2::ZERO), (centre + radius).min(Vec2::splat(t as f32)));
    for y in lo.y as usize..hi.y.ceil() as usize {
        for x in lo.x as usize..hi.x.ceil() as usize {
            let o = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            let r2 = ((o - centre) / radius).length_squared();
            if r2 >= 1.0 {
                continue;
            }
            // r * (1 - s + s r^2): monotonic for s < 1, unchanged at the rim.
            let p = centre + (o - centre) * (1.0 - strength * (1.0 - r2)) - Vec2::splat(0.5);
            let (x0, y0) = (p.x.floor() as i32, p.y.floor() as i32);
            let (fx, fy) = (p.x - x0 as f32, p.y - y0 as f32);
            let mix = |a: [f32; 4], b: [f32; 4], f: f32| std::array::from_fn::<f32, 4, _>(|c| a[c] + (b[c] - a[c]) * f);
            let v = mix(mix(texel(x0, y0), texel(x0 + 1, y0), fx), mix(texel(x0, y0 + 1), texel(x0 + 1, y0 + 1), fx), fy);
            let i = (y * t + x) * 4;
            let a = v[3].max(1e-6);
            rgba[i..i + 4].copy_from_slice(&[(v[0] / a) as u8, (v[1] / a) as u8, (v[2] / a) as u8, (v[3] * 255.0) as u8]);
        }
    }
}

/// Bilinear sample, 0..1 RGB.
fn sample(frame: &Frame, p: Vec2) -> [f32; 3] {
    let (w, h) = (frame.w as i32, frame.h as i32);
    let p = p - Vec2::splat(0.5);
    let (x0, y0) = (p.x.floor() as i32, p.y.floor() as i32);
    let (fx, fy) = (p.x - x0 as f32, p.y - y0 as f32);
    let at = |x: i32, y: i32, c: usize| frame.rgb[((y.clamp(0, h - 1) * w + x.clamp(0, w - 1)) * 3) as usize + c] as f32;
    std::array::from_fn(|c| {
        let top = at(x0, y0, c) * (1.0 - fx) + at(x0 + 1, y0, c) * fx;
        let bottom = at(x0, y0 + 1, c) * (1.0 - fx) + at(x0 + 1, y0 + 1, c) * fx;
        (top * (1.0 - fy) + bottom * fy) / 255.0
    })
}

/// CSS `brightness() contrast() saturate() hue-rotate()`, in that order, clamped per step like
/// the browser. `gain` is the automatic exposure on top of the manual brightness.
fn filter([r, g, b]: [f32; 3], cfg: &Calibration, gain: f32) -> [f32; 3] {
    let step = |c: f32| ((c * cfg.brightness * gain).clamp(0.0, 1.0) - 0.5) * cfg.contrast + 0.5;
    let (r, g, b) = (step(r).clamp(0.0, 1.0), step(g).clamp(0.0, 1.0), step(b).clamp(0.0, 1.0));
    let s = cfg.saturation;
    let [r, g, b] = [
        (0.213 + 0.787 * s) * r + (0.715 - 0.715 * s) * g + (0.072 - 0.072 * s) * b,
        (0.213 - 0.213 * s) * r + (0.715 + 0.285 * s) * g + (0.072 - 0.072 * s) * b,
        (0.213 - 0.213 * s) * r + (0.715 - 0.715 * s) * g + (0.072 + 0.928 * s) * b,
    ]
    .map(|c| c.clamp(0.0, 1.0));
    if cfg.hue_rotate == 0.0 {
        return [r, g, b];
    }
    let (sin, cos) = cfg.hue_rotate.to_radians().sin_cos();
    [
        (0.213 + cos * 0.787 - sin * 0.213) * r + (0.715 - cos * 0.715 - sin * 0.715) * g + (0.072 - cos * 0.072 + sin * 0.928) * b,
        (0.213 - cos * 0.213 + sin * 0.143) * r + (0.715 + cos * 0.285 + sin * 0.140) * g + (0.072 - cos * 0.072 - sin * 0.283) * b,
        (0.213 - cos * 0.213 - sin * 0.787) * r + (0.715 - cos * 0.715 + sin * 0.715) * g + (0.072 + cos * 0.928 + sin * 0.072) * b,
    ]
    .map(|c| c.clamp(0.0, 1.0))
}

/// Gaussian blur of an alpha mask, approximated by three box passes per axis. Outside reads
/// as transparent, same as canvas `filter: blur()`.
#[cfg(test)]
fn blur(data: &mut [u8], w: usize, h: usize, sigma: f32) {
    let mut buf: Vec<f32> = data.iter().map(|&a| a as f32).collect();
    blur_f32(&mut buf, w, h, sigma);
    for (d, v) in data.iter_mut().zip(buf) {
        *d = v.round().clamp(0.0, 255.0) as u8;
    }
}

pub fn blur_f32(buf: &mut [f32], w: usize, h: usize, sigma: f32) {
    blur_xy(buf, w, h, Vec2::splat(sigma));
}

/// Gaussian blur with its own sigma per axis.
fn blur_xy(buf: &mut [f32], w: usize, h: usize, sigma: Vec2) {
    let radius = |s: f32| (((4.0 * s * s + 1.0).sqrt() - 1.0) * 0.5).round() as usize;
    let (rx, ry) = (radius(sigma.x), radius(sigma.y));
    if rx == 0 && ry == 0 {
        return;
    }
    // A zero radius pass is a plain copy, so both axes always run (ping-pong, no copies).
    let mut tmp = vec![0.0; buf.len()];
    for _ in 0..3 {
        box_pass(buf, &mut tmp, w, h, rx, 1, w);
        box_pass(&tmp, buf, h, w, ry, w, 1);
    }
}

/// One box pass along lines of `len` samples spaced `stride` apart, `lines` of them spaced
/// `line_stride` apart. Running sum, O(n) regardless of radius.
fn box_pass(src: &[f32], dst: &mut [f32], len: usize, lines: usize, r: usize, stride: usize, line_stride: usize) {
    let norm = 1.0 / (2 * r + 1) as f32;
    for line in 0..lines {
        let at = |i: usize| src[line * line_stride + i * stride];
        let mut sum: f32 = (0..=r.min(len - 1)).map(at).sum();
        for i in 0..len {
            dst[line * line_stride + i * stride] = sum * norm;
            if i + r + 1 < len {
                sum += at(i + r + 1);
            }
            if i >= r {
                sum -= at(i - r);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::Quat;

    #[test]
    fn blur_keeps_mass_and_softens_edge() {
        let (w, h) = (32, 32);
        let mut m = vec![0u8; w * h];
        for y in 8..24 {
            for x in 8..24 {
                m[y * w + x] = 255;
            }
        }
        let before: u32 = m.iter().map(|&v| v as u32).sum();
        blur(&mut m, w, h, 3.0);
        let after: u32 = m.iter().map(|&v| v as u32).sum();
        assert!((before as f32 - after as f32).abs() / (before as f32) < 0.02);
        assert!(m[16 * w + 8] > 60 && m[16 * w + 8] < 200, "edge is feathered");
        assert!(m[16 * w + 16] > 240, "centre stays nearly solid");
    }

    #[test]
    fn hue_rotate_turns_skin_lavender_and_keeps_whites() {
        let cfg = Calibration { brightness: 1.0, contrast: 1.0, saturation: 1.0, hue_rotate: -100.0, ..Calibration::default() };
        let [r, g, b] = filter([0.85, 0.65, 0.5], &cfg, 1.0); // skin
        assert!(b > g && r > g, "skin becomes purple-ish: {r:.2} {g:.2} {b:.2}");
        let white = filter([1.0, 1.0, 1.0], &cfg, 1.0);
        assert!(white.iter().all(|&c| c > 0.98), "whites stay white: {white:?}");
    }

    /// Smooth colours that differ everywhere on the face.
    fn pattern(x: Vec3) -> [f32; 3] {
        [0.5 + 0.4 * (x.x * 2.1).sin(), 0.5 + 0.4 * (x.y * 1.7).sin(), 0.5 + 0.4 * (x.x * 0.9 + x.y * 1.3).cos()]
    }

    fn neutral() -> Calibration {
        Calibration { brightness: 1.0, contrast: 1.0, saturation: 1.0, sharpen: 0.0, denoise: 0.0, auto_exposure: false, ..Calibration::default() }
    }

    /// Mean colour difference where both textures are solid, and how many such texels.
    fn diff(a: &Feature, b: &Feature) -> (f32, usize) {
        let (mut sum, mut n) = (0.0, 0);
        for (p, q) in a.rgba.chunks(4).zip(b.rgba.chunks(4)) {
            if p[3] > 200 && q[3] > 200 {
                sum += (0..3).map(|c| (p[c] as f32 - q[c] as f32).abs()).sum::<f32>() / 3.0;
                n += 1;
            }
        }
        (sum / n.max(1) as f32, n)
    }

    /// Live numbers only (no images): where the features land and how much of each mask is
    /// covered. `cargo test --release webcam_features -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn webcam_features() {
        let shared = crate::tracker::spawn(0, false);
        let (start, cfg) = (std::time::Instant::now(), neutral());
        let mut last = None;
        let mut n = 0;
        while start.elapsed().as_secs() < 8 {
            std::thread::sleep(std::time::Duration::from_millis(5));
            let Some(t) = shared.lock().unwrap().latest.clone() else { continue };
            if last == Some(std::sync::Arc::as_ptr(&t)) {
                continue;
            }
            last = Some(std::sync::Arc::as_ptr(&t));
            let (Some(lm), Some(head)) = (t.landmarks.as_ref(), t.head) else { continue };
            n += 1;
            if n % 25 != 1 {
                continue;
            }
            let proj = Projection::new(&head);
            let frontal: Vec<Vec2> = (0..468).map(|i| proj.to_frontal(lm[i].truncate(), VERTICES[i][2]).unwrap_or(Vec2::NAN)).collect();
            let c = |ring: &[usize]| ring.iter().map(|&i| frontal[i]).sum::<Vec2>() / ring.len() as f32;
            let canon = |ring: &[usize]| ring.iter().map(|&i| Vec3::from(VERTICES[i]).truncate()).sum::<Vec2>() / ring.len() as f32;
            let (y, x, _) = head.rotation.to_euler(bevy::math::EulerRot::YXZ);
            let Some([r, l, m]) = features(&t.frame, lm, Some(head), &cfg) else { continue };
            let solid = |f: &Feature| f.rgba.chunks(4).filter(|p| p[3] > 200).count();
            println!(
                "yaw {:5.1} pitch {:5.1} | live cm R {:5.2},{:5.2} L {:5.2},{:5.2} M {:5.2},{:5.2} | canon R {:5.2},{:5.2} M {:5.2},{:5.2} | model R {:.3},{:.3} L {:.3},{:.3} M {:.3},{:.3} | solid {} {} {}",
                y.to_degrees(), x.to_degrees(),
                c(RIGHT_EYE).x, c(RIGHT_EYE).y, c(LEFT_EYE).x, c(LEFT_EYE).y, c(LIPS).x, c(LIPS).y,
                canon(RIGHT_EYE).x, canon(RIGHT_EYE).y, canon(LIPS).x, canon(LIPS).y,
                r.center.x, r.center.y, l.center.x, l.center.y, m.center.x, m.center.y,
                solid(&r), solid(&l), solid(&m),
            );
        }
    }

    #[test]
    fn relief_domes_the_eyes_and_sinks_the_open_mouth() {
        let cfg = neutral();
        let (frame, mut lm, head) = crate::testface::render(Quat::IDENTITY, pattern);
        // Open the mouth: lower lip line and chin down by 1 cm (19 px per cm).
        for i in [14, 87, 178, 88, 95, 317, 402, 318, 324, 17, 84, 181, 91, 146, 314, 405, 321, 375] {
            lm[i].y += 19.0;
        }
        let [r, _, m] = features(&frame, &lm, Some(head), &cfg).unwrap();
        let g = RELIEF_CELLS + 1;
        let at = |f: &Feature, i: usize, j: usize| f.relief[j * g + i];
        let mid = g / 2;
        assert_eq!(r.relief.len(), g * g);
        assert!(at(&r, mid, mid) > 0.0, "eyeball stands out: {}", at(&r, mid, mid));
        assert_eq!(at(&r, 0, 0), 0.0, "nothing outside the cut");
        let lip = (0..g).map(|j| at(&m, mid, j)).fold(f32::MIN, f32::max);
        assert!(at(&m, mid, mid) < lip - 0.002, "open mouth lies behind the lips: {} vs {lip}", at(&m, mid, mid));
        let flat = Calibration { relief: 0.0, ..cfg };
        assert!(features(&frame, &lm, Some(head), &flat).unwrap()[0].relief.iter().all(|&h| h == 0.0));
    }

    #[test]
    fn bulge_magnifies_the_middle_and_keeps_the_rim() {
        let t = TEX as usize;
        // Vertical stripes 8 texels wide.
        let mut rgba: Vec<u8> = (0..t * t).flat_map(|i| if (i % t) / 8 % 2 == 0 { [255, 255, 255, 255] } else { [0, 0, 0, 255] }).collect();
        let before = rgba.clone();
        let c = Vec2::splat(t as f32 / 2.0);
        bulge(&mut rgba, c, Vec2::splat(60.0), 0.5);
        let row = |img: &[u8], y: usize| (0..t).map(|x| img[(y * t + x) * 4]).collect::<Vec<_>>();
        // Width of the stripe just right of the centre: 8 texels, magnified 2x in the middle.
        let stripe = |r: Vec<u8>| r[t / 2..].iter().take_while(|&&v| (v > 127) == (r[t / 2] > 127)).count();
        let (after, was) = (stripe(row(&rgba, t / 2)), stripe(row(&before, t / 2)));
        assert!(was == 8 && (14..=18).contains(&after), "stripe {was} -> {after}");
        assert_eq!(row(&rgba, 10), row(&before, 10), "outside the lens nothing changes");
    }

    #[test]
    fn denoise_averages_noise_and_passes_motion() {
        let (mut history, seen) = (Vec::new(), vec![1.0; 2]);
        let mut out = Vec::new();
        // Texel 0 flickers ±0.02 around 0.5 (webcam noise); texel 1 holds still.
        for f in 0..30 {
            let v = 0.5 + if f % 2 == 0 { 0.02 } else { -0.02 };
            let mut rgb = [vec![v, 0.3], vec![v, 0.3], vec![v, 0.3]];
            denoise(&mut history, &mut rgb, &seen, 0.5);
            out.push(rgb[0][0]);
        }
        assert!(out[20..].iter().all(|v| (v - 0.5).abs() < 0.012), "noise stays: {:?}", &out[20..]);
        // A blink: texel 0 jumps; it shows at once, not smeared over frames.
        let mut rgb = [vec![0.9, 0.3], vec![0.9, 0.3], vec![0.9, 0.3]];
        denoise(&mut history, &mut rgb, &seen, 0.5);
        assert_eq!(rgb[0][0], 0.9);
    }

    #[test]
    fn grow_keeps_the_shape() {
        // A long thin lens (an eye, a closed mouth) gets the same border on every side.
        let lens: Vec<Vec2> = (0..16).map(|i| i as f32 / 16.0 * std::f32::consts::TAU).map(|a| Vec2::new(3.0 * a.cos(), 0.5 * a.sin())).collect();
        let (lo, hi) = bounds(&grow(&lens, 0.2));
        assert!((hi.x - 3.2).abs() < 0.05 && (hi.y - 0.7).abs() < 0.05 && (lo + hi).length() < 1e-3, "{lo} {hi}");
        let (lo, hi) = bounds(&grow(&lens.iter().rev().copied().collect::<Vec<_>>(), 0.2));
        assert!((hi.x - 3.2).abs() < 0.05 && (lo.y + 0.7).abs() < 0.05, "either winding");
    }

    #[test]
    fn projection_round_trips() {
        let head = Head { rotation: Quat::from_rotation_y(0.6) * Quat::from_rotation_x(0.2), pivot: Vec2::new(300.0, 250.0), scale: 17.0 };
        let proj = Projection::new(&head);
        for v in [VERTICES[33], VERTICES[4], VERTICES[61]] {
            let f = proj.to_frontal(proj.to_image(Vec3::from(v)), v[2]).unwrap();
            assert!((f - Vec3::from(v).truncate()).length() < 1e-3, "{f:?} vs {v:?}");
        }
    }

    #[test]
    fn frontalization_is_pose_invariant() {
        let cfg = neutral();
        let at = |r: Quat| {
            let (frame, lm, head) = crate::testface::render(r, pattern);
            features(&frame, &lm, Some(head), &cfg).unwrap()
        };
        let front = at(Quat::IDENTITY);
        let [r, l, m] = &front;
        assert!(r.center.x > cfg.face_x && l.center.x < cfg.face_x, "mirrored: right eye shows on the right");
        assert!(m.center.y < r.center.y, "mouth below the eyes");
        for f in &front {
            assert!(f.rgba.chunks(4).filter(|p| p[3] > 200).count() > 500, "texture has content");
        }

        // Turned and tilted: same place, same size, and the same pixels.
        let turned = at(Quat::from_rotation_y(0.6) * Quat::from_rotation_x(0.25) * Quat::from_rotation_z(0.2));
        for (a, b) in front.iter().zip(&turned) {
            assert!(b.alpha > 0.9, "visible");
            assert!((a.half - b.half).length() < 1e-3, "size {:?} vs {:?}", a.half, b.half);
            assert!((a.center - b.center).length() < 1e-3, "place {:?} vs {:?}", a.center, b.center);
        }
        // Eyes: the same pixels. The mouth's inside is cut on a smooth sheet (`mouth_depth`),
        // close but not exact on this face with the canonical pout; above all, whole.
        for (k, (a, b)) in front.iter().zip(&turned).enumerate() {
            let (d, n) = diff(a, b);
            assert!(n > 400 && d < if k == 2 { 14.0 } else { 6.0 }, "texture differs by {d:.1} over {n} texels");
        }
        let solid = |f: &Feature| f.rgba.chunks(4).filter(|p| p[3] > 200).count() as f32;
        assert!(solid(&turned[2]) > 0.95 * solid(&front[2]), "mouth has holes when turned: {} vs {}", solid(&turned[2]), solid(&front[2]));

        // Pressed lips (both inner lip lines on one line) seen from below: the 3D cut folded
        // the lips over themselves and the model showed through.
        let (frame, mut lm, head) = crate::testface::render(Quat::from_rotation_x(-0.35), pattern);
        let inner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];
        let y = inner.iter().map(|&i| lm[i].y).sum::<f32>() / inner.len() as f32;
        for i in inner {
            lm[i].y = y;
        }
        let pressed = &features(&frame, &lm, Some(head), &cfg).unwrap()[2];
        assert!(solid(pressed) > 0.95 * solid(&front[2]), "pressed lips have holes: {} vs {}", solid(pressed), solid(&front[2]));

        // Far to the side the hidden eye fades instead of smearing; the near one stays exact.
        let side = at(Quat::from_rotation_y(1.2));
        let (r3, l3) = (&side[0], &side[1]);
        assert!(r3.alpha.min(l3.alpha) < 0.1 && r3.alpha.max(l3.alpha) > 0.9, "{} {}", r3.alpha, l3.alpha);
        let (near, near_front) = if r3.alpha > l3.alpha { (r3, &front[0]) } else { (l3, &front[1]) };
        let (d, n) = diff(near, near_front);
        assert!(n > 200 && d < 10.0, "near eye at 70 degrees differs by {d:.1} over {n} texels");
    }
}
