//! Test helper: MediaPipe's canonical face rendered into a frame at a known pose.

use crate::canonical_mesh::{TRIANGLES, VERTICES};
use crate::compose::Projection;
use crate::tracker::{Frame, Head};
use bevy::math::{Mat3, Quat, Vec2, Vec3};

/// The canonical face rendered into a frame at a pose, coloured by where each pixel is on
/// the face (so a correct frontalization must give the same texture at every pose), and
/// its landmarks. Z-buffered, so the nose really hides the far eye.
pub fn render(rotation: Quat, pattern: impl Fn(Vec3) -> [f32; 3]) -> (Frame, Vec<Vec3>, Head) {
    render_at(Head { rotation, pivot: Vec2::new(320.0, 330.0), scale: 19.0 }, pattern)
}

/// Same, at any pose including position and size.
pub fn render_at(head: Head, pattern: impl Fn(Vec3) -> [f32; 3]) -> (Frame, Vec<Vec3>, Head) {
    let (w, h) = (640u32, 480u32);
    let rotation = head.rotation;
    let proj = Projection::new(&head);
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    let mut depth = vec![f32::NEG_INFINITY; (w * h) as usize];
    let m = Mat3::from_quat(rotation);
    for tri in &TRIANGLES {
        let v = tri.map(|i| Vec3::from(VERTICES[i as usize]));
        let p = v.map(|x| proj.to_image(x));
        let z = v.map(|x| m.row(2).dot(x));
        let area = (p[1] - p[0]).perp_dot(p[2] - p[0]);
        if area.abs() < 1e-6 {
            continue;
        }
        let (lo, hi) = (p[0].min(p[1]).min(p[2]), p[0].max(p[1]).max(p[2]));
        for y in lo.y.max(0.0) as u32..(hi.y.ceil() as u32).min(h) {
            for x in lo.x.max(0.0) as u32..(hi.x.ceil() as u32).min(w) {
                let q = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let wa = (p[1] - q).perp_dot(p[2] - q) / area;
                let wb = (p[2] - q).perp_dot(p[0] - q) / area;
                let wc = 1.0 - wa - wb;
                if wa < 0.0 || wb < 0.0 || wc < 0.0 {
                    continue;
                }
                let i = (y * w + x) as usize;
                let d = z[0] * wa + z[1] * wb + z[2] * wc;
                if d > depth[i] {
                    depth[i] = d;
                    let c = pattern(v[0] * wa + v[1] * wb + v[2] * wc);
                    rgb[i * 3..i * 3 + 3].copy_from_slice(&c.map(|c| (c * 255.0) as u8));
                }
            }
        }
    }
    let mut lm: Vec<Vec3> = VERTICES.iter().map(|&x| proj.to_image(Vec3::from(x)).extend(0.0)).collect();
    lm.resize(478, lm[468 - 1]);
    (Frame { w, h, rgb }, lm, head)
}

