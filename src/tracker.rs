//! Webcam + face tracking, on its own thread. MediaPipe's Face Landmarker pipeline, rebuilt:
//! BlazeFace finds the face once, the 478-point landmark model then tracks it frame to frame
//! from its own previous result, and head pose comes from a weak-perspective fit against
//! MediaPipe's canonical face. Where the landmark model gives up (steep angles), optical flow
//! on the face carries the pose (flow.rs). Everything runs on-device with tract, no network.
//!
//! Frame and landmarks are published together, so the compositor always cuts the eyes out of
//! the exact frame the landmarks were measured on (the browser version needed
//! requestVideoFrameCallback gymnastics for that guarantee).

use crate::flow;
use bevy::math::{Mat3, Quat, Vec2, Vec3};
use std::sync::{Arc, Mutex};
use tract_tflite::prelude::*;

const DETECTOR: &[u8] = include_bytes!("../assets/face_detector.tflite");
const LANDMARKER: &[u8] = include_bytes!("../assets/face_landmarks_detector.tflite");

pub struct Frame {
    pub w: u32,
    pub h: u32,
    pub rgb: Vec<u8>,
}

pub struct Tracked {
    pub frame: Arc<Frame>,
    /// Landmarks in frame pixels (x right, y down, z in pixel scale, smaller = closer).
    pub landmarks: Option<Vec<Vec3>>,
    /// With the head too far turned for the landmark model, the pose comes from optical flow
    /// and the landmarks are a rigid canonical face at that pose (no expression).
    pub head: Option<Head>,
    /// For the mesh view: how the face was found, how sure the landmark model was, and the
    /// crop (corners, frame pixels) it looked at.
    pub how: &'static str,
    pub presence: f32,
    pub roi: Option<[Vec2; 4]>,
}

#[derive(Clone, Copy, Debug)]
pub struct Head {
    /// Canonical face -> camera (x right, y up, z toward camera).
    pub rotation: Quat,
    /// Where the head turns around (top of the neck), in frame pixels. Unlike any point on
    /// the face it stays put when you only turn, so following it moves the model only when
    /// you actually move.
    pub pivot: Vec2,
    /// Frame pixels per canonical unit (~cm). Fitted with the known rotation, so turning
    /// leaves it alone; only leaning in or out changes it.
    pub scale: f32,
}

/// Top of the neck in canonical face coordinates (cm): behind and below the eyes.
pub const NECK: Vec3 = Vec3::new(0.0, -3.0, -6.0);

#[derive(Default)]
pub struct TrackerState {
    pub latest: Option<Arc<Tracked>>,
    pub status: String,
    pub error: bool,
    pub running: bool,
}

pub type Shared = Arc<Mutex<TrackerState>>;

/// `full_frames`: always decode the whole camera image (the mesh view shows all of it);
/// otherwise only the face and a margin while it is tracked.
pub fn spawn(camera: u32, full_frames: bool) -> Shared {
    let shared: Shared = Arc::new(Mutex::new(TrackerState { status: "Kamera wird geöffnet …".into(), ..Default::default() }));
    let s = shared.clone();
    std::thread::spawn(move || {
        if let Err(e) = run(camera, full_frames, &s) {
            let mut st = s.lock().unwrap();
            st.status = e;
            st.error = true;
        }
    });
    shared
}

type Plan = TypedRunnableModel<TypedModel>;

fn load(bytes: &[u8]) -> TractResult<Plan> {
    tract_tflite::tflite().model_for_read(&mut &bytes[..])?.into_optimized()?.into_runnable()
}

/// What the tracker needs of the next camera frame.
#[derive(Default)]
struct Wants {
    /// The face is long gone: only every SEARCH_IDLE a frame is decoded at all (decoding 1080p
    /// costs ~12 ms, the search after it ~30-60 ms).
    resting: bool,
    /// Frame pixels [x0, y0, x1, y1] worth decoding; None = all of it.
    region: Option<[f32; 4]>,
}

fn run(camera: u32, full_frames: bool, shared: &Shared) -> Result<(), String> {
    // Capture + decode on their own thread into a "newest frame" slot, so inference always
    // works on the freshest frame and the camera never waits on it (no stale queue, ~2x fps).
    let slot: Arc<(Mutex<Option<Result<Frame, String>>>, std::sync::Condvar)> = Default::default();
    let writer = slot.clone();
    let wants: Arc<Mutex<Wants>> = Default::default();
    let wanted = wants.clone();
    std::thread::spawn(move || {
        let publish = |f| {
            *writer.0.lock().unwrap() = Some(f);
            writer.1.notify_one();
        };
        if let Some(dir) = std::env::var_os("AVATAR_REPLAY") {
            return replay(dir.into(), publish);
        }
        let record = std::env::var_os("AVATAR_RECORD").map(|dir| recorder(dir.into()));
        let start = std::time::Instant::now();
        match open_camera(camera) {
            Ok(mut cam) => {
                let (mut decoded, mut damaged) = (std::time::Instant::now() - SEARCH_IDLE, 0);
                loop {
                    let buf = match cam.frame() {
                        Ok(b) => b,
                        Err(e) => return publish(Err(format!("Webcam gestoppt: {e}"))),
                    };
                    let (resting, region) = {
                        let w = wanted.lock().unwrap();
                        (w.resting, w.region.filter(|_| !full_frames && record.is_none()))
                    };
                    if record.is_none() && resting && decoded.elapsed() < SEARCH_IDLE {
                        continue;
                    }
                    decoded = std::time::Instant::now();
                    let frame = decode(&buf, region);
                    // A damaged frame now and then is the webcam's doing: skip it. Only a run of
                    // them means the camera is broken.
                    damaged = if frame.is_err() { damaged + 1 } else { 0 };
                    if (1..30).contains(&damaged) {
                        continue;
                    }
                    if let (Some(tx), Ok(f)) = (&record, &frame) {
                        // Behind on writing: skip a frame rather than pile up memory.
                        let _ = tx.try_send((start.elapsed().as_millis(), Frame { w: f.w, h: f.h, rgb: f.rgb.clone() }));
                    }
                    let failed = frame.is_err();
                    publish(frame);
                    if failed {
                        return;
                    }
                }
            }
            Err(e) => publish(Err(e)),
        }
    });

    let started = std::time::Instant::now();
    shared.lock().unwrap().status = "Gesichtserkennung wird geladen …".into();
    let models = (load(DETECTOR), load(LANDMARKER));
    let (detector, landmarker) = match models {
        (Ok(d), Ok(l)) => (d, l),
        (Err(e), _) | (_, Err(e)) => return Err(format!("Gesichtserkennung fehlgeschlagen: {e}")),
    };
    let anchors = anchors();

    let mut roi: Option<Roi> = None;
    // Where the face was last seen, kept for a moment after losing it: re-detecting in a crop
    // around it makes the face several times larger for the detector than in the whole frame.
    let mut last_seen: Option<(Roi, std::time::Instant)> = None;
    let (mut anchor, mut flow): (Option<flow::Anchor>, Option<flow::Flow>) = (None, None);
    // Raw pose of the previous frame (None if it had no face): the speed for predictions.
    let mut last_head: Option<Head> = None;
    let mut last_roi: Option<Roi> = None;
    let (mut pose_filter, mut shape_filter) = (OneEuro::pose(), OneEuro::shapes());
    loop {
        let frame = Arc::new({
            let mut guard = slot.1.wait_while(slot.0.lock().unwrap(), |f| f.is_none()).unwrap();
            guard.take().unwrap()?
        });
        if !shared.lock().unwrap().running {
            let mut st = shared.lock().unwrap();
            st.status = "Läuft".into();
            st.running = true;
        }

        // Track from last frame's landmarks. MediaPipe sure: done. Unsure or lost: optical flow
        // from the last trusted frame carries the pose and decides whether the unsure
        // landmarks still fit. Nothing works: re-detect on this same frame, first near where
        // the face was, then in the whole frame.
        let now = std::time::Instant::now();
        let (mut landmarks, mut steep) = (None, false);
        let looked_at = roi.map(|r| [Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::Y].map(|c| r.to_frame(c)));
        let tracked = roi.and_then(|r| landmarks_in(&landmarker, &frame, r));
        // For AVATAR_TRACE: which path gave the face, and how sure the landmark model was.
        let (mut how, presence) = ("lost", tracked.as_ref().map_or(f32::NAN, |t| t.1));
        match tracked {
            Some((lm, presence)) if presence >= DETECT_PRESENCE => {
                flow = None;
                landmarks = Some(lm);
                how = "landmarks";
            }
            unsure => match (unsure, flow::step(&mut flow, anchor.as_ref(), &frame, now).or_else(|| {
                // The flow failed from this start; every later frame is further from it, so
                // retrying it each frame only burned time (and frame rate) while lost.
                anchor = None;
                None
            })) {
                (Some((lm, presence)), Some(by_flow)) if presence >= MIN_PRESENCE && head_pose(&lm).is_some_and(|h| agrees(&h, &by_flow)) => {
                    if let Some(f) = &mut flow {
                        f.verified(now);
                    }
                    landmarks = Some(lm);
                    how = "unsure landmarks, flow agrees";
                }
                (_, Some(by_flow)) => {
                    landmarks = Some(flow::synthetic(&by_flow));
                    steep = true;
                    how = "flow only";
                }
                (Some((lm, presence)), None) if presence >= TRACK_PRESENCE => {
                    landmarks = Some(lm);
                    how = "unsure landmarks, no flow";
                }
                _ => {}
            },
        }
        if landmarks.is_none() {
            flow = None;
            let near = last_seen.filter(|(_, t)| now - *t < REDETECT_NEAR).map(|(r, _)| Roi { size: r.size * 2.0, angle: 0.0, ..r });
            let whole = Roi { center: Vec2::new(frame.w as f32, frame.h as f32) * 0.5, size: frame.w.max(frame.h) as f32, angle: 0.0 };
            landmarks = near
                .into_iter()
                .chain([whole])
                .find_map(|region| {
                    let found = detect(&detector, &anchors, &frame, region)?;
                    landmarks_in(&landmarker, &frame, found).filter(|(_, p)| *p >= DETECT_PRESENCE).map(|(lm, _)| lm)
                });
            if landmarks.is_some() {
                how = "re-detected";
            }
        }
        // Long gone (away from the desk): search only every SEARCH_IDLE, the capture thread
        // skips decoding the frames in between. A face coming back is found within that.
        let gone = landmarks.is_none() && last_seen.is_none_or(|(_, t)| now - t > SEARCH_IDLE * 4);
        // Real landmarks become the flow's starting point should the next frame need it.
        if let (Some(lm), false) = (&landmarks, steep)
            && let Some(head) = head_pose(lm)
        {
            anchor = Some(flow::Anchor { frame: frame.clone(), landmarks: lm.clone(), head, before: last_head, at: now });
        }
        last_head = landmarks.as_deref().and_then(head_pose);
        if *TRACE {
            let (yaw, pitch) = last_head.map_or((f32::NAN, f32::NAN), |h| {
                let (y, x, _) = h.rotation.to_euler(bevy::math::EulerRot::YXZ);
                (y.to_degrees(), x.to_degrees())
            });
            eprintln!("trace: t {:6.2}  presence {presence:.2}  yaw {yaw:5.1} pitch {pitch:5.1}  {how}", (now - started).as_secs_f32());
        }
        // ROI from the raw landmarks (tracking wants accuracy), everything shown from the
        // filtered ones (viewers want stability).
        // Moved ahead by how far the face moved since the previous frame: on a fast move the
        // face otherwise lands half outside the crop and the landmark model loses it.
        let current = landmarks.as_deref().map(roi_from_landmarks);
        roi = current.map(|r| match last_roi {
            Some(l) => Roi { center: r.center + (r.center - l.center), ..r },
            None => r,
        });
        // Sure of the face: the next frame is decoded around where it will be. Unsure, flow,
        // lost: all of it (the flow and the search look further).
        let sure = matches!(how, "landmarks" | "re-detected");
        *wants.lock().unwrap() = Wants { resting: gone, region: roi.filter(|_| sure).map(|r| r.decode_region()) };
        last_roi = current;
        if let Some(r) = current {
            last_seen = Some((r, now));
        }
        let head = landmarks.as_deref().and_then(|lm| head_pose(&pose_filter.apply(lm, now)));
        let landmarks = landmarks.map(|lm| shape_filter.apply(&lm, now));
        // A dropout of a few frames keeps the filters (no jump when the face is back).
        if landmarks.is_none() && last_seen.is_none_or(|(_, t)| now - t > KEEP_FILTERS) {
            pose_filter.reset();
            shape_filter.reset();
        }

        let mut st = shared.lock().unwrap();
        st.status = if steep { "Starke Drehung, folgt über Bildpunkte" } else { "Läuft" }.into();
        st.latest = Some(Arc::new(Tracked { frame, landmarks, head, how, presence, roi: looked_at }));
    }
}

/// `AVATAR_RECORD=dir`: every camera frame as dir/<ms since start>.jpg, written off the
/// capture thread. Replayed with `AVATAR_REPLAY=dir`, so a tracking problem can be rerun
/// offline as often as needed.
fn recorder(dir: std::path::PathBuf) -> std::sync::mpsc::SyncSender<(u128, Frame)> {
    let _ = std::fs::create_dir_all(&dir);
    let (tx, rx) = std::sync::mpsc::sync_channel::<(u128, Frame)>(16);
    let rx = Arc::new(Mutex::new(rx));
    for _ in 0..3 {
        let (rx, dir) = (rx.clone(), dir.clone());
        std::thread::spawn(move || {
            while let Ok((ms, f)) = { let next = rx.lock().unwrap().recv(); next } {
                let Ok(file) = std::fs::File::create(dir.join(format!("{ms:08}.jpg"))) else { continue };
                let mut out = std::io::BufWriter::new(file);
                let _ = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 92).encode(&f.rgb, f.w, f.h, image::ExtendedColorType::Rgb8);
            }
        });
    }
    tx
}

/// Plays a `AVATAR_RECORD` directory back in real time instead of the camera, then stops.
fn replay(dir: std::path::PathBuf, publish: impl Fn(Result<Frame, String>)) {
    let mut files: Vec<(u64, std::path::PathBuf)> = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| Some((e.path().file_stem()?.to_str()?.parse().ok()?, e.path())))
        .collect();
    files.sort();
    let start = std::time::Instant::now();
    for (ms, path) in files {
        let Ok(img) = image::open(&path) else { continue };
        let img = img.to_rgb8();
        std::thread::sleep(std::time::Duration::from_millis(ms).saturating_sub(start.elapsed()));
        publish(Ok(Frame { w: img.width(), h: img.height(), rgb: img.into_raw() }));
    }
    publish(Err("Wiedergabe beendet".into()));
}

fn open_camera(camera: u32) -> Result<nokhwa::Camera, String> {
    use nokhwa::pixel_format::RgbFormat;
    use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};

    // Most pixels up to 1080p among modes that hold >= 25 fps: the eyes are cut from a
    // ~50 px patch, so every extra pixel there is visible sharpness. Plain "highest
    // resolution" picks multi-megapixel modes that crawl at a few fps.
    let any = RequestedFormat::new::<RgbFormat>(RequestedFormatType::None);
    let mut cam = nokhwa::Camera::new(CameraIndex::Index(camera), any).map_err(|e| format!("Keine Webcam: {e}"))?;
    let best = cam.compatible_camera_formats().ok().and_then(|formats| {
        use nokhwa::utils::FrameFormat::*;
        formats
            .into_iter()
            .filter(|f| matches!(f.format(), MJPEG | YUYV | NV12 | RAWRGB))
            .min_by_key(|f| {
                let pixels = f.resolution().x() as i64 * f.resolution().y() as i64;
                (f.frame_rate() < 25, pixels > 1920 * 1080, std::cmp::Reverse(pixels), std::cmp::Reverse(f.frame_rate()))
            })
    });
    if let Some(f) = best {
        let _ = cam.set_camera_requset(RequestedFormat::new::<RgbFormat>(RequestedFormatType::Exact(f)));
    }
    cam.open_stream().map_err(|e| format!("Keine Webcam: {e}"))?;
    Ok(cam)
}

fn decode(buf: &nokhwa::Buffer, region: Option<[f32; 4]>) -> Result<Frame, String> {
    if buf.source_frame_format() == nokhwa::utils::FrameFormat::MJPEG {
        let (w, h, rgb) = crate::jpeg::decode(buf.buffer(), region)?;
        return Ok(Frame { w, h, rgb });
    }
    let img = buf.decode_image::<nokhwa::pixel_format::RgbFormat>().map_err(|e| format!("Webcam-Bild nicht lesbar: {e}"))?;
    Ok(Frame { w: img.width(), h: img.height(), rgb: img.into_raw() })
}

/// One Euro filter over all landmarks (what MediaPipe runs in video mode, and what was missing:
/// raw landmarks jitter by a pixel or two, and through the pose fit that is a visibly shaking
/// head). Strong smoothing while still; the cutoff rises with speed so real motion stays
/// responsive. Speeds are in face widths per second, independent of resolution.
///
/// `whole_face` drives every point's cutoff from the mean speed of the whole face: noise
/// averages out over 478 points, real head motion doesn't, so it suits the pose. Per-point
/// speed (whole_face = false) lets a blink or an opening mouth through, so it suits the shapes.
struct OneEuro {
    whole_face: bool,
    min_cutoff_hz: f32,
    beta: f32,
    last: Option<(std::time::Instant, Vec<Vec3>, Vec<Vec3>)>,
}

impl OneEuro {
    const DERIVATIVE_CUTOFF_HZ: f32 = 1.0;

    fn pose() -> Self {
        Self { whole_face: true, min_cutoff_hz: 0.25, beta: 4.0, last: None }
    }

    fn shapes() -> Self {
        Self { whole_face: false, min_cutoff_hz: 1.5, beta: 6.0, last: None }
    }

    fn reset(&mut self) {
        self.last = None;
    }

    fn apply(&mut self, raw: &[Vec3], now: std::time::Instant) -> Vec<Vec3> {
        let Some((then, prev, prev_speed)) = self.last.take() else {
            self.last = Some((now, raw.to_vec(), vec![Vec3::ZERO; raw.len()]));
            return raw.to_vec();
        };
        let dt = (now - then).as_secs_f32().max(1e-3);
        let alpha = |cutoff: f32| 1.0 / (1.0 + 1.0 / (std::f32::consts::TAU * cutoff * dt));
        let face_width = (raw[454] - raw[234]).truncate().length().max(1.0);
        let a_d = alpha(Self::DERIVATIVE_CUTOFF_HZ);
        let speed: Vec<Vec3> = raw.iter().zip(&prev).zip(&prev_speed).map(|((x, p), dp)| dp.lerp((*x - *p) / dt, a_d)).collect();
        // Whole-face speed: translation plus rotation. The mean velocity alone cancels for a
        // turn or tilt (points move opposite ways), which kept fast turns at the slowest
        // cutoff and made them lag. Rotation shows as halves of the face moving differently
        // (left/right, top/bottom, front/back); means over hundreds of points, so per-point
        // noise still cancels.
        let mean_speed = if self.whole_face { group_speed(&prev, &speed) } else { 0.0 };
        let out: Vec<Vec3> = raw
            .iter()
            .zip(&prev)
            .zip(&speed)
            .map(|((x, p), d)| {
                let v = if self.whole_face { mean_speed } else { d.length() };
                p.lerp(*x, alpha(self.min_cutoff_hz + self.beta * v / face_width))
            })
            .collect();
        self.last = Some((now, out.clone(), speed));
        out
    }
}

/// Translation plus rotation speed of the whole face, from group-mean velocities.
fn group_speed(points: &[Vec3], speed: &[Vec3]) -> f32 {
    let centre = points.iter().copied().sum::<Vec3>() / points.len() as f32;
    let mean = |pick: &dyn Fn(Vec3) -> bool| {
        let (sum, n) = points.iter().zip(speed).filter(|(p, _)| pick(**p - centre)).fold((Vec3::ZERO, 0.0), |(s, n), (_, d)| (s + *d, n + 1.0));
        sum / f32::max(n, 1.0)
    };
    let split = |axis: usize| (mean(&|d: Vec3| d[axis] > 0.0) - mean(&|d: Vec3| d[axis] <= 0.0)).truncate().length() * 0.5;
    mean(&|_| true).truncate().length() + split(0) + split(1) + split(2)
}

/// Rotated square crop region, in frame pixels. `angle` is the direction of the face's
/// x axis (eye line) in image space, so the crop comes out with the face upright.
#[derive(Clone, Copy, Debug)]
struct Roi {
    center: Vec2,
    size: f32,
    angle: f32,
}

impl Roi {
    fn axes(&self) -> (Vec2, Vec2) {
        let (s, c) = self.angle.sin_cos();
        (Vec2::new(c, s), Vec2::new(-s, c))
    }
    /// Frame pixels [x0, y0, x1, y1] to decode for a frame tracked in this crop: its bounding
    /// box, plus a quarter of its size on every side for motion until that frame arrives.
    fn decode_region(&self) -> [f32; 4] {
        let (s, c) = self.angle.sin_cos();
        let half = self.size * (0.5 * (s.abs() + c.abs()) + 0.25);
        [self.center.x - half, self.center.y - half, self.center.x + half, self.center.y + half]
    }
    /// Normalized crop coords (0..1) -> frame pixels.
    fn to_frame(&self, n: Vec2) -> Vec2 {
        let (u, v) = self.axes();
        self.center + (u * (n.x - 0.5) + v * (n.y - 0.5)) * self.size
    }
}

/// Bilinear crop into an NHWC f32 tensor, mapping each channel to `lo..hi`. Outside the
/// frame reads as black, which is also how the detector's letterbox padding looks.
fn crop(frame: &Frame, roi: Roi, side: usize, lo: f32, hi: f32) -> Tensor {
    let (u, v) = roi.axes();
    let step = roi.size / side as f32;
    let origin = roi.center - (u + v) * roi.size * 0.5;
    let (w, h) = (frame.w as i32, frame.h as i32);
    let px = |x: i32, y: i32, c: usize| -> f32 {
        if x < 0 || y < 0 || x >= w || y >= h {
            0.0
        } else {
            frame.rgb[((y * w + x) * 3) as usize + c] as f32
        }
    };
    let scale = (hi - lo) / 255.0;
    let mut out = Vec::with_capacity(side * side * 3);
    for j in 0..side {
        for i in 0..side {
            let p = origin + (u * (i as f32 + 0.5) + v * (j as f32 + 0.5)) * step - Vec2::splat(0.5);
            let (x0, y0) = (p.x.floor() as i32, p.y.floor() as i32);
            let (fx, fy) = (p.x - x0 as f32, p.y - y0 as f32);
            for c in 0..3 {
                let top = px(x0, y0, c) * (1.0 - fx) + px(x0 + 1, y0, c) * fx;
                let bottom = px(x0, y0 + 1, c) * (1.0 - fx) + px(x0 + 1, y0 + 1, c) * fx;
                out.push(lo + (top * (1.0 - fy) + bottom * fy) * scale);
            }
        }
    }
    Tensor::from_shape(&[1, side, side, 3], &out).expect("crop shape")
}

/// BlazeFace short-range SSD anchors: 16x16 cells x2 plus 8x8 cells x6 = 896.
fn anchors() -> Vec<Vec2> {
    let mut a = Vec::with_capacity(896);
    for (cells, per_cell) in [(16, 2), (8, 6)] {
        for y in 0..cells {
            for x in 0..cells {
                let c = Vec2::new((x as f32 + 0.5) / cells as f32, (y as f32 + 0.5) / cells as f32);
                a.extend(std::iter::repeat_n(c, per_cell));
            }
        }
    }
    a
}

/// Best face from BlazeFace, as the landmark model's ROI: box scaled 1.5x, square, rotated
/// so the eyes are level (MediaPipe's face_detection_to_roi settings).
/// Presence MediaPipe is trusted at on its own (also needed for a fresh detection).
const DETECT_PRESENCE: f32 = 0.5;
/// `AVATAR_TRACE=1`: one line per camera frame on stderr, to see where tracking breaks.
static TRACE: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var_os("AVATAR_TRACE").is_some());
/// Below that it is unsure, typically at steep angles: its landmarks are still used if the
/// optical flow agrees with them (down to MIN_PRESENCE), or alone if there is no flow and
/// the previous frame already had a face here (down to TRACK_PRESENCE).
const TRACK_PRESENCE: f32 = 0.25;
const MIN_PRESENCE: f32 = 0.03;
const REDETECT_NEAR: std::time::Duration = std::time::Duration::from_secs(2);
const KEEP_FILTERS: std::time::Duration = std::time::Duration::from_millis(400);
/// Search interval while no face has been seen for 4x this long.
const SEARCH_IDLE: std::time::Duration = std::time::Duration::from_millis(250);

/// Best face in the square `region` of the frame (the whole frame, letterboxed, or a crop
/// around where the face was: BlazeFace short-range only sees 128 px, and a face of ~20 px
/// in a full 1080p frame is easily missed at a steep angle).
fn detect(detector: &Plan, anchors: &[Vec2], frame: &Frame, letterbox: Roi) -> Option<Roi> {
    let side = letterbox.size;
    let out = detector.run(tvec!(crop(frame, letterbox, 128, -1.0, 1.0).into())).ok()?;
    let regressors = out[0].as_slice::<f32>().ok()?;
    let scores = out[1].as_slice::<f32>().ok()?;

    // numFaces = 1: the single best anchor is enough, NMS would only matter for several faces.
    let (best, &logit) = scores.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1))?;
    if sigmoid(logit.clamp(-100.0, 100.0)) < 0.5 {
        return None;
    }
    let r = &regressors[best * 16..best * 16 + 16];
    let a = anchors[best];
    let at = |dx: f32, dy: f32| letterbox.to_frame(a + Vec2::new(dx, dy) / 128.0);
    let center = at(r[0], r[1]);
    let size = r[2].max(r[3]) / 128.0 * side;
    let (right_eye, left_eye) = (at(r[4], r[5]), at(r[6], r[7]));
    let eye_line = left_eye - right_eye;
    Some(Roi { center, size: size * 1.5, angle: eye_line.y.atan2(eye_line.x) })
}

/// The 478 landmarks inside `roi` and how sure the model is there is a face (0..1).
fn landmarks_in(landmarker: &Plan, frame: &Frame, roi: Roi) -> Option<(Vec<Vec3>, f32)> {
    let out = landmarker.run(tvec!(crop(frame, roi, 256, 0.0, 1.0).into())).ok()?;
    let presence = sigmoid(*out[1].as_slice::<f32>().ok()?.first()?);
    let raw = out[0].as_slice::<f32>().ok()?;
    let landmarks = raw
        .chunks_exact(3)
        .map(|p| {
            let xy = roi.to_frame(Vec2::new(p[0], p[1]) / 256.0);
            xy.extend(p[2] / 256.0 * roi.size)
        })
        .collect();
    Some((landmarks, presence))
}

/// Next frame's ROI from this frame's landmarks: box around them in the face's own rotated
/// frame, scaled 1.5x, square, like MediaPipe's tracking loop. The face's "right" direction
/// combines the eye line with the forehead-chin line turned 90 degrees: the eye line alone
/// gets short and wobbly when the head turns far, the vertical one when it nods far.
fn roi_from_landmarks(lm: &[Vec3]) -> Roi {
    let eye_line = lm[263].truncate() - lm[33].truncate();
    let down = lm[152].truncate() - lm[10].truncate();
    let right = eye_line + Vec2::new(down.y, -down.x) * 0.5;
    let angle = right.y.atan2(right.x);
    let probe = Roi { center: Vec2::ZERO, size: 1.0, angle };
    let (u, v) = probe.axes();
    let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
    for p in lm {
        let q = Vec2::new(p.truncate().dot(u), p.truncate().dot(v));
        lo = lo.min(q);
        hi = hi.max(q);
    }
    let mid = (lo + hi) * 0.5;
    Roi { center: u * mid.x + v * mid.y, size: (hi - lo).max_element() * 1.5, angle }
}

include!("procrustes_basis.rs");

/// Head pose from the 2D landmarks and the known 3D canonical face (weak perspective), over
/// the landmark subset and weights MediaPipe's geometry pipeline uses. The landmarks' z is
/// not used at all: it is a network guess with its own scale, and fitting it biased the
/// rotation and made the size change when you turned.
pub fn head_pose(lm: &[Vec3]) -> Option<Head> {
    let points: Vec<(Vec3, Vec2, f32)> = PROCRUSTES_BASIS.iter().map(|&(i, w, c)| (Vec3::from(c), lm[i].truncate(), w)).collect();
    fit_pose(&points)
}

/// Weak-perspective pose from canonical points (cm) and where they are in the frame (pixels,
/// y down), weighted. Fits image = M * canonical + t with M a 2x3 matrix (rows = scale *
/// first two rotation rows), then takes the nearest rotation.
pub fn fit_pose(points: &[(Vec3, Vec2, f32)]) -> Option<Head> {
    let total: f32 = points.iter().map(|p| p.2).sum();
    if !(total > 0.0) {
        return None;
    }
    // Image space -> camera space: y up.
    let live = |p: Vec2| Vec2::new(p.x, -p.y);
    let (mut lc, mut cc) = (Vec2::ZERO, Vec3::ZERO);
    for &(c, p, w) in points {
        lc += live(p) * w / total;
        cc += c * w / total;
    }
    // Normal equations: M = (sum w l c^T) (sum w c c^T)^-1, one row per image axis.
    let (mut lx, mut ly, mut cov) = (Vec3::ZERO, Vec3::ZERO, Mat3::ZERO);
    for &(c, p, w) in points {
        let (l, c) = (live(p) - lc, c - cc);
        lx += c * (l.x * w);
        ly += c * (l.y * w);
        cov += Mat3::from_cols(c * c.x, c * c.y, c * c.z) * w;
    }
    // The points must span all three dimensions (a patch of forehead alone can't tell a turn).
    let trace = cov.x_axis.x + cov.y_axis.y + cov.z_axis.z;
    if !(cov.determinant() > 1e-3 * (trace / 3.0).powi(3)) {
        return None;
    }
    let inv = cov.inverse(); // symmetric, so rows and columns agree
    let (a, b) = (inv * lx, inv * ly);
    let scale = (a.length() + b.length()) * 0.5;
    if !(scale > 0.0) {
        return None;
    }
    // Rotation rows ~ a/s, b/s; the third completes a right-handed frame.
    let rows = Mat3::from_cols(a, b, a.cross(b) / scale).transpose();
    let rotation = Quat::from_mat3(&polar_rotation(rows)?).normalize();
    let pivot = lc + Vec2::new(a.dot(NECK - cc), b.dot(NECK - cc));
    Some(Head { rotation, pivot: Vec2::new(pivot.x, -pivot.y), scale })
}

/// Whether two pose estimates describe the same head (MediaPipe unsure vs optical flow).
fn agrees(a: &Head, b: &Head) -> bool {
    a.rotation.angle_between(b.rotation) < 12f32.to_radians()
        && (a.scale / b.scale - 1.0).abs() < 0.15
        && (a.pivot - b.pivot).length() < 3.0 * b.scale
}

/// Orthogonal factor of the polar decomposition (Newton iteration). It is the rotation that
/// best maps canonical onto live, without needing a full SVD.
pub fn polar_rotation(h: Mat3) -> Option<Mat3> {
    if h.determinant() <= 1e-9 * h.to_cols_array().iter().map(|x| x.abs()).sum::<f32>().powi(3) {
        return None;
    }
    let mut r = h * (1.0 / h.to_cols_array().iter().map(|x| x * x).sum::<f32>().sqrt());
    for _ in 0..30 {
        r = (r + r.inverse().transpose()) * 0.5;
    }
    r.is_finite().then_some(r)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_count_matches_model() {
        assert_eq!(anchors().len(), 896);
    }

    #[test]
    fn procrustes_recovers_rotation() {
        // Canonical face rotated by a known pose, projected to image space like the model
        // output (y down, z away), at an arbitrary scale and offset.
        let truth = Quat::from_rotation_y(0.4) * Quat::from_rotation_x(-0.2) * Quat::from_rotation_z(0.1);
        let mut lm = vec![Vec3::ZERO; 478];
        for &(i, _, c) in &PROCRUSTES_BASIS {
            let p = truth * Vec3::from(c) * 23.0;
            lm[i] = Vec3::new(p.x + 320.0, -p.y + 240.0, -p.z);
        }
        let got = head_pose(&lm).unwrap().rotation;
        assert!(got.angle_between(truth) < 1e-3, "{got:?} vs {truth:?}");
    }

    #[test]
    fn head_scale_and_pivot_ignore_turning() {
        // Same head, same place, turned 45 degrees: scale and neck point must not move.
        // MediaPipe's z is squashed relative to x/y; make it so here too.
        // The landmarks' z must not matter at all (it has its own, unreliable scale).
        let place = |q: Quat, z: f32| {
            let mut lm = vec![Vec3::ZERO; 478];
            for &(i, _, c) in &PROCRUSTES_BASIS {
                let p = q * (Vec3::from(c) - NECK) * 23.0;
                lm[i] = Vec3::new(p.x + 320.0, -p.y + 240.0, -p.z * z);
            }
            head_pose(&lm).unwrap()
        };
        let turn = Quat::from_rotation_y(0.8) * Quat::from_rotation_x(0.2);
        for (z, tol) in [(1.0, 0.01), (0.5, 0.01), (0.0, 0.01)] {
            let (a, b) = (place(Quat::IDENTITY, z), place(turn, z));
            assert!((a.scale - 23.0).abs() < 0.3 && (b.scale - a.scale).abs() / a.scale < tol, "z {z}: {} {}", a.scale, b.scale);
            assert!((a.pivot - Vec2::new(320.0, 240.0)).length() < 1.0, "{:?}", a.pivot);
            assert!((b.pivot - a.pivot).length() < 1.0, "z {z}: {:?} vs {:?}", b.pivot, a.pivot);
        }
    }

    /// Face grid with a horizontal shift, the "eyelid" (point 159) raised by `lid`, and
    /// deterministic ~uniform noise in [-noise, noise].
    fn fake(shift: f32, lid: f32, noise: f32, k: usize) -> Vec<Vec3> {
        (0..478)
            .map(|i| {
                let base = Vec3::new(400.0 + (i % 23) as f32 * 11.0, 200.0 + (i / 23) as f32 * 12.0, 0.0);
                let n = |s: usize| ((((i * 7919 + k * 104729 + s * 31) % 1000) as f32) / 500.0 - 1.0) * noise;
                let lid = if i == 159 { lid } else { 0.0 };
                base + Vec3::new(shift + n(1), n(2) - lid, n(3))
            })
            .collect()
    }

    #[test]
    fn pose_filter_steadies_jitter_but_follows_a_turn() {
        use std::time::{Duration, Instant};
        let (mut f, t0) = (OneEuro::pose(), Instant::now());
        let at = |k: usize| t0 + Duration::from_millis(66 * k as u64);
        // Still head, 2 px jitter (~1.7 px/frame raw): output must barely move.
        let mut prev = f.apply(&fake(0.0, 0.0, 2.0, 0), at(0));
        let mut moved = 0.0;
        for k in 1..60 {
            let out = f.apply(&fake(0.0, 0.0, 2.0, k), at(k));
            moved += (out[100] - prev[100]).length();
            prev = out;
        }
        assert!(moved / 59.0 < 0.25, "still jitter {:.2} px/frame", moved / 59.0);
        // Head swings 200 px over 0.5 s: the output must keep up.
        for k in 60..68 {
            prev = f.apply(&fake((k - 59) as f32 * 25.0, 0.0, 2.0, k), at(k));
        }
        let lag = 200.0 - (prev[100].x - 400.0 - (100 % 23) as f32 * 11.0);
        assert!(lag < 60.0, "lags {lag:.0} px behind a fast turn");

        // A fast tilt: the face's mean barely moves, yet the output must keep up too.
        let mut f = OneEuro::pose();
        let base = fake(0.0, 0.0, 2.0, 0);
        let centre = base.iter().copied().sum::<Vec3>() / base.len() as f32;
        let tilt = |a: f32, k: usize| -> Vec<Vec3> {
            let r = Quat::from_rotation_z(a);
            fake(0.0, 0.0, 2.0, k).iter().map(|p| centre + r * (*p - centre)).collect()
        };
        for k in 0..30 {
            f.apply(&tilt(0.0, k), at(k));
        }
        let mut out = Vec::new();
        for k in 30..38 {
            out = f.apply(&tilt((k - 29) as f32 * 0.05, k), at(k)); // 0.4 rad in 0.5 s
        }
        let target = tilt(0.4, 0);
        let (o, t) = ((out[0] - centre).truncate(), (target[0] - centre).truncate());
        let got = o.angle_to(t).abs();
        assert!(got < 0.4 * 0.3, "tilt lags {:.0} percent behind", got / 0.4 * 100.0);
    }

    #[test]
    fn shape_filter_lets_a_blink_through() {
        use std::time::{Duration, Instant};
        let (mut f, t0) = (OneEuro::shapes(), Instant::now());
        let at = |k: usize| t0 + Duration::from_millis(66 * k as u64);
        for k in 0..30 {
            f.apply(&fake(0.0, 12.0, 1.0, k), at(k));
        }
        // Lid drops 12 px within two frames (a blink): most of it must show.
        f.apply(&fake(0.0, 6.0, 1.0, 30), at(30));
        let out = f.apply(&fake(0.0, 0.0, 1.0, 31), at(31));
        let shown = 12.0 - (fake(0.0, 0.0, 0.0, 0)[159].y - out[159].y);
        assert!(shown > 7.0, "blink shows {shown:.1} of 12 px");
    }

    #[test]
    fn detector_and_landmarker_run() {
        let frame = Frame { w: 64, h: 48, rgb: vec![128; 64 * 48 * 3] };
        let (d, l) = (load(DETECTOR).unwrap(), load(LANDMARKER).unwrap());
        let whole = Roi { center: Vec2::new(32.0, 24.0), size: 64.0, angle: 0.0 };
        assert!(detect(&d, &anchors(), &frame, whole).is_none(), "grey frame has no face");
        let near = Roi { center: Vec2::new(10.0, 10.0), size: 30.0, angle: 0.0 }; // partly outside
        assert!(detect(&d, &anchors(), &frame, near).is_none());
        let roi = Roi { center: Vec2::new(32.0, 24.0), size: 40.0, angle: 0.0 };
        let (_, presence) = landmarks_in(&l, &frame, roi).unwrap();
        assert!(presence < DETECT_PRESENCE, "grey frame has no face");
    }
}

/// Runs a recording through the tracker without a window:
/// `AVATAR_REPLAY=dir AVATAR_TRACE=1 cargo test --release replay_recording -- --ignored --nocapture`
#[cfg(test)]
#[test]
#[ignore]
fn replay_recording() {
    let shared = spawn(0, false);
    while !shared.lock().unwrap().error {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    println!("{}", shared.lock().unwrap().status);
}

/// Live check against the real webcam: `cargo test --release tracker::webcam -- --ignored --nocapture`.
/// Prints numbers only (face found, pose angles, frame-to-frame jitter), no images. Sit
/// still for the jitter number to mean anything.
#[cfg(test)]
#[test]
#[ignore]
fn webcam() {
    let shared = spawn(0, false);
    let start = std::time::Instant::now();
    let (mut frames, mut faces, mut steep, mut last) = (0, 0, 0, None::<*const Tracked>);
    let (mut prev, mut deltas) = (None::<Quat>, Vec::new());
    while start.elapsed().as_secs() < 10 {
        std::thread::sleep(std::time::Duration::from_millis(5));
        let st = shared.lock().unwrap();
        assert!(!st.error, "{}", st.status);
        let Some(t) = st.latest.clone() else { continue };
        if last == Some(Arc::as_ptr(&t)) {
            continue;
        }
        last = Some(Arc::as_ptr(&t));
        frames += 1;
        steep += st.status.contains("Bildpunkte") as usize;
        if let Some(head) = t.head {
            let q = head.rotation;
            faces += 1;
            if let Some(p) = prev {
                deltas.push(p.angle_between(q).to_degrees());
            }
            prev = Some(q);
            if faces % 30 == 1 {
                let (y, x, z) = q.to_euler(bevy::math::EulerRot::YXZ);
                println!(
                    "yaw {:.1} pitch {:.1} roll {:.1}  scale {:.2} px/cm  pivot {:.0},{:.0}",
                    y.to_degrees(), x.to_degrees(), z.to_degrees(), head.scale, head.pivot.x, head.pivot.y
                );
            }
        }
    }
    deltas.sort_by(f32::total_cmp);
    let pct = |p: f32| deltas.get((deltas.len() as f32 * p) as usize).copied().unwrap_or(f32::NAN);
    println!("{frames} frames in 10 s, face in {faces} ({steep} by optical flow); jitter per frame: median {:.2} deg, p90 {:.2} deg", pct(0.5), pct(0.9));
}
