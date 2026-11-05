//! Mouse overlay (`mouse` subcommand): button/wheel sprites plus a direction indicator, driven by
//! the input-overlay OBS plugin's websocket. Port of mouse-keyboard/mouse_overlay.html: tiny-skia
//! plays the role of the 2D canvas, the result is uploaded as one full-window image per frame.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use serde::Deserialize;
use std::f32::consts::{FRAC_PI_2, PI, TAU};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tiny_skia::{BlendMode, Color, FillRule, IntRect, Paint, PathBuilder, Pixmap, PixmapPaint, Rect, Stroke, Transform};

const W: u32 = 285;
const H: u32 = 421;
const CENTER: (f32, f32) = (143.5, 300.0);
const MIN_MOVE_PX: f64 = 3.0;
const IDLE: Duration = Duration::from_millis(250);
// Per-frame lerp factors of the original, tuned at 60 fps; rescaled by frame time in `render`.
const ANGLE_LERP: f32 = 0.18;
const ALPHA_LERP: f32 = 0.12;
const RADIUS: f32 = 78.0;

/// (atlas x, y, w, h, draw x, y, button index into `State::down`), drawn in this order.
/// The atlas has no pressed frames, so pressing tints the crop instead.
const PARTS: [(u32, u32, u32, u32, f32, f32, Option<usize>); 6] = [
    (1, 1, 139, 174, 2.0, 0.0, Some(0)),     // left
    (143, 1, 139, 174, 146.0, 0.0, Some(1)), // right
    (285, 246, 48, 95, 117.0, 79.0, None),   // wheel
    (285, 1, 40, 62, 0.0, 210.0, Some(2)),   // side button 1
    (284, 1, 41, 62, 11.0, 273.0, Some(3)),  // side button 2
    (328, 1, 283, 242, 2.0, 179.0, None),    // body
];

#[derive(Default)]
struct State {
    down: [bool; 4],
    last: Option<(f64, f64)>,
    target_angle: f32,
    last_move: Option<Instant>,
}

#[derive(Deserialize)]
struct Event {
    event_type: String,
    #[serde(default)]
    x: f64,
    #[serde(default)]
    y: f64,
    #[serde(default)]
    button: i32,
}

impl State {
    fn on_event(&mut self, e: Event) {
        match e.event_type.as_str() {
            "mouse_moved" | "mouse_dragged" => {
                if let Some((lx, ly)) = self.last {
                    let (dx, dy) = (e.x - lx, e.y - ly);
                    if dx.hypot(dy) >= MIN_MOVE_PX {
                        self.target_angle = dy.atan2(dx) as f32;
                        self.last_move = Some(Instant::now());
                    }
                }
                self.last = Some((e.x, e.y));
            }
            t @ ("mouse_pressed" | "mouse_released") => {
                let i = match e.button {
                    1 => 0,
                    2 => 1,
                    5 => 2,
                    4 => 3,
                    _ => return,
                };
                self.down[i] = t == "mouse_pressed";
            }
            _ => {}
        }
    }
}

fn listen(state: Arc<Mutex<State>>) {
    loop {
        if let Ok((mut ws, _)) = tungstenite::connect("ws://localhost:16899/") {
            while let Ok(msg) = ws.read() {
                if let Ok(text) = msg.to_text()
                    && let Ok(e) = serde_json::from_str(text)
                {
                    state.lock().unwrap().on_event(e);
                }
            }
        }
        std::thread::sleep(Duration::from_secs(2));
    }
}

struct Part {
    plain: Pixmap,
    pressed: Pixmap,
    pos: (f32, f32),
    button: Option<usize>,
}

#[derive(Resource)]
struct Overlay {
    parts: Vec<Part>,
    canvas: Pixmap,
    image: Handle<Image>,
    state: Arc<Mutex<State>>,
    shown_angle: f32,
    arrow_alpha: f32,
}

pub fn plugin(app: &mut App) {
    let atlas = Pixmap::decode_png(include_bytes!("../assets/mouse_body.png")).expect("mouse_body.png");
    let parts = PARTS
        .iter()
        .map(|&(x, y, w, h, dx, dy, button)| {
            let plain = atlas.clone_rect(IntRect::from_xywh(x as i32, y as i32, w, h).unwrap()).expect("atlas crop");
            let mut pressed = plain.clone();
            let tint = Paint { blend_mode: BlendMode::SourceAtop, ..paint(236, 180, 5, 0.55) };
            let rect = Rect::from_xywh(0.0, 0.0, w as f32, h as f32).unwrap();
            pressed.fill_rect(rect, &tint, Transform::identity(), None);
            Part { plain, pressed, pos: (dx, dy), button }
        })
        .collect();

    let state = Arc::new(Mutex::new(State::default()));
    let thread_state = state.clone();
    std::thread::spawn(move || listen(thread_state));

    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    let image = images.add(Image::new_fill(
        Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        TextureDimension::D2,
        &[0; 4],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    ));
    app.insert_resource(Overlay { parts, canvas: Pixmap::new(W, H).unwrap(), image, state, shown_angle: 0.0, arrow_alpha: 0.0 })
        .add_systems(Startup, setup)
        .add_systems(Update, render);
}

fn setup(mut commands: Commands, overlay: Res<Overlay>) {
    commands.spawn(Camera2d);
    commands.spawn((
        Node { position_type: PositionType::Absolute, width: percent(100), height: percent(100), ..default() },
        ImageNode::new(overlay.image.clone()),
    ));
}

fn paint(r: u8, g: u8, b: u8, a: f32) -> Paint<'static> {
    let mut p = Paint::default();
    p.set_color(Color::from_rgba8(r, g, b, (a * 255.0).round() as u8));
    p.anti_alias = true;
    p
}

fn angle_diff(a: f32, b: f32) -> f32 {
    let d = (b - a) % TAU;
    if d > PI {
        d - TAU
    } else if d < -PI {
        d + TAU
    } else {
        d
    }
}

fn render(mut overlay: ResMut<Overlay>, mut images: ResMut<Assets<Image>>, time: Res<Time>) {
    let o = &mut *overlay;
    let (down, target, moving) = {
        let s = o.state.lock().unwrap();
        (s.down, s.target_angle, s.last_move.is_some_and(|t| t.elapsed() <= IDLE))
    };
    let frames = time.delta_secs() * 60.0;
    let k = |lerp: f32| 1.0 - (1.0 - lerp).powf(frames);
    o.arrow_alpha += ((moving as u8 as f32) - o.arrow_alpha) * k(ALPHA_LERP);
    o.shown_angle += angle_diff(o.shown_angle, target) * k(ANGLE_LERP);

    o.canvas.fill(Color::TRANSPARENT);
    for p in &o.parts {
        let pm = if p.button.is_some_and(|i| down[i]) { &p.pressed } else { &p.plain };
        o.canvas.draw_pixmap(p.pos.0 as i32, p.pos.1 as i32, pm.as_ref(), &PixmapPaint::default(), Transform::identity(), None);
    }
    draw_indicator(&mut o.canvas, o.shown_angle, o.arrow_alpha);

    // tiny-skia is premultiplied, Bevy wants straight alpha.
    if let Some(img) = images.get_mut(&o.image)
        && let Some(data) = img.data.as_mut()
    {
        for (dst, px) in data.chunks_exact_mut(4).zip(o.canvas.pixels()) {
            let c = px.demultiply();
            dst.copy_from_slice(&[c.red(), c.green(), c.blue(), c.alpha()]);
        }
    }
}

fn draw_indicator(canvas: &mut Pixmap, angle: f32, arrow_alpha: f32) {
    let at_center = Transform::from_translate(CENTER.0, CENTER.1);
    let circle = |r: f32| PathBuilder::from_circle(0.0, 0.0, r).unwrap();

    let disc = circle(RADIUS);
    canvas.fill_path(&disc, &paint(250, 250, 250, 0.82), FillRule::Winding, at_center, None);
    let outline = Stroke { width: 3.0, ..Stroke::default() };
    canvas.stroke_path(&disc, &paint(30, 30, 30, 0.9), &outline, at_center, None);

    let ball = circle(RADIUS * 0.16 * (1.0 - arrow_alpha * 0.55));
    canvas.fill_path(&ball, &paint(30, 30, 30, 0.95), FillRule::Winding, at_center, None);

    if arrow_alpha <= 0.01 {
        return;
    }
    let t = Transform::from_rotate((angle + FRAC_PI_2).to_degrees()).post_translate(CENTER.0, CENTER.1);
    let tip_y = -RADIUS * 0.82;
    let base_y = tip_y + RADIUS * 0.46;
    let head_w = RADIUS * 0.32;
    let shaft_w = RADIUS * 0.075;
    let shaft_end = RADIUS * 0.30;

    let mut shaft = PathBuilder::new();
    shaft.move_to(-shaft_w, base_y);
    shaft.line_to(-shaft_w, shaft_end);
    shaft.line_to(shaft_w, shaft_end);
    shaft.line_to(shaft_w, base_y);
    shaft.close();
    canvas.fill_path(&shaft.finish().unwrap(), &paint(30, 30, 30, 0.95 * arrow_alpha), FillRule::Winding, t, None);

    let mut head = PathBuilder::new();
    head.move_to(0.0, tip_y);
    head.line_to(-head_w, base_y);
    head.line_to(head_w, base_y);
    head.close();
    let head = head.finish().unwrap();
    canvas.fill_path(&head, &paint(236, 180, 5, arrow_alpha), FillRule::Winding, t, None);
    let outline = Stroke { width: 2.5, ..Stroke::default() };
    canvas.stroke_path(&head, &paint(30, 30, 30, 0.95 * arrow_alpha), &outline, t, None);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_drive_state() {
        let ev = |s: &str| serde_json::from_str::<Event>(s).unwrap();
        let mut s = State::default();
        s.on_event(ev(r#"{"event_type":"mouse_pressed","button":1}"#));
        s.on_event(ev(r#"{"event_type":"mouse_pressed","button":5}"#));
        assert_eq!(s.down, [true, false, true, false]);
        s.on_event(ev(r#"{"event_type":"mouse_moved","x":10,"y":10}"#));
        s.on_event(ev(r#"{"event_type":"mouse_moved","x":10,"y":11}"#)); // under 3 px: ignored
        assert!(s.last_move.is_none());
        s.on_event(ev(r#"{"event_type":"mouse_moved","x":10,"y":20}"#));
        assert!((s.target_angle - FRAC_PI_2).abs() < 1e-6 && s.last_move.is_some());
        assert!((angle_diff(3.0, -3.0) - (TAU - 6.0)).abs() < 1e-5);
    }
}
