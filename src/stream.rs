//! Panel switch / `--serve`: the avatar as an OBS browser source with real transparency, no window capture.
//! A second camera renders the scene (without the panel) into a texture that is read back
//! every frame and pushed over a WebSocket to a tiny page served on the same port.

use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Condvar, Mutex};
use tungstenite::{Bytes, Message};

pub const PORT: u16 = 4460;

/// The page OBS loads: draws each frame (8-byte header w, h, then premultiplied RGBA, as the
/// GPU rendered it) on a transparent canvas, and reconnects when the app restarts. WebGL takes
/// premultiplied pixels as they are; only without it does the page unpremultiply for a 2D
/// canvas (which then premultiplies again internally).
const PAGE: &str = r#"<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;background:transparent;overflow:hidden}
canvas{display:block;width:100vw;height:100vh;object-fit:contain}
</style></head><body><canvas id="c"></canvas><script>
const c = document.getElementById('c');
const gl = c.getContext('webgl', { premultipliedAlpha: true, alpha: true, antialias: false });
let draw, clear;
if (gl) {
  const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; };
  const p = gl.createProgram();
  gl.attachShader(p, sh(gl.VERTEX_SHADER, 'attribute vec2 p; varying vec2 uv; void main() { uv = vec2(p.x, -p.y) * 0.5 + 0.5; gl_Position = vec4(p, 0.0, 1.0); }'));
  gl.attachShader(p, sh(gl.FRAGMENT_SHADER, 'precision mediump float; varying vec2 uv; uniform sampler2D t; void main() { gl_FragColor = texture2D(t, uv); }'));
  gl.linkProgram(p); gl.useProgram(p);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
  for (const k of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, k, gl.LINEAR);
  for (const k of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, k, gl.CLAMP_TO_EDGE);
  draw = (w, h, px) => {
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; gl.viewport(0, 0, w, h); }
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, px);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  };
  clear = () => { gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); };
} else {
  const g = c.getContext('2d');
  draw = (w, h, px) => {
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    for (let i = 0; i < px.length; i += 4) {
      const a = px[i + 3];
      if (a && a < 255) { px[i] = px[i] * 255 / a; px[i + 1] = px[i + 1] * 255 / a; px[i + 2] = px[i + 2] * 255 / a; }
    }
    g.putImageData(new ImageData(new Uint8ClampedArray(px.buffer, px.byteOffset, px.length), w, h), 0, 0);
  };
  clear = () => g.clearRect(0, 0, c.width, c.height);
}
function connect() {
  const ws = new WebSocket('ws://' + location.host + '/');
  ws.binaryType = 'arraybuffer';
  ws.onmessage = e => {
    const d = new DataView(e.data), w = d.getUint32(0, true), h = d.getUint32(4, true);
    draw(w, h, new Uint8Array(e.data, 8, w * h * 4));
  };
  ws.onclose = () => { clear(); setTimeout(connect, 1000); };
}
connect();
</script></body></html>"#;

/// Newest frame and a counter, so each connection sends every new frame once.
type Latest = Arc<(Mutex<(u64, Option<Bytes>)>, Condvar)>;

/// The browser source: idle until switched on, then renders and reads back every frame.
#[derive(Resource)]
pub struct Stream {
    latest: Latest,
    image: Handle<Image>,
    view: Entity,
    readback: Entity,
    listening: bool,
    on: bool,
    /// Why it could not start, for the panel.
    pub problem: Option<String>,
}

/// `view` renders into `image`; nothing runs until `Config::serve` is on.
pub fn start(commands: &mut Commands, image: Handle<Image>, view: Entity, (w, h): (u32, u32)) {
    let latest: Latest = Default::default();
    let frames = latest.clone();
    let readback = commands
        .spawn_empty()
        .observe(move |ev: On<ReadbackComplete>| {
            let data = &ev.event().data;
            let (w, h) = (w as usize, h as usize);
            // Rows are padded to the GPU's copy alignment.
            let stride = data.len() / h.max(1);
            let mut out = Vec::with_capacity(8 + w * h * 4);
            out.extend((w as u32).to_le_bytes());
            out.extend((h as u32).to_le_bytes());
            // Premultiplied as rendered (edges blend into the transparent clear colour); the page
            // shows it as such, so no per-pixel work here on the main thread.
            for row in data.chunks(stride).take(h) {
                out.extend_from_slice(&row[..w * 4]);
            }
            publish(&frames, Some(Bytes::from(out)));
        })
        .id();
    commands.insert_resource(Stream { latest, image, view, readback, listening: false, on: false, problem: None });
}

/// Follows the panel switch. The port is opened the first time and then kept; switching off
/// stops the render and readback and closes open pages, which clear and wait to reconnect.
pub fn switch(mut stream: ResMut<Stream>, mut avatar: ResMut<crate::avatar::Avatar>, mut cameras: Query<&mut Camera>, mut commands: Commands) {
    let on = avatar.config.serve;
    if on == stream.on {
        return;
    }
    if on && !stream.listening {
        match TcpListener::bind(("127.0.0.1", PORT)) {
            Ok(listener) => {
                let latest = stream.latest.clone();
                std::thread::spawn(move || serve(listener, latest));
                stream.listening = true;
                stream.problem = None;
                eprintln!("OBS browser source: http://localhost:{PORT}/");
            }
            Err(e) => {
                stream.problem = Some(format!("Port {PORT} ist belegt ({e}). Läuft die App schon ein zweites Mal?"));
                avatar.config.serve = false;
                return;
            }
        }
    }
    stream.on = on;
    if let Ok(mut camera) = cameras.get_mut(stream.view) {
        camera.is_active = on;
    }
    if on {
        commands.entity(stream.readback).insert(Readback::texture(stream.image.clone()));
    } else {
        commands.entity(stream.readback).remove::<Readback>();
        publish(&stream.latest, None);
    }
}

fn publish(latest: &Latest, frame: Option<Bytes>) {
    let (lock, wake) = &**latest;
    let mut l = lock.lock().unwrap();
    *l = (l.0 + 1, frame);
    wake.notify_all();
}

fn serve(listener: TcpListener, latest: Latest) {
    for stream in listener.incoming().flatten() {
        let latest = latest.clone();
        std::thread::spawn(move || client(stream, latest));
    }
}

/// One connection: the page for a plain request, frames for a WebSocket.
fn client(mut stream: TcpStream, latest: Latest) {
    let mut head = [0u8; 2048];
    let mut n = 0;
    // Look at the request without consuming it: the WebSocket handshake needs it whole.
    for _ in 0..50 {
        n = stream.peek(&mut head).unwrap_or(0);
        if n == 0 || n == head.len() || head[..n].windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    if !String::from_utf8_lossy(&head[..n]).to_ascii_lowercase().contains("upgrade: websocket") {
        let _ = stream.read(&mut head);
        let _ = write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{PAGE}", PAGE.len());
        return;
    }
    let Ok(mut ws) = tungstenite::accept(stream) else { return };
    let mut seen = 0;
    loop {
        let frame = {
            let (lock, wake) = &*latest;
            let l = wake.wait_while(lock.lock().unwrap(), |l| l.0 == seen).unwrap();
            seen = l.0;
            l.1.clone()
        };
        // None: switched off. Closing makes the page clear and retry.
        let Some(f) = frame else { return };
        // Sending blocks while the browser is busy, so a slow viewer just skips frames.
        if ws.send(Message::Binary(f)).is_err() {
            return;
        }
    }
}
