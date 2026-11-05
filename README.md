# avatar-rs

A face-tracked 3D stream avatar, native in Rust. The eyes and mouth are real cutouts from your webcam and sit on your
figure in relief. You bring the figure yourself, as a `.blend`, `.glb`, `.fbx`, `.obj` file and more. It also comes
with a mouse overlay, an OBS WebSocket client and global hotkeys. Everything is a single binary and runs fully offline.

| Layer | How |
|---|---|
| 3D figure | Bevy 0.17 (PBR), your own models via import |
| Face tracking | MediaPipe's TFLite models (BlazeFace + 478-point mesh) through `tract` on the CPU, head pose by weighted Procrustes against MediaPipe's canonical face, optical flow between detections |
| Eyes and mouth | frontalized webcam cutouts (tiny-skia) as mesh patches with relief on the figure's surface |
| Settings | egui panel, saved in `~/.config/avatar-rs/config.json` |

The app's interface is in German. The labels below are given in German with an English translation.

## Installation

### Arch Linux

Use the PKGBUILD from the repo. It builds the latest release for your CPU (`target-cpu=native`), installs it to
`/usr/bin/avatar-rs` and adds avatar-rs to your app menu:

```sh
git clone https://github.com/marcomaiermm/avatar-rs.git
cd avatar-rs/packaging/arch
makepkg -si
```

Recommended alongside:

```sh
sudo pacman -S --needed blender xdg-desktop-portal v4l-utils
sudo usermod -aG input "$USER"   # global hotkeys; log in again afterwards
```

- `blender` converts everything except glTF on import (`.blend`, `.fbx`, `.obj`, `.stl`, `.ply`, `.usd*`, `.abc`).
- `xdg-desktop-portal` shows the file dialog. On KDE, GNOME and Hyprland it is usually installed already, together
  with the matching backend such as `xdg-desktop-portal-kde`, `-gnome`, `-gtk` or `-hyprland`.
- You need `v4l-utils` for the webcam tip further down.

To update, run `git pull` in the repo and `makepkg -si` again.

### Prebuilt binary (Linux x86_64)

Every [release](https://github.com/marcomaiermm/avatar-rs/releases) has an `avatar-rs-vX.Y.Z-x86_64-linux.tar.gz`.
The binary needs a CPU from 2013 or later (x86-64-v3: Haswell or Zen) and Vulkan drivers. On Arch it also needs
`alsa-lib`, `libxkbcommon`, `systemd-libs` and `wayland`.

```sh
tar xzf avatar-rs-*-x86_64-linux.tar.gz
cd avatar-rs-*/
install -Dm755 avatar-rs ~/.local/bin/avatar-rs
# app menu entry and icon
install -Dm644 avatar-rs.desktop -t ~/.local/share/applications
install -Dm644 avatar-rs.svg -t ~/.local/share/icons/hicolor/scalable/apps
```

### From source

```sh
sudo pacman -S --needed rust clang nasm alsa-lib systemd-libs libxkbcommon wayland vulkan-icd-loader
# Debian/Ubuntu: sudo apt install cargo curl clang nasm libasound2-dev libudev-dev libwayland-dev libxkbcommon-dev
cargo install --locked --git https://github.com/marcomaiermm/avatar-rs
```

For the app menu entry, copy `packaging/linux/avatar-rs.desktop` and `packaging/linux/avatar-rs.svg` as shown for the
prebuilt binary. The entry runs `avatar-rs`, so `~/.cargo/bin` has to be on your `PATH`.

## Usage

From the app menu, start **avatar-rs**. Right-click the entry for "Open with settings" or "Mouse overlay". From a
terminal:

```
avatar-rs                       # avatar window 512x512, transparent
avatar-rs --edit                # with the settings panel (in the window: C)
avatar-rs --model figure.blend  # import and load your own figure
avatar-rs --debug               # no webcam, with a painted test face
avatar-rs --serve               # also as an OBS browser source (http://localhost:4460/)
avatar-rs --mesh                # debug: camera image with the face mesh next to the avatar
avatar-rs --camera 1            # another webcam
avatar-rs mouse                 # mouse overlay (input-overlay plugin, ws://localhost:16899)
```

## Your own figure

In the panel (key **C**), click **Importieren …** (Import) next to "Figur" (Figure) and pick the file. Supported
formats are `.glb`, `.gltf`, `.blend`, `.fbx`, `.obj`, `.stl`, `.ply`, `.usd`, `.usda`, `.usdc`, `.usdz` and `.abc`.

- glTF loads directly. Blender converts everything else once, headless, into a GLB stored in
  `~/.local/share/avatar-rs/models/`. If Blender is not on your `PATH` as `blender`, set `AVATAR_BLENDER=/path/to/blender`.
- Imported figures then appear in the list. The one you picked last loads on the next start. Without an import the
  app shows a plain sphere.
- The figure is fitted automatically. Where the face sits on it and how large it is, you set in the panel under
  "Gesicht" (Face). These settings are saved per figure.
- From a `.blend`, only materials that glTF understands come through (Principled BSDF with image textures). Bake
  procedural node setups into textures first.

## In OBS

**Recommended:** In the panel, switch on "Für OBS bereitstellen" (Provide for OBS). It stays saved; `--serve` does the
same. Then add a "Browser" source in OBS with the URL `http://localhost:4460/` at 512x512. This gives real
transparency, without the panel or debug view, and the window may be covered. If the app restarts, the source
reconnects on its own.

**Alternative:** capture the "avatar-rs" window, on Wayland via "Screen Capture (PipeWire)".

## Calibration

- **Rest pose:** The first ~1.5 s with a detected face define what "straight ahead" is and where you sit. To measure
  again, use "Neu ausrichten" (Re-align) in the panel or the hotkey action `recalibrate`.
- **True proportions:** Eye distance, mouth position and sizes come live from your face. Every frame is frontalized
  using your head pose and MediaPipe's canonical face. Blinking and mouth opening come through; turning and tilting
  are taken out.
- **In relief:** Eyes and mouth are mesh patches on the figure's surface, with relief for lids, eyeball and lips, and
  they are lit by the scene. You set the strength under "Gesicht → Lage auf der Figur" (Face → Placement on the
  figure) with "Relief" and "Beleuchtung" (Lighting).

## Hotkeys and OBS control

In `~/.config/avatar-rs/config.json` (the file is created the first time the panel saves):

```json
{
  "obs": { "url": "ws://127.0.0.1:4455", "password": "..." },
  "hotkeys": [
    { "keys": "Ctrl+Alt+1", "action": { "obsScene": "Gameplay" } },
    { "keys": "Ctrl+Alt+2", "action": { "obsToggleSource": { "scene": "Gameplay", "source": "Avatar" } } },
    { "keys": "Ctrl+Alt+F", "action": { "model": "/home/you/.local/share/avatar-rs/models/figure.glb" } },
    { "keys": "Ctrl+Alt+M", "action": "toggleMirror" },
    { "keys": "Ctrl+Alt+R", "action": "recalibrate" }
  ]
}
```

OBS needs its WebSocket server switched on: Tools → WebSocket Server Settings. On Linux the app reads the hotkeys
directly through evdev, so they work under Wayland and while another app has focus. Your user needs to be in the
`input` group for that.

## Tips

- **Avatar stutters in low light?** Many webcams halve their frame rate then:
  `v4l2-ctl -d /dev/video0 -c exposure_dynamic_framerate=0`

## Development

```sh
cargo run --release -- --edit
cargo test --release
cargo test --release tracker::webcam -- --ignored --nocapture   # live check with the webcam
```

| Variable | Effect |
|---|---|
| `AVATAR_TRACE=1` | one line per camera frame on stderr, to see where tracking breaks |
| `AVATAR_RECORD=dir` | saves every camera frame as a JPEG |
| `AVATAR_REPLAY=dir` | plays a recording back instead of the camera |
| `AVATAR_SCREENSHOT=out.png` | screenshot after 90 frames, then exit (`AVATAR_SCREENSHOT_FRAME=n`) |
| `AVATAR_BLENDER=path` | the Blender used for imports |

### Releases

Commits on `main` follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat: …`, `fix: …`,
`perf: …`). On a squash merge the PR title becomes the commit, and a workflow checks it. From these commits
[release-please](https://github.com/googleapis/release-please) maintains a release PR that bumps the version in
`Cargo.toml` and `packaging/arch/PKGBUILD` and adds the entry to `CHANGELOG.md`. Merging that PR creates the tag and
the GitHub release, and the Linux binary is attached to the release with build provenance.

## License and third-party components

avatar-rs started as a native port of [J-Pster/pster-live-avatar](https://github.com/J-Pster/pster-live-avatar)
and is, like the original, under the [PolyForm Noncommercial License 1.0.0](LICENSE): free to use, change and share
for any noncommercial purpose.

- Face models: the repo holds no models. The build downloads MediaPipe's
  [`face_landmarker.task`](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)
  (Apache 2.0) with `curl`, checks its SHA-256 and bakes the face detector and the landmark model into the binary.
  To build offline, set `AVATAR_FACE_LANDMARKER=/path/to/face_landmarker.task`.
- `vendor/tract-tflite`: tract-tflite 0.22.5 (MIT/Apache 2.0), extended with DEQUANTIZE and PRELU and with faster
  NHWC kernels for MediaPipe's models.
