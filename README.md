# avatar-rs

Ein 3D-Modell mit Face-Tracking als Stream-Avatar, nativ in Rust. Augen und Mund sind echte Ausschnitte aus der
Webcam und sitzen plastisch auf deiner Figur. Die Figur bringst du selbst mit, als `.blend`, `.glb`, `.fbx`, `.obj`
und mehr. Dazu gibt es ein Maus-Overlay, einen OBS-WebSocket-Client und globale Hotkeys. Alles steckt in einer
einzigen Binary und läuft komplett offline.

| Schicht | Umsetzung |
|---|---|
| 3D-Figur | Bevy 0.17 (PBR), eigene Modelle per Import |
| Face-Tracking | MediaPipes TFLite-Modelle (BlazeFace + 478-Punkt-Mesh) über `tract` auf der CPU, Kopf-Pose per gewichtetem Procrustes gegen MediaPipes kanonisches Gesicht, Optical Flow zwischen den Erkennungen |
| Augen und Mund | frontalisierte Webcam-Ausschnitte (tiny-skia) als Mesh-Patches mit Relief auf der Oberfläche der Figur |
| Einstellungen | egui-Panel, gespeichert in `~/.config/avatar-rs/config.json` |

## Installation

### Arch Linux

Mit dem PKGBUILD aus dem Repo. Es baut das letzte Release für deine CPU (`target-cpu=native`) und installiert es nach
`/usr/bin/avatar-rs`:

```sh
git clone https://github.com/marcomaiermm/avatar-rs.git
cd avatar-rs/packaging/arch
makepkg -si
```

Empfohlen dazu:

```sh
sudo pacman -S --needed blender xdg-desktop-portal v4l-utils
sudo usermod -aG input "$USER"   # globale Hotkeys, danach neu einloggen
```

- `blender` wandelt alles außer glTF beim Import um (`.blend`, `.fbx`, `.obj`, `.stl`, `.ply`, `.usd*`, `.abc`).
- `xdg-desktop-portal` zeigt den Dateidialog. Unter KDE, GNOME und Hyprland ist er meist schon installiert, dazu das
  passende Backend wie `xdg-desktop-portal-kde`, `-gnome`, `-gtk` oder `-hyprland`.
- `v4l-utils` brauchst du für die Webcam-Tipps weiter unten.

Updates: `git pull` im Repo und `makepkg -si` erneut ausführen.

### Fertige Binary (Linux x86_64)

Jedes [Release](https://github.com/marcomaiermm/avatar-rs/releases) hat ein
`avatar-rs-vX.Y.Z-x86_64-linux.tar.gz`. Die Binary braucht eine CPU ab 2013 (x86-64-v3: Haswell oder Zen),
Vulkan-Treiber und auf Arch `alsa-lib`, `libxkbcommon`, `systemd-libs` und `wayland`.

```sh
tar xzf avatar-rs-*-x86_64-linux.tar.gz
install -Dm755 avatar-rs-*/avatar-rs ~/.local/bin/avatar-rs
```

### Aus dem Quellcode

```sh
sudo pacman -S --needed rust clang nasm alsa-lib systemd-libs libxkbcommon wayland vulkan-icd-loader
# Debian/Ubuntu: sudo apt install cargo clang nasm libasound2-dev libudev-dev libwayland-dev libxkbcommon-dev
cargo install --locked --git https://github.com/marcomaiermm/avatar-rs
```

## Start

```
avatar-rs                       # Avatar-Fenster 512x512, transparent
avatar-rs --edit                # mit Einstellungs-Panel (im Fenster: C)
avatar-rs --model figur.blend   # eigene Figur importieren und laden
avatar-rs --debug               # ohne Webcam, mit gemaltem Testgesicht
avatar-rs --serve               # zusätzlich als OBS-Browserquelle (http://localhost:4460/)
avatar-rs --mesh                # Debug: Kamerabild mit Face-Mesh neben dem Avatar
avatar-rs --camera 1            # andere Webcam
avatar-rs mouse                 # Maus-Overlay (input-overlay-Plugin, ws://localhost:16899)
```

## Eigene Figur

Im Panel (Taste **C**) neben „Figur“ auf **Importieren …** klicken und die Datei wählen. Unterstützt werden `.glb`,
`.gltf`, `.blend`, `.fbx`, `.obj`, `.stl`, `.ply`, `.usd`, `.usda`, `.usdc`, `.usdz` und `.abc`.

- glTF lädt direkt. Alles andere wandelt Blender einmal headless in ein GLB um, abgelegt unter
  `~/.local/share/avatar-rs/models/`. Läuft Blender nicht über `blender` im `PATH`, setzt du den Pfad mit `AVATAR_BLENDER=/pfad/zu/blender`.
- Importierte Figuren stehen danach in der Liste. Die zuletzt gewählte lädt beim nächsten Start automatisch.
  Ohne Import zeigt die App eine schlichte Kugel.
- Die Figur wird automatisch eingepasst. Wo das Gesicht sitzt und wie groß es ist, stellst du im Panel unter
  „Gesicht“ ein. Das wird pro Figur gespeichert.
- Bei `.blend` kommen nur Materialien an, die glTF kennt (Principled BSDF mit Bildtexturen). Prozedurale
  Node-Setups vorher in Texturen backen.

## In OBS

**Empfohlen:** Im Panel „Für OBS bereitstellen“ einschalten (bleibt gespeichert; alternativ `--serve`). Dann in OBS
eine Quelle „Browser“ mit der URL `http://localhost:4460/` und 512x512 anlegen. Das ergibt echte Transparenz, ohne
Panel und Debug-Ansicht, und das Fenster darf verdeckt sein. Startet die App neu, verbindet sich die Quelle von selbst.

**Alternativ:** das Fenster „avatar-rs“ aufnehmen, unter Wayland über „Bildschirmaufnahme (PipeWire)“.

## Kalibrierung

- **Ruhepose:** Die ersten ~1,5 s mit erkanntem Gesicht legen fest, was „geradeaus“ ist und wo du sitzt. Neu
  messen geht über „Neu ausrichten“ im Panel oder über die Hotkey-Aktion `recalibrate`.
- **Gesicht in echten Proportionen:** Augenabstand, Mundposition und Größen kommen live aus deinem Gesicht. Jedes
  Bild wird über die Kopfhaltung und MediaPipes kanonisches Gesicht frontalisiert. Blinzeln und Mundöffnen bleiben
  dabei erhalten, Drehen und Neigen werden herausgerechnet.
- **Plastisch:** Augen und Mund sind Mesh-Patches auf der Oberfläche der Figur, mit Relief für Lider, Augapfel und
  Lippen und mit Szenenlicht. Die Stärke stellst du unter „Gesicht → Lage auf der Figur“ mit „Relief“ und „Beleuchtung“ ein.

## Hotkeys und OBS-Steuerung

In `~/.config/avatar-rs/config.json` (die Datei entsteht beim ersten Speichern im Panel):

```json
{
  "obs": { "url": "ws://127.0.0.1:4455", "password": "..." },
  "hotkeys": [
    { "keys": "Ctrl+Alt+1", "action": { "obsScene": "Gameplay" } },
    { "keys": "Ctrl+Alt+2", "action": { "obsToggleSource": { "scene": "Gameplay", "source": "Avatar" } } },
    { "keys": "Ctrl+Alt+F", "action": { "model": "/home/du/.local/share/avatar-rs/models/figur.glb" } },
    { "keys": "Ctrl+Alt+M", "action": "toggleMirror" },
    { "keys": "Ctrl+Alt+R", "action": "recalibrate" }
  ]
}
```

In OBS muss der WebSocket-Server aktiv sein: Werkzeuge → WebSocket-Servereinstellungen. Unter Linux liest die App
die Hotkeys direkt über evdev. Dadurch funktionieren sie auch unter Wayland und wenn eine andere App den Fokus hat.
Dafür muss dein Nutzer in der Gruppe `input` sein.

## Tipps

- **Ruckelt der Avatar bei wenig Licht?** Viele Webcams halbieren dann die Framerate:
  `v4l2-ctl -d /dev/video0 -c exposure_dynamic_framerate=0`

## Entwicklung

```sh
cargo run --release -- --edit
cargo test --release
cargo test --release tracker::webcam -- --ignored --nocapture   # Live-Check mit der Webcam
```

| Variable | Wirkung |
|---|---|
| `AVATAR_TRACE=1` | eine Zeile pro Kamerabild auf stderr, um zu sehen, wo das Tracking abreißt |
| `AVATAR_RECORD=dir` | speichert jedes Kamerabild als JPEG |
| `AVATAR_REPLAY=dir` | spielt eine Aufnahme statt der Kamera ab |
| `AVATAR_SCREENSHOT=out.png` | Screenshot nach 90 Frames, dann Ende (`AVATAR_SCREENSHOT_FRAME=n`) |
| `AVATAR_BLENDER=pfad` | Blender für den Import |

### Releases

Commits auf `main` folgen [Conventional Commits](https://www.conventionalcommits.org/) (`feat: …`, `fix: …`,
`perf: …`). Der PR-Titel wird beim Squash-Merge zum Commit und wird geprüft. Daraus pflegt
[release-please](https://github.com/googleapis/release-please) einen Release-PR mit neuer Version in `Cargo.toml` und
`packaging/arch/PKGBUILD` sowie dem Eintrag in `CHANGELOG.md`. Wird dieser PR gemergt, entstehen Tag und
GitHub-Release, und die Linux-Binary hängt mit Build-Provenance am Release.

## Herkunft und Drittkomponenten

- Ursprünglich ein nativer Port von [J-Pster/pster-live-avatar](https://github.com/J-Pster/pster-live-avatar).
- `assets/face_*.tflite`: aus MediaPipes `face_landmarker.task` (Apache 2.0).
- `vendor/tract-tflite`: tract-tflite 0.22.5 (MIT/Apache 2.0), ergänzt um DEQUANTIZE und PRELU sowie schnellere
  NHWC-Kernels für MediaPipes Modelle.
