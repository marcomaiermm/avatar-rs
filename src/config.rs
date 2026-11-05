//! Settings file: per-model calibration (what the panel edits), OBS connection, hotkeys.
//! Lives at <config dir>/avatar-rs/config.json, replacing the browser's localStorage.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Bump when a field changes meaning or scale. Saved calibration with another version is
/// discarded, otherwise an old tweak masks the new default and the change seems to do nothing.
pub const VERSION: u32 = 8;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default, rename_all = "camelCase")]
pub struct Calibration {
    pub version: u32,
    pub model_scale: f32,
    pub model_y: f32,
    pub turn_gain: f32,
    /// How far the model follows the head moving around in the camera image (1 = same
    /// fraction of the window as of the camera frame).
    pub follow: f32,
    /// How far leaning toward/away from the camera moves the model toward/away from the
    /// viewer (exponent on the face's size change; 1 = as much as you moved, 0 = off).
    pub depth_gain: f32,
    pub smoothing: f32,
    pub mirror: bool,
    /// Extra pitch offset (radians) on top of the auto-calibrated neutral pose. Negative lifts
    /// the chin. The neutral pose already cancels the webcam sitting above eye level.
    pub base_pitch: f32,
    /// Applied to the cutouts: eyes and mouth are shaded cavities and come out darker.
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    /// CSS hue-rotate on the cutouts, degrees. Moves skin tones while whites stay white and
    /// darks dark: about -100 gives the lavender eyes/lips look of the reference streams.
    pub hue_rotate: f32,
    /// Unsharp mask strength on the cutouts (0 = off).
    pub sharpen: f32,
    /// Temporal denoising of the cutouts, 0..1 (0 = off).
    pub denoise: f32,
    /// Measure the skin around the eyes each frame and brighten dim rooms automatically.
    pub auto_exposure: bool,
    /// Without a face for a moment, hide the model instead of resting it in its neutral pose.
    pub hide_when_lost: bool,
    /// Where the face sits on the model: the point between the eyes, in model units (the
    /// model spans -1..1), and the face's ear-to-ear width in the same units. Eye spacing,
    /// mouth position and all sizes come live from the real face, scaled by this, so the
    /// features keep natural proportions instead of being three independent patches.
    pub face_x: f32,
    pub face_y: f32,
    pub face_scale: f32,
    /// Pulls the features together (<1) or apart. Real proportions look long and sparse on a
    /// stylised object; cartoon faces sit closer.
    pub spacing: f32,
    /// How flat the features lie (0 = hug the curved surface, 1 = a flat card in front of
    /// it). On a narrow body the curved sides tilt away; a feature painted there grows when
    /// it turns toward the camera. Flatter = it foreshortens like a real face instead.
    pub flatten: f32,
    /// How much the eyes and mouth stand out in 3D (lids, eyeball, lips, the open mouth's
    /// depth; 0 = flat, 1 = real depth at the features' size).
    pub relief: f32,
    /// How much of the features' colour the scene light shades (0 = shown exactly as the
    /// camera saw it, unlit; more = shading and gloss like the body, a little darker).
    pub shading: f32,
    /// Manual nudges in model units on top of the live layout. eye_x is symmetric: positive
    /// moves both eyes apart.
    pub eye_x: f32,
    pub eye_y: f32,
    pub mouth_x: f32,
    pub mouth_y: f32,
    /// Exaggeration of the eyes/mouth on top of the real proportions.
    pub eye_size: f32,
    /// Big-eye lens on the iris (0 = off, 1 = twice as big in the middle).
    pub eye_bulge: f32,
    /// Extra eye height: rounder, cartoon eyes (0 = real shape).
    pub eye_round: f32,
    pub mouth_size: f32,
    /// Crop box margin around the inflated contour; >1 keeps the mask off the crop edge.
    pub eye_margin: f32,
    pub mouth_margin: f32,
    /// How much the raw contour is inflated before it becomes the mask (skin around it).
    pub eye_dilation: f32,
    pub mouth_dilation: f32,
    pub eye_softness: f32,
    pub mouth_softness: f32,
}

impl Default for Calibration {
    /// Grenade framing from the original avatar.html; face layout retuned for the
    /// proportional mode (tight masks, natural spacing, calmer turns).
    fn default() -> Self {
        Self {
            version: VERSION,
            model_scale: 0.80,
            model_y: -0.26,
            turn_gain: 1.0,
            follow: 0.8,
            depth_gain: 1.0,
            smoothing: 0.35,
            mirror: true,
            base_pitch: 0.0,
            brightness: 1.26,
            contrast: 1.06,
            saturation: 1.08,
            hue_rotate: 0.0,
            sharpen: 0.8,
            denoise: 0.5,
            auto_exposure: true,
            hide_when_lost: false,
            face_x: 0.0,
            face_y: 0.3,
            face_scale: 1.3,
            spacing: 0.8,
            flatten: 0.6,
            relief: 1.0,
            shading: 0.5,
            eye_x: 0.0,
            eye_y: 0.0,
            mouth_x: 0.0,
            mouth_y: 0.0,
            eye_size: 1.5,
            eye_bulge: 0.0,
            eye_round: 0.0,
            mouth_size: 1.25,
            eye_margin: 1.20,
            mouth_margin: 1.16,
            eye_dilation: 1.15,
            mouth_dilation: 1.06,
            eye_softness: 0.42,
            mouth_softness: 0.32,
        }
    }
}

/// How a setting's value reads in the panel. Stored values stay in their own units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Unit {
    /// 0.8 reads "80 %".
    Percent,
    /// Extra on top of 1: 1.15 reads "+15 %".
    Extra,
    /// 1.0 reads "1,00×".
    Times,
    Degrees,
    /// Stored in radians, shown in degrees.
    Radians,
}

/// One control in the settings panel: JSON key, label, one-sentence help.
pub enum Setting {
    Slider { key: &'static str, label: &'static str, help: &'static str, min: f32, max: f32, unit: Unit },
    Toggle { key: &'static str, label: &'static str, help: &'static str },
}

const fn slider(key: &'static str, label: &'static str, help: &'static str, min: f32, max: f32, unit: Unit) -> Setting {
    Setting::Slider { key, label, help, min, max, unit }
}
const fn toggle(key: &'static str, label: &'static str, help: &'static str) -> Setting {
    Setting::Toggle { key, label, help }
}

/// The settings panel: tabs by what you want to change, then sections, then controls.
/// The expert cut-out knobs live on their own tab so the everyday ones stay few.
pub const PANEL: &[(&str, &[(&str, &[Setting])])] = {
    use Unit::*;
    &[
        ("Bewegung", &[
            ("Platz im Bild", &[
                slider("modelScale", "Größe", "Wie groß die Figur im Fenster ist.", 0.3, 2.5, Percent),
                slider("modelY", "Höhe", "Schiebt die Figur nach oben oder unten.", -1.0, 1.0, Percent),
            ]),
            ("Mitgehen", &[
                slider("turnGain", "Drehung", "Wie stark die Figur deine Kopfdrehung mitmacht. 1× ist genau so stark wie du.", 0.0, 2.5, Times),
                slider("follow", "Seitlich mitgehen", "Wie weit die Figur mitwandert, wenn du dich im Kamerabild bewegst.", 0.0, 2.0, Percent),
                slider("depthGain", "Vor und zurück", "Wie stark die Figur näher kommt, wenn du dich zur Kamera lehnst.", 0.0, 1.5, Percent),
                slider("smoothing", "Ruhe", "Glättet Zittern. Mehr Ruhe macht die Figur etwas träger.", 0.0, 0.9, Percent),
                slider("basePitch", "Grundneigung", "Neigt die Ruhehaltung. Negativ hebt das Kinn.", -0.6, 0.6, Radians),
                toggle("mirror", "Spiegeln", "Seitenverkehrt wie ein Spiegel. Für die meisten Streams an lassen."),
                toggle("hideWhenLost", "Ohne Gesicht ausblenden", "Blendet die Figur aus, solange die Kamera dich nicht sieht. Schaust du nur nach unten, bleibt sie sichtbar."),
            ]),
        ]),
        ("Gesicht", &[
            ("Lage auf der Figur", &[
                slider("faceX", "Links / rechts", "Wo das Gesicht auf der Figur sitzt.", -1.0, 1.0, Percent),
                slider("faceY", "Oben / unten", "Wo das Gesicht auf der Figur sitzt.", -1.0, 1.0, Percent),
                slider("faceScale", "Größe", "Größe des ganzen Gesichts. Augen und Mund behalten ihre Proportionen.", 0.2, 2.5, Percent),
                slider("spacing", "Abstand", "Rückt Augen und Mund zusammen oder auseinander. Cartoon-Gesichter sitzen enger.", 0.4, 1.5, Percent),
                slider("flatten", "Flach anliegen", "Wie flach das Gesicht auf der Oberfläche liegt. Hilft bei schmalen, runden Figuren.", 0.0, 1.0, Percent),
                slider("relief", "Relief", "Wie plastisch Augen und Mund sind: Lider, Augapfel, Lippen, der offene Mund. 0 ist flach.", 0.0, 3.0, Percent),
                slider("shading", "Beleuchtung", "Wie stark das Licht der Szene Augen und Mund schattiert und glänzen lässt. 0 zeigt sie genau wie die Kamera.", 0.0, 1.0, Percent),
            ]),
            ("Augen", &[
                slider("eyeSize", "Größe", "Größe der Augen zusätzlich zu deinen echten Proportionen.", 0.3, 2.5, Percent),
                slider("eyeRound", "Runder", "Macht die Augen höher und runder, wie im Cartoon.", 0.0, 1.0, Percent),
                slider("eyeBulge", "Große Iris", "Vergrößert Iris und Pupille in der Augenmitte.", 0.0, 1.0, Percent),
                slider("eyeX", "Abstand", "Schiebt beide Augen auseinander oder zusammen.", -0.5, 0.5, Percent),
                slider("eyeY", "Höhe", "Schiebt beide Augen nach oben oder unten.", -0.5, 0.5, Percent),
            ]),
            ("Mund", &[
                slider("mouthSize", "Größe", "Größe des Mundes zusätzlich zu deinen echten Proportionen.", 0.3, 2.5, Percent),
                slider("mouthX", "Links / rechts", "Schiebt den Mund zur Seite.", -0.5, 0.5, Percent),
                slider("mouthY", "Höhe", "Schiebt den Mund nach oben oder unten.", -0.5, 0.5, Percent),
            ]),
        ]),
        ("Bild", &[
            ("Farbe und Licht", &[
                slider("brightness", "Helligkeit", "Helligkeit von Augen und Mund.", 0.5, 2.5, Percent),
                slider("contrast", "Kontrast", "Kontrast von Augen und Mund.", 0.5, 2.0, Percent),
                slider("saturation", "Farbsättigung", "0 % ist schwarzweiß.", 0.0, 2.0, Percent),
                slider("hueRotate", "Farbton", "Verschiebt die Farben. Um −100° werden Augen und Lippen lila, Weiß bleibt weiß.", -180.0, 180.0, Degrees),
                slider("denoise", "Entrauschen", "Glättet das Rauschen der Webcam in Augen und Mund, vor allem in dunklen Räumen. Sehr viel lässt Bewegungen kurz nachziehen.", 0.0, 1.0, Percent),
                slider("sharpen", "Schärfe", "Schärft Augen und Mund nach. Zu viel macht sie körnig.", 0.0, 2.0, Percent),
                toggle("autoExposure", "Automatisch aufhellen", "Hellt Augen und Mund in dunklen Räumen anhand deiner Haut auf."),
            ]),
        ]),
        ("Feinschnitt", &[
            ("Augen-Ausschnitt", &[
                slider("eyeDilation", "Rand", "Wie viel Haut um das Auge mit ausgeschnitten wird.", 1.0, 2.2, Extra),
                slider("eyeMargin", "Spielraum", "Luft zwischen Ausschnitt und Bildkante. Mehr verhindert abgeschnittene Ränder.", 1.0, 2.0, Extra),
                slider("eyeSoftness", "Weiche Kante", "Wie weich der Ausschnitt in die Figur übergeht.", 0.0, 0.95, Percent),
            ]),
            ("Mund-Ausschnitt", &[
                slider("mouthDilation", "Rand", "Wie viel Haut um die Lippen mit ausgeschnitten wird.", 1.0, 2.2, Extra),
                slider("mouthMargin", "Spielraum", "Luft zwischen Ausschnitt und Bildkante. Mehr verhindert abgeschnittene Ränder.", 1.0, 2.0, Extra),
                slider("mouthSoftness", "Weiche Kante", "Wie weich der Ausschnitt in die Figur übergeht.", 0.0, 0.95, Percent),
            ]),
        ]),
    ]
};

impl Calibration {
    /// Field access by JSON name, so the panel can be driven by the PANEL table.
    pub fn field(&mut self, name: &str) -> &mut f32 {
        match name {
            "modelScale" => &mut self.model_scale,
            "modelY" => &mut self.model_y,
            "turnGain" => &mut self.turn_gain,
            "follow" => &mut self.follow,
            "depthGain" => &mut self.depth_gain,
            "faceX" => &mut self.face_x,
            "faceY" => &mut self.face_y,
            "faceScale" => &mut self.face_scale,
            "spacing" => &mut self.spacing,
            "flatten" => &mut self.flatten,
            "relief" => &mut self.relief,
            "shading" => &mut self.shading,
            "smoothing" => &mut self.smoothing,
            "basePitch" => &mut self.base_pitch,
            "brightness" => &mut self.brightness,
            "contrast" => &mut self.contrast,
            "saturation" => &mut self.saturation,
            "hueRotate" => &mut self.hue_rotate,
            "sharpen" => &mut self.sharpen,
            "denoise" => &mut self.denoise,
            "eyeX" => &mut self.eye_x,
            "eyeY" => &mut self.eye_y,
            "mouthX" => &mut self.mouth_x,
            "mouthY" => &mut self.mouth_y,
            "eyeSize" => &mut self.eye_size,
            "eyeDilation" => &mut self.eye_dilation,
            "eyeMargin" => &mut self.eye_margin,
            "eyeSoftness" => &mut self.eye_softness,
            "eyeBulge" => &mut self.eye_bulge,
            "eyeRound" => &mut self.eye_round,
            "mouthSize" => &mut self.mouth_size,
            "mouthDilation" => &mut self.mouth_dilation,
            "mouthMargin" => &mut self.mouth_margin,
            "mouthSoftness" => &mut self.mouth_softness,
            _ => panic!("unknown calibration field {name}"),
        }
    }

    /// Switch access by JSON name, for the panel's toggles.
    pub fn flag(&mut self, name: &str) -> &mut bool {
        match name {
            "mirror" => &mut self.mirror,
            "autoExposure" => &mut self.auto_exposure,
            "hideWhenLost" => &mut self.hide_when_lost,
            _ => panic!("unknown calibration switch {name}"),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(default)]
pub struct Obs {
    pub url: String,
    pub password: String,
}

impl Default for Obs {
    fn default() -> Self {
        Self { url: "ws://127.0.0.1:4455".into(), password: String::new() }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum Action {
    /// Swap the avatar model, e.g. a second model as an "expression".
    Model(String),
    ToggleMirror,
    TogglePanel,
    /// Re-measure the neutral head pose: look straight at the camera for a moment.
    Recalibrate,
    ObsScene(String),
    /// Toggle a source's visibility in a scene.
    ObsToggleSource { scene: String, source: String },
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Hotkey {
    /// e.g. "Ctrl+Alt+1", "Ctrl+Shift+F9".
    pub keys: String,
    pub action: Action,
}

/// Bump when Rest's measurements change meaning: an old rest pose is dropped and re-measured.
pub const REST_VERSION: u32 = 1;

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct Rest {
    #[serde(default)]
    pub version: u32,
    /// Head rotation, quaternion xyzw.
    pub rotation: [f32; 4],
    /// Neck pivot, in frame heights from the frame centre (y down).
    pub center: [f32; 2],
    /// Head scale (frame heights per canonical unit), turn-invariant.
    pub width: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct Config {
    pub calibration: BTreeMap<String, Calibration>,
    /// Rest pose measured while looking straight at the camera. Depends on where the webcam
    /// is mounted and where you sit, not on the model, so it is shared by all models.
    pub rest: Option<Rest>,
    pub obs: Obs,
    pub hotkeys: Vec<Hotkey>,
    /// The figure shown at start: the last one picked in the panel.
    pub model: Option<String>,
    /// Imported figures (glTF files), for the panel's list.
    pub models: Vec<String>,
    /// The webcam picked in the panel, by name: indices change when cameras are plugged in.
    pub camera: Option<String>,
    /// Serve the avatar as an OBS browser source (panel switch, or `--serve`).
    pub serve: bool,
    /// Why the file on disk could not be used, for the panel to show. Not saved.
    #[serde(skip)]
    pub problem: Option<String>,
    /// The unusable file could not be moved aside: saving would destroy it, so nothing is.
    #[serde(skip)]
    read_only: bool,
}

impl Config {
    pub fn path() -> PathBuf {
        dirs::config_dir().unwrap_or_else(|| ".".into()).join("avatar-rs").join("config.json")
    }

    pub fn load() -> Self {
        Self::load_from(&Self::path())
    }

    /// A file that does not parse (a typo in hand-written hotkeys, a half-written file) is
    /// never overwritten: it may hold the hotkeys and the OBS password. It is moved aside
    /// and the app starts from defaults, saying so in the panel.
    fn load_from(path: &std::path::Path) -> Self {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Self::default(),
            Err(e) => return Self::unusable(format!("{} ist nicht lesbar ({e}). Änderungen werden nicht gespeichert.", path.display())),
        };
        let mut cfg: Self = match serde_json::from_str(&text) {
            Ok(cfg) => cfg,
            Err(e) => {
                let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
                let backup = path.with_extension(format!("json.{secs}.bak"));
                return match std::fs::rename(path, &backup) {
                    Ok(()) => Self {
                        problem: Some(format!("Die Einstellungsdatei war fehlerhaft ({e}). Sie liegt gesichert unter {}, bis dahin gelten Standardwerte.", backup.display())),
                        ..Self::default()
                    },
                    Err(_) => Self::unusable(format!("Die Einstellungsdatei {} ist fehlerhaft ({e}). Änderungen werden nicht gespeichert.", path.display())),
                };
            }
        };
        cfg.calibration.retain(|_, c| c.version == VERSION);
        cfg.rest = cfg.rest.filter(|r| r.version == REST_VERSION);
        cfg
    }

    fn unusable(problem: String) -> Self {
        eprintln!("{problem}");
        Self { problem: Some(problem), read_only: true, ..Self::default() }
    }

    pub fn save(&self) {
        if !self.read_only {
            self.save_to(&Self::path());
        }
    }

    /// Written next to the file, then renamed over it: a crash mid-write leaves the old file
    /// whole instead of half a file.
    fn save_to(&self, path: &std::path::Path) {
        let tmp = path.with_extension("json.tmp");
        let write = std::fs::create_dir_all(path.parent().unwrap())
            .and_then(|_| std::fs::write(&tmp, serde_json::to_string_pretty(self).unwrap()))
            .and_then(|_| std::fs::rename(&tmp, path));
        if let Err(e) = write {
            eprintln!("could not save {}: {e}", path.display());
        }
    }

    pub fn calibration(&mut self, model: &str) -> &mut Calibration {
        self.calibration.entry(model.to_string()).or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broken_file_is_kept_and_saving_is_atomic() {
        let dir = std::env::temp_dir().join(format!("avatar-config-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        let broken = r#"{ "obs": { "password": "secret" }, "hotkeys": [ oops ] }"#;
        std::fs::write(&path, broken).unwrap();

        let cfg = Config::load_from(&path);
        assert!(cfg.problem.is_some(), "the panel is told");
        let files: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).collect();
        let backup = files.iter().find(|p| p.to_string_lossy().ends_with(".bak")).expect("moved aside");
        assert_eq!(std::fs::read_to_string(backup).unwrap(), broken, "kept byte for byte");
        assert!(!path.exists());

        // Saving now writes a fresh, valid file and leaves the backup and no temp file behind.
        cfg.save_to(&path);
        assert!(Config::load_from(&path).problem.is_none());
        assert_eq!(std::fs::read_to_string(backup).unwrap(), broken);
        assert!(!dir.join("config.json.tmp").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn every_panel_field_exists_and_old_files_merge() {
        let mut c = Calibration::default();
        for (_, sections) in PANEL {
            for (_, settings) in *sections {
                for s in *settings {
                    match s {
                        Setting::Slider { key, .. } => _ = c.field(key),
                        Setting::Toggle { key, .. } => _ = c.flag(key),
                    }
                }
            }
        }
        // A file missing newer keys keeps the defaults for them instead of failing.
        let parsed: Calibration = serde_json::from_str(r#"{"version":7,"modelY":0.5}"#).unwrap();
        assert_eq!(parsed.model_y, 0.5);
        assert_eq!(parsed.eye_size, Calibration::default().eye_size);
    }
}
