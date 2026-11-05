//! Global hotkeys that fire while another app has focus.
//! Linux (Wayland has no global-shortcut API for plain clients): read the keyboards' evdev nodes
//! directly, which needs membership in the `input` group. Elsewhere: the global-hotkey crate.

use crate::config::{Action, Hotkey};
use std::sync::mpsc;

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use evdev::{EventType, KeyCode};
    use std::sync::Arc;

    const CTRL: u8 = 1;
    const SHIFT: u8 = 2;
    const ALT: u8 = 4;
    const SUPER: u8 = 8;
    const MODIFIER_KEYS: [(KeyCode, u8); 8] = [
        (KeyCode::KEY_LEFTCTRL, CTRL),
        (KeyCode::KEY_RIGHTCTRL, CTRL),
        (KeyCode::KEY_LEFTSHIFT, SHIFT),
        (KeyCode::KEY_RIGHTSHIFT, SHIFT),
        (KeyCode::KEY_LEFTALT, ALT),
        (KeyCode::KEY_RIGHTALT, ALT),
        (KeyCode::KEY_LEFTMETA, SUPER),
        (KeyCode::KEY_RIGHTMETA, SUPER),
    ];

    /// "Ctrl+Alt+1" -> (modifier bits, key). Case-insensitive.
    pub fn parse(keys: &str) -> Option<(u8, KeyCode)> {
        let (mut mods, mut key) = (0, None);
        for part in keys.split('+').map(|p| p.trim().to_ascii_uppercase()) {
            let bit = match part.as_str() {
                "CTRL" | "CONTROL" => CTRL,
                "SHIFT" => SHIFT,
                "ALT" => ALT,
                "SUPER" | "META" | "WIN" => SUPER,
                _ if key.is_none() => {
                    key = Some(main_key(&part)?);
                    continue;
                }
                _ => return None, // two main keys
            };
            mods |= bit;
        }
        Some((mods, key?))
    }

    fn main_key(name: &str) -> Option<KeyCode> {
        let evdev_name = match name {
            "SPACE" | "ENTER" | "TAB" | "HOME" | "END" | "PAGEUP" | "PAGEDOWN" | "INSERT" | "DELETE" | "UP" | "DOWN" | "LEFT" | "RIGHT" => {
                format!("KEY_{name}")
            }
            "ESCAPE" => "KEY_ESC".into(),
            _ => {
                let is_f = name.strip_prefix('F').and_then(|n| n.parse::<u8>().ok()).is_some_and(|n| (1..=24).contains(&n));
                if name.len() == 1 && name.as_bytes()[0].is_ascii_alphanumeric() || is_f {
                    format!("KEY_{name}")
                } else if let Some(n) = name.strip_prefix("NUMPAD").filter(|n| n.len() == 1 && n.as_bytes()[0].is_ascii_digit()) {
                    format!("KEY_KP{n}")
                } else {
                    return None;
                }
            }
        };
        evdev_name.parse().ok()
    }

    pub fn spawn(hotkeys: Vec<Hotkey>) -> mpsc::Receiver<Action> {
        let (tx, rx) = mpsc::channel();
        let bound: Arc<Vec<_>> = Arc::new(
            hotkeys
                .into_iter()
                .filter_map(|h| match parse(&h.keys) {
                    Some((mods, key)) => Some((mods, key, h.action)),
                    None => {
                        eprintln!("hotkeys: cannot parse {:?}, skipping", h.keys);
                        None
                    }
                })
                .collect(),
        );
        if bound.is_empty() {
            return rx;
        }
        let mut opened = 0;
        for (path, mut dev) in evdev::enumerate() {
            if !dev.supported_keys().is_some_and(|k| k.contains(KeyCode::KEY_A)) {
                continue;
            }
            opened += 1;
            let (tx, bound) = (tx.clone(), bound.clone());
            std::thread::spawn(move || {
                let mut held = [false; 8]; // per device, parallel to MODIFIER_KEYS
                loop {
                    let events = match dev.fetch_events() {
                        Ok(e) => e,
                        Err(e) => return eprintln!("hotkeys: {}: {e}", path.display()),
                    };
                    for ev in events.filter(|e| e.event_type() == EventType::KEY && e.value() != 2) {
                        let key = KeyCode::new(ev.code());
                        if let Some(i) = MODIFIER_KEYS.iter().position(|(k, _)| *k == key) {
                            held[i] = ev.value() == 1;
                            continue;
                        }
                        if ev.value() != 1 {
                            continue;
                        }
                        let mods = MODIFIER_KEYS.iter().zip(held).filter(|(_, h)| *h).fold(0, |m, ((_, b), _)| m | b);
                        for (_, _, action) in bound.iter().filter(|(m, k, _)| *m == mods && *k == key) {
                            let _ = tx.send(action.clone());
                        }
                    }
                }
            });
        }
        if opened == 0 {
            eprintln!("hotkeys: no readable keyboard in /dev/input; add your user to the `input` group (then log in again)");
        }
        rx
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn parses_hotkey_strings() {
            assert_eq!(parse("Ctrl+Alt+1"), Some((CTRL | ALT, KeyCode::KEY_1)));
            assert_eq!(parse("ctrl + shift + f9"), Some((CTRL | SHIFT, KeyCode::KEY_F9)));
            assert_eq!(parse("Super+P"), Some((SUPER, KeyCode::KEY_P)));
            assert_eq!(parse("Win+Escape"), Some((SUPER, KeyCode::KEY_ESC)));
            assert_eq!(parse("Control+PageUp"), Some((CTRL, KeyCode::KEY_PAGEUP)));
            assert_eq!(parse("Alt+Numpad7"), Some((ALT, KeyCode::KEY_KP7)));
            assert_eq!(parse("F24"), Some((0, KeyCode::KEY_F24)));
            for bad in ["Ctrl+Shift", "", "Ctrl+F25", "Ctrl+A+B", "Ctrl+Banana", "Ctrl+F0"] {
                assert_eq!(parse(bad), None, "{bad}");
            }
        }
    }
}

#[cfg(target_os = "linux")]
pub use linux::spawn;

/// Must be called from main() before app.run(): the manager has to be created on the main thread,
/// and on Windows/macOS its events are delivered by the event loop winit/Bevy runs there.
// ponytail: untested, written without a non-Linux machine to try it on.
#[cfg(not(target_os = "linux"))]
pub fn spawn(hotkeys: Vec<Hotkey>) -> mpsc::Receiver<Action> {
    use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState, hotkey::HotKey};
    let (tx, rx) = mpsc::channel();
    let manager = match GlobalHotKeyManager::new() {
        Ok(m) => Box::leak(Box::new(m)), // the hotkeys die with the manager
        Err(e) => {
            eprintln!("hotkeys: {e}");
            return rx;
        }
    };
    let mut actions = std::collections::HashMap::new();
    for h in hotkeys {
        match h.keys.parse::<HotKey>().map_err(|e| e.to_string()).and_then(|k| manager.register(k).map(|_| k).map_err(|e| e.to_string())) {
            Ok(k) => drop(actions.insert(k.id(), h.action)),
            Err(e) => eprintln!("hotkeys: {:?}: {e}, skipping", h.keys),
        }
    }
    std::thread::spawn(move || {
        for ev in GlobalHotKeyEvent::receiver().iter().filter(|e| e.state == HotKeyState::Pressed) {
            if let Some(a) = actions.get(&ev.id) {
                let _ = tx.send(a.clone());
            }
        }
    });
    rx
}
