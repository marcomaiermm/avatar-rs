//! MediaPipe's face models are not in the repo. The build fetches Google's
//! face_landmarker.task (Apache 2.0), checks its hash and unpacks the two models the tracker
//! bakes in. Offline: AVATAR_FACE_LANDMARKER=/path/to/face_landmarker.task.

use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::process::Command;

const URL: &str = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const SHA256: &str = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff";
const MODELS: [&str; 2] = ["face_detector.tflite", "face_landmarks_detector.tflite"];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=AVATAR_FACE_LANDMARKER");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    if MODELS.iter().all(|m| out.join(m).exists()) {
        return;
    }
    let task = match std::env::var_os("AVATAR_FACE_LANDMARKER") {
        Some(path) => PathBuf::from(path),
        None => {
            let path = out.join("face_landmarker.task");
            let ok = Command::new("curl").args(["-sSfL", "--retry", "3", "-o"]).arg(&path).arg(URL).status().is_ok_and(|s| s.success());
            assert!(ok, "could not download {URL} with curl; set AVATAR_FACE_LANDMARKER to a downloaded copy");
            path
        }
    };
    let data = std::fs::read(&task).unwrap_or_else(|e| panic!("{}: {e}", task.display()));
    let hash = format!("{:x}", Sha256::digest(&data));
    assert_eq!(hash, SHA256, "{} is not the expected face_landmarker.task", task.display());
    // The .task is a zip with stored (uncompressed) entries: walk the local file headers.
    let u16_at = |i: usize| u16::from_le_bytes([data[i], data[i + 1]]) as usize;
    let u32_at = |i: usize| u32::from_le_bytes(data[i..i + 4].try_into().unwrap()) as usize;
    // Google pads before and between the entries (alignment): look for the next header.
    let next = |from: usize| data[from..].windows(4).position(|w| w == b"PK\x03\x04").map(|i| from + i);
    let mut at = next(0);
    while let Some(here) = at.filter(|&i| data.len() >= i + 30) {
        let (method, size, name_len, extra) = (u16_at(here + 8), u32_at(here + 18), u16_at(here + 26), u16_at(here + 28));
        let name = String::from_utf8_lossy(&data[here + 30..here + 30 + name_len]);
        let start = here + 30 + name_len + extra;
        if MODELS.contains(&name.as_ref()) {
            assert_eq!(method, 0, "{name} is compressed");
            std::fs::write(out.join(name.as_ref()), &data[start..start + size]).unwrap();
        }
        at = next(start + size);
    }
    for m in MODELS {
        assert!(out.join(m).exists(), "{m} missing from face_landmarker.task");
    }
}
