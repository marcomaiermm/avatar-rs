//! Bring your own figure. glTF loads as it is; every other format Blender converts to a GLB
//! once, headless, into <data dir>/avatar-rs/models. Blender is only needed for those.

use std::path::{Path, PathBuf};
use std::process::Command;

pub const FORMATS: &[&str] = &["glb", "gltf", "blend", "fbx", "obj", "stl", "ply", "usd", "usda", "usdc", "usdz", "abc"];

/// Runs inside Blender: `-- <source> <target.glb>`. A .blend is already open; anything else
/// is imported into an empty scene (no default cube).
const TO_GLB: &str = r#"
import bpy, sys
src, out = sys.argv[sys.argv.index("--") + 1:]
ext = src.rsplit(".", 1)[-1].lower()
if ext != "blend":
    bpy.ops.wm.read_factory_settings(use_empty=True)
    {
        "fbx": bpy.ops.import_scene.fbx,
        "obj": bpy.ops.wm.obj_import,
        "stl": bpy.ops.wm.stl_import,
        "ply": bpy.ops.wm.ply_import,
        "abc": bpy.ops.wm.alembic_import,
    }.get(ext, bpy.ops.wm.usd_import)(filepath=src)
bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", export_apply=True)
"#;

pub fn models_dir() -> PathBuf {
    dirs::data_dir().unwrap_or_else(|| ".".into()).join("avatar-rs").join("models")
}

/// The file to load: a .gltf stays where it is (its buffers and textures sit next to it),
/// everything else ends up as a GLB in the models directory.
pub fn import(src: &Path) -> Result<PathBuf, String> {
    import_into(src, &models_dir())
}

fn import_into(src: &Path, dir: &Path) -> Result<PathBuf, String> {
    let src = std::fs::canonicalize(src).map_err(|e| format!("{}: {e}", src.display()))?;
    let ext = src.extension().and_then(|e| e.to_str()).unwrap_or_default().to_lowercase();
    if !FORMATS.contains(&ext.as_str()) {
        return Err(format!("„.{ext}“ geht nicht. Möglich: {}", FORMATS.join(", ")));
    }
    if ext == "gltf" || src.parent() == Some(dir) {
        return Ok(src);
    }
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let name = src.file_stem().unwrap().to_string_lossy();
    let out = dir.join(format!("{name}.glb"));
    if ext == "glb" {
        std::fs::copy(&src, &out).map_err(|e| format!("{}: {e}", out.display()))?;
        return Ok(out);
    }
    // Into a side file first: a failed re-import keeps the model that worked.
    let tmp = dir.join(format!("{name}.import.glb"));
    let blender = std::env::var_os("AVATAR_BLENDER").unwrap_or_else(|| "blender".into());
    let run = Command::new(&blender)
        .args(["-b", "--factory-startup"])
        .args((ext == "blend").then_some(&src))
        .args(["--python-exit-code", "1", "--python-expr", TO_GLB, "--"])
        .arg(&src)
        .arg(&tmp)
        .output()
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => format!("Für .{ext} wird Blender gebraucht (Arch: pacman -S blender)."),
            _ => format!("Blender startet nicht: {e}"),
        })?;
    if !run.status.success() || !tmp.exists() {
        let log = String::from_utf8_lossy(&run.stdout) + String::from_utf8_lossy(&run.stderr);
        let tail: Vec<&str> = log.lines().filter(|l| !l.trim().is_empty()).collect();
        return Err(format!("Blender konnte {} nicht umwandeln:\n{}", src.display(), tail[tail.len().saturating_sub(4)..].join("\n")));
    }
    std::fs::rename(&tmp, &out).map_err(|e| format!("{}: {e}", out.display()))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obj_becomes_a_glb_and_gltf_stays_put() {
        let dir = std::env::temp_dir().join(format!("avatar-import-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let gltf = dir.join("a.gltf");
        std::fs::write(&gltf, "{}").unwrap();
        assert_eq!(import_into(&gltf, &dir.join("models")).unwrap(), gltf.canonicalize().unwrap());
        std::fs::write(dir.join("a.xyz"), "").unwrap();
        assert!(import_into(&dir.join("a.xyz"), &dir).is_err());
        if Command::new("blender").arg("--version").output().is_err() {
            eprintln!("no blender, conversion not checked");
            return;
        }
        let obj = dir.join("tri.obj");
        std::fs::write(&obj, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").unwrap();
        let out = import_into(&obj, &dir.join("models")).unwrap();
        assert!(std::fs::read(&out).unwrap().starts_with(b"glTF"), "{}", out.display());
        std::fs::write(dir.join("bad.fbx"), "not an fbx").unwrap();
        assert!(import_into(&dir.join("bad.fbx"), &dir.join("models")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
