mod hotkeys;
mod mouse;
mod obs;
mod avatar;
mod canonical_mesh;
mod compose;
mod flow;
mod import;
mod jpeg;
mod config;
mod tracker;
mod stream;
#[cfg(test)]
mod testface;

use bevy::prelude::*;
use bevy::window::{CompositeAlphaMode, WindowResolution};

const USAGE: &str = "\
avatar-rs [options]          avatar window (capture it in OBS)
avatar-rs mouse              mouse overlay (needs the input-overlay OBS plugin)

  --edit           open the calibration panel (or press C in the window)
  --debug          no webcam: synthetic head motion, to check the composition
  --mesh           camera image with the tracked face mesh next to the avatar
  --serve          OBS browser source with transparency at http://localhost:4460/
  --model FILE     your figure: .glb .gltf .blend .fbx .obj .stl .ply .usd* .abc
                   (all but glTF need Blender), or sphere
  --camera N       webcam index (default 0)
  --size WxH       window size (default 512x512)
";

#[derive(Clone, Debug)]
pub struct Args {
    pub edit: bool,
    pub debug: bool,
    pub mesh: bool,
    pub serve: bool,
    /// The figure to load (a glTF file or "sphere"); None = the last one from the panel.
    pub model: Option<String>,
    pub camera: u32,
    pub width: u32,
    pub height: u32,
}

fn parse_args() -> Result<(bool, Args), String> {
    let mut args = Args { edit: false, debug: false, mesh: false, serve: false, model: None, camera: 0, width: 512, height: 512 };
    let mut mouse = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut value = || it.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "mouse" => mouse = true,
            "--edit" => args.edit = true,
            "--debug" => args.debug = true,
            "--mesh" => args.mesh = true,
            "--serve" => args.serve = true,
            "--model" => {
                let v = value()?;
                args.model = Some(match v.as_str() {
                    avatar::SPHERE => v,
                    file => import::import(file.as_ref())?.to_string_lossy().into_owned(),
                });
            }
            "--camera" => args.camera = value()?.parse().map_err(|_| "--camera needs a number")?,
            "--size" => {
                let v = value()?;
                let (w, h) = v.split_once('x').ok_or("--size is WxH")?;
                (args.width, args.height) = (w.parse().map_err(|_| "bad width")?, h.parse().map_err(|_| "bad height")?);
            }
            "-h" | "--help" => return Err(String::new()),
            _ => return Err(format!("unknown argument {a}")),
        }
    }
    Ok((mouse, args))
}

/// Transparent, fixed-pixel window: what you capture in OBS (Window/Screen Capture).
fn window(title: &str, w: u32, h: u32) -> WindowPlugin {
    WindowPlugin {
        primary_window: Some(Window {
            title: title.into(),
            // Wayland app_id / X11 WM_CLASS: the launcher's icon for the window (avatar-rs.desktop).
            name: Some("avatar-rs".into()),
            resolution: WindowResolution::new(w, h).with_scale_factor_override(1.0),
            transparent: true,
            composite_alpha_mode: if cfg!(target_os = "macos") { CompositeAlphaMode::PostMultiplied } else { CompositeAlphaMode::PreMultiplied },
            ..default()
        }),
        ..default()
    }
}

fn main() {
    let (mouse, args) = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("{e}\n");
            }
            eprint!("{USAGE}");
            std::process::exit(2);
        }
    };

    let mut app = App::new();
    app.insert_resource(ClearColor(Color::NONE));
    // 60 fps is plenty: the camera delivers 30, OBS records 30/60. Uncapped, Bevy renders at the
    // monitor's rate (120+ Hz) for nothing. Mouse moves over the window still wake it early.
    let tick = bevy::winit::UpdateMode::reactive_low_power(std::time::Duration::from_secs_f64(1.0 / 60.0));
    app.insert_resource(bevy::winit::WinitSettings { focused_mode: tick, unfocused_mode: tick });
    if mouse {
        app.add_plugins(DefaultPlugins.set(window("avatar-rs mouse", 285, 421)).disable::<bevy::audio::AudioPlugin>());
        mouse::plugin(&mut app);
    } else {
        let assets = AssetPlugin { unapproved_path_mode: bevy::asset::UnapprovedPathMode::Allow, ..default() };
        app.add_plugins(DefaultPlugins.set(window("avatar-rs", args.width + if args.mesh { avatar::MESH_VIEW_W } else { 0 }, args.height)).set(assets).disable::<bevy::audio::AudioPlugin>())
            .add_plugins(bevy_egui::EguiPlugin::default());
        // The avatar picks the UI's camera itself (see avatar::setup).
        app.insert_resource(bevy_egui::EguiGlobalSettings { auto_create_primary_context: false, ..default() });
        avatar::plugin(&mut app);
        let config = config::Config::load();
        // Global hotkeys must be set up on the main thread before the event loop starts.
        app.insert_resource(avatar::Actions(std::sync::Mutex::new(hotkeys::spawn(config.hotkeys.clone()))));
        app.insert_resource(avatar::ObsOut(obs::spawn(config.obs.clone())));
        let state = avatar::new_state(args, config);
        app.insert_resource(state);
    }
    if let Ok(path) = std::env::var("AVATAR_SCREENSHOT") {
        app.add_systems(Update, move |mut commands: Commands, mut frame: Local<u32>, mut exit: MessageWriter<AppExit>| {
            *frame += 1;
            if *frame == std::env::var("AVATAR_SCREENSHOT_FRAME").ok().and_then(|f| f.parse().ok()).unwrap_or(90) {
                use bevy::render::view::screenshot::{Screenshot, save_to_disk};
                commands.spawn(Screenshot::primary_window()).observe(save_to_disk(path.clone()));
            }
            if *frame == std::env::var("AVATAR_SCREENSHOT_FRAME").ok().and_then(|f| f.parse().ok()).unwrap_or(90) + 10 {
                exit.write(AppExit::Success);
            }
        });
    }
    app.run();
}
