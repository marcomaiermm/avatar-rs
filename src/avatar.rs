//! The avatar window. Three layers, which is what makes the effect work:
//!   1. a 3D model gives the silhouette                    -> Bevy PBR
//!   2. face tracking turns and tilts that model           -> tracker.rs
//!   3. real eyes and mouth cut from the webcam            -> compose.rs, as textures on mesh
//!      patches that hug the model surface: part of the body, not stickers on top
//! Without 2 and 3 this is a PNGTuber. The tracking and the real pixels are the whole effect.

use crate::config::{self, Action, Calibration, Config, PANEL, Setting, Unit};
use crate::{compose, tracker, Args};
use bevy::asset::RenderAssetUsages;
use bevy::camera::primitives::MeshAabb;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::scene::SceneInstanceReady;
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

const GOLD: Color = Color::srgb(236.0 / 255.0, 180.0 / 255.0, 5.0 / 255.0);
/// The built-in stand-in until a figure is imported.
pub const SPHERE: &str = "sphere";

#[derive(Resource)]
pub struct Avatar {
    pub args: Args,
    pub config: Config,
    pub model: String,
    pub panel_open: bool,
    tracker: Option<tracker::Shared>,
    smoothed: Quat,
    /// Smoothed placement from head position/distance: (x, y) offset in world units, scale.
    placed: Vec3,
    /// Tracker result the feature patches were last built from (pointer identity).
    shown_frame: usize,
    /// Raw poses collected while measuring the rest pose; Some while calibrating.
    rest_samples: Option<Vec<(f32, Quat, Vec2, f32)>>,
    /// When the tracker last had a face (seconds): short dropouts keep the last picture,
    /// long ones ease the model back to neutral instead of freezing mid-turn.
    face_seen: f32,
    /// The face was last seen touching the bottom of the camera image: gone from view by
    /// looking down (at the keyboard), not lost. The model keeps its last, lowered pose.
    left_at_bottom: bool,
    surface: Arc<Surface>,
    /// Eyes and mouth being cut and baked for the newest tracker frame, off the render loop.
    job: Option<bevy::tasks::Task<Option<[compose::Feature; 3]>>>,
    debug_face: Option<Arc<tracker::Frame>>,
}

impl Avatar {
    fn calibration(&mut self) -> &mut Calibration {
        let model = self.model.clone();
        self.config.calibration(&model)
    }
}

/// The model's front surface as a depth map in head-local space (orthographic, looking down
/// -z): where the eye and mouth patches are laid onto the real model instead of a sphere.
/// Rasterized once per model so per-frame lookups are O(1). Only the part under the face
/// counts (front-most mesh there): patches must not crawl onto a bow or hat next to it.
#[derive(Default)]
struct Surface {
    model: String,
    depth: Vec<f32>,
    /// Which mesh is front-most per cell; cells of other meshes are misses.
    part: Vec<u16>,
    /// The body's texture coordinates per cell, for its normal and roughness maps.
    uv: Vec<Vec2>,
    body: u16,
}

const DEPTH_CELLS: usize = 256;
/// Half extent of the depth map; models are normalized to fit -1..1.
const DEPTH_RANGE: f32 = 1.6;

impl Surface {
    fn rasterize(model: String, tris: &[(u16, [Vec3; 3], [Vec2; 3])], face: Vec2) -> Self {
        let n = DEPTH_CELLS;
        let cell = 2.0 * DEPTH_RANGE / n as f32;
        let mut depth = vec![f32::NEG_INFINITY; n * n];
        let mut part = vec![u16::MAX; n * n];
        let mut uv = vec![Vec2::ZERO; n * n];
        let to_cell = |v: f32| ((v + DEPTH_RANGE) / cell - 0.5).clamp(0.0, (n - 1) as f32);
        for (id, [a, b, c], [ua, ub, uc]) in tris {
            let (a2, b2, c2) = (a.truncate(), b.truncate(), c.truncate());
            let d = (b2 - a2).perp_dot(c2 - a2);
            if d.abs() < 1e-12 {
                continue;
            }
            let (lo, hi) = (a2.min(b2).min(c2), a2.max(b2).max(c2));
            for y in to_cell(lo.y).ceil() as usize..=to_cell(hi.y).floor() as usize {
                for x in to_cell(lo.x).ceil() as usize..=to_cell(hi.x).floor() as usize {
                    let q = Vec2::new((x as f32 + 0.5) * cell - DEPTH_RANGE, (y as f32 + 0.5) * cell - DEPTH_RANGE);
                    let (u, v) = ((b2 - q).perp_dot(c2 - q) / d, (c2 - q).perp_dot(a2 - q) / d);
                    if u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
                        let z = a.z * u + b.z * v + c.z * (1.0 - u - v);
                        if z > depth[y * n + x] {
                            depth[y * n + x] = z;
                            part[y * n + x] = *id;
                            uv[y * n + x] = *ua * u + *ub * v + *uc * (1.0 - u - v);
                        }
                    }
                }
            }
        }
        let at = |q: f32| to_cell(q).round() as usize;
        let body = part[at(face.y) * n + at(face.x)];
        Self { model, depth, part, uv, body }
    }

    /// Middle of the body's silhouette at height y: the face is centred on it, so a bent
    /// model (a banana-shaped pickle) gets its face on the front instead of on one flank.
    fn middle_x(&self, y: f32) -> f32 {
        let n = DEPTH_CELLS;
        if self.depth.is_empty() {
            return 0.0;
        }
        let row = (((y + DEPTH_RANGE) / (2.0 * DEPTH_RANGE) * n as f32) as usize).min(n - 1);
        let hits: Vec<usize> = (0..n).filter(|&x| self.part[row * n + x] == self.body).collect();
        match (hits.first(), hits.last()) {
            (Some(&a), Some(&b)) => ((a + b) as f32 * 0.5 + 0.5) / n as f32 * 2.0 * DEPTH_RANGE - DEPTH_RANGE,
            _ => 0.0,
        }
    }

    /// Front surface depth and texture coordinate at head-local (x, y), bilinear over the
    /// cells that hit the body.
    fn at(&self, q: Vec2) -> Option<(f32, Vec2)> {
        if self.depth.is_empty() {
            return None;
        }
        let n = DEPTH_CELLS;
        let p = (q + DEPTH_RANGE) / (2.0 * DEPTH_RANGE) * n as f32 - Vec2::splat(0.5);
        let (x0, y0) = (p.x.floor() as i32, p.y.floor() as i32);
        let (fx, fy) = (p.x - x0 as f32, p.y - y0 as f32);
        let (mut sum, mut uv, mut weight) = (0.0, Vec2::ZERO, 0.0);
        for (dx, dy, w) in [(0, 0, (1.0 - fx) * (1.0 - fy)), (1, 0, fx * (1.0 - fy)), (0, 1, (1.0 - fx) * fy), (1, 1, fx * fy)] {
            let (x, y) = (x0 + dx, y0 + dy);
            if x < 0 || y < 0 || x >= n as i32 || y >= n as i32 {
                continue;
            }
            let i = y as usize * n + x as usize;
            if self.part[i] == self.body && w > 0.0 {
                sum += self.depth[i] * w;
                uv += self.uv[i] * w;
                weight += w;
            }
        }
        (weight > 0.0).then(|| (sum / weight, uv / weight))
    }
}

/// One eye or the mouth: a textured grid that is re-laid on the model surface every frame.
struct Patch {
    /// The real pixels: emissive (already exposed by the camera, so not lit a second time), and
    /// through the black base colour only the alpha.
    image: Handle<Image>,
    mesh: Handle<Mesh>,
    material: Handle<StandardMaterial>,
    entity: Entity,
}

#[derive(Resource)]
struct Patches([Patch; 3]);

/// Grid resolution of a patch: the model's curvature and the feature's own relief.
const PATCH_CELLS: u32 = compose::RELIEF_CELLS as u32;
/// Lift above the surface (head-local units) so the patch never z-fights the skin under it.
const PATCH_LIFT: f32 = 0.012;

fn patch_mesh() -> Mesh {
    use bevy::mesh::{Indices, PrimitiveTopology};
    let n = PATCH_CELLS;
    let mut uv = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            uv.push([i as f32 / n as f32, j as f32 / n as f32]);
        }
    }
    let mut idx = Vec::new();
    for j in 0..n {
        for i in 0..n {
            let a = j * (n + 1) + i;
            idx.extend([a, a + n + 1, a + 1, a + 1, a + n + 1, a + n + 2]);
        }
    }
    let count = uv.len();
    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0f32; 3]; count])
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0f32, 0.0, 1.0]; count])
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, vec![[1.0f32; 4]; count])
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uv)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_1, vec![[0.0f32; 2]; count])
        .with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, vec![[1.0f32, 0.0, 0.0, 1.0]; count])
        .with_inserted_indices(Indices::U32(idx))
}

/// A patch laid on the model, per vertex: position, colour (alpha clips it to the body),
/// the body's normal and texture coordinate there (so it is lit and textured like the body
/// under it) and a tangent for the body's normal map. Plus the patch's overall normal.
struct Laid {
    pos: Vec<[f32; 3]>,
    col: Vec<[f32; 4]>,
    normal: Vec<[f32; 3]>,
    uv: Vec<[f32; 2]>,
    tangent: Vec<[f32; 4]>,
    facing: Vec3,
}

/// Patch vertices laid onto the surface: a grid over the feature's box on the face plane,
/// each vertex dropped onto the model's front depth, then pulled toward a flat card in front
/// of it by `flatten`. Vertex alpha clips the patch to the body's silhouette, so a big mouth
/// can't hang off the side.
fn patch_vertices(f: &compose::Feature, surface: &Surface, flatten: f32) -> Laid {
    let n = PATCH_CELLS as f32;
    let sphere = |q: Vec2| (0.94f32 * 0.94 - q.length_squared()).max(0.0).sqrt();
    let has_surface = !surface.depth.is_empty();
    let at = |q: Vec2| if has_surface { surface.at(q) } else { Some((sphere(q), q * 0.5 + Vec2::splat(0.5))) };
    let depth = |q: Vec2| at(q).map(|a| a.0);
    let eps = DEPTH_RANGE * 2.0 / DEPTH_CELLS as f32 * 2.0;
    let gradient = |q: Vec2| {
        let d = |a: Vec2| depth(q + a).zip(depth(q - a)).map_or(0.0, |(p, m)| (p - m) / (2.0 * eps));
        Vec2::new(d(Vec2::X * eps), d(Vec2::Y * eps))
    };
    let grid: Vec<(Vec2, Option<(f32, Vec2)>)> = (0..=PATCH_CELLS)
        .flat_map(|j| (0..=PATCH_CELLS).map(move |i| (i, j)))
        .map(|(i, j)| {
            let q = f.center + f.half * Vec2::new(i as f32 / n * 2.0 - 1.0, 1.0 - j as f32 / n * 2.0);
            (q, at(q))
        })
        .collect();
    // The card sits at the patch's front-most point, so flattening never sinks into the body.
    let front = grid.iter().filter_map(|g| g.1.map(|a| a.0)).fold(f32::NEG_INFINITY, f32::max);
    let fallback = if front.is_finite() { front } else { 0.0 };
    let mut laid = Laid { pos: Vec::new(), col: Vec::new(), normal: Vec::new(), uv: Vec::new(), tangent: Vec::new(), facing: Vec3::Z };
    // The feature's relief on top: raised so its lowest point (the open mouth) is at the
    // surface, never inside the body. Its slope tilts the normal, so the light shades it.
    let g = PATCH_CELLS as usize + 1;
    let relief = |k: usize| f.relief.get(k).copied().unwrap_or(0.0);
    let floor = (0..g * g).map(relief).fold(0.0f32, f32::min);
    let cell = f.half * 2.0 / n;
    let slope = |k: usize| {
        let (i, j) = (k % g, k / g);
        let d = |a: usize, b: usize, span: usize| (relief(b) - relief(a)) / span as f32;
        let (l, r) = (j * g + i.saturating_sub(1), j * g + (i + 1).min(g - 1));
        let (u, b) = (j.saturating_sub(1) * g + i, (j + 1).min(g - 1) * g + i);
        // Grid rows run top down, so +y is toward the previous row.
        Vec2::new(d(l, r, (i + 1).min(g - 1) - i.saturating_sub(1)) / cell.x, d(b, u, (j + 1).min(g - 1) - j.saturating_sub(1)) / cell.y)
    };
    for (k, (q, hit)) in grid.into_iter().enumerate() {
        let (z, alpha, normal, uv) = match hit {
            Some((z, uv)) => {
                let g = gradient(q);
                let normal = Vec3::new(-g.x, -g.y, 1.0).normalize();
                let inside = [Vec2::X, -Vec2::X, Vec2::Y, -Vec2::Y].iter().all(|d| depth(q + *d * eps).is_some());
                // Fade only where the body itself turns edge-on (past ~75 degrees).
                (z + (front - z) * flatten, if inside { ((normal.z - 0.2) / 0.15).clamp(0.0, 1.0) } else { 0.0 }, normal, uv)
            }
            None => (fallback, 0.0, Vec3::Z, Vec2::ZERO),
        };
        laid.pos.push([q.x, q.y, z + PATCH_LIFT + relief(k) - floor]);
        laid.col.push([1.0, 1.0, 1.0, alpha]);
        // The body's normal (not the flattened card's: lit like the body under it), tilted by
        // the feature's own slope.
        let s = slope(k);
        laid.normal.push((normal + Vec3::new(-s.x, -s.y, 0.0)).normalize().to_array());
        laid.uv.push(uv.to_array());
    }
    laid.tangent = tangents(&laid.pos, &laid.normal, &laid.uv);
    let g = gradient(f.center);
    laid.facing = Vec3::new(-g.x, -g.y, 1.0).normalize().lerp(Vec3::Z, flatten).normalize();
    laid
}

/// Per-vertex tangents of the patch grid along the body texture's u, for the body's normal
/// map. Mikktspace convention: w says whether cross(normal, tangent) runs along +v or -v.
fn tangents(pos: &[[f32; 3]], normal: &[[f32; 3]], uv: &[[f32; 2]]) -> Vec<[f32; 4]> {
    let m = PATCH_CELLS as usize + 1;
    (0..m * m)
        .map(|k| {
            let (i, j) = (k % m, k / m);
            let d = |a: usize, b: usize| (Vec3::from(pos[b]) - Vec3::from(pos[a]), Vec2::from(uv[b]) - Vec2::from(uv[a]));
            let (pi, ti) = d(j * m + i.saturating_sub(1), j * m + (i + 1).min(m - 1));
            let (pj, tj) = d(j.saturating_sub(1) * m + i, (j + 1).min(m - 1) * m + i);
            // Solve the position change per unit u and v from the two grid directions.
            let det = ti.x * tj.y - tj.x * ti.y;
            if det.abs() < 1e-12 {
                return [1.0, 0.0, 0.0, 1.0];
            }
            let du = (pi * tj.y - pj * ti.y) / det;
            let dv = (pj * ti.x - pi * tj.x) / det;
            let n = Vec3::from(normal[k]);
            let t = (du - n * n.dot(du)).normalize_or(Vec3::X);
            [t.x, t.y, t.z, if n.cross(t).dot(dv) < 0.0 { -1.0 } else { 1.0 }]
        })
        .collect()
}

/// Patch material: the body's own gloss and surface detail (its roughness and normal maps,
/// on the body's texture coordinates = UV_1 of the patch); colour from the baked textures.
fn patch_material(host: Option<&StandardMaterial>, image: &Handle<Image>) -> StandardMaterial {
    use bevy::pbr::UvChannel;
    let h = host.cloned().unwrap_or_default();
    StandardMaterial {
        // Black base x texture: no colour, only the texture's alpha (times the vertex alpha).
        // One texture instead of a black-with-alpha copy plus an opaque one: Bevy creates a new
        // GPU texture on every change, so each one saved is an allocation per camera frame.
        base_color: Color::BLACK,
        base_color_texture: Some(image.clone()),
        emissive: LinearRgba::WHITE,
        emissive_texture: Some(image.clone()),
        perceptual_roughness: h.perceptual_roughness,
        metallic: h.metallic,
        reflectance: h.reflectance,
        metallic_roughness_texture: h.metallic_roughness_texture,
        metallic_roughness_channel: UvChannel::Uv1,
        normal_map_texture: h.normal_map_texture,
        normal_map_channel: UvChannel::Uv1,
        flip_normal_map_y: h.flip_normal_map_y,
        occlusion_texture: h.occlusion_texture,
        occlusion_channel: UvChannel::Uv1,
        // No clearcoat: Bevy layers it over emission, which would dim the real pixels to ~4%.
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        ..default()
    }
}

/// How the scene light falls on a feature: `lit` of its colour is the base colour (shaded by
/// the light, with its relief's normals), the rest emissive (exactly the camera's pixels).
/// The two add up to the same brightness where the light falls square on. Eyes are wet and
/// lips moist: glossier than the body, so the key light leaves a highlight on them.
fn shade(m: &mut StandardMaterial, lit: f32, mouth: bool) {
    m.base_color = Color::linear_rgb(lit * LIT_GAIN, lit * LIT_GAIN, lit * LIT_GAIN);
    m.emissive = LinearRgba::rgb(1.0 - lit, 1.0 - lit, 1.0 - lit);
    m.perceptual_roughness = if mouth { 0.38 } else { 0.22 };
    m.reflectance = 0.5;
}

/// Base colour gain that makes a lit feature, facing the key light, as bright as the camera's
/// pixels shown unlit (measured on the test face; the scene's lights are dim).
const LIT_GAIN: f32 = 1.5;

/// Eyes and mouth for one tracker frame: cut from the camera and placed.
fn compose_patches(t: &tracker::Tracked, surface: &Surface, cfg: &Calibration) -> Option<[compose::Feature; 3]> {
    let mut features = compose::features(&t.frame, t.landmarks.as_ref()?, t.head, cfg)?;
    // The face moves as one rigid piece onto the body's middle line (at its own centre
    // height): shifting each feature separately bent the face along a curved model.
    let centre_y = features.iter().map(|f| f.center.y).sum::<f32>() / 3.0;
    let shift = surface.middle_x(centre_y);
    for f in &mut features {
        f.center.x += shift;
    }
    Some(features)
}

/// Actions arriving from the global hotkey thread.
#[derive(Resource)]
pub struct Actions(pub Mutex<Receiver<Action>>);

/// Requests to the OBS thread.
#[derive(Resource)]
pub struct ObsOut(pub std::sync::mpsc::Sender<Action>);

#[derive(Component)]
struct HeadPivot;

#[derive(Component)]
struct ModelRoot;

/// Marks a model root whose meshes are spawned and fitted, so its surface can be read.
#[derive(Component)]
struct Ready;

pub fn plugin(app: &mut App) {
    app.add_systems(Startup, setup)
        .add_systems(Update, (handle_actions, toggle_panel, crate::stream::switch, swap_model, build_surface, track).chain())
        .add_systems(EguiPrimaryContextPass, (mesh_view, panel).chain());
}

pub fn new_state(args: Args, mut config: Config) -> Avatar {
    config.serve |= args.serve;
    // Debug mode skips the webcam entirely: it only checks the model's composition and turn.
    let tracker = (!args.debug).then(|| tracker::spawn(args.camera, args.mesh));
    // A remembered figure whose file is gone falls back to the sphere instead of nothing.
    let model = args.model.clone().or(config.model.clone()).filter(|m| m == SPHERE || std::path::Path::new(m).exists()).unwrap_or_else(|| SPHERE.into());
    let args_debug = args.debug;
    Avatar {
        panel_open: args.edit,
        args,
        config,
        model,
        tracker,
        smoothed: Quat::IDENTITY,
        placed: Vec3::new(0.0, 0.0, 1.0),
        shown_frame: 0,
        rest_samples: None,
        face_seen: f32::NEG_INFINITY,
        left_at_bottom: false,
        surface: Arc::default(),
        job: None,
        debug_face: args_debug.then(|| Arc::new(fake_face())),
    }
}

fn setup(
    avatar: Res<Avatar>,
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // With the mesh view the avatar keeps its own square on the right.
    let viewport = avatar.args.mesh.then(|| bevy::camera::Viewport {
        physical_position: UVec2::new(MESH_VIEW_W, 0),
        physical_size: UVec2::new(avatar.args.width, avatar.args.height),
        ..default()
    });
    let camera = |camera: Camera| {
        (
            Camera3d::default(),
            camera,
            Projection::from(PerspectiveProjection { fov: compose::FOV_DEG.to_radians(), near: 0.1, far: 100.0, ..default() }),
            // Three.js-style output: no tone mapping, and exposure 1 so light intensities mean
            // the same as in the browser version.
            Tonemapping::None,
            bevy::camera::Exposure { ev100: -(1.2f32).log2() },
            Transform::from_xyz(0.0, 0.0, compose::CAMERA_Z),
        )
    };
    let window = commands.spawn(camera(Camera { viewport, ..default() })).id();
    // The panel's camera is chosen here, not left to bevy_egui (it would take whichever
    // camera came first). With the mesh view the UI gets its own camera over the whole
    // window; on the avatar's camera it would be squeezed into the avatar's square.
    if avatar.args.mesh {
        commands.spawn((Camera2d, Camera { order: 1, clear_color: ClearColorConfig::None, ..default() }, bevy_egui::PrimaryEguiContext));
    } else {
        commands.entity(window).insert(bevy_egui::PrimaryEguiContext);
    }
    // Same view into a texture for the OBS browser source: no panel, no mesh view. Idle
    // until switched on in the panel.
    let size = (avatar.args.width, avatar.args.height);
    let mut target = Image::new_target_texture(size.0, size.1, TextureFormat::Rgba8UnormSrgb);
    target.texture_descriptor.usage |= bevy::render::render_resource::TextureUsages::COPY_SRC;
    let target = images.add(target);
    let view = commands
        .spawn(camera(Camera { order: -1, is_active: false, target: bevy::camera::RenderTarget::Image(target.clone().into()), clear_color: ClearColorConfig::Custom(Color::NONE), ..default() }))
        .id();
    crate::stream::start(&mut commands, target, view, size);
    // Three's ambient is irradiance (divided by pi in the BRDF), Bevy's is radiance.
    commands.insert_resource(AmbientLight { color: Color::WHITE, brightness: 1.25 / std::f32::consts::PI, ..default() });
    commands.spawn((
        DirectionalLight { color: Color::srgb_u8(0xff, 0xf3, 0xd6), illuminance: 2.1, ..default() },
        Transform::from_xyz(2.0, 3.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight { color: GOLD, illuminance: 1.5, ..default() },
        Transform::from_xyz(-3.0, 1.0, -2.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    let pivot = commands.spawn((HeadPivot, Transform::default(), Visibility::default())).id();
    let patches = std::array::from_fn(|_| {
        let side = compose::TEX;
        let mut texture = || {
            images.add(Image::new_fill(
                Extent3d { width: side, height: side, depth_or_array_layers: 1 },
                TextureDimension::D2,
                &[0, 0, 0, 0],
                TextureFormat::Rgba8UnormSrgb,
                RenderAssetUsages::default(),
            ))
        };
        let image = texture();
        let material = materials.add(patch_material(None, &image));
        let mesh = meshes.add(patch_mesh());
        let entity = commands
            .spawn((
                Mesh3d(mesh.clone()),
                MeshMaterial3d(material.clone()),
                Transform::default(),
                Visibility::Hidden,
                // The patch moves every frame; its spawn-time bounds would cull it wrongly.
                bevy::camera::visibility::NoFrustumCulling,
                ChildOf(pivot),
            ))
            .id();
        Patch { image, mesh, material, entity }
    });
    commands.insert_resource(Patches(patches));
}

fn handle_actions(actions: Option<Res<Actions>>, obs: Option<Res<ObsOut>>, mut avatar: ResMut<Avatar>) {
    let Some(actions) = actions else { return };
    let rx = actions.0.lock().unwrap();
    while let Ok(action) = rx.try_recv() {
        match action {
            Action::Model(m) => avatar.model = m,
            Action::ToggleMirror => {
                let c = avatar.calibration();
                c.mirror = !c.mirror;
                avatar.config.save();
            }
            Action::TogglePanel => avatar.panel_open = !avatar.panel_open,
            Action::Recalibrate => avatar.rest_samples = Some(Vec::new()),
            obs_action => {
                if let Some(obs) = &obs {
                    let _ = obs.0.send(obs_action);
                }
            }
        }
    }
}

fn toggle_panel(keys: Res<ButtonInput<KeyCode>>, mut contexts: EguiContexts, mut avatar: ResMut<Avatar>) {
    let typing = contexts.ctx_mut().is_ok_and(|c| c.wants_keyboard_input());
    if keys.just_pressed(KeyCode::KeyC) && !typing {
        avatar.panel_open = !avatar.panel_open;
    }
}

/// (Re)spawns the model whenever `avatar.model` changes, including the first frame.
#[allow(clippy::too_many_arguments)]
fn swap_model(
    mut commands: Commands,
    avatar: Res<Avatar>,
    mut shown: Local<Option<String>>,
    pivot: Single<Entity, With<HeadPivot>>,
    old: Query<Entity, With<ModelRoot>>,
    assets: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if shown.as_deref() == Some(avatar.model.as_str()) {
        return;
    }
    *shown = Some(avatar.model.clone());
    for e in &old {
        commands.entity(e).despawn();
    }
    let root = commands.spawn((ModelRoot, Transform::default(), Visibility::default(), ChildOf(*pivot))).id();
    if avatar.model == SPHERE {
        let mesh = meshes.add(Sphere::new(1.0).mesh().uv(64, 48));
        let material = materials.add(StandardMaterial { base_color: Color::srgb(0.72, 0.76, 0.7), perceptual_roughness: 0.6, ..default() });
        commands.spawn((Mesh3d(mesh), MeshMaterial3d(material), ChildOf(root)));
        commands.entity(root).insert(Ready);
    } else {
        let scene = assets.load(GltfAssetLabel::Scene(0).from_asset(avatar.model.clone()));
        commands.entity(root).insert(SceneRoot(scene)).observe(fit_glb);
    }
}

/// Every mesh under `root` with its transform relative to `root`'s parent (the head pivot)
/// when `include_root`, else relative to `root` itself.
fn meshes_under(
    root: Entity,
    include_root: bool,
    children: &Query<&Children>,
    node: impl Fn(Entity) -> Option<(Mat4, Option<AssetId<Mesh>>)>,
) -> Vec<(Mat4, AssetId<Mesh>, Entity)> {
    let base = if include_root { node(root).map_or(Mat4::IDENTITY, |n| n.0) } else { Mat4::IDENTITY };
    let mut out = Vec::new();
    let mut stack: Vec<(Entity, Mat4)> = children.get(root).into_iter().flatten().map(|&c| (c, base)).collect();
    while let Some((e, parent)) = stack.pop() {
        let Some((local, mesh)) = node(e) else { continue };
        let m = parent * local;
        out.extend(mesh.map(|id| (m, id, e)));
        stack.extend(children.get(e).into_iter().flatten().map(|&c| (c, m)));
    }
    out
}

/// Normalize any GLB into the same framing as the procedural heads (largest side = 2,
/// centred), otherwise every downloaded model would show at its own size and offset.
fn fit_glb(
    ev: On<SceneInstanceReady>,
    mut commands: Commands,
    children: Query<&Children>,
    mut nodes: Query<(&mut Transform, Option<&Mesh3d>)>,
    meshes: Res<Assets<Mesh>>,
    parts: Query<&MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let (mut lo, mut hi) = (Vec3::MAX, Vec3::MIN);
    let found = meshes_under(ev.entity, false, &children, |e| nodes.get(e).ok().map(|(t, m)| (t.to_matrix(), m.map(|m| m.id()))));
    for (m, id, e) in found {
        // ponytail: Bevy 0.17 renders KHR_materials_anisotropy as white/black noise (Blender 5
        // exports it for satin, tangents or not). Off until a Bevy update renders it right.
        if let Ok(part) = parts.get(e)
            && materials.get(&part.0).is_some_and(|m| m.anisotropy_strength != 0.0)
        {
            materials.get_mut(&part.0).unwrap().anisotropy_strength = 0.0;
        }
        let Some(aabb) = meshes.get(id).and_then(|m| m.compute_aabb()) else { continue };
        for i in 0..8 {
            let s = Vec3::new([-1.0, 1.0][i & 1], [-1.0, 1.0][(i >> 1) & 1], [-1.0, 1.0][i >> 2]);
            let p = m.transform_point3(Vec3::from(aabb.center) + Vec3::from(aabb.half_extents) * s);
            lo = lo.min(p);
            hi = hi.max(p);
        }
    }
    if lo.x <= hi.x {
        let k = 2.0 / (hi - lo).max_element();
        if let Ok((mut t, _)) = nodes.get_mut(ev.entity) {
            *t = Transform::from_translation(-(lo + hi) * 0.5 * k).with_scale(Vec3::splat(k));
        }
    }
    commands.entity(ev.entity).insert(Ready);
}

#[allow(clippy::too_many_arguments)]
fn build_surface(
    mut avatar: ResMut<Avatar>,
    root: Query<Entity, (With<ModelRoot>, With<Ready>)>,
    children: Query<&Children>,
    nodes: Query<(&Transform, Option<&Mesh3d>)>,
    meshes: Res<Assets<Mesh>>,
    parts: Query<&MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    patches: Res<Patches>,
) {
    use bevy::mesh::VertexAttributeValues;
    let Ok(root) = root.single() else { return };
    if avatar.surface.model == avatar.model {
        return;
    }
    let mut tris = Vec::new();
    let found = meshes_under(root, true, &children, |e| nodes.get(e).ok().map(|(t, m)| (t.to_matrix(), m.map(|m| m.id()))));
    for (part, (m, id, _)) in found.iter().enumerate() {
        let Some(mesh) = meshes.get(*id) else { continue };
        let Some(VertexAttributeValues::Float32x3(pos)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) else { continue };
        let uv = match mesh.attribute(Mesh::ATTRIBUTE_UV_0) {
            Some(VertexAttributeValues::Float32x2(uv)) => uv.clone(),
            _ => vec![[0.0; 2]; pos.len()],
        };
        let index: Vec<usize> = match mesh.indices() {
            Some(ix) => ix.iter().collect(),
            None => (0..pos.len()).collect(),
        };
        let at = |i: usize| m.transform_point3(Vec3::from(pos[i]));
        let uv = |i: usize| Vec2::from(uv[i]);
        tris.extend(index.chunks_exact(3).map(|t| (part as u16, [at(t[0]), at(t[1]), at(t[2])], [uv(t[0]), uv(t[1]), uv(t[2])])));
    }
    let face = { let c = avatar.calibration(); Vec2::new(c.face_x, c.face_y) };
    let surface = Surface::rasterize(avatar.model.clone(), &tris, face);
    // The features take the gloss and surface detail of the body part under the face.
    let host = found.get(surface.body as usize).and_then(|p| parts.get(p.2).ok()).and_then(|m| materials.get(&m.0)).cloned();
    for p in &patches.0 {
        if let Some(m) = materials.get_mut(&p.material) {
            *m = patch_material(host.as_ref(), &p.image);
        }
    }
    avatar.surface = Arc::new(surface);
}

/// Without a face for this long the model eases back to its neutral pose.
const FACE_LOST: f32 = 1.0;
/// Without a face for this long the eyes and mouth disappear.
const HOLD_FEATURES: f32 = 0.3;

/// Head pose from the tracker, then the eye/mouth patches for the same frame.
#[allow(clippy::too_many_arguments)]
fn track(
    time: Res<Time>,
    mut avatar: ResMut<Avatar>,
    mut pivot: Single<&mut Transform, With<HeadPivot>>,
    head: Single<Entity, With<HeadPivot>>,
    patches: Res<Patches>,
    mut visibility: Query<&mut Visibility>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let cfg = avatar.calibration().clone();
    let now = time.elapsed_secs();
    let latest = match &avatar.debug_face {
        Some(face) => Some(Arc::new(tracker::Tracked { frame: face.clone(), landmarks: Some(fake_landmarks()), head: Some(fake_head()), how: "test face", presence: 1.0, roi: None })),
        None => avatar.tracker.as_ref().and_then(|t| t.lock().unwrap().latest.clone()),
    };

    if avatar.args.debug {
        let t = now;
        pivot.rotation = Quat::from_rotation_x((t * 0.7).sin() * 0.25)
            * Quat::from_rotation_y(t.sin() * 0.5)
            * Quat::from_rotation_z((t * 0.5).sin() * 0.15);
    } else if let Some((head, frame)) = latest.as_ref().and_then(|l| Some((l.head?, &l.frame))) {
        // Where the head is in the camera image and how big (= how close), in frame heights.
        // Both are turn-invariant (neck pivot, fitted scale): turning only rotates.
        let q = head.rotation;
        let center = (head.pivot - Vec2::new(frame.w as f32, frame.h as f32) * 0.5) / frame.h as f32;
        let width = head.scale / frame.h as f32;

        // Self-calibration: the first moments with a face (or on request) define the rest
        // pose: "straight ahead" (cancels however the webcam is mounted), plus where you sit.
        if avatar.config.rest.is_none() && avatar.rest_samples.is_none() {
            avatar.rest_samples = Some(Vec::new());
        }
        if let Some(samples) = &mut avatar.rest_samples {
            samples.push((now, q, center, width));
            if now - samples[0].0 > 1.5 && samples.len() >= 10 {
                let first = Vec4::from(samples[0].1);
                let n = samples.len() as f32;
                let rotation = samples.iter().fold(Vec4::ZERO, |acc, s| {
                    let v = Vec4::from(s.1);
                    acc + if v.dot(first) < 0.0 { -v } else { v }
                });
                let center = samples.iter().map(|s| s.2).sum::<Vec2>() / n;
                let width = samples.iter().map(|s| s.3).sum::<f32>() / n;
                avatar.config.rest = Some(config::Rest { version: config::REST_VERSION, rotation: rotation.normalize().to_array(), center: center.to_array(), width });
                avatar.rest_samples = None;
                avatar.config.save();
            }
        }
        let rest = avatar.config.rest.unwrap_or(config::Rest { version: config::REST_VERSION, rotation: [0.0, 0.0, 0.0, 1.0], center: [0.0, 0.0], width });
        let q = Quat::from_array(rest.rotation).normalize().inverse() * q;
        // Mirrored display: reflect the pose across the YZ plane, which flips yaw and roll.
        let q = if cfg.mirror { Quat::from_xyzw(q.x, -q.y, -q.z, q.w) } else { q };
        // Gain on the angle, not the matrix: exaggerates the turn without distorting the
        // model. The mount-height pitch goes after the gain so it stays a constant offset.
        let target = Quat::IDENTITY.slerp(q, cfg.turn_gain) * Quat::from_rotation_x(cfg.base_pitch);

        // Moving in the camera frame moves the model across the window by the same fraction
        // (times `follow`); leaning in brings it closer (`grow` = how much bigger your face
        // got). The window spans 2 * tan(fov/2) * CAMERA_Z world units vertically at the
        // model's rest depth.
        let view = 2.0 * (compose::FOV_DEG.to_radians() * 0.5).tan() * compose::CAMERA_Z;
        let d = (center - Vec2::from(rest.center)) * view * cfg.follow;
        let side = if cfg.mirror { -1.0 } else { 1.0 };
        let grow = (width / rest.width.max(1e-3)).powf(cfg.depth_gain).clamp(0.6, 1.8);
        let placed = Vec3::new(d.x * side, -d.y, grow);

        // Per-frame factor defined at 60 fps, rescaled so 144 Hz monitors smooth the same.
        let follow = 1.0 - cfg.smoothing.powf(time.delta_secs() * 60.0);
        avatar.smoothed = avatar.smoothed.slerp(target, follow);
        avatar.placed = avatar.placed.lerp(placed, follow);
        pivot.rotation = avatar.smoothed;
        avatar.face_seen = now;
        let lowest = latest.as_ref().and_then(|l| l.landmarks.as_ref()).map_or(0.0, |lm| lm[..468].iter().fold(0.0f32, |m, p| m.max(p.y)));
        avatar.left_at_bottom = lowest > frame.h as f32 * 0.95;
    } else if now - avatar.face_seen > FACE_LOST && !avatar.left_at_bottom {
        let ease = 1.0 - 0.9f32.powf(time.delta_secs() * 60.0);
        avatar.smoothed = avatar.smoothed.slerp(Quat::from_rotation_x(cfg.base_pitch), ease);
        avatar.placed = avatar.placed.lerp(Vec3::new(0.0, 0.0, 1.0), ease);
        pivot.rotation = avatar.smoothed;
    }
    // Real depth, not a zoom: `grow` times bigger means 1/grow of the camera distance, so
    // perspective and parallax change like a real object coming closer. x/y scale with the
    // distance, so the model stays at the same spot in the window.
    let dist = compose::CAMERA_Z / avatar.placed.z;
    pivot.scale = Vec3::splat(cfg.model_scale);
    pivot.translation = Vec3::new(avatar.placed.x, cfg.model_y + avatar.placed.y, 0.0) * (dist / compose::CAMERA_Z) + Vec3::Z * (compose::CAMERA_Z - dist);

    // Lost for a moment: optionally gone instead of standing still at rest.
    let gone = cfg.hide_when_lost && !avatar.args.debug && !avatar.left_at_bottom && now - avatar.face_seen > FACE_LOST;
    if let Ok(mut v) = visibility.get_mut(*head) {
        v.set_if_neq(if gone { Visibility::Hidden } else { Visibility::Inherited });
    }

    let hide = |visibility: &mut Query<&mut Visibility>| {
        for p in &patches.0 {
            if let Ok(mut v) = visibility.get_mut(p.entity) {
                *v = Visibility::Hidden;
            }
        }
    };
    let Some(tracked) = latest.filter(|l| l.landmarks.is_some()) else {
        // A few lost frames keep the last eyes/mouth up (no flicker at steep angles).
        if now - avatar.face_seen > HOLD_FEATURES && !avatar.left_at_bottom {
            avatar.job = None;
            hide(&mut visibility);
        }
        return;
    };
    // Start on the newest tracker frame once the previous one is done (the render loop runs
    // faster than the camera); the patches update when it lands, a frame or so later.
    let id = Arc::as_ptr(&tracked) as usize;
    if avatar.job.is_none() && (id != avatar.shown_frame || avatar.debug_face.is_some()) {
        avatar.shown_frame = id;
        let (surface, cfg) = (avatar.surface.clone(), cfg.clone());
        avatar.job = Some(bevy::tasks::AsyncComputeTaskPool::get().spawn(async move { compose_patches(&tracked, &surface, &cfg) }));
    }
    let Some(job) = avatar.job.take_if(|j| j.is_finished()) else { return };
    let Some(baked) = bevy::tasks::block_on(job) else {
        hide(&mut visibility);
        return;
    };
    for (feature, patch) in baked.into_iter().zip(&patches.0) {
        if feature.alpha <= 0.0 {
            if let Ok(mut v) = visibility.get_mut(patch.entity) {
                *v = Visibility::Hidden;
            }
            continue;
        }
        let mut laid = patch_vertices(&feature, &avatar.surface, cfg.flatten);
        // Opaque unless the patch itself turns edge-on to the camera; a half-transparent
        // patch let the model's gloss shine through the mouth. In the vertex alpha, not the
        // material: changing a material re-prepares it, the mesh is rewritten anyway.
        let fade = ((pivot.rotation * laid.facing).z - 0.1) / 0.2;
        let fade = fade.clamp(0.0, 1.0) * feature.alpha;
        for c in &mut laid.col {
            c[3] *= fade;
        }
        if let Some(mesh) = meshes.get_mut(&patch.mesh) {
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, laid.pos);
            mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, laid.col);
            mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, laid.normal);
            mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, laid.uv);
            mesh.insert_attribute(Mesh::ATTRIBUTE_TANGENT, laid.tangent);
        }
        if let Some(image) = images.get_mut(&patch.image) {
            image.data = Some(feature.rgba);
        }
        // Bevy uploads a changed image as a new GPU texture; the material only binds it when
        // it is marked changed too, otherwise it keeps showing the first frame forever.
        if let Some(m) = materials.get_mut(&patch.material) {
            shade(m, cfg.shading, patch.entity == patches.0[2].entity);
        }
        if let Ok(mut v) = visibility.get_mut(patch.entity) {
            *v = Visibility::Inherited;
        }
    }
}

/// Width of the `--mesh` view left of the avatar.
pub const MESH_VIEW_W: u32 = 480;

/// `--mesh`: the camera image (mirrored like the avatar) with the tracked face mesh, eye and
/// lip contours, and the crop the landmark model looked at, plus how the face was found.
/// Mesh colour: green = landmark model sure, yellow = unsure, orange = optical flow only.
fn mesh_view(mut contexts: EguiContexts, mut avatar: ResMut<Avatar>, mut shown: Local<Option<(usize, egui::TextureHandle, String)>>) -> Result {
    if !avatar.args.mesh {
        return Ok(());
    }
    let ctx = contexts.ctx_mut()?;
    let mirror = avatar.calibration().mirror;
    let latest = avatar.tracker.as_ref().and_then(|t| t.lock().unwrap().latest.clone());
    if let Some(t) = latest.filter(|t| shown.as_ref().is_none_or(|s| s.0 != Arc::as_ptr(t) as usize)) {
        let image = draw_mesh(&t, mirror);
        let (yaw, pitch) = t.head.map_or((f32::NAN, f32::NAN), |h| {
            let (y, x, _) = h.rotation.to_euler(EulerRot::YXZ);
            (y.to_degrees(), x.to_degrees())
        });
        let text = format!("{}   presence {:.2}   yaw {yaw:.0}°  pitch {pitch:.0}°", t.how, t.presence);
        match &mut *shown {
            Some((id, tex, old)) => {
                tex.set(image, egui::TextureOptions::LINEAR);
                (*id, *old) = (Arc::as_ptr(&t) as usize, text);
            }
            None => *shown = Some((Arc::as_ptr(&t) as usize, ctx.load_texture("mesh", image, egui::TextureOptions::LINEAR), text)),
        }
    }
    egui::SidePanel::left("mesh").exact_width(MESH_VIEW_W as f32).resizable(false).frame(egui::Frame::new().fill(egui::Color32::from_gray(18))).show(ctx, |ui| {
        match &*shown {
            Some((_, tex, text)) => {
                ui.image((tex.id(), tex.size_vec2()));
                ui.colored_label(egui::Color32::WHITE, text);
            }
            None => {
                ui.label("waiting for the camera...");
            }
        }
    });
    Ok(())
}

fn draw_mesh(t: &tracker::Tracked, mirror: bool) -> egui::ColorImage {
    use tiny_skia::{Color as Sk, Paint, PathBuilder, Pixmap, Stroke, Transform};
    let f = &t.frame;
    let s = MESH_VIEW_W as f32 / f.w as f32;
    let (w, h) = (MESH_VIEW_W, ((f.h as f32 * s) as u32).max(1));
    let mut pm = Pixmap::new(w, h).unwrap();
    for (i, px) in pm.data_mut().chunks_exact_mut(4).enumerate() {
        let (x, y) = (i as u32 % w, i as u32 / w);
        let x = if mirror { w - 1 - x } else { x };
        let (fx, fy) = ((((x as f32 + 0.5) / s) as u32).min(f.w - 1), (((y as f32 + 0.5) / s) as u32).min(f.h - 1));
        let k = ((fy * f.w + fx) * 3) as usize;
        px.copy_from_slice(&[f.rgb[k], f.rgb[k + 1], f.rgb[k + 2], 255]);
    }
    let at = |p: Vec2| {
        let x = p.x * s;
        (if mirror { w as f32 - x } else { x }, p.y * s)
    };
    let mut draw = |loops: &mut dyn Iterator<Item = Vec<Vec2>>, (r, g, b, a): (u8, u8, u8, u8), width: f32| {
        let mut pb = PathBuilder::new();
        for pts in loops {
            let (x, y) = at(pts[0]);
            pb.move_to(x, y);
            for p in &pts[1..] {
                let (x, y) = at(*p);
                pb.line_to(x, y);
            }
            pb.close();
        }
        if let Some(path) = pb.finish() {
            let mut paint = Paint::default();
            paint.set_color(Sk::from_rgba8(r, g, b, a));
            paint.anti_alias = true;
            pm.stroke_path(&path, &paint, &Stroke { width, ..Default::default() }, Transform::identity(), None);
        }
    };
    if let Some(roi) = t.roi {
        draw(&mut std::iter::once(roi.to_vec()), (255, 210, 0, 255), 2.0);
    }
    if let Some(lm) = &t.landmarks {
        let colour = match t.how {
            "flow only" => (255, 140, 0, 150),
            "landmarks" | "re-detected" | "test face" => (60, 255, 120, 110),
            _ => (255, 240, 60, 150),
        };
        let p = |i: u16| lm[i as usize].truncate();
        draw(&mut crate::canonical_mesh::TRIANGLES.iter().map(|t| t.map(p).to_vec()), colour, 0.6);
        draw(&mut [compose::RIGHT_EYE, compose::LEFT_EYE, compose::LIPS].into_iter().map(|r| r.iter().map(|&i| lm[i].truncate()).collect()), (0, 220, 255, 255), 1.5);
    }
    egui::ColorImage::from_rgba_premultiplied([w as usize, h as usize], pm.data())
}

/// The app's one accent: the gold of its rim light. Selection, changed values, the main action.
const ACCENT: egui::Color32 = egui::Color32::from_rgb(236, 180, 5);
const OK: egui::Color32 = egui::Color32::from_rgb(120, 205, 130);
const ERROR: egui::Color32 = egui::Color32::from_rgb(255, 107, 107);
const PANEL_W: f32 = 320.0;
/// Name column of the settings rows: fixed, so every slider lines up across sections.
const LABEL_W: f32 = 112.0;

/// Calm dark styling for the panel: one accent, roomier rows, readable sizes.
fn style_panel(ctx: &egui::Context) {
    use egui::{CornerRadius, FontId, Stroke, TextStyle};
    ctx.style_mut(|s| {
        let v = &mut s.visuals;
        v.panel_fill = egui::Color32::from_rgb(21, 21, 25);
        v.window_fill = egui::Color32::from_rgb(28, 28, 33);
        v.extreme_bg_color = egui::Color32::from_rgb(14, 14, 17);
        v.selection.bg_fill = ACCENT.linear_multiply(0.75);
        v.selection.stroke = Stroke::new(1.0, ACCENT);
        v.slider_trailing_fill = true;
        for w in [&mut v.widgets.inactive, &mut v.widgets.hovered, &mut v.widgets.active, &mut v.widgets.open, &mut v.widgets.noninteractive] {
            w.corner_radius = CornerRadius::same(6);
        }
        v.widgets.inactive.weak_bg_fill = egui::Color32::from_rgb(38, 38, 44);
        v.widgets.inactive.bg_fill = egui::Color32::from_rgb(52, 52, 60);
        v.widgets.hovered.weak_bg_fill = egui::Color32::from_rgb(50, 50, 58);
        s.spacing.item_spacing = egui::vec2(8.0, 8.0);
        s.spacing.button_padding = egui::vec2(10.0, 5.0);
        s.spacing.interact_size.y = 24.0;
        s.text_styles.insert(TextStyle::Heading, FontId::proportional(17.0));
        s.text_styles.insert(TextStyle::Body, FontId::proportional(14.0));
        s.text_styles.insert(TextStyle::Button, FontId::proportional(14.0));
        s.text_styles.insert(TextStyle::Small, FontId::proportional(12.0));
    });
}

/// "Kugel", or the file name of an imported figure.
fn figure_name(model: &str) -> String {
    match model {
        SPHERE => "Kugel".into(),
        other => std::path::Path::new(other).file_stem().map_or(other.into(), |s| s.to_string_lossy().into_owned()),
    }
}

/// A stored value as the panel shows it (80 %, +15 %, 1,00×, 12°) and back.
fn shown(v: f64, unit: Unit) -> f64 {
    match unit {
        Unit::Percent => v * 100.0,
        Unit::Extra => (v - 1.0) * 100.0,
        Unit::Times | Unit::Degrees => v,
        Unit::Radians => v.to_degrees(),
    }
}
fn stored(x: f64, unit: Unit) -> f64 {
    match unit {
        Unit::Percent => x / 100.0,
        Unit::Extra => 1.0 + x / 100.0,
        Unit::Times | Unit::Degrees => x,
        Unit::Radians => x.to_radians(),
    }
}
fn format_value(v: f64, unit: Unit) -> String {
    let x = shown(v, unit);
    let x = if x.abs() < 0.05 { 0.0 } else { x };
    match unit {
        Unit::Percent => format!("{x:.0} %"),
        Unit::Extra => format!("+{x:.0} %"),
        Unit::Times => format!("{x:.2}×").replace('.', ","),
        Unit::Degrees | Unit::Radians => format!("{x:.0}°"),
    }
}
fn parse_value(text: &str, unit: Unit) -> Option<f64> {
    let number: String = text.chars().filter(|c| c.is_ascii_digit() || matches!(c, '-' | '−' | ',' | '.')).collect();
    let x: f64 = number.replace('−', "-").replace(',', ".").parse().ok()?;
    Some(stored(x, unit))
}

/// One settings row: the name (gold when changed; double-click resets it, hover explains
/// it) and its control.
fn setting_row(ui: &mut egui::Ui, setting: &Setting, cfg: &mut Calibration, defaults: &mut Calibration) -> bool {
    let (key, label, help) = match setting {
        Setting::Slider { key, label, help, .. } | Setting::Toggle { key, label, help } => (*key, *label, *help),
    };
    let (is_default, default_text) = match setting {
        Setting::Slider { unit, .. } => ((*cfg.field(key) - *defaults.field(key)).abs() < 1e-4, format_value(*defaults.field(key) as f64, *unit)),
        Setting::Toggle { .. } => (*cfg.flag(key) == *defaults.flag(key), if *defaults.flag(key) { "an".into() } else { "aus".into() }),
    };
    let tip = if is_default { format!("{help}\n\nStandard: {default_text}") } else { format!("{help}\n\nStandard: {default_text} · Doppelklick auf den Namen setzt zurück") };
    let name = egui::RichText::new(label).color(if is_default { ui.visuals().text_color() } else { ACCENT });
    let name = egui::Label::new(name).sense(egui::Sense::click()).truncate();
    let mut changed = false;
    ui.horizontal(|ui| {
        // Sliders: name in a fixed column, then the slider. Switches: the usual box, then its name.
        let name = match setting {
            Setting::Slider { .. } => {
                let column = egui::Layout::left_to_right(egui::Align::Center);
                ui.allocate_ui_with_layout(egui::vec2(LABEL_W, 24.0), column, |ui| {
                    ui.set_min_width(LABEL_W);
                    ui.add(name)
                })
                .inner
            }
            Setting::Toggle { .. } => {
                changed |= ui.checkbox(cfg.flag(key), "").on_hover_text(&tip).changed();
                ui.add(name)
            }
        }
        .on_hover_text(&tip);
        if name.double_clicked() && !is_default {
            match setting {
                Setting::Slider { .. } => *cfg.field(key) = *defaults.field(key),
                Setting::Toggle { .. } => *cfg.flag(key) = *defaults.flag(key),
            }
            changed = true;
        }
        if let &Setting::Slider { min, max, unit, .. } = setting {
            let step = match unit {
                Unit::Times => 0.05,
                Unit::Degrees => 1.0,
                Unit::Radians => 0.5f64.to_radians(),
                Unit::Percent | Unit::Extra => 0.01,
            };
            let slider = egui::Slider::new(cfg.field(key), min..=max)
                .step_by(step)
                .custom_formatter(move |v, _| format_value(v, unit))
                .custom_parser(move |t| parse_value(t, unit));
            changed |= ui.add(slider).on_hover_text(&tip).changed();
        }
    });
    changed
}

/// What the tracker is doing, in a few words, and whether it is fine, unsure or broken.
fn tracking_state(avatar: &Avatar) -> (String, egui::Color32) {
    let Some(tracker) = &avatar.tracker else { return ("Testmodus ohne Webcam".into(), ACCENT) };
    let t = tracker.lock().unwrap();
    let face = t.latest.as_ref().is_some_and(|l| l.landmarks.is_some());
    match () {
        _ if t.error => (t.status.clone(), ERROR),
        _ if !t.running => (t.status.clone(), ACCENT),
        _ if avatar.rest_samples.is_some() => ("Richte aus: schau geradeaus in die Kamera".into(), ACCENT),
        _ if !face => ("Kein Gesicht im Bild".into(), ACCENT),
        _ if t.status != "Läuft" => (t.status.clone(), ACCENT),
        _ => ("Gesicht erkannt".into(), OK),
    }
}

#[allow(clippy::too_many_arguments)]
fn panel(
    mut contexts: EguiContexts,
    mut avatar: ResMut<Avatar>,
    stream: Res<crate::stream::Stream>,
    mut tab: Local<usize>,
    mut confirm_reset: Local<Option<f64>>,
    mut styled: Local<bool>,
    mut unsaved: Local<bool>,
    mut importing: Local<Option<std::sync::mpsc::Receiver<Option<Result<std::path::PathBuf, String>>>>>,
    mut import_error: Local<Option<String>>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    // Saved once the mouse is let go, not on every frame of dragging a slider (a file write
    // each time, on the render thread).
    if *unsaved && !ctx.input(|i| i.pointer.any_down()) {
        avatar.config.save();
        *unsaved = false;
    }
    if !*styled {
        style_panel(ctx);
        *styled = true;
    }
    let (status, status_color) = tracking_state(&avatar);
    // With the panel closed, only what needs attention shows over the avatar.
    let (status, status_color) = match &avatar.config.problem {
        Some(_) if !avatar.panel_open => ("Problem mit der Einstellungsdatei: C öffnet die Einstellungen".to_string(), ERROR),
        _ => (status, status_color),
    };
    if !avatar.panel_open && status_color != OK && status != "Kein Gesicht im Bild" {
        egui::Area::new("status".into()).fixed_pos([8.0 + if avatar.args.mesh { MESH_VIEW_W as f32 } else { 0.0 }, 8.0]).show(ctx, |ui| {
            egui::Frame::new().fill(egui::Color32::from_black_alpha(210)).inner_margin(egui::Margin::symmetric(10, 6)).corner_radius(6).show(ui, |ui| {
                ui.colored_label(status_color, &status);
            });
        });
    }
    if !avatar.panel_open {
        return Ok(());
    }

    let now = ctx.input(|i| i.time);
    let mut changed = false;
    let frame = egui::Frame::new().fill(ctx.style().visuals.panel_fill).inner_margin(egui::Margin::symmetric(18, 16));
    egui::SidePanel::right("panel").exact_width(PANEL_W).resizable(false).frame(frame).show(ctx, |ui| {
        // Figure and tracking state.
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Figur").strong());
            let mut model = avatar.model.clone();
            let list: Vec<String> = std::iter::once(SPHERE.to_string()).chain(avatar.config.models.iter().filter(|m| std::path::Path::new(m).exists()).cloned()).collect();
            egui::ComboBox::from_id_salt("figure").selected_text(figure_name(&model)).width(130.0).show_ui(ui, |ui| {
                for m in &list {
                    ui.selectable_value(&mut model, m.clone(), figure_name(m));
                }
            });
            // The file dialog and Blender take seconds: off the render thread.
            let help = format!("Eigene Figur laden: {}. Alles außer glTF wandelt Blender um.", crate::import::FORMATS.join(", "));
            if ui.add_enabled(importing.is_none(), egui::Button::new(if importing.is_some() { "Lädt …" } else { "Importieren …" })).on_hover_text(help).clicked() {
                let (tx, rx) = std::sync::mpsc::channel();
                std::thread::spawn(move || {
                    let file = rfd::FileDialog::new().add_filter("3D-Modell", crate::import::FORMATS).pick_file();
                    let _ = tx.send(file.map(|f| crate::import::import(&f)));
                });
                *importing = Some(rx);
                *import_error = None;
            }
            if let Some(done) = importing.as_ref().and_then(|rx| rx.try_recv().ok()) {
                *importing = None;
                match done {
                    Some(Ok(path)) => {
                        let path = path.to_string_lossy().into_owned();
                        if !avatar.config.models.contains(&path) {
                            avatar.config.models.push(path.clone());
                        }
                        model = path;
                    }
                    Some(Err(e)) => *import_error = Some(e),
                    None => {}
                }
            }
            if model != avatar.model {
                avatar.config.model = Some(model.clone());
                changed = true;
            }
            avatar.model = model;
        });
        if let Some(e) = import_error.as_ref() {
            ui.label(egui::RichText::new(e).small().color(ERROR));
        }
        ui.horizontal(|ui| {
            let (rect, _) = ui.allocate_exact_size(egui::vec2(8.0, 18.0), egui::Sense::hover());
            ui.painter().circle_filled(rect.center(), 4.0, status_color);
            ui.label(egui::RichText::new(&status).color(ui.visuals().weak_text_color()));
        });
        if let Some(problem) = avatar.config.problem.clone() {
            ui.add_space(4.0);
            egui::Frame::new().fill(ERROR.linear_multiply(0.12)).corner_radius(6).inner_margin(egui::Margin::same(10)).show(ui, |ui| {
                ui.label(egui::RichText::new(problem).color(ERROR));
                if ui.button("Verstanden").clicked() {
                    avatar.config.problem = None;
                }
            });
        }
        ui.add_space(6.0);

        // The one thing to do after sitting down: tell the app where "straight ahead" is.
        let calibrating = avatar.rest_samples.is_some();
        let text = egui::RichText::new(if calibrating { "Richte aus …" } else { "Neu ausrichten" }).strong().color(egui::Color32::from_rgb(24, 20, 6));
        let button = egui::Button::new(text).fill(if calibrating { ACCENT.linear_multiply(0.6) } else { ACCENT }).min_size(egui::vec2(ui.available_width(), 36.0));
        if ui.add_enabled(!calibrating, button).on_hover_text("Legt fest, was „geradeaus“ ist und wo du sitzt. Dauert etwa 2 Sekunden.").clicked() {
            avatar.rest_samples = Some(Vec::new());
        }
        ui.label(egui::RichText::new("Schau dabei 2 Sekunden geradeaus in die Kamera.").small().color(ui.visuals().weak_text_color()));
        ui.add_space(10.0);

        // OBS picks the avatar up as a browser source, with real transparency.
        let url = format!("http://localhost:{}/", crate::stream::PORT);
        ui.horizontal(|ui| {
            let help = format!("In OBS eine Quelle „Browser“ mit dieser Adresse anlegen, {}×{} Pixel. Echte Transparenz, ohne Panel, das Fenster darf verdeckt sein.", avatar.args.width, avatar.args.height);
            if ui.checkbox(&mut avatar.config.serve, egui::RichText::new("Für OBS bereitstellen").strong()).on_hover_text(help).changed() {
                changed = true;
            }
            if avatar.config.serve {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("Adresse kopieren").clicked() {
                        ui.ctx().copy_text(url.clone());
                    }
                });
            }
        });
        if let Some(problem) = &stream.problem {
            ui.label(egui::RichText::new(problem).small().color(ERROR));
        } else if avatar.config.serve {
            ui.label(egui::RichText::new(format!("Browserquelle: {url}")).small().color(ui.visuals().weak_text_color()));
        }
        ui.add_space(10.0);

        // Tabs by what you want to change.
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            for (i, (title, _)) in PANEL.iter().enumerate() {
                let selected = *tab == i;
                let text = egui::RichText::new(*title).color(if selected { egui::Color32::from_rgb(24, 20, 6) } else { ui.visuals().text_color() });
                let b = egui::Button::new(text).fill(if selected { ACCENT } else { egui::Color32::TRANSPARENT }).stroke(egui::Stroke::NONE);
                if ui.add(b).clicked() {
                    *tab = i;
                }
            }
        });
        ui.separator();

        // Settings, scrolling between the fixed header and footer.
        let footer = 46.0;
        let model = avatar.model.clone();
        let mut defaults = Calibration::default();
        egui::ScrollArea::vertical().max_height(ui.available_height() - footer).auto_shrink([false, false]).show(ui, |ui| {
            let cfg = avatar.config.calibration(&model);
            // Panel minus margins, name column, gaps and the value box.
            ui.spacing_mut().slider_width = PANEL_W - 36.0 - LABEL_W - 16.0 - 64.0 - 12.0;
            for (k, (title, settings)) in PANEL[*tab].1.iter().enumerate() {
                ui.add_space(if k == 0 { 4.0 } else { 14.0 });
                ui.label(egui::RichText::new(*title).heading().strong());
                ui.add_space(2.0);
                ui.spacing_mut().item_spacing.y = 6.0;
                for setting in *settings {
                    changed |= setting_row(ui, setting, cfg, &mut defaults);
                }
                ui.spacing_mut().item_spacing.y = 8.0;
            }
            if PANEL[*tab].0 == "Feinschnitt" {
                ui.add_space(14.0);
                ui.label(egui::RichText::new("Teilen").heading().strong());
                if ui.button("Einstellungen kopieren").on_hover_text("Kopiert alle Werte dieser Figur als JSON, zum Teilen oder Sichern.").clicked() {
                    ui.ctx().copy_text(serde_json::to_string_pretty(&*cfg).unwrap());
                }
                ui.label(egui::RichText::new(format!("Gespeichert in {}", Config::path().display())).small().color(ui.visuals().weak_text_color()));
            }
        });

        // Footer: everything saves on its own; resetting asks once.
        ui.separator();
        ui.horizontal(|ui| {
            let armed = confirm_reset.is_some_and(|until| now < until);
            let text = if armed { egui::RichText::new(format!("Wirklich alles für {} zurücksetzen?", figure_name(&model))).color(ERROR) } else { egui::RichText::new("Alles zurücksetzen") };
            if ui.button(text).clicked() {
                if armed {
                    *avatar.calibration() = Calibration::default();
                    changed = true;
                    *confirm_reset = None;
                } else {
                    *confirm_reset = Some(now + 4.0);
                }
            }
            if !armed {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(egui::RichText::new("Speichert automatisch").small().color(ui.visuals().weak_text_color()));
                });
            }
        });
    });
    *unsaved |= changed;
    Ok(())
}

// ------------------------------------------------------------------ test face (--debug)

/// A painted face (two eyes, a mouth) so --debug shows the cutout composition without a
/// webcam. The eyes blink and the mouth talks through the landmark rings, not the pixels.
/// Test-mode head: frontal, in the middle of a 640x480 frame.
fn fake_head() -> tracker::Head {
    tracker::Head { rotation: Quat::IDENTITY, pivot: Vec2::new(320.0, 330.0), scale: 19.0 }
}

/// A painted face whose landmarks are the canonical mesh, so the test mode runs the same
/// frontalization as a real face.
fn fake_face() -> tracker::Frame {
    use crate::canonical_mesh::VERTICES;
    let (w, h) = (640u32, 480u32);
    let head = fake_head();
    let at = |ring: &[usize]| ring.iter().map(|&i| Vec3::from(VERTICES[i]).truncate()).sum::<Vec2>() / ring.len() as f32;
    let (right, left, lips) = (at(compose::RIGHT_EYE), at(compose::LEFT_EYE), at(compose::LIPS));
    let inside = |p: Vec2, c: Vec2, r: Vec2| ((p - c) / r).length_squared() <= 1.0;
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            // Pixel -> canonical cm (frontal head, so the projection is just a scale).
            let p = (Vec2::new(x as f32 - head.pivot.x, head.pivot.y - y as f32)) / head.scale + tracker::NECK.truncate();
            let mut c = [225, 185, 160];
            for e in [right, left] {
                if inside(p, e, Vec2::new(1.35, 0.6)) {
                    c = [245, 245, 240];
                }
                if inside(p, e, Vec2::splat(0.55)) {
                    c = [70, 45, 30];
                }
                if inside(p, e, Vec2::splat(0.25)) {
                    c = [10, 10, 10];
                }
            }
            if inside(p, lips, Vec2::new(2.3, 0.75)) {
                c = [175, 65, 75];
            }
            if inside(p, lips, Vec2::new(1.8, 0.3)) {
                c = [60, 20, 25];
            }
            rgb.extend(c);
        }
    }
    tracker::Frame { w, h, rgb }
}

fn fake_landmarks() -> Vec<Vec3> {
    let proj = compose::Projection::new(&fake_head());
    let mut lm: Vec<Vec3> = crate::canonical_mesh::VERTICES.iter().map(|&v| proj.to_image(Vec3::from(v)).extend(0.0)).collect();
    lm.resize(478, lm[467]);
    lm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tangents_follow_the_texture() {
        // Flat grid, u along +x, v running down (glTF): tangent +x, cross(n, t) = +y = -v.
        let m = PATCH_CELLS as usize + 1;
        let pos: Vec<[f32; 3]> = (0..m * m).map(|k| [(k % m) as f32, -((k / m) as f32), 0.0]).collect();
        let uv: Vec<[f32; 2]> = (0..m * m).map(|k| [(k % m) as f32 * 0.1, (k / m) as f32 * 0.1]).collect();
        for t in tangents(&pos, &vec![[0.0, 0.0, 1.0]; m * m], &uv) {
            assert!((Vec3::new(t[0], t[1], t[2]) - Vec3::X).length() < 1e-4 && t[3] == -1.0, "{t:?}");
        }
    }
}

