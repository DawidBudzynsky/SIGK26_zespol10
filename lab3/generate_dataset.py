"""Render a dataset of Phong-shaded sphere images for the neural renderer.

Uses moderngl in standalone (headless) mode with the same shaders supplied with
the project. Each sample saves:
  - data/<idx>.png        (128x128 RGB rendering)
  - data/labels.json      (parameters per index)

Spec (Projekt 3, sec. 2.3):
  - object position: each component in [-20, 20]
  - diffuse color: each component in [0, 255]
  - shininess in [3, 20]
  - light position: each component in [-20, 20]
  - material ambient = [76,76,76], material specular = [255,255,255]
  - light ambient = [25,25,25], light/specular = [255,255,255]
  - 128x128, 3000 images, sphere object, fixed camera.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image
from pyrr import Matrix44


REPO_ROOT = Path(__file__).resolve().parent.parent
SHADER_DIR = REPO_ROOT / "SIGK___Projekt_3" / "resources" / "shaders" / "phong"
MODEL_PATH = REPO_ROOT / "SIGK___Projekt_3" / "resources" / "models" / "sphere.obj"


# Phong shader source - identical math to the provided phong.frag/phong.vert
# but with all 5 constants exposed as uniforms so the spec's exact values can
# be plugged in without editing GLSL.
VERT_SRC = """
#version 330 core
uniform mat4 model_view_projection;
uniform mat4 model_matrix;
in vec3 in_position;
in vec3 in_normal;
out vec3 v_position;
out vec3 v_normal;
void main() {
    v_position = (model_matrix * vec4(in_position, 1.0)).xyz;
    v_normal   = normalize((model_matrix * vec4(in_normal, 0.0)).xyz);
    gl_Position = model_view_projection * vec4(in_position, 1.0);
}
"""

FRAG_SRC = """
#version 330 core
in vec3 v_position;
in vec3 v_normal;
out vec4 f_color;

uniform vec3  material_ambient;
uniform vec3  material_diffuse;
uniform vec3  material_specular;
uniform float material_shininess;

uniform vec3  light_position;
uniform vec3  light_ambient;
uniform vec3  light_diffuse;
uniform vec3  light_specular;

uniform vec3  camera_position;

void main() {
    vec3 ambient  = light_ambient * material_ambient;

    vec3 N = normalize(v_normal);
    vec3 L = normalize(light_position - v_position);
    float cosNL = clamp(dot(N, L), 0.0, 1.0);
    vec3 diffuse = light_diffuse * material_diffuse * cosNL;

    vec3 V = normalize(camera_position - v_position);
    vec3 R = reflect(-L, N);
    float cosRV = clamp(dot(R, V), 0.0, 1.0);
    vec3 specular = light_specular * pow(cosRV, material_shininess) * material_specular;

    vec3 phong_color = clamp(ambient + diffuse + specular, 0.0, 1.0);
    f_color = vec4(phong_color, 1.0);
}
"""


def load_sphere_mesh():
    """Parse the .obj sphere into vertex+normal arrays for ModernGL."""
    import pywavefront
    scene = pywavefront.Wavefront(str(MODEL_PATH), collect_faces=True, create_materials=True)
    mesh = list(scene.materials.values())[0]
    # PyWavefront stores interleaved vertex_format in mesh.vertex_format.
    # For sphere.obj we expect "N3F_V3F".
    fmt = mesh.vertex_format
    verts = np.array(mesh.vertices, dtype=np.float32)
    if fmt == "N3F_V3F":
        verts = verts.reshape(-1, 6)
        normals = verts[:, 0:3]
        positions = verts[:, 3:6]
    elif fmt == "T2F_N3F_V3F":
        verts = verts.reshape(-1, 8)
        normals = verts[:, 2:5]
        positions = verts[:, 5:8]
    elif fmt == "V3F":
        positions = verts.reshape(-1, 3)
        normals = np.zeros_like(positions)
        # face normals fall-back (sphere ships with normals so this is unlikely)
    else:
        raise RuntimeError(f"Unsupported vertex format: {fmt}")
    interleaved = np.concatenate([positions, normals], axis=1).astype(np.float32)
    return interleaved


def project_to_clip(obj_pos, camera_pos, fov_deg=45.0, aspect=1.0,
                    near=0.1, far=1000.0):
    """Return clip-space coords of the object center (the model translation)."""
    model, mvp = build_mvp(obj_pos, camera_pos, fov_deg, aspect, near, far)
    # pyrr stores matrices row-major intended for row-vector multiply: clip = p @ mvp.
    p = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    clip = p @ np.asarray(mvp).astype(np.float32)
    return clip


def is_visible(obj_pos, camera_pos, ndc_margin=0.85, depth_range=(3.0, 30.0)):
    """Object is visible if its center, projected to NDC, lies within
    [-ndc_margin, ndc_margin]^2 and at a sensible camera-space depth.
    The margin (<1.0) keeps the sphere on-screen even though radius=1.

    Per spec hint #2 we reject the corner cases the [-20, 20]^3 range
    permits (sphere fully off-screen / behind the camera)."""
    clip = project_to_clip(obj_pos, camera_pos)
    w = clip[3]
    if w <= depth_range[0] or w >= depth_range[1]:
        return False
    ndc = clip[:3] / w
    return abs(ndc[0]) < ndc_margin and abs(ndc[1]) < ndc_margin


def sample_params(rng: random.Random, camera_pos) -> dict:
    """Randomize a scene per spec 2.3 with rejection sampling on object
    position so the sphere is visible (spec hint #2)."""
    while True:
        obj_pos = [rng.uniform(-20.0, 20.0) for _ in range(3)]
        if is_visible(obj_pos, camera_pos):
            break
    return {
        "obj_pos":   obj_pos,
        "diffuse":   [rng.randint(0, 255) for _ in range(3)],
        "shininess": rng.uniform(3.0, 20.0),
        "light_pos": [rng.uniform(-20.0, 20.0) for _ in range(3)],
    }


def build_mvp(obj_pos, camera_pos, fov_deg=45.0, aspect=1.0, near=0.1, far=1000.0):
    model = Matrix44.from_translation(obj_pos, dtype=np.float32)
    proj = Matrix44.perspective_projection(fov_deg, aspect, near, far, dtype=np.float32)
    view = Matrix44.look_at(
        camera_pos, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), dtype=np.float32,
    )
    mvp = proj * view * model
    return model, mvp


def render_scene(ctx, prog, vao, fbo, params, camera_pos):
    obj_pos = params["obj_pos"]
    model, mvp = build_mvp(obj_pos, camera_pos)

    fbo.use()
    ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
    ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)

    prog["model_view_projection"].write(np.ascontiguousarray(mvp).astype("f4").tobytes())
    prog["model_matrix"].write(np.ascontiguousarray(model).astype("f4").tobytes())

    prog["material_ambient"].value = (76.0 / 255.0,) * 3
    prog["material_specular"].value = (1.0, 1.0, 1.0)
    prog["material_diffuse"].value = tuple(c / 255.0 for c in params["diffuse"])
    prog["material_shininess"].value = float(params["shininess"])

    prog["light_position"].value = tuple(params["light_pos"])
    prog["light_ambient"].value = (25.0 / 255.0,) * 3
    prog["light_diffuse"].value = (1.0, 1.0, 1.0)
    prog["light_specular"].value = (1.0, 1.0, 1.0)

    prog["camera_position"].value = tuple(camera_pos)

    vao.render()

    raw = fbo.read(components=3, alignment=1)
    img = np.frombuffer(raw, dtype=np.uint8).reshape(fbo.size[1], fbo.size[0], 3)
    img = np.flipud(img).copy()
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=str(Path(__file__).parent / "data"))
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cam", type=str, default="5,5,15",
                        help="camera xyz (comma separated)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cam = tuple(float(x) for x in args.cam.split(","))
    rng = random.Random(args.seed)

    ctx = moderngl.create_standalone_context()
    prog = ctx.program(vertex_shader=VERT_SRC, fragment_shader=FRAG_SRC)
    vbo_data = load_sphere_mesh()
    vbo = ctx.buffer(vbo_data.tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "3f 3f", "in_position", "in_normal")])

    color_tex = ctx.texture((args.size, args.size), 4)
    depth_rb = ctx.depth_renderbuffer((args.size, args.size))
    fbo = ctx.framebuffer(color_attachments=[color_tex], depth_attachment=depth_rb)

    labels = []
    for i in range(args.n):
        params = sample_params(rng, cam)
        img = render_scene(ctx, prog, vao, fbo, params, cam)
        Image.fromarray(img).save(out_dir / f"{i:05d}.png")
        labels.append({
            "idx": i,
            "obj_pos":   params["obj_pos"],
            "diffuse":   params["diffuse"],
            "shininess": params["shininess"],
            "light_pos": params["light_pos"],
            "camera":    list(cam),
        })
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{args.n}")

    meta = {
        "n": args.n,
        "size": args.size,
        "seed": args.seed,
        "camera": list(cam),
        "samples": labels,
    }
    with open(out_dir / "labels.json", "w") as f:
        json.dump(meta, f)

    print(f"Wrote {args.n} images + labels.json to {out_dir}")


if __name__ == "__main__":
    main()
