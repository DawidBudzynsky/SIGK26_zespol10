from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import moderngl
import numpy as np
import pywavefront
from PIL import Image
from pyrr import Matrix44


REPO_ROOT = Path(__file__).resolve().parents[1]
SHADER_DIR = REPO_ROOT / "SIGK___Projekt_3" / "SIGK___Projekt_3" / "resources" / "shaders" / "phong"
SPHERE_OBJ = REPO_ROOT / "SIGK___Projekt_3" / "SIGK___Projekt_3" / "resources" / "models" / "sphere.obj"

CAMERA = (5.0, 5.0, 15.0)   # zgodnie z phong_window.on_render
FOV = 45.0
NEAR, FAR = 0.1, 1000.0


def load_sphere(path: Path):
    scene = pywavefront.Wavefront(str(path), collect_faces=True, create_materials=True)
    mat = list(scene.materials.values())[0]
    fmt = mat.vertex_format
    arr = np.array(mat.vertices, dtype=np.float32)
    if fmt == "N3F_V3F":
        arr = arr.reshape(-1, 6)
        normals, positions = arr[:, 0:3], arr[:, 3:6]
    elif fmt == "T2F_N3F_V3F":
        arr = arr.reshape(-1, 8)
        normals, positions = arr[:, 2:5], arr[:, 5:8]
    else:
        raise RuntimeError(f"Unsupported vertex format: {fmt}")
    return np.concatenate([positions, normals], axis=1).astype(np.float32)


def build_mvp(obj_pos):
    model = Matrix44.from_translation(obj_pos, dtype=np.float32)
    proj = Matrix44.perspective_projection(FOV, 1.0, NEAR, FAR, dtype=np.float32)
    view = Matrix44.look_at(CAMERA, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), dtype=np.float32)
    return model, proj * view * model


def is_visible(obj_pos, ndc_margin: float = 1.05, depth_range=(4.0, 22.0)) -> bool:
    """Project sphere centre to NDC. Reject only positions that put the sphere
    fully off-screen, za blisko lub za daleko (PDF wskazówka #2).

    ndc_margin = 1.05 dopuszcza częściowe wychodzenie kuli poza kadr, tak
    żeby nie wymuszać wycentrowania (wskazówka #3).

    depth_range w przestrzeni widoku (czyli `w` z clip space) ograniczamy
    do (4, 22), bo z [-20, 20]^3 sporo pozycji wpada na ekstremalnie daleko
    (kula < 5 px) — sieć i tak nie miałaby czego się uczyć.
    """
    _, mvp = build_mvp(obj_pos)
    p = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    clip = p @ np.asarray(mvp).astype(np.float32)
    w = clip[3]
    if w <= depth_range[0] or w >= depth_range[1]:
        return False
    ndc = clip[:3] / w
    return abs(ndc[0]) < ndc_margin and abs(ndc[1]) < ndc_margin


def sample_scene(rng: random.Random) -> dict:
    while True:
        obj_pos = [rng.uniform(-20.0, 20.0) for _ in range(3)]
        if is_visible(obj_pos):
            return {
                "obj_pos":   obj_pos,
                "diffuse":   [rng.randint(0, 255) for _ in range(3)],
                "shininess": rng.uniform(3.0, 20.0),
                "light_pos": [rng.uniform(-20.0, 20.0) for _ in range(3)],
            }


def render(ctx, prog, vao, fbo, params: dict):
    model, mvp = build_mvp(params["obj_pos"])
    fbo.use()
    ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
    ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)

    prog["model_view_projection"].write(np.ascontiguousarray(mvp).astype("f4").tobytes())
    prog["model_matrix"].write(np.ascontiguousarray(model).astype("f4").tobytes())
    prog["material_diffuse"].value   = tuple(c / 255.0 for c in params["diffuse"])
    prog["material_shininess"].value = float(params["shininess"])
    prog["light_position"].value     = tuple(params["light_pos"])
    prog["camera_position"].value    = CAMERA

    vao.render()
    raw = fbo.read(components=3, alignment=1)
    img = np.frombuffer(raw, dtype=np.uint8).reshape(fbo.size[1], fbo.size[0], 3)
    return np.flipud(img).copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",  default=str(Path(__file__).parent / "data"))
    parser.add_argument("--n",    type=int, default=3000)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    vert = (SHADER_DIR / "phong.vert").read_text()
    frag = (SHADER_DIR / "phong.frag").read_text()
    ctx = moderngl.create_standalone_context()
    prog = ctx.program(vertex_shader=vert, fragment_shader=frag)

    vbo = ctx.buffer(load_sphere(SPHERE_OBJ).tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "3f 3f", "in_position", "in_normal")])

    color_tex = ctx.texture((args.size, args.size), 4)
    depth_rb  = ctx.depth_renderbuffer((args.size, args.size))
    fbo = ctx.framebuffer(color_attachments=[color_tex], depth_attachment=depth_rb)

    labels = []
    for i in range(args.n):
        params = sample_scene(rng)
        img = render(ctx, prog, vao, fbo, params)
        Image.fromarray(img).save(out_dir / f"{i:05d}.png")
        labels.append({"idx": i, **params})
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{args.n}")

    meta = {
        "n": args.n, "size": args.size, "seed": args.seed,
        "camera": list(CAMERA), "fov": FOV,
        "samples": labels,
    }
    with open(out_dir / "labels.json", "w") as f:
        json.dump(meta, f)
    print(f"wrote {args.n} images + labels.json to {out_dir}")


if __name__ == "__main__":
    main()
