# backend/app/vitonhd_wrapper.py
# -*- coding: utf-8 -*-
import sys
import os
import shutil
import subprocess
from io import BytesIO
from pathlib import Path
import uuid

from PIL import Image
import numpy as np
import cv2
from PIL import Image, ImageOps
import numpy as np

# (Opcional) Soporte HEIC/HEIF si subes fotos de iPhone
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

def _normalize_mask(mask_path: Path):
    """
    Normaliza una máscara (cloth o parsing) para eliminar saturación
    (todo blanco o todo negro) y obtener bordes grises suavizados.
    """
    im = Image.open(mask_path).convert("L")
    arr = np.array(im, dtype=np.float32)
    # Normaliza percentiles 2–98 para suavizar bordes
    lo, hi = np.percentile(arr, (2, 98))
    arr = np.clip((arr - lo) / max(hi - lo, 1e-5), 0, 1) * 255
    im = Image.fromarray(arr.astype(np.uint8))
    im.save(mask_path)


def _detect_backend_root() -> Path:
    """Detecta la carpeta 'backend/' de forma robusta (sin usar run_id)."""
    this_file = globals().get("__file__", None)
    if this_file:
        return Path(this_file).resolve().parents[1]

    cwd = Path.cwd().resolve()
    if (cwd / "VITON-HD").exists() and (cwd / "app").exists():
        return cwd
    if (cwd.name == "app") and (cwd.parent / "VITON-HD").exists():
        return cwd.parent
    for p in [cwd, *cwd.parents]:
        if (p / "VITON-HD").exists():
            return p

    raise RuntimeError("No pude detectar la carpeta 'backend'. Ejecuta uvicorn parado en 'backend/'")


ROOT = _detect_backend_root()
VITON = ROOT / "VITON-HD"
APP_DIR = ROOT / "app"
RESULTS = APP_DIR / "storage" / "results"
TEMP = APP_DIR / "storage" / "tmp"
RESULTS.mkdir(parents=True, exist_ok=True)
TEMP.mkdir(parents=True, exist_ok=True)


# ---------- Utilidades robustas de imagen (sin run_id global) ----------
import traceback

def _append_debug_line(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")

def _open_image_strict(path: Path) -> Image.Image:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Archivo inválido o vacío: {path}")
    im = Image.open(path)
    im.load()
    im = im.convert("RGB")
    buf = BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _safe_remove_bg(img: Image.Image) -> Image.Image:
    try:
        from rembg import remove
    except ImportError as e:
        raise RuntimeError("Falta 'rembg' u 'onnxruntime'. Instala con: pip install rembg onnxruntime") from e

    raw_buf = BytesIO()
    img.save(raw_buf, format="PNG")
    raw_bytes = raw_buf.getvalue()

    out_bytes = remove(raw_bytes)
    if not out_bytes:
        raise RuntimeError("rembg devolvió bytes vacíos.")
    out_img = Image.open(BytesIO(out_bytes))
    out_img.load()
    return out_img.convert("RGBA")


def _binary_mask_from_alpha(rgba: Image.Image) -> Image.Image:
    alpha = np.array(rgba.split()[-1])
    mask01 = (alpha > 0).astype(np.uint8) * 255
    return Image.fromarray(mask01, mode="L")


def _morph_cleanup(mask: Image.Image) -> Image.Image:
    arr = np.array(mask)
    kernel = np.ones((3, 3), np.uint8)
    arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel, iterations=2)
    arr = cv2.morphologyEx(arr, cv2.MORPH_OPEN, kernel, iterations=1)
    return Image.fromarray(arr, mode="L")


def _save_parse_from_person(person_img_path: Path, parse_dir: Path):
    """
    Ejecuta el PGN sobre la imagen de persona para generar el parsing (segmentación).
    """
    print("[STEP] parse")
    parse_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(person_img_path).stem
    parse_dst = parse_dir / f"{stem}.png"

    # Ejecutar el PGN
    cmd = [
        r"C:\Users\esteb\miniconda3\envs\viton-pgn\python.exe",
        str(ROOT / "CIHP_PGN" / "inf_pgn.py"),
        "--image", str(person_img_path),
        "--output", str(parse_dir)
    ]
    print(f"[PGN] Ejecutando: {' '.join(cmd)}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""  # fuerza CPU si quieres
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Error en PGN:\n{proc.stderr}")

    print("[STEP] parse ensure jpg")

    # 🔧 Normalizar máscara para evitar fondos planos
    _normalize_mask(parse_dst)


def _save_cloth_mask(cloth_img_path: Path, cmask_dir: Path) -> None:
    """
    Genera cloth-mask/<cloth>.(png y jpg) como máscara binaria.
    El repo espera usually 'cloth.jpg' en cloth-mask, por eso guardamos ambos.
    """
    cmask_dir.mkdir(parents=True, exist_ok=True)

    base = Image.open(cloth_img_path).convert("RGB")
    gray = np.array(base.convert("L"))
    # Umbral por Otsu -> binario
    _, arr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Componente conectado mayor (evita agujeros/ruido)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((arr > 0).astype(np.uint8), connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + int(np.argmax(areas))
        arr = np.where(labels == keep, 255, 0).astype(np.uint8)

    # Cierre morfológico
    arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    mask = Image.fromarray(arr, mode="L")

    stem = cloth_img_path.stem
    out_mask_png = cmask_dir / f"{stem}.png"
    out_mask_jpg = cmask_dir / f"{stem}.jpg"
    mask.save(out_mask_png)  # L
    mask.convert("L").save(out_mask_jpg, quality=95, subsampling=0)

def _save_openpose_img(person_img_path: Path, openpose_img_dir: Path) -> Path:
    openpose_img_dir.mkdir(parents=True, exist_ok=True)
    base = _open_image_strict(person_img_path)  # RGB normalizado
    out_png = openpose_img_dir / f"{person_img_path.stem}_rendered.png"
    base.convert("RGB").save(out_png, format="PNG")   # <--- forzamos RGB
    return out_png


def _save_openpose_json(person_img_path: Path, openpose_json_dir: Path) -> None:
    """
    Genera JSON tipo OpenPose (BODY_25) usando MediaPipe Pose.
    Tolera selfies / plano medio: si faltan caderas/piernas, las estima.
    Lanza RuntimeError si no detecta nada razonable (nunca placeholders 0).
    """
    import json, cv2, math
    import mediapipe as mp

    openpose_json_dir.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.imread(str(person_img_path))
    if img_bgr is None:
        raise RuntimeError(f"No pude leer la imagen para pose: {person_img_path}")
    H, W = img_bgr.shape[:2]

    to_px = lambda lm: (float(lm.x * W), float(lm.y * H), float(lm.visibility))

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, enable_segmentation=False) as pose:
        res = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            raise RuntimeError("MediaPipe no detectó pose. Intenta foto con buena luz y más del torso visible.")

        lm = res.pose_landmarks.landmark

        # indices MediaPipe útiles
        NOSE=0; RE=2; LE=1; REAR=4; LEAR=3
        RSH=12; LSH=11; REL=14; LEL=13; RWR=16; LWR=15
        RHIP=24; LHIP=23; RKN=26; LKN=25; RANK=28; LANK=27

        def get(i): 
            return to_px(lm[i])

        def mid(a,b):
            ax,ay,ac = get(a); bx,by,bc = get(b)
            return [(ax+bx)/2.0, (ay+by)/2.0, (ac+bc)/2.0]

        # Neck ≈ punto medio hombros
        neck = mid(LSH, RSH)

        # Si faltan caderas/rodillas/tobillos, estimar a partir de hombros:
        def ensure_vis(p, minv=0.01):  # evita 0 exacto
            x,y,c = p
            return [float(max(0,min(W-1,x))), float(max(0,min(H-1,y))), float(max(minv,c))]

        # “Proyección” vertical a partir de hombros para estimar caderas/piernas si no visibles
        def project_down(src, factor_h=0.9):
            x,y,c = src
            return [x, min(H-1, y + H*factor_h*0.25), max(c, 0.2)]

        # tomar puntos (si vis<1e-3 se considera ausente)
        def safe(i, fallback):
            x,y,c = get(i)
            if c < 1e-3:
                return ensure_vis(fallback)
            return ensure_vis([x,y,c])

        # caderas: si faltan, proyecta desde hombros
        hipR_est = project_down(get(RSH))
        hipL_est = project_down(get(LSH))
        hip_mid  = [(hipR_est[0]+hipL_est[0])/2, (hipR_est[1]+hipL_est[1])/2, (hipR_est[2]+hipL_est[2])/2]

        RHIPp = safe(RHIP, hipR_est)
        LHIPp = safe(LHIP, hipL_est)
        MIDHIP = ensure_vis([(RHIPp[0]+LHIPp[0])/2, (RHIPp[1]+LHIPp[1])/2, (RHIPp[2]+LHIPp[2])/2])

        # rodillas/tobillos: si faltan, proyecta hacia abajo desde cadera
        def knee_from_hip(hip): return project_down(hip, factor_h=1.2)
        def ankle_from_knee(knee): return project_down(knee, factor_h=1.2)

        RKNp  = safe(RKN, knee_from_hip(RHIPp))
        LKNp  = safe(LKN, knee_from_hip(LHIPp))
        RANKp = safe(RANK, ankle_from_knee(RKNp))
        LANKp = safe(LANK, ankle_from_knee(LKNp))

        # ojos/orejas: si faltan, usa nariz
        NOSEp = ensure_vis(get(NOSE))
        REp   = safe(RE, NOSEp)
        LEp   = safe(LE, NOSEp)
        REARp = safe(REAR, REp)
        LEARp = safe(LEAR, LEp)

        # codos/muñecas: caen a hombros si faltan
        RELp = safe(REL, get(RSH))
        LELp = safe(LEL, get(LSH))
        RWRp = safe(RWR, RELp)
        LWRp = safe(LWR, LELp)

        # BODY_25 (0..24)
        body25 = [
            NOSEp,                          # 0 nose
            ensure_vis(neck),               # 1 neck (aprox)
            ensure_vis(get(RSH)),           # 2 RShoulder
            RELp,                           # 3 RElbow
            RWRp,                           # 4 RWrist
            ensure_vis(get(LSH)),           # 5 LShoulder
            LELp,                           # 6 LElbow
            LWRp,                           # 7 LWrist
            MIDHIP,                         # 8 mid-hip
            RHIPp,                          # 9 RHip
            RKNp,                           #10 RKnee
            RANKp,                          #11 RAnkle
            LHIPp,                          #12 LHip
            LKNp,                           #13 LKnee
            LANKp,                          #14 LAnkle
            REp,                            #15 REye
            LEp,                            #16 LEye
            REARp,                          #17 REar
            LEARp,                          #18 LEar
            [0,0,0.0], [0,0,0.0], [0,0,0.0],#19..21 left toes/heel (sin info)
            [0,0,0.0], [0,0,0.0], [0,0,0.0] #22..24 right toes/heel (sin info)
        ]

        flat = []
        for x,y,c in body25:
            flat += [float(x), float(y), float(c)]

        data = {"version": 1.3, "people":[{
            "person_id":[-1],
            "pose_keypoints_2d": flat,
            "face_keypoints_2d":[],
            "hand_left_keypoints_2d":[],
            "hand_right_keypoints_2d":[],
            "pose_keypoints_3d":[],
            "face_keypoints_3d":[],
            "hand_left_keypoints_3d":[],
            "hand_right_keypoints_3d":[]
        }]}

        stem = person_img_path.stem
        json_main = openpose_json_dir / f"{stem}_keypoints.json"
        json_rnd  = openpose_json_dir / f"{stem}_rendered_keypoints.json"
        with open(json_main, "w", encoding="utf-8") as f:
            json.dump(data, f)
        with open(json_rnd,  "w", encoding="utf-8") as f:
            json.dump(data, f)

        # Validación fuerte: no aceptar todo-cero
        if sum(flat) == 0.0:
            raise RuntimeError("Pose inválida: MediaPipe produjo todo ceros.")

def _assert_openpose_has_points(openpose_json_dir: Path, person_img_path: Path):
    import json
    stem = Path(person_img_path).stem
    jp = openpose_json_dir / f"{stem}_keypoints.json"
    if not jp.exists():
        raise RuntimeError(f"Falta {jp}")
    data = json.loads(jp.read_text(encoding="utf-8"))
    pts = data.get("people", [{}])[0].get("pose_keypoints_2d", [])
    if not pts or sum(pts) == 0.0:
        raise RuntimeError("OpenPose JSON inválido: todos los puntos son 0. Revisa la foto (luz, encuadre).")

def _find_first_image(*roots: Path) -> Path | None:
    exts = (".png", ".jpg", ".jpeg", ".webp")
    for root in roots:
        if not root or not Path(root).exists():
            continue
        root = Path(root)
        # 1) imágenes directamente en el directorio
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                return p
        # 2) búsqueda recursiva
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                return p
    return None

def _fit_on_canvas(img: Image.Image, out_w=768, out_h=1024, bg=(255, 255, 255)) -> Image.Image:
    """Redimensiona manteniendo aspecto y coloca en un lienzo out_w×out_h."""
    img = img.convert("RGBA")
    w, h = img.size
    scale = min(out_w / w, out_h / h)
    nw, nh = int(w * scale), int(h * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (out_w, out_h), (*bg, 255))
    canvas.paste(img, ((out_w - nw)//2, (out_h - nh)//2), img)
    return canvas.convert("RGB")

def _force_mode(img_path: Path, mode: str) -> None:
    """Abre y guarda la imagen garantizando el modo/paleta deseado."""
    try:
        im = Image.open(img_path)
        if im.mode != mode:
            im = im.convert(mode)
        # Para JPG exige RGB/L; para PNG da igual, pero conservamos
        if img_path.suffix.lower() in (".jpg", ".jpeg") and mode not in ("RGB", "L"):
            im = im.convert("RGB")
        im.save(img_path)
    except Exception:
        # Si falla, no bloqueamos el pipeline; el repo puede seguir
        pass

# ---------- Función principal (único lugar donde aparece run_id) ----------

def _normalize_dataset_tree(test_root: Path) -> None:
    """
    Normaliza TODOS los .png/.jpg dentro de datasets/test para evitar RGBA.
    
    image-parse/*  => 'L'
    cloth-mask/*   => 'L'
    lo demás (image/, cloth/, openpose-img/...) => 'RGB'
    """
    exts = {".png", ".jpg", ".jpeg"}
    for p in test_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            try:
                im = Image.open(p)
                parent = p.parent.name.lower()
                if parent in ("image-parse", "cloth-mask"):
                    target = "L"
                else:
                    target = "RGB"
                if im.mode != target:
                    im = im.convert(target)
                    im.save(p)
            except Exception:# no bloqueamos, seguimos con el resto
                pass

def _dump_tree_modes(root: Path, outfile: Path) -> None:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    lines = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            try:
                im = Image.open(p)
                lines.append(f"{p.relative_to(root)} :: {im.mode} :: {im.size}")
            except Exception as e:
                lines.append(f"{p.relative_to(root)} :: ERROR :: {e}")
    _append_debug_line(outfile, "\n".join(lines))   # <<<< append, no write_text


def _normalize_dataset_tree(test_root: Path) -> None:
    """
    Normaliza TODOS los .png/.jpg dentro de datasets/test para evitar RGBA.
    
    image-parse/
  => 'L'
    
    cloth-mask/*   => 'L'
    lo demás (image/, cloth/, openpose-img/...) => 'RGB'"""
    exts = {".png", ".jpg", ".jpeg"}
    for p in test_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            try:
                im = Image.open(p)
                parent = p.parent.name.lower()
                target = "L" if parent in ("image-parse", "cloth-mask") else "RGB"
                if im.mode != target:
                    im = im.convert(target)
                    im.save(p)
            except Exception:
                pass

def _cloth_mask_from_alpha_or_rembg(cloth_img_path: Path, cmask_dir: Path) -> None:
    """
    Genera la máscara de la prenda usando transparencias (alpha)
    o rembg si no hay alpha disponible. Mucho más potente que Otsu.
    """
    cmask_dir.mkdir(parents=True, exist_ok=True)
    im = Image.open(cloth_img_path)

    # Si ya viene con canal alpha, lo usamos directamente
    if im.mode == "RGBA":
        alpha = np.array(im.split()[-1])
        mask_arr = (alpha > 0).astype(np.uint8) * 255
    else:
        # Intentar rembg para fondo complejo
        try:
            from rembg import remove
            buf = BytesIO()
            im.convert("RGBA").save(buf, format="PNG")
            result = remove(buf.getvalue())
            rgba = Image.open(BytesIO(result)).convert("RGBA")
            alpha = np.array(rgba.split()[-1])
            mask_arr = (alpha > 0).astype(np.uint8) * 255
        except ImportError:
            # Último recurso: Otsu
            gray = np.array(im.convert("L"))
            _, mask_arr = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

    # Limpieza morfológica
    kernel = np.ones((5, 5), np.uint8)
    mask_arr = cv2.morphologyEx(mask_arr, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = Image.fromarray(mask_arr, mode="L")

    stem = cloth_img_path.stem
    out_mask_png = cmask_dir / f"{stem}.png"
    out_mask_jpg = cmask_dir / f"{stem}.jpg"
    mask.save(out_mask_png)
    mask.save(out_mask_jpg, quality=95, subsampling=0)
    _normalize_mask(out_mask_png)

def _classify_garment(cmask_path: Path, canvas_hw=(1024, 768)) -> str:
    """
    Heurística muy simple para decidir si la prenda es superior/inferior/accesorio
    según la máscara (posición del centroide y relación alto/ancho).
    """
    if not cmask_path.exists():
        return "upper"
    m = Image.open(cmask_path).convert("L")
    m = m.resize((canvas_hw[1], canvas_hw[0]), Image.NEAREST)  # (W,H) -> (768,1024)
    arr = (np.array(m) > 0).astype(np.uint8)

    h, w = arr.shape
    ys, xs = np.where(arr > 0)
    if len(ys) < 50:   # área min
        return "accessory"

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    bbox_h = max(1, y_max - y_min + 1)
    bbox_w = max(1, x_max - x_min + 1)
    cy = ys.mean() / h
    ar = bbox_h / bbox_w

    # Regla: si está mayormente abajo y es muy alto -> lower
    if cy > 0.55 and ar > 1.2:
        return "lower"
    # Si ocupa poco y está arriba -> accesorio (bufanda, gorro)
    if cy < 0.35 and (bbox_h * bbox_w) / (h * w) < 0.18:
        return "accessory"
    # default
    return "upper"


def _detect_face_bbox(person_img_path: Path) -> tuple | None:
    """
    Devuelve (x, y, w, h) del rostro con HaarCascade (sin descargar modelos).
    """
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        img = cv2.imread(str(person_img_path))
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                              minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces) == 0:
            return None
        # coge la cara más grande
        faces = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)
        return tuple(map(int, faces[0]))
    except Exception:
        return None


def _feathered_paste(dst_rgb: Image.Image, src_rgb: Image.Image, box: tuple, feather: int = 25):
    """
    Pega src sobre dst dentro de 'box' usando una máscara con borde suavizado (feather).
    'box' es (x, y, w, h) en píxeles.
    """
    x, y, w, h = box
    w = max(1, w); h = max(1, h)
    src_crop = src_rgb.crop((x, y, x + w, y + h))
    mask = Image.new("L", (w, h), 255)
    # feather con erosion/dilate gauss
    mask_arr = np.array(mask, dtype=np.float32)
    # crea contorno suavizado
    k = max(3, feather | 1)
    mask_arr = cv2.GaussianBlur(mask_arr, (k, k), 0)
    mask = Image.fromarray(mask_arr.clip(0, 255).astype(np.uint8), "L")

    dst_rgb.paste(src_crop, (x, y), mask)


def _restore_face_and_regions(final_img_path: Path, person_img_path: Path,
                              garment_kind: str, canvas=(768, 1024)) -> None:
    """
    Post-procesa la imagen generada:
      - Restaura cara/cuello con feather.
      - Si la prenda parece 'upper', preserva más la parte inferior (evita artefacto en piernas).
    Guarda el resultado sobre final_img_path.
    """
    out = Image.open(final_img_path).convert("RGB")
    person = Image.open(person_img_path).convert("RGB")
    # asegúrate mismo tamaño (768x1024)
    if out.size != (canvas[0], canvas[1]):
        out = out.resize((canvas[0], canvas[1]), Image.LANCZOS)
    if person.size != (canvas[0], canvas[1]):
        person = person.resize((canvas[0], canvas[1]), Image.LANCZOS)

    # 1) Cara/cuello
    bbox = _detect_face_bbox(person_img_path)
    if bbox:
        x, y, w, h = bbox
        # expandir bbox para incluir cuello (15% abajo y 10% lados)
        expand_x = int(w * 0.1)
        expand_y_top = int(h * 0.1)
        expand_y_bot = int(h * 0.35)
        x = max(0, x - expand_x)
        y = max(0, y - expand_y_top)
        w = min(canvas[0] - x, w + 2 * expand_x)
        h = min(canvas[1] - y, h + expand_y_bot + expand_y_top)
        _feathered_paste(out, person, (x, y, w, h), feather=31)

    # 2) Si la prenda es 'upper', preserva más la zona inferior
    if garment_kind == "upper":
        # Línea de corte suave a 60% de la altura
        W, H = canvas
        y_cut = int(H * 0.60)
        if y_cut < H - 5:
            band_h = H - y_cut
            band_src = person.crop((0, y_cut, W, H))
            # máscara feather vertical
            band_mask = Image.new("L", (W, band_h), 255)
            m = np.array(band_mask, dtype=np.float32)
            m = cv2.GaussianBlur(m, (31, 31), 0)
            band_mask = Image.fromarray(m.clip(0, 255).astype(np.uint8), "L")
            out.paste(band_src, (0, y_cut), band_mask)

    out.save(final_img_path, quality=95, subsampling=0)

def run_viton_hd(person_path: str, cloth_path: str) -> str:
    """
    Ejecuta el flujo de VITON-HD completo y devuelve la ruta del resultado generado.
    Maneja logs y errores detalladamente.
    """
    import uuid
    import traceback

    # === VARIABLES BASE SIEMPRE DEFINIDAS ===
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    run_dir  = TEMP / run_id                              # ...\app\storage\tmp\run_xxxx

    # Dataset temporal (TMP)
    test_dir  = run_dir / "datasets" / "test"
    img_dir   = test_dir / "image"
    parse_dir = test_dir / "image-parse"
    cloth_dir = test_dir / "cloth"
    cmask_dir = test_dir / "cloth-mask"
    op_img_dir  = test_dir / "openpose-img"
    op_json_dir = test_dir / "openpose-json"

    # Directorios de salida y logs (RESULTS)
    out_dir = RESULTS / run_id / "try-on"
    log_dir = RESULTS / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    pre_debug = log_dir / "pre_debug.txt"

    # Crear estructura del dataset (TMP)
    for d in (img_dir, parse_dir, cloth_dir, cmask_dir, op_img_dir, op_json_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Archivo de pares en TMP/datasets
    pairs_txt = run_dir / "datasets" / "test" / "test_pairs.txt"
    pairs_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(pairs_txt, "w", encoding="utf-8", newline="\n") as f:
        f.write("person.jpg cloth.jpg\n")

    print(f"[DEBUG] run_id definido correctamente: {run_id}")
    print(f"[DEBUG] log_dir = {log_dir}")
    print(f"[DEBUG] pre_debug = {pre_debug}")

    # =========================
    # PREPROCESADO (try/except)
    # =========================
    try:
        _append_debug_line(pre_debug, "[STEP] copy inputs")
        person_dst = img_dir / "person.jpg"
        cloth_dst  = cloth_dir / "cloth.jpg"
        shutil.copy2(person_path, person_dst)
        shutil.copy2(cloth_path, cloth_dst)

        _append_debug_line(pre_debug, "[STEP] parse")
        _save_parse_from_person(person_dst, parse_dir)

        _append_debug_line(pre_debug, "[STEP] parse ensure jpg")
        p_png = parse_dir / "person.png"
        p_jpg = parse_dir / "person.jpg"
        if p_png.exists() and not p_jpg.exists():
            Image.open(p_png).convert("L").save(p_jpg, quality=95, subsampling=0)

        _append_debug_line(pre_debug, "[STEP] cloth-mask (alpha/rembg)")
        _cloth_mask_from_alpha_or_rembg(cloth_dst, cmask_dir)

        _append_debug_line(pre_debug, "[STEP] openpose placeholders")
        _save_openpose_img(person_dst, op_img_dir)
        _save_openpose_json(person_dst, op_json_dir)
        _assert_openpose_has_points(op_json_dir, person_dst)

        _append_debug_line(pre_debug, "[STEP] normalize tree")
        test_root = run_dir / "datasets" / "test"
        _normalize_dataset_tree(test_root)

        _append_debug_line(pre_debug, "[STEP] dump modes")
        _dump_tree_modes(test_root, pre_debug)

        _append_debug_line(pre_debug, "[STEP] write pairs")
        # escribir pairs (asegurando carpeta padre)
        pairs_txt.parent.mkdir(parents=True, exist_ok=True)
        with open(pairs_txt, "w", encoding="utf-8", newline="\n") as f:
            f.write("person.jpg cloth.jpg\n")
        _append_debug_line(pre_debug, f"[OK] pairs -> {pairs_txt}")

    except Exception:
        # Si algo del prepro falla, deja volcado y aborta con mensaje claro
        try:
            test_root = run_dir / "datasets" / "test"
            _dump_tree_modes(test_root, pre_debug)
            _append_debug_line(pre_debug, "[ERROR] Exception in prepro:")
            _append_debug_line(pre_debug, traceback.format_exc())
        except Exception:
            pass
        # >>> OJO: el raise AHORA SÍ está dentro del except (no fuera)
        raise RuntimeError(f"Fallo en preprocesado antes de test.py. Revisa {pre_debug}")

    # =========================
    # EJECUCIÓN DE test.py
    # =========================
    log_file = log_dir / "vitonhd_test_log.txt"

    # (Por si acaso) normaliza canales otra vez antes de correr
    test_root = run_dir / "datasets" / "test"
    _normalize_dataset_tree(test_root)

    cmd = [
        sys.executable,
        str(VITON / "test.py"),
        "--name", "alias",
        "--dataset_dir", str(run_dir / "datasets"),
        "--dataset_list", str(pairs_txt),
        "--checkpoint_dir", str(VITON / "checkpoints"),
        "--save_dir", str(log_dir),
        "--load_height", "1024",
        "--load_width", "768",
    ]

    print(f"[VITON-HD] PYTHON={sys.executable}")
    print(f"[VITON-HD] run_id={run_id}")
    print(f"[VITON-HD] DATASET_DIR={run_dir / 'datasets'}")
    print(f"[VITON-HD] LOG={log_file}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    #env["CUDA_VISIBLE_DEVICES"] = ""  # fuerza CPU*
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env.get("PYTHONPATH", "")) if env.get("PYTHONPATH") else str(ROOT)

    proc = subprocess.run(
        cmd,
        cwd=str(VITON),
        env=env,
        capture_output=True,
        text=True,
        shell=False,
    )

    # Guardar stdout/stderr SIEMPRE
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== CMD ===\n")
        f.write(" ".join(cmd) + "\n\n")
        f.write("=== STDOUT ===\n")
        f.write(proc.stdout or "")
        f.write("\n\n=== STDERR ===\n")
        f.write(proc.stderr or "")

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
        tail_text = "\n".join(tail)
        raise RuntimeError(
            f"test.py falló.\nRevisa el log: {log_file}\nÚltimas líneas:\n{tail_text}"
        )

    # =========================
    # BÚSQUEDA DE RESULTADO
    # =========================
    found = None
    for root, _, files in os.walk(log_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                found = Path(root) / f
                print(f"[INFO] Imagen final encontrada en: {found}")
                break
        if found:
            break

    if not found:
        raise RuntimeError(
            f"VITON-HD error: No se encontró ninguna imagen en {log_dir}. "
            f"Revisa el log {log_file} por si la lista quedó vacía."
        ) 

    target = RESULTS / run_id / "try-on" / "person.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.open(found).convert("RGB").save(target, quality=95, subsampling=0)

    cmask_path = (cmask_dir / "cloth.png")
    if not cmask_path.exists():
        cmask_path = (cmask_dir / "cloth.jpg") if (cmask_dir / "cloth.jpg").exists() else cmask_path

    garment_kind = _classify_garment(cmask_path)
    _restore_face_and_regions(target, person_dst, garment_kind, canvas=(768, 1024))

    return str(target) 
