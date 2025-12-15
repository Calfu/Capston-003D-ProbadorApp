from dotenv import load_dotenv
import os

# Cargar .env desde la ruta ABSOLUTA
env_path = os.path.join(os.path.dirname(__file__), ".env")
print("Cargando .env desde:", env_path)
load_dotenv(dotenv_path=env_path)

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pathlib import Path
from io import BytesIO
from PIL import Image
import tempfile, base64
from fastapi.staticfiles import StaticFiles



# 🔹 Importaciones de rutas
from app.routes import usuarios, productos, empleados, probador, carrito  # ⬅️ AÑADIDO carrito
from app.routes.assistant import router as assistant_router
# 🔹 Importación del motor de VITON-HD
from app.vitonhd_wrapper import run_viton_hd


# ============================================================
# 🚀 Inicialización de la aplicación
# ============================================================
app = FastAPI(title="Probador Virtual API")
app.mount("/imagenes", StaticFiles(directory="app/storage/imagenes"), name="imagenes")
app.include_router(assistant_router, prefix="/api")

# ============================================================
# 🌐 Configuración de CORS
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8081",
        "http://127.0.0.1:8081",
        "http://localhost:19006",
        "http://127.0.0.1:19006",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 🛰️ Middleware para registrar requests del frontend
# ============================================================
@app.middleware("http")
async def log_request(request: Request, call_next):
    if request.url.path.startswith("/usuarios/registro"):
        print("🛰️ Llego un request a /usuarios/registro")
        print("📦 Headers:", dict(request.headers))
        try:
            body = await request.json()
            print("📩 Body recibido bruto:", body)
        except Exception as e:
            print("⚠️ No se pudo leer el body:", e)
    response = await call_next(request)
    return response

# ============================================================
# 🔗 Registro de rutas
# ============================================================
app.include_router(usuarios.router)
app.include_router(productos.router)
app.include_router(empleados.router)
app.include_router(probador.router)
app.include_router(carrito.router)  # ⬅️ NUEVO

# ============================================================
# ❤️ Endpoint de verificación del servidor
# ============================================================
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

# ============================================================
# 👕 Endpoint del probador virtual (VITON-HD)
# ============================================================
@app.post("/tryon")
async def tryon(user: UploadFile = File(...), garment: UploadFile = File(...)):
    if not (user.content_type or "").startswith("image/") or not (garment.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Sube archivos de imagen válidos (JPEG/PNG/WEBP).")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # Guardar imágenes temporales
        user_img = Image.open(BytesIO(await user.read())).convert("RGB")
        garment_img = Image.open(BytesIO(await garment.read())).convert("RGBA")

        user_path = tmp / "user.jpg"
        garment_path = tmp / "garment.png"

        user_img.save(user_path, quality=95, subsampling=0)
        garment_img.save(garment_path)

        # Ejecutar modelo de VITON-HD
        out_path = run_viton_hd(str(user_path), str(garment_path))

        # Leer resultado y convertirlo a base64
        out_img = Image.open(out_path).convert("RGB")
        buf = BytesIO()
        out_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return JSONResponse({"result": b64})