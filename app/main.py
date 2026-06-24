import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv() # Load .env 
import jwt
from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    required_env_vars = [
        "SUPABASE_URL",
        "SUPABASE_SERVICE_KEY",
        "SUPABASE_JWT_SECRET",
        "GEMINI_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME",
        "GROQ_API_KEY",
    ]

    missing = [var for var in required_env_vars if not os.getenv(var)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Check your .env file."
        )

    print("All environment variables loaded")
    print(f"Supabase: {os.getenv('SUPABASE_URL')}")
    print(f"Pinecone index: {os.getenv('PINECONE_INDEX_NAME')}")
    print("Medha API is ready")

    yield  # server runs here

    # ── Shutdown ──
    print("Medha API shutting down")


# ─── App instance ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Medha Dairy API",
    description="Backend for Medha — the personal AI diary companion",
    version="1.0.0",
    lifespan=lifespan,
    # Disable docs in production for security
    # Set ENVIRONMENT=production in Render env vars to disable
    docs_url=None if os.getenv("ENVIRONMENT") == "production" else "/docs",
    redoc_url=None if os.getenv("ENVIRONMENT") == "production" else "/redoc",
)


# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # TODO: restrict in production
    allow_credentials=True,
    allow_methods=["*"],        # GET, POST, PUT, DELETE etc.
    allow_headers=["*"],        # Authorization, Content-Type etc.
)


# ─── Routers ──────────────────────────────────────────────────────────────────

from app.routers import ingest, chat  # noqa: E402
from app.routers import ingest, chat, plan, memories as memories_router, people as people_router

app.include_router(people_router.router, 
                   prefix="/api",
                     tags=["People"])


app.include_router(
    ingest.router,
    prefix="/api",      # all ingest routes become /api/...
   tags=["Ingestion"], # groups them in the /docs Swagger UI
)

app.include_router(
    chat.router,
    prefix="/api",      # all chat routes become /api/...
    tags=["Chat"],
)

from app.routers import ingest, chat, plan   # add plan

app.include_router(
    plan.router,
    prefix="/api",
    tags=["Planning"],
)

from app.routers import ingest, chat, plan, memories as memories_router

app.include_router(
    memories_router.router,
    prefix="/api",
    tags=["Memories"],
)

from app.routers import astro as astro_router
app.include_router(astro_router.router, prefix="/api", tags=["Astro"])


from app.routers import payment as payments_router
app.include_router(payments_router.router, prefix="/api", tags=["Payments"])
#_____________Extented feature for add socisl diary feature____________________
from app.routers import social as social_router   # ← add this

app.include_router(social_router.router, prefix="/api", tags=["Social"])  # ← add this

# ─── Health check endpoints ───────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "service":     "Medha API",
        "status":      "ok",
        "version":     "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
    }

@app.post("/debug/token", tags=["Debug"])
async def debug_token(request: Request):
    
    body = await request.json()
    token = body.get("token", "")
    try:
        
        header  = jwt.get_unverified_header(token)
        payload = jwt.decode(token, options={"verify_signature": False})
        return {"header": header, "payload_keys": list(payload.keys())}
    except Exception as e:
        return {"error": str(e)}


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy"}