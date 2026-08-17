from fastapi import FastAPI

from app.api.requirements import router as requirements_router
from app.api.similarity import router as similarity_router

app = FastAPI(
    title="Cotizador AI2",
    version="0.1.0",
)

app.include_router(requirements_router)
app.include_router(similarity_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "cotizador-ai2",
    }
