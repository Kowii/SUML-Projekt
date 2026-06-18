"""Health-check route — used by Docker and load balancers to verify service liveness."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Return a simple healthy status so Docker healthchecks can detect readiness."""
    return {"status": "healthy"}
