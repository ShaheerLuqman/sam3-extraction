from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("/runs")
def list_runs(request: Request) -> dict:
    """Every tracking run, newest first, without the prompt payloads."""
    runs = request.app.state.runs.list()
    return {"runs": runs, "count": len(runs)}


@router.get("/runs/{run_id}")
def get_run(request: Request, run_id: str) -> dict:
    """One run in full, including the instances that were prompted."""
    run = request.app.state.runs.get(run_id)
    if run is None:
        raise HTTPException(404, "unknown run")
    return run


@router.delete("/runs/{run_id}")
def delete_run(request: Request, run_id: str) -> dict:
    """Forget a run and delete the video/JSON it produced."""
    if not request.app.state.runs.delete(run_id):
        raise HTTPException(404, "unknown run")
    return {"deleted": True}
