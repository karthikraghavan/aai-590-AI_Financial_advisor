from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.services import (
    BASE_DIR,
    INPUT_USER_DATA,
    build_client_profile_from_excel,
    build_client_profile_from_payload,
    build_full_recommendation,
    generate_asset_allocation,
    generate_financial_plan,
    get_market_analysis,
)

app = FastAPI(
    title="AI Financial Advisor API",
    description="Inference endpoint (KMeans k9) and agentic RAG run (plan + allocation).",
    version="0.2.0",
)


class InferencePayload(BaseModel):
    monthly_income: float
    monthly_expense_total: float
    actual_savings: float
    debt_to_income_ratio: float
    investment_amount: float
    emergency_fund: float
    credit_score: float
    loan_payment: Optional[float] = 0.0
    budget_goal: Optional[float] = 0.0
    savings_rate: Optional[float] = None
    subscription_services: Optional[int] = 0
    income_type: Optional[str] = ""
    rent_or_mortgage: Optional[float] = 0.0
    name: Optional[str] = None
    model_name: Optional[str] = Field(
        default="kmeans_k3.pkl",
        description="Override model file name under results/. Defaults to kmeans_k9.pkl.",
    )

    class Config:
        extra = "allow"


class AgenticRequest(BaseModel):
    file_path: Optional[str] = Field(
        default=None, description="Excel file path (relative to project root or absolute) for profile creation."
    )
    payload: Optional[InferencePayload] = Field(default=None, description="Manual payload for profile creation.")
    market_data: Optional[Dict[str, Any]] = None
    use_live_market: bool = True
    model_name: Optional[str] = "kmeans_k9.pkl"

    class Config:
        extra = "allow"


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (BASE_DIR / path_str).resolve()
    return path


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/infer")
def infer_cluster(payload: InferencePayload) -> Dict[str, Any]:
    try:
        #profile = build_client_profile_from_payload(
         #   payload.dict(exclude={"model_name"}, exclude_none=True),
          #  model_name=payload.model_name,
        #)
        profile = build_client_profile_from_excel(
            INPUT_USER_DATA / "input.xlsx",
            model_name=payload.model_name,
        )  # For consistent feature engineering
        cluster = profile.get("cluster", {})
        persona = next(iter(cluster.values()), "Unknown")
        return {"cluster": cluster, "persona": persona, "client_profile": profile.get("client_profile")}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/agentic/run")
def agentic_rag(request: AgenticRequest) -> Dict[str, Any]:
    try:
        if request.payload:
            profile = build_client_profile_from_payload(
                request.payload.dict(exclude_none=True, exclude={"model_name"}),
                model_name=request.model_name,
            )
        elif request.file_path:
            path = _resolve_path(request.file_path)
            profile = build_client_profile_from_excel(path, model_name=request.model_name)
        else:
            raise HTTPException(status_code=400, detail="Provide either payload or file_path.")

        market = request.market_data or get_market_analysis(use_live=request.use_live_market)
        plan = generate_financial_plan(market, profile)
        allocation = generate_asset_allocation(market, profile)

        return {
            "client_profile": profile,
            "market": market,
            "financial_plan": plan,
            "allocation": allocation,
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/agentic/full-run")
def full_agentic_recommendation(request: AgenticRequest) -> Dict[str, Any]:
    try:
        if request.payload:
            profile = build_client_profile_from_payload(
                request.payload.dict(exclude_none=True, exclude={"model_name"}),
                model_name=request.model_name,
            )
        elif request.file_path:
            path = _resolve_path(request.file_path)
            profile = build_client_profile_from_excel(path, model_name=request.model_name)
        else:
            raise HTTPException(status_code=400, detail="Provide either payload or file_path.")

        return build_full_recommendation(
            client_profile=profile,
            market_data=request.market_data,
            use_live_market=request.use_live_market,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
