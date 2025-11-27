from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "results"
VECTOR_STORE_DIR = BASE_DIR / "faiss_vector_store"
PROMPTS_DIR = DATA_DIR / "prompts"
DOCS_DIR = DATA_DIR / "docs"
INPUT_USER_DATA = DATA_DIR / "userdata"

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

SEGMENT_FEATURES = [
    "monthly_income",
    "monthly_expense_total",
    "actual_savings",
    "debt_to_income_ratio",
    "investment_amount",
    "emergency_fund",
    "credit_score",
]

SEGMENT_LABELS = {
    "0": "Financial Achievers",
    "1": "Steady Planners",
    "2": "Financially Stressed",
}

BEST_K = 3


class ProfileError(Exception):
    """Raised when profile processing fails."""


def get_openai_key() -> Optional[str]:
    api_key = os.getenv("OPEN_AI_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key.strip()
    return None


def _load_scaler() -> joblib:
    scaler_path = BASE_DIR / "robust_scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    return joblib.load(scaler_path)


def _load_kmeans(model_name: Optional[str] = None) -> joblib:
    name = model_name or f"kmeans_k{BEST_K}.pkl"
    model_path = MODEL_DIR / name
    if not model_path.exists():
        raise FileNotFoundError(f"KMeans model not found at {model_path}")
    return joblib.load(model_path)


def _safe_log1p(series: pd.Series) -> pd.Series:
    return np.log1p(np.where(series < 0, 0, series))


def _fix_outliers_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["actual_savings"] = _safe_log1p(df["actual_savings"])
    df["debt_to_income_ratio"] = np.clip(df["debt_to_income_ratio"], 0, 1.2)
    df["investment_amount"] = _safe_log1p(df["investment_amount"])
    df["credit_score"] = np.clip(df["credit_score"], 300, 850)
    return df


def _prepare_data_from_excel(user_data_path: Path) -> pd.DataFrame:
    if not user_data_path.exists():
        raise FileNotFoundError(f"User data file not found at {user_data_path}")

    userprofiledata = pd.read_excel(user_data_path)
    userprofiledata = userprofiledata.dropna(how="all")
    userprofiledata = userprofiledata.drop(userprofiledata.columns[:2], axis=1).iloc[1:].reset_index(drop=True)
    userprofiledata = userprofiledata.T
    userprofiledata.columns = userprofiledata.iloc[0]
    userprofiledata = userprofiledata.drop(userprofiledata.index[0]).reset_index(drop=True)

    userprofiledata = userprofiledata.rename(
        columns={
            "Name": "name",
            "Monthly Income": "monthly_income",
            "Monthly Expense": "monthly_expense_total",
            "Actual Savings": "actual_savings",
            "Budget Goal": "budget_goal",
            "Investment amount": "investment_amount",
            "Emergency Fund": "emergency_fund",
            "Credit Score": "credit_score",
            "Loan Payment": "loan_payment",
            "Debt to Income ratio": "debt_to_income_ratio",
            "Savings rate": "savings_rate",
            "# of Subscription services": "subscription_services",
            "Income Type": "income_type",
            "Rent or Mortgage": "rent_or_mortgage",
        }
    )

    data = userprofiledata[SEGMENT_FEATURES]
    data = data.apply(pd.to_numeric, errors="coerce")
    return data


def _prepare_dataframe_from_payload(payload: Dict[str, Any]) -> pd.DataFrame:
    row = {feature: payload.get(feature) for feature in SEGMENT_FEATURES}
    missing = [k for k, v in row.items() if v is None]
    if missing:
        raise ProfileError(f"Missing required numeric fields: {', '.join(missing)}")
    df = pd.DataFrame([row])
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def _infer_cluster_label(df: pd.DataFrame, model_name: Optional[str] = None) -> str:
    clean_df = _fix_outliers_for_inference(df)
    scaler = _load_scaler()
    scaled = scaler.transform(clean_df)
    scaled_df = pd.DataFrame(scaled, index=df.index, columns=df.columns)
    kmeans = _load_kmeans(model_name=model_name)
    print(model_name)
    print("KMeans names:", getattr(kmeans, "feature_names_in_", None))
    label = str(kmeans.predict(scaled_df)[0])
    return label


def build_client_profile_from_excel(path: Path, model_name: Optional[str] = None) -> Dict[str, Any]:
    processed_data = _prepare_data_from_excel(path)
    #logger.info("Processed data (from %s): %s", path, processed_data.to_dict(orient="records"))
    cluster_label = _infer_cluster_label(processed_data, model_name=model_name)
    profile = processed_data.to_dict(orient="records")
    return {"client_profile": profile, "cluster": {cluster_label: SEGMENT_LABELS.get(cluster_label, "Unknown")}}


def build_client_profile_from_payload(payload: Dict[str, Any], model_name: Optional[str] = None) -> Dict[str, Any]:
    processed_data = _prepare_dataframe_from_payload(payload)
    cluster_label = _infer_cluster_label(processed_data, model_name=model_name)
    profile = [payload]
    return {"client_profile": profile, "cluster": {cluster_label: SEGMENT_LABELS.get(cluster_label, "Unknown")}}


def get_ytd_return(ticker: str) -> Optional[float]:
    year_start = datetime(datetime.today().year, 1, 1)
    data = yf.download(ticker, start=year_start, progress=False)
    if len(data) < 2:
        return None
    first_open = float(data["Open"].iloc[0])
    last_close = float(data["Close"].iloc[-1])
    ytd_return = (last_close - first_open) / first_open * 100
    return round(ytd_return, 2)


def get_latest_close(ticker: str) -> Optional[float]:
    data = yf.Ticker(ticker).history(period="2d")
    if "Close" in data and len(data) > 0:
        return float(data["Close"].iloc[-1])
    return None


def get_market_analysis(use_live: bool = True) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "S&P 500 YTD (%)": None,
        "AGG ETF YTD (%)": None,
        "VIX Current": None,
    }
    if not use_live:
        return results
    try:
        results["S&P 500 YTD (%)"] = get_ytd_return("^GSPC")
        results["AGG ETF YTD (%)"] = get_ytd_return("AGG")
        results["VIX Current"] = get_latest_close("^VIX")
    except Exception as exc:  # noqa: BLE001
        results["error"] = f"Unable to fetch live market data: {exc}"
    return results


def _load_prompt(tag: str) -> str:
    path = PROMPTS_DIR / ("asset_allocation.txt" if tag == "AA" else "financial_planning.txt")
    if not path.exists():
        raise FileNotFoundError(f"Prompt file missing at {path}")
    return path.read_text()


def _document_path(tag: str) -> Path:
    return DOCS_DIR / (
        "PortfolioRecommendationMorningstar.pdf" if tag == "AA" else "Guide_to_Financial_Planning_Process.pdf"
    )


def _vector_store_path(tag: str) -> Path:
    # prefer prebuilt stores under faiss_vector_store; fallback to build in place
    name = "asset_allocation_vector_store" if tag == "AA" else "financial_planning_vector_store"
    candidate_paths: List[Path] = [
        DATA_DIR / "faiss_vector_store" / name,
        VECTOR_STORE_DIR / name,
    ]
    for path in candidate_paths:
        if path.exists():
            return path
    return candidate_paths[0]


def _load_vector_store(tag: str, embeddings: OpenAIEmbeddings) -> FAISS:
    store_path = _vector_store_path(tag)
    doc_path = _document_path(tag)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    if store_path.exists():
        return FAISS.load_local(str(store_path), embeddings, allow_dangerous_deserialization=True)

    loader = PyPDFLoader(str(doc_path))
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(store_path))
    return vector_store


def _normalize_client_profile(profile: Any) -> Dict[str, Any]:
    if isinstance(profile, str):
        try:
            return json.loads(profile)
        except json.JSONDecodeError as exc:
            raise ProfileError(f"Unable to parse client profile string: {exc}") from exc
    if isinstance(profile, dict):
        return profile
    raise ProfileError("Client profile must be a JSON string or dictionary.")


def _normalize_market(market: Any) -> Dict[str, Any]:
    if market is None:
        return {}
    if isinstance(market, str):
        try:
            return json.loads(market)
        except json.JSONDecodeError as exc:
            raise ProfileError(f"Unable to parse market data string: {exc}") from exc
    if isinstance(market, dict):
        return market
    raise ProfileError("Market data must be a JSON string or dictionary.")


def generate_prompt_response(market_json: Any, client_profile: Any, tag: str) -> Dict[str, Any]:
    profile_obj = _normalize_client_profile(client_profile)
    market_obj = _normalize_market(market_json)

    cluster_dict = profile_obj.get("cluster") or {}
    prompt_segment = list(cluster_dict.values())[0] if cluster_dict else "Unknown"

    api_key = get_openai_key()
    if not api_key:
        raise ValueError("OPEN_AI_KEY or OPENAI_API_KEY is not set.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    vector_store = _load_vector_store(tag, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 15})

    docs = retriever.invoke("portfolio recommendation" if tag == "AA" else "financial planning")
    context = "\n\n".join([d.page_content for d in docs])

    prompt_template = _load_prompt(tag)
    prompt = ChatPromptTemplate.from_template(prompt_template.strip())

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    chain = (
        {
            "context": lambda _: context,
            "prompt_profile": lambda _: json.dumps(profile_obj, indent=2),
            "prompt_segment": lambda _: prompt_segment,
            "prompt_market": lambda _: json.dumps(market_obj, indent=2),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    output = chain.invoke({})
    parsed: Optional[Any] = None
    try:
        parsed = json.loads(output)
    except Exception:
        parsed = None

    return {"raw": output, "parsed": parsed}


def generate_financial_plan(market_json: Any, client_profile: Any) -> Dict[str, Any]:
    return generate_prompt_response(market_json, client_profile, tag="FP")


def generate_asset_allocation(market_json: Any, client_profile: Any) -> Dict[str, Any]:
    return generate_prompt_response(market_json, client_profile, tag="AA")


def build_full_recommendation(
    client_profile: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None, use_live_market: bool = True
) -> Dict[str, Any]:
    market = market_data or get_market_analysis(use_live=use_live_market)
    financial_plan = generate_financial_plan(market, client_profile)
    allocation = generate_asset_allocation(market, client_profile)
    return {"client_profile": client_profile, "market": market, "financial_plan": financial_plan, "allocation": allocation}
