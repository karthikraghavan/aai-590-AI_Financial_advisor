import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import requests
import streamlit as st

# Simple Streamlit UI that calls the FastAPI backend to run agentic financial advice.
# Assumes the FastAPI server is running (default: http://localhost:8000).

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
APP_DIR = Path(__file__).resolve().parent
IMAGE_PATH = APP_DIR / "images" / "usd-logo-primary-thumb.png"

def compute_derived_fields(monthly_income: float, monthly_expense: float, loan_payment: float) -> Dict[str, float]:
    actual_savings = max(monthly_income - monthly_expense, 0)
    savings_rate = actual_savings / monthly_income if monthly_income else 0
    debt_to_income_ratio = loan_payment / monthly_income if monthly_income else 0
    return {
        "actual_savings": round(actual_savings, 2),
        "savings_rate": round(savings_rate, 4),
        "debt_to_income_ratio": round(debt_to_income_ratio, 4),
    }


def load_logo_base64() -> Optional[str]:
    logo_path = IMAGE_PATH
    with logo_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def make_allocation_pie(allocation: Any):
    parsed = allocation.get("parsed") if isinstance(allocation, dict) else allocation
    alloc_list: List[Any] = []
    if isinstance(parsed, dict):
        alloc_list = parsed.get("portfolio_recommendation") or parsed.get("allocation") or []
    elif isinstance(parsed, list):
        alloc_list = parsed

    labels: List[str] = []
    sizes: List[float] = []
    for row in alloc_list:
        if not isinstance(row, dict):
            continue
        pct = row.get("target_allocation_percentage") or row.get("target_allocation") or row.get("percentage")
        if pct is None:
            continue
        labels.append(row.get("asset_class") or row.get("asset") or "Asset")
        sizes.append(float(pct))

    if not sizes:
        return None

    fig, ax = plt.subplots(figsize=(2, 2), dpi=300)

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        radius=0.6,             
        textprops={"fontsize": 4}
    )

    for t in autotexts:
        t.set_fontsize(4)

    ax.set_aspect("equal")

    plt.tight_layout()          # keeps everything inside the box
    return fig

def post_agentic(payload: Dict[str, Any], base_url: str) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/agentic/finadvice"
    resp = requests.post(url, json={"payload": payload, "use_live_market": True}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _html_list(items: Dict[str, Any] | List[Any] | str) -> str:
    if isinstance(items, dict):
        lines = [f"<li><strong>{k}:</strong> {v}</li>" for k, v in items.items()]
    elif isinstance(items, list):
        lines = []
        for v in items:
            if isinstance(v, dict):
                # Render dict as key: value pairs in one line
                inner = "; ".join([f"{k}: {val}" for k, val in v.items()])
                lines.append(f"<li>{inner}</li>")
            else:
                lines.append(f"<li>{v}</li>")
    elif isinstance(items, str):
        return f"<p>{items}</p>"
    else:
        return "<p></p>"
    return f"<ul>{''.join(lines)}</ul>"


def _format_recommendations(advice: Any) -> List[str]:
    if isinstance(advice, list):
        return [
            "<br>".join([f"{k}: {v}" for k, v in rec.items()]) + "<br><br>"
            if isinstance(rec, dict) else str(rec)
            for rec in advice
        ]
    if isinstance(advice, dict) and "recommendations" in advice:
        return _format_recommendations(advice["recommendations"])
    if isinstance(advice, dict):
        return [f"{k}: {v}" for k, v in advice.items()]
    if isinstance(advice, str):
        return [advice]
    return []


def _format_allocation(alloc_table: Any) -> List[str]:
    formatted: List[str] = []
    if isinstance(alloc_table, list):
        for row in alloc_table:
            if isinstance(row, dict):
                asset = row.get("asset_class") or row.get("asset") or "Asset"
                pct = row.get("target_allocation_percentage") or row.get("target_allocation") or row.get("percentage")
                examples = row.get("example_instruments") or row.get("examples") or []
                examples_str = "; ".join(examples) if isinstance(examples, list) else str(examples)
                pct_str = f"{pct}%" if pct is not None else ""
                formatted.append(f"{asset} – {pct_str} ({examples_str})")
            else:
                formatted.append(str(row))
    elif isinstance(alloc_table, dict):
        formatted = [f"{k}: {v}" for k, v in alloc_table.items()]
    elif isinstance(alloc_table, str):
        formatted = [alloc_table]
    return formatted


def render_summary_html(
    name: Optional[str],
    client_profile: Dict[str, Any],
    financial_plan: Dict[str, Any],
    allocation: Dict[str, Any],
) -> str:
    title_name = name or "Client"
    segment = next(iter(client_profile.get("cluster", {}).values()), "N/A")
    profile_rows = client_profile.get("client_profile") or []
    profile_summary = profile_rows[0] if profile_rows else {}

    plan_parsed = financial_plan.get("parsed") or {}
    raw_advice = plan_parsed.get("Financial Advise") or plan_parsed.get("financial_advice") or plan_parsed or {}
    advice_list = _format_recommendations(raw_advice)

    alloc_parsed = allocation.get("parsed") or {}
    alloc_table = alloc_parsed.get("portfolio_recommendation") or alloc_parsed.get("allocation") or {}
    rationale = alloc_parsed.get("rationale", "")
    compliance = alloc_parsed.get("compliance_notes") or "Recommendations provided for informational purposes only."
    alloc_list = _format_allocation(alloc_table)

    html_parts = [
        f"<h2>{title_name}'s Financial Summary</h2>",
        f"<h4>Financial Segment</h4><p>{segment}</p>",
        "<h4>Client Profile Summary</h4>",
        _html_list(profile_summary),
        "<h4>Financial Advice</h4>",
        _html_list(advice_list),
        "<h4>Asset Allocation</h4>",
        _html_list(alloc_list),
        f"<p><em>Rationale:</em> {rationale}</p>" if rationale else "",
    ]
    return "\n".join([part for part in html_parts if part])

def render_compliance_notes(
        allocation: Dict[str, Any],
) -> str:
    alloc_parsed = allocation.get("parsed") or {}
    compliance = alloc_parsed.get("compliance_notes") or "Recommendations provided for informational purposes only."
    html_comp = f"<p style='font-size:0.9em;color:#555;'><strong>Compliance Notes:</strong> {compliance}</p>"
    return html_comp

def main() -> None:
    st.set_page_config(page_title="AI Financial Advisor", layout="wide")
    logo_b64 = load_logo_base64()
    header_html = "<h1 style='text-align:center; margin: 0;font-color:blue'>FinAdvisorAI - AI Driven Financial Advisor</h1>"
    if logo_b64:
        header_html = (
            "<div style='position:relative; padding-top:0; padding-bottom:10px;'>"
            f"<img src='data:image/png;base64,{logo_b64}' style='position:absolute; right:10px; top:-40px; height:120px;' />"
            f"{header_html}"
            "</div>"
        )

    st.markdown(header_html, unsafe_allow_html=True)
    st.caption("Enter client details. Derived fields (Actual Savings, Debt-to-Income, Savings Rate) are auto-calculated.")

    with st.sidebar:
        st.subheader("API settings")
        api_url = st.text_input("FastAPI base URL", value=API_BASE_URL)
        st.caption("Default assumes the API is running locally: http://localhost:8000")

    with st.form("client_form"):
        name = st.text_input("Name")
        monthly_income = st.number_input("Monthly Income", min_value=0.0, step=100.0)
        monthly_expense = st.number_input("Monthly Expense", min_value=0.0, step=100.0)
        budget_goal = st.number_input("Budget Goal", min_value=0.0, step=100.0)
        investment_amount = st.number_input("Investment amount", min_value=0.0, step=100.0)
        emergency_fund = st.number_input("Emergency Fund", min_value=0.0, step=100.0)
        credit_score = st.number_input("Credit Score", min_value=0, max_value=850, step=1)
        loan_payment = st.number_input("Loan Payment", min_value=0.0, step=50.0)
        subscription_services = st.number_input("# of Subscription services", min_value=0, step=1)
        income_type = st.text_input("Income Type", value="salary")
        rent_or_mortgage = st.number_input("Rent or Mortgage", min_value=0.0, step=100.0)

        derived = compute_derived_fields(monthly_income, monthly_expense, loan_payment)

        st.markdown(
            f"**Derived fields**  \n"
            f"- Actual Savings: {derived['actual_savings']}  \n"
            f"- Debt to Income ratio: {derived['debt_to_income_ratio']}  \n"
            f"- Savings rate: {derived['savings_rate']}"
        )

        submitted = st.form_submit_button("Generate Advice")

    if submitted:
        payload = {
            "name": name,
            "monthly_income": monthly_income,
            "monthly_expense_total": monthly_expense,
            "budget_goal": budget_goal,
            "investment_amount": investment_amount,
            "emergency_fund": emergency_fund,
            "credit_score": credit_score,
            "loan_payment": loan_payment,
            "subscription_services": int(subscription_services),
            "income_type": income_type,
            "rent_or_mortgage": rent_or_mortgage,
            # derived fields
            "actual_savings": derived["actual_savings"],
            "debt_to_income_ratio": derived["debt_to_income_ratio"],
            "savings_rate": derived["savings_rate"],
        }

        try:
            with st.spinner("Calling FastAPI and generating advice..."):
                response = post_agentic(payload, api_url)
            st.success("Advice generated.")
            html = render_summary_html(
                name=name,
                client_profile=response.get("client_profile", {}),
                financial_plan=response.get("financial_plan", {}),
                allocation=response.get("allocation", {}),
            )
            st.markdown(html, unsafe_allow_html=True)
            fig = make_allocation_pie(response.get("allocation", {}))
            if fig:
                st.pyplot(fig)
            html_comp = render_compliance_notes(
                allocation=response.get("allocation", {}),
            )
            st.markdown(html_comp, unsafe_allow_html=True)
            
        except requests.HTTPError as exc:
            st.error(f"API error: {exc.response.text if exc.response else exc}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to generate advice: {exc}")


if __name__ == "__main__":
    main()
