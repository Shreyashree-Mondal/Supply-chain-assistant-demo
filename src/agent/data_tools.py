"""Data-backed LangChain tools: exact policy lookup + knowledge-base search."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
from langchain_core.tools import tool

from src import config
from src.llm.rag import Retriever

POLICY_COLS = ["Product Name", "Category Name", "Active_Status", "Data_Confidence", "Forecast_Method",
               "Latest_Forecast", "Monthly_Sigma", "Backtest_MAPE", "Lead_Time_Days", "Product Price",
               "Safety_Stock", "Reorder_Point", "EOQ", "Inventory_Risk", "ABC_Class", "Recommended_Action"]


RENAME = {"Latest_Forecast": "forecast_units_per_month", "Monthly_Sigma": "forecast_error_std_units_per_month",
          "Backtest_MAPE": "backtest_mape_pct", "Lead_Time_Days": "lead_time_days",
          "Product Price": "unit_price_usd", "Safety_Stock": "safety_stock_units",
          "Reorder_Point": "reorder_point_units", "EOQ": "eoq_units"}


@lru_cache(maxsize=1)
def _policy() -> pd.DataFrame:
    return pd.read_csv(config.POLICY_CSV)


@lru_cache(maxsize=1)
def _retriever() -> Retriever:
    return Retriever.load()


@tool
def lookup_product_policy(product_id: int) -> dict:
    """Look up one product's real inventory policy by its Product Card Id: forecast, safety stock,
    reorder point, EOQ, price, risk tier, ABC class, data confidence and recommended action.
    Use this for any question about a specific product."""
    df = _policy()
    row = df[df["Product Card Id"] == int(product_id)]
    if row.empty:
        return {"error": f"No product with id {product_id} in the policy table."}
    r = row.iloc[0]
    out = {"product_id": int(product_id)}
    for c in POLICY_COLS:
        v = r[c]
        out[RENAME.get(c, c)] = None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)
    return out


@tool
def search_knowledge_base(query: str) -> list:
    """Search the project knowledge base (inventory concepts, model notes, data summaries, assumptions)
    and return the top matching passages with their source ids. Use this to explain concepts or
    assumptions. Do NOT use it to get a specific product's numbers; use lookup_product_policy."""
    hits = _retriever().search(query, k=3)
    return [{"source": h["id"], "score": h["score"], "text": h["text"][:600]} for h in hits]


@tool
def find_product(name_query: str) -> list:
    """Find products whose name contains the given text. Returns up to 5 matches with their
    Product Card Id, name and status. Use this when the user names a product but gives no id."""
    df = _policy()
    m = df[df["Product Name"].str.contains(name_query, case=False, regex=False, na=False)].head(5)
    return [{"product_id": int(r["Product Card Id"]), "name": r["Product Name"], "status": r["Active_Status"]}
            for _, r in m.iterrows()]


@tool
def compare_replenishment_policies(product_id: int) -> dict:
    """Simulate 30 years of daily demand for one product and compare the current (reorder point, EOQ)
    policy with a tuned policy (reorder point x1.5, order quantity about EOQ), at normal demand and at
    +30% demand. Returns annual cost and fill rate for each. Results depend on an ASSUMED lost-sale cost
    of 25% of unit price per unit, which the project config does not define."""
    from src.optimization.rl_env import (InventoryEnv, STOCKOUT_COST_FRAC, evaluate, load_product, sq_policy)
    try:
        prod = load_product(int(product_id))
    except IndexError:
        return {"error": f"No product with id {product_id} in the policy table."}

    def tuned(env, obs):
        return 2 if env.on_hand + sum(env.pipeline) <= prod["rop"] * 1.5 else 0

    out = {"product_id": int(product_id), "assumed_lost_sale_cost_pct_of_price": STOCKOUT_COST_FRAC * 100}
    for label, scale in [("normal_demand", 1.0), ("demand_plus_30pct", 1.3)]:
        cur = evaluate(lambda: InventoryEnv(prod, demand_scale=scale), sq_policy, episodes=30)
        new = evaluate(lambda: InventoryEnv(prod, demand_scale=scale), tuned, episodes=30)
        out[label] = {"current_policy": {k: round(v, 3) for k, v in cur.items()},
                      "tuned_policy": {k: round(v, 3) for k, v in new.items()}}
    return out


@tool
def days_of_supply(units_on_hand: float, avg_daily_demand: float) -> float:
    """Days of supply = units on hand divided by average daily demand."""
    if avg_daily_demand <= 0:
        return float("nan")
    return round(units_on_hand / avg_daily_demand, 2)


@tool
def service_level_to_z(service_level_percent: float) -> float:
    """Convert a target cycle service level in percent (for example 95) to the normal z-score."""
    from statistics import NormalDist
    p = service_level_percent / 100.0
    if not 0 < p < 1:
        return float("nan")
    return round(NormalDist().inv_cdf(p), 3)


@tool
def holding_cost_per_unit(unit_price: float, annual_holding_rate_percent: float) -> float:
    """Annual holding cost per unit = unit price times the annual holding rate (given in percent, for example 20)."""
    return round(unit_price * annual_holding_rate_percent / 100.0, 4)
