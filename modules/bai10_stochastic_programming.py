import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

try:
    import pyomo.environ as pyo
    PYOMO_AVAILABLE = True
except Exception:
    PYOMO_AVAILABLE = False

try:
    import pulp
    PULP_AVAILABLE = True
except Exception:
    PULP_AVAILABLE = False


# =========================================================
# BÀI 10 — QUY HOẠCH NGẪU NHIÊN HAI GIAI ĐOẠN
# Two-stage Stochastic Programming under Uncertainty
# =========================================================


# ---------------------------------------------------------
# 0. STYLE + COMPONENTS
# ---------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 12px;
            border-radius: 14px;
        }
        .small-note {
            font-size: 14px;
            opacity: 0.84;
            line-height: 1.55;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def sticker_header(icon, title, subtitle):
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, rgba(14,165,233,0.18), rgba(139,92,246,0.14));
            padding: 18px 22px;
            border-radius: 18px;
            margin: 10px 0 18px 0;
            border: 1px solid rgba(255,255,255,0.08);
        ">
            <div style="font-size: 28px; font-weight: 850;">{icon} {title}</div>
            <div style="font-size: 16px; opacity: 0.88; margin-top: 6px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def policy_card(icon, title, body, tone="info"):
    colors = {
        "info": "#0ea5e9",
        "success": "#22c55e",
        "warning": "#f59e0b",
        "danger": "#ef4444",
        "purple": "#8b5cf6",
        "gray": "#64748b",
    }
    color = colors.get(tone, "#0ea5e9")

    st.markdown(
        f"""
        <div style="
            border-left: 6px solid {color};
            background: rgba(255,255,255,0.04);
            padding: 16px 18px;
            border-radius: 14px;
            margin-bottom: 12px;
        ">
            <div style="font-size: 22px; font-weight: 750; margin-bottom: 6px;">
                {icon} {title}
            </div>
            <div style="font-size: 16px; line-height: 1.62;">
                {body}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------
def get_items():
    return ["I", "D", "AI", "H"]


def get_item_names():
    return {
        "I": "Hạ tầng số",
        "D": "Chuyển đổi số",
        "AI": "Trí tuệ nhân tạo",
        "H": "Nhân lực số",
    }


def get_scenario_data():
    df = pd.DataFrame({
        "scenario": ["s1", "s2", "s3", "s4"],
        "scenario_name": ["Lạc quan", "Cơ sở", "Bi quan", "Khủng hoảng"],
        "world_growth_pct": [3.5, 2.8, 1.5, 0.2],
        "fdi_usd_billion": [32.0, 27.0, 20.0, 12.0],
        "export_growth_pct": [12.0, 8.0, 3.0, -5.0],
        "probability": [0.30, 0.45, 0.20, 0.05],
    })
    df["scenario_label"] = df["scenario"] + " — " + df["scenario_name"]
    return df


def get_beta_data():
    items = get_items()

    beta_base = {
        "I": 1.00,
        "D": 1.10,
        "AI": 1.25,
        "H": 0.95,
    }

    beta_s = {
        ("s1", "I"): 1.25, ("s1", "D"): 1.35, ("s1", "AI"): 1.55, ("s1", "H"): 1.05,
        ("s2", "I"): 1.00, ("s2", "D"): 1.10, ("s2", "AI"): 1.25, ("s2", "H"): 0.95,
        ("s3", "I"): 0.75, ("s3", "D"): 0.85, ("s3", "AI"): 0.90, ("s3", "H"): 1.00,
        ("s4", "I"): 0.40, ("s4", "D"): 0.50, ("s4", "AI"): 0.55, ("s4", "H"): 1.10,
    }

    beta_df = pd.DataFrame({
        "Hạng mục": ["I", "D", "AI", "H"],
        "Tên hạng mục": [get_item_names()[j] for j in items],
        "β cơ bản": [beta_base[j] for j in items],
        "s1 Lạc quan": [beta_s[("s1", j)] for j in items],
        "s2 Cơ sở": [beta_s[("s2", j)] for j in items],
        "s3 Bi quan": [beta_s[("s3", j)] for j in items],
        "s4 Khủng hoảng": [beta_s[("s4", j)] for j in items],
    })

    beta_long = []
    for s in ["s1", "s2", "s3", "s4"]:
        for j in items:
            beta_long.append({
                "scenario": s,
                "scenario_name": get_scenario_data().set_index("scenario").loc[s, "scenario_name"],
                "item": j,
                "item_name": get_item_names()[j],
                "beta": beta_s[(s, j)],
            })

    return beta_base, beta_s, beta_df, pd.DataFrame(beta_long)


def expected_beta_s():
    scen = get_scenario_data()
    _, beta_s, _, _ = get_beta_data()
    items = get_items()

    out = {}
    for j in items:
        out[j] = sum(
            scen.loc[scen["scenario"] == s, "probability"].iloc[0] * beta_s[(s, j)]
            for s in scen["scenario"]
        )
    return out


def expected_macro_indicators():
    scen = get_scenario_data()
    return {
        "world_growth_expected": float((scen["world_growth_pct"] * scen["probability"]).sum()),
        "fdi_expected": float((scen["fdi_usd_billion"] * scen["probability"]).sum()),
        "export_growth_expected": float((scen["export_growth_pct"] * scen["probability"]).sum()),
        "prob_bad": float(scen.loc[scen["scenario"].isin(["s3", "s4"]), "probability"].sum()),
    }


# ---------------------------------------------------------
# 2. SOLVER HELPERS
# ---------------------------------------------------------
def available_pyomo_solver():
    if not PYOMO_AVAILABLE:
        return None

    for name in ["glpk", "cbc"]:
        try:
            solver = pyo.SolverFactory(name)
            if solver.available(False):
                return name
        except Exception:
            pass

    return None


def solve_with_pulp(
    model_type="SP",
    fixed_x=None,
    scenario_single=None,
    use_expected_beta=False,
    first_budget=65000,
    recourse_budget=15000,
    total_budget=80000,
    item_cap_share=0.55,
    min_H_first=0,
    min_DAI_first=0,
    ai_link=0.5,
    robust=False,
    scenario_optimum=None,
):
    """
    model_type:
    - SP: stochastic program with scenario-specific y[s,j]
    - EV: deterministic expected-value model, one recourse vector y[j]
    - WS: wait-and-see for one scenario
    - FIXED_X_EVAL: evaluate fixed first-stage x under stochastic recourse
    - ROBUST_REGRET: minimize maximum regret
    """

    if not PULP_AVAILABLE:
        return None

    items = get_items()
    scen = get_scenario_data()
    beta_base, beta_s, _, _ = get_beta_data()
    beta_ev = expected_beta_s()
    scenarios = scen["scenario"].tolist()
    p = dict(zip(scen["scenario"], scen["probability"]))

    model = pulp.LpProblem(f"VN_Two_Stage_{model_type}", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("x", items, lowBound=0, cat="Continuous")

    if model_type in ["EV", "WS"]:
        y = pulp.LpVariable.dicts("y", items, lowBound=0, cat="Continuous")
    else:
        y = pulp.LpVariable.dicts("y", (scenarios, items), lowBound=0, cat="Continuous")

    # First-stage constraints
    if fixed_x is None:
        model += pulp.lpSum(x[j] for j in items) <= first_budget, "C1_first_stage_budget"
        for j in items:
            model += x[j] <= item_cap_share * first_budget, f"C2_item_cap_first_{j}"
        if min_H_first > 0:
            model += x["H"] >= min_H_first, "C3_min_H_first_stage"
        if min_DAI_first > 0:
            model += x["D"] + x["AI"] >= min_DAI_first, "C4_min_DAI_first_stage"
    else:
        for j in items:
            model += x[j] == fixed_x.get(j, 0.0), f"FIX_x_{j}"

    # Recourse constraints
    if model_type == "EV":
        model += pulp.lpSum(y[j] for j in items) <= recourse_budget, "R1_recourse_budget_EV"
        model += y["AI"] <= ai_link * x["H"], "R2_AI_recourse_link_EV"

    elif model_type == "WS":
        model += pulp.lpSum(y[j] for j in items) <= recourse_budget, f"R1_recourse_budget_{scenario_single}"
        model += y["AI"] <= ai_link * x["H"], f"R2_AI_recourse_link_{scenario_single}"

    else:
        for s in scenarios:
            model += pulp.lpSum(y[s][j] for j in items) <= recourse_budget, f"R1_recourse_budget_{s}"
            model += y[s]["AI"] <= ai_link * x["H"], f"R2_AI_recourse_link_{s}"

    # Objective
    first_value = pulp.lpSum(beta_base[j] * x[j] for j in items)

    if model_type == "EV":
        second_value = pulp.lpSum(beta_ev[j] * y[j] for j in items)
        model += first_value + second_value, "OBJ_expected_value"

    elif model_type == "WS":
        second_value = pulp.lpSum(beta_s[(scenario_single, j)] * y[j] for j in items)
        model += first_value + second_value, f"OBJ_wait_and_see_{scenario_single}"

    elif robust:
        # For robust regret: convert to minimization of R, done by maximizing -R.
        R = pulp.LpVariable("max_regret", lowBound=0, cat="Continuous")

        for s in scenarios:
            scenario_value = first_value + pulp.lpSum(beta_s[(s, j)] * y[s][j] for j in items)
            model += R >= scenario_optimum[s] - scenario_value, f"REGRET_{s}"

        model += -R, "OBJ_minimize_max_regret"

    else:
        second_value = pulp.lpSum(
            p[s] * pulp.lpSum(beta_s[(s, j)] * y[s][j] for j in items)
            for s in scenarios
        )
        model += first_value + second_value, "OBJ_stochastic_expected"

    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    status = pulp.LpStatus[model.status]

    if status != "Optimal":
        return {
            "status": status,
            "objective": np.nan,
            "x": {},
            "y": {},
            "details": pd.DataFrame(),
            "constraints": pd.DataFrame(),
            "scenario_values": pd.DataFrame(),
        }

    x_val = {j: float(pulp.value(x[j])) for j in items}

    y_val = {}
    if model_type in ["EV", "WS"]:
        y_val = {j: float(pulp.value(y[j])) for j in items}
    else:
        y_val = {
            s: {j: float(pulp.value(y[s][j])) for j in items}
            for s in scenarios
        }

    # Details
    rows = []
    if model_type in ["EV", "WS"]:
        if model_type == "EV":
            sname = "EV"
            beta_used = beta_ev
        else:
            sname = scenario_single
            beta_used = {j: beta_s[(scenario_single, j)] for j in items}

        for j in items:
            rows.append({
                "stage": "First-stage",
                "scenario": sname,
                "item": j,
                "item_name": get_item_names()[j],
                "budget": x_val[j],
                "beta": beta_base[j],
                "value": beta_base[j] * x_val[j],
            })
            rows.append({
                "stage": "Second-stage",
                "scenario": sname,
                "item": j,
                "item_name": get_item_names()[j],
                "budget": y_val[j],
                "beta": beta_used[j],
                "value": beta_used[j] * y_val[j],
            })
    else:
        for j in items:
            rows.append({
                "stage": "First-stage",
                "scenario": "all",
                "item": j,
                "item_name": get_item_names()[j],
                "budget": x_val[j],
                "beta": beta_base[j],
                "value": beta_base[j] * x_val[j],
            })
        for s in scenarios:
            for j in items:
                rows.append({
                    "stage": "Second-stage",
                    "scenario": s,
                    "item": j,
                    "item_name": get_item_names()[j],
                    "budget": y_val[s][j],
                    "beta": beta_s[(s, j)],
                    "value": beta_s[(s, j)] * y_val[s][j],
                })

    details = pd.DataFrame(rows)

    # Scenario values for stochastic-style x,y
    scenario_rows = []
    if model_type in ["SP", "FIXED_X_EVAL", "ROBUST_REGRET"] or robust:
        for s in scenarios:
            first_v = sum(beta_base[j] * x_val[j] for j in items)
            second_v = sum(beta_s[(s, j)] * y_val[s][j] for j in items)
            scenario_rows.append({
                "scenario": s,
                "scenario_name": scen.set_index("scenario").loc[s, "scenario_name"],
                "probability": p[s],
                "first_stage_value": first_v,
                "second_stage_value": second_v,
                "total_value_if_s": first_v + second_v,
                "weighted_value": p[s] * (first_v + second_v),
            })
    elif model_type == "EV":
        first_v = sum(beta_base[j] * x_val[j] for j in items)
        second_v = sum(beta_ev[j] * y_val[j] for j in items)
        scenario_rows.append({
            "scenario": "EV",
            "scenario_name": "Expected-value deterministic",
            "probability": 1.0,
            "first_stage_value": first_v,
            "second_stage_value": second_v,
            "total_value_if_s": first_v + second_v,
            "weighted_value": first_v + second_v,
        })
    elif model_type == "WS":
        first_v = sum(beta_base[j] * x_val[j] for j in items)
        second_v = sum(beta_s[(scenario_single, j)] * y_val[j] for j in items)
        scenario_rows.append({
            "scenario": scenario_single,
            "scenario_name": scen.set_index("scenario").loc[scenario_single, "scenario_name"],
            "probability": 1.0,
            "first_stage_value": first_v,
            "second_stage_value": second_v,
            "total_value_if_s": first_v + second_v,
            "weighted_value": first_v + second_v,
        })

    scenario_values = pd.DataFrame(scenario_rows)

    # Constraints table
    constraints_rows = []
    for name, cons in model.constraints.items():
        constraints_rows.append({
            "constraint": name,
            "slack": cons.slack,
            "shadow_price": cons.pi,
            "binding?": abs(cons.slack) <= 1e-5,
        })
    constraints = pd.DataFrame(constraints_rows)

    objective_value = float(pulp.value(model.objective))

    # For robust, report max regret as positive.
    if robust:
        try:
            R_val = float(pulp.value(model.variablesDict()["max_regret"]))
        except Exception:
            R_val = np.nan
        objective_value = -objective_value
    else:
        R_val = np.nan

    return {
        "status": status,
        "objective": objective_value,
        "max_regret": R_val,
        "x": x_val,
        "y": y_val,
        "details": details,
        "constraints": constraints,
        "scenario_values": scenario_values,
    }


def solve_with_pyomo_if_available(
    model_type="SP",
    fixed_x=None,
    scenario_single=None,
    use_expected_beta=False,
    first_budget=65000,
    recourse_budget=15000,
    item_cap_share=0.55,
    min_H_first=0,
    min_DAI_first=0,
    ai_link=0.5,
):
    """
    Pyomo implementation for SP/EV/WS/FIXED_X_EVAL.
    If solver unavailable, caller should use PuLP fallback.
    """

    solver_name = available_pyomo_solver()
    if not PYOMO_AVAILABLE or solver_name is None:
        return None

    items = get_items()
    scen = get_scenario_data()
    beta_base, beta_s, _, _ = get_beta_data()
    beta_ev = expected_beta_s()
    scenarios = scen["scenario"].tolist()
    p = dict(zip(scen["scenario"], scen["probability"]))

    if model_type in ["EV", "WS"]:
        scenarios_model = ["EV"] if model_type == "EV" else [scenario_single]
    else:
        scenarios_model = scenarios

    m = pyo.ConcreteModel()
    m.J = pyo.Set(initialize=items)
    m.S = pyo.Set(initialize=scenarios_model)

    m.beta = pyo.Param(m.J, initialize=beta_base)

    if model_type == "EV":
        beta_second = {("EV", j): beta_ev[j] for j in items}
        prob_second = {"EV": 1.0}
    elif model_type == "WS":
        beta_second = {(scenario_single, j): beta_s[(scenario_single, j)] for j in items}
        prob_second = {scenario_single: 1.0}
    else:
        beta_second = {(s, j): beta_s[(s, j)] for s in scenarios for j in items}
        prob_second = p

    m.beta_s = pyo.Param(m.S, m.J, initialize=beta_second)
    m.p = pyo.Param(m.S, initialize=prob_second)

    m.x = pyo.Var(m.J, within=pyo.NonNegativeReals)
    m.y = pyo.Var(m.S, m.J, within=pyo.NonNegativeReals)

    if fixed_x is None:
        m.first_budget = pyo.Constraint(expr=sum(m.x[j] for j in m.J) <= first_budget)

        def item_cap_rule(m, j):
            return m.x[j] <= item_cap_share * first_budget
        m.item_cap = pyo.Constraint(m.J, rule=item_cap_rule)

        if min_H_first > 0:
            m.min_H = pyo.Constraint(expr=m.x["H"] >= min_H_first)
        if min_DAI_first > 0:
            m.min_DAI = pyo.Constraint(expr=m.x["D"] + m.x["AI"] >= min_DAI_first)
    else:
        def fixed_x_rule(m, j):
            return m.x[j] == fixed_x.get(j, 0.0)
        m.fixed_x = pyo.Constraint(m.J, rule=fixed_x_rule)

    def budget2_rule(m, s):
        return sum(m.y[s, j] for j in m.J) <= recourse_budget
    m.budget2 = pyo.Constraint(m.S, rule=budget2_rule)

    def ai_link_rule(m, s):
        return m.y[s, "AI"] <= ai_link * m.x["H"]
    m.ai_link = pyo.Constraint(m.S, rule=ai_link_rule)

    def obj_rule(m):
        first = sum(m.beta[j] * m.x[j] for j in m.J)
        second = sum(
            m.p[s] * sum(m.beta_s[s, j] * m.y[s, j] for j in m.J)
            for s in m.S
        )
        return first + second

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    solver = pyo.SolverFactory(solver_name)
    res = solver.solve(m, tee=False)

    status = str(res.solver.termination_condition)

    if "optimal" not in status.lower():
        return {
            "status": status,
            "objective": np.nan,
            "x": {},
            "y": {},
            "details": pd.DataFrame(),
            "constraints": pd.DataFrame(),
            "scenario_values": pd.DataFrame(),
        }

    x_val = {j: float(pyo.value(m.x[j])) for j in items}
    y_val = {
        s: {j: float(pyo.value(m.y[s, j])) for j in items}
        for s in scenarios_model
    }

    rows = []
    for j in items:
        rows.append({
            "stage": "First-stage",
            "scenario": "all",
            "item": j,
            "item_name": get_item_names()[j],
            "budget": x_val[j],
            "beta": beta_base[j],
            "value": beta_base[j] * x_val[j],
        })

    for s in scenarios_model:
        for j in items:
            rows.append({
                "stage": "Second-stage",
                "scenario": s,
                "item": j,
                "item_name": get_item_names()[j],
                "budget": y_val[s][j],
                "beta": beta_second[(s, j)],
                "value": beta_second[(s, j)] * y_val[s][j],
            })

    details = pd.DataFrame(rows)

    scenario_rows = []
    for s in scenarios_model:
        first_v = sum(beta_base[j] * x_val[j] for j in items)
        second_v = sum(beta_second[(s, j)] * y_val[s][j] for j in items)
        scenario_rows.append({
            "scenario": s,
            "scenario_name": "Expected-value" if s == "EV" else scen.set_index("scenario").loc[s, "scenario_name"],
            "probability": prob_second[s],
            "first_stage_value": first_v,
            "second_stage_value": second_v,
            "total_value_if_s": first_v + second_v,
            "weighted_value": prob_second[s] * (first_v + second_v),
        })

    scenario_values = pd.DataFrame(scenario_rows)

    return {
        "status": "Optimal",
        "objective": float(pyo.value(m.obj)),
        "x": x_val,
        "y": y_val if model_type not in ["EV", "WS"] else y_val[scenarios_model[0]],
        "details": details,
        "constraints": pd.DataFrame(),
        "scenario_values": scenario_values,
        "solver": solver_name,
    }


def solve_model(
    model_type="SP",
    fixed_x=None,
    scenario_single=None,
    first_budget=65000,
    recourse_budget=15000,
    item_cap_share=0.55,
    min_H_first=0,
    min_DAI_first=0,
    ai_link=0.5,
    prefer_pyomo=True,
):
    if prefer_pyomo and model_type in ["SP", "EV", "WS", "FIXED_X_EVAL"]:
        pyomo_res = solve_with_pyomo_if_available(
            model_type=model_type,
            fixed_x=fixed_x,
            scenario_single=scenario_single,
            first_budget=first_budget,
            recourse_budget=recourse_budget,
            item_cap_share=item_cap_share,
            min_H_first=min_H_first,
            min_DAI_first=min_DAI_first,
            ai_link=ai_link,
        )
        if pyomo_res is not None:
            return pyomo_res

    return solve_with_pulp(
        model_type=model_type,
        fixed_x=fixed_x,
        scenario_single=scenario_single,
        first_budget=first_budget,
        recourse_budget=recourse_budget,
        item_cap_share=item_cap_share,
        min_H_first=min_H_first,
        min_DAI_first=min_DAI_first,
        ai_link=ai_link,
    )


def evaluate_fixed_x_stochastic(
    x_fixed,
    first_budget=65000,
    recourse_budget=15000,
    item_cap_share=0.55,
    min_H_first=0,
    min_DAI_first=0,
    ai_link=0.5,
):
    return solve_model(
        model_type="FIXED_X_EVAL",
        fixed_x=x_fixed,
        first_budget=first_budget,
        recourse_budget=recourse_budget,
        item_cap_share=item_cap_share,
        min_H_first=min_H_first,
        min_DAI_first=min_DAI_first,
        ai_link=ai_link,
        prefer_pyomo=True,
    )


def solve_all_metrics(
    first_budget=65000,
    recourse_budget=15000,
    item_cap_share=0.55,
    min_H_first=0,
    min_DAI_first=0,
    ai_link=0.5,
):
    scenarios = get_scenario_data()["scenario"].tolist()
    scen_probs = dict(zip(get_scenario_data()["scenario"], get_scenario_data()["probability"]))

    sp = solve_model(
        model_type="SP",
        first_budget=first_budget,
        recourse_budget=recourse_budget,
        item_cap_share=item_cap_share,
        min_H_first=min_H_first,
        min_DAI_first=min_DAI_first,
        ai_link=ai_link,
    )

    ev = solve_model(
        model_type="EV",
        first_budget=first_budget,
        recourse_budget=recourse_budget,
        item_cap_share=item_cap_share,
        min_H_first=min_H_first,
        min_DAI_first=min_DAI_first,
        ai_link=ai_link,
    )

    eev = evaluate_fixed_x_stochastic(
        ev["x"],
        first_budget=first_budget,
        recourse_budget=recourse_budget,
        item_cap_share=item_cap_share,
        min_H_first=min_H_first,
        min_DAI_first=min_DAI_first,
        ai_link=ai_link,
    )

    ws_results = {}
    scenario_optimum = {}
    for s in scenarios:
        ws = solve_model(
            model_type="WS",
            scenario_single=s,
            first_budget=first_budget,
            recourse_budget=recourse_budget,
            item_cap_share=item_cap_share,
            min_H_first=min_H_first,
            min_DAI_first=min_DAI_first,
            ai_link=ai_link,
        )
        ws_results[s] = ws
        scenario_optimum[s] = ws["objective"]

    ws_expected = sum(scen_probs[s] * ws_results[s]["objective"] for s in scenarios)

    vss = sp["objective"] - eev["objective"]
    evpi = ws_expected - sp["objective"]

    robust = solve_with_pulp(
        model_type="ROBUST_REGRET",
        first_budget=first_budget,
        recourse_budget=recourse_budget,
        item_cap_share=item_cap_share,
        min_H_first=min_H_first,
        min_DAI_first=min_DAI_first,
        ai_link=ai_link,
        robust=True,
        scenario_optimum=scenario_optimum,
    )

    return {
        "SP": sp,
        "EV": ev,
        "EEV": eev,
        "WS": ws_results,
        "WS_expected": ws_expected,
        "VSS": vss,
        "EVPI": evpi,
        "ROBUST": robust,
        "scenario_optimum": scenario_optimum,
    }


def x_to_dataframe(x_dict, label):
    items = get_items()
    return pd.DataFrame({
        "solution": label,
        "item": items,
        "item_name": [get_item_names()[j] for j in items],
        "budget": [x_dict.get(j, 0.0) for j in items],
    })


def y_to_dataframe(y_obj, label):
    items = get_items()

    if isinstance(y_obj, dict) and all(j in y_obj for j in items):
        return pd.DataFrame({
            "solution": label,
            "scenario": "single",
            "item": items,
            "item_name": [get_item_names()[j] for j in items],
            "budget": [y_obj.get(j, 0.0) for j in items],
        })

    rows = []
    for s, ydict in y_obj.items():
        for j in items:
            rows.append({
                "solution": label,
                "scenario": s,
                "item": j,
                "item_name": get_item_names()[j],
                "budget": ydict.get(j, 0.0),
            })
    return pd.DataFrame(rows)


def scenario_value_table_for_solution(solution, solution_name):
    vals = solution["scenario_values"].copy()
    vals["solution"] = solution_name
    return vals


# ---------------------------------------------------------
# 3. SECTIONS
# ---------------------------------------------------------
def show_context():
    st.header("10.1. Bối cảnh Việt Nam")

    sticker_header(
        "🌏🇻🇳",
        "Ra quyết định ngân sách số khi tương lai chưa chắc chắn",
        "Việt Nam có độ mở thương mại rất cao, nên chính sách đầu tư số 2026–2030 phải tính đến cầu xuất khẩu, FDI, tăng trưởng thế giới và cú sốc địa - chính trị."
    )

    policy_card(
        "🎯",
        "Vấn đề chính sách trung tâm",
        "Chính phủ phải quyết định cơ cấu ngân sách 5 năm ngay từ đầu kỳ, nhưng chưa biết kinh tế thế giới sẽ lạc quan, cơ sở, bi quan hay khủng hoảng. Quy hoạch ngẫu nhiên hai giai đoạn giúp tách quyết định “làm ngay” và quyết định “điều chỉnh sau khi biết kịch bản”.",
        "success"
    )

    policy_card(
        "🧭",
        "Tư duy two-stage",
        "Giai đoạn 1 là quyết định here-and-now: phân bổ 65.000 tỷ VND cho I, D, AI, H. Giai đoạn 2 là quyết định wait-and-see: dùng 15.000 tỷ VND dự phòng để điều chỉnh theo từng kịch bản.",
        "purple"
    )

    policy_card(
        "⚖️",
        "Đánh đổi chính sách",
        "Nếu đầu tư quá mạnh vào AI ngay từ đầu, hiệu quả cao trong kịch bản lạc quan nhưng dễ thiếu khả năng chống chịu khi khủng hoảng. Nếu đầu tư nhiều hơn vào H, hiệu quả ngắn hạn có thể thấp hơn nhưng khả năng thích ứng và hấp thụ cú sốc tốt hơn.",
        "warning"
    )

    scen = get_scenario_data()
    macro = expected_macro_indicators()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Độ mở thương mại", "≈180%", "xuất nhập khẩu/GDP")
    c2.metric("FDI kỳ vọng", f"{macro['fdi_expected']:.1f}", "tỷ USD/năm")
    c3.metric("Xuất khẩu kỳ vọng", f"{macro['export_growth_expected']:.1f}%", "tăng trưởng")
    c4.metric("Xác suất xấu", f"{macro['prob_bad']*100:.0f}%", "bi quan + khủng hoảng")

    st.subheader("Bảng 10.1 — Bốn kịch bản kinh tế toàn cầu tác động tới Việt Nam")
    st.dataframe(scen[[
        "scenario", "scenario_name", "world_growth_pct",
        "fdi_usd_billion", "export_growth_pct", "probability"
    ]].round(3), use_container_width=True)

    st.subheader("Ảnh 10.1 — Scenario tree: quyết định trước, điều chỉnh sau")

    labels = [
        "Năm 0: quyết định x",
        "s1 Lạc quan",
        "s2 Cơ sở",
        "s3 Bi quan",
        "s4 Khủng hoảng",
        "y_s1 điều chỉnh",
        "y_s2 điều chỉnh",
        "y_s3 điều chỉnh",
        "y_s4 điều chỉnh",
        "GDP gain kỳ vọng",
    ]

    fig_tree = go.Figure(data=[go.Sankey(
        node=dict(
            label=labels,
            pad=18,
            thickness=18,
            line=dict(color="black", width=0.35),
        ),
        link=dict(
            source=[0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            target=[1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9],
            value=[30, 45, 20, 5, 30, 45, 20, 5, 30, 45, 20, 5],
        )
    )])
    fig_tree.update_layout(
        title="Ảnh 10.1 — Cây kịch bản hai giai đoạn trong hoạch định ngân sách số",
        height=560
    )
    st.plotly_chart(fig_tree, use_container_width=True)

    st.subheader("Ảnh 10.2 — FDI và tăng trưởng xuất khẩu theo xác suất kịch bản")

    fig_macro = px.scatter(
        scen,
        x="fdi_usd_billion",
        y="export_growth_pct",
        size="probability",
        color="world_growth_pct",
        hover_name="scenario_label",
        text="scenario",
        title="Kịch bản nào tạo môi trường thuận lợi hoặc rủi ro cho đầu tư số?",
        labels={
            "fdi_usd_billion": "FDI Việt Nam, tỷ USD/năm",
            "export_growth_pct": "Tăng trưởng xuất khẩu, %",
            "world_growth_pct": "Tăng trưởng thế giới, %",
            "probability": "Xác suất",
        }
    )
    fig_macro.update_traces(textposition="top center")
    fig_macro.update_layout(height=520)
    st.plotly_chart(fig_macro, use_container_width=True)

    st.info(
        "Thông điệp mở đầu: bài toán không hỏi “kịch bản nào chắc chắn xảy ra”, mà hỏi “nên chuẩn bị ngân sách thế nào để nếu kịch bản nào xảy ra thì vẫn có khả năng thích ứng”."
    )


def show_scenario_tree():
    st.header("10.2. Cấu trúc kịch bản")

    sticker_header(
        "🌲🎲",
        "Scenario Tree: xác suất không thay thế dự báo, nhưng giúp ra quyết định tốt hơn",
        "Mỗi kịch bản kết hợp tăng trưởng thế giới, FDI vào Việt Nam và tăng trưởng xuất khẩu. Xác suất dùng để tính kỳ vọng trong mô hình SP."
    )

    scen = get_scenario_data()

    st.dataframe(scen[[
        "scenario", "scenario_name", "world_growth_pct",
        "fdi_usd_billion", "export_growth_pct", "probability"
    ]].round(3), use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        fig_prob = px.pie(
            scen,
            names="scenario_label",
            values="probability",
            hole=0.42,
            title="Ảnh 10.3 — Cơ cấu xác suất kịch bản"
        )
        fig_prob.update_layout(height=470)
        st.plotly_chart(fig_prob, use_container_width=True)

    with c2:
        scen_long = scen.melt(
            id_vars=["scenario", "scenario_name", "probability"],
            value_vars=["world_growth_pct", "fdi_usd_billion", "export_growth_pct"],
            var_name="Chỉ báo",
            value_name="Giá trị"
        )
        scen_long["Chỉ báo"] = scen_long["Chỉ báo"].replace({
            "world_growth_pct": "Tăng trưởng TG (%)",
            "fdi_usd_billion": "FDI VN (tỷ USD)",
            "export_growth_pct": "Xuất khẩu VN tăng (%)",
        })

        fig_bar = px.bar(
            scen_long,
            x="scenario_name",
            y="Giá trị",
            color="Chỉ báo",
            barmode="group",
            title="Ảnh 10.4 — Ba tín hiệu vĩ mô trong từng kịch bản"
        )
        fig_bar.update_layout(height=470)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Ảnh 10.5 — Bản đồ rủi ro kịch bản")

    fig_risk = px.scatter(
        scen,
        x="world_growth_pct",
        y="export_growth_pct",
        size="fdi_usd_billion",
        color="probability",
        hover_name="scenario_label",
        text="scenario",
        title="Tăng trưởng thế giới thấp + xuất khẩu giảm là vùng rủi ro chính sách",
        labels={
            "world_growth_pct": "Tăng trưởng thế giới, %",
            "export_growth_pct": "Tăng trưởng xuất khẩu Việt Nam, %",
            "fdi_usd_billion": "FDI, tỷ USD",
            "probability": "Xác suất",
        }
    )
    fig_risk.update_traces(textposition="top center")
    fig_risk.update_layout(height=520)
    st.plotly_chart(fig_risk, use_container_width=True)

    scenario_logic = pd.DataFrame({
        "Kịch bản": ["s1 Lạc quan", "s2 Cơ sở", "s3 Bi quan", "s4 Khủng hoảng"],
        "Tư duy chính sách": [
            "Có thể đẩy mạnh AI và D vì cầu, FDI và xuất khẩu thuận lợi.",
            "Duy trì cân bằng giữa AI, D và H.",
            "Cần tăng khả năng thích ứng; H và D quan trọng hơn.",
            "Ưu tiên H như hàng hóa bảo hiểm, giảm phụ thuộc vào AI mở rộng nhanh.",
        ],
        "Rủi ro nếu quyết định sai": [
            "Bỏ lỡ cơ hội tăng tốc nếu quá thận trọng.",
            "Thiếu linh hoạt nếu chỉ theo kế hoạch trung bình.",
            "AI-heavy có thể kém hiệu quả khi cầu yếu.",
            "Thiếu H làm giảm khả năng chuyển đổi việc làm và phục hồi sau sốc.",
        ],
    })

    st.subheader("Bảng 10.2 — Diễn giải chính sách của từng kịch bản")
    st.dataframe(scenario_logic, use_container_width=True)


def show_math_model():
    st.header("10.3. Mô hình toán học")

    sticker_header(
        "🧮🎯",
        "Two-stage stochastic programming: quyết định tốt không chỉ sinh lợi, mà còn giữ quyền điều chỉnh",
        "Mô hình tách phần chắc chắn phải quyết định ngay và phần linh hoạt điều chỉnh sau khi kịch bản được quan sát."
    )

    st.subheader("1️⃣ Giai đoạn 1 — Here-and-now decision")

    st.latex(r"""
    x = (x_I, x_D, x_{AI}, x_H)
    """)

    st.latex(r"""
    \sum_{j \in J} x_j \leq 65{,}000,\quad x_j \geq 0
    """)

    st.markdown("""
    `x_j` là ngân sách giai đoạn đầu cho hạng mục `j`, đơn vị **tỷ VND**.  
    Mô hình giả định tổng ngân sách 5 năm là 80.000 tỷ VND, trong đó 65.000 tỷ VND được phân bổ trước,
    còn 15.000 tỷ VND giữ làm dự phòng điều chỉnh.
    """)

    st.subheader("2️⃣ Giai đoạn 2 — Wait-and-see recourse decision")

    st.latex(r"""
    y_s = (y^s_I, y^s_D, y^s_{AI}, y^s_H),\quad s \in S
    """)

    st.latex(r"""
    \sum_{j \in J} y^s_j \leq 15{,}000,\quad y^s_j \geq 0,\quad \forall s \in S
    """)

    st.markdown("""
    `y^s_j` là ngân sách bổ sung cho hạng mục `j` nếu kịch bản `s` xảy ra.
    Mỗi kịch bản có một quyết định recourse riêng, vì sau khi biết trạng thái thế giới, Chính phủ có thể điều chỉnh linh hoạt.
    """)

    st.subheader("3️⃣ Ràng buộc hấp thụ AI bằng nhân lực")

    st.latex(r"""
    y^s_{AI} \leq 0.5x_H,\quad \forall s \in S
    """)

    policy_card(
        "👩‍💻🤖",
        "Ý nghĩa của ràng buộc AI-H",
        "Đầu tư bổ sung vào AI trong tương lai không thể mở rộng vô hạn nếu giai đoạn đầu không đầu tư đủ vào nhân lực số. Đây là ràng buộc năng lực hấp thụ: muốn AI chạy được, phải có người vận hành, dữ liệu, kỹ năng và quản trị.",
        "purple"
    )

    st.subheader("4️⃣ Dạng tổng quát của mô hình two-stage")

    st.latex(r"""
    \min \; c'x + \sum_{s \in S}p_s Q(x,s)
    """)

    st.latex(r"""
    Q(x,s)=
    \min \{q'y_s:\;T_sx+Wy_s=h_s,\;y_s\geq 0\}
    """)

    st.markdown("""
    Đây là dạng chuẩn của quy hoạch ngẫu nhiên hai giai đoạn. Trong bài này, ta dùng dạng đơn giản hóa theo hướng
    **tối đa hóa GDP gain kỳ vọng**, vì mục tiêu học tập là hiểu first-stage, recourse, VSS và EVPI.
    """)

    st.subheader("5️⃣ Dạng tính toán dùng trong dashboard")

    st.latex(r"""
    \max
    \sum_{j \in J}\beta_jx_j
    +
    \sum_{s \in S}p_s
    \left[
    \sum_{j \in J}\beta^s_jy^s_j
    \right]
    """)

    st.latex(r"""
    \text{s.t.}\quad
    \sum_jx_j\leq 65{,}000,\quad
    \sum_jy^s_j\leq 15{,}000,\quad
    y^s_{AI}\leq 0.5x_H,\quad
    x_j,y^s_j\geq 0
    """)

    st.markdown("""
    Vì các biến chỉ xuất hiện tuyến tính trong mục tiêu và ràng buộc, bài toán là **linear programming**.
    Do đó có thể giải bằng Pyomo + GLPK/CBC hoặc PuLP/CBC. Module ưu tiên Pyomo nếu môi trường có solver;
    nếu không có GLPK/CBC cho Pyomo, dashboard tự dùng PuLP/CBC để bảo đảm chạy được.
    """)

    variable_df = pd.DataFrame({
        "Ký hiệu": [
            "J", "S", "x_j", "y^s_j", "p_s", "β_j", "β^s_j",
            "65.000", "15.000", "y^s_AI ≤ 0.5x_H"
        ],
        "Ý nghĩa": [
            "Tập hạng mục đầu tư: I, D, AI, H",
            "Tập kịch bản: s1, s2, s3, s4",
            "Quyết định đầu tư giai đoạn 1",
            "Quyết định điều chỉnh giai đoạn 2 theo kịch bản",
            "Xác suất kịch bản",
            "Hệ số hiệu quả cơ bản của đầu tư ban đầu",
            "Hệ số hiệu quả đầu tư bổ sung trong kịch bản s",
            "Trần ngân sách first-stage, tỷ VND",
            "Trần ngân sách recourse mỗi kịch bản, tỷ VND",
            "Khả năng mở rộng AI phụ thuộc nhân lực số ban đầu",
        ],
        "Vai trò trong mô hình": [
            "Không gian quyết định chính sách",
            "Không gian bất định",
            "Here-and-now",
            "Wait-and-see",
            "Tính kỳ vọng",
            "Lợi ích chắc chắn hơn",
            "Lợi ích phụ thuộc trạng thái thế giới",
            "Cam kết kế hoạch đầu kỳ",
            "Quỹ dự phòng linh hoạt",
            "Ràng buộc năng lực hấp thụ AI",
        ],
    })

    st.subheader("Bảng 10.3 — Chú giải biến và ràng buộc")
    st.dataframe(variable_df, use_container_width=True)

    st.subheader("6️⃣ Các chỉ số đánh giá chất lượng quyết định")

    c1, c2 = st.columns(2)

    with c1:
        st.latex(r"""
        VSS = z_{SP} - z_{EEV}
        """)
        st.markdown("""
        **VSS** đo lợi ích của việc giải bài toán ngẫu nhiên thay vì dùng lời giải trung bình rồi đem đi áp dụng cho thế giới bất định.
        """)

    with c2:
        st.latex(r"""
        EVPI = z_{WS} - z_{SP}
        """)
        st.markdown("""
        **EVPI** đo giá trị của thông tin hoàn hảo: nếu biết trước chắc chắn kịch bản nào xảy ra, ta có thể cải thiện bao nhiêu.
        """)

    policy_card(
        "🔎",
        "Điểm cần nhớ",
        "VSS trả lời: “Có đáng tư duy theo xác suất không?” EVPI trả lời: “Thông tin dự báo hoàn hảo đáng giá bao nhiêu?” Hai chỉ số này rất phù hợp với hoạch định chính sách Việt Nam khi đối mặt COVID-19, bão Yagi, biến động FDI và chuỗi cung ứng.",
        "success"
    )


def show_beta_table():
    st.header("10.4. Bảng hệ số β theo kịch bản")

    sticker_header(
        "🔥🧊",
        "Hệ số β thay đổi theo trạng thái thế giới",
        "AI và D có hiệu quả cao trong kịch bản thuận lợi, nhưng H trở thành hạng mục có tính bảo hiểm trong khủng hoảng."
    )

    beta_base, beta_s, beta_df, beta_long = get_beta_data()

    st.dataframe(beta_df.round(3), use_container_width=True)

    st.subheader("Ảnh 10.6 — Heatmap β theo kịch bản và hạng mục")

    heat = beta_long.pivot(index="item_name", columns="scenario_name", values="beta")
    fig_heat = px.imshow(
        heat,
        text_auto=".2f",
        aspect="auto",
        title="Hệ số hiệu quả β^s_j: hạng mục nào mạnh trong từng kịch bản?"
    )
    fig_heat.update_layout(height=520)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Ảnh 10.7 — β của từng hạng mục thay đổi qua kịch bản")

    fig_line = px.line(
        beta_long,
        x="scenario_name",
        y="beta",
        color="item_name",
        markers=True,
        title="AI mạnh khi thuận lợi; H chống chịu tốt hơn trong khủng hoảng"
    )
    fig_line.update_layout(height=500, xaxis_title="Kịch bản", yaxis_title="β theo kịch bản")
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("Ảnh 10.8 — So sánh β cơ bản và β kỳ vọng")

    beta_ev = expected_beta_s()
    compare = pd.DataFrame({
        "item": get_items(),
        "item_name": [get_item_names()[j] for j in get_items()],
        "β cơ bản": [beta_base[j] for j in get_items()],
        "β kỳ vọng theo xác suất": [beta_ev[j] for j in get_items()],
    })

    compare_long = compare.melt(
        id_vars=["item", "item_name"],
        value_vars=["β cơ bản", "β kỳ vọng theo xác suất"],
        var_name="Loại β",
        value_name="β"
    )

    fig_compare = px.bar(
        compare_long,
        x="item_name",
        y="β",
        color="Loại β",
        barmode="group",
        text="β",
        title="So sánh hiệu quả cơ bản và hiệu quả kỳ vọng"
    )
    fig_compare.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_compare.update_layout(height=480, xaxis_title="Hạng mục")
    st.plotly_chart(fig_compare, use_container_width=True)

    insight_df = pd.DataFrame({
        "Hạng mục": ["I", "D", "AI", "H"],
        "Mẫu hình β": [
            "Giảm mạnh trong khủng hoảng",
            "Tốt khi thuận lợi, yếu khi khủng hoảng",
            "Cao nhất khi lạc quan, giảm mạnh khi khủng hoảng",
            "Ổn định hơn và cao nhất trong khủng hoảng",
        ],
        "Hàm ý chính sách": [
            "Hạ tầng số cần đầu tư nhưng không nên là công cụ duy nhất chống sốc.",
            "Chuyển đổi số phụ thuộc cầu và năng lực doanh nghiệp.",
            "AI có upside cao nhưng rủi ro chu kỳ lớn.",
            "Nhân lực số là hàng hóa bảo hiểm, giúp hấp thụ cú sốc và chuyển đổi việc làm.",
        ],
    })

    st.subheader("Bảng 10.4 — Diễn giải chính sách của β")
    st.dataframe(insight_df, use_container_width=True)

    policy_card(
        "🛡️",
        "Vì sao β_H cao trong khủng hoảng?",
        "Khi kinh tế khủng hoảng, cầu yếu và FDI giảm, các dự án AI hoặc hạ tầng mới có thể kém hiệu quả hơn. Ngược lại, nhân lực qua đào tạo có thể chuyển đổi việc làm, vận hành công nghệ linh hoạt hơn và giúp nền kinh tế phục hồi nhanh hơn.",
        "warning"
    )


def show_programming_solution():
    st.header("10.5. Giải yêu cầu lập trình")

    sticker_header(
        "💻📊",
        "Từ Pyomo/PuLP đến VSS, EVPI và Robust Regret",
        "Phần này giải SP, EV, WS và Robust Optimization; sau đó kiểm tra ràng buộc, so sánh first-stage decision và diễn giải giá trị của bất định."
    )

    if not PULP_AVAILABLE and not PYOMO_AVAILABLE:
        st.error("Cần cài `pulp` hoặc `pyomo` để chạy Bài 10.")
        return None

    st.subheader("Thiết lập tham số mô hình")

    c1, c2, c3, c4 = st.columns(4)

    first_budget = c1.number_input(
        "First-stage budget, tỷ VND",
        min_value=30000,
        max_value=80000,
        value=65000,
        step=5000,
        key="bai10_first_budget"
    )

    recourse_budget = c2.number_input(
        "Recourse budget/kịch bản, tỷ VND",
        min_value=5000,
        max_value=30000,
        value=15000,
        step=2500,
        key="bai10_recourse_budget"
    )

    item_cap_share = c3.slider(
        "Trần mỗi hạng mục first-stage",
        min_value=0.25,
        max_value=1.00,
        value=0.55,
        step=0.05,
        key="bai10_item_cap"
    )

    ai_link = c4.slider(
        "Hệ số hấp thụ AI: y_AI ≤ λx_H",
        min_value=0.10,
        max_value=1.00,
        value=0.50,
        step=0.05,
        key="bai10_ai_link"
    )

    c5, c6 = st.columns(2)

    min_H_first = c5.number_input(
        "Sàn H first-stage, tỷ VND",
        min_value=0,
        max_value=int(first_budget),
        value=0,
        step=1000,
        key="bai10_min_h"
    )

    min_DAI_first = c6.number_input(
        "Sàn D+AI first-stage, tỷ VND",
        min_value=0,
        max_value=int(first_budget),
        value=0,
        step=1000,
        key="bai10_min_dai"
    )

    solver_note = "Pyomo + GLPK/CBC" if available_pyomo_solver() else "PuLP/CBC fallback"
    st.info(f"Solver đang dùng: **{solver_note}**. Nếu Streamlit Cloud chưa có GLPK/CBC cho Pyomo, module tự dùng PuLP/CBC để vẫn chạy được.")

    if min_H_first + min_DAI_first > first_budget:
        st.error("Mô hình có nguy cơ không khả thi: sàn H + sàn D+AI lớn hơn ngân sách first-stage.")
        return None

    with st.spinner("Đang giải SP, EV, WS và Robust Regret..."):
        metrics = solve_all_metrics(
            first_budget=first_budget,
            recourse_budget=recourse_budget,
            item_cap_share=item_cap_share,
            min_H_first=min_H_first,
            min_DAI_first=min_DAI_first,
            ai_link=ai_link,
        )

    sp = metrics["SP"]
    ev = metrics["EV"]
    eev = metrics["EEV"]
    robust = metrics["ROBUST"]

    if sp["status"] != "Optimal":
        st.error(f"SP không tối ưu. Trạng thái: {sp['status']}")
        return None

    # -----------------------------------------------------
    # 10.5.1
    # -----------------------------------------------------
    st.subheader("Câu 10.5.1 — Mô hình SP hai giai đoạn và first-stage tối ưu")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("SP objective", f"{sp['objective']:,.2f}")
    k2.metric("First-stage used", f"{sum(sp['x'].values()):,.0f}", "tỷ VND")
    k3.metric("VSS", f"{metrics['VSS']:,.2f}")
    k4.metric("EVPI", f"{metrics['EVPI']:,.2f}")

    x_sp_df = x_to_dataframe(sp["x"], "SP")
    st.markdown("#### Quyết định first-stage tối ưu của SP")
    st.dataframe(x_sp_df.round(3), use_container_width=True)

    fig_x_sp = px.bar(
        x_sp_df,
        x="item_name",
        y="budget",
        color="item_name",
        text="budget",
        title="Ảnh 10.9 — First-stage decision x của stochastic program"
    )
    fig_x_sp.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_x_sp.update_layout(height=470, xaxis_title="Hạng mục", yaxis_title="Tỷ VND", showlegend=False)
    st.plotly_chart(fig_x_sp, use_container_width=True)

    st.markdown("#### Recourse decision y theo từng kịch bản")
    y_sp_df = y_to_dataframe(sp["y"], "SP")
    st.dataframe(y_sp_df.round(3), use_container_width=True)

    fig_y_sp = px.bar(
        y_sp_df,
        x="scenario",
        y="budget",
        color="item_name",
        barmode="stack",
        text="budget",
        title="Ảnh 10.10 — Recourse y_s: điều chỉnh sau khi biết kịch bản"
    )
    fig_y_sp.update_layout(height=520, xaxis_title="Kịch bản", yaxis_title="Tỷ VND")
    st.plotly_chart(fig_y_sp, use_container_width=True)

    st.markdown("#### Kiểm tra ràng buộc và shadow price")
    if sp["constraints"].empty:
        st.warning("Mô hình chạy bằng Pyomo nên bảng shadow price có thể không hiển thị trên Streamlit Cloud. Kết quả tối ưu vẫn được báo cáo.")
    else:
        st.dataframe(sp["constraints"].round(5), use_container_width=True)

    # -----------------------------------------------------
    # 10.5.2
    # -----------------------------------------------------
    st.subheader("Câu 10.5.2 — So sánh EV, SP và nghiệm xác định từng kịch bản")

    x_ev_df = x_to_dataframe(ev["x"], "EV")
    x_robust_df = x_to_dataframe(robust["x"], "Robust regret")

    ws_x_frames = []
    for s, obj in metrics["WS"].items():
        ws_x_frames.append(x_to_dataframe(obj["x"], f"WS {s}"))
    x_ws_df = pd.concat(ws_x_frames, ignore_index=True)

    x_compare = pd.concat([x_sp_df, x_ev_df, x_robust_df, x_ws_df], ignore_index=True)

    st.dataframe(x_compare.round(3), use_container_width=True)

    fig_compare_x = px.bar(
        x_compare,
        x="item_name",
        y="budget",
        color="solution",
        barmode="group",
        title="Ảnh 10.11 — So sánh first-stage decision: SP, EV, WS và Robust"
    )
    fig_compare_x.update_layout(height=560, xaxis_title="Hạng mục", yaxis_title="Tỷ VND")
    st.plotly_chart(fig_compare_x, use_container_width=True)

    scenario_value_rows = []
    scenario_value_rows.append(scenario_value_table_for_solution(sp, "SP"))
    scenario_value_rows.append(scenario_value_table_for_solution(eev, "EEV — EV fixed x"))
    scenario_values = pd.concat(scenario_value_rows, ignore_index=True)

    st.markdown("#### Giá trị theo kịch bản khi áp dụng SP và EV fixed-x")
    st.dataframe(scenario_values.round(3), use_container_width=True)

    fig_sv = px.bar(
        scenario_values,
        x="scenario",
        y="total_value_if_s",
        color="solution",
        barmode="group",
        text="total_value_if_s",
        title="Ảnh 10.12 — Nếu kịch bản xảy ra, SP và EV fixed-x tạo giá trị bao nhiêu?"
    )
    fig_sv.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_sv.update_layout(height=500)
    st.plotly_chart(fig_sv, use_container_width=True)

    # -----------------------------------------------------
    # 10.5.3
    # -----------------------------------------------------
    st.subheader("Câu 10.5.3 — Tính VSS và EVPI")

    summary_df = pd.DataFrame({
        "Chỉ số": [
            "SP — Stochastic Program",
            "EV — Expected Value deterministic",
            "EEV — EV solution evaluated under uncertainty",
            "WS expected — Perfect information",
            "VSS = SP - EEV",
            "EVPI = WS - SP",
        ],
        "Giá trị": [
            sp["objective"],
            ev["objective"],
            eev["objective"],
            metrics["WS_expected"],
            metrics["VSS"],
            metrics["EVPI"],
        ],
        "Ý nghĩa": [
            "Lời giải xét toàn bộ phân phối kịch bản",
            "Lời giải dùng kịch bản trung bình",
            "Giá trị thực tế nếu dùng x_EV trong thế giới bất định",
            "Giá trị kỳ vọng nếu biết trước hoàn hảo kịch bản",
            "Lợi ích của tư duy xác suất",
            "Giá trị tối đa của thông tin hoàn hảo",
        ]
    })

    st.dataframe(summary_df.round(3), use_container_width=True)

    fig_metrics = px.bar(
        summary_df,
        x="Chỉ số",
        y="Giá trị",
        text="Giá trị",
        title="Ảnh 10.13 — SP, EEV, WS và hai chỉ số VSS/EVPI"
    )
    fig_metrics.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_metrics.update_layout(height=520, xaxis_tickangle=-20)
    st.plotly_chart(fig_metrics, use_container_width=True)

    if metrics["VSS"] > 1e-5:
        policy_card(
            "✅",
            "VSS dương",
            f"VSS = {metrics['VSS']:,.2f}. Điều này cho thấy việc xét phân phối xác suất của kịch bản tạo giá trị tốt hơn so với chỉ dùng một kịch bản trung bình.",
            "success"
        )
    else:
        policy_card(
            "ℹ️",
            "VSS gần bằng 0",
            "Trong cấu hình tuyến tính hiện tại, lời giải EV và SP có thể trùng nhau. Đây không phải lỗi; nó cho thấy bộ ràng buộc và hệ số đang làm cho quyết định trung bình đủ đại diện. Hãy tăng sàn H, giảm trần hạng mục hoặc thay đổi hệ số AI-H để thấy VSS rõ hơn.",
            "info"
        )

    policy_card(
        "🔮",
        "EVPI là trần giá trị của thông tin dự báo",
        f"EVPI = {metrics['EVPI']:,.2f}. Nếu EVPI lớn, Chính phủ có lý do đầu tư vào năng lực dự báo, dữ liệu sớm, cảnh báo rủi ro chuỗi cung ứng và phân tích kịch bản.",
        "purple"
    )

    # -----------------------------------------------------
    # 10.5.4
    # -----------------------------------------------------
    st.subheader("Câu 10.5.4 — Robust optimization: cực tiểu hóa regret xấu nhất")

    robust_summary = pd.DataFrame([
        {
            "Mô hình": "SP",
            "Objective/Expected value": sp["objective"],
            "Max regret": np.nan,
            "x_I": sp["x"]["I"],
            "x_D": sp["x"]["D"],
            "x_AI": sp["x"]["AI"],
            "x_H": sp["x"]["H"],
        },
        {
            "Mô hình": "Robust regret",
            "Objective/Expected value": np.nan,
            "Max regret": robust["max_regret"],
            "x_I": robust["x"].get("I", np.nan),
            "x_D": robust["x"].get("D", np.nan),
            "x_AI": robust["x"].get("AI", np.nan),
            "x_H": robust["x"].get("H", np.nan),
        }
    ])

    st.dataframe(robust_summary.round(3), use_container_width=True)

    robust_x = pd.concat([
        x_to_dataframe(sp["x"], "SP"),
        x_to_dataframe(robust["x"], "Robust regret"),
    ], ignore_index=True)

    fig_robust = px.bar(
        robust_x,
        x="item_name",
        y="budget",
        color="solution",
        barmode="group",
        text="budget",
        title="Ảnh 10.14 — SP và Robust regret khác nhau ở cơ cấu first-stage nào?"
    )
    fig_robust.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_robust.update_layout(height=500, xaxis_title="Hạng mục", yaxis_title="Tỷ VND")
    st.plotly_chart(fig_robust, use_container_width=True)

    policy_card(
        "🛡️",
        "Robust regret là tư duy chống hối tiếc",
        "SP tối đa hóa giá trị kỳ vọng. Robust regret chọn quyết định sao cho nếu kịch bản xấu xảy ra, mức hối tiếc so với quyết định hoàn hảo trong kịch bản đó là nhỏ nhất có thể. Đây là tư duy phù hợp khi Nhà nước muốn giảm rủi ro chính trị - xã hội của quyết định sai.",
        "warning"
    )

    return metrics


def show_policy_discussion():
    st.header("10.6. Câu hỏi thảo luận chính sách")

    sticker_header(
        "🧭🇻🇳",
        "Từ VSS, EVPI đến bài học chính sách cho Việt Nam",
        "Phần này chuyển kết quả mô hình thành lập luận chính sách: có nên đầu tư thêm vào nhân lực số, dữ liệu dự báo và năng lực chống chịu?"
    )

    if not PULP_AVAILABLE and not PYOMO_AVAILABLE:
        st.error("Cần cài `pulp` hoặc `pyomo` để chạy phần thảo luận.")
        return

    metrics = solve_all_metrics(
        first_budget=65000,
        recourse_budget=15000,
        item_cap_share=0.55,
        min_H_first=0,
        min_DAI_first=0,
        ai_link=0.5,
    )

    sp = metrics["SP"]
    ev = metrics["EV"]
    robust = metrics["ROBUST"]

    # -----------------------------------------------------
    # Câu a
    # -----------------------------------------------------
    st.subheader("a) So với lời giải xác định, SP đầu tư H nhiều hơn hay ít hơn? Vì sao?")

    x_compare = pd.concat([
        x_to_dataframe(sp["x"], "SP"),
        x_to_dataframe(ev["x"], "EV"),
        x_to_dataframe(robust["x"], "Robust regret"),
    ], ignore_index=True)

    st.dataframe(x_compare.round(3), use_container_width=True)

    fig_a = px.bar(
        x_compare,
        x="item_name",
        y="budget",
        color="solution",
        barmode="group",
        text="budget",
        title="Minh chứng câu a — So sánh đầu tư H giữa SP, EV và Robust"
    )
    fig_a.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_a.update_layout(height=500, xaxis_title="Hạng mục", yaxis_title="Tỷ VND")
    st.plotly_chart(fig_a, use_container_width=True)

    h_sp = sp["x"].get("H", 0.0)
    h_ev = ev["x"].get("H", 0.0)
    h_robust = robust["x"].get("H", 0.0)

    if h_sp > h_ev + 1e-5:
        direction = "nhiều hơn"
        tone = "success"
    elif h_sp < h_ev - 1e-5:
        direction = "ít hơn"
        tone = "warning"
    else:
        direction = "xấp xỉ bằng"
        tone = "info"

    policy_card(
        "👩‍💻",
        f"SP đầu tư H {direction} EV",
        f"Trong cấu hình mặc định, H_SP = {h_sp:,.0f} tỷ VND, H_EV = {h_ev:,.0f} tỷ VND. Nếu SP đầu tư H nhiều hơn, nguyên nhân là H giúp mở khóa recourse AI và chống chịu trong kịch bản xấu. Nếu bằng nhau, điều đó cho thấy ràng buộc và hệ số hiện tại chưa đủ mạnh để làm H khác biệt giữa SP và EV.",
        tone
    )

    st.markdown("""
    **Diễn giải sâu:**  
    Trong mô hình này, H có hai vai trò. Thứ nhất, H tạo lợi ích trực tiếp thông qua β_H. Thứ hai, H là điều kiện để mở rộng AI ở giai đoạn hai qua ràng buộc `y_AI^s ≤ 0.5x_H`.  
    Vì vậy, H không chỉ là “đào tạo”, mà là **quyền chọn chính sách**: đầu tư H hôm nay giúp Chính phủ có khả năng triển khai AI linh hoạt hơn ngày mai.
    """)

    st.info(
        "Gợi ý khi thuyết trình: nếu kết quả SP và EV quá giống nhau, hãy chạy thêm kịch bản có sàn H hoặc trần hạng mục thấp hơn. Khi đó vai trò bảo hiểm của H thường hiện rõ hơn."
    )

    # -----------------------------------------------------
    # Câu b
    # -----------------------------------------------------
    st.subheader("b) VSS dương nói lên điều gì về tư duy xác suất trong hoạch định chính sách Việt Nam?")

    vss = metrics["VSS"]
    evpi = metrics["EVPI"]

    summary = pd.DataFrame({
        "Chỉ số": ["SP", "EEV", "VSS", "WS expected", "EVPI"],
        "Giá trị": [
            sp["objective"],
            metrics["EEV"]["objective"],
            vss,
            metrics["WS_expected"],
            evpi,
        ],
        "Câu hỏi chính sách": [
            "Nếu xét toàn bộ phân phối kịch bản thì đạt bao nhiêu?",
            "Nếu dùng lời giải trung bình rồi đem áp dụng vào thế giới bất định thì đạt bao nhiêu?",
            "Tư duy xác suất tạo thêm bao nhiêu giá trị?",
            "Nếu biết trước hoàn hảo kịch bản thì đạt bao nhiêu?",
            "Thông tin hoàn hảo đáng giá tối đa bao nhiêu?",
        ]
    })

    st.dataframe(summary.round(3), use_container_width=True)

    fig_b = px.bar(
        summary,
        x="Chỉ số",
        y="Giá trị",
        text="Giá trị",
        title="Minh chứng câu b — VSS và EVPI trong hoạch định chính sách"
    )
    fig_b.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_b.update_layout(height=480)
    st.plotly_chart(fig_b, use_container_width=True)

    if vss > 1e-5:
        policy_card(
            "✅",
            "VSS dương là bằng chứng cho tư duy xác suất",
            f"VSS = {vss:,.2f} cho thấy việc ra quyết định dựa trên phân phối xác suất tốt hơn việc lấy một kịch bản trung bình. Với Việt Nam, điều này ủng hộ cách hoạch định chính sách theo nhiều kịch bản: cơ sở, AI-centric, xanh hóa, khủng hoảng.",
            "success"
        )
    else:
        policy_card(
            "ℹ️",
            "VSS gần 0 vẫn có ý nghĩa",
            "VSS gần 0 cho thấy trong cấu hình tuyến tính đơn giản, lời giải trung bình và lời giải ngẫu nhiên có thể trùng nhau. Đây là một kết quả kiểm tra mô hình quan trọng, không phải lỗi. Khi thêm ràng buộc năng lực, sàn H hoặc rủi ro khủng hoảng mạnh hơn, VSS thường trở nên rõ hơn.",
            "info"
        )

    st.markdown("""
    **Hàm ý chính sách:**  
    Tư duy xác suất giúp Việt Nam tránh ba lỗi phổ biến trong hoạch định:
    
    1. **Lỗi trung bình:** chỉ thiết kế chính sách cho kịch bản cơ sở, bỏ qua đuôi rủi ro.
    2. **Lỗi phản ứng chậm:** không giữ dự phòng hoặc năng lực điều chỉnh khi cú sốc xảy ra.
    3. **Lỗi đầu tư lệch:** tập trung vào hạng mục có hiệu quả cao trong thuận lợi, nhưng yếu trong khủng hoảng.
    """)

    # -----------------------------------------------------
    # Câu c
    # -----------------------------------------------------
    st.subheader("c) COVID-19 và bão Yagi cho thấy Việt Nam có dưới đầu tư vào H như hàng hóa bảo hiểm không?")

    scen = get_scenario_data()
    _, _, beta_df, beta_long = get_beta_data()

    crisis_beta = beta_long[beta_long["scenario"] == "s4"].sort_values("beta", ascending=False)

    st.dataframe(crisis_beta[["item_name", "beta"]].round(3), use_container_width=True)

    fig_c1 = px.bar(
        crisis_beta,
        x="item_name",
        y="beta",
        color="item_name",
        text="beta",
        title="Minh chứng câu c — Trong khủng hoảng, H có β cao nhất"
    )
    fig_c1.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_c1.update_layout(height=470, showlegend=False)
    st.plotly_chart(fig_c1, use_container_width=True)

    stress_df = pd.DataFrame({
        "Cú sốc thực tế": ["COVID-19 2020–2022", "Bão Yagi 2024", "Đứt gãy chuỗi cung ứng", "Biến động FDI/xuất khẩu"],
        "Tác động tới nền kinh tế": [
            "Gián đoạn sản xuất, dịch chuyển lao động, tăng nhu cầu kỹ năng số và làm việc từ xa.",
            "Thiệt hại hạ tầng, gián đoạn logistics, yêu cầu phục hồi nhanh và quản trị rủi ro.",
            "Doanh nghiệp cần chuyển đổi nhà cung ứng, số hóa vận hành và tăng năng lực dự báo.",
            "Tăng trưởng phụ thuộc cầu thế giới, nên cần khả năng thích ứng của lao động và doanh nghiệp.",
        ],
        "Vì sao H là hàng hóa bảo hiểm?": [
            "Lao động có kỹ năng số dễ chuyển việc, làm việc từ xa và thích nghi với mô hình mới.",
            "Nhân lực địa phương có kỹ năng giúp phục hồi, vận hành hệ thống số và quản lý dữ liệu thiệt hại.",
            "Nhân lực số giúp doanh nghiệp tái cấu trúc quy trình nhanh hơn.",
            "Khi FDI/xuất khẩu yếu, kỹ năng số giúp chuyển sang thị trường, sản phẩm và mô hình kinh doanh mới.",
        ],
    })

    st.dataframe(stress_df, use_container_width=True)

    fig_c2 = go.Figure(data=[go.Sankey(
        node=dict(
            label=[
                "Cú sốc thực tế",
                "COVID-19",
                "Bão Yagi",
                "Đứt gãy chuỗi cung ứng",
                "Nhu cầu thích ứng",
                "Nhân lực số H",
                "Phục hồi nhanh hơn",
                "Giảm tổn thương lao động",
            ],
            pad=18,
            thickness=18,
            line=dict(color="black", width=0.3),
        ),
        link=dict(
            source=[0, 0, 0, 1, 2, 3, 4, 5, 5],
            target=[1, 2, 3, 4, 4, 4, 5, 6, 7],
            value=[35, 25, 20, 30, 25, 20, 50, 30, 20],
        )
    )])
    fig_c2.update_layout(
        title="Minh chứng câu c — H như hàng hóa bảo hiểm trước cú sốc",
        height=560
    )
    st.plotly_chart(fig_c2, use_container_width=True)

    policy_card(
        "🛡️",
        "H là hàng hóa bảo hiểm của nền kinh tế số",
        "COVID-19 và bão Yagi cho thấy cú sốc không chỉ phá vỡ sản xuất, mà còn kiểm tra năng lực thích ứng của con người, doanh nghiệp và chính quyền. Nhân lực số giúp chuyển đổi việc làm, vận hành dữ liệu, tổ chức làm việc từ xa, phục hồi logistics và triển khai dịch vụ công trong khủng hoảng.",
        "success"
    )

    st.markdown("""
    **Trả lời trọng tâm:**  
    Có khả năng Việt Nam đang dưới đầu tư vào nhân lực số nếu chỉ nhìn H như một khoản chi giáo dục thông thường. Trong mô hình Bài 10, H có vai trò giống **bảo hiểm chính sách**:
    
    - Khi thuận lợi, H hỗ trợ hấp thụ AI và D.
    - Khi bi quan hoặc khủng hoảng, H giúp lao động chuyển đổi việc làm.
    - Khi có cú sốc tự nhiên hoặc y tế, H giúp chính quyền và doanh nghiệp vận hành trên nền tảng số.
    
    Vì vậy, đầu tư vào H không chỉ để tăng năng suất, mà còn để tăng **resilience** — năng lực chống chịu và phục hồi.
    """)

    st.warning(
        "Đánh đổi chính sách: đầu tư H có thể không tạo GDP gain ngắn hạn cao như AI trong kịch bản lạc quan, nhưng lại làm giảm rủi ro tổn thương khi khủng hoảng. Đây là lý do các bài toán chính sách không nên chỉ tối đa hóa lợi ích kỳ vọng, mà cần xem thêm robust regret, VSS và EVPI."
    )


# ---------------------------------------------------------
# 4. RENDER
# ---------------------------------------------------------
def render():
    st.title("🎲 Bài 10 — Quy hoạch ngẫu nhiên hai giai đoạn dưới bất định")
    inject_css()

    st.markdown("""
    Bài 10 xây dựng mô hình **two-stage stochastic programming** cho quyết định đầu tư số Việt Nam 2026–2030.
    Giai đoạn 1 là quyết định **here-and-now**; giai đoạn 2 là quyết định **wait-and-see recourse** theo kịch bản.
    Module cũng tính **VSS**, **EVPI** và mở rộng **robust regret**.
    """)

    tabs = st.tabs([
        "10.1 Bối cảnh",
        "10.2 Scenario Tree",
        "10.3 Mô hình toán",
        "10.4 Hệ số β",
        "10.5 Giải lập trình",
        "10.6 Chính sách",
    ])

    with tabs[0]:
        show_context()

    with tabs[1]:
        show_scenario_tree()

    with tabs[2]:
        show_math_model()

    with tabs[3]:
        show_beta_table()

    with tabs[4]:
        show_programming_solution()

    with tabs[5]:
        show_policy_discussion()
