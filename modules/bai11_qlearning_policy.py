import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except Exception:
    GYM_AVAILABLE = False

try:
    from stable_baselines3 import DQN
    SB3_AVAILABLE = True
except Exception:
    SB3_AVAILABLE = False


# =========================================================
# BÀI 11 — Q-LEARNING CHO CHÍNH SÁCH KINH TẾ THÍCH NGHI
# Vietnam Economy as a Markov Decision Process
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
# 1. ACTIONS, STATES, LABELS
# ---------------------------------------------------------
def get_level_names():
    return {
        0: "low",
        1: "medium",
        2: "high",
    }


def get_state_factor_names():
    return {
        "gdp": "GDP growth",
        "digital": "Digital index",
        "ai": "AI capacity",
        "unemployment": "Unemployment risk",
    }


def get_action_data():
    data = pd.DataFrame({
        "action_id": [0, 1, 2, 3, 4],
        "action_code": ["a0", "a1", "a2", "a3", "a4"],
        "policy_name": [
            "Truyền thống",
            "Cân bằng",
            "Số hóa nhanh",
            "AI dẫn dắt",
            "Bao trùm",
        ],
        "K_share": [0.70, 0.40, 0.25, 0.20, 0.30],
        "D_share": [0.10, 0.25, 0.45, 0.20, 0.20],
        "AI_share": [0.10, 0.15, 0.15, 0.45, 0.10],
        "H_share": [0.10, 0.20, 0.15, 0.15, 0.40],
        "logic": [
            "Ưu tiên vốn vật chất, phù hợp khi hạ tầng cơ bản còn thiếu.",
            "Cân bằng giữa K, D, AI, H; ít cực đoan, dễ giải trình.",
            "Đẩy nhanh hạ tầng số và chuyển đổi số doanh nghiệp.",
            "Tăng tốc AI, phù hợp khi nền kinh tế đã có nền tảng số và nhân lực đủ tốt.",
            "Ưu tiên nhân lực số, giảm rủi ro thất nghiệp và tăng khả năng hấp thụ công nghệ.",
        ]
    })

    return data


def action_allocation_dict():
    df = get_action_data()
    return {
        int(row["action_id"]): np.array(
            [row["K_share"], row["D_share"], row["AI_share"], row["H_share"]],
            dtype=float
        )
        for _, row in df.iterrows()
    }


def state_to_index(state):
    g, d, ai, u = [int(x) for x in state]
    return g, d, ai, u


def state_to_label(state):
    lv = get_level_names()
    state = np.array(state, dtype=int)
    return (
        f"GDP={lv[int(state[0])]}, "
        f"D={lv[int(state[1])]}, "
        f"AI={lv[int(state[2])]}, "
        f"U={lv[int(state[3])]}"
    )


def action_to_label(action_id):
    df = get_action_data().set_index("action_id")
    row = df.loc[int(action_id)]
    return f"{row['action_code']} — {row['policy_name']}"


def all_states_df():
    rows = []
    lv = get_level_names()

    for g in range(3):
        for d in range(3):
            for ai in range(3):
                for u in range(3):
                    rows.append({
                        "gdp": g,
                        "digital": d,
                        "ai": ai,
                        "unemployment": u,
                        "state_tuple": (g, d, ai, u),
                        "state_label": f"GDP={lv[g]}, D={lv[d]}, AI={lv[ai]}, U={lv[u]}",
                    })

    return pd.DataFrame(rows)


def discretize_gdp_growth(growth_pct):
    if growth_pct < 4.8:
        return 0
    if growth_pct < 6.8:
        return 1
    return 2


def discretize_digital(D):
    if D < 24:
        return 0
    if D < 34:
        return 1
    return 2


def discretize_ai(AI):
    if AI < 95:
        return 0
    if AI < 125:
        return 1
    return 2


def discretize_unemployment_risk(U):
    if U < 3.5:
        return 0
    if U < 5.0:
        return 1
    return 2


# ---------------------------------------------------------
# 2. GYMNASIUM ENVIRONMENT
# ---------------------------------------------------------
if GYM_AVAILABLE:
    class VietnamEconomyEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(
            self,
            T=10,
            annual_budget=1000.0,
            seed=42,
            shock_prob=0.12,
            reward_weights=(0.40, 0.25, 0.20, 0.15),
            start_state=None,
        ):
            super().__init__()

            self.action_space = spaces.Discrete(5)
            self.observation_space = spaces.MultiDiscrete([3, 3, 3, 3])

            self.T = int(T)
            self.annual_budget = float(annual_budget)
            self.shock_prob = float(shock_prob)
            self.w = np.array(reward_weights, dtype=float)
            self.allocation = action_allocation_dict()
            self.rng = np.random.default_rng(seed)

            self.alpha_K = 0.33
            self.alpha_L = 0.42
            self.alpha_D = 0.10
            self.alpha_AI = 0.08
            self.alpha_H = 0.07

            self.start_state = start_state
            self.reset(seed=seed)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)

            if seed is not None:
                self.rng = np.random.default_rng(seed)

            self.t = 0

            # Baseline close to VN 2026 from previous modules
            self.K = 27500.0
            self.L = 54.0
            self.D = 20.3
            self.AI = 86.0
            self.H = 30.0
            self.U = 4.2
            self.A = 1.0

            self.prev_Y = self.production()
            self.last_info = {}

            if self.start_state is None:
                # Vietnam 2026: GDP growth medium, D medium, AI low, U medium
                self.state = np.array([1, 1, 0, 1], dtype=np.int64)
            else:
                self.state = np.array(self.start_state, dtype=np.int64)

                # Align continuous variables roughly with state for simulation
                self.K = [25000, 27500, 31000][int(self.state[0])]
                self.D = [18, 26, 38][int(self.state[1])]
                self.AI = [80, 110, 145][int(self.state[2])]
                self.U = [3.0, 4.3, 5.8][int(self.state[3])]
                self.H = [24, 31, 40][2 - int(self.state[3])] if int(self.state[3]) != 1 else 30
                self.prev_Y = self.production()

            return self.state.copy(), {}

        def production(self):
            return (
                self.A *
                (self.K ** self.alpha_K) *
                (self.L ** self.alpha_L) *
                (self.D ** self.alpha_D) *
                (self.AI ** self.alpha_AI) *
                (self.H ** self.alpha_H)
            )

        def step(self, action):
            action = int(action)
            a = self.allocation[action]

            K_share, D_share, AI_share, H_share = a
            budget = self.annual_budget

            # Random external shock
            shock = self.rng.random() < self.shock_prob
            shock_strength = self.rng.uniform(0.02, 0.07) if shock else 0.0

            # Store previous values
            old_Y = self.prev_Y
            old_U = self.U

            # Investment dynamics
            self.K = (1 - 0.045) * self.K + K_share * budget
            self.D = (1 - 0.055) * self.D + D_share * budget / 70.0
            self.AI = (1 - 0.070) * self.AI + AI_share * budget / 14.0
            self.H = (1 - 0.020) * self.H + H_share * budget / 120.0

            # TFP spillover: D, AI, H improve productivity
            self.A = self.A * (
                1
                + 0.0018 * self.D / 100
                + 0.0016 * self.AI / 100
                + 0.0024 * self.H / 100
            )

            # Labor mild growth
            self.L = self.L * 1.003

            new_Y_raw = self.production()
            new_Y = new_Y_raw * (1 - shock_strength)

            # Annual GDP growth percentage
            gdp_growth_pct = (new_Y / old_Y - 1.0) * 100

            # Unemployment risk dynamics:
            # AI-heavy raises short-run transition risk if H is not high.
            # H-heavy lowers risk; D improves matching.
            ai_pressure = 1.7 * AI_share * (1.0 if self.H < 34 else 0.55)
            human_buffer = 1.4 * H_share
            digital_matching = 0.35 * D_share

            self.U = (
                self.U
                + ai_pressure
                - human_buffer
                - digital_matching
                + (1.2 * shock_strength * 10)
                + self.rng.normal(0, 0.08)
            )
            self.U = float(np.clip(self.U, 2.2, 7.5))

            # Cyber risk: AI and D raise attack surface, H reduces governance risk.
            cyber_risk = max(
                0.0,
                0.20 * AI_share + 0.08 * D_share - 0.11 * H_share + 0.05 * int(shock)
            )

            # Emission proxy: K and AI infrastructure raise energy pressure; D and H reduce it partly.
            emission = max(
                0.0,
                0.28 * K_share + 0.20 * AI_share - 0.07 * D_share - 0.05 * H_share
            )

            # Normalized welfare components
            delta_gdp = gdp_growth_pct / 10.0
            delta_unemployment = (self.U - old_U) / 3.0

            reward = (
                self.w[0] * delta_gdp
                - self.w[1] * delta_unemployment
                - self.w[2] * cyber_risk
                - self.w[3] * emission
            )

            # Mild penalty for too aggressive AI when AI high but H not sufficient
            if AI_share >= 0.45 and self.H < 34:
                reward -= 0.04

            # Update state
            self.prev_Y = new_Y
            state_next = np.array([
                discretize_gdp_growth(gdp_growth_pct),
                discretize_digital(self.D),
                discretize_ai(self.AI),
                discretize_unemployment_risk(self.U),
            ], dtype=np.int64)

            self.state = state_next
            self.t += 1
            terminated = self.t >= self.T
            truncated = False

            self.last_info = {
                "year_index": self.t,
                "action": action,
                "gdp_growth_pct": gdp_growth_pct,
                "Y": new_Y,
                "K": self.K,
                "D": self.D,
                "AI": self.AI,
                "H": self.H,
                "U": self.U,
                "cyber_risk": cyber_risk,
                "emission": emission,
                "reward": reward,
                "shock": shock,
                "shock_strength": shock_strength,
                "state_label": state_to_label(self.state),
            }

            return self.state.copy(), float(reward), terminated, truncated, self.last_info.copy()


# ---------------------------------------------------------
# 3. TRAINING AND EVALUATION
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def train_q_learning(
    episodes=10000,
    alpha=0.10,
    gamma=0.95,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_episodes=5000,
    seed=42,
    shock_prob=0.12,
):
    if not GYM_AVAILABLE:
        return None

    env = VietnamEconomyEnv(seed=seed, shock_prob=shock_prob)
    rng = np.random.default_rng(seed)

    Q = np.zeros((3, 3, 3, 3, 5), dtype=float)
    episode_rewards = []
    episode_eps = []
    action_counts = np.zeros(5, dtype=int)

    for ep in range(int(episodes)):
        s, _ = env.reset(seed=seed + ep)
        eps = max(eps_end, eps_start - ep / eps_decay_episodes * (eps_start - eps_end))
        total_reward = 0.0

        while True:
            if rng.random() < eps:
                a = env.action_space.sample()
            else:
                a = int(np.argmax(Q[tuple(s)]))

            s2, r, done, truncated, info = env.step(a)

            old = Q[tuple(s) + (a,)]
            target = r + gamma * np.max(Q[tuple(s2)]) * (0 if done else 1)
            Q[tuple(s) + (a,)] = old + alpha * (target - old)

            total_reward += r
            action_counts[a] += 1
            s = s2

            if done or truncated:
                break

        episode_rewards.append(total_reward)
        episode_eps.append(eps)

    curve = pd.DataFrame({
        "episode": np.arange(1, episodes + 1),
        "total_reward": episode_rewards,
        "epsilon": episode_eps,
    })
    curve["rolling_reward_200"] = curve["total_reward"].rolling(200, min_periods=1).mean()

    action_df = get_action_data().copy()
    action_df["training_action_count"] = action_counts
    action_df["training_action_share_pct"] = action_df["training_action_count"] / action_df["training_action_count"].sum() * 100

    return {
        "Q": Q,
        "curve": curve,
        "action_df": action_df,
        "episodes": episodes,
        "alpha": alpha,
        "gamma": gamma,
        "eps_start": eps_start,
        "eps_end": eps_end,
        "seed": seed,
        "shock_prob": shock_prob,
    }


def extract_policy(Q):
    states = all_states_df()
    rows = []

    for _, row in states.iterrows():
        s = row["state_tuple"]
        q_values = Q[s]
        best_action = int(np.argmax(q_values))

        rows.append({
            "gdp": s[0],
            "digital": s[1],
            "ai": s[2],
            "unemployment": s[3],
            "state_label": row["state_label"],
            "best_action": best_action,
            "best_action_label": action_to_label(best_action),
            "max_Q": float(q_values[best_action]),
            "Q_a0": float(q_values[0]),
            "Q_a1": float(q_values[1]),
            "Q_a2": float(q_values[2]),
            "Q_a3": float(q_values[3]),
            "Q_a4": float(q_values[4]),
        })

    return pd.DataFrame(rows)


def evaluate_policy(policy_type, Q=None, fixed_action=None, episodes=500, seed=123, start_state=None, shock_prob=0.12):
    if not GYM_AVAILABLE:
        return pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(seed)
    episode_rows = []
    trace_rows = []

    for ep in range(int(episodes)):
        env = VietnamEconomyEnv(seed=seed + ep, start_state=start_state, shock_prob=shock_prob)
        s, _ = env.reset(seed=seed + ep)
        total_reward = 0.0

        for t in range(env.T):
            if policy_type == "q_policy":
                a = int(np.argmax(Q[tuple(s)]))
            elif policy_type == "fixed":
                a = int(fixed_action)
            elif policy_type == "random":
                a = int(rng.integers(0, 5))
            else:
                a = 1

            s2, r, done, truncated, info = env.step(a)
            total_reward += r

            trace_rows.append({
                "episode": ep,
                "t": t + 1,
                "policy": policy_type if policy_type != "fixed" else f"fixed_a{fixed_action}",
                "state_before": state_to_label(s),
                "action": a,
                "action_label": action_to_label(a),
                "reward": r,
                "state_after": state_to_label(s2),
                "gdp_growth_pct": info["gdp_growth_pct"],
                "Y": info["Y"],
                "D": info["D"],
                "AI": info["AI"],
                "H": info["H"],
                "U": info["U"],
                "cyber_risk": info["cyber_risk"],
                "emission": info["emission"],
                "shock": info["shock"],
            })

            s = s2
            if done or truncated:
                break

        episode_rows.append({
            "episode": ep,
            "policy": policy_type if policy_type != "fixed" else f"fixed_a{fixed_action}",
            "total_reward": total_reward,
        })

    return pd.DataFrame(episode_rows), pd.DataFrame(trace_rows)


def evaluate_multiple_policies(Q, episodes=500, seed=2026, shock_prob=0.12):
    frames = []
    traces = []

    for name, kwargs in [
        ("π* Q-learning", {"policy_type": "q_policy", "Q": Q}),
        ("Rule a1 — Cân bằng", {"policy_type": "fixed", "fixed_action": 1}),
        ("Rule a3 — AI dẫn dắt", {"policy_type": "fixed", "fixed_action": 3}),
        ("Random", {"policy_type": "random"}),
    ]:
        ep_df, tr_df = evaluate_policy(
            episodes=episodes,
            seed=seed,
            shock_prob=shock_prob,
            **kwargs
        )
        ep_df["policy_label"] = name
        tr_df["policy_label"] = name
        frames.append(ep_df)
        traces.append(tr_df)

    return pd.concat(frames, ignore_index=True), pd.concat(traces, ignore_index=True)


def policy_for_selected_states(Q):
    test_states = pd.DataFrame({
        "case": [
            "Việt Nam 2026 thực tế",
            "Suy giảm: GDP thấp, D thấp, U cao",
            "Nền tảng mạnh: GDP cao, AI cao, U thấp",
            "Chuyển đổi số thấp nhưng GDP trung bình",
            "AI thấp, thất nghiệp cao",
        ],
        "state": [
            (1, 1, 0, 1),
            (0, 0, 0, 2),
            (2, 2, 2, 0),
            (1, 0, 1, 1),
            (1, 1, 0, 2),
        ],
        "interpretation": [
            "Điểm xuất phát mô phỏng: tăng trưởng trung bình, số hóa trung bình, AI thấp, rủi ro thất nghiệp trung bình.",
            "Kinh tế yếu, số hóa yếu và thất nghiệp cao: cần chính sách quick win và bảo vệ lao động.",
            "Nền kinh tế đã tốt, AI mạnh, thất nghiệp thấp: có thể củng cố, tránh quá nóng.",
            "GDP chưa xấu nhưng số hóa thấp: cần tăng D để mở nền tảng.",
            "AI còn thấp nhưng rủi ro lao động cao: cần phối hợp H và D trước khi đẩy AI.",
        ]
    })

    rows = []
    for _, row in test_states.iterrows():
        s = tuple(row["state"])
        a = int(np.argmax(Q[s]))
        qvals = Q[s]

        rows.append({
            "case": row["case"],
            "state_label": state_to_label(s),
            "chosen_action": a,
            "chosen_action_label": action_to_label(a),
            "max_Q": qvals[a],
            "Q_a0": qvals[0],
            "Q_a1": qvals[1],
            "Q_a2": qvals[2],
            "Q_a3": qvals[3],
            "Q_a4": qvals[4],
            "interpretation": row["interpretation"],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# 4. SECTIONS
# ---------------------------------------------------------
def show_context():
    st.header("11.1. Bối cảnh Việt Nam")

    sticker_header(
        "🤖🇻🇳",
        "Học tăng cường: từ chính sách cố định sang chính sách thích nghi",
        "Bài 11 xem nền kinh tế Việt Nam như môi trường, chính sách là hành động, và phần thưởng phản ánh phúc lợi xã hội."
    )

    policy_card(
        "🎯",
        "Câu hỏi trung tâm",
        "Nếu trạng thái nền kinh tế thay đổi theo thời gian — tăng trưởng thấp/cao, số hóa yếu/mạnh, AI thấp/cao, thất nghiệp thấp/cao — thì chính sách ngân sách có nên cố định không? Q-learning minh họa cách học một chính sách thích nghi theo trạng thái.",
        "success"
    )

    policy_card(
        "⚖️",
        "Đánh đổi chính sách",
        "Một hành động AI-centric có thể tăng năng suất khi nền tảng AI và nhân lực đủ tốt, nhưng cũng có thể làm tăng rủi ro thất nghiệp và an ninh mạng nếu nền kinh tế chưa sẵn sàng. Ngược lại, chính sách bao trùm có thể giảm rủi ro lao động nhưng chậm tạo bước nhảy năng suất.",
        "warning"
    )

    policy_card(
        "🧭",
        "Nguyên tắc trách nhiệm",
        "AI hỗ trợ ra quyết định không thay thế trách nhiệm chính trị - xã hội. Mô hình RL trong bài này chỉ là công cụ minh họa, tạo bằng chứng định lượng và cảnh báo rủi ro; quyết định cuối cùng vẫn thuộc về cơ quan có thẩm quyền, với tham vấn xã hội và đánh giá tác động.",
        "purple"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số trạng thái", "81", "3⁴")
    c2.metric("Số hành động", "5", "a0–a4")
    c3.metric("Episode", "10 năm", "T = 10")
    c4.metric("Training", "10.000", "episodes")

    st.subheader("Ảnh 11.1 — MDP cho chính sách kinh tế thích nghi")

    labels = [
        "Trạng thái kinh tế s_t",
        "GDP growth",
        "Digital index",
        "AI capacity",
        "Unemployment risk",
        "Chính sách a_t",
        "K/D/AI/H allocation",
        "Môi trường kinh tế",
        "Reward welfare R_t",
        "Trạng thái mới s_{t+1}",
        "Q-learning cập nhật Q(s,a)",
    ]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            label=labels,
            pad=18,
            thickness=18,
            line=dict(color="black", width=0.35),
        ),
        link=dict(
            source=[0, 0, 0, 0, 5, 6, 7, 8, 9, 10],
            target=[1, 2, 3, 4, 6, 7, 8, 9, 10, 5],
            value=[20, 20, 20, 20, 80, 80, 80, 80, 80, 40],
        )
    )])
    fig.update_layout(
        title="Ảnh 11.1 — Vòng lặp MDP: quan sát → hành động → phần thưởng → trạng thái mới",
        height=560
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Bảng 11.1 — Vì sao RL phù hợp để minh họa chính sách thích nghi?")

    table = pd.DataFrame({
        "Thành phần RL": ["Environment", "State", "Action", "Reward", "Policy π", "Q-value"],
        "Trong bài toán Việt Nam": [
            "Nền kinh tế Việt Nam 2026–2035",
            "GDP growth, Digital index, AI capacity, Unemployment risk",
            "5 cấu trúc phân bổ ngân sách K/D/AI/H",
            "Welfare: tăng trưởng trừ thất nghiệp, cyber risk và emission",
            "Quy tắc chọn hành động theo trạng thái",
            "Giá trị kỳ vọng của hành động a tại trạng thái s",
        ],
        "Ý nghĩa chính sách": [
            "Chính sách tác động nhưng không kiểm soát hoàn toàn môi trường.",
            "Ra quyết định dựa trên tình trạng hiện tại, không dùng một kế hoạch cứng.",
            "Mỗi hành động là một ưu tiên phát triển khác nhau.",
            "Không tối đa hóa GDP đơn thuần; có cả rủi ro xã hội và môi trường.",
            "Chính sách thích nghi, có điều kiện.",
            "Bằng chứng định lượng để so sánh chính sách.",
        ]
    })
    st.dataframe(table, use_container_width=True)

    st.info(
        "Liên hệ Việt Nam: Nghị quyết 57-NQ/TW và QĐ 749/QĐ-TTg đều nhấn mạnh chuyển đổi số, đổi mới sáng tạo và năng lực công nghệ. RL giúp minh họa tư duy chính sách động: không có một hành động tối ưu cho mọi trạng thái."
    )


def show_mdp_model():
    st.header("11.2. Mô hình MDP đơn giản hóa")

    sticker_header(
        "🧮🕹️",
        "MDP = trạng thái + hành động + phần thưởng + chuyển trạng thái",
        "Bài toán được rời rạc hóa để sinh viên có thể cài đặt tabular Q-learning minh bạch và kiểm tra được."
    )

    st.subheader("1️⃣ Trạng thái rời rạc")

    st.latex(r"""
    s_t =
    (GDP_t,\;D_t,\;AI_t,\;U_t)
    """)

    st.latex(r"""
    GDP_t,D_t,AI_t,U_t \in \{low, medium, high\}
    """)

    st.latex(r"""
    |\mathcal{S}| = 3^4 = 81
    """)

    state_table = pd.DataFrame({
        "Yếu tố trạng thái": [
            "GDP growth",
            "Digital index",
            "AI capacity",
            "Unemployment risk",
        ],
        "Mức 0 — low": [
            "Tăng trưởng thấp",
            "Số hóa yếu",
            "Năng lực AI thấp",
            "Rủi ro thất nghiệp thấp",
        ],
        "Mức 1 — medium": [
            "Tăng trưởng trung bình",
            "Số hóa trung bình",
            "Năng lực AI trung bình",
            "Rủi ro thất nghiệp trung bình",
        ],
        "Mức 2 — high": [
            "Tăng trưởng cao",
            "Số hóa cao",
            "Năng lực AI cao",
            "Rủi ro thất nghiệp cao",
        ],
        "Ý nghĩa chính sách": [
            "Nền kinh tế đang cần kích thích hay củng cố?",
            "Có nền tảng để hấp thụ AI và dịch vụ số không?",
            "Có thể triển khai AI mạnh hay cần chuẩn bị thêm?",
            "Có cần ưu tiên bao trùm và đào tạo lại không?",
        ]
    })
    st.dataframe(state_table, use_container_width=True)

    st.subheader("2️⃣ Hành động chính sách")

    actions = get_action_data()
    st.dataframe(actions, use_container_width=True)

    fig_actions = px.bar(
        actions.melt(
            id_vars=["action_code", "policy_name"],
            value_vars=["K_share", "D_share", "AI_share", "H_share"],
            var_name="Hạng mục",
            value_name="Tỷ trọng"
        ),
        x="action_code",
        y="Tỷ trọng",
        color="Hạng mục",
        barmode="stack",
        text="Tỷ trọng",
        title="Ảnh 11.2 — 5 hành động ngân sách K/D/AI/H"
    )
    fig_actions.update_layout(height=520, xaxis_title="Hành động")
    st.plotly_chart(fig_actions, use_container_width=True)

    st.subheader("3️⃣ Reward function")

    st.latex(r"""
    R_t =
    w_1\Delta GDP_t
    -
    w_2\Delta Unemployment_t
    -
    w_3CyberRisk_t
    -
    w_4Emission_t
    """)

    st.latex(r"""
    w=(0.40,\;0.25,\;0.20,\;0.15)
    """)

    reward_df = pd.DataFrame({
        "Thành phần": ["ΔGDP", "ΔUnemployment", "CyberRisk", "Emission"],
        "Trọng số": [0.40, 0.25, 0.20, 0.15],
        "Dấu trong reward": ["+", "-", "-", "-"],
        "Diễn giải": [
            "Tăng trưởng cao làm welfare tăng.",
            "Thất nghiệp tăng làm welfare giảm.",
            "AI/D mở rộng nhanh có thể tăng rủi ro an ninh mạng.",
            "Đầu tư K và AI hạ tầng có thể tạo áp lực năng lượng/phát thải.",
        ],
    })
    st.dataframe(reward_df, use_container_width=True)

    fig_reward = px.bar(
        reward_df,
        x="Thành phần",
        y="Trọng số",
        color="Dấu trong reward",
        text="Trọng số",
        title="Ảnh 11.3 — Trọng số welfare trong reward function"
    )
    fig_reward.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_reward.update_layout(height=440)
    st.plotly_chart(fig_reward, use_container_width=True)

    st.subheader("4️⃣ Q-learning update")

    st.latex(r"""
    Q(s_t,a_t)
    \leftarrow
    Q(s_t,a_t)
    +
    \alpha
    \left[
    r_t+\gamma\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)
    \right]
    """)

    st.markdown("""
    Công thức trên cập nhật giá trị của hành động `a_t` tại trạng thái `s_t`.  
    Nếu hành động tạo reward cao và dẫn tới trạng thái tương lai tốt, Q-value tăng lên.
    Sau nhiều episode, chính sách được trích xuất bằng:
    """)

    st.latex(r"""
    \pi^*(s)=\arg\max_a Q(s,a)
    """)

    policy_card(
        "📌",
        "Điểm cần nhớ",
        "Q-learning không cần biết trước toàn bộ phương trình tối ưu. Agent học qua thử nghiệm lặp lại. Tuy nhiên trong chính sách công, thử nghiệm trực tiếp trên xã hội là không thể tùy tiện; vì vậy mô hình chỉ nên dùng như mô phỏng hỗ trợ ra quyết định.",
        "warning"
    )

    st.subheader("Ảnh 11.4 — Chuyển trạng thái mô phỏng bằng Cobb-Douglas mở rộng")

    st.latex(r"""
    Y_t =
    A_tK_t^{0.33}L_t^{0.42}D_t^{0.10}AI_t^{0.08}H_t^{0.07}
    """)

    st.markdown("""
    Môi trường dùng hàm Cobb-Douglas mở rộng để mô phỏng sản lượng.  
    Hành động phân bổ ngân sách làm thay đổi K, D, AI, H; từ đó ảnh hưởng đến GDP growth, thất nghiệp, cyber risk và emission.
    """)


def show_training_solution():
    st.header("11.3. Giải yêu cầu lập trình")

    if not GYM_AVAILABLE:
        st.error("Chưa cài `gymnasium`. Hãy thêm `gymnasium` vào requirements.txt.")
        return None

    sticker_header(
        "💻🧠",
        "Huấn luyện Q-learning và so sánh chính sách",
        "Phần này cài đặt môi trường gymnasium, huấn luyện tabular Q-learning, trích xuất π*(s), rồi so sánh với các rule-based policies."
    )

    st.subheader("Thiết lập tham số huấn luyện")

    c1, c2, c3, c4 = st.columns(4)

    episodes = c1.number_input(
        "Số episodes",
        min_value=1000,
        max_value=30000,
        value=10000,
        step=1000,
        key="bai11_episodes"
    )

    alpha = c2.slider(
        "Learning rate α",
        min_value=0.01,
        max_value=0.50,
        value=0.10,
        step=0.01,
        key="bai11_alpha"
    )

    gamma = c3.slider(
        "Discount γ",
        min_value=0.50,
        max_value=0.99,
        value=0.95,
        step=0.01,
        key="bai11_gamma"
    )

    shock_prob = c4.slider(
        "Xác suất cú sốc",
        min_value=0.00,
        max_value=0.40,
        value=0.12,
        step=0.02,
        key="bai11_shock_prob"
    )

    seed = st.number_input(
        "Seed tái lập kết quả",
        min_value=1,
        max_value=99999,
        value=42,
        step=1,
        key="bai11_seed"
    )

    # -----------------------------------------------------
    # 11.3.1 + 11.3.2
    # -----------------------------------------------------
    st.subheader("Câu 11.3.1–11.3.2 — Cài đặt Env và huấn luyện Q-learning")

    with st.spinner("Đang huấn luyện Q-learning..."):
        result = train_q_learning(
            episodes=int(episodes),
            alpha=float(alpha),
            gamma=float(gamma),
            eps_start=1.0,
            eps_end=0.05,
            eps_decay_episodes=5000,
            seed=int(seed),
            shock_prob=float(shock_prob),
        )

    Q = result["Q"]
    curve = result["curve"]
    action_train = result["action_df"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Episodes", f"{episodes:,}")
    k2.metric("α", f"{alpha:.2f}")
    k3.metric("γ", f"{gamma:.2f}")
    k4.metric("Q-table shape", str(Q.shape))

    st.markdown("#### Learning curve")

    fig_curve = px.line(
        curve,
        x="episode",
        y=["total_reward", "rolling_reward_200"],
        title="Ảnh 11.5 — Learning curve: reward theo episode"
    )
    fig_curve.update_layout(height=520, yaxis_title="Tổng reward trong episode")
    st.plotly_chart(fig_curve, use_container_width=True)

    fig_eps = px.line(
        curve,
        x="episode",
        y="epsilon",
        title="Ảnh 11.6 — Epsilon giảm dần: từ khám phá sang khai thác"
    )
    fig_eps.update_layout(height=420, yaxis_title="epsilon")
    st.plotly_chart(fig_eps, use_container_width=True)

    st.markdown("#### Tần suất hành động trong quá trình training")
    st.dataframe(action_train[[
        "action_code", "policy_name", "training_action_count", "training_action_share_pct"
    ]].round(3), use_container_width=True)

    fig_action_train = px.bar(
        action_train,
        x="action_code",
        y="training_action_share_pct",
        color="policy_name",
        text="training_action_share_pct",
        title="Ảnh 11.7 — Agent đã khám phá hành động nào nhiều nhất?"
    )
    fig_action_train.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_action_train.update_layout(height=450, yaxis_title="Tỷ trọng trong training, %")
    st.plotly_chart(fig_action_train, use_container_width=True)

    # -----------------------------------------------------
    # 11.3.3
    # -----------------------------------------------------
    st.subheader("Câu 11.3.3 — Trích xuất chính sách π*(s)")

    policy_df = extract_policy(Q)
    selected_states = policy_for_selected_states(Q)

    st.markdown("#### Chính sách tại Việt Nam 2026 và 4 trạng thái giả định")
    st.dataframe(selected_states.round(4), use_container_width=True)

    fig_selected = px.bar(
        selected_states.melt(
            id_vars=["case", "state_label", "chosen_action_label"],
            value_vars=["Q_a0", "Q_a1", "Q_a2", "Q_a3", "Q_a4"],
            var_name="Action",
            value_name="Q-value"
        ),
        x="case",
        y="Q-value",
        color="Action",
        barmode="group",
        title="Ảnh 11.8 — Q-value tại các trạng thái chính sách quan trọng"
    )
    fig_selected.update_layout(height=560, xaxis_tickangle=-20)
    st.plotly_chart(fig_selected, use_container_width=True)

    st.markdown("#### Policy map toàn bộ 81 trạng thái")
    st.dataframe(policy_df[[
        "state_label", "best_action_label", "max_Q", "Q_a0", "Q_a1", "Q_a2", "Q_a3", "Q_a4"
    ]].round(4), use_container_width=True)

    action_counts = policy_df.groupby("best_action_label", as_index=False).agg(
        n_states=("state_label", "count")
    )
    action_counts["share_pct"] = action_counts["n_states"] / 81 * 100

    fig_policy_share = px.pie(
        action_counts,
        names="best_action_label",
        values="n_states",
        hole=0.42,
        title="Ảnh 11.9 — Chính sách π* chọn hành động nào trên 81 trạng thái?"
    )
    fig_policy_share.update_layout(height=500)
    st.plotly_chart(fig_policy_share, use_container_width=True)

    heat_policy = policy_df.copy()
    heat_policy["GDP-D state"] = "GDP" + heat_policy["gdp"].astype(str) + "_D" + heat_policy["digital"].astype(str)
    heat_policy["AI-U state"] = "AI" + heat_policy["ai"].astype(str) + "_U" + heat_policy["unemployment"].astype(str)

    pivot = heat_policy.pivot_table(
        index="GDP-D state",
        columns="AI-U state",
        values="best_action",
        aggfunc="first"
    )

    fig_policy_heat = px.imshow(
        pivot,
        text_auto=True,
        aspect="auto",
        title="Ảnh 11.10 — Heatmap policy π*: hành động tối ưu theo nhóm trạng thái"
    )
    fig_policy_heat.update_layout(height=560)
    st.plotly_chart(fig_policy_heat, use_container_width=True)

    # -----------------------------------------------------
    # 11.3.4
    # -----------------------------------------------------
    st.subheader("Câu 11.3.4 — So sánh π* với rule-based policies")

    with st.spinner("Đang đánh giá π*, a1, a3 và random..."):
        compare_df, trace_df = evaluate_multiple_policies(
            Q,
            episodes=500,
            seed=int(seed) + 1000,
            shock_prob=float(shock_prob),
        )

    summary = compare_df.groupby("policy_label", as_index=False).agg(
        mean_reward=("total_reward", "mean"),
        std_reward=("total_reward", "std"),
        min_reward=("total_reward", "min"),
        max_reward=("total_reward", "max"),
    )

    st.dataframe(summary.round(4), use_container_width=True)

    fig_compare = px.box(
        compare_df,
        x="policy_label",
        y="total_reward",
        color="policy_label",
        title="Ảnh 11.11 — Phân phối reward tích lũy: π* vs rule-based"
    )
    fig_compare.update_layout(height=520, xaxis_title="Chính sách", yaxis_title="Tổng reward 10 năm")
    st.plotly_chart(fig_compare, use_container_width=True)

    fig_bar = px.bar(
        summary.sort_values("mean_reward", ascending=False),
        x="policy_label",
        y="mean_reward",
        error_y="std_reward",
        color="policy_label",
        text="mean_reward",
        title="Ảnh 11.12 — Reward trung bình của các chính sách"
    )
    fig_bar.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_bar.update_layout(height=480, xaxis_title="Chính sách", yaxis_title="Mean reward")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Typical one-episode trace for q policy
    q_trace = trace_df[trace_df["policy_label"] == "π* Q-learning"].query("episode == 0").copy()

    st.markdown("#### Một quỹ đạo minh họa của π* trong 10 năm")
    st.dataframe(q_trace[[
        "t", "state_before", "action_label", "reward", "state_after",
        "gdp_growth_pct", "D", "AI", "H", "U", "cyber_risk", "emission", "shock"
    ]].round(4), use_container_width=True)

    fig_trace = px.line(
        q_trace,
        x="t",
        y=["reward", "gdp_growth_pct", "U", "cyber_risk", "emission"],
        title="Ảnh 11.13 — Một quỹ đạo π*: reward, tăng trưởng, thất nghiệp, cyber risk, emission"
    )
    fig_trace.update_layout(height=530)
    st.plotly_chart(fig_trace, use_container_width=True)

    # -----------------------------------------------------
    # 11.3.5
    # -----------------------------------------------------
    st.subheader("Câu 11.3.5 — Huấn luyện DQN trực tiếp trên web")

    st.warning(
        "Huấn luyện DQN trực tiếp trên Streamlit Cloud có thể mất thời gian và tốn tài nguyên. "
        "Nên dùng số bước nhỏ để demo. Với bài nộp chính thức, tabular Q-learning vẫn là kết quả chính."
    )

    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.monitor import Monitor
        sb3_ready = True
    except Exception as e:
        sb3_ready = False
        st.error("Chưa cài được stable-baselines3 hoặc import DQN bị lỗi.")
        st.code(str(e))

    if sb3_ready:
        total_timesteps = st.slider(
            "Số bước huấn luyện DQN",
            min_value=500,
            max_value=10000,
            value=2000,
            step=500,
            key="bai11_dqn_timesteps",
        )

        learning_rate = st.select_slider(
            "Learning rate",
            options=[1e-4, 5e-4, 1e-3, 5e-3],
            value=1e-3,
            key="bai11_dqn_learning_rate",
        )

        gamma_dqn = st.slider(
            "Gamma DQN",
            min_value=0.80,
            max_value=0.99,
            value=0.95,
            step=0.01,
            key="bai11_dqn_gamma",
        )

        with st.expander("Xem mã minh họa cấu hình DQN"):
            st.code(
                """
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

env = VietnamEconomyEnv()
env = Monitor(env)

model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=200,
    batch_size=32,
    gamma=0.95,
    exploration_fraction=0.4,
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[64, 64]),
    verbose=0,
    seed=42,
)

model.learn(total_timesteps=2000)
                """,
                language="python",
            )

        if st.button("Train DQN trực tiếp trên web", key="bai11_train_dqn_button"):
            with st.spinner("Đang huấn luyện DQN, vui lòng chờ..."):
                try:
                    env = VietnamEconomyEnv()
                    env = Monitor(env)

                    model = DQN(
                        "MlpPolicy",
                        env,
                        learning_rate=learning_rate,
                        buffer_size=10000,
                        learning_starts=200,
                        batch_size=32,
                        gamma=gamma_dqn,
                        exploration_fraction=0.4,
                        exploration_final_eps=0.05,
                        policy_kwargs=dict(net_arch=[64, 64]),
                        verbose=0,
                        seed=42,
                    )

                    model.learn(total_timesteps=total_timesteps)

                    st.success("Huấn luyện DQN hoàn tất.")

                    eval_env = VietnamEconomyEnv()
                    obs, info = eval_env.reset(seed=42)

                    total_reward = 0.0
                    actions = []
                    rewards = []

                    done = False
                    step_count = 0

                    while not done and step_count < 50:
                        action, _states = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = eval_env.step(action)

                        done = bool(terminated or truncated)
                        total_reward += float(reward)
                        actions.append(int(action))
                        rewards.append(float(reward))
                        step_count += 1

                    col1, col2 = st.columns(2)
                    col1.metric("Tổng reward đánh giá", f"{total_reward:.3f}")
                    col2.metric("Số bước đánh giá", step_count)

                    eval_df = pd.DataFrame({
                        "Bước": list(range(1, len(actions) + 1)),
                        "Action": actions,
                        "Reward": rewards,
                    })

                    st.dataframe(eval_df, use_container_width=True)

                    fig_reward = px.line(
                        eval_df,
                        x="Bước",
                        y="Reward",
                        markers=True,
                        title="Reward theo bước sau khi train DQN"
                    )
                    fig_reward.update_layout(
                        height=420,
                        xaxis_title="Bước",
                        yaxis_title="Reward"
                    )
                    st.plotly_chart(fig_reward, use_container_width=True)

                except Exception as e:
                    st.error(
                        "DQN chưa chạy được. Kiểm tra lại VietnamEconomyEnv hoặc cấu hình thư viện."
                    )
                    st.code(str(e))

    policy_card(
        "🧠",
        "DQN có thể cải thiện khi nào?",
        "DQN hữu ích khi trạng thái liên tục, số trạng thái lớn hoặc chính sách cần học đặc trưng phi tuyến. "
        "Trong bài này trạng thái chỉ có 81 mức nên tabular Q-learning minh bạch hơn, dễ giải thích hơn "
        "và phù hợp với yêu cầu học tập.",
        "info",
    )

    return {
        "Q": Q,
        "curve": curve,
        "policy_df": policy_df,
        "selected_states": selected_states,
        "compare_df": compare_df,
        "trace_df": trace_df,
        "summary": summary,
    }


def show_policy_discussion():
    st.header("11.4. Câu hỏi thảo luận chính sách")

    if not GYM_AVAILABLE:
        st.error("Cần cài `gymnasium` để chạy phần thảo luận.")
        return

    sticker_header(
        "🧭🇻🇳",
        "Từ π*(s) đến quy trình chính sách có trách nhiệm",
        "Phần này dùng kết quả Q-learning để trả lời: khi nào quick win, khi nào consolidation, và làm sao dùng AI mà không thay thế quyết định chính trị - xã hội."
    )

    result = train_q_learning(
        episodes=10000,
        alpha=0.10,
        gamma=0.95,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_episodes=5000,
        seed=42,
        shock_prob=0.12,
    )
    Q = result["Q"]

    # -----------------------------------------------------
    # Câu a
    # -----------------------------------------------------
    st.subheader("a) GDP thấp, D thấp, U cao — π*(s) chọn gì? Có phải quick win không?")

    s_bad = (0, 0, 0, 2)
    a_bad = int(np.argmax(Q[s_bad]))
    q_bad = pd.DataFrame({
        "Action": [f"a{i}" for i in range(5)],
        "Policy": [action_to_label(i) for i in range(5)],
        "Q-value": [Q[s_bad + (i,)] for i in range(5)],
    })

    c1, c2 = st.columns(2)
    c1.metric("Trạng thái", state_to_label(s_bad))
    c2.metric("π*(s)", action_to_label(a_bad))

    st.dataframe(q_bad.round(4), use_container_width=True)

    fig_a = px.bar(
        q_bad,
        x="Action",
        y="Q-value",
        color="Policy",
        text="Q-value",
        title="Minh chứng câu a — Q-value khi GDP thấp, D thấp, U cao"
    )
    fig_a.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_a.update_layout(height=480)
    st.plotly_chart(fig_a, use_container_width=True)

    if a_bad in [2, 4, 1]:
        quick_win_msg = (
            "Có. Hành động được chọn nghiêng về quick win hoặc ổn định xã hội: số hóa nhanh, bao trùm hoặc cân bằng. "
            "Khi GDP thấp và thất nghiệp cao, chính sách không nên chỉ đẩy AI mạnh; cần ưu tiên nền tảng số, đào tạo và giảm rủi ro lao động."
        )
        tone = "success"
    else:
        quick_win_msg = (
            "Chưa hẳn. Nếu agent chọn AI dẫn dắt hoặc truyền thống, cần kiểm tra lại reward weights hoặc giả định môi trường, vì trạng thái này thường cần quick win về số hóa và bao trùm."
        )
        tone = "warning"

    policy_card(
        "⚡",
        "Quick win trong trạng thái suy giảm",
        quick_win_msg,
        tone
    )

    st.markdown("""
    **Diễn giải chính sách:**  
    Khi GDP thấp, D thấp và U cao, nền kinh tế vừa thiếu động lực tăng trưởng vừa có rủi ro xã hội.  
    Một chính sách “quick win” hợp lý thường có ba đặc điểm:
    
    - cải thiện nhanh nền tảng số để giảm chi phí giao dịch;
    - tạo hoặc giữ việc làm thông qua đào tạo lại;
    - tránh triển khai AI quá nhanh khi năng lực hấp thụ thấp.
    
    Vì vậy, hành động a2 hoặc a4 thường dễ giải thích hơn a3 trong trạng thái này.
    """)

    # -----------------------------------------------------
    # Câu b
    # -----------------------------------------------------
    st.subheader("b) GDP cao, AI cao, U thấp — chính sách chọn gì? Có phù hợp consolidation không?")

    s_good = (2, 2, 2, 0)
    a_good = int(np.argmax(Q[s_good]))
    q_good = pd.DataFrame({
        "Action": [f"a{i}" for i in range(5)],
        "Policy": [action_to_label(i) for i in range(5)],
        "Q-value": [Q[s_good + (i,)] for i in range(5)],
    })

    c3, c4 = st.columns(2)
    c3.metric("Trạng thái", state_to_label(s_good))
    c4.metric("π*(s)", action_to_label(a_good))

    st.dataframe(q_good.round(4), use_container_width=True)

    fig_b = px.bar(
        q_good,
        x="Action",
        y="Q-value",
        color="Policy",
        text="Q-value",
        title="Minh chứng câu b — Q-value khi GDP cao, AI cao, U thấp"
    )
    fig_b.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_b.update_layout(height=480)
    st.plotly_chart(fig_b, use_container_width=True)

    if a_good in [1, 4, 0]:
        consolidation_msg = (
            "Có xu hướng phù hợp với consolidation. Khi GDP và AI đã cao, chính sách có thể chuyển sang củng cố nền tảng, cân bằng rủi ro, đào tạo và tránh quá nóng công nghệ."
        )
        tone_b = "success"
    else:
        consolidation_msg = (
            "Nếu agent vẫn chọn AI dẫn dắt, điều đó hàm ý mô hình đánh giá lợi ích biên của AI còn lớn. Tuy nhiên về chính sách thực tế cần kiểm tra cyber risk, độc quyền dữ liệu và bất bình đẳng kỹ năng."
        )
        tone_b = "warning"

    policy_card(
        "🏗️",
        "Consolidation khi nền tảng đã mạnh",
        consolidation_msg,
        tone_b
    )

    st.markdown("""
    **Diễn giải chính sách:**  
    Khi GDP cao, AI cao và U thấp, vấn đề không còn là “kích hoạt tăng trưởng bằng mọi giá”.
    Trọng tâm chuyển sang consolidation:
    
    - củng cố an ninh dữ liệu và quản trị AI;
    - đầu tư H để giữ năng lực vận hành;
    - tránh phụ thuộc quá mức vào một công nghệ;
    - dùng AI để nâng chất lượng dịch vụ công, năng suất và đổi mới sáng tạo.
    """)

    # -----------------------------------------------------
    # Câu c
    # -----------------------------------------------------
    st.subheader("c) Tích hợp π* vào hoạch định chính sách Việt Nam thế nào để không thay thế quyết định chính trị - xã hội?")

    governance = pd.DataFrame({
        "Bước": [
            "1. Mô phỏng kỹ thuật",
            "2. Kiểm định dữ liệu",
            "3. Tham vấn chuyên gia",
            "4. Đánh giá tác động xã hội",
            "5. Thử nghiệm có kiểm soát",
            "6. Quyết định chính trị",
            "7. Giám sát và cập nhật",
        ],
        "Vai trò của π*(s)": [
            "Đề xuất hành động ứng viên theo trạng thái.",
            "Kiểm tra độ nhạy với giả định reward, shock, trạng thái.",
            "So sánh với tri thức ngành, địa phương, doanh nghiệp, lao động.",
            "Đánh giá việc làm, bất bình đẳng, môi trường, an ninh dữ liệu.",
            "Pilot nhỏ, có cơ chế dừng nếu rủi ro vượt ngưỡng.",
            "Cơ quan có thẩm quyền chọn phương án, không giao quyền cho thuật toán.",
            "Cập nhật mô hình khi dữ liệu thực tế thay đổi.",
        ],
        "Rào chắn trách nhiệm": [
            "Không tự động ban hành chính sách.",
            "Công khai giả định và dữ liệu.",
            "Có phản biện ngoài mô hình.",
            "Không tối đa hóa reward hẹp.",
            "Không thử nghiệm tùy tiện trên nhóm dễ tổn thương.",
            "Minh bạch trách nhiệm giải trình.",
            "Theo dõi sai lệch và tác dụng phụ.",
        ]
    })

    st.dataframe(governance, use_container_width=True)

    fig_c = go.Figure(data=[go.Sankey(
        node=dict(
            label=[
                "Dữ liệu kinh tế",
                "Mô hình RL π*(s)",
                "Khuyến nghị kỹ thuật",
                "Chuyên gia & địa phương",
                "Đánh giá xã hội",
                "Cơ quan chính sách",
                "Quyết định cuối cùng",
                "Giám sát hậu kiểm",
            ],
            pad=18,
            thickness=18,
            line=dict(color="black", width=0.35),
        ),
        link=dict(
            source=[0, 1, 2, 2, 3, 4, 5, 6],
            target=[1, 2, 3, 4, 5, 5, 6, 7],
            value=[40, 40, 20, 20, 20, 20, 40, 40],
        )
    )])
    fig_c.update_layout(
        title="Minh chứng câu c — AI/RL là đầu vào, không phải người ra quyết định",
        height=560
    )
    st.plotly_chart(fig_c, use_container_width=True)

    policy_card(
        "⚖️",
        "Nguyên tắc không thay thế quyết định chính trị - xã hội",
        "π*(s) nên được tích hợp như một lớp phân tích trong dashboard chính sách: tạo khuyến nghị, cảnh báo rủi ro và so sánh kịch bản. Quyết định cuối cùng phải đi qua quy trình thể chế, tham vấn xã hội, đánh giá tác động và trách nhiệm giải trình.",
        "danger"
    )

    st.markdown("""
    **Liên hệ Việt Nam:**  
    Với tinh thần Nghị quyết 57-NQ/TW và QĐ 749/QĐ-TTg, AI có thể giúp tăng chất lượng phân tích chính sách, nhưng không thể thay thế vai trò Nhà nước trong cân bằng tăng trưởng, công bằng, an sinh, an ninh và chủ quyền dữ liệu.  
    Cách dùng phù hợp là:
    
    - dùng π*(s) để tạo phương án tham khảo;
    - công khai reward function và giả định;
    - chạy phân tích nhạy cảm;
    - yêu cầu chuyên gia và bên chịu tác động phản biện;
    - quyết định cuối cùng do cơ quan có thẩm quyền chịu trách nhiệm.
    """)


# ---------------------------------------------------------
# 5. RENDER
# ---------------------------------------------------------
def render():
    st.title("🧠 Bài 11 — Q-learning cho chính sách kinh tế thích nghi")
    inject_css()

    st.markdown("""
    Bài 11 mô phỏng nền kinh tế Việt Nam như một **Markov Decision Process (MDP)**.
    Agent học chính sách phân bổ ngân sách K/D/AI/H bằng **tabular Q-learning** qua nhiều episode,
    sau đó so sánh với các chính sách rule-based.
    """)

    tabs = st.tabs([
        "11.1 Bối cảnh",
        "11.2 MDP Model",
        "11.3 Q-learning",
        "11.4 Chính sách",
    ])

    with tabs[0]:
        show_context()

    with tabs[1]:
        show_mdp_model()

    with tabs[2]:
        show_training_solution()

    with tabs[3]:
        show_policy_discussion()
