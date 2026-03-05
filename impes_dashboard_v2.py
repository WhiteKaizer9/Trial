import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="IMPES Core Flood Dashboard", layout="wide")

IN3_PER_ML = 0.0610237441
IN3_PER_FT3 = 1728.0
FT3_PER_RB = 5.615

def in3min_to_stbday(q_in3_min: float, Boi: float) -> float:
    q_ft3_min = q_in3_min / IN3_PER_FT3
    q_RB_min = q_ft3_min / FT3_PER_RB
    q_STB_min = q_RB_min / max(Boi, 1e-12)
    return q_STB_min * 1440.0

def pore_volume_in3(L_in: float, D_in: float, phi: float) -> float:
    area = np.pi * (D_in ** 2) / 4.0
    return phi * area * L_in

def default_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Total length of core (in)", 12.0),
            ("Core diameter (in)", 3.0),
            ("Number of grid blocks (Nx)", 12),
            ("Porosity, ϕ", 0.22),
            ("Permeability (md)", 78.17),
            ("Total injection rate (mL/min)", 0.20),
            ("Oil viscosity, μo (cp)", 4.80),
            ("Water viscosity, μw (cp)", 0.85),
            ("Oil compressibility, co (1/psi)", 7.5e-6),
            ("Water compressibility, cw (1/psi)", 3.0e-6),
            ("Rock compressibility, cϕ (1/psi)", 3.5e-6),
            ("Initial oil FVF, Boi (RB/STB)", 1.28),
            ("Residual water saturation, Swr", 0.15),
            ("Residual oil saturation, Sorw", 0.20),
            ("Max water rel perm, krw*", 0.32),
            ("Max oil rel perm, kro*", 0.90),
            ("Water rel perm exponent, nw", 2.20),
            ("Oil rel perm exponent, no", 3.65),
            ("Initial oil pressure (psia)", 1800.0),
            ("Flowing pressure at outlet, p_well (psia)", 1800.0),
            ("Time step Δt (min)", 10.0),
            ("End time (min)", 1000.0),
        ],
        columns=["Parameter", "Value"],
    )

def rel_perm(Sw, Swr, Sorw, krw_star, kro_star, nw, no):
    denom = max(1.0 - Swr - Sorw, 1e-12)
    Se = np.clip((Sw - Swr) / denom, 0.0, 1.0)
    krw = krw_star * (Se ** nw)
    kro = kro_star * ((1.0 - Se) ** no)
    return krw, kro

def upstream(val_left, val_right, p_left, p_right):
    return val_left if p_left >= p_right else val_right

@st.cache_data(show_spinner=False)
def simulate_impes(param_map: dict) -> dict:
    L = float(param_map["Total length of core (in)"])
    D = float(param_map["Core diameter (in)"])
    Nx = int(param_map["Number of grid blocks (Nx)"])
    phi = float(param_map["Porosity, ϕ"])
    k_md = float(param_map["Permeability (md)"])
    q_inj = float(param_map["Total injection rate (mL/min)"]) * IN3_PER_ML
    mu_o = float(param_map["Oil viscosity, μo (cp)"])
    mu_w = float(param_map["Water viscosity, μw (cp)"])
    co = float(param_map["Oil compressibility, co (1/psi)"])
    cw = float(param_map["Water compressibility, cw (1/psi)"])
    cphi = float(param_map["Rock compressibility, cϕ (1/psi)"])
    Boi = float(param_map["Initial oil FVF, Boi (RB/STB)"])
    Swr = float(param_map["Residual water saturation, Swr"])
    Sorw = float(param_map["Residual oil saturation, Sorw"])
    krw_star = float(param_map["Max water rel perm, krw*"])
    kro_star = float(param_map["Max oil rel perm, kro*"])
    nw = float(param_map["Water rel perm exponent, nw"])
    no = float(param_map["Oil rel perm exponent, no"])
    p_init = float(param_map["Initial oil pressure (psia)"])
    p_well = float(param_map["Flowing pressure at outlet, p_well (psia)"])
    dt = float(param_map["Time step Δt (min)"])
    t_end = float(param_map["End time (min)"])

    dx = L / Nx
    area = np.pi * (D ** 2) / 4.0
    VR = dx * area
    x_centers = np.arange(Nx) * dx + dx / 2.0

    Nt = int(np.floor(t_end / dt)) + 1
    t = np.arange(Nt) * dt

    Sw = np.ones(Nx) * Swr
    p = np.ones(Nx) * p_init

    Sw_hist = np.zeros((Nt, Nx))
    p_hist = np.zeros((Nt, Nx))
    A_hist = np.zeros((Nt, Nx, Nx))
    RHS_hist = np.zeros((Nt, Nx))
    qo_hist = np.zeros(Nt)
    qw_hist = np.zeros(Nt)

    PV_in3 = pore_volume_in3(L, D, phi)
    OOIP_in3 = PV_in3 * (1.0 - Swr)
    OOIP_STB = (OOIP_in3 / IN3_PER_FT3) / FT3_PER_RB / max(Boi, 1e-12)
    cum_oil_STB = np.zeros(Nt)

    Sw_hist[0] = Sw
    p_hist[0] = p

    for n in range(Nt - 1):
        krw, kro = rel_perm(Sw, Swr, Sorw, krw_star, kro_star, nw, no)
        lam_w = krw / mu_w
        lam_o = kro / mu_o
        lam_t = lam_w + lam_o
        ct = cphi + Sw * cw + (1.0 - Sw) * co

        A = np.zeros((Nx, Nx))
        RHS = np.zeros(Nx)

        T01 = 6.328e-4 * k_md * upstream(lam_t[0], lam_t[1], p[0], p[1]) * area / dx
        acc0 = VR * phi * ct[0] / dt
        A[0, 0] = -(T01 + acc0)
        A[0, 1] = T01
        RHS[0] = -acc0 * p[0] - q_inj

        for i in range(1, Nx - 1):
            TL = 6.328e-4 * k_md * upstream(lam_t[i - 1], lam_t[i], p[i - 1], p[i]) * area / dx
            TR = 6.328e-4 * k_md * upstream(lam_t[i], lam_t[i + 1], p[i], p[i + 1]) * area / dx
            acc = VR * phi * ct[i] / dt
            A[i, i - 1] = TL
            A[i, i] = -(TL + TR + acc)
            A[i, i + 1] = TR
            RHS[i] = -acc * p[i]

        i = Nx - 1
        TL = 6.328e-4 * k_md * upstream(lam_t[i - 1], lam_t[i], p[i - 1], p[i]) * area / dx
        acc = VR * phi * ct[i] / dt
        WI = 6.328e-4 * k_md * lam_t[i] * area / (dx / 2.0)
        A[i, i - 1] = TL
        A[i, i] = -(TL + acc + WI)
        RHS[i] = -acc * p[i] - WI * p_well

        p_new = np.linalg.solve(A, RHS)

        Sw_new = Sw.copy()
        Tw01 = 6.328e-4 * k_md * upstream(lam_w[0], lam_w[1], p_new[0], p_new[1]) * area / dx
        Sw_new[0] = (
            (1.0 + (cphi + cw) * (p_new[0] - p[0])) * Sw[0]
            + (dt / (VR * phi)) * (Tw01 * (p_new[1] - p_new[0]) + q_inj)
        )

        for i in range(1, Nx - 1):
            TwL = 6.328e-4 * k_md * upstream(lam_w[i - 1], lam_w[i], p_new[i - 1], p_new[i]) * area / dx
            TwR = 6.328e-4 * k_md * upstream(lam_w[i], lam_w[i + 1], p_new[i], p_new[i + 1]) * area / dx
            Sw_new[i] = (
                (1.0 + (cphi + cw) * (p_new[i] - p[i])) * Sw[i]
                + (dt / (VR * phi)) * (TwR * (p_new[i + 1] - p_new[i]) - TwL * (p_new[i] - p_new[i - 1]))
            )

        i = Nx - 1
        TwL = 6.328e-4 * k_md * upstream(lam_w[i - 1], lam_w[i], p_new[i - 1], p_new[i]) * area / dx
        Tw_out = 6.328e-4 * k_md * lam_w[i] * area / (dx / 2.0)
        Sw_new[i] = (
            (1.0 + (cphi + cw) * (p_new[i] - p[i])) * Sw[i]
            + (dt / (VR * phi)) * (Tw_out * (p_well - p_new[i]) - TwL * (p_new[i] - p_new[i - 1]))
        )

        Sw_new = np.clip(Sw_new, Swr, 1.0 - Sorw)

        krwN, kroN = rel_perm(np.array([Sw_new[-1]]), Swr, Sorw, krw_star, kro_star, nw, no)
        lam_wN = float(krwN[0] / mu_w)
        lam_oN = float(kroN[0] / mu_o)
        lam_tN = lam_wN + lam_oN
        WI_new = 6.328e-4 * k_md * lam_tN * area / (dx / 2.0)

        qt = WI_new * (p_new[-1] - p_well)
        if lam_tN > 0.0:
            qw = qt * (lam_wN / lam_tN)
            qo = qt * (lam_oN / lam_tN)
        else:
            qw, qo = 0.0, 0.0

        Sw = Sw_new
        p = p_new
        Sw_hist[n + 1] = Sw
        p_hist[n + 1] = p
        A_hist[n + 1] = A
        RHS_hist[n + 1] = RHS
        qo_hist[n + 1] = qo
        qw_hist[n + 1] = qw

        qo_stbday = in3min_to_stbday(qo, Boi)
        qo_stbmin = qo_stbday / 1440.0
        cum_oil_STB[n + 1] = cum_oil_STB[n] + qo_stbmin * dt

    qo_STBday = np.array([in3min_to_stbday(q, Boi) for q in qo_hist])
    qw_STBday = np.array([in3min_to_stbday(q, Boi) for q in qw_hist])
    RF = np.where(OOIP_STB > 0.0, cum_oil_STB / OOIP_STB, 0.0)

    return {"t": t, "x_centers": x_centers, "dx": dx, "D": D, "Sw": Sw_hist, "p": p_hist, "A": A_hist, "RHS": RHS_hist, "qo_STBday": qo_STBday, "qw_STBday": qw_STBday, "RF": RF}

def compact_line_plot(x, y, xlabel, ylabel, height=2.2, marker=False):
    fig, ax = plt.subplots(figsize=(5.8, height))
    y = np.asarray(y, dtype=float)
    if marker:
        ax.plot(x, y, marker="o", ms=3)
    else:
        ax.plot(x, y)
    if y.size > 0 and np.all(np.isfinite(y)):
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        span = y_max - y_min
        if span < 1e-12:
            pad = max(abs(y_min) * 0.02, 1e-6)
        else:
            pad = max(span * 0.08, abs((y_max + y_min) * 0.5) * 0.005, 1e-6)
        ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig

st.title("IMPES Core Flood Dashboard")
st.caption("Editable inputs, working IMPES engine, compact plots, and a stable animation.")

col_left, col_right = st.columns([1.15, 0.85], gap="large")

with col_left:
    edited = st.data_editor(default_table(), use_container_width=True, num_rows="fixed", hide_index=True)
    run = st.button("Run simulation", type="primary")

with col_right:
    st.subheader("Display options")
    show_dense_lines = st.checkbox("Show dense overlays for 1.1 and 1.2", value=True)
    show_matrix = st.checkbox("Show A matrix and RHS in inspector", value=True)

param_map = dict(zip(edited["Parameter"], edited["Value"]))

if "results" not in st.session_state:
    st.session_state["results"] = simulate_impes(param_map)
if run:
    st.cache_data.clear()
    st.session_state["results"] = simulate_impes(param_map)

res = st.session_state["results"]
t = res["t"]
x_centers = res["x_centers"]
Sw_hist = res["Sw"]
p_hist = res["p"]
A_hist = res["A"]
RHS_hist = res["RHS"]
qo_STBday = res["qo_STBday"]
qw_STBday = res["qw_STBday"]
RF = res["RF"]
dx = res["dx"]
D = res["D"]

tab1, tab2 = st.tabs(["Deliverables", "Animated Inspector"])

with tab1:
    st.subheader("Deliverables 1.1–1.6")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**1.1 Sw(x) at every time step**")
        fig_sw_hm = go.Figure(data=go.Heatmap(x=x_centers, y=t, z=Sw_hist, colorscale=[[0.0, "black"], [1.0, "blue"]], colorbar=dict(title="Sw")))
        fig_sw_hm.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="x (in)", yaxis_title="time (min)")
        st.plotly_chart(fig_sw_hm, use_container_width=True)
        if show_dense_lines:
            fig = plt.figure(figsize=(5.8, 2.25))
            ax = fig.add_subplot(111)
            for n in range(Sw_hist.shape[0]):
                ax.plot(x_centers, Sw_hist[n, :], linewidth=0.7, alpha=0.35)
            ax.set_xlabel("x (in)")
            ax.set_ylabel("Sw")
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig)
    with c2:
        st.markdown("**1.2 p(x) at every time step**")
        fig_p_hm = go.Figure(data=go.Heatmap(x=x_centers, y=t, z=p_hist, colorscale="Viridis", colorbar=dict(title="psi")))
        fig_p_hm.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="x (in)", yaxis_title="time (min)")
        st.plotly_chart(fig_p_hm, use_container_width=True)
        if show_dense_lines:
            fig = plt.figure(figsize=(5.8, 2.25))
            ax = fig.add_subplot(111)
            for n in range(p_hist.shape[0]):
                ax.plot(x_centers, p_hist[n, :], linewidth=0.7, alpha=0.35)
            ax.set_xlabel("x (in)")
            ax.set_ylabel("p (psi)")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.markdown("**1.3 First block vs time**")
        st.pyplot(compact_line_plot(t, p_hist[:, 0], "time (min)", "p1 (psi)", marker=True))
        st.pyplot(compact_line_plot(t, Sw_hist[:, 0], "time (min)", "Sw1", marker=True))
    with c4:
        st.markdown("**1.4 Last block vs time**")
        st.pyplot(compact_line_plot(t, p_hist[:, -1], "time (min)", "p12 (psi)", marker=True))
        st.pyplot(compact_line_plot(t, Sw_hist[:, -1], "time (min)", "Sw12", marker=True))

    c5, c6 = st.columns(2, gap="large")
    with c5:
        st.markdown("**1.5 Production rates**")
        st.pyplot(compact_line_plot(t, qo_STBday, "time (min)", "qo (STB/day)", height=2.3, marker=True))
        st.pyplot(compact_line_plot(t, qw_STBday, "time (min)", "qw (STB/day)", height=2.3, marker=True))
    with c6:
        st.markdown("**1.6 Recovery factor**")
        st.pyplot(compact_line_plot(t, RF, "time (min)", "RF", marker=True))

with tab2:
    st.subheader("Animated Inspector")
    st.caption("Stable 2D animation: bottom blue = water, top black = oil, thin colored strip = pressure.")
    H = D
    YMAX = H + 0.18 * H
    p_min = float(np.min(p_hist))
    p_max = float(np.max(p_hist))

    def pressure_color(value: float) -> str:
        frac = 0.5 if p_max - p_min < 1e-12 else (value - p_min) / (p_max - p_min)
        frac = max(0.0, min(1.0, frac))
        colors = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]
        idx = min(int(frac * (len(colors) - 1)), len(colors) - 1)
        return colors[idx]

    def frame_shapes(n: int):
        shapes = []
        for i in range(len(x_centers)):
            x0 = i * dx
            x1 = (i + 1) * dx
            sw = float(Sw_hist[n, i])
            water_h = max(0.0, min(H, sw * H))
            shapes.append(dict(type="rect", x0=x0, x1=x1, y0=H, y1=YMAX, fillcolor=pressure_color(float(p_hist[n, i])), line=dict(width=0)))
            shapes.append(dict(type="rect", x0=x0, x1=x1, y0=0.0, y1=water_h, fillcolor="blue", line=dict(width=0)))
            shapes.append(dict(type="rect", x0=x0, x1=x1, y0=water_h, y1=H, fillcolor="black", line=dict(width=0)))
            shapes.append(dict(type="line", x0=x0, x1=x1, y0=H, y1=H, line=dict(width=1.3, color="rgba(255,255,255,0.55)")))
        for b in range(len(x_centers) + 1):
            xb = b * dx
            shapes.append(dict(type="line", x0=xb, x1=xb, y0=0.0, y1=YMAX, line=dict(width=2, color="rgba(255,255,255,0.65)")))
        return shapes

    fig_anim = go.Figure()
    fig_anim.add_trace(go.Scatter(x=[], y=[]))
    fig_anim.update_layout(
    height=280,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(range=[0.0, x_centers[-1] + dx / 2.0], title="core length (in)"),
    yaxis=dict(range=[0.0, YMAX], title="fill / pressure strip", showgrid=False, zeroline=False),
    shapes=frame_shapes(0),

    # Global text color (helps slider labels/current value)
    font=dict(color="black"),

    updatemenus=[dict(
        type="buttons",
        x=0.0,
        y=1.18,

        # Button styling so text is always readable
        bgcolor="rgba(30,30,30,0.85)",          # dark background
        bordercolor="rgba(255,255,255,0.7)",
        borderwidth=1,
        font=dict(color="white"),               # white text

        buttons=[
            dict(
                label="Play",
                method="animate",
                args=[None, {"frame": {"duration": 220, "redraw": True},
                             "fromcurrent": True,
                             "transition": {"duration": 0}}]
            ),
            dict(
                label="Pause",
                method="animate",
                args=[[None], {"frame": {"duration": 0, "redraw": True},
                               "mode": "immediate",
                               "transition": {"duration": 0}}]
            ),
        ],
    )],

    sliders=[dict(
        x=0.02,
        y=-0.12,
        len=0.96,

        # Slider text styling
        font=dict(color="black"),
        currentvalue=dict(prefix="time step n = ", font=dict(color="black")),

        steps=[
            dict(
                method="animate",
                args=[[str(n)], {"mode": "immediate",
                                 "frame": {"duration": 0, "redraw": True},
                                 "transition": {"duration": 0}}],
                label=str(n)
            )
            for n in range(len(t))
        ],
    )],
)


    
    fig_anim.frames = [go.Frame(name=str(n), layout=dict(shapes=frame_shapes(n))) for n in range(len(t))]
    st.plotly_chart(fig_anim, use_container_width=True)

    c7, c8 = st.columns([1.2, 0.8], gap="large")
    with c7:
        fig_hm = go.Figure(data=go.Heatmap(x=x_centers, y=t, z=p_hist, colorscale="Viridis", colorbar=dict(title="psi")))
        fig_hm.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="x (in)", yaxis_title="time (min)")
        st.plotly_chart(fig_hm, use_container_width=True)
    with c8:
        n_idx = st.slider("Select time index n", 0, len(t) - 1, 0, 1)
        st.metric("time (min)", f"{t[n_idx]:.1f}")
        st.metric("p12 (psi)", f"{p_hist[n_idx, -1]:.4f}")
        st.metric("Sw12", f"{Sw_hist[n_idx, -1]:.4f}")
        st.metric("qo (STB/day)", f"{qo_STBday[n_idx]:.4f}")
        st.metric("qw (STB/day)", f"{qw_STBday[n_idx]:.4f}")

    if show_matrix:
        with st.expander("Pressure matrix A and RHS at selected time"):
            st.dataframe(pd.DataFrame(A_hist[n_idx]), use_container_width=True, height=250)
            st.dataframe(pd.DataFrame(RHS_hist[n_idx], columns=["RHS"]), use_container_width=True, height=250)
