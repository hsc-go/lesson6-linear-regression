import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025      # 수업의 기준 기간 끝
MIN_DAYS = 300        # 연간 최소 관측일 수

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="centered")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_yearly() -> pd.DataFrame:
    df = pd.read_csv(URL, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year
    df = df[df["연도"] <= LAST_YEAR]  # 2025년 이후 데이터 제외

    yearly = (
        df.groupby("연도")["평균기온"]
        .agg(연평균기온="mean", 관측일수="count")
        .reset_index()
    )
    yearly = yearly[yearly["관측일수"] >= MIN_DAYS]  # 관측일 300일 미만 해 제외
    return yearly.reset_index(drop=True)


st.title("🌡️ 기온 예측기")
st.caption("서울 연평균기온으로 만든 회귀 직선으로 연도별 기온을 예측합니다.")

try:
    yearly = load_yearly()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요: {e}")
    st.stop()

if len(yearly) < 2:
    st.error("조건을 만족하는 해가 2개 미만이라 회귀 직선을 만들 수 없어요.")
    st.stop()

x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)

slope, intercept = np.polyfit(x, y, 1)
r = np.corrcoef(x, y)[0, 1]

# 최근 20년만으로 구한 직선
RECENT_YEARS = 20
recent = yearly[yearly["연도"] > yearly["연도"].max() - RECENT_YEARS]
x_rec = recent["연도"].to_numpy(dtype=float)
y_rec = recent["연평균기온"].to_numpy(dtype=float)
if len(recent) >= 2:
    slope_rec, intercept_rec = np.polyfit(x_rec, y_rec, 1)
else:
    slope_rec, intercept_rec = None, None

# 슬라이더와 예상 기온
year = st.slider("예측할 연도", min_value=1900, max_value=2100, value=2050, step=1)
pred = slope * year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1.2rem 0;">
        <div style="font-size:1.1rem; opacity:0.7;">{year}년 예상 연평균기온</div>
        <div style="font-size:4.5rem; font-weight:700; line-height:1.2;">{pred:.2f} °C</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 100년당 기울기 (크게)
st.markdown(
    f"""
    <div style="text-align:center; padding:0.6rem 0 1.2rem;">
        <div style="font-size:1.1rem; opacity:0.7;">전체 기간 추세: 100년에</div>
        <div style="font-size:3.5rem; font-weight:700; line-height:1.2; color:crimson;">
            {slope * 100:+.2f} °C
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 회귀 직선 정보
c1, c2, c3 = st.columns(3)
c1.metric("직선을 만든 해의 개수", f"{len(yearly)}개")
c2.metric("시작 연도", f"{int(yearly['연도'].min())}년")
c3.metric("끝 연도", f"{int(yearly['연도'].max())}년")

st.metric("상관계수 (r)", f"{r:.3f}")

st.subheader("기울기 비교 (100년당 변화)")


def slope_card(title, sub, value, color):
    val = "계산 불가" if value is None else f"{value * 100:+.2f} °C"
    return f"""
    <div style="text-align:center; padding:1rem 0.5rem; border:1px solid rgba(128,128,128,0.35);
                border-radius:12px;">
        <div style="font-size:1rem; font-weight:600;">{title}</div>
        <div style="font-size:0.85rem; opacity:0.7;">{sub}</div>
        <div style="font-size:2.4rem; font-weight:700; color:{color}; line-height:1.4;">{val}</div>
    </div>
    """


k1, k2 = st.columns(2)
k1.markdown(
    slope_card(
        "전체 기간",
        f"{int(x.min())}~{int(x.max())}년 ({len(yearly)}개 해)",
        slope,
        "crimson",
    ),
    unsafe_allow_html=True,
)
k2.markdown(
    slope_card(
        f"최근 {RECENT_YEARS}년",
        f"{int(x_rec.min())}~{int(x_rec.max())}년 ({len(recent)}개 해)",
        slope_rec,
        "darkorange",
    ),
    unsafe_allow_html=True,
)

if slope_rec is not None and slope != 0:
    ratio = slope_rec / slope
    st.caption(
        f"최근 {RECENT_YEARS}년의 상승 속도는 전체 기간의 약 {ratio:.1f}배예요. "
        "(기간이 짧을수록 해마다의 들쭉날쭉함에 크게 흔들리니 해석에 주의하세요.)"
    )

st.caption(
    f"회귀식: 기온 = {slope:.5f} × 연도 + {intercept:.3f}  |  "
    f"기준: {LAST_YEAR}년까지, 관측일 {MIN_DAYS}일 이상인 해만 사용"
)

# 그래프
line_x = np.array([min(x.min(), year), max(x.max(), year)])
line_y = slope * line_x + intercept

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=7, opacity=0.8),
        hovertemplate="%{x}년<br>%{y:.2f} °C<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3, color="crimson"),
    )
)
if slope_rec is not None:
    rx = np.array([x_rec.min(), x_rec.max()])
    fig.add_trace(
        go.Scatter(
            x=rx,
            y=slope_rec * rx + intercept_rec,
            mode="lines",
            name=f"최근 {RECENT_YEARS}년 직선",
            line=dict(width=3, color="darkorange", dash="dash"),
        )
    )
fig.add_trace(
    go.Scatter(
        x=[year],
        y=[pred],
        mode="markers",
        name=f"{year}년 예상",
        marker=dict(size=14, color="gold", symbol="star", line=dict(width=1, color="black")),
        hovertemplate=f"{year}년 예상<br>{pred:.2f} °C<extra></extra>",
    )
)
fig.update_layout(
    title=f"연도별 평균기온 산점도와 회귀 직선 (r = {r:.3f})",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=10, r=10, t=80, b=10),
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("연평균기온 표 보기"):
    st.dataframe(
        yearly.rename(columns={"연평균기온": "연평균기온(°C)"}).round(2),
        use_container_width=True,
        hide_index=True,
    )
