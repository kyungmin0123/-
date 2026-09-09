import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 기온 예측기")
st.write("서울의 연평균기온 데이터를 바탕으로 회귀분석을 수행하고 미래 기온을 예측합니다.")

# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

try:
    df = pd.read_csv(DATA_URL, encoding="utf-8")

except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.error(e)
    st.stop()

# --------------------------------------------------
# 데이터 전처리
# --------------------------------------------------

# 날짜를 datetime으로 변환
df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

# 평균기온을 숫자로 변환
df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

# 날짜 또는 평균기온이 없는 행 제거
df = df.dropna(subset=["날짜", "평균기온"])

# 연도 추출
df["연도"] = df["날짜"].dt.year

# 2025년 이후 자료 제외
df = df[df["연도"] <= 2025]

# --------------------------------------------------
# 연도별 관측일 수와 평균기온 계산
# --------------------------------------------------

yearly = (
    df.groupby("연도")
    .agg(
        평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

# 관측일이 300일 이상인 해만 사용
yearly = yearly[yearly["관측일수"] >= 300].copy()

# 연도순 정렬
yearly = yearly.sort_values("연도").reset_index(drop=True)

# 데이터가 충분하지 않은 경우
if len(yearly) < 2:
    st.error("회귀분석을 수행할 수 있는 데이터가 충분하지 않습니다.")
    st.stop()

# --------------------------------------------------
# 회귀분석
# --------------------------------------------------

X = yearly[["연도"]]
y = yearly["평균기온"]

model = LinearRegression()
model.fit(X, y)

# 회귀선에 사용할 값
yearly["회귀예측"] = model.predict(X)

# 회귀식 계수
slope = model.coef_[0]
intercept = model.intercept_

# 상관계수
correlation = yearly["연도"].corr(yearly["평균기온"])

# --------------------------------------------------
# 슬라이더
# --------------------------------------------------

st.subheader("📅 예상 연도 선택")

selected_year = st.slider(
    "예측할 연도",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

# 선택한 연도의 예상 평균기온
predicted_temp = model.predict(
    np.array([[selected_year]])
)[0]

# --------------------------------------------------
# 예상 기온 크게 표시
# --------------------------------------------------

st.metric(
    label=f"{selected_year}년 예상 연평균기온",
    value=f"{predicted_temp:.2f} °C"
)

# --------------------------------------------------
# 회귀분석 정보
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "회귀에 사용된 연도 수",
        f"{len(yearly)}년"
    )

with col2:
    st.metric(
        "시작 연도",
        f"{int(yearly['연도'].min())}년"
    )

with col3:
    st.metric(
        "끝 연도",
        f"{int(yearly['연도'].max())}년"
    )

with col4:
    st.metric(
        "상관계수",
        f"{correlation:.3f}"
    )

# --------------------------------------------------
# 회귀식 표시
# --------------------------------------------------

st.write(
    f"**회귀식:** 평균기온 = "
    f"{slope:.4f} × 연도 + {intercept:.2f}"
)

st.caption(
    "※ 1900~2100년 예측값은 관측자료에 맞춘 선형회귀를 바탕으로 한 추정값입니다."
)

# --------------------------------------------------
# Plotly 그래프
# --------------------------------------------------

fig = go.Figure()

# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        text=[
            f"{year}년<br>평균기온: {temp:.2f}℃<br>관측일수: {days}일"
            for year, temp, days in zip(
                yearly["연도"],
                yearly["평균기온"],
                yearly["관측일수"]
            )
        ],
        hovertemplate="%{text}<extra></extra>",
        marker=dict(size=7)
    )
)

# 회귀직선
regression_years = np.linspace(
    yearly["연도"].min(),
    yearly["연도"].max(),
    200
)

regression_temps = model.predict(
    regression_years.reshape(-1, 1)
)

fig.add_trace(
    go.Scatter(
        x=regression_years,
        y=regression_temps,
        mode="lines",
        name="회귀직선",
        line=dict(width=3)
    )
)

# 선택한 연도의 예측값
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예상값",
        marker=dict(size=14, symbol="star")
    )
)

fig.update_layout(
    title="서울 연평균기온과 회귀직선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="x unified",
    height=600,
    template="plotly_white"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# --------------------------------------------------
# 사용 데이터 표
# --------------------------------------------------

with st.expander("📊 회귀분석에 사용된 연도별 데이터 보기"):
    display_df = yearly.copy()
    display_df["평균기온"] = display_df["평균기온"].round(2)
    display_df["회귀예측"] = display_df["회귀예측"].round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# 데이터 조건 안내
# --------------------------------------------------

st.info(
    "분석 조건: 2025년까지의 자료만 사용하며, "
    "각 연도의 관측일수가 300일 이상인 경우에만 회귀분석에 포함했습니다."
)
