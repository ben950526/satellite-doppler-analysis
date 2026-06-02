import streamlit as st
from datetime import datetime, timedelta
from skyfield.api import load, EarthSatellite, utc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import uniform_filter1d
import tempfile

st.set_page_config(page_title="衛星星間通訊分析工具", layout="wide")
st.title("🛰️ 兩顆衛星星間通訊分析工具")
st.markdown("**上傳兩個 TLE 檔案，並選擇想分析的時間範圍**")

# 側邊欄
with st.sidebar:
    st.header("分析參數")
    start_date_input = st.date_input("開始日期", datetime(2023, 11, 11).date())
    end_date_input = st.date_input("結束日期", datetime(2024, 5, 11).date())
    step_hours = st.slider("取樣間隔 (小時)", 1, 24, 6)
    comm_threshold = st.slider("可通訊距離門檻 (km)", 500, 5000, 2000, step=100)
    smooth_window = st.slider("徑向速度平滑程度", 3, 15, 7)
    
    # ==================== 新增：中心頻率可調 ====================
    st.subheader("📡 通訊頻率設定")
    center_freq_ghz = st.number_input(
        "中心頻率 (GHz)", 
        min_value=1.0, 
        max_value=100.0, 
        value=30.0, 
        step=0.1,
        help="Ka-band 常用 26~40 GHz"
    )
    st.caption(f"目前使用頻率：{center_freq_ghz} GHz")

# 上傳檔案
col1, col2 = st.columns(2)
with col1:
    tle1 = st.file_uploader("衛星1 TLE 檔案", type=["txt"])
with col2:
    tle2 = st.file_uploader("衛星2 TLE 檔案", type=["txt"])

if tle1 and tle2:
    if st.button("🚀 開始分析", type="primary", use_container_width=True):
        with st.spinner("計算中，請稍候..."):
            try:
                # 儲存檔案
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f1:
                    f1.write(tle1.getvalue())
                    path1 = f1.name
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f2:
                    f2.write(tle2.getvalue())
                    path2 = f2.name

                ts = load.timescale()

                def load_all_tles(filename):
                    tles = []
                    with open(filename, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f if line.strip()]
                    i = 0
                    while i < len(lines) - 1:
                        line1 = lines[i]
                        line2 = lines[i + 1]
                        if line1.startswith('1 ') and line2.startswith('2 '):
                            try:
                                sat = EarthSatellite(line1, line2, "SAT", ts)
                                tles.append({
                                    'epoch': sat.epoch.utc_datetime(),
                                    'line1': line1,
                                    'line2': line2
                                })
                            except:
                                pass
                        i += 2
                    tles.sort(key=lambda x: x['epoch'])
                    return tles

                tle_list1 = load_all_tles(path1)
                tle_list2 = load_all_tles(path2)

                start_date = datetime.combine(start_date_input, datetime.min.time(), tzinfo=utc)
                end_date = datetime.combine(end_date_input, datetime.min.time(), tzinfo=utc)
                step_seconds = step_hours * 3600

                st.info(f"📅 分析期間：{start_date.date()} 到 {end_date.date()} | 中心頻率：{center_freq_ghz} GHz")

                # ====================== 計算 ======================
                current = start_date
                all_times = []
                all_distances = []
                all_radial_vel = []
                all_doppler = []

                progress_bar = st.progress(0)

                while current < end_date:
                    batch_end = min(current + timedelta(days=30), end_date)
                    num_steps = int((batch_end - current).total_seconds() / step_seconds) + 1
                    dt_list = [current + timedelta(seconds=i*step_seconds) for i in range(num_steps)]
                    times_batch = ts.utc(dt_list)

                    for t, dt in zip(times_batch, dt_list):
                        try:
                            sat1 = EarthSatellite(tle_list1[-1]['line1'] if dt > tle_list1[-1]['epoch'] else tle_list1[0]['line1'],
                                                 tle_list1[-1]['line2'] if dt > tle_list1[-1]['epoch'] else tle_list1[0]['line2'], "SAT1", ts)
                            sat2 = EarthSatellite(tle_list2[-1]['line1'] if dt > tle_list2[-1]['epoch'] else tle_list2[0]['line1'],
                                                 tle_list2[-1]['line2'] if dt > tle_list2[-1]['epoch'] else tle_list2[0]['line2'], "SAT2", ts)

                            pos1 = sat1.at(t).position.km
                            vel1 = sat1.at(t).velocity.km_per_s
                            pos2 = sat2.at(t).position.km
                            vel2 = sat2.at(t).velocity.km_per_s

                            diff_pos = pos1 - pos2
                            dist = np.linalg.norm(diff_pos)

                            rel_vel = vel1 - vel2
                            if dist > 500.0:
                                radial_vel = np.dot(rel_vel, diff_pos) / dist
                                radial_vel = -radial_vel                    # 正值 = 正在靠近
                                radial_vel = np.clip(radial_vel, -1.5, 1.5)
                            else:
                                radial_vel = 0.0

                            # 使用使用者設定的中心頻率計算 Doppler
                            fc = center_freq_ghz * 1e9
                            doppler = (radial_vel / 299792.458) * fc

                            all_times.append(t.utc_datetime())
                            all_distances.append(dist)
                            all_radial_vel.append(radial_vel)
                            all_doppler.append(doppler if dist < comm_threshold else np.nan)
                        except:
                            pass

                    progress = (current - start_date).total_seconds() / (end_date - start_date).total_seconds()
                    progress_bar.progress(min(int(progress*100), 100))
                    current = batch_end

                # 轉換資料
                all_distances = np.array(all_distances)
                all_radial_vel = np.array(all_radial_vel)
                all_doppler = np.array(all_doppler)

                st.success("✅ 分析完成！")

                # ====================== 畫圖（保留你原本的 Plotly） ======================
                # 建立 DataFrame
                df_plot = pd.DataFrame({
                    'Time (UTC)': all_times,
                    'Distance (km)': all_distances,
                    'Radial Velocity (km/s)': all_radial_vel,
                    'Doppler Shift (kHz)': np.array(all_doppler)/1000
                })

                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    subplot_titles=("Distance", "Radial Relative Velocity", "Doppler Shift")
                )

                # 距離圖
                fig.add_trace(go.Scatter(x=df_plot['Time (UTC)'], y=df_plot['Distance (km)'],
                                       mode='lines', name='Distance', line=dict(color='blue')), row=1, col=1)
                fig.add_hline(y=comm_threshold, line_dash="dash", line_color="red",
                              annotation_text=f"{comm_threshold} km Threshold", row=1, col=1)

                # 徑向速度
                radial_vel_smooth = uniform_filter1d(all_radial_vel, size=smooth_window, mode='nearest')
                fig.add_trace(go.Scatter(x=df_plot['Time (UTC)'], y=radial_vel_smooth,
                                       mode='lines', name='Radial Velocity (Smoothed)', line=dict(color='orange')), row=2, col=1)

                # Doppler
                dopp_plot = np.where(all_distances < comm_threshold, all_doppler/1000, np.nan)
                fig.add_trace(go.Scatter(x=df_plot['Time (UTC)'], y=dopp_plot,
                                       mode='lines', name='Doppler Shift', line=dict(color='green')), row=3, col=1)

                # 重要：強制顯示 X 軸時間
                fig.update_xaxes(title_text="Time (UTC)", row=3, col=1, showticklabels=True, tickformat="%Y-%m-%d %H:%M")
                fig.update_xaxes(showticklabels=True, tickformat="%Y-%m-%d", row=1, col=1)
                fig.update_xaxes(showticklabels=True, tickformat="%Y-%m-%d", row=2, col=1)

                fig.update_layout(
                    height=950,
                    title_text="衛星星間通訊分析<br>(數值皆為衛星1 相對於衛星2)<br>滑鼠懸停可查看精確數值",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                fig.update_yaxes(title_text="Distance (km)", row=1, col=1)
                fig.update_yaxes(title_text="Radial Velocity (km/s)", row=2, col=1)
                fig.update_yaxes(title_text="Doppler Shift (kHz)", row=3, col=1)

                st.plotly_chart(fig, use_container_width=True)

                # 數據表格與下載
                st.subheader("📋 原始數據表格")
                st.dataframe(df_plot.round(4), use_container_width=True)

                csv = df_plot.to_csv(index=False).encode('utf-8')
                st.download_button("📥 下載完整數據 CSV", csv, "satellite_analysis.csv", "text/csv")

            except Exception as e:
                st.error(f"發生錯誤：{e}")

else:
    st.info("請上傳兩個 TLE 檔案")

st.caption("太空專題分析工具 | 中心頻率可自訂")