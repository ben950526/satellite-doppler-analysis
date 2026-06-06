import streamlit as st
from bisect import bisect_left
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
    step_minutes = st.slider(
        "取樣間隔 (分鐘)",
        1,
        720,
        10,
        help="LEO 衛星軌道週期約 90 分鐘；建議使用 5 到 10 分鐘，12 小時會嚴重漏看幾何變化。"
    )
    comm_threshold = st.slider("可通訊距離門檻 (km)", 500, 5000, 2000, step=100)
    smooth_window = st.slider("徑向速度平滑程度", 3, 15, 7)
    display_window = st.selectbox("圖表顯示範圍", ["全部", "1 天", "7 天", "30 天"], index=0)
    display_start_input = st.date_input("圖表起始日期", datetime(2023, 11, 11).date())
    
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
                        if lines[i].startswith('1 ') and lines[i + 1].startswith('2 '):
                            line1 = lines[i]
                            line2 = lines[i + 1]
                            i += 2
                        elif i + 2 < len(lines) and lines[i + 1].startswith('1 ') and lines[i + 2].startswith('2 '):
                            line1 = lines[i + 1]
                            line2 = lines[i + 2]
                            i += 3
                        else:
                            i += 1
                            continue

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
                    tles.sort(key=lambda x: x['epoch'])
                    return tles

                tle_list1 = load_all_tles(path1)
                tle_list2 = load_all_tles(path2)

                if not tle_list1 or not tle_list2:
                    st.error("TLE 檔案解析失敗：請確認檔案包含標準 line 1 / line 2 TLE。")
                    st.stop()

                def build_satellite_selector(tle_list, sat_name):
                    epochs = [item['epoch'] for item in tle_list]
                    cache = {}

                    def select(dt):
                        idx = bisect_left(epochs, dt)
                        if idx == 0:
                            selected_idx = 0
                        elif idx == len(epochs):
                            selected_idx = len(epochs) - 1
                        else:
                            before = epochs[idx - 1]
                            after = epochs[idx]
                            selected_idx = idx if abs(after - dt) < abs(dt - before) else idx - 1

                        if selected_idx not in cache:
                            tle = tle_list[selected_idx]
                            cache[selected_idx] = EarthSatellite(tle['line1'], tle['line2'], sat_name, ts)
                        return cache[selected_idx], epochs[selected_idx], selected_idx

                    return select

                select_sat1 = build_satellite_selector(tle_list1, "SAT1")
                select_sat2 = build_satellite_selector(tle_list2, "SAT2")

                st.caption(
                    f"衛星1 TLE：{len(tle_list1)} 筆，{tle_list1[0]['epoch'].date()} 到 {tle_list1[-1]['epoch'].date()}；"
                    f"衛星2 TLE：{len(tle_list2)} 筆，{tle_list2[0]['epoch'].date()} 到 {tle_list2[-1]['epoch'].date()}"
                )

                start_date = datetime.combine(start_date_input, datetime.min.time(), tzinfo=utc)
                end_date = datetime.combine(end_date_input, datetime.min.time(), tzinfo=utc)
                step_seconds = step_minutes * 60

                st.info(
                    f"📅 分析期間：{start_date.date()} 到 {end_date.date()} | "
                    f"取樣間隔：{step_minutes} 分鐘 | 中心頻率：{center_freq_ghz} GHz"
                )
                if step_minutes > 30:
                    st.warning(
                        "目前取樣間隔超過 30 分鐘，只適合看長期趨勢。"
                        "LEO 衛星的短週期距離、徑向速度與 Doppler 波形會發生 aliasing，"
                        "建議用 5 到 10 分鐘檢查圖形是否合理。"
                    )

                # ====================== 計算 ======================
                all_times = []
                all_distances = []
                all_radial_vel = []
                all_doppler = []
                all_tle_age1_days = []
                all_tle_age2_days = []
                all_tle_index1 = []
                all_tle_index2 = []

                progress_bar = st.progress(0)
                total_seconds = (end_date - start_date).total_seconds()
                if total_seconds <= 0:
                    st.error("結束日期必須晚於開始日期。")
                    st.stop()

                num_steps = int(total_seconds / step_seconds) + 1
                dt_list = [start_date + timedelta(seconds=i * step_seconds) for i in range(num_steps)]
                if dt_list[-1] < end_date:
                    dt_list.append(end_date)
                times_batch = ts.utc(dt_list)

                for step_index, (t, dt) in enumerate(zip(times_batch, dt_list), start=1):
                    try:
                        sat1, epoch1, tle_index1 = select_sat1(dt)
                        sat2, epoch2, tle_index2 = select_sat2(dt)

                        pos1 = sat1.at(t).position.km
                        vel1 = sat1.at(t).velocity.km_per_s
                        pos2 = sat2.at(t).position.km
                        vel2 = sat2.at(t).velocity.km_per_s

                        diff_pos = pos1 - pos2
                        dist = np.linalg.norm(diff_pos)

                        rel_vel = vel1 - vel2
                        if dist > 1e-6:
                            range_rate = np.dot(rel_vel, diff_pos) / dist
                            radial_vel = -range_rate                    # 正值 = 正在靠近
                        else:
                            radial_vel = 0.0

                        # 使用使用者設定的中心頻率計算 Doppler
                        fc = center_freq_ghz * 1e9
                        doppler = (radial_vel / 299792.458) * fc

                        all_times.append(t.utc_datetime())
                        all_distances.append(dist)
                        all_radial_vel.append(radial_vel)
                        all_doppler.append(doppler)
                        all_tle_age1_days.append(abs((dt - epoch1).total_seconds()) / 86400)
                        all_tle_age2_days.append(abs((dt - epoch2).total_seconds()) / 86400)
                        all_tle_index1.append(tle_index1)
                        all_tle_index2.append(tle_index2)
                    except:
                        pass

                    progress_bar.progress(min(int(step_index / len(dt_list) * 100), 100))

                # 轉換資料
                all_distances = np.array(all_distances)
                all_radial_vel = np.array(all_radial_vel)
                all_doppler = np.array(all_doppler)
                all_tle_age1_days = np.array(all_tle_age1_days)
                all_tle_age2_days = np.array(all_tle_age2_days)
                all_tle_index1 = np.array(all_tle_index1)
                all_tle_index2 = np.array(all_tle_index2)

                if len(all_times) == 0:
                    st.error("沒有產生任何分析資料點，請確認 TLE 時間範圍與分析日期。")
                    st.stop()

                max_tle_age = max(float(np.max(all_tle_age1_days)), float(np.max(all_tle_age2_days)))
                if max_tle_age > 14:
                    st.warning(
                        f"部分時間點使用的 TLE 與分析時間相差最遠 {max_tle_age:.1f} 天。"
                        "LEO 衛星使用太舊的 TLE 會讓距離與 Doppler 明顯失真，建議上傳涵蓋分析期間的歷史 TLE。"
                    )

                in_range = all_distances <= comm_threshold
                if not np.any(in_range):
                    st.warning(
                        "目前取樣點沒有任何距離低於通訊門檻的時刻。"
                        "Doppler 圖仍會顯示全時段頻移；若要找近距離通訊窗口，請縮小取樣間隔或提高距離門檻。"
                    )

                st.success("✅ 分析完成！")

                # ====================== 畫圖（保留你原本的 Plotly） ======================
                fc = center_freq_ghz * 1e9
                radial_vel_smooth = uniform_filter1d(all_radial_vel, size=smooth_window, mode='nearest')
                doppler_smooth_khz = (radial_vel_smooth / 299792.458) * fc / 1000
                tle_changed = np.r_[False, (np.diff(all_tle_index1) != 0) | (np.diff(all_tle_index2) != 0)]
                radial_jump = np.r_[False, np.abs(np.diff(radial_vel_smooth)) > 1.0]
                doppler_jump = np.r_[False, np.abs(np.diff(doppler_smooth_khz)) > 100.0]
                plot_break = tle_changed & (radial_jump | doppler_jump)
                radial_vel_plot = radial_vel_smooth.copy()
                doppler_plot = doppler_smooth_khz.copy()
                radial_vel_plot[plot_break] = np.nan
                doppler_plot[plot_break] = np.nan

                # 建立 DataFrame
                df_plot = pd.DataFrame({
                    'Time (UTC)': all_times,
                    'Distance (km)': all_distances,
                    'Radial Velocity (km/s)': all_radial_vel,
                    'Doppler Shift (kHz)': np.array(all_doppler)/1000,
                    'Radial Velocity Smoothed (km/s)': radial_vel_smooth,
                    'Doppler Shift Smoothed (kHz)': doppler_smooth_khz,
                    'Within Threshold': in_range,
                    'TLE Switched': tle_changed,
                    'Plot Break': plot_break,
                    'SAT1 TLE Age (days)': all_tle_age1_days,
                    'SAT2 TLE Age (days)': all_tle_age2_days
                })

                display_days = {
                    "1 天": 1,
                    "7 天": 7,
                    "30 天": 30
                }.get(display_window)

                if display_days is None:
                    df_display = df_plot
                else:
                    display_start = datetime.combine(display_start_input, datetime.min.time(), tzinfo=utc)
                    display_start = max(display_start, start_date)
                    display_end = min(display_start + timedelta(days=display_days), end_date)
                    df_display = df_plot[
                        (df_plot['Time (UTC)'] >= display_start)
                        & (df_plot['Time (UTC)'] <= display_end)
                    ]

                    if df_display.empty:
                        st.warning("目前圖表顯示範圍沒有資料，已改顯示完整分析範圍。")
                        df_display = df_plot
                    else:
                        st.info(f"目前圖表顯示：{display_start.date()} 到 {display_end.date()}（{display_window}）")

                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    subplot_titles=("Distance", "Radial Relative Velocity", "Doppler Shift")
                )

                # 距離圖
                fig.add_trace(go.Scatter(x=df_display['Time (UTC)'], y=df_display['Distance (km)'],
                                       mode='lines', name='Distance', line=dict(color='blue')), row=1, col=1)
                fig.add_hline(y=comm_threshold, line_dash="dash", line_color="red",
                              annotation_text=f"{comm_threshold} km Threshold", row=1, col=1)

                # 徑向速度
                fig.add_trace(go.Scatter(x=df_display['Time (UTC)'], y=radial_vel_plot[df_display.index],
                                       mode='lines', name='Radial Velocity (Smoothed)', line=dict(color='orange')), row=2, col=1)

                # Doppler
                fig.add_trace(go.Scatter(x=df_display['Time (UTC)'], y=doppler_plot[df_display.index],
                                       mode='lines', name='Doppler Shift (Smoothed)', line=dict(color='green')), row=3, col=1)
                display_marker_mask = df_display['Within Threshold'] & ~df_display['Plot Break']
                fig.add_trace(go.Scatter(
                    x=df_display.loc[display_marker_mask, 'Time (UTC)'],
                    y=df_display.loc[display_marker_mask, 'Doppler Shift Smoothed (kHz)'],
                    mode='markers',
                    name='Doppler Within Threshold',
                    marker=dict(color='red', size=4)
                ), row=3, col=1)

                # 重要：強制顯示 X 軸時間
                fig.update_xaxes(
                    title_text="Time (UTC)",
                    row=3,
                    col=1,
                    showticklabels=True,
                    tickformat="%Y-%m-%d %H:%M"
                )
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

