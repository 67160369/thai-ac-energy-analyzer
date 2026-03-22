import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# =========================
# โหลดและเตรียมข้อมูล
# =========================
df = pd.read_csv("dataset2/weather-energy-data-update.csv")
df = df.dropna()
df = df[df['month'].isin([1,2,3,4,5,6,7,8,9,10,11,12])]
df['month'] = df['month'].astype(int)

def get_season(m):
    if m in [12, 1, 2]:  return 'Winter'
    elif m in [3, 4, 5]: return 'Spring'
    elif m in [6, 7, 8]: return 'Summer'
    else:                return 'Autumn'

df['season'] = df['month'].apply(get_season)

season_order  = ['Winter', 'Spring', 'Summer', 'Autumn']
season_labels = {'Winter': 'Winter ❄', 'Spring': 'Spring 🌸',
                 'Summer': 'Summer ☀', 'Autumn': 'Autumn 🍂'}
season_colors = {'Winter': '#5B9BD5', 'Spring': '#70AD47',
                 'Summer': '#FF6B35', 'Autumn': '#C9A84C'}

season_stats = df.groupby('season').agg(
    avg_temp=('temp_dry','mean'),
    avg_kwh=('kWh','mean'),
    max_kwh=('kWh','max'),
    total_kwh=('kWh','sum')
).reindex(season_order)

# หาฤดูที่ใช้ไฟเยอะสุด/น้อยสุด
max_season = season_stats['avg_kwh'].idxmax()
min_season = season_stats['avg_kwh'].idxmin()

# =========================
# Figure หลัก
# =========================
fig = plt.figure(figsize=(22, 26))
fig.patch.set_facecolor('#F8F9FA')
fig.suptitle("Seasonal Energy Analysis\nTemperature vs Electricity Consumption",
             fontsize=20, fontweight='bold', y=0.99)

gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.38,
                       top=0.95, bottom=0.04, left=0.06, right=0.97)

# ==========================================
# ROW 0: Stats Cards + annotation สรุป
# ==========================================
ax_cards = fig.add_subplot(gs[0, :])
ax_cards.axis('off')

for i, season in enumerate(season_order):
    row = season_stats.loc[season]
    x   = 0.10 + i * 0.22
    c   = season_colors[season]
    lbl = season_labels[season]

    rect = plt.Rectangle((x-0.09, 0.05), 0.18, 0.88,
                          transform=ax_cards.transAxes, color=c, alpha=0.15, zorder=1)
    ax_cards.add_patch(rect)
    rect2 = plt.Rectangle((x-0.09, 0.05), 0.18, 0.88,
                           transform=ax_cards.transAxes,
                           fill=False, edgecolor=c, linewidth=2.5, zorder=2)
    ax_cards.add_patch(rect2)

    # badge ฤดูที่ใช้ไฟเยอะสุด/น้อยสุด
    if season == max_season:
        ax_cards.text(x, 0.97, '🔥 ใช้ไฟเยอะสุด', ha='center', va='center',
                      transform=ax_cards.transAxes, fontsize=8,
                      color='white', fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='#e74c3c', alpha=0.9))
    elif season == min_season:
        ax_cards.text(x, 0.97, '✅ ใช้ไฟน้อยสุด', ha='center', va='center',
                      transform=ax_cards.transAxes, fontsize=8,
                      color='white', fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='#27ae60', alpha=0.9))

    ax_cards.text(x, 0.82, lbl, ha='center', va='center',
                  transform=ax_cards.transAxes, fontsize=13, fontweight='bold', color=c)
    ax_cards.text(x, 0.62, "Avg Temp", ha='center', va='center',
                  transform=ax_cards.transAxes, fontsize=9, color='gray')
    ax_cards.text(x, 0.48, f"{row['avg_temp']:.1f} °C", ha='center', va='center',
                  transform=ax_cards.transAxes, fontsize=16, fontweight='bold', color='#333')
    ax_cards.text(x, 0.30, "Avg kWh/hr", ha='center', va='center',
                  transform=ax_cards.transAxes, fontsize=9, color='gray')
    ax_cards.text(x, 0.15, f"{row['avg_kwh']:.3f} kWh", ha='center', va='center',
                  transform=ax_cards.transAxes, fontsize=16, fontweight='bold', color=c)

# ==========================================
# ROW 1: Scatter แยกฤดู ครบ 4 กราฟ
# ==========================================
scatter_positions = [(1,0),(1,1),(1,2),(1,3)]
for i, season in enumerate(season_order):
    ax = fig.add_subplot(gs[scatter_positions[i][0], scatter_positions[i][1]])
    sdf = df[df['season'] == season]
    c   = season_colors[season]
    lbl = season_labels[season]

    ax.scatter(sdf['temp_dry'], sdf['kWh'], alpha=0.15, s=5, color=c)

    z = np.polyfit(sdf['temp_dry'], sdf['kWh'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(sdf['temp_dry'].min(), sdf['temp_dry'].max(), 100)
    ax.plot(x_line, p(x_line), color='black', linewidth=2, linestyle='--', label='Trend')

    ax.set_title(lbl, fontsize=11, fontweight='bold', color=c)
    ax.set_xlabel("Temperature (°C)", fontsize=9)
    ax.set_ylabel("kWh", fontsize=9)
    ax.set_facecolor('#FAFAFA')

    r = np.corrcoef(sdf['temp_dry'], sdf['kWh'])[0, 1]
    ax.text(0.05, 0.92, f"r = {r:.3f}", transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    # annotation trend direction
    direction = '↑ ยิ่งร้อนใช้มากขึ้น' if z[0] > 0 else '↓ ยิ่งร้อนใช้น้อยลง'
    ax.text(0.05, 0.80, direction, transform=ax.transAxes, fontsize=8, color='#555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fffde7', alpha=0.85))

# ==========================================
# ROW 2: Hourly Pattern + Monthly avg
# ==========================================
ax_hourly = fig.add_subplot(gs[2, :3])
peak_hours = {}
for season in season_order:
    sdf    = df[df['season'] == season]
    hourly = sdf.groupby('hour')['kWh'].mean()
    c      = season_colors[season]
    lbl    = season_labels[season]
    ax_hourly.plot(hourly.index, hourly.values, color=c, linewidth=2.5,
                   marker='o', markersize=4, label=lbl)
    peak_h = hourly.idxmax()
    peak_v = hourly.max()
    peak_hours[season] = (peak_h, peak_v)
    ax_hourly.annotate(f'Peak {int(peak_h)}:00\n{peak_v:.3f}kWh',
                       xy=(peak_h, peak_v),
                       xytext=(peak_h + 0.8, peak_v + 0.012),
                       fontsize=7.5, color=c, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color=c, lw=1.2))

ax_hourly.set_title("Hourly Energy Pattern by Season (with Peak Annotations)",
                    fontsize=12, fontweight='bold')
ax_hourly.set_xlabel("Hour of Day", fontsize=10)
ax_hourly.set_ylabel("Avg kWh", fontsize=10)
ax_hourly.legend(loc='upper left', fontsize=9)
ax_hourly.set_xticks(range(0, 24, 1))
ax_hourly.set_xticklabels([f'{h}:00' for h in range(24)], rotation=45, fontsize=7)
ax_hourly.set_facecolor('#FAFAFA')
ax_hourly.grid(axis='y', alpha=0.3)

# Monthly bar
ax_monthly = fig.add_subplot(gs[2, 3])
monthly = df.groupby('month')['kWh'].mean()
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
month_colors = [season_colors[get_season(int(m))] for m in monthly.index]
ax_monthly.bar(range(len(monthly)), monthly.values, color=month_colors, width=0.7)
ax_monthly.set_title("Monthly Avg kWh", fontsize=12, fontweight='bold')
ax_monthly.set_xticks(range(len(monthly)))
ax_monthly.set_xticklabels(month_names, fontsize=8, rotation=30)
ax_monthly.set_ylabel("Avg kWh", fontsize=10)
ax_monthly.set_facecolor('#FAFAFA')
legend_elements = [Patch(facecolor=season_colors[s], label=season_labels[s])
                   for s in season_order]
ax_monthly.legend(handles=legend_elements, fontsize=7, loc='upper right')

# ==========================================
# ROW 3: Boxplot + Temp Bin + Season Summary
# ==========================================
ax_box = fig.add_subplot(gs[3, 0])
box_data = [df[df['season'] == s]['kWh'].values for s in season_order]
bp = ax_box.boxplot(box_data, patch_artist=True, showfliers=False)
for patch, season in zip(bp['boxes'], season_order):
    patch.set_facecolor(season_colors[season])
    patch.set_alpha(0.7)
ax_box.set_title("kWh Distribution by Season", fontsize=11, fontweight='bold')
ax_box.set_xticklabels([season_labels[s] for s in season_order], fontsize=7, rotation=10)
ax_box.set_ylabel("kWh", fontsize=10)
ax_box.set_facecolor('#FAFAFA')

ax_bin = fig.add_subplot(gs[3, 1])
bins   = [-20, -10, 0, 10, 20, 30]
labels = ['<-10°C', '-10~0°C', '0~10°C', '10~20°C', '20~30°C']
df['temp_bin'] = pd.cut(df['temp_dry'], bins=bins, labels=labels)
bin_avg = df.groupby('temp_bin', observed=True)['kWh'].mean()
bin_colors = ['#5B9BD5','#85C1E9','#70AD47','#FF6B35','#E74C3C']
bars_bin = ax_bin.bar(range(len(bin_avg)), bin_avg.values, color=bin_colors, width=0.6)
ax_bin.set_title("Avg kWh by Temperature Range", fontsize=11, fontweight='bold')
ax_bin.set_xticks(range(len(bin_avg)))
ax_bin.set_xticklabels(labels, fontsize=8, rotation=10)
ax_bin.set_ylabel("Avg kWh", fontsize=10)
ax_bin.set_facecolor('#FAFAFA')
for j, val in enumerate(bin_avg.values):
    ax_bin.text(j, val + 0.003, f'{val:.3f}', ha='center', fontsize=8, fontweight='bold')

# Season Summary — เพิ่มตัวเลขบน bar
ax_sum = fig.add_subplot(gs[3, 2:])
s_avg_kwh  = season_stats['avg_kwh']
s_avg_temp = season_stats['avg_temp']
x     = np.arange(len(season_order))
width = 0.35
ax_sum2 = ax_sum.twinx()

b1 = ax_sum.bar(x - width/2, s_avg_kwh.values, width,
                color=[season_colors[s] for s in season_order], alpha=0.85, label='Avg kWh')
b2 = ax_sum2.bar(x + width/2, s_avg_temp.values, width,
                 color=[season_colors[s] for s in season_order], alpha=0.35, label='Avg Temp °C')

# เพิ่มตัวเลขบน bar
for bar, val in zip(b1, s_avg_kwh.values):
    ax_sum.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}\nkWh', ha='center', fontsize=8, fontweight='bold', color='#333')

for bar, val in zip(b2, s_avg_temp.values):
    ypos = bar.get_height() + 0.2 if val >= 0 else bar.get_height() - 0.8
    ax_sum2.text(bar.get_x() + bar.get_width()/2, ypos,
                 f'{val:.1f}°C', ha='center', fontsize=8, color='gray')

ax_sum.set_title("Season Summary: Avg kWh vs Avg Temperature",
                 fontsize=11, fontweight='bold')
ax_sum.set_xticks(x)
ax_sum.set_xticklabels([season_labels[s] for s in season_order], fontsize=9)
ax_sum.set_ylabel("Avg kWh", fontsize=10, color='#333')
ax_sum2.set_ylabel("Avg Temp (°C)", fontsize=10, color='gray')
ax_sum.set_facecolor('#FAFAFA')
ax_sum.legend(loc='upper left', fontsize=9)
ax_sum2.legend(loc='upper right', fontsize=9)

plt.savefig("seasonal_analysis_v2.png", dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("✅ seasonal_analysis.png สำเร็จ")