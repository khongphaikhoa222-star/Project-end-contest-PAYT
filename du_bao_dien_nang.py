import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from dateutil.easter import easter
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

TRAIN_END = pd.Timestamp("2009-12-31 23:00")    # train: đến hết 2009, test: năm 2010
MAX_GAP_GIO = 3                                 # chỉ nội suy khoảng thiếu tối đa 3 giờ
CAC_COT_BAT_BUOC = ["Date", "Time", "Global_active_power"]


# ============ 0. CHỌN FILE DỮ LIỆU (cửa sổ File Explorer) ============
def chon_file():
    """Ưu tiên: đường dẫn truyền qua dòng lệnh -> hộp thoại chọn file -> nhập tay."""
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        return sys.argv[1]
    duong_dan = ""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)          # đưa hộp thoại lên trên cùng
        duong_dan = filedialog.askopenfilename(
            title="Chọn file dữ liệu điện năng (household_power_consumption)",
            initialdir=os.getcwd(),
            filetypes=[("Dữ liệu văn bản", "*.txt *.csv"), ("Tất cả các file", "*.*")],
        )
        root.destroy()
    except Exception:                               # môi trường không có giao diện
        duong_dan = input("Nhập đường dẫn tới file dữ liệu: ").strip().strip('"')
    if not duong_dan or not os.path.isfile(duong_dan):
        sys.exit("Chưa chọn file hợp lệ. Thoát chương trình.")
    return duong_dan


DATA_PATH = chon_file()
THU_MUC_HINH = os.path.join(os.path.dirname(os.path.abspath(DATA_PATH)), "bieu_do")
os.makedirs(THU_MUC_HINH, exist_ok=True)
print("Đã chọn file:", DATA_PATH)

# ============ 1. ĐỌC & LÀM SẠCH DỮ LIỆU ============
df = pd.read_csv(DATA_PATH, sep=";", na_values=["?"], low_memory=False)
thieu_cot = [c for c in CAC_COT_BAT_BUOC if c not in df.columns]
if thieu_cot:
    sys.exit(f"File không đúng định dạng, thiếu cột: {thieu_cot}. "
             f"Các cột hiện có: {list(df.columns)}")

df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
df = df.drop(columns=["Date", "Time"]).set_index("datetime").sort_index()
print("Kích thước:", df.shape, "| Từ", df.index.min(), "đến", df.index.max())
print("Tỷ lệ thiếu (%):\n", (df.isna().mean() * 100).round(2))

# Trung bình công suất (kW) trong 1 giờ = số kWh dùng trong giờ đó
hourly = df["Global_active_power"].resample("h").mean()

# Chỉ điền khoảng thiếu NGẮN (<= MAX_GAP_GIO giờ), khoảng dài giữ NaN
la_nan = hourly.isna()
do_dai = la_nan.groupby((la_nan != la_nan.shift()).cumsum()).transform("sum")
hourly_clean = hourly.where(~(la_nan & (do_dai <= MAX_GAP_GIO)), hourly.interpolate(method="time"))
print("Số giờ còn thiếu (khoảng dài):", hourly_clean.isna().sum())

# Ngoại lai: Z-score và IQR (chỉ để quan sát, vẫn giữ vì là giờ cao điểm thật)
x = hourly_clean.dropna()
q1, q3 = x.quantile(0.25), x.quantile(0.75)
print(f"Ngoại lai Z-score: {(np.abs((x - x.mean()) / x.std()) > 3).sum()} điểm | "
      f"IQR: {((x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))).sum()} điểm")

# ============ 2. KHÁM PHÁ DỮ LIỆU ============
data = hourly_clean.to_frame("y")
data["hour"], data["dow"], data["month"] = data.index.hour, data.index.dayofweek, data.index.month
data["is_weekend"] = (data["dow"] >= 5).astype(int)

for k in [1, 24, 168]:
    print(f"Tương quan y(t) với y(t-{k}h): {data['y'].corr(data['y'].shift(k)):.3f}")

# ============ 3. KIỂM ĐỊNH WELCH T-TEST (tự cài) ============
def welch(a, b):
    n1, n2, v1, v2 = len(a), len(b), a.var(), b.var()
    t = (a.mean() - b.mean()) / np.sqrt(v1 / n1 + v2 / n2)
    dfw = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    d = (a.mean() - b.mean()) / np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return t, 2 * stats.t.sf(abs(t), dfw), d      # t, p-value, Cohen's d

daily = data["y"].resample("D").mean().dropna()
cuoi_tuan = daily[daily.index.dayofweek >= 5]; ngay_thuong = daily[daily.index.dayofweek < 5]
dong = daily[daily.index.month.isin([12, 1, 2])]; he = daily[daily.index.month.isin([6, 7, 8])]
kd = {}
for ten, a, b in [("Cuối tuần vs ngày thường", cuoi_tuan, ngay_thuong), ("Mùa đông vs mùa hè", dong, he)]:
    kd[ten] = welch(a, b)
    print(f"{ten}: t={kd[ten][0]:.2f} (scipy {stats.ttest_ind(a, b, equal_var=False).statistic:.2f}), "
          f"p={kd[ten][1]:.2e}, d={kd[ten][2]:.2f}")

# ============ 4. TẠO ĐẶC TRƯNG ============
def ngay_le_phap(n1, n2):
    ds = []
    for n in range(n1, n2 + 1):
        ds += [pd.Timestamp(n, m, d) for m, d in [(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)]]
        e = pd.Timestamp(easter(n))
        ds += [e + pd.Timedelta(days=k) for k in (1, 39, 50)]
    return pd.DatetimeIndex(ds)

LE = ngay_le_phap(2006, 2011)

s, idx = hourly_clean, hourly_clean.index
feat = pd.DataFrame({"y": s})
for ten, gt, ck in [("hour", idx.hour, 24), ("dow", idx.dayofweek, 7), ("month", idx.month - 1, 12)]:
    feat[ten + "_sin"] = np.sin(2 * np.pi * gt / ck)
    feat[ten + "_cos"] = np.cos(2 * np.pi * gt / ck)
feat["is_weekend"] = (idx.dayofweek >= 5).astype(int)
feat["is_holiday"] = idx.normalize().isin(LE).astype(int)
feat["lag_1"], feat["lag_24"], feat["lag_168"] = s.shift(1), s.shift(24), s.shift(168)
feat["roll24_lag24"] = s.shift(24).rolling(24).mean()     # không nhìn trộm tương lai

CAL = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend", "is_holiday"]
DAY_AHEAD = CAL + ["lag_24", "lag_168", "roll24_lag24"]
HOUR_AHEAD = DAY_AHEAD + ["lag_1"]
feat = feat.dropna(subset=["y"] + HOUR_AHEAD)

train, test = feat[feat.index <= TRAIN_END], feat[feat.index > TRAIN_END]
if len(train) == 0 or len(test) == 0:
    sys.exit("Dữ liệu không có đủ giai đoạn train (≤ 2009) và test (2010). "
             "Hãy kiểm tra lại file hoặc chỉnh TRAIN_END.")
y_tr, y_te = train["y"].values, test["y"].values
print(f"Train: {len(train)} giờ | Test: {len(test)} giờ")

# ============ 5. MÔ HÌNH HỒI QUY ============
def chuan_bi(cols):
    """Z-score theo train + thêm cột bias toàn số 1."""
    mu, sd = train[cols].mean(), train[cols].std(ddof=0).replace(0, 1)
    return [np.c_[np.ones(len(d)), ((d[cols] - mu) / sd).values] for d in (train, test)]

def hoi_quy_dai_so(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)          # giải X^T X θ = X^T y

def gradient_descent(X, y, lr=0.05, n_iter=5000):
    theta = np.zeros(X.shape[1])
    for _ in range(n_iter):
        theta -= lr * (2 / len(y)) * (X.T @ (X @ theta - y))
    return theta

def danh_gia(y, yh):
    mse = np.mean((y - yh) ** 2)
    return {"MAE": np.mean(np.abs(y - yh)), "MSE": mse, "RMSE": np.sqrt(mse),
            "R2": 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)}

Xtr_d, Xte_d = chuan_bi(DAY_AHEAD)
Xtr_c, Xte_c = chuan_bi(CAL)
Xtr_h, Xte_h = chuan_bi(HOUR_AHEAD)
theta = hoi_quy_dai_so(Xtr_d, y_tr)
theta_gd = gradient_descent(Xtr_d, y_tr)
print("Chênh lệch dự báo lớn nhất giữa đại số và GD:", np.abs(Xte_d @ (theta - theta_gd)).max().round(5), "kW")

du_bao = {
    "Baseline: hôm qua": test["lag_24"].values,
    "Baseline: tuần trước": test["lag_168"].values,
    "Hồi quy chỉ lịch": Xte_c @ hoi_quy_dai_so(Xtr_c, y_tr),
    "NumPy đại số (24h)": Xte_d @ theta,
    "NumPy GD (24h)": Xte_d @ theta_gd,
    "NumPy đại số (1h)": Xte_h @ hoi_quy_dai_so(Xtr_h, y_tr),
    "sklearn LinearRegression (24h)": LinearRegression(fit_intercept=False).fit(Xtr_d, y_tr).predict(Xte_d),
    "Gradient Boosting (24h)": HistGradientBoostingRegressor(random_state=42).fit(Xtr_d[:, 1:], y_tr).predict(Xte_d[:, 1:]),
}
bang = pd.DataFrame({k: danh_gia(y_te, v) for k, v in du_bao.items()}).T.round(4)
print(bang)

# Giờ mô hình dự báo kém nhất
res = pd.Series((y_te - du_bao["NumPy đại số (24h)"]) ** 2, index=test.index)
rmse_gio = res.groupby(test.index.hour).mean() ** 0.5
print(f"Giờ khó dự báo nhất: {rmse_gio.idxmax()}h (RMSE {rmse_gio.max():.3f} kW)")

# ============ 6. PHÁT HIỆN GIỜ CAO ĐIỂM (phân loại) ============
nguong = np.quantile(y_tr, 0.90)

def phan_loai(yt, yp):
    t, p = yt > nguong, yp > nguong
    TP, FP, FN = (t & p).sum(), (~t & p).sum(), (t & ~p).sum()
    prec = TP / (TP + FP) if TP + FP else 0
    rec = TP / (TP + FN) if TP + FN else 0
    return {"Accuracy": (t == p).mean(), "Precision": prec, "Recall": rec,
            "F1": 2 * prec * rec / (prec + rec) if prec + rec else 0}

cls = {k: phan_loai(y_te, du_bao[k]) for k in ["Baseline: hôm qua", "NumPy đại số (24h)", "NumPy đại số (1h)", "Gradient Boosting (24h)"]}
cls["Luôn đoán bình thường"] = phan_loai(y_te, np.zeros_like(y_te))   # cho thấy Accuracy dễ đánh lừa
print(f"Ngưỡng cao điểm (P90 của train): {nguong:.2f} kW")
print(pd.DataFrame(cls).T.round(3))

# ============ 7. XU HƯỚNG THEO THÁNG & DỰ BÁO 12 THÁNG TỚI ============
thang_tb = hourly_clean.resample("MS").mean()
du_so_lieu = hourly_clean.resample("MS").count() / hourly_clean.resample("MS").size() >= 0.9
m = thang_tb[du_so_lieu].dropna()                      # bỏ tháng đầu/cuối bị cụt
goc = m.index[0]

def thiet_ke(idx):
    t = ((idx.year - goc.year) * 12 + (idx.month - goc.month)).astype(float)
    dummy = np.eye(12)[idx.month - 1]                  # 12 cột one-hot theo tháng (đóng vai trò hệ số tự do)
    return np.c_[t, dummy]                             # cột 0: xu hướng; 12 cột sau: hiệu ứng mùa

beta, *_ = np.linalg.lstsq(thiet_ke(m.index), m.values, rcond=None)
fit_m = thiet_ke(m.index) @ beta
r2_m = danh_gia(m.values, fit_m)["R2"]
sd_res = np.std(m.values - fit_m, ddof=13)

tuong_lai = pd.date_range(m.index[-1] + pd.offsets.MonthBegin(1), periods=12, freq="MS")
dub_tl = thiet_ke(tuong_lai) @ beta
print(f"R² dữ liệu tháng: {r2_m:.3f} (số tháng dùng: {len(m)})")
print("Dự báo 12 tháng tới (kW):\n", pd.Series(dub_tl, index=tuong_lai.strftime("%Y-%m")).round(3).to_string())

# ============ 8. NHẬN XÉT DỮ LIỆU ============
theo_gio = data.groupby("hour")["y"].mean()
theo_thu = data.groupby("dow")["y"].mean()
theo_thang = data.groupby("month")["y"].mean()
TEN_THU = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"]
KHUNG = {"Đêm (0-6h)": range(0, 6), "Sáng (6-12h)": range(6, 12),
         "Chiều (12-18h)": range(12, 18), "Tối (18-24h)": range(18, 24)}
tb_khung = pd.Series({k: theo_gio[list(v)].mean() for k, v in KHUNG.items()})
tb_chung = data["y"].mean()

print("\n" + "=" * 12 + " NHẬN XÉT DỮ LIỆU " + "=" * 12)

print(f"\n[1] Theo giờ trong ngày (trung bình chung {tb_chung:.2f} kW)")
print(f"- Dùng NHIỀU nhất lúc {theo_gio.idxmax()}h ({theo_gio.max():.2f} kW), "
      f"dùng ÍT nhất lúc {theo_gio.idxmin()}h ({theo_gio.min():.2f} kW), "
      f"chênh nhau {theo_gio.max() / theo_gio.min():.1f} lần.")
print(f"- Khung giờ nhiều nhất: {tb_khung.idxmax()} ({tb_khung.max():.2f} kW); "
      f"ít nhất: {tb_khung.idxmin()} ({tb_khung.min():.2f} kW).")

print("\n[2] Theo ngày trong tuần")
print(f"- Nhiều nhất: {TEN_THU[theo_thu.idxmax()]} ({theo_thu.max():.2f} kW); "
      f"ít nhất: {TEN_THU[theo_thu.idxmin()]} ({theo_thu.min():.2f} kW).")
p_tuan = kd["Cuối tuần vs ngày thường"]
print(f"- Cuối tuần TB {cuoi_tuan.mean():.2f} kW, ngày thường TB {ngay_thuong.mean():.2f} kW "
      f"(cuối tuần {'cao' if cuoi_tuan.mean() > ngay_thuong.mean() else 'thấp'} hơn "
      f"{abs(cuoi_tuan.mean() / ngay_thuong.mean() - 1):.0%}); p={p_tuan[1]:.1e} -> "
      f"{'khác biệt có ý nghĩa thống kê' if p_tuan[1] < 0.05 else 'chưa đủ bằng chứng khác biệt'}.")

print("\n[3] Theo mùa / tháng")
print(f"- Nhiều nhất: tháng {theo_thang.idxmax()} ({theo_thang.max():.2f} kW); "
      f"ít nhất: tháng {theo_thang.idxmin()} ({theo_thang.min():.2f} kW).")
print(f"- Mùa đông (12-2) TB {dong.mean():.2f} kW so với mùa hè (6-8) {he.mean():.2f} kW: "
      f"đông {'cao' if dong.mean() > he.mean() else 'thấp'} hơn {abs(dong.mean() / he.mean() - 1):.0%} "
      f"(p={kd['Mùa đông vs mùa hè'][1]:.1e}, d={kd['Mùa đông vs mùa hè'][2]:.2f}).")

tb_gan = m.iloc[-12:].mean()                  # 12 tháng gần nhất có dữ liệu
tb_tl = dub_tl.mean()
doi = (tb_tl / tb_gan - 1) * 100
huong = "TĂNG" if beta[0] > 0 else "GIẢM"
print("\n[4] Xu hướng & dự báo năm tới "
      f"({tuong_lai[0]:%m/%Y} → {tuong_lai[-1]:%m/%Y})")
print(f"- Xu hướng dài hạn (đã trừ mùa): {huong} {abs(beta[0] * 12):.3f} kW/năm.")
print(f"- Dự báo TB năm tới {tb_tl:.2f} kW, so với 12 tháng gần nhất {tb_gan:.2f} kW ({doi:+.1f}%).")
print(f"- Tháng dự báo dùng nhiều nhất: {tuong_lai[dub_tl.argmax()]:%m/%Y} ({dub_tl.max():.2f} kW); "
      f"ít nhất: {tuong_lai[dub_tl.argmin()]:%m/%Y} ({dub_tl.min():.2f} kW).")
print(f"- Khoảng bất định thô mỗi tháng: ±{1.96 * sd_res:.2f} kW (R² mô hình tháng = {r2_m:.2f}).")
print("- Lưu ý: chỉ có khoảng 4 năm dữ liệu của MỘT hộ gia đình, nên dự báo dài hạn chỉ mang tính tham khảo.")

mh, base = bang.loc["NumPy đại số (24h)"], bang.loc[["Baseline: hôm qua", "Baseline: tuần trước"], "RMSE"].min()
print("\n[5] Chất lượng mô hình dự báo theo giờ")
print(f"- Mô hình chính (trước 24h): MAE={mh['MAE']:.3f}, RMSE={mh['RMSE']:.3f} kW, R²={mh['R2']:.3f}; "
      f"giảm RMSE {(1 - mh['RMSE'] / base) * 100:.1f}% so với baseline tốt nhất.")
print(f"- Giờ cao điểm: Precision={cls['NumPy đại số (24h)']['Precision']:.2f}, "
      f"Recall={cls['NumPy đại số (24h)']['Recall']:.2f}, F1={cls['NumPy đại số (24h)']['F1']:.2f}.")

# ============ 9. VẼ BIỂU ĐỒ ============
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10})
MAU = {"thuc": "#1f2937", "mh": "#2563eb", "base": "#f59e0b", "tl": "#dc2626", "xanh": "#10b981"}
MODEL_CHINH = "NumPy đại số (24h)"

def luu(fig, ten):
    fig.tight_layout()
    duong_dan = os.path.join(THU_MUC_HINH, ten)
    fig.savefig(duong_dan, dpi=150, bbox_inches="tight")
    print("  đã lưu:", duong_dan)

print("\n" + "=" * 12 + " BIỂU ĐỒ " + "=" * 12)

# --- Hình 1: tổng quan chuỗi thời gian ---
fig1, ax = plt.subplots(2, 1, figsize=(12, 7))
ax[0].plot(daily.index, daily.values, lw=0.6, color=MAU["mh"], alpha=0.6, label="Trung bình ngày")
ax[0].plot(daily.rolling(30, min_periods=15).mean(), lw=1.8, color=MAU["tl"], label="Trung bình trượt 30 ngày")
ax[0].axvline(TRAIN_END, color="gray", ls="--"); ax[0].text(TRAIN_END, ax[0].get_ylim()[1] * 0.95, " test →", color="gray")
ax[0].set(title="Công suất tiêu thụ trung bình theo ngày", ylabel="kW"); ax[0].legend()
ax[1].plot(m.index, m.values, "o-", color=MAU["thuc"], ms=3, label="Thực tế (tháng)")
ax[1].plot(m.index, fit_m, color=MAU["mh"], label=f"Mô hình xu hướng + mùa (R²={r2_m:.2f})")
ax[1].plot(tuong_lai, dub_tl, "o--", color=MAU["tl"], ms=4, label="Dự báo 12 tháng tới")
ax[1].fill_between(tuong_lai, dub_tl - 1.96 * sd_res, dub_tl + 1.96 * sd_res, color=MAU["tl"], alpha=0.15, label="Khoảng bất định ~95%")
ax[1].set(title="Xu hướng theo tháng & dự báo 12 tháng tới", ylabel="kW"); ax[1].legend(ncol=2)
luu(fig1, "01_tong_quan_va_du_bao_thang.png")

# --- Hình 2: mẫu hình tiêu thụ ---
fig2, ax = plt.subplots(2, 2, figsize=(13, 8))
ax[0, 0].plot(theo_gio.index, theo_gio.values, "o-", color=MAU["mh"])
ax[0, 0].set(title="Trung bình theo giờ trong ngày", xlabel="Giờ", ylabel="kW", xticks=range(0, 24, 3))
ax[0, 1].bar(TEN_THU, theo_thu.values, color=[MAU["tl"] if i >= 5 else MAU["mh"] for i in range(7)])
ax[0, 1].set(title="Trung bình theo ngày trong tuần (đỏ = cuối tuần)", ylabel="kW")
ax[0, 1].tick_params(axis="x", rotation=30)
ax[1, 0].bar(theo_thang.index, theo_thang.values, color=MAU["xanh"])
ax[1, 0].set(title="Trung bình theo tháng", xlabel="Tháng", ylabel="kW", xticks=range(1, 13))
pv = data.pivot_table(values="y", index="dow", columns="hour", aggfunc="mean")
im = ax[1, 1].imshow(pv.values, aspect="auto", cmap="YlOrRd")
ax[1, 1].set(title="Nhiệt đồ: thứ × giờ (kW)", xlabel="Giờ", yticks=range(7), yticklabels=TEN_THU)
ax[1, 1].grid(False); fig2.colorbar(im, ax=ax[1, 1])
luu(fig2, "02_mau_hinh_tieu_thu.png")

# --- Hình 3: phân phối & tự tương quan ---
fig3, ax = plt.subplots(1, 3, figsize=(15, 4.5))
ax[0].hist(x.values, bins=60, color=MAU["mh"], alpha=0.85)
ax[0].axvline(nguong, color=MAU["tl"], ls="--", label=f"Ngưỡng cao điểm P90 = {nguong:.2f}")
ax[0].set(title="Phân phối công suất theo giờ", xlabel="kW", ylabel="Số giờ"); ax[0].legend()
box_data = [daily[daily.index.dayofweek < 5].values, daily[daily.index.dayofweek >= 5].values]
ax[1].boxplot(box_data, patch_artist=True,
              boxprops=dict(facecolor="#bfdbfe"), medianprops=dict(color=MAU["tl"]))
ax[1].set_xticklabels(["Ngày thường", "Cuối tuần"])   # tránh tham số labels/tick_labels khác nhau giữa các bản matplotlib
t_, p_, d_ = kd["Cuối tuần vs ngày thường"]
ax[1].set(title=f"Cuối tuần vs ngày thường (p={p_:.1e}, d={d_:.2f})", ylabel="kW (TB ngày)")
y_ser = data["y"]
lags = range(1, 200)
ax[2].plot(list(lags), [y_ser.corr(y_ser.shift(k)) for k in lags], color=MAU["thuc"], lw=1)
for k in (24, 168):
    ax[2].axvline(k, color=MAU["tl"], ls=":", alpha=0.8); ax[2].text(k + 1, 0.9, f"{k}h", color=MAU["tl"])
ax[2].set(title="Tự tương quan theo độ trễ", xlabel="Độ trễ (giờ)", ylabel="Hệ số tương quan")
luu(fig3, "03_phan_phoi_va_tu_tuong_quan.png")

# --- Hình 4: thực tế vs dự báo ---
n_tuan = min(168, len(test))
sl = slice(0, n_tuan)
fig4, ax = plt.subplots(2, 2, figsize=(14, 8))
ax[0, 0].plot(test.index[sl], y_te[sl], color=MAU["thuc"], lw=1.6, label="Thực tế")
ax[0, 0].plot(test.index[sl], du_bao[MODEL_CHINH][sl], color=MAU["mh"], lw=1.3, label=MODEL_CHINH)
ax[0, 0].plot(test.index[sl], du_bao["Baseline: hôm qua"][sl], color=MAU["base"], lw=1, ls="--", label="Baseline: hôm qua")
ax[0, 0].set(title="Dự báo vs thực tế – tuần đầu của tập test", ylabel="kW"); ax[0, 0].legend(fontsize=8)
ax[0, 0].tick_params(axis="x", rotation=25)
yp = du_bao[MODEL_CHINH]
ax[0, 1].scatter(y_te, yp, s=3, alpha=0.25, color=MAU["mh"])
lim = [0, max(y_te.max(), yp.max())]
ax[0, 1].plot(lim, lim, color=MAU["tl"], ls="--", label="Dự báo hoàn hảo")
ax[0, 1].set(title="Thực tế vs dự báo (từng giờ)", xlabel="Thực tế (kW)", ylabel="Dự báo (kW)"); ax[0, 1].legend()
sai_so = y_te - yp
ax[1, 0].hist(sai_so, bins=60, color=MAU["xanh"], alpha=0.85)
ax[1, 0].axvline(0, color=MAU["tl"], ls="--")
ax[1, 0].set(title=f"Phân phối sai số (TB={sai_so.mean():.3f}, SD={sai_so.std():.3f})", xlabel="Thực tế − dự báo (kW)", ylabel="Số giờ")
ax[1, 1].bar(rmse_gio.index, rmse_gio.values, color=MAU["mh"])
ax[1, 1].bar(rmse_gio.idxmax(), rmse_gio.max(), color=MAU["tl"], label=f"Khó nhất: {rmse_gio.idxmax()}h")
ax[1, 1].set(title="RMSE theo giờ trong ngày", xlabel="Giờ", ylabel="RMSE (kW)", xticks=range(0, 24, 3)); ax[1, 1].legend()
luu(fig4, "04_thuc_te_vs_du_bao.png")

# --- Hình 5: so sánh mô hình ---
fig5, ax = plt.subplots(1, 3, figsize=(16, 5))
thu_tu = bang.sort_values("RMSE")
mau_cot = [MAU["tl"] if k == MODEL_CHINH else (MAU["base"] if k.startswith("Baseline") else MAU["mh"]) for k in thu_tu.index]
ax[0].barh(thu_tu.index, thu_tu["RMSE"], color=mau_cot); ax[0].invert_yaxis()
ax[0].set(title="RMSE các mô hình (thấp hơn = tốt hơn)", xlabel="kW")
for i, v in enumerate(thu_tu["RMSE"]):
    ax[0].text(v, i, f" {v:.3f}", va="center", fontsize=8)
ax[1].barh(thu_tu.index, thu_tu["R2"], color=mau_cot); ax[1].invert_yaxis()
ax[1].set(title="R² các mô hình (cao hơn = tốt hơn)", xlabel="R²"); ax[1].set_yticklabels([])
cls_df = pd.DataFrame(cls).T
cls_df[["Accuracy", "Precision", "Recall", "F1"]].plot.bar(ax=ax[2], width=0.8)
ax[2].set(title="Phát hiện giờ cao điểm (P90)", ylim=(0, 1.05)); ax[2].tick_params(axis="x", rotation=30)
ax[2].legend(fontsize=8, ncol=2)
luu(fig5, "05_so_sanh_mo_hinh.png")

# --- Hình 6: dự báo 12 tháng tới dạng bảng cột ---
fig6, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(tuong_lai.strftime("%m/%Y"), dub_tl, yerr=1.96 * sd_res, capsize=3, color=MAU["mh"], alpha=0.85)
ax.axhline(tb_gan, color=MAU["tl"], ls="--", label=f"TB 12 tháng gần nhất = {tb_gan:.2f} kW")
ax.set(title="Dự báo công suất trung bình 12 tháng tới (±95% thô)", ylabel="kW"); ax.legend()
ax.tick_params(axis="x", rotation=30)
luu(fig6, "06_du_bao_12_thang.png")

print(f"\nHoàn tất. Toàn bộ biểu đồ được lưu trong thư mục: {THU_MUC_HINH}")
plt.show()      # hiện tất cả cửa sổ biểu đồ; đóng cửa sổ để kết thúc chương trình
