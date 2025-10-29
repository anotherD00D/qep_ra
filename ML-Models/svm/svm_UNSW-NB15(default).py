import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import ipaddress
import matplotlib.pyplot as plt
import matlab.engine

eng = matlab.engine.start_matlab()

folder = Path(r'dataset')
files = list(folder.glob("+"))
csv_files = list(folder.glob("*.csv"))
feature_file = list(folder.glob("*.xls"))

feature_df = pd.read_csv(feature_file[0], encoding='cp1252')
col_names = feature_df['Name'].astype(str).tolist()

data_df = pd.read_csv(csv_files[0], header=None, names = col_names, low_memory=False).iloc[:10000]

# if len(csv_files) > 1:
#     for file in csv_files:
#         add_df = pd.read_csv(file, header=None, names=col_names, low_memory=False)
#         data_df = pd.concat([data_df, add_df])

X = data_df.drop(['attack_cat', 'Label'], axis='columns')
y = data_df['Label']

def cbs_with_pandas(
    X: pd.DataFrame,
    y: pd.Series,
    redundancy_threshold: float = 0.8,
    redundancy_method: str = "spearman",  # 'spearman' or 'pearson'
    max_features: int | None = None,
):
    """
    CBS-like selection using pandas correlations only:
      1) Relevance = |corr(x_i, y)| via corrwith (Pearson).
      2) Greedy redundancy removal: keep feature if
         max_j |corr(x_i, x_j)| < threshold for already selected features.

    Handles categoricals by factorizing (no one-hot).
    """
    # y numeric
    if not np.issubdtype(y.dtype, np.number):
        y = pd.Series(pd.factorize(y, sort=True)[0], index=y.index)
    y = y.astype(float)

    # Encode X: keep numerics; factorize non-numerics; fill NaNs
    X_enc = pd.DataFrame(index=X.index)
    for c in X.columns:
        s = X[c]
        if np.issubdtype(s.dtype, np.number):
            X_enc[c] = s.astype(float)
            if X_enc[c].isna().any():
                X_enc[c] = X_enc[c].fillna(X_enc[c].median())
        else:
            codes, _ = pd.factorize(s, sort=True)  # NaN -> -1
            X_enc[c] = pd.Series(codes, index=X.index).astype(np.int32)

    # 1) Relevance scores (Pearson)
    relevance = X_enc.corrwith(y, method="pearson").abs().fillna(0.0)
    ranked = relevance.sort_values(ascending=False).index.tolist()

    # 2) Greedy redundancy filter
    selected = []
    for col in ranked:
        if relevance[col] <= 0:
            continue
        ok = True
        for kept in selected:
            rho = X_enc[col].corr(X_enc[kept], method=redundancy_method)
            if pd.isna(rho):
                rho = 0.0
            if abs(rho) >= redundancy_threshold:
                ok = False
                break
        if ok:
            selected.append(col)
            if max_features and len(selected) >= max_features:
                break

    return selected, X[selected]

selected_cols, X_reduced = cbs_with_pandas(
    X, y,
    redundancy_threshold=0.8,      # tighten to 0.7 to prune more
    redundancy_method="spearman",  # robust to monotonic nonlinearity
    max_features=200               # or None for no cap
)

def to_numeric_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 1) First try coercing to numeric where it mostly works
    for c in out.columns:
        s = out[c]
        if np.issubdtype(s.dtype, np.number):
            continue
        sn = pd.to_numeric(s, errors="coerce")
        # if most values became numeric, keep it
        if sn.notna().mean() >= 0.95:
            # fill a few NaNs that appeared
            out[c] = sn.fillna(sn.median())
        else:
            out[c] = s  # keep original for next step

    # 2) Convert remaining non-numeric columns:
    for c in out.select_dtypes(exclude=[np.number]).columns:
        s = out[c].astype(str)

        # If it looks like IPv4, map to integer
        is_ipv4 = s.str.match(r"^(?:\d{1,3}\.){3}\d{1,3}$", na=False).mean() > 0.9
        if is_ipv4:
            def ipv4_to_int(x):
                try:
                    return int(ipaddress.IPv4Address(x))
                except Exception:
                    return np.nan
            out[c] = s.map(ipv4_to_int).astype("float64")  # becomes numeric
            out[c] = out[c].fillna(out[c].median())
        else:
            # Fallback: factorize (categorical → integer codes)
            codes, _ = pd.factorize(out[c], sort=True)  # NaN -> -1
            out[c] = pd.Series(codes, index=out.index).astype("float64")

    # Final tidy: fill any leftovers and downcast
    out = out.apply(lambda col: col.fillna(col.median()) if col.isna().any() else col)
    return out.astype(np.float32, copy=False)

# Convert your reduced set
Xr = to_numeric_safe(X_reduced)
yr = y  # unchanged

X_train, X_test, y_train, y_test = train_test_split(
    Xr, yr, test_size=0.2, random_state=42,
    stratify=y
)

rbf_svm = make_pipeline(
    StandardScaler(),  
    SVC()              
)

rbf_svm.fit(X_train, y_train)
y_pred = rbf_svm.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, zero_division=0))

# y → numeric for correlation/plotting
if not np.issubdtype(yr.dtype, np.number):
    y_plot = pd.Series(pd.factorize(yr, sort=True)[0], index=yr.index)
else:
    y_plot = yr

# pick top-2 features by |corr(feature, y)|
relevance = Xr.corrwith(y_plot).abs().sort_values(ascending=False)
feat1, feat2 = relevance.index[:2].tolist()
print(f"[Plot] Using features: {feat1!r}, {feat2!r}  "
      f"(|corr|: {relevance[feat1]:.3f}, {relevance[feat2]:.3f})")

X2 = Xr[[feat1, feat2]]

# (optional) downsample for speed while keeping balance
PLOT_MAX = 50000
if len(X2) > PLOT_MAX:
    X2_small, _, y_small, _ = train_test_split(
        X2, y_plot, train_size=PLOT_MAX, random_state=42,
        stratify=y_plot if y_plot.nunique() > 1 else None, shuffle=True
    )
else:
    X2_small, y_small = X2, y_plot

# split and train a tiny 2-D RBF SVM (defaults)
Xtr, Xte, ytr, yte = train_test_split(
    X2_small, y_small, test_size=0.2, random_state=42,
    stratify=y_small if y_small.nunique() > 1 else None
)
svm2d = make_pipeline(StandardScaler(), SVC())  # defaults: RBF, C=1.0, gamma='scale'
svm2d.fit(Xtr, ytr)
print("[Plot] 2D holdout accuracy:", accuracy_score(yte, svm2d.predict(Xte)))

# mesh grid (smaller = faster)
x_min, x_max = X2_small[feat1].min(), X2_small[feat1].max()
y_min, y_max = X2_small[feat2].min(), X2_small[feat2].max()
pad_x = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
pad_y = 0.05 * (y_max - y_min if y_max > y_min else 1.0)

xx, yy = np.meshgrid(
    np.linspace(x_min - pad_x, x_max + pad_x, 200),
    np.linspace(y_min - pad_y, y_max + pad_y, 200)
)
grid = pd.DataFrame({feat1: xx.ravel(), feat2: yy.ravel()})
Z = svm2d.predict(grid.values).reshape(xx.shape)

# plot decision regions + points
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.25)               # decision regions
# scatter at most a few thousand points to keep rendering snappy
plot_n = min(len(X2_small), 3000)
idx = np.random.RandomState(42).choice(len(X2_small), size=plot_n, replace=False)
plt.scatter(X2_small.iloc[idx, 0], X2_small.iloc[idx, 1],
            c=y_small.iloc[idx], s=10, alpha=0.8)
plt.xlabel(feat1)
plt.ylabel(feat2)
plt.title("RBF SVM Decision Boundary on Top-2 Target-Correlated Features")
plt.tight_layout()
plt.show()
# ==== end 2-D plot block ====
eng.quit()