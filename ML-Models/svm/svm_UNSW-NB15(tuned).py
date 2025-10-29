import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import ipaddress
import matplotlib.pyplot as plt
import matlab.engine

# eng = matlab.engine.start_matlab()

folder = Path(r'dataset')
csv_files = list(folder.glob("*.csv"))
feature_file = list(folder.glob("*.xls"))

feature_df = pd.read_csv(feature_file[0], encoding='cp1252')
col_names = feature_df['Name'].astype(str).tolist()

data_df = pd.read_csv(csv_files[0], header=None, names = col_names, low_memory=False).iloc[:10000]

if len(csv_files) > 1:
    for file in csv_files:
        add_df = pd.read_csv(file, header=None, names=col_names, low_memory=False)
        data_df = pd.concat([data_df, add_df])

data_df = data_df.iloc[:10000]

X = data_df.drop(['attack_cat', 'Label'], axis='columns')
y = data_df['Label']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

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

sel_cols, X_train_sel = cbs_with_pandas(
    X_train_raw, y_train,
    redundancy_threshold=0.8,
    redundancy_method="spearman",
    max_features=200
)
X_test_sel = X_test_raw[sel_cols]

X_train_num = to_numeric_safe(X_train_sel)
X_test_num  = to_numeric_safe(X_test_sel)

scaler = StandardScaler()
Xs_train = scaler.fit_transform(X_train_num)
Xs_test  = scaler.transform(X_test_num)

var_std = float(np.var(Xs_train))
std_dev = float(np.std(Xs_train))         
gamma   = 1.0 / (Xs_train.shape[1] * var_std)
C       = 1.0 / std_dev
print(f"[Scaled-like] var={var_std:.6g} -> gamma={gamma:.6g}, C={C:.6g}")

svc = SVC(kernel="rbf", gamma=gamma, C=C)
svc.fit(Xs_train, y_train)
y_pred = svc.predict(Xs_test)
print(classification_report(y_test, y_pred, zero_division=0))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")