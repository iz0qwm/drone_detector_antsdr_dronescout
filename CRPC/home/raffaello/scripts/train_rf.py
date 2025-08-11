# /home/raffaello/scripts/train_rf.py
import re, yaml, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

CFG = yaml.safe_load(open('/home/raffaello/config/settings.yaml'))
CSV = CFG['paths']['datasets']['rf']
OUT = Path(CFG['paths']['models']['rf'])

# --- Load ---
df = pd.read_csv(CSV)

# --- Label engineering: maker dal campo "type" ---
def norm_maker(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).strip())
    maps = {
        "JR PROPO": "JR",
        "RadioMaster": "RADIOMASTER",
        "Radiolink": "RADIOLINK",
        "YunZhuo": "YUNZHUO",
        "DAUTEL": "AUTEL",  # conferma se non è refuso
    }
    for k, v in maps.items():
        if s.upper().startswith(k.upper()):
            return v
    return s.split(" ")[0].upper() if s else "UNKNOWN"

df["maker"] = df["type"].map(norm_maker)

y = df["maker"]
X = df.drop(columns=["type","maker"], errors="ignore")

# Drop colonne quasi vuote
if "vts_bw" in X.columns and X["vts_bw"].isna().mean() > 0.5:
    X = X.drop(columns=["vts_bw"])

# Merge maker rari (<3 campioni) -> OTHER
vc = y.value_counts()
print("Makers (before):\n", vc)

rare = vc[vc < 3].index.tolist()
if rare:
    print("\nMerging rare makers into 'OTHER':", rare)
    y = y.where(~y.isin(rare), "OTHER")

vc_after = y.value_counts()
print("\nMakers (after):\n", vc_after)

# --- Se dopo il merge resta 1 sola classe, allena e salva, stop ---
if len(vc_after) < 2:
    print("\n⚠ Una sola classe dopo il merge: alleno su tutto e salvo.")
    final_pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(with_mean=False),
        RandomForestClassifier(
            class_weight="balanced_subsample",
            n_estimators=300,
            min_samples_leaf=2,
            random_state=0
        )
    )
    final_pipe.fit(X, y)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipe, OUT)
    print("✅ Salvato modello RF in:", OUT)
    raise SystemExit(0)

# --- K-Fold stratificata (dinamica) ---
min_per_class = int(vc_after.min())
n_splits = max(2, min(3, min_per_class))  # 2 o 3, in base al minimo per classe
print(f"\nUserò StratifiedKFold con n_splits={n_splits}")

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
f1s = []

for fold, (tr, te) in enumerate(skf.split(X, y), 1):
    pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(with_mean=False),
        RandomForestClassifier(
            class_weight="balanced_subsample",
            n_estimators=300,
            min_samples_leaf=2,
            random_state=fold
        )
    )
    pipe.fit(X.iloc[tr], y.iloc[tr])
    y_pred = pipe.predict(X.iloc[te])
    f1 = f1_score(y.iloc[te], y_pred, average="macro", zero_division=0)
    f1s.append(f1)
    print(f"\n=== Fold {fold} macro-F1: {f1:.3f} ===")
    print(classification_report(y.iloc[te], y_pred, zero_division=0))

print(f"\n>>> Macro-F1 medio su {n_splits} fold: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

# --- Allena su 100% e salva ---
final_pipe = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(with_mean=False),
    RandomForestClassifier(
        class_weight="balanced_subsample",
        n_estimators=300,
        min_samples_leaf=2,
        random_state=0
    )
)
final_pipe.fit(X, y)
OUT.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(final_pipe, OUT)
print("✅ Salvato modello RF in:", OUT)

