import sys, joblib, numpy as np, pandas as pd

MODEL_PATH = "/home/raffaello/models/rf_model.pkl"
CSV_PATH = sys.argv[1]

# Carica modello e CSV
MODEL = joblib.load(MODEL_PATH)
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

# Ricava l'elenco delle feature usate in fit
def get_expected_features(model):
    # se il modello (o uno step) espone feature_names_in_
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "named_steps"):
        for _, step in model.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    raise RuntimeError("Impossibile ricavare le feature dal modello: nessun 'feature_names_in_' trovato.")

expected = get_expected_features(MODEL)

# (opzionale) rimuovi eventuale colonna target se per sbaglio nel CSV
for tgt in ("target", "label", "y"):
    if tgt in df.columns and tgt not in expected:
        df.drop(columns=[tgt], inplace=True)

# Calcola differenze
missing = [c for c in expected if c not in df.columns]
extra   = [c for c in df.columns if c not in expected]

# Aggiungi mancanti come NaN e droppa extra
for c in missing:
    df[c] = np.nan
if extra:
    df.drop(columns=extra, inplace=True)

# Ordina colonne come atteso
df = df[expected]

# Predici
pred = MODEL.predict(df)
print(pred)

# Debug utile
if missing or extra:
    print("INFO -> missing:", missing, "| extra droppate:", extra, file=sys.stderr)

