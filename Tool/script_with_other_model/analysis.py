import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carica il dataset
csv_path = "minilm_out.csv"  # Modifica con il percorso del tuo file

df = pd.read_csv(csv_path)

# Filtra solo la CWE specificata (es. "CWE-1")
cwe_target = "CWE-1"  # Modifica se necessario
df_cwe = df[df["candidate_cwe_1"] == cwe_target]

# Disegna la distribuzione
plt.figure(figsize=(10, 6))
sns.histplot(df_cwe["probcwe_1"], kde=True, bins=30)
plt.title(f"Distribuzione di Predict Probability per {cwe_target}")
plt.xlabel("Predict Probability")
plt.ylabel("Frequenza")
plt.grid()
plt.savefig("distribuzione_predict_probability.png")
plt.close()

# Boxplot per la distribuzione delle probabilità
plt.figure(figsize=(8, 5))
sns.boxplot(y=df_cwe["probcwe_1"])
plt.title(f"Boxplot di Predict Probability per {cwe_target}")
plt.ylabel("Predict Probability")
plt.grid()
plt.savefig("boxplot_predict_probability.png")
plt.close()

# Scatter plot delle probabilità per diverse CWE
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["candidate_cwe_1"], y=df["probcwe_1"], alpha=0.5)
plt.title("Scatter Plot delle Probabilità per Diverse CWE")
plt.xlabel("CWE")
plt.ylabel("Predict Probability")
plt.xticks(rotation=90)
plt.grid()
plt.savefig("scatterplot_predict_probability.png")
plt.close()
