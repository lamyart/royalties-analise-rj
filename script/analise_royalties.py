import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. SIMULAÇÃO DE DADOS (BASE REALISTA)
# -----------------------------

np.random.seed(42)

municipios = [
    "Campos dos Goytacazes", "Macaé", "Rio das Ostras",
    "Niterói", "Cabo Frio", "Angra dos Reis",
    "Maricá", "São João da Barra"
]

anos = list(range(2015, 2023))

data = []

for m in municipios:
    base_royalty = np.random.uniform(50, 500)
    
    for ano in anos:
        royalties = base_royalty * np.random.uniform(0.8, 1.2)
        
        renda = np.random.uniform(500, 1500) + royalties * 0.5
        educacao = np.random.uniform(0.4, 0.9)
        saneamento = np.random.uniform(0.3, 0.95)
        
        data.append([m, ano, royalties, renda, educacao, saneamento])

df = pd.DataFrame(data, columns=[
    "municipio", "ano", "royalties", "renda", "educacao", "saneamento"
])

# -----------------------------
# 2. ETL (LIMPEZA E PREPARAÇÃO)
# -----------------------------

# Padronização
df["municipio"] = df["municipio"].str.upper()

# Verificar nulos
print("Valores nulos:\n", df.isnull().sum())

# -----------------------------
# 3. ANÁLISE EXPLORATÓRIA
# -----------------------------

print("\nResumo estatístico:\n", df.describe())

# Média por município
media_municipios = df.groupby("municipio").mean(numeric_only=True)

print("\nMédia por município:\n", media_municipios)

# -----------------------------
# 4. CORRELAÇÃO
# -----------------------------

correlacao = df[["royalties", "renda", "educacao", "saneamento"]].corr()

print("\nCorrelação:\n", correlacao)

# -----------------------------
# 5. GRÁFICOS
# -----------------------------

# 5.1 Scatter plot: royalties vs renda
plt.figure()
plt.scatter(df["royalties"], df["renda"])
plt.xlabel("Royalties")
plt.ylabel("Renda")
plt.title("Royalties vs Renda")
plt.show()

# 5.2 Evolução dos royalties (exemplo de 1 município)
cidade_exemplo = "MACAÉ"

df_cidade = df[df["municipio"] == cidade_exemplo]

plt.figure()
plt.plot(df_cidade["ano"], df_cidade["royalties"])
plt.xlabel("Ano")
plt.ylabel("Royalties")
plt.title(f"Evolução dos Royalties - {cidade_exemplo}")
plt.show()

# 5.3 Comparação média de royalties por município
plt.figure()
media_municipios["royalties"].sort_values().plot(kind="bar")
plt.title("Royalties Médios por Município")
plt.ylabel("Royalties")
plt.xticks(rotation=45)
plt.show()

# -----------------------------
# 6. INSIGHT SIMPLES
# -----------------------------

print("\nInsight básico:")

if correlacao.loc["royalties", "renda"] > 0.5:
    print("Existe forte relação positiva entre royalties e renda.")
else:
    print("Não há relação forte entre royalties e renda.")
