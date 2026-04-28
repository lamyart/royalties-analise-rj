import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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
    base_royalty = np.random.uniform(50_000_000, 800_000_000)
    
    for ano in anos:
        royalties = base_royalty * np.random.uniform(0.8, 1.2)
        
        renda = np.random.uniform(800, 2500)
        educacao = np.random.uniform(0.4, 0.9)
        saneamento = np.random.uniform(0.3, 0.95)
        populacao = np.random.randint(50000, 500000)
        
        data.append([m, ano, royalties, renda, educacao, saneamento, populacao])

df = pd.DataFrame(data, columns=[
    "municipio", "ano", "royalties", "renda",
    "educacao", "saneamento", "populacao"
])

# -----------------------------
# 2. ETL (LIMPEZA)
# -----------------------------

df["municipio"] = df["municipio"].str.upper().str.strip()

# -----------------------------
# 3. FEATURE ENGINEERING
# -----------------------------

df["royalties_per_capita"] = df["royalties"] / df["populacao"]

df["indice_social"] = (
    df["renda"] +
    df["educacao"] * 1000 +
    df["saneamento"] * 1000
) / 3

# -----------------------------
# 4. ANÁLISE
# -----------------------------

print("\nResumo estatístico:\n")
print(df.describe())

media_municipios = df.groupby("municipio").mean(numeric_only=True)

print("\nMédia por município:\n")
print(media_municipios)

correlacao = df[[
    "royalties_per_capita",
    "renda",
    "educacao",
    "saneamento",
    "indice_social"
]].corr()

print("\nCorrelação:\n")
print(correlacao)

# -----------------------------
# 5. CRIAR PASTA RESULTADOS
# -----------------------------

os.makedirs("resultados", exist_ok=True)

# -----------------------------
# 6. GRÁFICOS
# -----------------------------

# Scatter principal
plt.figure()
plt.scatter(df["royalties_per_capita"], df["indice_social"])
plt.xlabel("Royalties per capita")
plt.ylabel("Índice Social")
plt.title("Royalties vs Qualidade de Vida")
plt.savefig("resultados/scatter_royalties_vs_social.png")
plt.close()

# Evolução de município
cidade_exemplo = "MACAÉ"
df_cidade = df[df["municipio"] == cidade_exemplo]

plt.figure()
plt.plot(df_cidade["ano"], df_cidade["royalties_per_capita"])
plt.xlabel("Ano")
plt.ylabel("Royalties per capita")
plt.title(f"Evolução - {cidade_exemplo}")
plt.savefig("resultados/evolucao_municipio.png")
plt.close()

# Ranking
plt.figure()
media_municipios["royalties_per_capita"].sort_values().plot(kind="bar")
plt.title("Royalties per capita médio por município")
plt.xticks(rotation=45)
plt.savefig("resultados/ranking_municipios.png")
plt.close()

# -----------------------------
# 7. EXPORTAR DADOS
# -----------------------------

df.to_csv("resultados/base_tratada.csv", index=False)

# -----------------------------
# 8. INSIGHT AUTOMÁTICO
# -----------------------------

print("\nInsight:")

if correlacao.loc["royalties_per_capita", "indice_social"] > 0.5:
    print("Existe relação positiva entre royalties e qualidade de vida.")
else:
    print("Relação fraca: possível ineficiência na alocação dos royalties.")

print("\nArquivos gerados na pasta 'resultados/'")
