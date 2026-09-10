"""Analyse des échanges commerciaux du Luxembourg (1995-2024) : stationnarité,
VAR et causalité de Granger, cointégration (Engle-Granger) et modèle
correcteur d'erreur (VECM) avec prévision.

Récupère les séries d'exportations/importations directement depuis l'API
Eurostat (`namq_10_exi`) : aucune donnée locale n'est nécessaire.
"""

from pathlib import Path

import eurostat
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import gridspec
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import acf, grangercausalitytests, pacf
from statsmodels.tsa.vector_ar.vecm import VECM

ROOT_DIR = Path(__file__).resolve().parent.parent
FIGURE_DIR = ROOT_DIR / "figure"
TABLE_DIR = ROOT_DIR / "table"

CRISE_FIN_START = pd.Timestamp("2008-07-01")
CRISE_FIN_END = pd.Timestamp("2010-06-30")
COVID_START = pd.Timestamp("2020-01-01")
COVID_END = pd.Timestamp("2021-09-30")


### Partie 1 : récupération et description des séries ###

def recuperer_serie(na_item, nom_colonne):
    """Récupère une série (export=P6, import=P7) du Luxembourg depuis
    Eurostat, en log, restreinte aux 120 premiers trimestres (1995-2024)."""
    df = eurostat.get_data_df("namq_10_exi")
    df = df[
        (df["unit"] == "CLV10_MEUR")
        & (df["na_item"] == na_item)
        & (df["s_adj"] == "SCA")
        & (df[r"geo\TIME_PERIOD"] == "LU")
    ]
    df = df.dropna(axis=1)
    df = df.iloc[:, 5:]
    df = df.T
    df = df.rename(columns={df.columns[0]: nom_colonne})
    df = np.log(df)
    return df.head(120)


def tableau_fac_fap(serie, nlags=32):
    """Tableau FAC/FAP/Q-stat/Prob pour une série (niveau ou différenciée)."""
    fac = acf(
        serie, adjusted=0, nlags=nlags, qstat=True, fft=True,
        alpha=None, bartlett_confint=True, missing="none",
    )
    table = pd.DataFrame(fac).T.rename(columns={0: "FAC", 1: "Q-stat", 2: "Prob"})
    fap = pacf(serie, nlags=nlags, method="ywmle", alpha=None)
    fap_table = pd.DataFrame(fap).iloc[1:, :].reset_index(drop=True).rename(columns={0: "FAP"})
    table.insert(1, "FAP", fap_table)
    table.iloc[:, 0] = table.iloc[:, 0].shift(-1)
    table = table.drop(table.index[-1]).reset_index(drop=True)

    table[["FAC", "FAP", "Prob"]] = table[["FAC", "FAP", "Prob"]].round(3)
    table[["Q-stat"]] = table[["Q-stat"]].round(2)
    table.insert(0, "Lag", range(1, len(table) + 1))
    return table


def graphique_fac_fap(serie, table, titre_serie, filename, lags=32):
    """Figure GridSpec : FAC + FAP + tableau, pour une série."""
    tg = table.set_index("Lag").T.reset_index().rename(columns={"index": "Lag"})

    fig = plt.figure(figsize=(25, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[20, 3])

    ax0 = fig.add_subplot(gs[0, 0])
    sm.graphics.tsa.plot_acf(serie, lags=lags, zero=False, ax=ax0)
    ax0.set_title(f"Autocorrélation (FAC) de {titre_serie}", fontsize=22, fontweight="bold")

    ax1 = fig.add_subplot(gs[0, 1])
    sm.graphics.tsa.plot_pacf(serie, lags=lags, zero=False, ax=ax1)
    ax1.set_title(f"Autocorrélation Partielle (FAP) de {titre_serie}", fontsize=22, fontweight="bold")

    ax2 = fig.add_subplot(gs[1, :])
    ax2.axis("off")
    cell_table = ax2.table(
        cellText=tg.values, colLabels=tg.columns, cellLoc="center", loc="upper center"
    )
    cell_table.scale(1.05, 1.5)
    cell_table.auto_set_font_size(False)
    cell_table.set_fontsize(12.5)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def graphique_series_niveau(xm):
    """Figure 'hero' : séries d'exportations/importations en log, 1995-2024."""
    plt.figure(figsize=(22, 12))

    sns.lineplot(data=xm, x=xm.index, y="lnX", color="blue", label="Exportations")
    sns.scatterplot(data=xm, x=xm.index, y="lnX", color="blue")
    sns.lineplot(data=xm, x=xm.index, y="lnM", color="red", label="Importations")
    sns.scatterplot(data=xm, x=xm.index, y="lnM", color="red")

    plt.ylim(top=xm[["lnX", "lnM"]].max().max() * 1.025)

    plt.axvspan(CRISE_FIN_START, CRISE_FIN_END, color="grey", alpha=0.3)
    plt.axvspan(COVID_START, COVID_END, color="grey", alpha=0.3)

    plt.text(
        x=CRISE_FIN_START + (CRISE_FIN_END - CRISE_FIN_START) / 2,
        y=xm[["lnX", "lnM"]].max().max() * 0.96, s="Crise financière",
        fontsize=16, ha="center", va="top", color="black",
    )
    plt.text(
        x=COVID_START + (COVID_END - COVID_START) / 2,
        y=xm[["lnX", "lnM"]].max().max() * 1.01, s="COVID-19",
        fontsize=16, ha="center", va="top", color="black",
    )

    plt.xlabel(None)
    plt.ylabel("Valeur (volumes chaînés 2010, M€) — échelle log", fontsize=18)
    plt.title(
        "Série des exportations et des importations – Données trimestrielles, "
        "désaisonnalisées et corrigées des effets de calendrier",
        fontsize=20, loc="left",
    )
    plt.suptitle("Flux commerciaux du Luxembourg (1995-2024)", fontsize=35, fontweight="bold")
    plt.legend(title="Série", title_fontsize=16, fontsize=14, loc="lower right", bbox_to_anchor=(1, 0))
    plt.grid(True)
    plt.xticks(pd.date_range(start="1995", periods=31, freq="YE"), fontsize=12, rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.yticks(fontsize=12)

    plt.savefig(FIGURE_DIR / "01-flux-commerciaux-1995-2024.png", bbox_inches="tight")
    plt.close()


### Partie 2 : VAR et causalité de Granger ###

def analyse_granger(xm):
    """VAR sur les séries différenciées + tests de causalité de Granger."""
    xmd = xm.diff().fillna(0).rename(columns={"lnX": "ΔlnX", "lnM": "ΔlnM"})
    var_xmd = VAR(xmd)

    # Comparaison de plusieurs ordres de retard (le README documente un
    # retard optimal de 2 selon le critère BIC) : les 4 summaries sont
    # sauvegardées pour rendre ce choix vérifiable.
    for lag in range(1, 5):
        result = var_xmd.fit(lag)
        with open(TABLE_DIR / f"var-lag-selection-{lag}.txt", "w") as f:
            f.write(str(result.summary()))

    var_xmd_2 = var_xmd.fit(2)

    tests = {
        "m-to-x": var_xmd_2.test_causality("ΔlnX", ["ΔlnM"], kind="wald", signif=0.05),
        "x-to-x": var_xmd_2.test_causality("ΔlnX", ["ΔlnX"], kind="wald", signif=0.01),
        "x-to-m": var_xmd_2.test_causality("ΔlnM", ["ΔlnX"], kind="wald", signif=0.1),
        "m-to-m": var_xmd_2.test_causality("ΔlnM", ["ΔlnM"], kind="wald", signif=0.05),
    }
    for name, test in tests.items():
        with open(TABLE_DIR / f"test-granger-{name}.txt", "w") as f:
            f.write(test.summary().as_text())

    # Test de causalité de Granger "brut" (statsmodels), pour comparaison.
    grangercausalitytests(xmd, 2, addconst=True)


### Partie 3 : cointégration (Engle-Granger) ###

def regression_cointegration(y, x, label):
    """MCO de `y` sur `x` (+ constante) et sauvegarde du summary."""
    model = sm.OLS(y, sm.add_constant(x)).fit()
    with open(TABLE_DIR / f"model-cointegration-{label}.txt", "w") as f:
        f.write(model.summary().as_text())
    return model


def graphique_residus(residus, titre, filename):
    """Diagramme en barres des résidus d'une régression de cointégration."""
    residus_flat = np.array(residus).flatten()

    plt.figure(figsize=(20, 8))
    plt.bar(range(len(residus_flat)), residus_flat)
    plt.axhline(y=0, color="r", linestyle="-")
    plt.xlabel("Index", fontsize=16)
    plt.ylabel("Résidus", fontsize=16)
    plt.title(titre, fontsize=16, fontweight="bold")
    plt.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close()


def retard(df, column_name, n):
    """Ajoute à `df` les `n` premières colonnes décalées de `column_name`."""
    for i in range(1, n + 1):
        df[f"{column_name}(-{i})"] = df[column_name].shift(i, fill_value=0)
    return df


def analyse_cointegration(x, m):
    """Régressions croisées X~M et M~X, résidus, et tableau récapitulatif."""
    model_xm = regression_cointegration(x["X"], m["M"], "xm")
    model_mx = regression_cointegration(m["M"], x["X"], "mx")

    residu_xm = pd.DataFrame({"Valeurs_ajustées": model_xm.fittedvalues, "u": model_xm.resid})
    residu_mx = pd.DataFrame({"Valeurs_ajustées": model_mx.fittedvalues, "u": model_mx.resid})

    graphique_residus(
        residu_xm["u"], "Graphique des résidus exportation sur importation de la Luxembourg",
        "03-residus-cointegration-exp-imp.png",
    )
    graphique_residus(
        residu_mx["u"], "Graphique des résidus importation sur exportation de la Luxembourg",
        "residus-cointegration-imp-exp.png",
    )

    tableau_res = residu_xm[["u"]].rename(columns={"u": "uXM"}).join(
        residu_mx[["u"]].rename(columns={"u": "uMX"})
    )
    tableau_res.to_csv(TABLE_DIR / "residus-xm-mx.csv", index=True)

    fac_fap_residu_xm = tableau_fac_fap(residu_xm["u"])
    graphique_fac_fap(
        residu_xm["u"], fac_fap_residu_xm,
        "des résidus exportation sur importation de la Luxembourg",
        "acf-pacf-residus-xm.png",
    )
    fac_fap_residu_mx = tableau_fac_fap(residu_mx["u"])
    graphique_fac_fap(
        residu_mx["u"], fac_fap_residu_mx,
        "des résidus importation sur exportation de la Luxembourg",
        "acf-pacf-residus-mx.png",
    )

    return residu_xm, residu_mx


def selection_retards_vecm(residu_xm, residu_mx):
    """Sélection du nombre de retards pour la VECM à partir des résidus de
    cointégration (régressions emboîtées 0 à 4 retards, dans chaque sens)."""
    for label, residu, colonne in (("xm", residu_xm, "ΔuXM"), ("mx", residu_mx, "ΔuMX")):
        diff = residu.drop("Valeurs_ajustées", axis=1).diff().fillna(0).rename(columns={"u": "Δu"})
        diff = retard(diff, "Δu", 4)
        retarde = residu.shift(fill_value=0).rename(columns={"u": "u(-1)"}).drop("Valeurs_ajustées", axis=1)
        combine = retarde.join(diff, how="left").rename(columns={"Δu": colonne})

        y = combine[colonne]
        for n_retards in range(4, -1, -1):
            colonnes_x = ["u(-1)"] + [f"Δu(-{i})" for i in range(1, n_retards + 1)]
            x = sm.add_constant(combine[colonnes_x])
            model = sm.OLS(y, x).fit()
            with open(TABLE_DIR / f"model-lag-selection-{label}-{n_retards}.txt", "w") as f:
                f.write(model.summary().as_text())


### Partie 4 : VECM et prévision ###

def analyse_vecm(x, m):
    """Estime la VECM(2) et sauvegarde son summary."""
    serie = pd.concat([x, m], axis=1)
    serie.columns = ["logX", "logM"]

    vecm = VECM(serie, k_ar_diff=2, coint_rank=1, deterministic="co")
    vecm_res = vecm.fit()

    with open(TABLE_DIR / "vecm.txt", "w") as f:
        f.write(vecm_res.summary().as_text())

    return serie, vecm_res


def previsions_vecm(serie, vecm_res, n_steps=8):
    """Prévision à 8 trimestres, graphique et tableaux (niveau + croissance)."""
    forecast = vecm_res.predict(steps=n_steps)
    forecast_df = pd.DataFrame(forecast, columns=serie.columns)

    quarters = [f"{year}-Q{quarter}" for year in [2025, 2026] for quarter in range(1, 5)]
    forecast_df.index = quarters[:n_steps]

    ### Graphique : séries historiques + prévisions, remises en niveau ###
    serie_exp = np.exp(serie)
    forecast_exp = np.exp(forecast_df)

    serie_exp.index = pd.to_datetime(serie_exp.index)
    forecast_exp.index = pd.to_datetime(forecast_exp.index)
    full_index = serie_exp.index.union(forecast_exp.index)

    plt.figure(figsize=(22, 12))

    plt.plot(serie_exp.index, serie_exp["logX"], label="Exportation historique", color="red", linestyle="-")
    plt.plot(serie_exp.index, serie_exp["logM"], label="Importation historique", color="blue", linestyle="-")
    plt.plot(forecast_exp.index, forecast_exp["logX"], label="Exportation prévision", color="peachpuff", linestyle="--")
    plt.plot(forecast_exp.index, forecast_exp["logM"], label="Importation prévision", color="lightskyblue", linestyle="--")

    plt.scatter(serie_exp.index, serie_exp["logX"], color="red")
    plt.scatter(serie_exp.index, serie_exp["logM"], color="blue")
    plt.scatter(forecast_exp.index, forecast_exp["logX"], color="peachpuff")
    plt.scatter(forecast_exp.index, forecast_exp["logM"], color="lightskyblue")

    y_max = max(serie_exp[["logX", "logM"]].max().max(), forecast_exp[["logX", "logM"]].max().max())
    plt.ylim(top=y_max * 1.05)

    plt.axvspan(CRISE_FIN_START, CRISE_FIN_END, color="grey", alpha=0.3)
    plt.axvspan(COVID_START, COVID_END, color="grey", alpha=0.3)
    plt.text(
        x=CRISE_FIN_START + (CRISE_FIN_END - CRISE_FIN_START) / 2, y=y_max * 0.6,
        s="Crise financière", fontsize=16, ha="center", va="bottom", color="black",
    )
    plt.text(
        x=COVID_START + (COVID_END - COVID_START) / 2, y=y_max * 0.95,
        s="COVID-19", fontsize=16, ha="center", va="bottom", color="black",
    )

    plt.xlabel(None)
    plt.ylabel("Valeur (volumes chaînés 2010, M€)", fontsize=18)
    plt.title(
        "Séries historiques et prévisions – Données trimestrielles, "
        "désaisonnalisées et corrigées des effets de calendrier",
        fontsize=20, loc="left",
    )
    plt.suptitle("Flux commerciaux du Luxembourg (1995–2024)", fontsize=35, fontweight="bold")
    plt.legend(title="Série", title_fontsize=16, fontsize=14, loc="lower right", bbox_to_anchor=(1, 0))
    plt.grid(True)
    plt.xlim(left=pd.Timestamp("1995-01-01"))
    plt.xticks(pd.date_range(start="1995-01-01", end=full_index.max(), freq="YS"), fontsize=12, rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.yticks(fontsize=12)
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / "04-previsions-vecm.png", bbox_inches="tight")
    plt.close()

    ### Tableaux : prévision en niveau + taux de croissance ###
    forecast_exp.columns = ["Exportation", "Importation"]
    forecast_exp = forecast_exp.round(2)
    forecast_exp.index.name = "Annee"
    forecast_exp.to_html(TABLE_DIR / "forecast.html", index=True)

    growth_rates_percent = (forecast_exp[["Exportation", "Importation"]].pct_change() * 100).round(2)
    growth_rates_percent.columns = ["growth_X", "growth_M"]
    growth_rates_percent.index.name = "Annee"
    growth_rates_percent.to_html(TABLE_DIR / "forecast-growth-rate.html", index=True)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    ### Partie 1 : récupération et description des séries ###
    x = recuperer_serie("P6", "X")
    m = recuperer_serie("P7", "M")
    x_diff = x.diff().dropna()
    m_diff = m.diff().dropna()

    x.to_excel(TABLE_DIR / "x-level.xlsx", index=False)
    m.to_excel(TABLE_DIR / "m-level.xlsx", index=False)

    fac_fap_x = tableau_fac_fap(x)
    fac_fap_x.to_excel(TABLE_DIR / "facfap-x-level.xlsx", index=False)
    graphique_fac_fap(x, fac_fap_x, "l'exportation de la Luxembourg", "02-acf-pacf-exportations.png")

    fac_fap_x_diff = tableau_fac_fap(x_diff)
    fac_fap_x_diff.to_excel(TABLE_DIR / "facfap-x-diff.xlsx", index=False)
    graphique_fac_fap(x_diff, fac_fap_x_diff, "l'exportation de la Luxembourg (différenciée)", "acf-pacf-x-diff.png")

    fac_fap_m = tableau_fac_fap(m)
    fac_fap_m.to_excel(TABLE_DIR / "facfap-m-level.xlsx", index=False)
    graphique_fac_fap(m, fac_fap_m, "l'importation de la Luxembourg", "acf-pacf-m-level.png")

    fac_fap_m_diff = tableau_fac_fap(m_diff)
    fac_fap_m_diff.to_excel(TABLE_DIR / "facfap-m-diff.xlsx", index=False)
    graphique_fac_fap(m_diff, fac_fap_m_diff, "l'importation de la Luxembourg (différenciée)", "acf-pacf-m-diff.png")

    xm = pd.concat([x, m], axis=1).rename(columns={"X": "lnX", "M": "lnM"})
    xm.index = pd.to_datetime(pd.date_range(start="1995-01-01", periods=120, freq="QS"))
    graphique_series_niveau(xm)

    ### Partie 2 : VAR et causalité de Granger ###
    analyse_granger(xm)

    ### Partie 3 : cointégration ###
    residu_xm, residu_mx = analyse_cointegration(x, m)
    selection_retards_vecm(residu_xm, residu_mx)

    ### Partie 4 : VECM et prévision ###
    serie, vecm_res = analyse_vecm(x, m)
    previsions_vecm(serie, vecm_res)

    print("Terminé.")


if __name__ == "__main__":
    main()
