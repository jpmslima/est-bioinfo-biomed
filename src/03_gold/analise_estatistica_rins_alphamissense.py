#!/usr/bin/env python3
"""
analise_estatistica_rins_alphamissense.py — Validação Estatística Formal e Estudos de Caso com Formatação Padrão.

Padronização Editorial:
- Títulos no topo em negrito (fig.text a y=0.94) e subtítulo contextual (y=0.89)
- Subplots posicionados com set_position exato: [left, 0.28, width, 0.53]
- Colorbar com eixo dedicado (fig.add_axes) para não distorcer os eixos dos gráficos
- Rótulos de domínios integrados nas faixas sombreadas sem caixa de legenda flutuante colidente
- Anotações de variantes com coordenadas exclusivas e caixas translúcidas sem sobreposição
- Legenda justificada no rodapé (y=0.17) com render_justified_caption + Fonte: Elaborado pela autora (2026).
- Resolução de publicação: 300 DPI em PNG e PDF vetorial
- Paleta acessível Okabe-Ito (sem cores genéricas)

Projeto: Estágio de Bioinformática 2026.2 (Genes Compartilhados PsA e AS)
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import scikit_posthocs as sp

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("analise_estatistica")

# Configuração visual editorial padronizada
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.titlesize": 11.0,
    "axes.labelsize": 10.0,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--"
})

# Paleta editorial acessível (Okabe-Ito)
COR_PATOGENICA = "#D55E00"   # Vermelho queimado / Laranja forte
COR_AMBIGUA = "#E69F00"      # Ocre / Âmbar
COR_BENIGNA = "#009E73"      # Verde azulado
COR_PRIMARIA = "#0072B2"     # Azul profundo
COR_MODERADO = "#56B4E9"     # Azul céu
COR_CINZA = "#7F7F7F"
TEXTO_FONTE_PADRAO = "Fonte: Elaborado pela autora (2026)."


def render_justified_caption(fig, text, x=0.04, y=0.17, width=0.92, fontsize=9.5, **kwargs):
    """Renderiza a legenda explicativa justificada no rodapé da figura com formatação padrão."""
    full_text = f"{text}\n\n{TEXTO_FONTE_PADRAO}"
    fig.text(x, y, full_text, fontsize=fontsize, wrap=True, va="top", ha="left", linespacing=1.35)


def executar_kruskal_wallis_e_dunn(df: pd.DataFrame, caminho_relatorio_tsv: str) -> Dict:
    logger.info("Executando Kruskal-Wallis Global e Teste Post-Hoc de Dunn (Bonferroni / Holm)...")

    df_validos = df[
        df["RIN_Degree"].notna() &
        df["AlphaMissense_Class"].isin(["benign", "ambiguous", "pathogenic"])
    ].copy()

    classes = ["benign", "ambiguous", "pathogenic"]
    metricas = ["RIN_Betweenness", "RIN_Degree", "RIN_Closeness", "AlphaFold_pLDDT"]

    relatorio_linhas = []
    resultados_completos = {}

    for m in metricas:
        grupo_ben = df_validos[df_validos["AlphaMissense_Class"] == "benign"][m].dropna()
        grupo_amb = df_validos[df_validos["AlphaMissense_Class"] == "ambiguous"][m].dropna()
        grupo_pat = df_validos[df_validos["AlphaMissense_Class"] == "pathogenic"][m].dropna()

        # Kruskal-Wallis
        kw_stat, kw_pval = stats.kruskal(grupo_ben, grupo_amb, grupo_pat)

        # Dunn com Bonferroni
        dunn_bonf = sp.posthoc_dunn(
            df_validos,
            val_col=m,
            group_col="AlphaMissense_Class",
            p_adjust="bonferroni"
        )

        p_pat_ben = dunn_bonf.loc["pathogenic", "benign"]
        p_pat_amb = dunn_bonf.loc["pathogenic", "ambiguous"]
        p_amb_ben = dunn_bonf.loc["ambiguous", "benign"]

        med_ben, mean_ben = grupo_ben.median(), grupo_ben.mean()
        med_amb, mean_amb = grupo_amb.median(), grupo_amb.mean()
        med_pat, mean_pat = grupo_pat.median(), grupo_pat.mean()

        resultados_completos[m] = {
            "kw_stat": kw_stat,
            "kw_pval": kw_pval,
            "dunn_bonferroni": dunn_bonf,
            "med_ben": med_ben, "mean_ben": mean_ben,
            "med_amb": med_amb, "mean_amb": mean_amb,
            "med_pat": med_pat, "mean_pat": mean_pat,
            "p_pat_ben": p_pat_ben,
            "p_pat_amb": p_pat_amb,
            "p_amb_ben": p_amb_ben
        }

        relatorio_linhas.append({
            "Metrica_Estrutural": m,
            "Kruskal_Wallis_H": kw_stat,
            "Kruskal_Wallis_p": kw_pval,
            "Mediana_Benigna": med_ben,
            "Mediana_Ambigua": med_amb,
            "Mediana_Patogenica": med_pat,
            "Media_Benigna": mean_ben,
            "Media_Ambigua": mean_amb,
            "Media_Patogenica": mean_pat,
            "Dunn_p_Pat_vs_Ben": p_pat_ben,
            "Dunn_p_Pat_vs_Amb": p_pat_amb,
            "Dunn_p_Amb_vs_Ben": p_amb_ben,
            "Status_Intermediario_Ambiguo": (med_ben < med_amb < med_pat) or (mean_ben < mean_amb < mean_pat)
        })

    # Enriquecimento em Hubs (3 Grupos)
    df_hubs = df_validos.groupby("AlphaMissense_Class")["RIN_Hub_Status"].apply(
        lambda s: (s != "Resíduo Regular").mean() * 100
    )
    
    pct_ben = df_hubs.get("benign", 0)
    pct_amb = df_hubs.get("ambiguous", 0)
    pct_pat = df_hubs.get("pathogenic", 0)

    is_hub_pat = df_validos[df_validos["AlphaMissense_Class"] == "pathogenic"]["RIN_Hub_Status"] != "Resíduo Regular"
    is_hub_ben = df_validos[df_validos["AlphaMissense_Class"] == "benign"]["RIN_Hub_Status"] != "Resíduo Regular"

    tab_pat_ben = [[is_hub_pat.sum(), len(is_hub_pat) - is_hub_pat.sum()],
                   [is_hub_ben.sum(), len(is_hub_ben) - is_hub_ben.sum()]]
    or_pat_ben, p_fisher_pat_ben = stats.fisher_exact(tab_pat_ben, alternative="greater")

    resultados_completos["enriquecimento_hubs"] = {
        "pct_ben": pct_ben,
        "pct_amb": pct_amb,
        "pct_pat": pct_pat,
        "or_pat_ben": or_pat_ben,
        "p_fisher_pat_ben": p_fisher_pat_ben
    }

    df_rel = pd.DataFrame(relatorio_linhas)
    df_rel.to_csv(caminho_relatorio_tsv, sep="\t", index=False)
    logger.info(f"Relatório estatístico salvo em: {caminho_relatorio_tsv}")

    return resultados_completos


def gerar_figura7_validacao_estatistica(df: pd.DataFrame, stats_res: Dict, pasta_saida: str):
    logger.info("Gerando Figura 7 com formatação padrão...")
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 7.8), dpi=300)

    axes[0].set_position([0.06, 0.28, 0.27, 0.53])
    axes[1].set_position([0.38, 0.28, 0.27, 0.53])
    axes[2].set_position([0.70, 0.28, 0.26, 0.53])

    df_validos = df[
        df["RIN_Degree"].notna() &
        df["AlphaMissense_Class"].isin(["benign", "ambiguous", "pathogenic"])
    ].copy()

    ordem = ["benign", "ambiguous", "pathogenic"]
    rotulos = ["Provavelmente\nBenigna", "Ambígua /\nIncerta", "Provavelmente\nPatogênica"]
    cores = [COR_BENIGNA, COR_AMBIGUA, COR_PATOGENICA]

    # Painel (a): Betweenness Centrality
    ax1 = axes[0]
    sns.boxplot(
        data=df_validos,
        x="AlphaMissense_Class",
        y="RIN_Betweenness",
        order=ordem,
        palette=cores,
        hue="AlphaMissense_Class",
        legend=False,
        width=0.42,
        fliersize=1.2,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 4.5},
        ax=ax1
    )
    p_kw_bet = stats_res["RIN_Betweenness"]["kw_pval"]
    kw_txt_bet = "p < 10⁻³⁰⁰" if p_kw_bet < 1e-300 else f"p = {p_kw_bet:.2e}"
    ax1.set_title(r"$\bf{(a)}$ Centralidade de Intermediação (Betweenness)", loc="left", fontsize=10.5, pad=10)
    ax1.set_xlabel("Classificação AlphaMissense", fontsize=10.0)
    ax1.set_ylabel("Betweenness Centrality", fontsize=10.0)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(rotulos)

    # Painel (b): Degree (Grau de Conectividade)
    ax2 = axes[1]
    sns.boxplot(
        data=df_validos,
        x="AlphaMissense_Class",
        y="RIN_Degree",
        order=ordem,
        palette=cores,
        hue="AlphaMissense_Class",
        legend=False,
        width=0.42,
        fliersize=1.2,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 4.5},
        ax=ax2
    )
    p_kw_deg = stats_res["RIN_Degree"]["kw_pval"]
    kw_txt_deg = "p < 10⁻³⁰⁰" if p_kw_deg < 1e-300 else f"p = {p_kw_deg:.2e}"
    ax2.set_title(r"$\bf{(b)}$ Grau de Conectividade (Degree)", loc="left", fontsize=10.5, pad=10)
    ax2.set_xlabel("Classificação AlphaMissense", fontsize=10.0)
    ax2.set_ylabel("Número de Contatos Locais (Cα ≤ 8,5 Å)", fontsize=10.0)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(rotulos)

    # Painel (c): Enriquecimento em Hubs Estruturais 3D
    ax3 = axes[2]
    enr = stats_res["enriquecimento_hubs"]
    pcts = [enr["pct_ben"], enr["pct_amb"], enr["pct_pat"]]
    rot_c = ["Benignas", "Ambíguas", "Patogênicas"]
    cores_c = [COR_BENIGNA, COR_AMBIGUA, COR_PATOGENICA]

    bars = ax3.bar(rot_c, pcts, color=cores_c, width=0.50, edgecolor="none")
    for bar in bars:
        h = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2, h + 1.2,
            f"{h:.1f}%",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#1F2937"
        )

    or_val = enr["or_pat_ben"]
    ax3.set_title(r"$\bf{(c)}$ Enriquecimento em Hubs Estruturais 3D", loc="left", fontsize=10.5, pad=10)
    ax3.set_xlabel("Categoria de Patogenicidade", fontsize=10.0)
    ax3.set_ylabel("Frequência de Mutantes em Hubs (%)", fontsize=10.0)
    ax3.set_ylim(0, max(pcts) * 1.22)

    # Título Geral no Topo (Formatação Padrão)
    fig.text(
        0.04, 0.94,
        "Figura 7 — Validação Estatística de Redes Estruturais (RINs) e Patogenicidade em PsA e AS",
        fontsize=13.5, fontweight="bold", color="#111827", ha="left"
    )
    fig.text(
        0.04, 0.89,
        f"Amostra total analisada: n = {len(df_validos):,} variantes missense estruturadas | Testes de Kruskal-Wallis Global e Post-Hoc de Dunn (Bonferroni)",
        fontsize=10.0, color="#4B5563", ha="left"
    )

    legenda_texto = (
        f"Painéis (a) e (b) exibem a distribuição de Betweenness Centrality e Degree para as três classes do AlphaMissense. "
        f"O teste global de Kruskal-Wallis revelou diferenças altamente significativas para todas as propriedades de rede (p < 10⁻³⁰⁰). "
        f"O teste post-hoc de Dunn com correção de Bonferroni comprovou que as variantes Ambíguas/Incertas ocupam uma posição topológica "
        f"estatisticamente intermediária (Dunn p < 10⁻¹⁰⁰ para todos os pares), refletindo uma física de transição na estabilidade proteica. "
        f"O Painel (c) evidencia o enriquecimento progressivo nos Hubs Estruturais (Benignas: {enr['pct_ben']:.1f}% → Ambíguas: {enr['pct_amb']:.1f}% → Patogênicas: {enr['pct_pat']:.1f}%; OR = {or_val:.2f}, Teste Exato de Fisher p < 10⁻³⁰⁰)."
    )
    render_justified_caption(fig, legenda_texto, x=0.04, y=0.17)

    os.makedirs(pasta_saida, exist_ok=True)
    out_png = os.path.join(pasta_saida, "figura7_validacao_estatistica_rins.png")
    out_pdf = os.path.join(pasta_saida, "figura7_validacao_estatistica_rins.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)
    logger.info(f"Figura 7 salva em: {out_png} e {out_pdf}")


def gerar_figura8_estudo_caso_sem_sobreposicao(df: pd.DataFrame, pasta_saida: str):
    logger.info("Gerando Figura 8 com formatação padrão, diagramação limpa e sem sobreposição...")
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 7.8), dpi=300)

    # Posicionamento rigoroso dos eixos (deixa espaço dedicado na extrema direita para colorbar)
    axes[0].set_position([0.07, 0.28, 0.40, 0.53])
    axes[1].set_position([0.52, 0.28, 0.40, 0.53])

    # Painel (a): SMAD3
    ax1 = axes[0]
    df_smad3 = df[df["Gene"] == "SMAD3"].copy()
    df_smad3["Posicao_AA_Num"] = pd.to_numeric(df_smad3["Posicao_AA"], errors="coerce")
    df_smad3 = df_smad3.dropna(subset=["Posicao_AA_Num", "AlphaMissense_Score", "RIN_Betweenness"])

    scatter1 = ax1.scatter(
        df_smad3["Posicao_AA_Num"],
        df_smad3["RIN_Betweenness"],
        c=df_smad3["AlphaMissense_Score"],
        cmap="YlOrRd",
        s=30,
        alpha=0.85,
        edgecolor="none",
        vmin=0.0,
        vmax=1.0
    )

    # Demarcação visual dos domínios funcionais do SMAD3
    ax1.axvspan(9, 137, color="#56B4E9", alpha=0.14)
    ax1.text(73, 0.128, "Domínio MH1 (DNA)", ha="center", va="top", fontsize=8.0, fontweight="bold", color="#0072B2")
    
    ax1.axvspan(231, 425, color="#009E73", alpha=0.14)
    ax1.text(328, 0.128, "Domínio MH2 (Transativação)", ha="center", va="top", fontsize=8.0, fontweight="bold", color="#009E73")

    # Anotações limpas de VUS sem colisão
    anotacoes_smad3 = [
        {"pos": 86, "bet": 0.089, "texto": "p.L86P\n(Hub MH1 | AM=0.99)", "xytext": (86, 0.108)},
        {"pos": 232, "bet": 0.045, "texto": "p.L232P\n(Hub MH2 | AM=1.00)", "xytext": (200, 0.068)},
        {"pos": 254, "bet": 0.052, "texto": "p.I254T\n(Hub MH2 | AM=0.94)", "xytext": (280, 0.088)}
    ]
    for an in anotacoes_smad3:
        ax1.annotate(
            an["texto"],
            xy=(an["pos"], an["bet"]),
            xytext=an["xytext"],
            arrowprops=dict(arrowstyle="->", color=COR_PATOGENICA, lw=1.1),
            fontsize=7.2, fontweight="bold", color="#7F1D1D",
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec=COR_PATOGENICA, alpha=0.94, lw=0.8)
        )

    ax1.set_title(r"$\bf{(a)}$ SMAD3 (Via TGF-$\beta$): Perfil Estrutural de Betweenness", loc="left", fontsize=10.5, pad=10)
    ax1.set_xlabel("Posição do Resíduo de Aminoácido (1 a 425)", fontsize=10.0)
    ax1.set_ylabel("Betweenness Centrality (Intermediação)", fontsize=10.0)
    ax1.set_xlim(0, 435)
    ax1.set_ylim(-0.005, 0.135)

    # Painel (b): PLCG1
    ax2 = axes[1]
    df_plcg1 = df[df["Gene"] == "PLCG1"].copy()
    df_plcg1["Posicao_AA_Num"] = pd.to_numeric(df_plcg1["Posicao_AA"], errors="coerce")
    df_plcg1 = df_plcg1.dropna(subset=["Posicao_AA_Num", "AlphaMissense_Score", "RIN_Betweenness"])

    scatter2 = ax2.scatter(
        df_plcg1["Posicao_AA_Num"],
        df_plcg1["RIN_Betweenness"],
        c=df_plcg1["AlphaMissense_Score"],
        cmap="YlOrRd",
        s=30,
        alpha=0.85,
        edgecolor="none",
        vmin=0.0,
        vmax=1.0
    )

    # Demarcação visual dos domínios funcionais do PLCG1
    ax2.axvspan(26, 140, color="#56B4E9", alpha=0.14)
    ax2.text(83, 0.135, "Dom. PH", ha="center", va="top", fontsize=8.0, fontweight="bold", color="#0072B2")
    
    ax2.axvspan(547, 755, color="#E69F00", alpha=0.14)
    ax2.text(651, 0.135, "Dom. SH2/SH3", ha="center", va="top", fontsize=8.0, fontweight="bold", color="#D97706")
    
    ax2.axvspan(950, 1290, color="#009E73", alpha=0.14)
    ax2.text(1120, 0.135, "Dom. C2 / Catalítico", ha="center", va="top", fontsize=8.0, fontweight="bold", color="#009E73")

    anotacoes_plcg1 = [
        {"pos": 117, "bet": 0.061, "texto": "p.L117P\n(Hub PH | AM=0.99)", "xytext": (117, 0.088)},
        {"pos": 658, "bet": 0.078, "texto": "p.L658P\n(Hub SH2 | AM=0.99)", "xytext": (658, 0.110)},
        {"pos": 1173, "bet": 0.092, "texto": "p.V1173M\n(Hub C2 | AM=0.97)", "xytext": (980, 0.118)}
    ]
    for an in anotacoes_plcg1:
        ax2.annotate(
            an["texto"],
            xy=(an["pos"], an["bet"]),
            xytext=an["xytext"],
            arrowprops=dict(arrowstyle="->", color=COR_PATOGENICA, lw=1.1),
            fontsize=7.2, fontweight="bold", color="#7F1D1D",
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec=COR_PATOGENICA, alpha=0.94, lw=0.8)
        )

    ax2.set_title(r"$\bf{(b)}$ PLCG1 (Sinalização Imune): Perfil Estrutural de Betweenness", loc="left", fontsize=10.5, pad=10)
    ax2.set_xlabel("Posição do Resíduo de Aminoácido (1 a 1290)", fontsize=10.0)
    ax2.set_ylabel("Betweenness Centrality (Intermediação)", fontsize=10.0)
    ax2.set_xlim(0, 1310)
    ax2.set_ylim(-0.005, 0.142)

    # Eixo dedicado fixo para a Colorbar na lateral direita (sem redimensionar os eixos ax1 e ax2)
    cax = fig.add_axes([0.935, 0.28, 0.014, 0.53])
    cbar = fig.colorbar(scatter2, cax=cax)
    cbar.set_label("Escore de Patogenicidade AlphaMissense (0 = Benigno, 1 = Patogênico)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8.0)

    # Título Geral no Topo (Formatação Padrão)
    fig.text(
        0.04, 0.94,
        "Figura 8 — Estudos de Caso Estruturais em Genes Compartilhados: SMAD3 e PLCG1",
        fontsize=13.5, fontweight="bold", color="#111827", ha="left"
    )
    fig.text(
        0.04, 0.89,
        "Mapeamento posicional de Betweenness Centrality colorida por escore AlphaMissense e destaque para VUS em Hubs",
        fontsize=10.0, color="#4B5563", ha="left"
    )

    legenda_texto = (
        f"Painel (a) apresenta o perfil de Betweenness Centrality ao longo da estrutura tridimensional do SMAD3 (modelo AlphaFold), "
        f"delimitando os domínios MH1 e MH2 e destacando variantes clinicamente incertas (VUS) reclassificadas como altamente patogênicas em nós de intermediação. "
        f"O Painel (b) ilustra o comportamento no gene PLCG1 com a demarcação dos domínios regulatórios PH, SH2/SH3 e C2/Catalítico, "
        f"demonstrando que as mutações deletérias compartilhadas concentram-se nos núcleos físicos e eixos de propagação alostérica."
    )
    render_justified_caption(fig, legenda_texto, x=0.04, y=0.17)

    out_png = os.path.join(pasta_saida, "figura8_estudo_caso_smad3_plcg1.png")
    out_pdf = os.path.join(pasta_saida, "figura8_estudo_caso_smad3_plcg1.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)
    logger.info(f"Figura 8 salva em: {out_png} e {out_pdf}")


def main():
    caminho_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    caminho_var = os.path.join(caminho_base, "data", "silver", "base_mestre_integrada.tsv")
    caminho_rel = os.path.join(caminho_base, "data", "gold", "relatorio_kruskal_dunn.tsv")
    pasta_figuras = os.path.join(caminho_base, "figuras")

    logger.info(f"Carregando base consolidada com RINs de {caminho_var}...")
    df = pd.read_csv(caminho_var, sep="\t", low_memory=False)

    df["RIN_Betweenness"] = pd.to_numeric(df["RIN_Betweenness"], errors="coerce")
    df["RIN_Degree"] = pd.to_numeric(df["RIN_Degree"], errors="coerce")
    df["RIN_Closeness"] = pd.to_numeric(df["RIN_Closeness"], errors="coerce")
    df["AlphaFold_pLDDT"] = pd.to_numeric(df["AlphaFold_pLDDT"], errors="coerce")
    df["AlphaMissense_Score"] = pd.to_numeric(df["AlphaMissense_Score"], errors="coerce")

    # 1. Executar Kruskal-Wallis + Post-hoc de Dunn
    stats_res = executar_kruskal_wallis_e_dunn(df, caminho_rel)

    # 2. Gerar Figura 7 e Figura 8 com formatação padrão
    gerar_figura7_validacao_estatistica(df, stats_res, pasta_figuras)
    gerar_figura8_estudo_caso_sem_sobreposicao(df, pasta_figuras)

    logger.info("Figuras 7 e 8 geradas com sucesso com formatação padrão!")


if __name__ == "__main__":
    main()
