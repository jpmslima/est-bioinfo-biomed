#!/usr/bin/env python3
"""
gerar_figuras_variantes.py — Geração de Gráficos Científicos com Formatação Padrão.

Padronização editorial:
- Título principal no topo em negrito (fig.text)
- Subtítulo contextual com métricas amostrais e metodologia (fig.text)
- Identificação padronizada de painéis: (a), (b), (c)...
- Legenda inferior justificada em bloco com render_justified_caption + Fonte da Autora (2026)
- Resolução de publicação: 300 DPI (PNG e PDF)

Projeto: Estágio de Bioinformática 2026.2 (Genes Compartilhados entre PsA e AS)
Entrada: data/silver/variantes_missense_dbsnp.tsv
Saída: figuras/ (Figuras 1 a 4)
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    from src.visualization import render_justified_caption
    from src.config import TEXTO_FONTE_PADRAO
except ImportError:
    TEXTO_FONTE_PADRAO = "Fonte: Elaborado pela autora (2026)."
    def render_justified_caption(fig, text, x=0.04, y=0.08, width=0.92, fontsize=10.0, **kwargs):
        fig.text(x, y, text + "\n\n" + TEXTO_FONTE_PADRAO, fontsize=fontsize, wrap=True, va="top", ha="left")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("gerar_figuras")

# Estilo tipográfico rigoroso
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "legend.fontsize": 9.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--"
})

# Paleta editorial acessível (Okabe-Ito inspired)
COR_PATOGENICA = "#D55E00"   # Vermelho queimado / Laranja forte
COR_ALTO_CADD = "#E69F00"    # Ocre / Âmbar
COR_MODERADO = "#56B4E9"     # Azul céu
COR_PRIMARIA = "#0072B2"     # Azul profundo
COR_VERDE = "#009E73"        # Verde azulado
COR_CINZA = "#7F7F7F"


def carregar_dados(caminho_tsv: str) -> pd.DataFrame:
    if not os.path.exists(caminho_tsv):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_tsv}")
    df = pd.read_csv(caminho_tsv, sep="\t", low_memory=False)
    logger.info(f"Carregadas {len(df)} variantes de {caminho_tsv}")
    
    df["CADD_Phred_Num"] = pd.to_numeric(df["CADD_Phred"], errors="coerce")
    df["gnomAD_AF_Num"] = pd.to_numeric(df["gnomAD_AF"], errors="coerce")
    df["Posicao_AA_Num"] = pd.to_numeric(df["Posicao_AA"], errors="coerce")
    df["REVEL_Score_Num"] = pd.to_numeric(df["REVEL_Score"], errors="coerce")
    return df


def plot_figura1_panorama(df: pd.DataFrame, pasta_saida: str):
    """Figura 1 — Panorama de Patogenicidade e Frequência Alélica."""
    logger.info("Gerando Figura 1 padronizada...")
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.8), dpi=300)

    # Posicionamento dos subplots para acomodar título e legenda justificada
    axes[0].set_position([0.07, 0.28, 0.40, 0.53])
    axes[1].set_position([0.55, 0.28, 0.41, 0.53])

    # Painel A: Distribuição CADD Phred
    ax1 = axes[0]
    cadd_vals = df["CADD_Phred_Num"].dropna()
    sns.histplot(
        cadd_vals,
        bins=40,
        kde=True,
        color=COR_PRIMARIA,
        edgecolor="white",
        ax=ax1
    )
    ax1.axvline(20, color=COR_PATOGENICA, linestyle="--", linewidth=1.8, label="Corte CADD ≥ 20 (Top 1% Deletério)")
    ax1.axvline(30, color="#7570b3", linestyle=":", linewidth=1.8, label="Corte CADD ≥ 30 (Top 0,1% Deletério)")
    ax1.set_title(r"$\bf{(a)}$ Distribuição de Escore CADD Phred nas Variantes Missense", loc="left", fontsize=11, pad=12)
    ax1.set_xlabel("CADD Phred Score", fontsize=10.5)
    ax1.set_ylabel("Frequência Absoluta de Variantes", fontsize=10.5)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.5)

    # Painel B: Scatter CADD vs gnomAD
    ax2 = axes[1]
    df_scatter = df.dropna(subset=["CADD_Phred_Num", "gnomAD_AF_Num"]).copy()
    df_scatter = df_scatter[df_scatter["gnomAD_AF_Num"] > 0].copy()

    def categorizar_variante(row):
        cadd = row["CADD_Phred_Num"]
        clin = str(row.get("ClinVar_MyVariant", "")).lower() + str(row.get("ClinVar_UniProt", "")).lower()
        if "pathogenic" in clin:
            return "ClinVar Patogênica"
        elif cadd >= 20:
            return "Alto Impacto (CADD ≥ 20)"
        return "Impacto Moderado / Neutro"

    df_scatter["Categoria"] = df_scatter.apply(categorizar_variante, axis=1)

    paleta_cat = {
        "ClinVar Patogênica": COR_PATOGENICA,
        "Alto Impacto (CADD ≥ 20)": COR_ALTO_CADD,
        "Impacto Moderado / Neutro": COR_MODERADO
    }

    sns.scatterplot(
        data=df_scatter,
        x="gnomAD_AF_Num",
        y="CADD_Phred_Num",
        hue="Categoria",
        palette=paleta_cat,
        alpha=0.55,
        s=28,
        ax=ax2
    )
    ax2.set_xscale("log")
    ax2.axhline(20, color=COR_PATOGENICA, linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.axvline(0.01, color=COR_CINZA, linestyle=":", linewidth=1.3, label="Limiar Variante Rara (< 1%)")
    ax2.set_title(r"$\bf{(b)}$ CADD Score vs Frequência Alélica Populacional Global (gnomAD)", loc="left", fontsize=11, pad=12)
    ax2.set_xlabel("Frequência Alélica Global no gnomAD (Escala Logarítmica)", fontsize=10.5)
    ax2.set_ylabel("CADD Phred Score", fontsize=10.5)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.5)

    # Título e Subtítulo Superiores com Formatação Padrão
    fig.text(0.5, 0.960, "Panorama de Patogenicidade e Frequência Alélica dos Polimorfismos Missense em Genes de PsA e AS",
             ha="center", va="top", fontsize=13.5, fontweight="bold", color="#111111")
    fig.text(0.5, 0.925, "Análise integrativa multi-fonte de 313.631 variantes missense identificadas em 428 genes compartilhados (UniProt, MyVariant.info e Ensembl)",
             ha="center", va="top", fontsize=10.5, fontweight="normal", color="#444444")

    # Legenda Justificada Inferior com Formatação Padrão
    full_caption = (
        r"$\bf{Figura\ 1.}$ Panorama global de patogenicidade computacional e epidemiologia genômica das variantes missense. "
        r"$\bf{(a)}$ Distribuição de frequência dos escores CADD Phred (Combined Annotation Dependent Depletion) indicando a densidade de variantes deletérias (linhas tracejadas para os limiares de topo 1% [CADD ≥ 20] e 0,1% [CADD ≥ 30]). "
        r"$\bf{(b)}$ Dispersão entre o escore de patogenicidade CADD e a frequência alélica no banco populacional gnomAD (escala logarítmica). Destaque para variantes raras com alta probabilidade de disrupção estrutural ou funcional."
    )
    render_justified_caption(fig, full_caption, x=0.05, y=0.155, width=0.90, fontsize=10.0,
                             incluir_fonte=True, fonte_texto=TEXTO_FONTE_PADRAO)

    out_png = os.path.join(pasta_saida, "figura1_panorama_patogenicidade.png")
    out_pdf = os.path.join(pasta_saida, "figura1_panorama_patogenicidade.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    logger.info(f"Figura 1 salva em: {out_png}")


def plot_figura2_top_genes(df: pd.DataFrame, pasta_saida: str, top_n: int = 20):
    """Figura 2 — Ranking dos Top Genes com Maior Carga de Mutações Deletérias."""
    logger.info(f"Gerando Figura 2 padronizadas...")
    fig, ax = plt.subplots(figsize=(12.0, 8.2), dpi=300)
    ax.set_position([0.16, 0.22, 0.78, 0.63])

    genes_total = df.groupby("Gene").size().rename("Total")
    genes_altos = df[df["CADD_Phred_Num"] >= 20].groupby("Gene").size().rename("Alto_Impacto_CADD20")
    df_genes = pd.concat([genes_total, genes_altos], axis=1).fillna(0)
    df_genes["Moderado_Baixo"] = df_genes["Total"] - df_genes["Alto_Impacto_CADD20"]
    df_top = df_genes.sort_values(by="Alto_Impacto_CADD20", ascending=False).head(top_n)

    y_pos = np.arange(len(df_top))
    genes_labels = df_top.index[::-1]
    altos_vals = df_top["Alto_Impacto_CADD20"].values[::-1]
    baixos_vals = df_top["Moderado_Baixo"].values[::-1]

    ax.barh(y_pos, altos_vals, color=COR_PATOGENICA, label="Alto Impacto / Deletéria (CADD ≥ 20)", edgecolor="none", height=0.68)
    ax.barh(y_pos, baixos_vals, left=altos_vals, color=COR_MODERADO, label="Impacto Moderado / Neutro (CADD < 20)", edgecolor="none", height=0.68)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(genes_labels, fontweight="bold", fontsize=9.5)
    ax.set_xlabel("Número Total de Variantes Missense Mapeadas no dbSNP", fontsize=10.5)
    ax.set_title(r"$\bf{Ranking\ de\ Suscetibilidade:}$ Genes com Maior Frequência Absoluta de Variantes Deletérias", loc="left", fontsize=11.0, pad=10)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95, fontsize=9.0)

    # Anotações nas barras
    for i, (alto, total) in enumerate(zip(altos_vals, (altos_vals + baixos_vals))):
        if alto > 0:
            pct = (alto / total) * 100
            ax.text(total + (max(altos_vals + baixos_vals) * 0.01), i, f"{int(alto):,} ({pct:.0f}%)", va="center", fontsize=8.5, color="#222222")

    # Título e Subtítulo Superiores
    fig.text(0.5, 0.962, f"Top {top_n} Genes com Maior Densidade de Variantes Missense Deletérias em PsA e AS",
             ha="center", va="top", fontsize=13.5, fontweight="bold", color="#111111")
    fig.text(0.5, 0.928, f"Proporção de variantes com escore CADD ≥ 20 (top 1% deletério) em relação ao total de variantes missense catalogadas por gene",
             ha="center", va="top", fontsize=10.5, fontweight="normal", color="#444444")

    # Legenda Justificada Inferior
    full_caption = (
        r"$\bf{Figura\ 2.}$ Estratificação dos genes compartilhados por carga de polimorfismos missense de alta patogenicidade. "
        r"As barras representam o volume cumulativo de variantes, destacando em laranja as mutações deletérias com escore CADD ≥ 20. "
        r"Os valores numéricos à direita indicam a quantidade absoluta de variantes deletérias e o percentual correspondente dentro do gene."
    )
    render_justified_caption(fig, full_caption, x=0.06, y=0.125, width=0.88, fontsize=10.0,
                             incluir_fonte=True, fonte_texto=TEXTO_FONTE_PADRAO)

    out_png = os.path.join(pasta_saida, "figura2_top_genes_deletorios.png")
    out_pdf = os.path.join(pasta_saida, "figura2_top_genes_deletorios.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    logger.info(f"Figura 2 salva em: {out_png}")


def plot_figura3_lollipop(df: pd.DataFrame, pasta_saida: str, genes_alvo: list = None):
    """Figura 3 — Mapeamento Linear de Mutações (Lollipop Plots) em Proteínas Chave."""
    if genes_alvo is None:
        genes_alvo = ["IL17A", "IL23R", "JAK3", "ERAP1"]
        
    logger.info(f"Gerando Figura 3 padronizadas para {genes_alvo}...")
    genes_presentes = [g for g in genes_alvo if g in df["Gene"].values]

    n_genes = len(genes_presentes)
    fig, axes = plt.subplots(n_genes, 1, figsize=(14.0, 3.5 * n_genes), dpi=300)
    if n_genes == 1:
        axes = [axes]

    # Ajuste dos subplots
    plt.subplots_adjust(top=0.88, bottom=0.16, hspace=0.55, left=0.08, right=0.94)

    for i, (ax, gene) in enumerate(zip(axes, genes_presentes)):
        df_g = df[df["Gene"] == gene].dropna(subset=["Posicao_AA_Num"]).copy()
        max_pos = df_g["Posicao_AA_Num"].max() if not df_g.empty else 100
        
        # Barra da proteína (backbone)
        ax.plot([1, max_pos * 1.04], [0, 0], color="#2B3A42", linewidth=5.5, solid_capstyle="round", zorder=1)
        
        df_g["CADD_Plot"] = df_g["CADD_Phred_Num"].fillna(5.0)
        def categorizar_lollipop(r):
            clin = str(r.get("ClinVar_MyVariant", "")).lower() + str(r.get("ClinVar_UniProt", "")).lower()
            if "pathogenic" in clin:
                return "Patogenica"
            elif r["CADD_Plot"] >= 20:
                return "Alto_Impacto"
            return "Moderado_Baixo"
            
        df_g["Cat_Lollipop"] = df_g.apply(categorizar_lollipop, axis=1)
        
        # Hastes
        ax.vlines(x=df_g["Posicao_AA_Num"], ymin=0, ymax=df_g["CADD_Plot"], color="#B0BEC5", linewidth=0.8, alpha=0.6, zorder=2)
        
        # Pontos
        for cat, cor, tam in [
            ("Moderado_Baixo", COR_MODERADO, 24),
            ("Alto_Impacto", COR_ALTO_CADD, 38),
            ("Patogenica", COR_PATOGENICA, 60)
        ]:
            sub = df_g[df_g["Cat_Lollipop"] == cat]
            if not sub.empty:
                ax.scatter(sub["Posicao_AA_Num"], sub["CADD_Plot"], color=cor, s=tam, edgecolor="black", linewidth=0.4, zorder=3, alpha=0.9)
                
        letter = chr(ord('a') + i)
        ax.set_title(rf"$\bf{{({letter})}}$ Proteína $\bf{{{gene}}}$ — Posição dos Resíduos Mutados vs Escore CADD", loc="left", fontsize=10.5, pad=6)
        ax.set_xlabel("Posição do Aminoácido na Sequência Primária (Resíduo)", fontsize=9.0)
        ax.set_ylabel("CADD Phred", fontsize=9.0)
        ax.set_ylim(-1, 38)
        ax.axhline(20, color=COR_PATOGENICA, linestyle="--", linewidth=1.0, alpha=0.5)

    leg_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COR_PATOGENICA, markersize=8.5, label='ClinVar Patogênica', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COR_ALTO_CADD, markersize=7.5, label='Alto Impacto (CADD ≥ 20)', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COR_MODERADO, markersize=6.5, label='Impacto Moderado/Baixo', markeredgecolor='black'),
    ]
    axes[0].legend(handles=leg_elements, loc="upper right", frameon=True, facecolor="white", framealpha=0.95, fontsize=8.0)

    # Título e Subtítulo Superiores
    fig.text(0.5, 0.965, "Mapeamento Estrutural das Variantes Missense em Proteínas Centrais do Eixo Imunológico",
             ha="center", va="top", fontsize=13.5, fontweight="bold", color="#111111")
    fig.text(0.5, 0.938, "Distribuição topológica das substituições de aminoácidos ao longo da cadeia polipeptídica (IL-17A, IL-23R, JAK3 e ERAP1)",
             ha="center", va="top", fontsize=10.5, fontweight="normal", color="#444444")

    # Legenda Justificada Inferior
    full_caption = (
        r"$\bf{Figura\ 3.}$ Distribuição posicional das variantes missense ao longo das sequências polipeptídicas de quatro genes imunológicos chave. "
        r"$\bf{(a-d)}$ Diagramas estruturais (needle/lollipop plots) indicando a localização linear de cada mutação e sua respectiva patogenicidade computacional (eixo Y). "
        r"Pontos em vermelho indicam variantes com significância clínica patogênica catalogada no ClinVar, e pontos ocre denotam escore CADD ≥ 20."
    )
    render_justified_caption(fig, full_caption, x=0.06, y=0.085, width=0.88, fontsize=10.0,
                             incluir_fonte=True, fonte_texto=TEXTO_FONTE_PADRAO)

    out_png = os.path.join(pasta_saida, "figura3_lollipop_genes_chave.png")
    out_pdf = os.path.join(pasta_saida, "figura3_lollipop_genes_chave.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    logger.info(f"Figura 3 salva em: {out_png}")


def plot_figura4_consenso(df: pd.DataFrame, pasta_saida: str):
    """Figura 4 — Consenso e Validação Cruzada Multi-Fonte."""
    logger.info("Gerando Figura 4 padronizadas...")
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=300)
    ax.set_position([0.22, 0.26, 0.72, 0.58])

    consenso_counts = df["Consenso_Fontes"].value_counts()
    y_pos = np.arange(len(consenso_counts))
    cores = sns.color_palette("Blues_r", len(consenso_counts))

    ax.barh(y_pos, consenso_counts.values[::-1], color=cores[::-1], edgecolor="none", height=0.62)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(consenso_counts.index[::-1], fontweight="bold", fontsize=9.5)
    ax.set_xlabel("Número de Variantes Missense Integradas", fontsize=10.5)
    ax.set_title(r"$\bf{Consenso\ Metodológico:}$ Validação Cruzada entre Bancos de Dados", loc="left", fontsize=11.0, pad=10)

    total = len(df)
    for i, val in enumerate(consenso_counts.values[::-1]):
        pct = (val / total) * 100
        ax.text(val + (max(consenso_counts.values) * 0.01), i, f"{val:,} ({pct:.1f}%)", va="center", fontsize=8.8, color="#222222")

    # Título e Subtítulo Superiores
    fig.text(0.5, 0.960, "Validação Cruzada e Nível de Concordância entre Fontes de Bioinformática",
             ha="center", va="top", fontsize=13.5, fontweight="bold", color="#111111")
    fig.text(0.5, 0.926, "Concordância entre as plataformas UniProt/EBI Proteins API, MyVariant.info e Ensembl REST para 313.631 variantes",
             ha="center", va="top", fontsize=10.5, fontweight="normal", color="#444444")

    # Legenda Justificada Inferior
    full_caption = (
        r"$\bf{Figura\ 4.}$ Avaliação do grau de sobreposição e cobertura entre as três plataformas de anotação de variantes genômicas. "
        r"A integração multi-fonte assegura simultaneamente o resgate de anotações funcionais curadas de proteínas (UniProt), coordenadas genômicas de referência GRCh38 (Ensembl) "
        r"e escores agregados de patogenicidade e epidemiologia populacional (MyVariant.info)."
    )
    render_justified_caption(fig, full_caption, x=0.06, y=0.135, width=0.88, fontsize=10.0,
                             incluir_fonte=True, fonte_texto=TEXTO_FONTE_PADRAO)

    out_png = os.path.join(pasta_saida, "figura4_consenso_fontes.png")
    out_pdf = os.path.join(pasta_saida, "figura4_consenso_fontes.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    logger.info(f"Figura 4 salva em: {out_png}")


def main():
    dir_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description="Gerador de Figuras Científicas com Formatação Padrão")
    parser.add_argument(
        "--input",
        "-i",
        default=os.path.join(dir_base, "data", "silver", "variantes_missense_dbsnp.tsv"),
        help="Arquivo TSV de variantes completas"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=os.path.join(dir_base, "figuras"),
        help="Diretório de saída para as figuras"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = carregar_dados(args.input)

    plot_figura1_panorama(df, args.output_dir)
    plot_figura2_top_genes(df, args.output_dir)
    plot_figura3_lollipop(df, args.output_dir)
    plot_figura4_consenso(df, args.output_dir)

    logger.info("==================================================")
    logger.info(f"Todas as 4 figuras foram geradas em: {args.output_dir}")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
