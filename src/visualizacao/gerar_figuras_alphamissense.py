#!/usr/bin/env python3
"""
gerar_figuras_alphamissense.py — Figuras Científicas AlphaMissense com Formatação Padrão.

Formatação padrão para o projeto estagio_2026.2:
- Resolução de publicação: 300 DPI (PNG e PDF)
- Título principal no topo em negrito (fig.text)
- Subtítulo contextual com métricas amostrais e metodologia (fig.text)
- Identificação padronizada de painéis: (a), (b)...
- Legenda inferior justificada em bloco + Fonte: Elaborado pela autora (2026).
- Paleta de cores acessível (Okabe-Ito inspired)

Projeto: Estágio de Bioinformática 2026.2 (Genes Compartilhados entre PsA e AS)
Entradas:
  - data/silver/variantes_dbsnp_alphamissense.tsv
  - data/gold/vus_reclassificadas_hubs.tsv
Saída: figuras/ (Figura 5 e Figura 6)
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

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("gerar_figuras_am")

# Configuração tipográfica editorial
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

# Cores acessíveis (Okabe-Ito)
COR_PATOGENICA = "#D55E00"   # Vermelho queimado / Laranja forte
COR_AMBIGUA = "#E69F00"      # Ocre / Âmbar
COR_BENIGNA = "#009E73"      # Verde azulado
COR_PRIMARIA = "#0072B2"     # Azul profundo
COR_MODERADO = "#56B4E9"     # Azul céu
COR_CINZA = "#7F7F7F"
TEXTO_FONTE_PADRAO = "Fonte: Elaborado pela autora (2026)."


def render_justified_caption(fig, text, x=0.04, y=0.09, width=0.92, fontsize=9.5, **kwargs):
    """Renderiza a legenda explicativa justificada no rodapé da figura."""
    full_text = f"{text}\n\n{TEXTO_FONTE_PADRAO}"
    fig.text(x, y, full_text, fontsize=fontsize, wrap=True, va="top", ha="left", linespacing=1.35)


def plot_figura5_alphamissense_concordancia(df: pd.DataFrame, pasta_saida: str):
    """Figura 5 — Panorama de Predição Estrutural AlphaMissense e Concordância com CADD."""
    logger.info("Gerando Figura 5 com formatação padrão (AlphaMissense)...")
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.8), dpi=300)

    # Posicionamento dos eixos
    axes[0].set_position([0.07, 0.28, 0.40, 0.53])
    axes[1].set_position([0.55, 0.28, 0.41, 0.53])

    df_am = df[df["AlphaMissense_Score"].notna()].copy()
    total_am = len(df_am)

    # Painel (a): Distribuição contínua dos escores AlphaMissense
    ax1 = axes[0]
    scores = df_am["AlphaMissense_Score"]
    
    n, bins, patches = ax1.hist(scores, bins=45, edgecolor="white", linewidth=0.5, alpha=0.9)
    for b_left, b_right, patch in zip(bins[:-1], bins[1:], patches):
        center = (b_left + b_right) / 2
        if center >= 0.564:
            patch.set_facecolor(COR_PATOGENICA)
        elif center < 0.340:
            patch.set_facecolor(COR_BENIGNA)
        else:
            patch.set_facecolor(COR_AMBIGUA)

    ax1.axvline(0.340, color="#333333", linestyle="--", linewidth=1.2, label="Limiar Benigno (<0.340)")
    ax1.axvline(0.564, color="#111111", linestyle="--", linewidth=1.2, label="Limiar Patogênico (≥0.564)")

    ax1.set_title("(a) Distribuição de Patogenicidade Estrutural (AlphaMissense)", loc="left", fontweight="bold", pad=10)
    ax1.set_xlabel("Escore de Patogenicidade AlphaMissense (0 a 1)")
    ax1.set_ylabel("Densidade de Variantes Missense")
    ax1.legend(loc="upper center", frameon=True, facecolor="white", edgecolor="none")

    # Painel (b): Concordância AlphaMissense vs CADD Phred
    ax2 = axes[1]
    df_am_cadd = df_am[df_am["CADD_Num"].notna()].copy()
    
    # Boxplot por classe do AlphaMissense
    ordem_classes = ["likely_benign", "ambiguous", "likely_pathogenic"]
    rotulos_classes = ["Provavelmente\nBenigna", "Ambígua /\nIncerta", "Provavelmente\nPatogênica"]
    cores_classes = [COR_BENIGNA, COR_AMBIGUA, COR_PATOGENICA]

    sns.boxplot(
        data=df_am_cadd,
        x="AlphaMissense_Class",
        y="CADD_Num",
        order=ordem_classes,
        palette=cores_classes,
        width=0.45,
        fliersize=1.5,
        ax=ax2
    )

    ax2.axhline(20.0, color=COR_PATOGENICA, linestyle=":", linewidth=1.2, label="Limiar CADD ≥ 20 (Top 1%)")
    ax2.set_title("(b) Validação Cruzada: AlphaMissense vs. CADD Phred Score", loc="left", fontweight="bold", pad=10)
    ax2.set_xlabel("Classificação Categórica do AlphaMissense")
    ax2.set_ylabel("CADD Phred Score")
    ax2.set_xticklabels(rotulos_classes)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")

    # Título Geral no Topo
    fig.text(
        0.04, 0.94,
        "Figura 5 — Integração com Predições do AlphaMissense em Genes de PsA e AS",
        fontsize=13.5, fontweight="bold", color="#111827", ha="left"
    )
    fig.text(
        0.04, 0.89,
        f"Amostra total avaliada: n = {total_am:,} variantes missense mapeadas | Mapeamento estrutural DeepMind / Science (2023)",
        fontsize=10.0, color="#4B5563", ha="left"
    )

    # Legenda Justificada no Rodapé
    legenda_texto = (
        f"Painel (a) exibe a densidade de distribuição dos escores contínuos do AlphaMissense para as variantes "
        f"mapeadas nos genes compartilhados entre PsA e AS, delimitando as faixas canônicas de corte estrutural "
        f"(Benigno < 0,340; Ambíguo 0,340-0,564; Patogênico ≥ 0,564). O Painel (b) apresenta a validação cruzada "
        f"através do boxplot de dispersão dos escores CADD Phred entre as classes do AlphaMissense, evidenciando "
        f"concordância preditiva significativa e forte suporte evolutivo e funcional."
    )
    render_justified_caption(fig, legenda_texto, x=0.04, y=0.17)

    # Salvar
    os.makedirs(pasta_saida, exist_ok=True)
    out_png = os.path.join(pasta_saida, "figura5_alphamissense_concordancia.png")
    out_pdf = os.path.join(pasta_saida, "figura5_alphamissense_concordancia.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)
    logger.info(f"Figura 5 salva em: {out_png} e {out_pdf}")


def plot_figura6_reclassificacao_vus(df_vus: pd.DataFrame, df_total: pd.DataFrame, pasta_saida: str):
    """Figura 6 — Reclassificação de Variantes de Significado Incerto (VUS) pelo AlphaMissense."""
    logger.info("Gerando Figura 6 com formatação padrão (Reclassificação de VUS)...")
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.8), dpi=300)

    axes[0].set_position([0.07, 0.28, 0.40, 0.53])
    axes[1].set_position([0.55, 0.28, 0.41, 0.53])

    # Painel (a): Distribuição das Classes do AlphaMissense em Variantes Incertas/Sem Anotação
    ax1 = axes[0]
    cond_incertas = df_total["Status_Clinico_Consolidado"].isin([
        "VUS / Significado Incerto ou Conflitante",
        "Sem Anotação Clínica / Experimental"
    ])
    df_incertas_am = df_total[cond_incertas & df_total["AlphaMissense_Score"].notna()].copy()
    
    contagem_classes = df_incertas_am["AlphaMissense_Class"].value_counts()
    classes_order = ["likely_pathogenic", "ambiguous", "likely_benign"]
    valores = [contagem_classes.get(c, 0) for c in classes_order]
    rotulos = ["Provavelmente\nPatogênica (Deletéria)", "Ambígua /\nIncerta", "Provavelmente\nBenigna"]
    cores = [COR_PATOGENICA, COR_AMBIGUA, COR_BENIGNA]

    bars = ax1.bar(rotulos, valores, color=cores, width=0.55, edgecolor="none")
    for bar in bars:
        h = bar.get_height()
        pct = (h / sum(valores)) * 100 if sum(valores) > 0 else 0
        ax1.text(
            bar.get_x() + bar.get_width() / 2, h + (max(valores) * 0.015),
            f"{h:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1F2937"
        )

    ax1.set_title("(a) Predição Estrutural de Variantes Clinicamente Incertas / VUS", loc="left", fontweight="bold", pad=10)
    ax1.set_ylabel("Número de Variantes dbSNP")
    ax1.set_ylim(0, max(valores) * 1.18)

    # Painel (b): Top Genes Compartilhados com Mais Candidatas Deletérias Reclassificadas
    ax2 = axes[1]
    top_genes = df_vus["Gene"].value_counts().head(12)
    y_pos = np.arange(len(top_genes))

    bars2 = ax2.barh(y_pos, top_genes.values, color=COR_PRIMARIA, height=0.6, edgecolor="none")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_genes.index)
    ax2.invert_yaxis()  # Maior no topo

    for bar in bars2:
        w = bar.get_width()
        ax2.text(
            w + (max(top_genes.values) * 0.015), bar.get_y() + bar.get_height() / 2,
            f"{int(w):,}",
            ha="left", va="center", fontsize=8.5, fontweight="bold", color="#1F2937"
        )

    ax2.set_title("(b) Top Genes com Variantes Incertas Reclassificadas como Deletérias", loc="left", fontweight="bold", pad=10)
    ax2.set_xlabel("Número de Variantes Candidatas Prioritárias (AlphaMissense + CADD/REVEL)")
    ax2.set_xlim(0, max(top_genes.values) * 1.15)

    # Título Geral no Topo
    total_reclass = len(df_vus)
    fig.text(
        0.04, 0.94,
        "Figura 6 — Reclassificação Funcional de VUS e Proposição de Candidatas Críticas",
        fontsize=13.5, fontweight="bold", color="#111827", ha="left"
    )
    fig.text(
        0.04, 0.89,
        f"Total de variantes incertas reclassificadas com alta confiança deletéria: n = {total_reclass:,} variantes | Estágio 2026.2",
        fontsize=10.0, color="#4B5563", ha="left"
    )

    # Legenda Justificada no Rodapé
    legenda_texto = (
        f"Painel (a) demonstra o impacto da aplicação do modelo AlphaMissense sobre o conjunto de variantes do dbSNP "
        f"sem validação experimental definitiva ou classificadas como VUS no ClinVar, permitindo a elucidação funcional "
        f"de mutações antes incertas. O Painel (b) destaca os principais genes compartilhados entre PsA e AS enriquecidos "
        f"em variantes candidatas de alta prioridade (AlphaMissense ≥ 0,564 com suporte consensual de CADD ≥ 20 e/ou REVEL ≥ 0,5)."
    )
    render_justified_caption(fig, legenda_texto, x=0.04, y=0.17)

    # Salvar
    out_png = os.path.join(pasta_saida, "figura6_reclassificacao_vus_alphamissense.png")
    out_pdf = os.path.join(pasta_saida, "figura6_reclassificacao_vus_alphamissense.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)
    logger.info(f"Figura 6 salva em: {out_png} e {out_pdf}")


def main():
    dir_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description="Geração de Figuras AlphaMissense com formatação padrão.")
    parser.add_argument(
        "--integrada",
        default=os.path.join(dir_base, "data", "silver", "variantes_dbsnp_alphamissense.tsv"),
        help="Caminho da tabela integrada com AlphaMissense."
    )
    parser.add_argument(
        "--vus",
        default=os.path.join(dir_base, "data", "gold", "vus_reclassificadas_hubs.tsv"),
        help="Caminho da tabela de VUS reclassificadas."
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(dir_base, "figuras"),
        help="Diretório de saída para salvar as figuras."
    )
    args = parser.parse_args()

    if not os.path.exists(args.integrada):
        raise FileNotFoundError(f"Arquivo não encontrado: {args.integrada}")
    if not os.path.exists(args.vus):
        raise FileNotFoundError(f"Arquivo não encontrado: {args.vus}")

    df_total = pd.read_csv(args.integrada, sep="\t", low_memory=False)
    df_total["CADD_Num"] = pd.to_numeric(df_total["CADD_Phred"], errors="coerce")
    df_total["REVEL_Num"] = pd.to_numeric(df_total["REVEL_Score"], errors="coerce")
    df_total["AlphaMissense_Score"] = pd.to_numeric(df_total["AlphaMissense_Score"], errors="coerce")

    df_vus = pd.read_csv(args.vus, sep="\t", low_memory=False)

    plot_figura5_alphamissense_concordancia(df_total, args.output_dir)
    plot_figura6_reclassificacao_vus(df_vus, df_total, args.output_dir)


if __name__ == "__main__":
    main()
