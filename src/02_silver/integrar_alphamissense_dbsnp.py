#!/usr/bin/env python3
"""
integrar_alphamissense_dbsnp.py — Cruzamento Multi-Fonte dbSNP × AlphaMissense e Anotação de VUS.

Projeto: Estágio de Bioinformática 2026.2 (Genes Compartilhados PsA e AS)
Entradas:
  - estagio_2026.2/tabela_missense_dbsnp_completa.tsv
  - estagio_2026.2/alphamissense_predicoes_genes_estagio.tsv
Saídas:
  - estagio_2026.2/tabela_missense_alphamissense_integrada.tsv (e .csv)
  - estagio_2026.2/tabela_vus_reclassificadas_alphamissense.tsv
"""

import argparse
import logging
import os
import re
import sys
import time
from typing import Optional
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("integrar_alphamissense")

AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O"
}

HGVS_REGEX = re.compile(r"p\.([A-Za-z]{1,3})([0-9]+)([A-Za-z]{1,3})")


def padronizar_variante_proteica(
    hgvs_p: Optional[str],
    aa_ref: Optional[str],
    pos_aa: Optional[str],
    aa_alt: Optional[str]
) -> Optional[str]:
    """Padroniza mutações missense para a notação de 1 letra: ex: R134W, C146W."""
    ref = str(aa_ref).strip().upper() if pd.notna(aa_ref) else ""
    pos = str(pos_aa).strip() if pd.notna(pos_aa) else ""
    alt = str(aa_alt).strip().upper() if pd.notna(aa_alt) else ""

    if pos.endswith(".0"):
        pos = pos[:-2]

    if ref and pos.isdigit() and alt:
        ref_1 = AA3_TO_AA1.get(ref, ref if len(ref) == 1 else "")
        alt_1 = AA3_TO_AA1.get(alt, alt if len(alt) == 1 else "")
        if ref_1 and alt_1 and ref_1 != alt_1:
            return f"{ref_1}{pos}{alt_1}"

    if pd.notna(hgvs_p):
        hgvs_str = str(hgvs_p).strip()
        m = HGVS_REGEX.match(hgvs_str)
        if m:
            r_str, p_str, a_str = m.group(1).upper(), m.group(2), m.group(3).upper()
            r_1 = AA3_TO_AA1.get(r_str, r_str if len(r_str) == 1 else "")
            a_1 = AA3_TO_AA1.get(a_str, a_str if len(a_str) == 1 else "")
            if r_1 and a_1 and r_1 != a_1:
                return f"{r_1}{p_str}{a_1}"

    return None


def classificar_clinvar(row: pd.Series) -> str:
    """Consolida a anotação clínica de ClinVar/UniProt para identificar VUS e benigno/patogênico."""
    cv_myv = str(row.get("ClinVar_MyVariant", "")).lower() if pd.notna(row.get("ClinVar_MyVariant")) else ""
    cv_uni = str(row.get("ClinVar_UniProt", "")).lower() if pd.notna(row.get("ClinVar_UniProt")) else ""
    combined = f"{cv_myv} {cv_uni}".strip()

    if not combined or combined == "nan nan" or combined == "nan":
        return "Sem Anotação Clínica / Experimental"
    
    if "pathogenic" in combined or "patogênica" in combined or "pathogenic/likely_pathogenic" in combined:
        return "Patogênica / Provavelmente Patogênica"
    elif "uncertain" in combined or "vus" in combined or "significance" in combined or "conflicting" in combined:
        return "VUS / Significado Incerto ou Conflitante"
    elif "benign" in combined or "benigna" in combined:
        return "Benigna / Provavelmente Benigna"
    else:
        return "Outras / Não Conclusiva"


def executar_integracao(
    caminho_dbsnp: str,
    caminho_am: str,
    caminho_saida_completa: str,
    caminho_saida_vus: str
):
    logger.info("Iniciando cruzamento dbSNP × AlphaMissense...")
    start_time = time.time()

    # 1. Carregar tabela completa dbSNP
    logger.info(f"Carregando tabela completa dbSNP de {caminho_dbsnp}...")
    df_var = pd.read_csv(caminho_dbsnp, sep="\t", low_memory=False)
    logger.info(f"Variantes dbSNP carregadas: {len(df_var):,} registros")

    # 2. Padronizar variantes proteicas
    logger.info("Padronizando variantes proteicas para notação canônica (ex: R134W)...")
    df_var["Protein_Variant_AM"] = df_var.apply(
        lambda r: padronizar_variante_proteica(
            r.get("HGVS_p"), r.get("AA_Ref"), r.get("Posicao_AA"), r.get("AA_Alt")
        ),
        axis=1
    )

    df_var["uniprot_id_clean"] = df_var["UniProt_ID"].astype(str).str.strip()
    df_var["variant_clean"] = df_var["Protein_Variant_AM"].astype(str).str.strip()

    # 3. Carregar predições AlphaMissense dos genes
    logger.info(f"Carregando predições AlphaMissense de {caminho_am}...")
    df_am = pd.read_csv(caminho_am, sep="\t")
    logger.info(f"Predições AlphaMissense carregadas: {len(df_am):,} registros")

    df_am["uniprot_id_clean"] = df_am["uniprot_id"].astype(str).str.strip()
    df_am["variant_clean"] = df_am["protein_variant"].astype(str).str.strip()
    df_am = df_am.drop_duplicates(subset=["uniprot_id_clean", "variant_clean"])

    # 4. Cruzamento direto via merge vetorizado O(N)
    logger.info("Executando merge relacional vetorizado dbSNP ⨝ AlphaMissense...")
    df_merged = pd.merge(
        df_var,
        df_am[["uniprot_id_clean", "variant_clean", "am_pathogenicity", "am_class"]],
        on=["uniprot_id_clean", "variant_clean"],
        how="left"
    )

    df_merged["AlphaMissense_Score"] = pd.to_numeric(df_merged["am_pathogenicity"], errors="coerce")
    df_merged["AlphaMissense_Class"] = df_merged["am_class"].fillna("nao_mapeado")

    # Classificação Amigável
    def rotulo_am(score, cls):
        if pd.isna(score):
            return "Sem Predição AM"
        if cls == "likely_pathogenic" or score >= 0.564:
            return "Provavelmente Patogênica (AM ≥ 0.564)"
        elif cls == "likely_benign" or score < 0.340:
            return "Provavelmente Benigna (AM < 0.340)"
        else:
            return "Ambígua / Incerta (0.340 ≤ AM < 0.564)"

    df_merged["AlphaMissense_Categoria"] = [
        rotulo_am(s, c) for s, c in zip(df_merged["AlphaMissense_Score"], df_merged["AlphaMissense_Class"])
    ]

    # Status Clínico Consolidado
    df_merged["Status_Clinico_Consolidado"] = df_merged.apply(classificar_clinvar, axis=1)

    # Limpar colunas temporárias
    df_export = df_merged.drop(columns=["uniprot_id_clean", "variant_clean", "am_pathogenicity", "am_class"])

    # 5. Salvar Tabela Completa Integrada
    logger.info(f"Salvando tabela completa integrada em {caminho_saida_completa}...")
    df_export.to_csv(caminho_saida_completa, sep="\t", index=False)
    
    caminho_csv = caminho_saida_completa.replace(".tsv", ".csv")
    df_export.to_csv(caminho_csv, index=False)

    # 6. Identificar e Reclassificar VUS e Não Anotadas
    logger.info("Processando reclassificação de VUS e variantes sem comprovação clínica...")
    
    df_export["CADD_Num"] = pd.to_numeric(df_export["CADD_Phred"], errors="coerce")
    df_export["REVEL_Num"] = pd.to_numeric(df_export["REVEL_Score"], errors="coerce")

    cond_clinica_incerta = df_export["Status_Clinico_Consolidado"].isin([
        "VUS / Significado Incerto ou Conflitante",
        "Sem Anotação Clínica / Experimental"
    ])
    cond_am_patogenica = (df_export["AlphaMissense_Class"] == "likely_pathogenic") | (df_export["AlphaMissense_Score"] >= 0.564)
    cond_cadd_alto = df_export["CADD_Num"] >= 20.0
    cond_revel_alto = df_export["REVEL_Num"] >= 0.5

    df_vus_reclass = df_export[
        cond_clinica_incerta & cond_am_patogenica & (cond_cadd_alto | cond_revel_alto)
    ].copy()

    def definir_consenso(r):
        evidencias = []
        if r["AlphaMissense_Score"] >= 0.564:
            evidencias.append(f"AlphaMissense ({r['AlphaMissense_Score']:.3f})")
        if r["CADD_Num"] >= 20.0:
            evidencias.append(f"CADD ({r['CADD_Num']:.1f})")
        if r["REVEL_Num"] >= 0.5:
            evidencias.append(f"REVEL ({r['REVEL_Num']:.3f})")
        return " + ".join(evidencias)

    df_vus_reclass["Evidencias_Deleterias"] = df_vus_reclass.apply(definir_consenso, axis=1)
    df_vus_reclass["Nivel_Prioridade"] = np.where(
        (df_vus_reclass["AlphaMissense_Score"] >= 0.8) & (df_vus_reclass["CADD_Num"] >= 25.0),
        "Prioridade Máxima (Candidata Crítica)",
        "Alta Prioridade Funcional"
    )

    df_vus_reclass = df_vus_reclass.sort_values(
        by=["AlphaMissense_Score", "CADD_Num"], ascending=[False, False]
    ).drop(columns=["CADD_Num", "REVEL_Num"])

    logger.info(f"Total de VUS / Não Anotadas Reclassificadas como Deletérias: {len(df_vus_reclass):,}")
    df_vus_reclass.to_csv(caminho_saida_vus, sep="\t", index=False)

    elapsed = time.time() - start_time
    logger.info(
        f"Integração finalizada com sucesso em {elapsed:.1f}s!\n"
        f"- Tabela integrada: {caminho_saida_completa} ({len(df_export):,} variantes)\n"
        f"- Tabela de VUS reclassificadas: {caminho_saida_vus} ({len(df_vus_reclass):,} variantes candidatas)"
    )


def main():
    dir_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description="Cruzamento dbSNP × AlphaMissense e reclassificação de VUS.")
    parser.add_argument(
        "--dbsnp",
        default=os.path.join(dir_base, "data", "silver", "variantes_missense_dbsnp.tsv"),
        help="Caminho da tabela missense dbSNP completa."
    )
    parser.add_argument(
        "--alphamissense",
        default=os.path.join(dir_base, "data", "bronze", "alphamissense_raw_stream.tsv"),
        help="Caminho da tabela de predições AlphaMissense dos genes."
    )
    parser.add_argument(
        "--output-completa",
        default=os.path.join(dir_base, "data", "silver", "variantes_dbsnp_alphamissense.tsv"),
        help="Caminho de saída da tabela missense integrada com AlphaMissense."
    )
    parser.add_argument(
        "--output-vus",
        default=os.path.join(dir_base, "data", "gold", "vus_reclassificadas_hubs.tsv"),
        help="Caminho de saída da tabela de VUS reclassificadas."
    )
    args = parser.parse_args()

    executar_integracao(
        args.dbsnp,
        args.alphamissense,
        args.output_completa,
        args.output_vus
    )


if __name__ == "__main__":
    main()
