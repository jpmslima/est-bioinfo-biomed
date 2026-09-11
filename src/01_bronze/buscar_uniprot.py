#!/usr/bin/env python3
"""
buscar_uniprot.py — Busca de Códigos e Metadados UniProt para Lista de Genes (Homo sapiens).

Projeto: Estágio de Bioinformática 2026.1
Arquivo de entrada padrão: estagio_2026.1/GENES EM COMUM - PsA e AS - Página1.tsv

Uso:
    python3 estagio_2026.1/buscar_uniprot.py
    python3 estagio_2026.1/buscar_uniprot.py --input "estagio_2026.1/GENES EM COMUM - PsA e AS - Página1.tsv" --output "estagio_2026.1/genes_uniprot_mapeamento.tsv"
"""

import argparse
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional
import urllib.parse
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("buscar_uniprot")

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"


def extrair_genes_do_arquivo(caminho_arquivo: str) -> List[Dict[str, str]]:
    """
    Lê o arquivo TSV/TXT/CSV e extrai os símbolos de genes,
    tratando múltiplas colunas, espaços e barras de sinônimos (ex: MT-CO2 / COX2).
    """
    if not os.path.exists(caminho_arquivo):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")

    genes_lista: List[Dict[str, str]] = []
    seen = set()

    with open(caminho_arquivo, "r", encoding="utf-8-sig") as f:
        for num_linha, linha in enumerate(f, start=1):
            linha_limpa = linha.strip()
            if not linha_limpa or linha_limpa.startswith("#"):
                continue

            # Quebra por tabs ou múltiplos espaços se houver mais de um gene na linha
            tokens = [t.strip() for t in re.split(r"[\t;]", linha_limpa) if t.strip()]
            for token in tokens:
                # Trata casos com barra como "MT-CO2 / COX2"
                partes = [p.strip() for p in token.split("/") if p.strip()]
                for gene_simbolo in partes:
                    gene_norm = gene_simbolo.upper()
                    if gene_norm not in seen:
                        seen.add(gene_norm)
                        genes_lista.append({
                            "linha_origem": num_linha,
                            "gene_raw": token,
                            "gene_query": gene_simbolo
                        })

    logger.info(f"Total de genes únicos identificados: {len(genes_lista)}")
    return genes_lista


def consultar_uniprot_gene(gene_symbol: str, session: requests.Session, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """
    Consulta o UniProt REST API para um símbolo de gene em Homo sapiens (taxon:9606).
    Prioriza entradas revisadas (Swiss-Prot).
    """
    # 1. Tentativa com correspondência exata do gene e Homo sapiens
    query = f'(gene_exact:"{gene_symbol}" OR gene:"{gene_symbol}") AND organism_id:9606'
    params = {
        "query": query,
        "fields": "accession,id,gene_names,protein_name,reviewed,organism_name,length",
        "format": "json",
        "size": 10
    }

    for tentativa in range(1, max_retries + 1):
        try:
            resp = session.get(UNIPROT_SEARCH_URL, params=params, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                if not results:
                    return None

                # Ordenar para priorizar: 1) Reviewed (Swiss-Prot), 2) Primary gene name exato
                best_entry = None
                for entry in results:
                    reviewed = entry.get("entryType") == "UniProtKB reviewed (Swiss-Prot)"
                    genes = entry.get("genes", [])
                    primary_gene = genes[0].get("geneName", {}).get("value", "") if genes else ""
                    
                    # Se for reviewed e o primary gene for idêntico
                    if reviewed and primary_gene.upper() == gene_symbol.upper():
                        best_entry = entry
                        break
                    # Senão, se for reviewed guarda como candidato
                    if reviewed and best_entry is None:
                        best_entry = entry

                if best_entry is None:
                    best_entry = results[0]

                # Extrai informações limpas
                accession = best_entry.get("primaryAccession", "")
                entry_name = best_entry.get("uniProtkbId", "")
                reviewed = "Swiss-Prot (Reviewed)" if best_entry.get("entryType") == "UniProtKB reviewed (Swiss-Prot)" else "TrEMBL (Unreviewed)"
                
                # Nome da proteína
                protein_desc = best_entry.get("proteinDescription", {})
                rec_name = protein_desc.get("recommendedName", {}).get("fullName", {}).get("value", "")
                if not rec_name:
                    sub_names = protein_desc.get("submissionNames", [])
                    if sub_names:
                        rec_name = sub_names[0].get("fullName", {}).get("value", "")
                if not rec_name:
                    rec_name = "N/A"

                # Nomes de genes e sinônimos
                gene_primary = ""
                synonyms = []
                if best_entry.get("genes"):
                    g_info = best_entry["genes"][0]
                    gene_primary = g_info.get("geneName", {}).get("value", "")
                    for syn in g_info.get("synonyms", []):
                        synonyms.append(syn.get("value", ""))

                length = best_entry.get("sequence", {}).get("length", "")

                return {
                    "uniprot_id": accession,
                    "entry_name": entry_name,
                    "curation_status": reviewed,
                    "protein_name": rec_name,
                    "primary_gene": gene_primary,
                    "synonyms": ", ".join(synonyms) if synonyms else "",
                    "sequence_length_aa": length,
                    "url": f"https://www.uniprot.org/uniprotkb/{accession}/entry"
                }

            elif resp.status_code == 429:
                time.sleep(2 * tentativa)
            else:
                logger.warning(f"Erro HTTP {resp.status_code} para gene '{gene_symbol}'")
        except requests.RequestException as e:
            if tentativa == max_retries:
                logger.error(f"Falha de rede ao consultar '{gene_symbol}': {e}")
            time.sleep(1.5 * tentativa)

    return None


def main():
    dir_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description="Mapeador de genes para UniProt IDs (Homo sapiens)")
    parser.add_argument(
        "--input",
        "-i",
        default=os.path.join(dir_base, "data", "bronze", "genes_input_raw.tsv"),
        help="Caminho do arquivo de entrada com os nomes dos genes"
    )
    parser.add_argument(
        "--output-tsv",
        "-ot",
        default=os.path.join(dir_base, "data", "silver", "genes_uniprot_curados.tsv"),
        help="Caminho do arquivo TSV de saída"
    )
    parser.add_argument(
        "--output-csv",
        "-oc",
        default=os.path.join(dir_base, "data", "silver", "genes_uniprot_curados.csv"),
        help="Caminho do arquivo CSV de saída"
    )
    args = parser.parse_args()

    logger.info(f"Carregando arquivo de entrada: {args.input}")
    genes = extrair_genes_do_arquivo(args.input)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Bioinformatics-Internship-Pipeline/1.0 (research_pipeline)"
    })

    resultados = []
    total = len(genes)
    logger.info(f"Iniciando busca no UniProtKB para {total} genes...")

    for idx, item in enumerate(genes, start=1):
        gene_query = item["gene_query"]
        if idx % 25 == 0 or idx == 1 or idx == total:
            logger.info(f"Processando [{idx}/{total}]: {gene_query}")

        info = consultar_uniprot_gene(gene_query, session)
        if info:
            resultados.append({
                "Gene_Original": item["gene_raw"],
                "Gene_Query": gene_query,
                "Status": "Mapeado",
                "UniProt_ID": info["uniprot_id"],
                "Entry_Name": info["entry_name"],
                "Curacao": info["curation_status"],
                "Nome_Proteina": info["protein_name"],
                "Gene_Primario_UniProt": info["primary_gene"],
                "Sinonimos": info["synonyms"],
                "Tamanho_aa": info["sequence_length_aa"],
                "Link_UniProt": info["url"],
            })
        else:
            # Não encontrado no UniProtKB de proteínas codificantes
            # Geralmente ncRNA, lncRNA, miRNA, pseudogene ou símbolo obsoleto
            motivo = "Não codificante / miRNA / lncRNA / Pseudogene / Não encontrado"
            resultados.append({
                "Gene_Original": item["gene_raw"],
                "Gene_Query": gene_query,
                "Status": "Não Encontrado no UniProtKB (Proteína)",
                "UniProt_ID": "-",
                "Entry_Name": "-",
                "Curacao": "-",
                "Nome_Proteina": motivo,
                "Gene_Primario_UniProt": "-",
                "Sinonimos": "-",
                "Tamanho_aa": "-",
                "Link_UniProt": "-"
            })
        
        # Respeita taxa do servidor UniProt
        time.sleep(0.05)

    df = pd.DataFrame(resultados)
    
    # Salvar saídas
    df.to_csv(args.output_tsv, sep="\t", index=False)
    df.to_csv(args.output_csv, sep=",", index=False)

    n_sucesso = (df["Status"] == "Mapeado").sum()
    n_nao_enc = (df["Status"] != "Mapeado").sum()
    
    logger.info("==================================================")
    logger.info(f"Processamento concluído com sucesso!")
    logger.info(f"Total de genes consultados: {total}")
    logger.info(f"Genes mapeados para UniProt ID: {n_sucesso} ({n_sucesso/total*100:.1f}%)")
    logger.info(f"Genes sem proteína correspondente (ex: RNA/pseudogene): {n_nao_enc} ({n_nao_enc/total*100:.1f}%)")
    logger.info(f"Arquivo TSV salvo em: {args.output_tsv}")
    logger.info(f"Arquivo CSV salvo em: {args.output_csv}")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
