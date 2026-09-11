#!/usr/bin/env python3
"""
extrair_alphamissense_genes.py — Extração em Streaming das Predições AlphaMissense para Genes UniProt.

Projeto: Estágio de Bioinformática 2026.2 (Genes Compartilhados PsA e AS)
Fonte: Google Cloud Storage (DeepMind Technologies - AlphaMissense_aa_substitutions.tsv.gz)
Entrada: estagio_2026.2/genes_uniprot_mapeamento.tsv
Saída: estagio_2026.2/alphamissense_predicoes_genes_estagio.tsv
"""

import argparse
import csv
import gzip
import io
import logging
import os
import sys
import time
from typing import Dict, Set
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("extrair_alphamissense")

ALPHAMISSENSE_URL = "https://storage.googleapis.com/dm_alphamissense/AlphaMissense_aa_substitutions.tsv.gz"


def carregar_uniprot_ids(caminho_tsv: str) -> Dict[str, str]:
    """Carrega o conjunto de UniProt IDs mapeados e o respectivo símbolo do gene."""
    if not os.path.exists(caminho_tsv):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_tsv}")

    df = pd.read_csv(caminho_tsv, sep="\t")
    df_validos = df[df["Status"] == "Mapeado"].copy()

    uniprot_to_gene = {}
    for _, row in df_validos.iterrows():
        uid = str(row["UniProt_ID"]).strip()
        gene = str(row["Gene_Query"]).strip()
        if uid and uid.lower() != "nan":
            uniprot_to_gene[uid] = gene

    logger.info(f"Carregados {len(uniprot_to_gene)} UniProt IDs únicos a partir de {caminho_tsv}")
    return uniprot_to_gene


def extrair_predicoes_streaming(
    uniprot_to_gene: Dict[str, str],
    url_fonte: str,
    caminho_saida: str
):
    """
    Realiza o download em streaming do arquivo compactado .tsv.gz do AlphaMissense,
    descompacta em memória em fluxo e filtra exclusivamente as linhas dos UniProt IDs do projeto.
    """
    uids_alvo: Set[str] = set(uniprot_to_gene.keys())
    logger.info(f"Iniciando conexão de streaming com {url_fonte}...")

    session = requests.Session()
    response = session.get(url_fonte, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    logger.info(f"Tamanho do arquivo remoto compactado: {total_size / (1024*1024):.2f} MB")

    os.makedirs(os.path.dirname(os.path.abspath(caminho_saida)), exist_ok=True)
    
    total_linhas_lidas = 0
    total_predicoes_encontradas = 0
    genes_encontrados = set()
    start_time = time.time()

    decompressor = gzip.GzipFile(fileobj=response.raw)
    reader = io.TextIOWrapper(decompressor, encoding="utf-8", errors="replace")

    with open(caminho_saida, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out, delimiter="\t")
        writer.writerow([
            "Gene",
            "uniprot_id",
            "protein_variant",
            "am_pathogenicity",
            "am_class"
        ])

        for line in reader:
            total_linhas_lidas += 1
            if line.startswith("#"):
                continue
            
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue

            uid = parts[0].strip()
            if uid in uids_alvo:
                gene_symbol = uniprot_to_gene[uid]
                variant = parts[1].strip()
                patho = parts[2].strip()
                am_class = parts[3].strip()

                writer.writerow([
                    gene_symbol,
                    uid,
                    variant,
                    patho,
                    am_class
                ])
                total_predicoes_encontradas += 1
                genes_encontrados.add(uid)

            if total_linhas_lidas % 5000000 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Processadas {total_linhas_lidas:,} linhas | "
                    f"Predições encontradas: {total_predicoes_encontradas:,} | "
                    f"Genes com dados: {len(genes_encontrados)}/{len(uids_alvo)} | "
                    f"Tempo: {elapsed:.1f}s"
                )

    tempo_total = time.time() - start_time
    logger.info(
        f"Extração concluída com sucesso em {tempo_total:.1f}s!\n"
        f"- Total de linhas varridas no AlphaMissense: {total_linhas_lidas:,}\n"
        f"- Total de predições salvas: {total_predicoes_encontradas:,}\n"
        f"- Genes com predições mapeadas: {len(genes_encontrados)} de {len(uids_alvo)}\n"
        f"- Arquivo gerado: {caminho_saida}"
    )


def main():
    dir_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description="Extração de predições AlphaMissense para genes UniProt.")
    parser.add_argument(
        "--input",
        default=os.path.join(dir_base, "data", "silver", "genes_uniprot_curados.tsv"),
        help="Caminho do arquivo TSV de mapeamento UniProt."
    )
    parser.add_argument(
        "--output",
        default=os.path.join(dir_base, "data", "bronze", "alphamissense_raw_stream.tsv"),
        help="Caminho do arquivo TSV de saída para predições do AlphaMissense."
    )
    parser.add_argument(
        "--url",
        default=ALPHAMISSENSE_URL,
        help="URL direta do arquivo AlphaMissense_aa_substitutions.tsv.gz."
    )
    args = parser.parse_args()

    uniprot_to_gene = carregar_uniprot_ids(args.input)
    extrair_predicoes_streaming(uniprot_to_gene, args.url, args.output)


if __name__ == "__main__":
    main()
