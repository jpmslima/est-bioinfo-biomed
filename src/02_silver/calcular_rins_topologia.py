#!/usr/bin/env python3
"""
calcular_rins_topologia.py — Pipeline de Alta Performance com ProcessPoolExecutor para Cálculo de RINs.

Projeto: Estágio de Bioinformática 2026.2 (Genes Compartilhados PsA e AS)
Etapas:
  1. Leitura dos 426 PDBs já disponíveis no cache local
  2. Construção paralela multi-core dos grafos de contato inter-resíduos (Cα-Cα <= 8.5 Å)
  3. Cálculo das métricas topológicas:
     - RIN_Degree (Conectividade local)
     - RIN_Betweenness (Intermediação estrutural)
     - RIN_Closeness (Acessibilidade global)
     - AlphaFold_pLDDT (Confiança estrutural 3D)
     - RIN_Hub_Status (Hub Global / Hub de Intermediação / Hub Local / Resíduo Regular)
  4. Cruzamento relacional com a base de variantes missense e AlphaMissense

Entradas:
  - estagio_2026.2/genes_uniprot_mapeamento.tsv
  - estagio_2026.2/tabela_missense_alphamissense_integrada.tsv
Saídas:
  - estagio_2026.2/tabela_rins_topologia_residuos.tsv
  - estagio_2026.2/tabela_missense_alphamissense_rins_integrada.tsv
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(ROOT_DIR, "cache_alphafold_pdb")
DISTANCE_CUTOFF = 8.5  # Angstroms


def log_msg(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def processar_um_pdb(args_tuple: Tuple[str, str, str, float]) -> List[Dict]:
    uid, gene_symbol, cache_dir, distance_cutoff = args_tuple
    caminho_pdb = os.path.join(cache_dir, f"{uid}.pdb")
    
    if not os.path.exists(caminho_pdb) or os.path.getsize(caminho_pdb) < 1000:
        return []

    parser = PDBParser(QUIET=True)
    try:
        estrutura = parser.get_structure(uid, caminho_pdb)
    except Exception:
        return []

    residuos_info = []
    coordenadas = []

    for model in estrutura:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                if "CA" in residue:
                    ca_atom = residue["CA"]
                    res_name = residue.get_resname().strip().upper()
                    res_1 = protein_letters_3to1.get(res_name, "X")
                    pos_aa = residue.id[1]
                    plddt = ca_atom.get_bfactor()
                    coords = ca_atom.get_coord()

                    residuos_info.append({
                        "gene": gene_symbol,
                        "uniprot_id": uid,
                        "posicao_aa": pos_aa,
                        "residuo_ref": res_1,
                        "plddt": plddt
                    })
                    coordenadas.append(coords)
        break

    if not coordenadas or len(coordenadas) < 3:
        return []

    coords_arr = np.array(coordenadas)
    matriz_dist = squareform(pdist(coords_arr, metric="euclidean"))

    n_res = len(residuos_info)
    G = nx.Graph()
    G.add_nodes_from(range(n_res))

    for i in range(n_res):
        for j in range(i + 1, n_res):
            if matriz_dist[i, j] <= distance_cutoff:
                G.add_edge(i, j)

    # Métricas Topológicas
    degrees = dict(G.degree())
    if n_res <= 500:
        betweenness = nx.betweenness_centrality(G, normalized=True)
    else:
        betweenness = nx.betweenness_centrality(G, k=min(n_res, 120), normalized=True, seed=42)

    # Closeness otimizado
    closeness = nx.closeness_centrality(G)

    bet_vals = list(betweenness.values())
    deg_vals = list(degrees.values())
    p90_bet = np.percentile(bet_vals, 90) if bet_vals else 0
    p90_deg = np.percentile(deg_vals, 90) if deg_vals else 0

    registros = []
    for i, info in enumerate(residuos_info):
        deg = degrees.get(i, 0)
        bet = betweenness.get(i, 0.0)
        close = closeness.get(i, 0.0)

        is_hub_bet = bet >= p90_bet and bet > 0
        is_hub_deg = deg >= p90_deg and deg > 0

        if is_hub_bet and is_hub_deg:
            hub_status = "Hub Global (Betweenness + Degree)"
        elif is_hub_bet:
            hub_status = "Hub de Intermediação (Betweenness)"
        elif is_hub_deg:
            hub_status = "Hub Local (Degree)"
        else:
            hub_status = "Resíduo Regular"

        registros.append({
            "Gene": info["gene"],
            "UniProt_ID": info["uniprot_id"],
            "Posicao_AA": info["posicao_aa"],
            "Residuo_Ref": info["residuo_ref"],
            "AlphaFold_pLDDT": round(float(info["plddt"]), 2),
            "RIN_Degree": deg,
            "RIN_Betweenness": round(float(bet), 6),
            "RIN_Closeness": round(float(close), 6),
            "RIN_Hub_Status": hub_status
        })

    return registros


def executar_pipeline_rins(
    caminho_genes_tsv: str,
    caminho_saida_rins: str,
    caminho_var_integrada: str,
    caminho_saida_final: str,
    num_workers: int = 6
):
    start_time = time.time()
    log_msg("=== Início do Pipeline de RINs e Topologia Estrutural AlphaFold (Multi-Core) ===")

    df_genes = pd.read_csv(caminho_genes_tsv, sep="\t")
    df_validos = df_genes[df_genes["Status"] == "Mapeado"].drop_duplicates(subset=["UniProt_ID"])

    tarefas = []
    for _, row in df_validos.iterrows():
        uid = str(row["UniProt_ID"]).strip()
        gene = str(row["Gene_Query"]).strip()
        if uid and uid.lower() != "nan":
            tarefas.append((uid, gene, CACHE_DIR, DISTANCE_CUTOFF))

    log_msg(f"Processando topologia de rede em paralelo com {num_workers} processos para {len(tarefas)} proteínas...")

    todos_residuos = []
    processados = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(processar_um_pdb, item): item for item in tarefas}
        for future in as_completed(futures):
            processados += 1
            res_list = future.result()
            if res_list:
                todos_residuos.extend(res_list)

            if processados % 50 == 0 or processados == len(tarefas):
                elapsed_parcial = time.time() - start_time
                log_msg(
                    f"Progresso 3D: {processados}/{len(tarefas)} proteínas | "
                    f"Resíduos estruturados: {len(todos_residuos):,} | "
                    f"Tempo: {elapsed_parcial:.1f}s"
                )

    # 1. Salvar tabela de resíduos
    df_rins = pd.DataFrame(todos_residuos)
    log_msg(f"Salvando {len(df_rins):,} resíduos em {caminho_saida_rins}...")
    df_rins.to_csv(caminho_saida_rins, sep="\t", index=False)

    # 2. Cruzar com a tabela consolidada de variantes
    log_msg("Cruzando métricas topológicas com a tabela consolidada de variantes missense...")
    df_var = pd.read_csv(caminho_var_integrada, sep="\t", low_memory=False)

    df_var["Posicao_AA_Num"] = pd.to_numeric(df_var["Posicao_AA"], errors="coerce")
    df_rins["Posicao_AA_Num"] = pd.to_numeric(df_rins["Posicao_AA"], errors="coerce")

    df_final = pd.merge(
        df_var,
        df_rins[[
            "UniProt_ID", "Posicao_AA_Num",
            "AlphaFold_pLDDT", "RIN_Degree", "RIN_Betweenness", "RIN_Closeness", "RIN_Hub_Status"
        ]],
        on=["UniProt_ID", "Posicao_AA_Num"],
        how="left"
    ).drop(columns=["Posicao_AA_Num"])

    log_msg(f"Salvando base final consolidada (dbSNP + UniProt + AlphaMissense + RINs) em {caminho_saida_final}...")
    df_final.to_csv(caminho_saida_final, sep="\t", index=False)

    caminho_csv = caminho_saida_final.replace(".tsv", ".csv")
    df_final.to_csv(caminho_csv, index=False)

    total_time = time.time() - start_time
    log_msg(f"Pipeline de RINs finalizado com sucesso em {total_time:.1f} segundos!")
    log_msg(f"✓ Total de resíduos estruturados: {len(df_rins):,}")
    log_msg(f"✓ Total de variantes anotadas com RINs: {len(df_final):,}")


def main():
    dir_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument(
        "--genes",
        default=os.path.join(dir_base, "data", "silver", "genes_uniprot_curados.tsv")
    )
    parser.add_argument(
        "--output-rins",
        default=os.path.join(dir_base, "data", "silver", "topologia_rins_residuos.tsv")
    )
    parser.add_argument(
        "--var-integrada",
        default=os.path.join(dir_base, "data", "silver", "variantes_dbsnp_alphamissense.tsv")
    )
    parser.add_argument(
        "--output-final",
        default=os.path.join(dir_base, "data", "silver", "base_mestre_integrada.tsv")
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6
    )
    args = parser.parse_args()

    executar_pipeline_rins(
        args.genes,
        args.output_rins,
        args.var_integrada,
        args.output_final,
        args.workers
    )


if __name__ == "__main__":
    main()
