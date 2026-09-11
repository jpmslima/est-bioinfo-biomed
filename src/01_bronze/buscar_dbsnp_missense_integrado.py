#!/usr/bin/env python3
"""
buscar_dbsnp_missense_integrado.py — Pipeline Integrado Multi-Fonte de Variantes Missense / dbSNP.

Integração de 3 abordagens complementares em Bioinformática:
1. UniProt / EBI Proteins Variation API (Impacto Estrutural, posições AA e patogenicidade anotada)
2. MyVariant.info API (Scores CADD, REVEL, gnomAD Allele Frequency, ClinVar multiconferido)
3. Ensembl REST API (Coordenadas genômicas GRCh38, transcritos e anotações canônicas)

Projeto: Estágio de Bioinformática 2026.1
Entrada: estagio_2026.1/genes_uniprot_mapeamento.tsv (ou lista de genes)

Uso:
    python3 estagio_2026.1/buscar_dbsnp_missense_integrado.py
    python3 estagio_2026.1/buscar_dbsnp_missense_integrado.py --limit 10   # Para teste rápido com 10 genes
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("dbsnp_missense")

EBI_VARIATION_URL = "https://www.ebi.ac.uk/proteins/api/variation"
MYVARIANT_QUERY_URL = "https://myvariant.info/v1/query"
ENSEMBL_REST_URL = "https://rest.ensembl.org"


def carregar_genes_uniprot(caminho_tsv: str) -> List[Dict[str, str]]:
    """Carrega os genes e seus respectivos UniProt IDs a partir da tabela gerada."""
    if not os.path.exists(caminho_tsv):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_tsv}")
    
    df = pd.read_csv(caminho_tsv, sep="\t")
    # Filtrar apenas entradas mapeadas com UniProt ID válido
    df_validos = df[df["Status"] == "Mapeado"].copy()
    
    genes_info = []
    for _, row in df_validos.iterrows():
        genes_info.append({
            "gene": str(row["Gene_Query"]).strip(),
            "uniprot_id": str(row["UniProt_ID"]).strip(),
            "nome_proteina": str(row.get("Nome_Proteina", "")).strip()
        })
    
    logger.info(f"Carregados {len(genes_info)} genes com UniProt ID válido de {caminho_tsv}")
    return genes_info


def extrair_variantes_uniprot(uniprot_id: str, gene_symbol: str, session: requests.Session) -> List[Dict[str, Any]]:
    """
    Abordagem 1: Consulta a API EBI/UniProt Variation para extrair mutações missense (SNVs na proteína).
    """
    url = f"{EBI_VARIATION_URL}/{uniprot_id}"
    headers = {"Accept": "application/json"}
    
    variantes = []
    try:
        resp = session.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            features = data.get("features", [])
            
            for feat in features:
                # Verifica se é variante pontual / missense
                feat_type = feat.get("type")
                wild_type = feat.get("wildType", "")
                mut_type = feat.get("mutatedType", "") or feat.get("alternativeSequence", "")
                
                # Critério missense: troca de 1 AA por outro 1 AA
                if len(wild_type) == 1 and len(mut_type) == 1 and wild_type != mut_type:
                    pos = feat.get("begin", "")
                    
                    # Extrair rsIDs do dbSNP em xrefs
                    rsids = []
                    for xref in feat.get("xrefs", []):
                        if xref.get("name") == "dbSNP":
                            rsids.append(xref.get("id"))
                    
                    rsid_str = "; ".join(rsids) if rsids else "Sem rsID UniProt"
                    
                    # Predições patogênicas (SIFT / PolyPhen no UniProt)
                    sift_val, sift_score = "", ""
                    polyphen_val, polyphen_score = "", ""
                    for pred in feat.get("predictions", []):
                        pred_alg = pred.get("predAlgorithmName", "").lower()
                        if "sift" in pred_alg:
                            sift_val = pred.get("predictionVal", "")
                            sift_score = str(pred.get("score", ""))
                        elif "polyphen" in pred_alg:
                            polyphen_val = pred.get("predictionVal", "")
                            polyphen_score = str(pred.get("score", ""))
                    
                    # ClinVar / Significância Clínica
                    clin_sig = []
                    for cs in feat.get("clinicalSignificances", []):
                        val = cs.get("type", "")
                        if val:
                            clin_sig.append(val)
                    
                    # Coordenada genômica se disponível
                    genomic_loc = feat.get("genomicLocation", "")
                    
                    variantes.append({
                        "Gene": gene_symbol,
                        "UniProt_ID": uniprot_id,
                        "dbSNP_rsID": rsid_str,
                        "Posicao_AA": pos,
                        "AA_Ref": wild_type,
                        "AA_Alt": mut_type,
                        "HGVS_p": f"p.{wild_type}{pos}{mut_type}",
                        "Genomic_Location": genomic_loc,
                        "SIFT_UniProt": f"{sift_val} ({sift_score})".strip(" ()"),
                        "PolyPhen_UniProt": f"{polyphen_val} ({polyphen_score})".strip(" ()"),
                        "ClinVar_UniProt": "; ".join(clin_sig) if clin_sig else "",
                        "Fonte_UniProt": True
                    })
        elif resp.status_code != 404:
            logger.debug(f"UniProt Variation HTTP {resp.status_code} para {uniprot_id} ({gene_symbol})")
    except Exception as e:
        logger.debug(f"Exceção ao consultar UniProt Variation {uniprot_id}: {e}")
        
    return variantes


def consultar_myvariant_gene(gene_symbol: str, session: requests.Session, max_hits: int = 100) -> List[Dict[str, Any]]:
    """
    Abordagem 2: Consulta MyVariant.info API para obter anotações ricas de missense (CADD, gnomAD, ClinVar, REVEL).
    """
    params = {
        "q": f"dbnsfp.genename:{gene_symbol}",
        "fields": "dbsnp.rsid,cadd.phred,gnomad_genome.af.af,gnomad_exome.af.af,clinvar.rcv.clinical_significance,dbnsfp.sift.pred,dbnsfp.polyphen2_hdiv.pred,dbnsfp.revel.score,dbnsfp.aapos,dbnsfp.aaref,dbnsfp.aaalt,dbnsfp.hgvsp",
        "size": max_hits
    }
    
    variantes_mv = []
    try:
        resp = session.get(MYVARIANT_QUERY_URL, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            hits = data.get("hits", [])
            for hit in hits:
                # dbSNP rsID
                dbsnp_info = hit.get("dbsnp", {})
                rsid = ""
                if isinstance(dbsnp_info, dict):
                    rsid = dbsnp_info.get("rsid", "")
                elif isinstance(dbsnp_info, list) and dbsnp_info:
                    rsid = dbsnp_info[0].get("rsid", "")
                elif isinstance(dbsnp_info, str):
                    rsid = dbsnp_info
                
                # CADD Phred
                cadd_info = hit.get("cadd", {})
                cadd_phred = cadd_info.get("phred", "") if isinstance(cadd_info, dict) else ""
                
                # gnomAD AF
                gnomad_g = hit.get("gnomad_genome", {}).get("af", {}).get("af", "") if isinstance(hit.get("gnomad_genome"), dict) else ""
                gnomad_e = hit.get("gnomad_exome", {}).get("af", {}).get("af", "") if isinstance(hit.get("gnomad_exome"), dict) else ""
                gnomad_af = gnomad_g or gnomad_e or ""
                
                # ClinVar
                clinvar_info = hit.get("clinvar", {})
                clin_sig = ""
                if isinstance(clinvar_info, dict):
                    rcv = clinvar_info.get("rcv", {})
                    if isinstance(rcv, dict):
                        clin_sig = str(rcv.get("clinical_significance", ""))
                    elif isinstance(rcv, list) and rcv:
                        clin_sig = str(rcv[0].get("clinical_significance", ""))
                
                # dbNSFP
                dbnsfp = hit.get("dbnsfp", {})
                hgvs_p, pos_aa, aa_ref, aa_alt = "", "", "", ""
                revel = ""
                if isinstance(dbnsfp, dict):
                    raw_hgvsp = dbnsfp.get("hgvsp", "")
                    if isinstance(raw_hgvsp, list):
                        hgvs_p = raw_hgvsp[0] if raw_hgvsp else ""
                    else:
                        hgvs_p = str(raw_hgvsp) if raw_hgvsp else ""
                    
                    pos_aa = str(dbnsfp.get("aapos", "") or "")
                    aa_ref = str(dbnsfp.get("aaref", "") or "")
                    aa_alt = str(dbnsfp.get("aaalt", "") or "")
                    revel = str(dbnsfp.get("revel", {}).get("score", "")) if isinstance(dbnsfp.get("revel"), dict) else ""
                
                # Se faltar pos_aa mas tiver hgvs_p (ex: p.Cys146Trp)
                if not pos_aa and hgvs_p:
                    m_pos = re.search(r"(\d+)", hgvs_p)
                    if m_pos:
                        pos_aa = m_pos.group(1)
                
                if rsid or hgvs_p or (pos_aa and aa_alt):
                    variantes_mv.append({
                        "Gene": gene_symbol,
                        "dbSNP_rsID": rsid if rsid else "Sem rsID MyVariant",
                        "Posicao_AA": pos_aa,
                        "AA_Ref": aa_ref,
                        "AA_Alt": aa_alt,
                        "HGVS_p": hgvs_p,
                        "CADD_Phred": cadd_phred,
                        "gnomAD_AF": gnomad_af,
                        "ClinVar_MyVariant": clin_sig,
                        "REVEL_Score": revel,
                        "Fonte_MyVariant": True
                    })
    except Exception as e:
        logger.debug(f"Exceção MyVariant para {gene_symbol}: {e}")
        
    return variantes_mv


def consultar_ensembl_coordenadas(gene_symbol: str, session: requests.Session) -> Dict[str, str]:
    """
    Abordagem 3: Consulta Ensembl REST API para obter metadados genômicos de referência (GRCh38).
    """
    url = f"{ENSEMBL_REST_URL}/lookup/symbol/homo_sapiens/{gene_symbol}"
    headers = {"Content-Type": "application/json"}
    
    try:
        resp = session.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "Ensembl_Gene_ID": data.get("id", ""),
                "Cromossomo": str(data.get("seq_region_name", "")),
                "Genomic_Start_GRCh38": str(data.get("start", "")),
                "Genomic_End_GRCh38": str(data.get("end", "")),
                "Strand": str(data.get("strand", "")),
                "Biotype": str(data.get("biotype", ""))
            }
    except Exception:
        pass
    
    return {
        "Ensembl_Gene_ID": "",
        "Cromossomo": "",
        "Genomic_Start_GRCh38": "",
        "Genomic_End_GRCh38": "",
        "Strand": "",
        "Biotype": ""
    }


def consolidar_variantes(
    var_uniprot: List[Dict[str, Any]],
    var_myvariant: List[Dict[str, Any]],
    ensembl_info: Dict[str, str],
    gene_symbol: str,
    uniprot_id: str
) -> List[Dict[str, Any]]:
    """
    Cruza e harmoniza os dados das 3 fontes gerando uma tabela unificada por variante.
    """
    # Dicionário indexado por chave unificada (Gene, Posicao_AA, AA_Alt) ou por rsID
    tabela_consolidada: Dict[str, Dict[str, Any]] = {}

    def get_chave(v: Dict[str, Any]) -> str:
        rsid = v.get("dbSNP_rsID", "").strip()
        pos = str(v.get("Posicao_AA", "")).strip()
        alt = str(v.get("AA_Alt", "")).strip()
        if rsid and rsid.startswith("rs"):
            return f"rs_{rsid}"
        if pos and alt:
            return f"pos_{pos}_{alt}"
        return f"hgvs_{v.get('HGVS_p', '')}"

    # 1. Insere UniProt
    for u in var_uniprot:
        k = get_chave(u)
        tabela_consolidada[k] = {
            "Gene": gene_symbol,
            "UniProt_ID": uniprot_id,
            "dbSNP_rsID": u.get("dbSNP_rsID", ""),
            "Posicao_AA": u.get("Posicao_AA", ""),
            "AA_Ref": u.get("AA_Ref", ""),
            "AA_Alt": u.get("AA_Alt", ""),
            "HGVS_p": u.get("HGVS_p", ""),
            "SIFT_UniProt": u.get("SIFT_UniProt", ""),
            "PolyPhen_UniProt": u.get("PolyPhen_UniProt", ""),
            "ClinVar_UniProt": u.get("ClinVar_UniProt", ""),
            "CADD_Phred": "",
            "gnomAD_AF": "",
            "ClinVar_MyVariant": "",
            "REVEL_Score": "",
            "Fonte_UniProt": True,
            "Fonte_MyVariant": False,
            "Fonte_Ensembl": bool(ensembl_info.get("Ensembl_Gene_ID")),
            **ensembl_info
        }

    # 2. Merge MyVariant
    for m in var_myvariant:
        k = get_chave(m)
        if k in tabela_consolidada:
            rec = tabela_consolidada[k]
            rec["Fonte_MyVariant"] = True
            if not rec["dbSNP_rsID"] or "Sem rsID" in rec["dbSNP_rsID"]:
                rec["dbSNP_rsID"] = m.get("dbSNP_rsID", "")
            if not rec["HGVS_p"] and m.get("HGVS_p"):
                rec["HGVS_p"] = m.get("HGVS_p")
            rec["CADD_Phred"] = m.get("CADD_Phred", "")
            rec["gnomAD_AF"] = m.get("gnomAD_AF", "")
            rec["ClinVar_MyVariant"] = m.get("ClinVar_MyVariant", "")
            rec["REVEL_Score"] = m.get("REVEL_Score", "")
        else:
            tabela_consolidada[k] = {
                "Gene": gene_symbol,
                "UniProt_ID": uniprot_id,
                "dbSNP_rsID": m.get("dbSNP_rsID", ""),
                "Posicao_AA": m.get("Posicao_AA", ""),
                "AA_Ref": m.get("AA_Ref", ""),
                "AA_Alt": m.get("AA_Alt", ""),
                "HGVS_p": m.get("HGVS_p", ""),
                "SIFT_UniProt": "",
                "PolyPhen_UniProt": "",
                "ClinVar_UniProt": "",
                "CADD_Phred": m.get("CADD_Phred", ""),
                "gnomAD_AF": m.get("gnomAD_AF", ""),
                "ClinVar_MyVariant": m.get("ClinVar_MyVariant", ""),
                "REVEL_Score": m.get("REVEL_Score", ""),
                "Fonte_UniProt": False,
                "Fonte_MyVariant": True,
                "Fonte_Ensembl": bool(ensembl_info.get("Ensembl_Gene_ID")),
                **ensembl_info
            }

    # 3. Adicionar coluna de consenso de fontes
    linhas_finais = []
    for rec in tabela_consolidada.values():
        fontes = []
        if rec.get("Fonte_UniProt"):
            fontes.append("UniProt")
        if rec.get("Fonte_MyVariant"):
            fontes.append("MyVariant")
        if rec.get("Fonte_Ensembl"):
            fontes.append("Ensembl")
        rec["Consenso_Fontes"] = " + ".join(fontes)
        linhas_finais.append(rec)

    return linhas_finais


def main():
    dir_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description="Download e integração de variantes missense dbSNP de 3 fontes")
    parser.add_argument(
        "--input",
        "-i",
        default=os.path.join(dir_base, "data", "silver", "genes_uniprot_curados.tsv"),
        help="Arquivo TSV com mapeamento de genes e UniProt IDs"
    )
    parser.add_argument(
        "--output-tsv",
        "-ot",
        default=os.path.join(dir_base, "data", "silver", "variantes_missense_dbsnp.tsv"),
        help="Tabela completa de variantes (.tsv)"
    )
    parser.add_argument(
        "--output-csv",
        "-oc",
        default=os.path.join(dir_base, "data", "silver", "variantes_missense_dbsnp.csv"),
        help="Tabela completa de variantes (.csv)"
    )
    parser.add_argument(
        "--output-prioritarias",
        "-op",
        default=os.path.join(dir_base, "data", "gold", "variantes_prioritarias_deleterias.tsv"),
        help="Tabela filtrada com variantes de alto impacto / patogênicas (.tsv)"
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limitar o número de genes processados (útil para testes rápidos)"
    )
    args = parser.parse_args()

    genes_lista = carregar_genes_uniprot(args.input)
    if args.limit:
        logger.info(f"Modo de teste: limitando processamento aos primeiros {args.limit} genes.")
        genes_lista = genes_lista[:args.limit]

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Bioinformatics-Internship-MultiSource/1.0 (research_pipeline)"
    })

    total_genes = len(genes_lista)
    logger.info(f"Iniciando extração multi-fonte para {total_genes} genes...")

    todas_variantes: List[Dict[str, Any]] = []

    for idx, item in enumerate(genes_lista, start=1):
        gene = item["gene"]
        uniprot_id = item["uniprot_id"]
        
        if idx % 10 == 0 or idx == 1 or idx == total_genes:
            logger.info(f"Processando [{idx}/{total_genes}]: Gene '{gene}' (UniProt: {uniprot_id})")

        # 1. UniProt Variation
        var_u = extrair_variantes_uniprot(uniprot_id, gene, session)
        
        # 2. MyVariant.info
        var_mv = consultar_myvariant_gene(gene, session)
        
        # 3. Ensembl REST
        ensembl_info = consultar_ensembl_coordenadas(gene, session)
        
        # Consolidação
        vars_gene = consolidar_variantes(var_u, var_mv, ensembl_info, gene, uniprot_id)
        todas_variantes.extend(vars_gene)
        
        # Intervalo respeitoso de requisições
        time.sleep(0.08)

    logger.info(f"Total de variantes missense integradas: {len(todas_variantes)}")

    if not todas_variantes:
        logger.warning("Nenhuma variante foi coletada.")
        return

    df = pd.DataFrame(todas_variantes)

    # Reordenar colunas de forma lógica e limpa
    cols_ordem = [
        "Gene", "dbSNP_rsID", "HGVS_p", "Posicao_AA", "AA_Ref", "AA_Alt",
        "UniProt_ID", "Ensembl_Gene_ID", "Cromossomo", "Genomic_Start_GRCh38", "Genomic_End_GRCh38", "Strand",
        "SIFT_UniProt", "PolyPhen_UniProt", "CADD_Phred", "REVEL_Score", "gnomAD_AF",
        "ClinVar_UniProt", "ClinVar_MyVariant", "Consenso_Fontes"
    ]
    cols_presentes = [c for c in cols_ordem if c in df.columns]
    resto_cols = [c for c in df.columns if c not in cols_ordem]
    df = df[cols_presentes + resto_cols]

    # Salvar tabela completa
    df.to_csv(args.output_tsv, sep="\t", index=False)
    df.to_csv(args.output_csv, sep=",", index=False)
    logger.info(f"Tabela completa salva em: {args.output_tsv}")
    logger.info(f"Tabela CSV salva em: {args.output_csv}")

    # Criar filtro de variantes prioritárias (CADD > 20, ou ClinVar Patogênica / Provavelmente Patogênica)
    def eh_prioritaria(row) -> bool:
        # CADD > 20
        cadd = row.get("CADD_Phred")
        try:
            if cadd and float(cadd) >= 20.0:
                return True
        except (ValueError, TypeError):
            pass
        # ClinVar
        cv_u = str(row.get("ClinVar_UniProt", "")).lower()
        cv_m = str(row.get("ClinVar_MyVariant", "")).lower()
        if "pathogenic" in cv_u or "pathogenic" in cv_m:
            return True
        return False

    df_prioritarias = df[df.apply(eh_prioritaria, axis=1)].copy()
    df_prioritarias.to_csv(args.output_prioritarias, sep="\t", index=False)
    logger.info(f"Tabela de variantes prioritárias (CADD>=20 ou ClinVar Patogênica) salva com {len(df_prioritarias)} variantes em: {args.output_prioritarias}")

    logger.info("==================================================")
    logger.info("Pipeline Multi-Fonte Concluído com Sucesso!")
    logger.info(f"Genes processados: {total_genes}")
    logger.info(f"Total de variantes coletadas: {len(df)}")
    logger.info(f"Variantes prioritárias (Alto Impacto / ClinVar): {len(df_prioritarias)}")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
