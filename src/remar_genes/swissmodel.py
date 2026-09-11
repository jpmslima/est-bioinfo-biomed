"""
src/remar_genes/swissmodel.py
Módulo 2: Modelagem 3D Homóloga de Mutantes via API do Swiss-Model e Validação por Ramachandran
Projeto: REMAR Genes — Espondiloartrites (PsA & AS)

Responsabilidades:
  1. Autenticação e submissão à REST API do Swiss-Model (Workspace Automodel)
     utilizando o modelo AlphaFold DB selvagem como molde (mold/template).
  2. Consulta de status e download padronizado: {UniprotID}_mut_{Mutation}.pdb.
  3. Cálculo de ângulos diedros da cadeia principal (Phi e Psi) a partir de coordenadas PDB.
  4. Geração dinâmica do Gráfico de Ramachandran interativo (Plotly) adaptando o método Rebel Pops.
"""

import os
import re
import math
import json
import logging
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple, Any

from Bio.PDB import PDBParser, PPBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SWISS_MODEL_API_URL = "https://swissmodel.expasy.org/automodel"


def aplicar_mutacao_na_sequencia(seq_wt: str, posicao_aa: int, aa_ref: str, aa_alt: str) -> Tuple[str, bool]:
    """
    Substitui o aminoácido de referência pelo aminoácido alternativo na sequência WT (1-indexed).
    Retorna a sequência mutada e um booleano de validação biológica.
    """
    if posicao_aa < 1 or posicao_aa > len(seq_wt):
        return seq_wt, False
    
    idx = posicao_aa - 1
    real_ref = seq_wt[idx]
    
    valido = (real_ref.upper() == aa_ref.upper())
    seq_list = list(seq_wt)
    seq_list[idx] = aa_alt.upper()
    return "".join(seq_list), valido


def submeter_job_swissmodel(
    token: str,
    target_sequence: str,
    mold_pdb_path: str,
    project_title: str = "REMAR_Genes_Variant"
) -> Dict[str, Any]:
    """
    Submete uma requisição de modelagem comparativa para a API do Swiss-Model.
    Utiliza a estrutura selvagem do AlphaFold DB (fornecida via mold_pdb_path) como molde.
    
    Args:
        token: Swiss-Model API Token do usuário.
        target_sequence: Sequência de aminoácidos da isoforma mutada.
        mold_pdb_path: Caminho para o arquivo PDB do AlphaFold (molde selvagem).
        project_title: Título identificador do job no Swiss-Model Workspace.
        
    Returns:
        Dicionário com status da submissão e project_id para acompanhamento.
    """
    token = token.strip() if token else ""
    if not token:
        return {
            "status": "erro",
            "mensagem": "Token da API do Swiss-Model não informado. Forneça o token no painel lateral."
        }

    if not os.path.exists(mold_pdb_path):
        return {
            "status": "erro",
            "mensagem": f"Arquivo PDB molde não encontrado em: {mold_pdb_path}"
        }

    try:
        with open(mold_pdb_path, "r", encoding="utf-8") as f:
            template_pdb_content = f.read()
    except Exception as e:
        return {"status": "erro", "mensagem": f"Falha ao ler molde PDB: {e}"}

    payload = {
        "target_sequences": [target_sequence],
        "project_title": project_title,
        "user_template": template_pdb_content
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        SWISS_MODEL_API_URL,
        data=req_data,
        headers={
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
            "User-Agent": "REMAR_Genes_Client/1.0"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp_body = resp.read().decode("utf-8")
            resp_json = json.loads(resp_body)
            project_id = resp_json.get("project_id")
            return {
                "status": "submetido",
                "project_id": project_id,
                "detalhes": resp_json,
                "mensagem": f"Job de modelagem submetido com sucesso. ID do Projeto: {project_id}"
            }
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8") if e.fp else str(e)
        logger.error(f"Erro HTTP Swiss-Model ({e.code}): {err_msg}")
        return {
            "status": "erro_http",
            "codigo": e.code,
            "mensagem": f"Erro HTTP {e.code} da API Swiss-Model: {err_msg}"
        }
    except Exception as e:
        logger.error(f"Falha de conexão com Swiss-Model API: {e}")
        return {"status": "erro_conexao", "mensagem": str(e)}


def consultar_e_baixar_modelo(
    token: str,
    project_id: str,
    output_dir: str,
    uniprot_id: str,
    mutacao: str
) -> Dict[str, Any]:
    """
    Verifica o status de execução de um job no Swiss-Model e, quando finalizado,
    baixa as coordenadas PDB e salva na nomenclatura estrita solicitada:
    {UniprotID}_mut_{Mutation}.pdb (ex.: Q16552_mut_A55S.pdb)
    """
    token = token.strip()
    url_status = f"{SWISS_MODEL_API_URL}/{project_id}"
    req = urllib.request.Request(
        url_status,
        headers={"Authorization": f"Token {token}"}
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            status_job = data.get("status")

            if status_job == "COMPLETED":
                models = data.get("models", [])
                if not models:
                    return {"status": "erro", "mensagem": "Nenhum modelo gerado pelo Swiss-Model."}
                
                # Obter URL do primeiro modelo (melhor ranking QMEANDisCo)
                coord_url = models[0].get("coordinates_url")
                if not coord_url:
                    return {"status": "erro", "mensagem": "URL das coordenadas PDB não encontrada."}

                # Download do PDB
                req_pdb = urllib.request.Request(coord_url, headers={"Authorization": f"Token {token}"})
                with urllib.request.urlopen(req_pdb, timeout=30) as p_resp:
                    pdb_bytes = p_resp.read()

                os.makedirs(output_dir, exist_ok=True)
                nome_padronizado = f"{uniprot_id}_mut_{mutacao}.pdb"
                caminho_final = os.path.join(output_dir, nome_padronizado)

                with open(caminho_final, "wb") as f_out:
                    f_out.write(pdb_bytes)

                return {
                    "status": "concluido",
                    "caminho_pdb": caminho_final,
                    "nome_arquivo": nome_padronizado,
                    "qmean": models[0].get("qmean"),
                    "gmqr": models[0].get("gmqr")
                }
            elif status_job in ("PENDING", "RUNNING"):
                return {"status": status_job, "mensagem": f"Job em processamento ({status_job})."}
            else:
                return {"status": "falha", "mensagem": f"Job falhou ou foi cancelado: {status_job}"}
    except Exception as e:
        return {"status": "erro_consulta", "mensagem": str(e)}


def gerar_mutante_in_silico_local(
    mold_pdb_path: str,
    output_dir: str,
    uniprot_id: str,
    mutacao: str,
    posicao_aa: int,
    aa_alt: str
) -> str:
    """
    Fallback local para ambiente de teste offline / demonstração interativa:
    Gera o arquivo PDB mutado no padrão oficial {UniprotID}_mut_{Mutation}.pdb
    atualizando o resíduo correspondente no esqueleto atômico do AlphaFold.
    """
    os.makedirs(output_dir, exist_ok=True)
    nome_padrao = f"{uniprot_id}_mut_{mutacao}.pdb"
    caminho_saida = os.path.join(output_dir, nome_padrao)

    mapa_3letras = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"
    }
    novo_res_3 = mapa_3letras.get(aa_alt.upper(), "ALA")

    with open(mold_pdb_path, "r", encoding="utf-8") as f_in, open(caminho_saida, "w", encoding="utf-8") as f_out:
        f_out.write(f"REMARK 000 MODELO MUTANTE GERADO IN SILICO: {uniprot_id} {mutacao}\n")
        f_out.write(f"REMARK 000 MOLDE ALPHA_FOLD: {os.path.basename(mold_pdb_path)}\n")
        for line in f_in:
            if line.startswith("ATOM  ") or line.startswith("HETATM"):
                try:
                    res_num = int(line[22:26].strip())
                    if res_num == posicao_aa:
                        # Substitui o código de 3 letras do resíduo (colunas 18-20)
                        line = line[:17] + f"{novo_res_3:>3}" + line[20:]
                except ValueError:
                    pass
            f_out.write(line)

    return caminho_saida


def classificar_regiao_ramachandran(phi: float, psi: float, res_name: str = "") -> str:
    """
    Classifica o par de ângulos (Phi, Psi) nas regiões clássicas de Ramachandran:
    - Favorecido (Core): Alfa-hélice direita, Folha-Beta, Alfa-hélice esquerda.
    - Permitido (Allowed): Margens de tolerância conformacional.
    - Outlier (Não-permitido): Tensão estérica ou desestabilização severa da cadeia.
    Inspirado nos contornos de Lovell et al. (2003) e adaptado no Rebel Pops.
    """
    # Folha Beta estendida / paralela / antiparalela
    if (-180 <= phi <= -40) and ((60 <= psi <= 180) or (-180 <= psi <= -140)):
        return "Favorecido"
    # Alfa-hélice direita regular
    if (-160 <= phi <= -25) and (-90 <= psi <= 25):
        return "Favorecido"
    # Alfa-hélice esquerda (típica em glicinas)
    if (20 <= phi <= 100) and (0 <= psi <= 80):
        return "Favorecido"

    # Regiões Permitidas expandidas
    if (-180 <= phi <= 0) and (-120 <= psi <= 180):
        return "Permitido"
    if (0 <= phi <= 120) and (-60 <= psi <= 100):
        return "Permitido"

    return "Outlier"


def calcular_angulos_ramachandran(pdb_path: str) -> pd.DataFrame:
    """
    Calcula os ângulos diedros Phi (φ) e Psi (ψ) em graus para todos os aminoácidos
    do modelo PDB utilizando Bio.PDB.
    
    Retorna:
        DataFrame com colunas: ['Residuo_Num', 'Residuo_Nome', 'Phi', 'Psi', 'Regiao_Ramachandran']
    """
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"Arquivo PDB não encontrado: {pdb_path}")

    parser = PDBParser(QUIET=True)
    estrutura = parser.get_structure("modelo", pdb_path)
    ppb = PPBuilder()

    registros = []

    for polypeptide in ppb.build_peptides(estrutura):
        phi_psi_lista = polypeptide.get_phi_psi_list()
        for idx, residuo in enumerate(polypeptide):
            phi, psi = phi_psi_lista[idx]
            if phi is not None and psi is not None:
                phi_graus = math.degrees(phi)
                psi_graus = math.degrees(psi)
                res_nome = residuo.get_resname()
                res_num = residuo.get_id()[1]
                regiao = classificar_regiao_ramachandran(phi_graus, psi_graus, res_nome)

                registros.append({
                    "Residuo_Num": res_num,
                    "Residuo_Nome": res_nome,
                    "Phi": round(phi_graus, 2),
                    "Psi": round(psi_graus, 2),
                    "Regiao_Ramachandran": regiao
                })

    df = pd.DataFrame(registros)
    return df


def plotar_ramachandran_plotly(
    df_rama: pd.DataFrame,
    posicao_mutada: Optional[int] = None,
    titulo: str = "Gráfico de Ramachandran Interativo (Qualidade de Enovelamento)"
) -> go.Figure:
    """
    Renderiza o Gráfico de Ramachandran dinâmico interativo via Plotly
    com contornos de regiões permitidas/favorecidas e destaque na mutação avaliada.
    """
    fig = go.Figure()

    # Contornos de fundo das regiões estereoquímicas clássicas
    # Região 1: Beta-Sheet
    fig.add_shape(
        type="rect", x0=-180, x1=-40, y0=60, y1=180,
        fillcolor="rgba(191, 219, 254, 0.4)", line=dict(color="rgba(59, 130, 246, 0.6)", width=1),
        layer="below"
    )
    fig.add_shape(
        type="rect", x0=-180, x1=-40, y0=-180, y1=-140,
        fillcolor="rgba(191, 219, 254, 0.4)", line=dict(color="rgba(59, 130, 246, 0.6)", width=1),
        layer="below"
    )
    # Região 2: Alfa-Hélice Direita
    fig.add_shape(
        type="rect", x0=-160, x1=-25, y0=-90, y1=25,
        fillcolor="rgba(187, 247, 208, 0.4)", line=dict(color="rgba(34, 197, 94, 0.6)", width=1),
        layer="below"
    )
    # Região 3: Alfa-Hélice Esquerda
    fig.add_shape(
        type="rect", x0=20, x1=100, y0=0, y1=80,
        fillcolor="rgba(254, 240, 138, 0.4)", line=dict(color="rgba(234, 179, 8, 0.6)", width=1),
        layer="below"
    )

    # Cores por categoria
    cores_map = {
        "Favorecido": "#1E40AF",  # Azul escuro
        "Permitido": "#0D9488",   # Verde-petróleo
        "Outlier": "#DC2626"      # Vermelho
    }

    for cat in ["Favorecido", "Permitido", "Outlier"]:
        df_cat = df_rama[df_rama["Regiao_Ramachandran"] == cat]
        if not df_cat.empty:
            fig.add_trace(go.Scatter(
                x=df_cat["Phi"],
                y=df_cat["Psi"],
                mode="markers",
                name=f"{cat} ({len(df_cat)})",
                marker=dict(
                    color=cores_map.get(cat, "#6B7280"),
                    size=7,
                    opacity=0.8,
                    line=dict(width=0.5, color="white")
                ),
                customdata=df_cat[["Residuo_Num", "Residuo_Nome", "Regiao_Ramachandran"]],
                hovertemplate=(
                    "<b>Resíduo:</b> %{customdata[1]}%{customdata[0]}<br>"
                    "<b>Phi (φ):</b> %{x}°<br>"
                    "<b>Psi (ψ):</b> %{y}°<br>"
                    "<b>Status:</b> %{customdata[2]}<extra></extra>"
                )
            ))

    # Destaque do resíduo mutado se fornecido
    if posicao_mutada is not None:
        df_mut = df_rama[df_rama["Residuo_Num"] == posicao_mutada]
        if not df_mut.empty:
            fig.add_trace(go.Scatter(
                x=df_mut["Phi"],
                y=df_mut["Psi"],
                mode="markers+text",
                name="Resíduo Mutado",
                text=[f"MUT: {r['Residuo_Nome']}{r['Residuo_Num']}" for _, r in df_mut.iterrows()],
                textposition="top right",
                marker=dict(
                    symbol="star",
                    size=16,
                    color="#F59E0B",
                    line=dict(width=2, color="#7C2D12")
                ),
                customdata=df_mut[["Residuo_Num", "Residuo_Nome", "Regiao_Ramachandran"]],
                hovertemplate=(
                    "⭐ <b>RESÍDUO MUTADO</b> ⭐<br>"
                    "<b>Resíduo:</b> %{customdata[1]}%{customdata[0]}<br>"
                    "<b>Phi (φ):</b> %{x}°<br>"
                    "<b>Psi (ψ):</b> %{y}°<br>"
                    "<b>Classificação:</b> %{customdata[2]}<extra></extra>"
                )
            ))

    # Métricas globais
    total = len(df_rama)
    n_fav = len(df_rama[df_rama["Regiao_Ramachandran"] == "Favorecido"])
    pct_fav = (n_fav / total * 100) if total > 0 else 0
    n_out = len(df_rama[df_rama["Regiao_Ramachandran"] == "Outlier"])
    pct_out = (n_out / total * 100) if total > 0 else 0

    fig.update_layout(
        title=f"<b>{titulo}</b><br><sup>Favorecidos: {pct_fav:.1f}% | Outliers: {pct_out:.1f}% (Total: {total} resíduos)</sup>",
        xaxis=dict(
            title="Ângulo Phi (φ) [Graus]",
            range=[-180, 180],
            zeroline=True,
            zerolinecolor="#9CA3AF",
            gridcolor="#E5E7EB"
        ),
        yaxis=dict(
            title="Ângulo Psi (ψ) [Graus]",
            range=[-180, 180],
            zeroline=True,
            zerolinecolor="#9CA3AF",
            gridcolor="#E5E7EB"
        ),
        plot_bgcolor="#F9FAFB",
        paper_bgcolor="#FFFFFF",
        width=720,
        height=620,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig
