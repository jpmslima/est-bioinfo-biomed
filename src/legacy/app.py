#!/usr/bin/env python3
"""
app.py — Plataforma REMAR Genes: Espondiloartrites (Artrite Psoriásica & Espondilite Anquilosante)
Interface Unificada Streamlit + Plotly + Biofísica Estrutural

Arquitetura:
  - Módulo 1: Anotação Funcional UniProtKB + Predição DTU (SignalP 6) + Alerta de Peptídeo Sinal x pLDDT
  - Módulo 2: Modelagem 3D Homóloga de Mutantes (Swiss-Model API) + Gráfico de Ramachandran Dinâmico (Rebel Pops)
  - Módulo 3: Visualização On-The-Fly (Plotly dinâmico sem figuras estáticas) + Topologia RINs
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Adicionar caminho src ao path
CAMINHO_RAIZ = os.path.dirname(os.path.abspath(__file__))
if os.path.join(CAMINHO_RAIZ, "src") not in sys.path:
    sys.path.insert(0, os.path.join(CAMINHO_RAIZ, "src"))

from remar_genes.annotation import (
    obter_features_uniprot,
    obter_predicoes_dtu,
    verificar_alerta_qualidade_peptideo_sinal
)
from remar_genes.swissmodel import (
    submeter_job_swissmodel,
    consultar_e_baixar_modelo,
    gerar_mutante_in_silico_local,
    calcular_angulos_ramachandran,
    plotar_ramachandran_plotly
)

# Configuração da página
st.set_page_config(
    page_title="REMAR Genes — Espondiloartrites (PsA & AS)",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo visual limpo, profissional e sem poluição
st.markdown("""
<style>
    .titulo-principal {
        font-size: 2.0rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .subtitulo {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.0rem;
    }
    .card-metrica {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .alerta-caixa {
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 5px solid;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Carregando base mestre integrada (Silver/Gold)...")
def carregar_dados_integrados() -> pd.DataFrame:
    """Carrega a tabela consolidada (dbSNP x AlphaMissense x RINs)."""
    caminhos_possiveis = [
        os.path.join(CAMINHO_RAIZ, "tabela_missense_alphamissense_rins_integrada.tsv"),
        os.path.join(CAMINHO_RAIZ, "data", "silver", "base_mestre_integrada.tsv"),
        os.path.join(CAMINHO_RAIZ, "data", "silver", "variantes_dbsnp_alphamissense.tsv")
    ]
    for p in caminhos_possiveis:
        if os.path.exists(p):
            cols_desejadas = [
                "Gene", "UniProt_ID", "dbSNP_rsID", "HGVS_p", "Posicao_AA", "AA_Ref", "AA_Alt",
                "AlphaMissense_Score", "AlphaMissense_Class", "CADD_Phred", "REVEL_Score",
                "AlphaFold_pLDDT", "RIN_Degree", "RIN_Betweenness", "RIN_Closeness", "RIN_Hub_Status",
                "ClinVar_UniProt", "gnomAD_AF"
            ]
            try:
                # Ler colunas existentes para economizar memória
                header = pd.read_csv(p, sep="\t", nrows=1)
                cols_usar = [c for c in cols_desejadas if c in header.columns]
                df = pd.read_csv(p, sep="\t", usecols=cols_usar, low_memory=False)
                # Conversões numéricas seguras
                for num_col in ["Posicao_AA", "AlphaMissense_Score", "CADD_Phred", "AlphaFold_pLDDT", "RIN_Degree", "RIN_Betweenness"]:
                    if num_col in df.columns:
                        df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
                return df
            except Exception as e:
                st.warning(f"Aviso ao carregar {p}: {e}")
    st.error("Nenhuma base integrada encontrada.")
    return pd.DataFrame()


@st.cache_data
def obter_pdb_molde(uniprot_id: str) -> Optional[str]:
    """Localiza o arquivo PDB do AlphaFold DB no cache local."""
    caminho_cache = os.path.join(CAMINHO_RAIZ, "data", "bronze", "alphafold_pdb_cache", f"{uniprot_id}.pdb")
    if os.path.exists(caminho_cache):
        return caminho_cache
    return None


def renderizar_visualizador_3dmol(pdb_content: str, res_destaque: Optional[int] = None, altura: int = 480):
    """Renderiza a estrutura tridimensional via 3Dmol.js."""
    res_destaque_script = ""
    if res_destaque:
        res_destaque_script = f"""
        viewer.addStyle({{resi: {res_destaque}}}, {{stick: {{color: 'red', radius: 0.35}}}});
        viewer.addSphere({{center: {{resi: {res_destaque}}}, radius: 1.8, color: 'orange', opacity: 0.75}});
        viewer.zoomTo({{resi: {res_destaque}}});
        """

    html = f"""
    <div id="container-3d" style="width: 100%; height: {altura}px; position: relative; border-radius: 8px; overflow: hidden; border: 1px solid #CBD5E1;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
        $(document).ready(function() {{
            let element = $('#container-3d');
            let config = {{ backgroundColor: '#F8FAFC' }};
            let viewer = $3Dmol.createViewer(element, config);
            let pdbData = `{pdb_content}`;
            viewer.addModel(pdbData, "pdb");
            
            // Coloração clássica por confiança estrutural pLDDT (B-factor)
            viewer.setStyle({{}}, {{cartoon: {{colorscheme: {{prop: 'b', gradient: 'roygb', min: 50, max: 90}}}}}});
            {res_destaque_script}
            viewer.zoomTo();
            viewer.render();
            viewer.spin("y", 0.3);
        }});
    </script>
    """
    components.html(html, height=altura + 10)


# ==========================================
# INTERFACE PRINCIPAL STREAMLIT
# ==========================================
df_mestre = carregar_dados_integrados()

st.markdown('<div class="titulo-principal">🧬 REMAR Genes — Espondiloartrites (PsA & AS)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitulo">Plataforma Unificada de Genômica Populacional, AlphaMissense, DTU Health Tech e Biofísica de Grafos 3D (RINs)</div>', unsafe_allow_html=True)

if df_mestre.empty:
    st.stop()

# BARRA LATERAL (FILTROS E SELEÇÃO)
st.sidebar.header("🔍 Seleção & Parâmetros")

lista_genes = sorted(df_mestre["Gene"].dropna().unique().tolist())
gene_padrao_idx = lista_genes.index("IL17A") if "IL17A" in lista_genes else 0
gene_selecionado = st.sidebar.selectbox("Selecione o Gene Alvo:", lista_genes, index=gene_padrao_idx)

# Filtrar dados do gene
df_gene = df_mestre[df_mestre["Gene"] == gene_selecionado].copy()
uniprot_id = df_gene["UniProt_ID"].dropna().iloc[0] if not df_gene.empty and "UniProt_ID" in df_gene.columns else "N/A"

st.sidebar.markdown(f"**UniProt ID:** `{uniprot_id}`")
st.sidebar.markdown(f"**Total de Variantes Missense:** `{len(df_gene):,}`")

st.sidebar.markdown("---")
st.sidebar.subheader("🔑 Swiss-Model API")
swiss_token = st.sidebar.text_input("Token Pessoal do Swiss-Model:", type="password", help="Obtido gratuitamente em https://swissmodel.expasy.org/account")
st.sidebar.caption("Caso não informado, o módulo usará o modelador in silico local para demonstração.")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Filtros Analíticos")
corte_cadd = st.sidebar.slider("Corte Mínimo CADD Phred:", min_value=0.0, max_value=40.0, value=20.0, step=1.0)
somente_hubs = st.sidebar.checkbox("Isolar apenas Hubs Estruturais (Top 10%)", value=False)

# Aplicar filtros no dataframe do gene
df_filtrado = df_gene[df_gene["CADD_Phred"] >= corte_cadd] if "CADD_Phred" in df_gene.columns else df_gene
if somente_hubs and "RIN_Hub_Status" in df_filtrado.columns:
    df_filtrado = df_filtrado[df_filtrado["RIN_Hub_Status"].str.contains("Hub", case=False, na=False)]

# CARDS DE MÉTRICAS SUPERIORES
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total no Gene", f"{len(df_gene):,}")
with col2:
    n_prior = len(df_gene[(df_gene["AlphaMissense_Score"] >= 0.564) & (df_gene["CADD_Phred"] >= 20.0)]) if "AlphaMissense_Score" in df_gene.columns else 0
    st.metric("Prioritárias Deletérias", f"{n_prior:,}")
with col3:
    n_hubs = len(df_gene[df_gene["RIN_Hub_Status"].str.contains("Hub", case=False, na=False)]) if "RIN_Hub_Status" in df_gene.columns else 0
    st.metric("Variantes em Hubs 3D", f"{n_hubs:,}")
with col4:
    plddt_medio = df_gene["AlphaFold_pLDDT"].mean() if "AlphaFold_pLDDT" in df_gene.columns else 0.0
    st.metric("pLDDT Médio 3D", f"{plddt_medio:.1f}")
with col5:
    st.metric("Variantes Filtradas", f"{len(df_filtrado):,}")

st.markdown("---")

# ABAS DA APLICAÇÃO
aba1, aba2, aba3, aba4 = st.tabs([
    "📈 Análise On-the-Fly (Lollipop & RINs)",
    "🏷️ DTU Health Tech & Peptídeo Sinal",
    "🧪 Modelagem de Mutantes & Ramachandran",
    "📋 Tabela de Variantes & Download"
])

# -------------------------------------------------------------
# ABA 1: GRÁFICOS DINÂMICOS ON-THE-FLY
# -------------------------------------------------------------
with aba1:
    st.subheader(f"📊 Distribuição Posicional e Biofísica das Variantes — {gene_selecionado}")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        # Gráfico Lollipop on-the-fly (Posição AA vs AlphaMissense Score)
        st.markdown("**Escore AlphaMissense ao longo da Sequência (Lollipop Plot):**")
        if not df_filtrado.empty and "Posicao_AA" in df_filtrado.columns and "AlphaMissense_Score" in df_filtrado.columns:
            fig_lol = px.scatter(
                df_filtrado,
                x="Posicao_AA",
                y="AlphaMissense_Score",
                color="AlphaMissense_Class" if "AlphaMissense_Class" in df_filtrado.columns else None,
                size="CADD_Phred" if "CADD_Phred" in df_filtrado.columns else None,
                hover_data=["HGVS_p", "dbSNP_rsID", "RIN_Degree", "AlphaFold_pLDDT"],
                color_discrete_map={"pathogenic": "#DC2626", "ambiguous": "#F59E0B", "benign": "#10B981"},
                labels={"Posicao_AA": "Posição do Aminoácido", "AlphaMissense_Score": "AlphaMissense Score"}
            )
            # Adicionar linha vertical tipo pirulito para as mais severas
            fig_lol.update_layout(
                height=450,
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_lol.add_hline(y=0.564, line_dash="dash", line_color="#DC2626", annotation_text="Corte Deletério (0.564)")
            fig_lol.add_hline(y=0.340, line_dash="dash", line_color="#10B981", annotation_text="Corte Benigno (0.340)")
            st.plotly_chart(fig_lol, use_container_width=True)
        else:
            st.info("Nenhuma variante atende aos filtros atuais.")

    with col_g2:
        # Gráfico de Redes Estruturais (Posição AA vs Betweenness Centrality)
        st.markdown("**Centralidade de Intermediação 3D (Betweenness Centrality - RINs):**")
        if not df_filtrado.empty and "Posicao_AA" in df_filtrado.columns and "RIN_Betweenness" in df_filtrado.columns:
            fig_bet = px.scatter(
                df_filtrado,
                x="Posicao_AA",
                y="RIN_Betweenness",
                color="AlphaMissense_Score" if "AlphaMissense_Score" in df_filtrado.columns else None,
                size="RIN_Degree" if "RIN_Degree" in df_filtrado.columns else None,
                color_continuous_scale="Viridis",
                hover_data=["HGVS_p", "RIN_Hub_Status", "AlphaFold_pLDDT"],
                labels={"Posicao_AA": "Posição do Aminoácido", "RIN_Betweenness": "Betweenness Centrality (Grafo 3D)"}
            )
            fig_bet.update_layout(height=450, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
            st.plotly_chart(fig_bet, use_container_width=True)
        else:
            st.info("Métricas de Betweenness não disponíveis para a seleção.")

    # Visualizador 3D do Molde AlphaFold
    st.markdown("---")
    st.subheader(f"🌐 Visualização Estrutural 3D AlphaFold — {gene_selecionado} (`{uniprot_id}`)")
    caminho_pdb_wt = obter_pdb_molde(uniprot_id)
    if caminho_pdb_wt:
        with open(caminho_pdb_wt, "r") as f:
            conteudo_pdb = f.read()
        renderizar_visualizador_3dmol(conteudo_pdb, altura=420)
        st.caption("🎨 Modelo colorido por pLDDT: Azul = Muito Alta (>90), Ciano = Alta (70-90), Amarelo = Baixa (50-70), Laranja = Muito Baixa (<50).")
    else:
        st.warning(f"Estrutura PDB AlphaFold para {uniprot_id} não encontrada em cache local.")

# -------------------------------------------------------------
# ABA 2: DTU HEALTH TECH & ALERTA DE PEPTÍDEO SINAL
# -------------------------------------------------------------
with aba2:
    st.subheader("🏷️ Anotações Funcionais (UniProt REST API) & Predições DTU Health Tech")
    
    with st.spinner("Consultando features funcionais na API UniProt e DTU..."):
        info_uniprot = obter_features_uniprot(uniprot_id)
        pred_dtu = obter_predicoes_dtu(uniprot_id)

    col_dtu1, col_dtu2 = st.columns([1, 1])
    
    with col_dtu1:
        st.markdown("#### 🔬 Predições de Endereçamento Celular (DTU)")
        sp_info = pred_dtu.get("signalp_6", {})
        tp_info = pred_dtu.get("targetp_2", {})
        dl_info = pred_dtu.get("deeploc_2", {})

        st.write(f"**SignalP 6.0:** Peptídeo Sinal: `{'SIM' if sp_info.get('tem_peptideo_sinal') else 'NÃO'}` | Tipo: `{sp_info.get('tipo_sinal')}` | Clivagem CS: `{sp_info.get('posicao_clivagem_cs')}`")
        st.write(f"**TargetP 2.0:** Predição de Trânsito: `{tp_info.get('predicao')}` (Prob SP: `{tp_info.get('prob_SP', 0.0):.2f}`)")
        st.write(f"**DeepLoc 2.0:** Localização Subcelular: `{dl_info.get('localizacao_primaria')}` | `{dl_info.get('tipo_membrana')}`")

    with col_dtu2:
        st.markdown("#### 🧬 Features Estruturais (UniProtKB)")
        st.write(f"**Proteína:** {info_uniprot.get('nome_proteina', 'N/A')}")
        st.write(f"**Tamanho:** {info_uniprot.get('tamanho_aa', 0)} aminoácidos")
        st.write(f"**Sítios Ativos/Ligação:** {len(info_uniprot.get('sitios_ativos', []))}")
        st.write(f"**Pontes Dissulfeto:** {len(info_uniprot.get('pontes_dissulfeto', []))}")
        st.write(f"**Sítios de Glicosilação:** {len(info_uniprot.get('glicosilacoes', []))}")

    # TESTE INTERATIVO DE ALERTA DE QUALIDADE ESTRUTURAL
    st.markdown("---")
    st.markdown("### ⚠️ Auditoria de Qualidade Estrutural (Peptídeo Sinal x AlphaFold pLDDT)")
    
    col_al1, col_al2 = st.columns([1, 2])
    with col_al1:
        # Seletor de posição para simulação/auditoria
        pos_teste = st.number_input("Inspecionar Posição de Aminoácido (AA):", min_value=1, max_value=max(info_uniprot.get("tamanho_aa", 500), 1), value=10)
        plddt_val = None
        if not df_gene.empty and "Posicao_AA" in df_gene.columns:
            m_pos = df_gene[df_gene["Posicao_AA"] == pos_teste]
            if not m_pos.empty and "AlphaFold_pLDDT" in m_pos.columns:
                plddt_val = float(m_pos["AlphaFold_pLDDT"].iloc[0])
        
        if plddt_val is None:
            plddt_val = st.slider("AlphaFold pLDDT simulado da posição:", min_value=0.0, max_value=100.0, value=48.0)
        else:
            st.write(f"pLDDT registrado no modelo: **{plddt_val:.1f}**")

    with col_al2:
        intervalo_sp = sp_info.get("intervalo")
        diagnostico = verificar_alerta_qualidade_peptideo_sinal(pos_teste, intervalo_sp, plddt_val)
        
        cor_borda = diagnostico["cor_alerta"]
        st.markdown(f"""
        <div class="alerta-caixa" style="background-color: #F8FAFC; border-left-color: {cor_borda};">
            <h4 style="color: {cor_borda}; margin-top: 0;">{diagnostico['titulo']}</h4>
            <p><strong>Diagnóstico:</strong> {diagnostico['mensagem']}</p>
            <p><em><strong>Recomendação Técnica:</strong> {diagnostico['recomendacao']}</em></p>
        </div>
        """, unsafe_allow_html=True)

    # Tabelas de Sítios Funcionais
    st.markdown("#### Detalhes de Sítios Ativos, Pontes Dissulfeto e Glicosilação")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if info_uniprot.get("sitios_ativos"):
            st.dataframe(pd.DataFrame(info_uniprot["sitios_ativos"]), use_container_width=True)
        else:
            st.info("Nenhum sítio ativo/ligação catalítica anotado formalmente no UniProtKB.")
    with col_t2:
        if info_uniprot.get("pontes_dissulfeto") or info_uniprot.get("glicosilacoes"):
            todos_ptm = info_uniprot.get("pontes_dissulfeto", []) + info_uniprot.get("glicosilacoes", [])
            st.dataframe(pd.DataFrame(todos_ptm), use_container_width=True)
        else:
            st.info("Sem pontes dissulfeto ou sítios de glicosilação mapeados.")

# -------------------------------------------------------------
# ABA 3: MODELAGEM SWISS-MODEL & RAMACHANDRAN
# -------------------------------------------------------------
with aba3:
    st.subheader("🧪 Modelagem 3D Homóloga de Mutantes & Avaliação por Ramachandran")
    st.markdown(r"""
    Este módulo executa a requisição de modelagem comparativa via **Swiss-Model API** (ou modelador in silico local) 
    utilizando a estrutura do **AlphaFold DB selvagem como molde**. Em seguida, avalia dinamicamente os ângulos diedros 
    $(\phi, \psi)$ no **Gráfico de Ramachandran** (método *Rebel Pops*).
    """)

    col_m1, col_m2 = st.columns([1, 2])
    
    with col_m1:
        # Seletor de mutação do gene
        opcoes_mut = []
        if not df_gene.empty and "HGVS_p" in df_gene.columns:
            opcoes_mut = df_gene["HGVS_p"].dropna().unique().tolist()
        
        mut_escolhida = st.selectbox("Selecione a Mutação Candidata:", opcoes_mut if opcoes_mut else ["p.A55S"])
        
        # Extrair dados da mutação
        linha_mut = df_gene[df_gene["HGVS_p"] == mut_escolhida].iloc[0] if mut_escolhida in df_gene["HGVS_p"].values else None
        
        pos_aa_mut = int(linha_mut["Posicao_AA"]) if linha_mut is not None and pd.notnull(linha_mut["Posicao_AA"]) else 55
        aa_ref_mut = str(linha_mut["AA_Ref"]) if linha_mut is not None and pd.notnull(linha_mut["AA_Ref"]) else "A"
        aa_alt_mut = str(linha_mut["AA_Alt"]) if linha_mut is not None and pd.notnull(linha_mut["AA_Alt"]) else "S"
        mut_codigo = f"{aa_ref_mut}{pos_aa_mut}{aa_alt_mut}"

        st.write(f"**Posição:** {pos_aa_mut} | **Ref:** {aa_ref_mut} $\\rightarrow$ **Alt:** {aa_alt_mut}")
        if linha_mut is not None and "AlphaMissense_Score" in linha_mut:
            st.write(f"**AlphaMissense:** `{linha_mut['AlphaMissense_Score']:.3f}` ({linha_mut.get('AlphaMissense_Class', '')})")
            st.write(f"**CADD Phred:** `{linha_mut.get('CADD_Phred', 0.0):.1f}`")

        caminho_molde_af = obter_pdb_molde(uniprot_id)
        btn_modelar = st.button("🚀 Gerar e Avaliar Modelo Mutante", type="primary")

    with col_m2:
        if btn_modelar:
            if not caminho_molde_af:
                st.error(f"Molde selvagem AlphaFold ({uniprot_id}.pdb) não encontrado.")
            else:
                pasta_saida = os.path.join(CAMINHO_RAIZ, "data", "modelos_mutantes")
                os.makedirs(pasta_saida, exist_ok=True)
                
                # Nomenclatura estrita requerida: {UniprotID}_mut_{Mutation}.pdb
                nome_esperado = f"{uniprot_id}_mut_{mut_codigo}.pdb"
                caminho_pdb_mut = os.path.join(pasta_saida, nome_esperado)

                with st.spinner("Construindo coordenadas do modelo mutante..."):
                    # 1. Tentar via Swiss-Model API caso token informado
                    if swiss_token:
                        seq_wt = info_uniprot.get("sequencia", "")
                        # Substituição
                        seq_mut = list(seq_wt)
                        if 1 <= pos_aa_mut <= len(seq_mut):
                            seq_mut[pos_aa_mut - 1] = aa_alt_mut
                        seq_mut_str = "".join(seq_mut)

                        res_sub = submeter_job_swissmodel(
                            token=swiss_token,
                            target_sequence=seq_mut_str,
                            mold_pdb_path=caminho_molde_af,
                            project_title=f"REMAR_{uniprot_id}_{mut_codigo}"
                        )
                        st.info(f"Swiss-Model API: {res_sub.get('mensagem')}")
                        if res_sub.get("status") == "submetido":
                            pid = res_sub.get("project_id")
                            # Baixar modelo
                            res_down = consultar_e_baixar_modelo(swiss_token, pid, pasta_saida, uniprot_id, mut_codigo)
                            if res_down.get("status") == "concluido":
                                caminho_pdb_mut = res_down.get("caminho_pdb")
                            else:
                                # Fallback para modelagem in silico imediata
                                caminho_pdb_mut = gerar_mutante_in_silico_local(
                                    caminho_molde_af, pasta_saida, uniprot_id, mut_codigo, pos_aa_mut, aa_alt_mut
                                )
                        else:
                            caminho_pdb_mut = gerar_mutante_in_silico_local(
                                caminho_molde_af, pasta_saida, uniprot_id, mut_codigo, pos_aa_mut, aa_alt_mut
                            )
                    else:
                        # Modelagem in silico local imediata
                        caminho_pdb_mut = gerar_mutante_in_silico_local(
                            caminho_molde_af, pasta_saida, uniprot_id, mut_codigo, pos_aa_mut, aa_alt_mut
                        )

                st.success(f"✅ Modelo Mutante pronto: `{os.path.basename(caminho_pdb_mut)}`")
                
                # Download direto do PDB padronizado
                with open(caminho_pdb_mut, "rb") as f_pdb:
                    st.download_button(
                        label=f"💾 Baixar PDB Mutante ({nome_esperado})",
                        data=f_pdb,
                        file_name=nome_esperado,
                        mime="chemical/x-pdb"
                    )

                # 2. Cálculo e Renderização do Gráfico de Ramachandran
                with st.spinner("Calculando ângulos de Ramachandran (Phi/Psi)..."):
                    df_rama = calcular_angulos_ramachandran(caminho_pdb_mut)
                    fig_rama = plotar_ramachandran_plotly(
                        df_rama,
                        posicao_mutada=pos_aa_mut,
                        titulo=f"Ramachandran: {nome_esperado}"
                    )
                    st.plotly_chart(fig_rama, use_container_width=True)

# -------------------------------------------------------------
# ABA 4: TABELA DE VARIANTES & EXPORTAÇÃO
# -------------------------------------------------------------
with aba4:
    st.subheader(f"📋 Variantes Filtradas para {gene_selecionado}")
    
    colunas_exibir = [
        c for c in [
            "HGVS_p", "dbSNP_rsID", "Posicao_AA", "AA_Ref", "AA_Alt",
            "AlphaMissense_Score", "AlphaMissense_Class", "CADD_Phred", "REVEL_Score",
            "AlphaFold_pLDDT", "RIN_Degree", "RIN_Betweenness", "RIN_Hub_Status", "ClinVar_UniProt"
        ] if c in df_filtrado.columns
    ]

    st.dataframe(
        df_filtrado[colunas_exibir].sort_values(by="AlphaMissense_Score", ascending=False) if "AlphaMissense_Score" in df_filtrado.columns else df_filtrado[colunas_exibir],
        use_container_width=True,
        height=400
    )

    # Botão de exportação TSV
    tsv_bytes = df_filtrado[colunas_exibir].to_csv(sep="\t", index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Baixar Variantes Filtradas de {gene_selecionado} (.tsv)",
        data=tsv_bytes,
        file_name=f"remar_genes_{gene_selecionado}_filtrado.tsv",
        mime="text/tab-separated-values"
    )
