#!/usr/bin/env python3
"""
app_interativo_psa_as.py — Interface Interativa Streamlit + Plotly + 3Dmol.js para Análise de Genes Compartilhados entre PsA e AS.

Projeto: Estágio de Bioinformática 2026.2
Recursos:
  - Seleção dinâmica entre todos os 428 genes mapeados (ex.: FOXF2, SMAD3, PLCG1, TYK2, IL17A)
  - Visualizador 3D Interativo de Proteínas (AlphaFold PDB via 3Dmol.js com rotação, zoom e destaque de resíduos mutados)
  - Gráficos gerados on-the-fly (sob demanda) via Plotly com hover, zoom e filtros dinâmicos
  - Integração de dbSNP, AlphaMissense, CADD, ClinVar, gnomAD e RINs (Redes de Interação de Resíduos)
  - Tabela filtrável e exportação em CSV/TSV
  - Painel de validação estatística e estudos de caso
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="PsA & AS — Genômica e Topologia Estrutural 3D",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS modernos
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .viewer-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

CAMINHO_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def obter_caminho_existente(rel_novo: str, rel_antigo: str) -> str:
    p_novo = os.path.join(CAMINHO_BASE, rel_novo)
    if os.path.exists(p_novo):
        return p_novo
    return os.path.join(CAMINHO_BASE, rel_antigo)

CAMINHO_VAR = obter_caminho_existente("data/silver/base_mestre_integrada.tsv", "tabela_missense_alphamissense_rins_integrada.tsv")
CAMINHO_GENES = obter_caminho_existente("data/silver/genes_uniprot_curados.tsv", "genes_uniprot_mapeamento.tsv")
CAMINHO_STAT = obter_caminho_existente("data/gold/relatorio_kruskal_dunn.tsv", "relatorio_kruskal_dunn_posthoc.tsv")
PASTA_PDB = obter_caminho_existente("data/bronze/alphafold_pdb_cache", "cache_alphafold_pdb")


@st.cache_data(show_spinner="Carregando catálogo de genes mapeados...")
def carregar_catalogo_genes():
    if os.path.exists(CAMINHO_GENES):
        df_g = pd.read_csv(CAMINHO_GENES, sep="\t")
        return df_g
    return pd.DataFrame()


@st.cache_data(show_spinner="Indexando base consolidada com dbSNP, AlphaMissense e RINs (74 MB)...")
def carregar_base_consolidada():
    if not os.path.exists(CAMINHO_VAR):
        st.error(f"Arquivo principal não encontrado: {CAMINHO_VAR}")
        return pd.DataFrame()
    df = pd.read_csv(CAMINHO_VAR, sep="\t", low_memory=False)
    
    # Conversões numéricas seguras e aliases
    if "dbSNP_rsID" in df.columns:
        df["rsID"] = df["dbSNP_rsID"]
    elif "rsID" in df.columns:
        df["dbSNP_rsID"] = df["rsID"]
    df["Posicao_AA_Num"] = pd.to_numeric(df["Posicao_AA"], errors="coerce")
    df["AlphaMissense_Score"] = pd.to_numeric(df["AlphaMissense_Score"], errors="coerce")
    df["CADD_Phred"] = pd.to_numeric(df["CADD_Phred"], errors="coerce")
    df["CADD_Plot_Size"] = df["CADD_Phred"].fillna(5.0).clip(lower=4.0, upper=35.0)
    df["RIN_Betweenness"] = pd.to_numeric(df["RIN_Betweenness"], errors="coerce")
    df["RIN_Degree"] = pd.to_numeric(df["RIN_Degree"], errors="coerce")
    df["RIN_Closeness"] = pd.to_numeric(df["RIN_Closeness"], errors="coerce")
    df["AlphaFold_pLDDT"] = pd.to_numeric(df["AlphaFold_pLDDT"], errors="coerce")
    return df


def carregar_pdb_texto(uniprot_id: str) -> str:
    """Carrega o arquivo PDB do cache local ou retorna string vazia."""
    caminho_pdb = os.path.join(PASTA_PDB, f"{uniprot_id}.pdb")
    if os.path.exists(caminho_pdb):
        with open(caminho_pdb, "r") as f:
            return f.read()
    return ""


def gerar_visualizador_3d_html(pdb_data: str, estilo_render: str = "cartoon", coloracao: str = "spectrum", res_destaque: int = None) -> str:
    """Gera visualizador molecular interativo 3Dmol.js embutido em HTML."""
    if not pdb_data:
        return "<div style='color: #64748B; padding: 20px;'>Estrutura 3D não disponível localmente para esta proteína.</div>"

    # Configuração de estilo e cor no 3Dmol.js
    if coloracao == "pLDDT (AlphaFold)":
        cor_script = """
            viewer.setColorByFunction({}, function(atom) {
                var b = atom.b;
                if (b >= 90) return '#0053D6'; // Muito alta (>90)
                if (b >= 70) return '#65CBF3'; // Alta (70-90)
                if (b >= 50) return '#FFDB13'; // Baixa (50-70)
                return '#FF7D45';             // Muito baixa (<50)
            });
        """
    elif coloracao == "spectrum":
        cor_script = "viewer.setStyle({}, {cartoon: {color: 'spectrum'}});"
    elif coloracao == "cyan":
        cor_script = "viewer.setStyle({}, {cartoon: {color: '#0072B2'}});"
    else:
        cor_script = "viewer.setStyle({}, {cartoon: {color: 'spectrum'}});"

    destaque_script = ""
    if res_destaque and res_destaque > 0:
        destaque_script = f"""
            viewer.addStyle({{resi: {res_destaque}}}, {{sphere: {{color: '#D55E00', radius: 1.8}}, stick: {{color: '#D55E00', radius: 0.4}}}});
            viewer.addLabel("Resíduo {res_destaque}", {{fontSize: 12, fontColor: '#FFFFFF', backgroundColor: '#D55E00', backgroundOpacity: 0.9}}, {{resi: {res_destaque}}});
        """

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
        <style>
            #container {{
                width: 100%;
                height: 480px;
                position: relative;
                border-radius: 8px;
                overflow: hidden;
                background-color: #0F172A;
            }}
        </style>
    </head>
    <body>
        <div id="container"></div>
        <script>
            let element = document.getElementById('container');
            let config = {{ backgroundColor: '#0F172A' }};
            let viewer = $3Dmol.createViewer(element, config);
            let pdbUri = `{pdb_data}`;
            viewer.addModel(pdbUri, "pdb");
            {cor_script}
            {destaque_script}
            viewer.zoomTo();
            viewer.render();
            viewer.spin("y", 0.3);
        </script>
    </body>
    </html>
    """
    return html_code


df_genes_cat = carregar_catalogo_genes()
df_master = carregar_base_consolidada()

# Paleta Acessível Okabe-Ito
PALETA_CORES = {
    "pathogenic": "#D55E00",
    "likely_pathogenic": "#D55E00",
    "ambiguous": "#E69F00",
    "likely_benign": "#009E73",
    "benign": "#009E73",
    "ClinVar Patogênica": "#D55E00",
    "Alto Impacto CADD": "#E69F00",
    "VUS Reclassificada": "#D55E00"
}


# ==============================================================================
# SIDEBAR — CONTROLES E FILTROS DINÂMICOS
# ==============================================================================
st.sidebar.title("🧬 Navegação & Filtros")

modo_visao = st.sidebar.radio(
    "Selecione o Módulo:",
    ["Explorador por Gene", "Visão Global do Genoma", "Estudos de Caso (SMAD3 / PLCG1)", "Relatório Estatístico"]
)

lista_genes_disponiveis = sorted(df_master["Gene"].dropna().unique()) if not df_master.empty else []

if modo_visao == "Explorador por Gene":
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Seleção do Gene Alvo")
    
    idx_padrao = lista_genes_disponiveis.index("SMAD3") if "SMAD3" in lista_genes_disponiveis else 0
    gene_selecionado = st.sidebar.selectbox(
        "Escolha um gene compartilhado (PsA / AS):",
        lista_genes_disponiveis,
        index=idx_padrao,
        help="Pesquise pelo símbolo oficial do gene (ex.: FOXF2, SMAD3, PLCG1, TYK2, IL17A...)"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Filtros de Patogenicidade e Rede")
    
    classes_am = st.sidebar.multiselect(
        "Classe AlphaMissense:",
        ["pathogenic", "ambiguous", "benign"],
        default=["pathogenic", "ambiguous", "benign"],
        format_func=lambda x: {"pathogenic": "Provavelmente Patogênica", "ambiguous": "Ambígua / Incerta", "benign": "Provavelmente Benigna"}.get(x, x)
    )
    
    min_cadd = st.sidebar.slider(
        "Corte Mínimo CADD Phred:",
        min_value=0.0,
        max_value=40.0,
        value=0.0,
        step=1.0,
        help="CADD ≥ 20 representa o top 1% de variantes mais deletérias do genoma."
    )
    
    apenas_hubs = st.sidebar.checkbox("Exibir apenas Hubs Estruturais (Top 10% Centralidade)", value=False)
    apenas_vus = st.sidebar.checkbox("Apenas VUS / Sem Anotação Clínica Prévia", value=False)


# ==============================================================================
# MÓDULO 1: EXPLORADOR POR GENE
# ==============================================================================
if modo_visao == "Explorador por Gene":
    st.markdown(f'<div class="main-header">Gene: {gene_selecionado}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Mapeamento integrado de polimorfismos (dbSNP), predições estruturais (AlphaMissense), Redes de Interação (RINs) e Visualização 3D AlphaFold</div>', unsafe_allow_html=True)

    # Filtrar dados do gene
    df_gene = df_master[df_master["Gene"] == gene_selecionado].copy()
    
    # Metadados do Gene
    info_gene = df_genes_cat[(df_genes_cat["Gene_Original"] == gene_selecionado) | (df_genes_cat["Gene_Query"] == gene_selecionado)] if not df_genes_cat.empty else pd.DataFrame()
    if not info_gene.empty and "UniProt_ID" in info_gene.columns:
        uniprot_id = info_gene["UniProt_ID"].values[0]
        nome_prot = info_gene["Nome_Proteina"].values[0] if "Nome_Proteina" in info_gene.columns else gene_selecionado
    elif "UniProt_ID" in df_gene.columns and len(df_gene["UniProt_ID"].dropna()) > 0:
        uniprot_id = df_gene["UniProt_ID"].dropna().values[0]
        nome_prot = gene_selecionado
    else:
        uniprot_id = "N/D"
        nome_prot = gene_selecionado

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    with col_m1:
        st.metric("UniProt ID", uniprot_id)
    with col_m2:
        st.metric("Total de Variantes Missense", f"{len(df_gene):,}")
    with col_m3:
        n_pat = len(df_gene[df_gene["AlphaMissense_Class"] == "pathogenic"])
        st.metric("Patogênicas (AlphaMissense)", f"{n_pat:,}")
    with col_m4:
        n_hubs = len(df_gene[df_gene["RIN_Hub_Status"] != "Resíduo Regular"])
        st.metric("Variantes em Hubs 3D", f"{n_hubs:,}")
    with col_m5:
        n_vus_prio = len(df_gene[
            (df_gene["Status_Clinico_Consolidado"].str.contains("VUS|Sem Anotação", case=False, na=False)) &
            (df_gene["AlphaMissense_Score"] >= 0.564) &
            (df_gene["RIN_Hub_Status"] != "Resíduo Regular")
        ])
        st.metric("VUS Reclassificadas em Hubs", f"{n_vus_prio:,}", delta="Alta Prioridade", delta_color="inverse")

    st.markdown("---")

    # Aplicar Filtros da Sidebar
    df_gene_filt = df_gene.copy()
    if classes_am:
        df_gene_filt = df_gene_filt[df_gene_filt["AlphaMissense_Class"].isin(classes_am)]
    if min_cadd > 0:
        df_gene_filt = df_gene_filt[df_gene_filt["CADD_Phred"] >= min_cadd]
    if apenas_hubs:
        df_gene_filt = df_gene_filt[df_gene_filt["RIN_Hub_Status"] != "Resíduo Regular"]
    if apenas_vus:
        df_gene_filt = df_gene_filt[df_gene_filt["Status_Clinico_Consolidado"].str.contains("VUS|Sem Anotação", case=False, na=False)]

    # ABAS DO GENE: ANÁLISE GRÁFICA + ESTRUTURA 3D
    tab_graficos, tab_3d, tab_tabela = st.tabs(["📊 Gráficos Interativos (Plotly)", "🧊 Visualizador 3D da Proteína (AlphaFold)", "📋 Catálogo Tabular de Variantes"])

    with tab_graficos:
        col_g1, col_g2 = st.columns([1.6, 1.0])

        with col_g1:
            st.subheader("📈 Perfil Topológico de Resíduos ao Longo da Estrutura")
            eixo_y_topologia = st.selectbox(
                "Selecione a Métrica Topológica (Eixo Y):",
                ["RIN_Betweenness", "RIN_Degree", "RIN_Closeness", "AlphaFold_pLDDT"],
                format_func=lambda x: {
                    "RIN_Betweenness": "Centralidade de Intermediação (Betweenness)",
                    "RIN_Degree": "Grau de Conectividade Local (Degree)",
                    "RIN_Closeness": "Acessibilidade Global (Closeness)",
                    "AlphaFold_pLDDT": "Confiança Estrutural AlphaFold (pLDDT)"
                }.get(x, x)
            )

            fig_scatter = px.scatter(
                df_gene_filt.dropna(subset=["Posicao_AA_Num", eixo_y_topologia]),
                x="Posicao_AA_Num",
                y=eixo_y_topologia,
                color="AlphaMissense_Score",
                color_continuous_scale="YlOrRd",
                range_color=[0.0, 1.0],
                size="CADD_Plot_Size",
                size_max=12,
                hover_name="Protein_Variant_AM",
                hover_data={
                    "Posicao_AA_Num": True,
                    eixo_y_topologia: ":.4f",
                    "AlphaMissense_Score": ":.4f",
                    "CADD_Phred": ":.1f",
                    "rsID": True,
                    "RIN_Hub_Status": True,
                    "Status_Clinico_Consolidado": True,
                    "CADD_Plot_Size": False
                },
                labels={
                    "Posicao_AA_Num": "Posição do Resíduo de Aminoácido na Proteína",
                    eixo_y_topologia: eixo_y_topologia.replace("_", " "),
                    "AlphaMissense_Score": "Escore AlphaMissense"
                },
                title=f"Dispersão Posicional das Variantes em {gene_selecionado} (Tamanho = CADD, Cor = AlphaMissense)"
            )
            fig_scatter.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20), coloraxis_colorbar=dict(title="AlphaMissense"))
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_g2:
            st.subheader("🎯 Distribuição de Classes & Impacto")
            cont_classes = df_gene_filt["AlphaMissense_Class"].value_counts().reset_index()
            cont_classes.columns = ["Classe", "Variantes"]
            
            mapa_nome = {"pathogenic": "Provavelmente Patogênica", "ambiguous": "Ambígua / Incerta", "benign": "Provavelmente Benigna"}
            cont_classes["Classe_Nome"] = cont_classes["Classe"].map(mapa_nome)
            
            fig_pie = px.pie(
                cont_classes,
                names="Classe_Nome",
                values="Variantes",
                color="Classe",
                color_discrete_map=PALETA_CORES,
                hole=0.45,
                title=f"Proporção de Classes AlphaMissense ({len(df_gene_filt):,} variantes)"
            )
            fig_pie.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=-0.2))
            st.plotly_chart(fig_pie, use_container_width=True)

    with tab_3d:
        st.subheader(f"🧊 Estrutura Tridimensional AlphaFold — {gene_selecionado} ({uniprot_id})")
        st.markdown("Interaja diretamente com a estrutura 3D da proteína: rotacione arrastando o mouse, amplie com o scroll e destaque posições mutadas específicas.")

        col_3d_ctrl1, col_3d_ctrl2, col_3d_ctrl3 = st.columns([1, 1, 1])
        with col_3d_ctrl1:
            esquema_cor = st.selectbox(
                "Esquema de Cores 3D:",
                ["spectrum", "pLDDT (AlphaFold)", "cyan"],
                format_func=lambda x: {
                    "spectrum": "Espectro N-to-C (Arco-Íris)",
                    "pLDDT (AlphaFold)": "Confiança Estrutural AlphaFold (pLDDT)",
                    "cyan": "Monocromático Azul"
                }.get(x, x)
            )
        with col_3d_ctrl2:
            # Lista de posições mutadas para destacar
            posicoes_mutadas = [0] + sorted([int(p) for p in df_gene["Posicao_AA_Num"].dropna().unique() if p > 0])
            res_destaque = st.selectbox(
                "Destacar Resíduo / Sítio Mutado em 3D:",
                posicoes_mutadas,
                format_func=lambda x: "Nenhum (Visão Geral)" if x == 0 else f"Resíduo {x} ({df_gene[df_gene['Posicao_AA_Num'] == x]['Protein_Variant_AM'].values[0] if len(df_gene[df_gene['Posicao_AA_Num'] == x]) > 0 else ''})"
            )
        with col_3d_ctrl3:
            if esquema_cor == "pLDDT (AlphaFold)":
                st.markdown("""
                **Legenda pLDDT:**
                - 🟦 Azul Escuro: Muito Alta (>90)
                - 🩵 Azul Claro: Alta (70-90)
                - 🟨 Amarelo: Baixa (50-70)
                - 🟧 Laranja: Muito Baixa (<50)
                """)

        pdb_conteudo = carregar_pdb_texto(uniprot_id)
        if pdb_conteudo:
            html_viewer = gerar_visualizador_3d_html(
                pdb_conteudo,
                coloracao=esquema_cor,
                res_destaque=res_destaque
            )
            components.html(html_viewer, height=500)
        else:
            st.warning(f"Arquivo PDB para o UniProt {uniprot_id} não encontrado no cache local.")

    with tab_tabela:
        st.subheader(f"📋 Tabela de Variantes Missense de {gene_selecionado} ({len(df_gene_filt):,} registros filtrados)")
        
        colunas_visiveis = [
            "rsID", "Protein_Variant_AM", "Posicao_AA_Num", "AlphaMissense_Score", "AlphaMissense_Class",
            "CADD_Phred", "REVEL_Score", "gnomAD_AF", "RIN_Betweenness", "RIN_Degree", "RIN_Hub_Status", "Status_Clinico_Consolidado"
        ]
        cols_existentes = [c for c in colunas_visiveis if c in df_gene_filt.columns]
        
        st.dataframe(
            df_gene_filt[cols_existentes].sort_values("AlphaMissense_Score", ascending=False),
            use_container_width=True,
            height=320
        )

        csv_bytes = df_gene_filt[cols_existentes].to_csv(sep="\t", index=False).encode('utf-8')
        st.download_button(
            label=f"⬇️ Exportar Variantes de {gene_selecionado} (.tsv)",
            data=csv_bytes,
            file_name=f"variantes_{gene_selecionado}_alphamissense_rins.tsv",
            mime="text/tab-separated-values"
        )


# ==============================================================================
# MÓDULO 2: VISÃO GLOBAL DO GENOMA COMPARTILHADO
# ==============================================================================
elif modo_visao == "Visão Global do Genoma":
    st.markdown('<div class="main-header">Visão Global dos 428 Genes Compartilhados (PsA & AS)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Métricas consolidadas de 313.631 variantes missense, 4,1M predições do AlphaMissense e 211.710 resíduos 3D modelados</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Genes Compartilhados", f"{df_master['Gene'].nunique():,}")
    with c2:
        st.metric("Variantes Missense Mapeadas", f"{len(df_master):,}")
    with c3:
        st.metric("Variantes Patogênicas (AM ≥ 0.564)", f"{len(df_master[df_master['AlphaMissense_Score'] >= 0.564]):,}")
    with c4:
        st.metric("Odds Ratio em Hubs Estruturais", "2,47", delta="Fisher p < 10⁻³⁰⁰")

    st.markdown("---")
    col_vg1, col_vg2 = st.columns(2)

    with col_vg1:
        st.subheader("🏆 Top Genes com Maior Acúmulo de Variantes Deletérias")
        top_n = st.slider("Quantidade de genes a exibir:", 5, 30, 15)
        top_pat = df_master[df_master["AlphaMissense_Score"] >= 0.564]["Gene"].value_counts().head(top_n).reset_index()
        top_pat.columns = ["Gene", "Variantes_Patogenicas"]

        fig_bar = px.bar(
            top_pat,
            x="Variantes_Patogenicas",
            y="Gene",
            orientation="h",
            color="Variantes_Patogenicas",
            color_continuous_scale="Reds",
            title=f"Top {top_n} Genes Compartilhados Mais Mutados (AlphaMissense Patogênico)"
        )
        fig_bar.update_layout(yaxis=dict(autorange="reversed"), height=520)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_vg2:
        st.subheader("📊 Concordância entre AlphaMissense e CADD Phred")
        fig_box = px.box(
            df_master.dropna(subset=["AlphaMissense_Class", "CADD_Phred"]).sample(min(15000, len(df_master))),
            x="AlphaMissense_Class",
            y="CADD_Phred",
            color="AlphaMissense_Class",
            color_discrete_map=PALETA_CORES,
            category_orders={"AlphaMissense_Class": ["benign", "ambiguous", "pathogenic"]},
            labels={"AlphaMissense_Class": "Classificação AlphaMissense", "CADD_Phred": "Escore CADD Phred"},
            title="Distribuição CADD Phred por Categoria AlphaMissense (Amostra n=15.000)"
        )
        fig_box.add_hline(y=20, line_dash="dash", line_color="#D55E00", annotation_text="Corte CADD ≥ 20 (Top 1%)")
        fig_box.update_layout(height=520, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)


# ==============================================================================
# MÓDULO 3: ESTUDOS DE CASO (SMAD3, PLCG1, TYK2)
# ==============================================================================
elif modo_visao == "Estudos de Caso (SMAD3 / PLCG1)":
    st.markdown('<div class="main-header">Estudos de Caso Estruturais em Genes-Chave</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Investigação aprofundada da topologia de redes de contato e reclassificação de variantes incertas</div>', unsafe_allow_html=True)

    gene_caso = st.selectbox("Selecione o Estudo de Caso:", ["SMAD3", "PLCG1", "TYK2", "FOXF2"])
    df_caso = df_master[df_master["Gene"] == gene_caso].copy()
    match_caso = df_genes_cat[(df_genes_cat["Gene_Original"] == gene_caso) | (df_genes_cat["Gene_Query"] == gene_caso)] if not df_genes_cat.empty else pd.DataFrame()
    if not match_caso.empty and "UniProt_ID" in match_caso.columns:
        uniprot_caso = match_caso["UniProt_ID"].values[0]
    elif "UniProt_ID" in df_caso.columns and len(df_caso["UniProt_ID"].dropna()) > 0:
        uniprot_caso = df_caso["UniProt_ID"].dropna().values[0]
    else:
        uniprot_caso = ""

    col_cs1, col_cs2 = st.columns([1.1, 1.0])
    with col_cs1:
        if gene_caso == "SMAD3":
            st.info("🔹 **SMAD3 (Via TGF-β / Fibrose & Inflamação):** Transdutor de sinal que regula genes pró-inflamatórios. Possui os domínios **MH1 (9-137)** (ligação ao DNA) e **MH2 (231-425)** (oligomerização/transativação).")
        elif gene_caso == "PLCG1":
            st.info("🔹 **PLCG1 (Fosfolipase C Gama 1 / Sinalização Imune):** Mediador da ativação de linfócitos T/B. Possui os domínios **PH (26-140)**, **SH2/SH3 (547-755)** e o núcleo catalítico/C2 **(950-1290)**.")
        elif gene_caso == "TYK2":
            st.info("🔹 **TYK2 (Tirosina-quinase 2 / Eixo IL-12/IL-23):** Alvo terapêutico central nas espondiloartropatias.")
        elif gene_caso == "FOXF2":
            st.info("🔹 **FOXF2 (Forkhead Box F2):** Fator de transcrição com domínio Forkhead de ligação ao DNA altamente estruturado.")

        fig_caso = px.scatter(
            df_caso.dropna(subset=["Posicao_AA_Num", "RIN_Betweenness"]),
            x="Posicao_AA_Num",
            y="RIN_Betweenness",
            color="AlphaMissense_Score",
            color_continuous_scale="YlOrRd",
            range_color=[0.0, 1.0],
            size="CADD_Plot_Size",
            size_max=12,
            hover_name="Protein_Variant_AM",
            hover_data={"Posicao_AA_Num": True, "RIN_Betweenness": ":.4f", "AlphaMissense_Score": ":.4f", "CADD_Phred": ":.1f", "RIN_Hub_Status": True, "CADD_Plot_Size": False},
            title=f"Perfil de Betweenness em {gene_caso}"
        )
        fig_caso.update_layout(height=440)
        st.plotly_chart(fig_caso, use_container_width=True)

    with col_cs2:
        st.subheader(f"Visualização 3D — {gene_caso}")
        pdb_caso = carregar_pdb_texto(uniprot_caso)
        if pdb_caso:
            html_caso = gerar_visualizador_3d_html(pdb_caso, coloracao="pLDDT (AlphaFold)")
            components.html(html_caso, height=440)

    st.subheader(f"Variantes Reclassificadas de Alta Prioridade em Hubs de {gene_caso}")
    df_vus_caso = df_caso[
        (df_caso["Status_Clinico_Consolidado"].str.contains("VUS|Sem Anotação", case=False, na=False)) &
        (df_caso["AlphaMissense_Score"] >= 0.80) &
        (df_caso["RIN_Hub_Status"] != "Resíduo Regular")
    ]
    st.dataframe(
        df_vus_caso[["Protein_Variant_AM", "Posicao_AA_Num", "AlphaMissense_Score", "CADD_Phred", "RIN_Betweenness", "RIN_Degree", "RIN_Hub_Status"]].sort_values("RIN_Betweenness", ascending=False),
        use_container_width=True
    )


# ==============================================================================
# MÓDULO 4: RELATÓRIO ESTATÍSTICO (KRUSKAL-WALLIS & DUNN)
# ==============================================================================
elif modo_visao == "Relatório Estatístico":
    st.markdown('<div class="main-header">Validação Estatística Formal (Kruskal-Wallis & Dunn Post-Hoc)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Demonstração estatística de que as propriedades topológicas das proteínas governam o impacto deletério das variantes</div>', unsafe_allow_html=True)

    if os.path.exists(CAMINHO_STAT):
        df_stat = pd.read_csv(CAMINHO_STAT, sep="\t")
        st.dataframe(df_stat, use_container_width=True)
    
    st.markdown("""
    ### 🔬 Conclusões Científicas da Análise Não-Paramétrica:
    1. **Kruskal-Wallis Global:** Confirmou que *Betweenness*, *Degree*, *Closeness* e *pLDDT* possuem distribuições significativamente distintas entre classes ($p < 10^{-300}$).
    2. **Teste Post-Hoc de Dunn:** As variantes **Ambíguas / Incertas** ocupam uma posição **estatisticamente intermediária** de conectividade física ($p < 10^{-300}$), provando que a indefinição algorítmica reflete resíduos em zonas de transição estrutural.
    3. **Enriquecimento em Hubs:** Odds Ratio de **2,47** ($p < 10^{-300}$), indicando que mutações patogênicas recaem com probabilidade 2,5 vezes maior sobre os nós centrais das redes de contato 3D.
    """)
