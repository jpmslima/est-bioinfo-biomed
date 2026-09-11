# REMAR Genes: Plataforma Integrada de Bioinformática Estrutural & Genômica
### Espondiloartrites: Artrite Psoriásica (PsA) & Espondilite Anquilosante (AS)

Repositório contendo o pipeline automatizado, análise estatística de redes de interação de resíduos (RINs) em estruturas tridimensionais (AlphaFold DB) e a aplicação interativa **REMAR Genes**, desenvolvida para a coorte de **428 genes humanos compartilhados** entre a **Artrite Psoriásica (PsA)** e a **Espondilite Anquilosante (AS)**.

O projeto é estruturado sob a **Arquitetura Medalhão de Dados (Bronze -> Silver -> Gold)**, integrando predições de inteligência artificial de saturação mutagênica (*AlphaMissense, Science 2023*), evidências consensuais de patogenicidade computacional (*ClinGen SVI, AJHG 2022*), topologia de redes alostéricas tridimensionais e modelagem conformacional homóloga com validação por ângulos diedrais de Ramachandran.

---

## 🏛️ Arquitetura de Dados (Medalhão) e Estrutura do Repositório

```
estagio_2026.2/
│
├── 🥉 data/bronze/                           # DADOS BRUTOS & ARQUIVOS DE ORIGEM (Raw Data)
│   ├── genes_input_raw.tsv                  # Lista inicial de genes da coorte clínica
│   ├── alphamissense_raw_stream.tsv         # Extração bruta de predições do catálogo AlphaMissense
│   ├── alphafold_pdb_cache/                 # Molde selvagem AlphaFold DB em formato PDB (.pdb)
│   └── uniprot_cache/                       # Respostas JSON brutas em cache da REST API UniProtKB
│
├── 🥈 data/silver/                           # DADOS LIMPOS, HIGIENIZADOS & INTEGRADOS (Cleaned & Conformed)
│   ├── genes_uniprot_curados.tsv            # 428 genes canônicos mapeados (Homo sapiens, Taxon ID 9606)
│   ├── variantes_missense_dbsnp.tsv         # Variantes missense populacionais padronizadas
│   ├── alphamissense_genes_psa_as.tsv       # Predições estruturadas AlphaMissense para a coorte
│   ├── topologia_rins_residuos.tsv          # 211.710 resíduos com Degree, Betweenness, Closeness e pLDDT
│   └── base_mestre_integrada.tsv            # Matriz relacional consolidada (313.631 variantes missense)
│
├── 🥇 data/gold/                             # TABELAS ANALÍTICAS & VALIDAÇÃO ESTATÍSTICA (Gold Insights)
│   ├── variantes_prioritarias_deleterias.tsv# Variantes prioritárias (AlphaMissense >= 0,564 e CADD >= 25,3)
│   ├── vus_reclassificadas_hubs.tsv         # Variantes de significado incerto (VUS) situadas em Hubs 3D
│   └── relatorio_kruskal_dunn.tsv           # Resultados do teste de Kruskal-Wallis e post-hoc de Dunn
│
├── 📁 src/                                   # PIPELINE MODULAR DE PROCESSAMENTO
│   ├── 01_bronze/                           # Extração e coleta de dados (UniProtKB, dbSNP, AlphaMissense)
│   │   ├── buscar_uniprot.py
│   │   ├── buscar_dbsnp_missense_integrado.py
│   │   └── extrair_alphamissense_genes.py
│   ├── 02_silver/                           # Cruzamento relacional, padronização e grafos de resíduos
│   │   ├── integrar_alphamissense_dbsnp.py
│   │   └── calcular_rins_topologia.py
│   ├── 03_gold/                             # Testes estatísticos de hipótese e bioestatística formal
│   │   └── analise_estatistica_rins_alphamissense.py
│   └── visualizacao/                        # Scripts para renderização de figuras em alta resolução
│       ├── gerar_figuras_variantes.py
│       └── gerar_figuras_alphamissense.py
│
├── 🧬 annotator.py                           # Módulo de anotações funcionais UniProtKB e predição DTU/SignalP
├── 📐 modeling.py                            # Módulo de modelagem Swiss-Model e cálculo diédrico de Ramachandran
├── 🚀 app_remar.py                           # Aplicação Web Interativa Principal (Streamlit)
└── 📄 README.md                              # Documentação técnica do repositório
```

---

## 🚀 Como Executar a Aplicação Web Interativa (Streamlit)

A aplicação **REMAR Genes** é executada via Streamlit e oferece um painel interativo com renderização molecular 3D em tempo real, gráficos dinâmicos de alta precisão e filtragem clínica sob demanda.

### 1. Pré-requisitos e Ambiente:
Recomenda-se ambiente Python 3.10 ou superior com os pacotes instalados:
```bash
pip install streamlit pandas numpy plotly requests biopython scipy
```

### 2. Inicialização da Aplicação:
Na raiz do repositório, execute:
```bash
streamlit run app_remar.py
```

---

## 🔬 Visão Geral dos Recursos da Aplicação (`app_remar.py`)

A plataforma está organizada em uma interface dividida em **5 abas especializadas**:

### 🎯 Barra Lateral (Menu de Seleção e Filtros Clínicos)
* **Seletor de Gene Global:** Busca e seleção instantânea entre os **428 genes** da coorte (ex.: `IL17A`, `SMAD3`, `PLCG1`, `TYK2`, `JAK1`, `STAT3`, `PPARG`, `RUNX3`...).
* **Filtro CADD Phred (ClinGen):** Controle deslizante calibrado nas diretrizes internacionais (padrão em CADD >= 25,3 para suporte patogênico PP3_Supporting; e CADD >= 28,1 para PP3_Moderate).
* **Filtro REVEL (ClinGen):** Seleção de níveis de evidência de patogenicidade (Supporting >= 0,644; Moderate >= 0,773; Strong >= 0,932).
* **Isolamento de Hubs Estruturais:** Caixa de seleção para focar exclusivamente nos resíduos que compõem o Top 10% de intermediação topológica 3D.

### 📊 Painel de KPIs Executivos (Topo da Tela)
* **Gene / Alvo:** Símbolo oficial e UniProt ID sincronizados.
* **Variantes Filtradas:** Total de variantes missense após os cortes clínicos ativos.
* **Patogênicas (AM):** Quantidade de variantes classificadas como Provavelmente Patogênicas (AlphaMissense >= 0,564).
* **Consenso ClinGen:** Variantes com concordância estrita de alto impacto (AlphaMissense >= 0,564 + CADD >= 25,3).
* **Hubs Estruturais:** Mutações que atingem resíduos críticos de conectividade física ou alostérica.
* **Confiança pLDDT Médio:** Nível médio de acurácia estrutural do molde AlphaFold DB.

---

### 🧬 Detalhamento das 5 Abas Funcionais

#### 1. Perfil Funcional & Topologia 3D
* **Gráfico Lollipop (AlphaMissense x Posição):** Visualização interativa de saturação mutagênica ao longo da cadeia primária. Cada resíduo é posicionado com altura proporcional ao score AlphaMissense, tamanho proporcional ao CADD Phred e cores padronizadas por classe patogênica. O *hover* detalhado exibe posição, resíduo, código do dbSNP (`rsID`), scores e classe.
* **Perfil Contínuo de Topologia RINs:** Gráfico de linha contínua e área preenchida ordenado de 1 a N, exibindo *Betweenness Centrality* (eixos alostéricos), *RIN Degree* (conectividade local) ou *Closeness Centrality* (acessibilidade global), destacando os Hubs do Top 10% e sobrepondo as mutações candidatas.
* **Visualizador Estrutural Selvagem 3D (3Dmol.js):** Renderização tridimensional interativa da proteína a partir do AlphaFold DB, com rotação em 360°, zoom, foco e coloração gradiente pela métrica de confiança biofísica pLDDT.

#### 2. Anotações Funcionais (UniProtKB & DTU)
* **Predição de Endereçamento e Peptídeo Sinal (DTU/SignalP):** Avaliação de sequência de clivagem de sinal (SP) para secreção e predição de compartimentalização celular (Membrana, Núcleo, Citosol, Retículo, Mitocôndria).
* **Anotações Estruturais UniProtKB REST API:** Consulta automatizada trazendo descrição da proteína, domínios funcionais mapeados, resíduos de sítio ativo catalítico, pontes dissulfeto covalentes e sítios de glicosilação.

#### 3. Modelagem Homóloga de Mutantes e Avaliação Conformacional
* **Modelagem in silico do Mutante:** Seleção de qualquer variante patogênica para gerar a conformação pontual sobre o molde selvagem do AlphaFold DB (`{UniProtID}_mut_{Mutacao}.pdb`).
* **Visualizador 3D do Mutante:** Renderização molecular destacando o resíduo mutado em vermelho com esferas/bastões e focalização automática por câmera (`zoomTo`).
* **Integração Swiss-Model API:** Suporte para envio de jobs remotos com token de usuário.
* **Validação Biofísica de Ramachandran:** Gráfico de contorno com cálculo dos ângulos de torção diedrais Phi (φ) e Psi (ψ) do esqueleto polipeptídico, posicionando o resíduo mutado como uma estrela vermelha para validação de estabilidade estrutural (regiões de Hélice-Alfa e Folha-Beta favorecidas).
* **Download de PDB:** Botão para exportar diretamente o arquivo estrutural `.pdb` do mutante gerado.

#### 4. Tabela Curada de Variantes e Exportação
* **Tabela Completa de Variantes:** Exibição estruturada contendo: `HGVS_p`, `dbSNP_rsID`, `Posicao_AA`, `AA_Ref`, `AA_Alt`, `AlphaMissense_Score`, `Class`, `CADD_Phred`, `REVEL_Score`, `AlphaFold_pLDDT`, `RIN_Degree`, `RIN_Betweenness`, `RIN_Hub_Status`, `ClinVar_UniProt` e `Status_Clinico_Consolidado`.
* **Exportação TSV:** Download sob demanda da tabela filtrada do gene (`remar_genes_{Gene}_filtradas.tsv`).
* **Glossário Completo de Aminoácidos:** Tabela expansível de consulta com código de 1 letra, 3 letras, nome em português, nome em inglês e propriedades físico-químicas de todos os 20 aminoácidos canônicos.

#### 5. Rigor Metodológico e Respaldo Internacional
* **AlphaMissense (Google DeepMind, Science 2023):** Detalhamento dos limiares calibrados de 0,340 (Benigno) e 0,564 (Patogênico) com precisão populacional mínima de 90%.
* **ClinGen SVI Working Group (Pejaver et al., AJHG 2022):** Justificativa da obsolescência do ponto de corte clássico de CADD >= 20,0 e fundamentação dos critérios formais ACMG PP3 de CADD >= 25,3 (Supporting), CADD >= 28,1 (Moderate) e intervalos do REVEL.
* **Estatística de Redes:** Explicação dos testes de Kruskal-Wallis e Dunn que demonstram a correlação biológica entre centralidade tridimensional e patogenicidade (p < 10^-300).

---

## 🔬 Execução do Pipeline Modular (`src/`)

Caso seja necessário reprocessar ou reconstruir os dados brutos a partir das fontes públicas:

### 1. Camada Bronze -> Coleta e Pré-processamento
```bash
python3 src/01_bronze/buscar_uniprot.py
python3 src/01_bronze/buscar_dbsnp_missense_integrado.py
python3 src/01_bronze/extrair_alphamissense_genes.py
```

### 2. Camada Silver -> Integração Relacional e Topologia de Redes 3D
```bash
python3 src/02_silver/integrar_alphamissense_dbsnp.py
python3 src/02_silver/calcular_rins_topologia.py
```

### 3. Camada Gold -> Análise Estatística e Figuras de Alta Resolução
```bash
python3 src/03_gold/analise_estatistica_rins_alphamissense.py
python3 src/visualizacao/gerar_figuras_variantes.py
python3 src/visualizacao/gerar_figuras_alphamissense.py
```

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.10+
* **Framework Web:** Streamlit
* **Visualização de Dados:** Plotly Graph Objects, 3Dmol.js (WebGL molecular)
* **Manipulação de Dados:** Pandas, NumPy, PyArrow
* **Bioinformática e Estruturas:** Biopython, AlphaFold Database (EBI-EMBL), UniProtKB REST API, Swiss-Model API
* **Bioestatística:** SciPy, Statsmodels
