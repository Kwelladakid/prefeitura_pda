# app.py

import os
import io
import zipfile
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import google.generativeai as genai


# =========================
# HTML único (tudo integrado)
# =========================
def build_full_html_page(
    title: str,
    df_html_table: str,
    figs: list,
    ia_text: str,
    subtitle_dashboard="Dashboard Interativo",
    subtitle_table="Planilha Limpa (Tabela)",
    subtitle_ia="Relatório da IA"
) -> bytes:
    # Concatena gráficos Plotly como HTML + inclui tabela e texto da IA em seções
    fig_parts = []
    for i, f in enumerate(figs):
        fig_parts.append(f.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False)))

    css = """
    body{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#222}
    .container{max-width:1200px;margin:0 auto}
    h1{margin:0 0 12px 0;font-size:28px}
    h2{margin:18px 0 10px 0;font-size:22px}
    .box{border:1px solid #e5e5e5;border-radius:10px;padding:16px;margin:16px 0}
    .muted{color:#666;font-size:12px}
    table{border-collapse:collapse;width:100%;font-size:13px}
    th,td{border:1px solid #ddd;padding:8px}
    th{background:#f5f7fb;text-align:left}
    pre{white-space:pre-wrap;word-wrap:break-word}
    """

    html = f"""<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<div class="container">
  <h1>{title}</h1>
  <div class="muted">Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}</div>

  <div class="box">
    <h2>{subtitle_table}</h2>
    {df_html_table}
  </div>

  <div class="box">
    <h2>{subtitle_dashboard}</h2>
    {''.join(fig_parts) if fig_parts else '<div class="muted">Nenhum gráfico gerado.</div>'}
  </div>

  <div class="box">
    <h2>{subtitle_ia}</h2>
    {"<pre>"+ia_text+"</pre>" if ia_text else '<div class="muted">Nenhum texto gerado pela IA.</div>'}
  </div>
</div>
</body>
</html>"""
    return html.encode("utf-8")


def build_dashboard_html(figs, title="Dashboard de Despesas"):
    parts = []
    for i, f in enumerate(figs):
        parts.append(f.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False)))
    html = f"""<html><head><meta charset="utf-8"><title>{title}</title></head>
    <body><h2 style="font-family:Arial;margin:16px 0;">{title}</h2>{''.join(parts)}</body></html>"""
    return html.encode("utf-8")


# =========================
# Configuração da página
# =========================
st.set_page_config(page_title="Analista de Despesas - Prefeitura", layout="wide")
st.title("🏛️ Analista de Despesas (Download do HTML Integrado)")
st.markdown("---")

# =========================
# Barra lateral
# =========================
st.sidebar.header("⚙️ Configurações")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
st.sidebar.caption("Dica: exporte GEMINI_API_KEY no seu ~/.zshrc para preencher automaticamente.")

# Estado para IA e gráficos
if "ia_text" not in st.session_state:
    st.session_state["ia_text"] = ""
if "export_figs" not in st.session_state:
    st.session_state["export_figs"] = []
if "dashboard_ready" not in st.session_state:
    st.session_state["dashboard_ready"] = False
if "full_html_bytes" not in st.session_state:
    st.session_state["full_html_bytes"] = None

# =========================
# Upload de arquivo
# =========================
st.subheader("📂 Carregar Planilha")
uploaded_file = st.file_uploader("Arraste seu arquivo CSV/XLSX/XLS aqui", type=["csv", "xlsx", "xls"])

if uploaded_file:
    # Leitura do arquivo
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        elif name.endswith(".xlsx"):
            df_raw = pd.read_excel(uploaded_file, engine="openpyxl")
        elif name.endswith(".xls"):
            df_raw = pd.read_excel(uploaded_file, engine="xlrd")
        else:
            raise ValueError("Formato não suportado. Envie CSV, XLSX ou XLS.")
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        st.stop()

    # =========================
    # Limpeza e tipagem
    # =========================
    df_clean = df_raw.copy()

    # Padroniza nomes de colunas
    df_clean.columns = (
        df_clean.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
    )

    # Remove linhas/colunas totalmente vazias e duplicatas
    df_clean = df_clean.dropna(how="all").dropna(axis=1, how="all").drop_duplicates()

    # DataFrame para cálculos/gráficos (mantém numéricos)
    df_calculo = df_clean.copy()
    for col in df_calculo.columns:
        if any(k in col for k in ["valor", "pago", "total", "quantidade", "preco", "custo", "despesa"]):
            df_calculo[col] = pd.to_numeric(df_calculo[col], errors="coerce").fillna(0)
        if any(k in col for k in ["data", "vencimento", "emissao"]):
            df_calculo[col] = pd.to_datetime(df_calculo[col], errors="coerce")

    # DataFrame para exibição segura (como HTML)
    df_display = df_calculo.copy()
    for col in df_display.columns:
        if not (pd.api.types.is_numeric_dtype(df_display[col]) or pd.api.types.is_datetime64_any_dtype(df_display[col])):
            df_display[col] = df_display[col].astype(str).replace("nan", "")

    # Geração de bytes para Excel/CSV
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
        df_clean.to_excel(writer, index=False, sheet_name="dados_limpos")
    excel_bytes = excel_buf.getvalue()
    csv_bytes = df_clean.to_csv(index=False).encode("utf-8-sig")

    # =========================
    # Abas principais
    # =========================
    tab_dash, tab_dados, tab_ia, tab_down = st.tabs(
        ["📊 Dashboard", "📋 Dados Limpos", "🤖 IA (Gemini)", "⬇️ Downloads"]
    )

    # -------- Dashboard --------
    with tab_dash:
        st.subheader("Análise Visual")
        cols_num = df_calculo.select_dtypes(include=["number"]).columns.tolist()
        cols_date = df_calculo.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
        cols_txt = [c for c in df_calculo.columns if c not in cols_num and c not in cols_date]
        export_figs = []

        if cols_num and cols_txt:
            c1, c2 = st.columns(2)
            with c1:
                eixo_x = st.selectbox("Categoria (texto):", cols_txt, key="x_dash")
                eixo_y = st.selectbox("Valor (numérico):", cols_num, key="y_dash")

                df_grouped = (
                    df_calculo.groupby(eixo_x)[eixo_y]
                    .sum()
                    .reset_index()
                    .sort_values(eixo_y, ascending=False)
                    .head(15)
                )
                fig_bar = px.bar(
                    df_grouped, x=eixo_x, y=eixo_y, color=eixo_x,
                    title=f"Top 15 por {eixo_x}", template="plotly_white"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                export_figs.append(fig_bar)

            with c2:
                fig_pie = px.pie(
                    df_grouped, names=eixo_x, values=eixo_y,
                    title=f"Distribuição de {eixo_y}", hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                export_figs.append(fig_pie)

            if cols_date:
                st.markdown("---")
                st.markdown("**Evolução dos Gastos no Tempo**")
                data_col = st.selectbox("Coluna de data:", cols_date, key="date_col")
                df_time = df_calculo.dropna(subset=[data_col]).copy()
                if not df_time.empty:
                    df_time = df_time.groupby(data_col)[eixo_y].sum().reset_index().sort_values(data_col)
                    fig_line = px.line(
                        df_time, x=data_col, y=eixo_y,
                        title="Linha do Tempo de Despesas", template="plotly_white"
                    )
                    st.plotly_chart(fig_line, use_container_width=True)
                    export_figs.append(fig_line)

            st.markdown("---")
            with st.expander("Gráfico Radar (opcional)"):
                if len(cols_num) >= 1 and cols_txt:
                    dim_col = st.selectbox("Dimensão (texto) para o Radar:", cols_txt, key="radar_dim")
                    val_cols = st.multiselect("Métricas numéricas para o Radar (2-6):", cols_num, default=cols_num[: min(3, len(cols_num))])
                    top_n = st.slider("Top N categorias por soma da 1ª métrica:", min_value=3, max_value=20, value=6, step=1)

                    if len(val_cols) >= 1:
                        base = (
                            df_calculo.groupby(dim_col)[val_cols]
                            .sum()
                            .sort_values(val_cols[0], ascending=False)
                            .head(top_n)
                            .reset_index()
                        )
                        base_norm = base.copy()
                        for c in val_cols:
                            maxv = base_norm[c].max() or 1
                            base_norm[c] = base_norm[c] / maxv

                        fig_radar = go.Figure()
                        categorias = val_cols
                        for _, row in base_norm.iterrows():
                            fig_radar.add_trace(go.Scatterpolar(
                                r=[row[c] for c in categorias] + [row[categorias[0]]],
                                theta=categorias + [categorias[0]],
                                fill='toself',
                                name=str(row[dim_col])
                            ))
                        fig_radar.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                            showlegend=True,
                            title=f"Radar normalizado de métricas por {dim_col}",
                            template="plotly_white",
                            margin=dict(l=20, r=20, t=60, b=20)
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)
                        export_figs.append(fig_radar)
                else:
                    st.info("Carregue dados com ao menos 1 coluna numérica e 1 categórica para ver o Radar.")
        else:
            st.warning("Não foi possível identificar colunas numéricas e categóricas para o dashboard.")

        # Guarda gráficos no estado
        st.session_state["export_figs"] = export_figs
        st.session_state["dashboard_ready"] = bool(export_figs)

    # -------- Dados Limpos --------
    with tab_dados:
        st.subheader("Visualização dos Dados")
        st.write(f"Total de registros: {len(df_display)}")
        st.dataframe(df_display, use_container_width=True)

    # -------- IA (Gemini) --------
    with tab_ia:
        st.subheader("Relatório da IA (Gerar dentro do App)")
        pergunta = st.text_area("Pergunta à IA (ex: Principais fornecedores? Gastos atípicos? Tendências?)", height=80)

        col_ia1, col_ia2 = st.columns(2)
        with col_ia1:
            if st.button("Gerar análise com IA"):
                if not gemini_key:
                    st.error("Insira a Gemini API Key na barra lateral.")
                elif not pergunta:
                    st.warning("Digite uma pergunta para a IA.")
                else:
                    try:
                        genai.configure(api_key=gemini_key)
                        modelos = list(genai.list_models())
                        modelos_validos = [
                            m.name for m in modelos
                            if hasattr(m, "supported_generation_methods")
                            and "generateContent" in m.supported_generation_methods
                        ]
                        prefer = ["models/gemini-1.5-flash", "models/gemini-1.5-pro", "models/gemini-pro"]
                        modelo_escolhido = next((p for p in prefer if p in modelos_validos), (modelos_validos[0] if modelos_validos else None))

                        if not modelo_escolhido:
                            st.error("Nenhum modelo compatível encontrado para sua chave.")
                        else:
                            model = genai.GenerativeModel(modelo_escolhido)
                            resumo = df_calculo.describe(include="all").astype(str).to_string()
                            amostra = df_display.head(15).to_string(index=False)

                            prompt = (
                                "Você é um auditor de despesas públicas. Responda de forma técnica, objetiva e acionável.\n\n"
                                f"PERGUNTA: {pergunta}\n\n"
                                f"RESUMO ESTATÍSTICO (pandas.describe):\n{resumo}\n\n"
                                f"AMOSTRA DE DADOS (15 linhas):\n{amostra}\n"
                            )

                            with st.spinner("IA analisando..."):
                                resp = model.generate_content(prompt)
                                text = getattr(resp, "text", "").strip()
                                st.session_state["ia_text"] = text or "Sem resposta da IA."
                                st.success("Relatório da IA gerado e salvo para o HTML integrado.")
                    except Exception as e:
                        st.error(f"Erro com a IA: {e}")
                        st.info("Verifique sua chave e permissões no Google AI Studio.")

        with col_ia2:
            if st.button("Limpar relatório da IA"):
                st.session_state["ia_text"] = ""
                st.info("Relatório da IA limpo.")

        if st.session_state.get("ia_text"):
            st.markdown("### Prévia do relatório da IA (texto)")
            st.write(st.session_state["ia_text"])

    # -------- Downloads --------
    with tab_down:
        st.subheader("Exportações e Geração do HTML Integrado")

        colA, colB = st.columns(2)
        with colA:
            st.download_button(
                "📥 Excel (dados limpos)",
                data=excel_bytes,
                file_name="despesas_processadas.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with colB:
            st.download_button(
                "📥 CSV (dados limpos)",
                data=csv_bytes,
                file_name="despesas_processadas.csv",
                mime="text/csv",
            )

        st.markdown("---")
        # HTML do dashboard (opcional)
        export_figs = st.session_state.get("export_figs", [])
        dashboard_ready = st.session_state.get("dashboard_ready", False)
        html_dash_bytes = None
        if dashboard_ready and export_figs:
            html_dash_bytes = build_dashboard_html(export_figs, title="Dashboard de Despesas")
            st.download_button(
                "⬇️ Dashboard (HTML apenas gráficos)",
                data=html_dash_bytes,
                file_name="dashboard_despesas.html",
                mime="text/html",
            )
        else:
            st.caption("Gere o dashboard na aba '📊 Dashboard' para baixar o HTML dos gráficos (opcional).")

        st.markdown("---")
        # HTML completo (index.html)
        limit_rows = st.slider("Linhas da tabela no HTML Integrado (0 = todas)", 0, 1000, 200, step=50)
        df_for_table = df_display.head(limit_rows) if limit_rows > 0 else df_display
        table_html = df_for_table.to_html(index=False, escape=False)
        ia_text = st.session_state.get("ia_text", "")

        # Gera o HTML integrado agora (sem pré-visualização)
        full_html_bytes = build_full_html_page(
            title="Portal de Despesas - Página Completa",
            df_html_table=table_html,
            figs=export_figs,
            ia_text=ia_text
        )

        st.download_button(
            "🧾 Baixar Página Completa (index.html)",
            data=full_html_bytes,
            file_name="index.html",
            mime="text/html",
        )

        # ZIP com index.html + planilhas
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("index.html", full_html_bytes)
            zf.writestr("despesas_processadas.xlsx", excel_bytes)
            zf.writestr("despesas_processadas.csv", csv_bytes)
        zip_buf.seek(0)
        st.download_button(
            "📦 Baixar Tudo (ZIP: index.html + planilhas)",
            data=zip_buf.getvalue(),
            file_name="pacote_publicacao.zip",
            mime="application/zip",
        )
else:
    st.info("💡 Faça upload de um arquivo para iniciar.")
