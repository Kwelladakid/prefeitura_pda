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

# NOVO: Matplotlib/Seaborn para gerar imagens estáticas p/ PDF (sem Chrome/Kaleido)
import matplotlib
matplotlib.use("Agg")  # backend sem UI, compatível com cloud
import matplotlib.pyplot as plt
import seaborn as sns

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader


# =========================
# Utilitários HTML/PDF
# =========================
def build_dashboard_html(figs, title="Dashboard de Despesas"):
    parts = []
    for i, f in enumerate(figs):
        parts.append(f.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False)))
    html = f"""<html><head><meta charset="utf-8"><title>{title}</title></head>
    <body><h2 style="font-family:Arial;margin:16px 0;">{title}</h2>{''.join(parts)}</body></html>"""
    return html.encode("utf-8")


def build_pdf_report(titulo, resumo_texto, figuras_png_bytes, author="Analista IA"):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    # Cabeçalho
    c.setTitle(titulo)
    c.setAuthor(author)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height - 50, titulo)
    c.setFont("Helvetica", 10)
    c.drawString(40, height - 68, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

    # Resumo
    text_obj = c.beginText(40, height - 100)
    text_obj.setFont("Helvetica", 11)
    for line in resumo_texto.splitlines():
        for chunk in [line[i:i+110] for i in range(0, len(line), 110)]:
            text_obj.textLine(chunk)
    c.drawText(text_obj)

    # Figuras (2 por página, se necessário)
    y = height - 300
    for i, png_bytes in enumerate(figuras_png_bytes):
        img = ImageReader(io.BytesIO(png_bytes))
        img_w, img_h = 500, 300
        c.drawImage(img, 40, y - img_h, width=img_w, height=img_h, preserveAspectRatio=True, mask='auto')
        y -= (img_h + 30)
        if y < 160 and i < len(figuras_png_bytes) - 1:
            c.showPage()
            c.setFont("Helvetica-Bold", 14)
            c.drawString(40, height - 50, titulo)
            c.setFont("Helvetica", 10)
            c.drawString(40, height - 68, f"Página gerada: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
            y = height - 120

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()


# =========================
# NOVO: Gráficos estáticos com Matplotlib para PDF
# =========================
def fig_to_png_bytes(plt_fig, tight=True, dpi=150):
    img_buf = io.BytesIO()
    if tight:
        plt_fig.tight_layout()
    plt_fig.savefig(img_buf, format="png", dpi=dpi)
    plt.close(plt_fig)
    img_buf.seek(0)
    return img_buf.getvalue()


def make_bar_png(df_grouped, cat_col, val_col, title):
    plt_fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_grouped, x=cat_col, y=val_col, ax=ax, palette="Blues_d")
    ax.set_title(title)
    ax.set_xlabel(cat_col)
    ax.set_ylabel(val_col)
    ax.tick_params(axis='x', rotation=45, ha='right')
    return fig_to_png_bytes(plt_fig)


def make_pie_png(df_grouped, cat_col, val_col, title):
    plt_fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(df_grouped[val_col], labels=df_grouped[cat_col], autopct='%1.1f%%', startangle=140)
    ax.set_title(title)
    return fig_to_png_bytes(plt_fig)


def make_line_png(df_time, date_col, val_col, title):
    plt_fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_time[date_col], df_time[val_col], marker='o', linewidth=2, color="#155DE9")
    ax.set_title(title)
    ax.set_xlabel("Data")
    ax.set_ylabel(val_col)
    plt.xticks(rotation=30, ha='right')
    plt.grid(alpha=0.3)
    return fig_to_png_bytes(plt_fig)


def make_radar_png(base_norm, dim_col, metrics, title):
    # Radar simples com Matplotlib (valores normalizados 0-1)
    labels = metrics
    num_labels = len(labels)
    angles = [n / float(num_labels) * 2 * 3.14159265 for n in range(num_labels)]
    angles += angles[:1]  # fecha o círculo

    plt_fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(3.14159265 / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], labels)
    ax.set_rlabel_position(0)
    ax.set_title(title, y=1.1)

    for _, row in base_norm.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=str(row[dim_col]))
        ax.fill(angles, values, alpha=0.1)

    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    return fig_to_png_bytes(plt_fig)


# =========================
# Config da página
# =========================
st.set_page_config(page_title="Analista de Despesas - Prefeitura", layout="wide")
st.title("🏛️ Analista de Despesas (Limpeza, Dashboard, IA)")
st.markdown("---")

# =========================
# Barra lateral
# =========================
st.sidebar.header("⚙️ Configurações")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
st.sidebar.caption("Dica: exporte GEMINI_API_KEY no seu ~/.zshrc para preencher automaticamente.")

# =========================
# Upload
# =========================
st.subheader("📂 Carregar Planilha")
uploaded_file = st.file_uploader("Arraste seu arquivo CSV/XLSX/XLS aqui", type=["csv", "xlsx", "xls"])

if uploaded_file:
    # Leitura
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

    # Limpeza e tipagem
    df_clean = df_raw.copy()
    df_clean.columns = (
        df_clean.columns.astype(str)
        .str.strip().str.lower().str.replace(" ", "_")
        .str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("utf-8")
    )
    df_clean = df_clean.dropna(how="all").dropna(axis=1, how="all").drop_duplicates()

    df_calculo = df_clean.copy()
    for col in df_calculo.columns:
        if any(k in col for k in ["valor", "pago", "total", "quantidade", "preco", "custo", "despesa"]):
            df_calculo[col] = pd.to_numeric(df_calculo[col], errors="coerce").fillna(0)
        if any(k in col for k in ["data", "vencimento", "emissao"]):
            df_calculo[col] = pd.to_datetime(df_calculo[col], errors="coerce")

    df_display = df_calculo.copy()
    for col in df_display.columns:
        if not (pd.api.types.is_numeric_dtype(df_display[col]) or pd.api.types.is_datetime64_any_dtype(df_display[col])):
            df_display[col] = df_display[col].astype(str).replace("nan", "")

    # Arquivos base
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
        df_clean.to_excel(writer, index=False, sheet_name="dados_limpos")
    excel_bytes = excel_buf.getvalue()
    csv_bytes = df_clean.to_csv(index=False).encode("utf-8-sig")

    # Abas
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
        pdf_ctx = {}  # NOVO: contexto para gerar PNGs com Matplotlib

        if cols_num and cols_txt:
            c1, c2 = st.columns(2)
            with c1:
                eixo_x = st.selectbox("Categoria (texto):", cols_txt, key="x_dash")
                eixo_y = st.selectbox("Valor (numérico):", cols_num, key="y_dash")

                df_grouped = (
                    df_calculo.groupby(eixo_x)[eixo_y]
                    .sum().reset_index().sort_values(eixo_y, ascending=False).head(15)
                )
                fig_bar = px.bar(df_grouped, x=eixo_x, y=eixo_y, color=eixo_x,
                                 title=f"Top 15 por {eixo_x}", template="plotly_white")
                st.plotly_chart(fig_bar, use_container_width=True)
                export_figs.append(fig_bar)
                # Para PDF
                pdf_ctx["bar"] = (df_grouped, eixo_x, eixo_y, f"Top 15 por {eixo_x}")

            with c2:
                fig_pie = px.pie(df_grouped, names=eixo_x, values=eixo_y,
                                 title=f"Distribuição de {eixo_y}", hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)
                export_figs.append(fig_pie)
                # Para PDF
                pdf_ctx["pie"] = (df_grouped, eixo_x, eixo_y, f"Distribuição de {eixo_y}")

            if cols_date:
                st.markdown("---")
                st.markdown("**Evolução dos Gastos no Tempo**")
                data_col = st.selectbox("Coluna de data:", cols_date, key="date_col")
                df_time = df_calculo.dropna(subset=[data_col]).copy()
                if not df_time.empty:
                    df_time = df_time.groupby(data_col)[eixo_y].sum().reset_index().sort_values(data_col)
                    fig_line = px.line(df_time, x=data_col, y=eixo_y,
                                       title="Linha do Tempo de Despesas", template="plotly_white")
                    st.plotly_chart(fig_line, use_container_width=True)
                    export_figs.append(fig_line)
                    # Para PDF
                    pdf_ctx["line"] = (df_time, data_col, eixo_y, "Linha do Tempo de Despesas")

            # Radar
            st.markdown("---")
            with st.expander("Gráfico Radar (opcional)"):
                if len(cols_num) >= 1 and cols_txt:
                    dim_col = st.selectbox("Dimensão (texto) para o Radar:", cols_txt, key="radar_dim")
                    val_cols = st.multiselect("Métricas numéricas para o Radar (2-6):", cols_num,
                                              default=cols_num[: min(3, len(cols_num))])
                    top_n = st.slider("Top N categorias por soma da 1ª métrica:",
                                      min_value=3, max_value=20, value=6, step=1)

                    if len(val_cols) >= 1:
                        base = (
                            df_calculo.groupby(dim_col)[val_cols]
                            .sum().sort_values(val_cols[0], ascending=False).head(top_n).reset_index()
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
                        # Para PDF
                        pdf_ctx["radar"] = (base_norm, dim_col, val_cols, f"Radar normalizado por {dim_col}")
                else:
                    st.info("Carregue dados com ao menos 1 coluna numérica e 1 categórica para ver o Radar.")
        else:
            st.warning("Não foi possível identificar colunas numéricas e categóricas para o dashboard.")

        # Guardar no estado
        st.session_state["export_figs"] = export_figs
        st.session_state["dashboard_ready"] = bool(export_figs)
        st.session_state["pdf_ctx"] = pdf_ctx

    # -------- Dados Limpos --------
    with tab_dados:
        st.subheader("Visualização Segura dos Dados")
        st.write(f"Total de registros: {len(df_display)}")
        st.dataframe(df_display, use_container_width=True)

    # -------- IA (Gemini) --------
    with tab_ia:
        st.subheader("Converse com os dados (Gemini)")
        pergunta = st.text_input("Pergunta (ex: Quem são os maiores fornecedores? Existem gastos atípicos?)")

        if pergunta:
            if not gemini_key:
                st.error("Insira a Gemini API Key na barra lateral.")
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
                        st.caption(f"Modelo em uso: {modelo_escolhido}")
                        model = genai.GenerativeModel(modelo_escolhido)
                        resumo = df_calculo.describe(include="all").astype(str).to_string()
                        amostra = df_display.head(15).to_string()
                        prompt = (
                            "Você é um auditor de despesas públicas. Responda de forma técnica e objetiva.\n\n"
                            f"PERGUNTA: {pergunta}\n\n"
                            f"RESUMO ESTATÍSTICO:\n{resumo}\n\n"
                            f"AMOSTRA (15 linhas):\n{amostra}\n"
                        )
                        with st.spinner("IA analisando..."):
                            resp = model.generate_content(prompt)
                            st.markdown("### Resposta da IA")
                            st.write(resp.text)
                except Exception as e:
                    st.error(f"Erro na conexão com a IA: {e}")
                    st.info("Verifique permissões da chave no Google AI Studio.")

    # -------- Downloads --------
    with tab_down:
        st.subheader("Exportações")
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

        dashboard_ready = st.session_state.get("dashboard_ready", False)
        export_figs = st.session_state.get("export_figs", [])
        pdf_ctx = st.session_state.get("pdf_ctx", {})

        html_bytes = None
        pdf_bytes = b""

        if dashboard_ready and export_figs:
            st.markdown("---")
            st.markdown("### Dashboard e Relatório")

            # HTML interativo com Plotly
            html_bytes = build_dashboard_html(export_figs, title="Dashboard de Despesas")

            # NOVO: gerar PNGs com Matplotlib para PDF (sem Chrome/Kaleido)
            figuras_png = []
            # Bar
            if "bar" in pdf_ctx:
                df_g, cx, cy, ttl = pdf_ctx["bar"]
                figuras_png.append(make_bar_png(df_g, cx, cy, ttl))
            # Pie
            if "pie" in pdf_ctx:
                df_g, cx, cy, ttl = pdf_ctx["pie"]
                figuras_png.append(make_pie_png(df_g, cx, cy, ttl))
            # Line
            if "line" in pdf_ctx:
                df_t, dc, cy, ttl = pdf_ctx["line"]
                if not df_t.empty:
                    figuras_png.append(make_line_png(df_t, dc, cy, ttl))
            # Radar
            if "radar" in pdf_ctx:
                base_norm, dim_col, metrics, ttl = pdf_ctx["radar"]
                if len(metrics) >= 1 and not base_norm.empty:
                    figuras_png.append(make_radar_png(base_norm, dim_col, metrics, ttl))

            # Resumo
            num_cols_list = df_calculo.select_dtypes(include=['number']).columns.tolist()
            cat_cols_list = [c for c in df_calculo.columns if c not in num_cols_list]
            resumo_linhas = [
                f"Total de registros: {len(df_clean)}",
                f"Colunas numéricas: {', '.join(num_cols_list) or 'nenhuma'}",
                f"Colunas não numéricas: {', '.join(cat_cols_list) or 'nenhuma'}",
            ]
            resumo_texto = "\n".join(resumo_linhas)

            pdf_bytes = build_pdf_report(
                titulo="Relatório de Despesas",
                resumo_texto=resumo_texto,
                figuras_png_bytes=figuras_png,
                author="Sistema IA"
            )

            c1, c2 = st.columns(2)
            with c1:
                if html_bytes:
                    st.download_button(
                        "⬇️ Dashboard (HTML interativo)",
                        data=html_bytes,
                        file_name="dashboard_despesas.html",
                        mime="text/html",
                    )
            with c2:
                if pdf_bytes:
                    st.download_button(
                        "🧾 Relatório (PDF)",
                        data=pdf_bytes,
                        file_name="relatorio_despesas.pdf",
                        mime="application/pdf",
                    )

            # ZIP com tudo
            st.markdown("---")
            if pdf_bytes and html_bytes:
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr("despesas_processadas.xlsx", excel_bytes)
                    zf.writestr("despesas_processadas.csv", csv_bytes)
                    zf.writestr("dashboard_despesas.html", html_bytes)
                    zf.writestr("relatorio_despesas.pdf", pdf_bytes)
                zip_buf.seek(0)
                st.download_button(
                    "📦 Baixar Tudo (ZIP)",
                    data=zip_buf.getvalue(),
                    file_name="pacote_relatorio_dashboard.zip",
                    mime="application/zip",
                )
        else:
            st.info("Gere o dashboard na aba '📊 Dashboard' para habilitar os downloads de HTML/PDF.")
else:
    st.info("💡 Faça upload de um arquivo para iniciar.")
