# app.py

import os
import io
import json
import zipfile
import subprocess
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import google.generativeai as genai

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader


# =========================
# Suporte a Kaleido + Chrome (Streamlit Cloud)
# =========================
def ensure_chrome_for_kaleido() -> bool:
    """
    Garante que o Kaleido consiga renderizar PNGs dos gráficos.
    Tenta instalar/descobrir o caminho do Google Chrome via 'plotly_get_chrome'.
    Requer 'plotly-get-chrome' no requirements.
    """
    try:
        path = subprocess.check_output(
            ["plotly_get_chrome", "--install", "--path-only"],
            text=True
        ).strip()
        if path:
            os.environ["PLOTLY_CHROME"] = path  # usado pelo Kaleido/Plotly
            return True
    except Exception as e:
        # Não quebra o app; apenas indica que PDF pode sair sem gráficos
        st.warning(f"Chrome não disponível para Kaleido: {e}")
    return False


# =========================
# Funções utilitárias (Downloads)
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
        text_obj.textLine(line[:115])
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
# Configuração da página
# =========================
st.set_page_config(page_title="Analista de Despesas - Prefeitura", layout="wide")
st.title("🏛️ Analista de Despesas (Limpeza, Dashboard, IA)")
st.markdown("---")

# Garante Chrome p/ Kaleido (não quebra se falhar)
ensure_chrome_for_kaleido()

# =========================
# Barra lateral
# =========================
st.sidebar.header("⚙️ Configurações")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
st.sidebar.caption("Dica: exporte GEMINI_API_KEY no seu ~/.zshrc para preencher automaticamente.")

# =========================
# Upload de arquivo
# =========================
st.subheader("📂 Carregar Planilha")
uploaded_file = st.file_uploader("Arraste seu arquivo CSV/XLSX/XLS aqui", type=["csv", "xlsx", "xls"])

if uploaded_file:
    # Leitura do arquivo (motores explícitos)
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
        # Heurística de colunas numéricas
        if any(k in col for k in ["valor", "pago", "total", "quantidade", "preco", "custo", "despesa"]):
            df_calculo[col] = pd.to_numeric(df_calculo[col], errors="coerce").fillna(0)

        # Datas
        if any(k in col for k in ["data", "vencimento", "emissao"]):
            df_calculo[col] = pd.to_datetime(df_calculo[col], errors="coerce")

    # DataFrame para exibição segura (evita ArrowTypeError)
    df_display = df_calculo.copy()
    for col in df_display.columns:
        if (not pd.api.types.is_numeric_dtype(df_display[col])) and (not pd.api.types.is_datetime64_any_dtype(df_display[col])):
            df_display[col] = df_display[col].astype(str).replace("nan", "")

    # Bytes para planilhas (Excel/CSV)
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
        df_clean.to_excel(writer, index=False, sheet_name="dados_limpos")
    excel_bytes = excel_buf.getvalue()
    csv_bytes = df_clean.to_csv(index=False).encode("utf-8-sig")

    # =========================
    # Interface em abas
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
        else:
            st.warning("Não foi possível identificar colunas numéricas e categóricas para o dashboard.")

        # Guardar no estado para downloads
        st.session_state["export_figs"] = export_figs
        st.session_state["dashboard_ready"] = bool(export_figs)

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

                    # Descobrir modelos disponíveis e escolher automaticamente
                    modelos = list(genai.list_models())
                    modelos_validos = [
                        m.name for m in modelos
                        if hasattr(m, "supported_generation_methods")
                        and "generateContent" in m.supported_generation_methods
                    ]

                    prefer = [
                        "models/gemini-1.5-flash",
                        "models/gemini-1.5-pro",
                        "models/gemini-pro",
                    ]
                    modelo_escolhido = None
                    for pref in prefer:
                        if pref in modelos_validos:
                            modelo_escolhido = pref
                            break
                    if not modelo_escolhido and modelos_validos:
                        modelo_escolhido = modelos_validos[0]

                    if not modelo_escolhido:
                        st.error("Nenhum modelo compatível encontrado para sua chave. Verifique permissões no Google AI Studio.")
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
                    st.info("Atualize a lib: pip install -U google-generativeai. Verifique permissões do modelo no AI Studio.")

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

        # Dashboard HTML e Relatório PDF (quando houver figuras)
        dashboard_ready = st.session_state.get("dashboard_ready", False)
        export_figs = st.session_state.get("export_figs", [])

        html_bytes = None
        pdf_bytes = b""

        if dashboard_ready and export_figs:
            st.markdown("---")
            st.markdown("### Dashboard e Relatório")

            # HTML interativo do dashboard
            html_bytes = build_dashboard_html(export_figs, title="Dashboard de Despesas")

            # PNGs das figuras para PDF (via kaleido) com fallback se não houver Chrome
            figuras_png = []
            try:
                for f in export_figs:
                    png = pio.to_image(f, format="png", width=1200, height=700, engine="kaleido")
                    figuras_png.append(png)
            except Exception as e:
                st.warning(f"Falha ao renderizar imagens para PDF (Kaleido/Chrome): {e}")
                st.info("O PDF será gerado sem gráficos. Para habilitar gráficos, adicione 'plotly-get-chrome' no requirements e reimplante.")

            # Resumo simples para o PDF
            resumo_linhas = [
                f"Total de registros: {len(df_clean)}",
                f"Colunas numéricas: {', '.join(df_calculo.select_dtypes(include=['number']).columns) or 'nenhuma'}",
                f"Colunas categóricas: {', '.join([c for c in df_calculo.columns if c not in df_calculo.select_dtypes(include=['number']).columns]) or 'nenhuma'}",
            ]
            resumo_texto = "\n".join(resumo_linhas)

            # Gera PDF (com ou sem imagens)
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

            # ZIP opcional com tudo
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
