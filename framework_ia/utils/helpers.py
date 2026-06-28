"""
utils/helpers.py
Captura de gráficos matplotlib y salida de texto para Streamlit.
"""
import sys
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st


def run_and_capture(func, *args, **kwargs):
    """
    Ejecuta `func(*args, **kwargs)` capturando:
      - Todos los plt.show() → lista de Figure objects
      - Todo print() → string de texto
    Retorna (figs: list, text: str)
    """
    captured_figs = []
    buffer = io.StringIO()
    original_show = plt.show
    old_stdout = sys.stdout

    def _capture_show():
        fig = plt.gcf()
        if fig.get_axes():
            # Crear una copia independiente de la figura antes de limpiarla
            import copy
            captured_figs.append(fig)
        plt.figure()   # nueva figura en blanco para el siguiente plot

    plt.show = _capture_show
    sys.stdout = buffer

    try:
        func(*args, **kwargs)
    except Exception as exc:
        sys.stdout = old_stdout
        plt.show = original_show
        raise exc
    finally:
        sys.stdout = old_stdout
        plt.show = original_show

    return captured_figs, buffer.getvalue()


def display_results(figs, text, cols=2, section_title="Resultados"):
    """
    Muestra en Streamlit las figuras y el texto capturados por run_and_capture.
    """
    if text.strip():
        with st.expander("📊 Salida numérica", expanded=True):
            st.code(text, language=None)

    if figs:
        st.markdown(f"**📉 {section_title}**")
        for i in range(0, len(figs), cols):
            row = figs[i: i + cols]
            if len(row) == 1:
                st.pyplot(row[0], use_container_width=True)
                plt.close(row[0])
            else:
                st_cols = st.columns(len(row))
                for j, f in enumerate(row):
                    with st_cols[j]:
                        st.pyplot(f, use_container_width=True)
                        plt.close(f)


def metric_card(label, value, delta=None, color="#2563EB"):
    """Mini tarjeta de métrica con HTML."""
    delta_html = ""
    if delta is not None:
        arrow = "▲" if float(str(delta).replace("%", "")) >= 0 else "▼"
        delta_html = f'<div style="font-size:.75rem;color:#64748B">{arrow} {delta}</div>'
    return f"""
    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-top:3px solid {color};
                border-radius:8px;padding:.8rem 1rem;text-align:center">
        <div style="font-size:.78rem;color:#64748B;text-transform:uppercase;
                    letter-spacing:.05em">{label}</div>
        <div style="font-size:1.4rem;font-weight:700;color:#1E293B">{value}</div>
        {delta_html}
    </div>"""


def section_header(icon, title, subtitle=""):
    """Encabezado de sección consistente."""
    sub = f'<p style="margin:0;color:#64748B;font-size:.85rem">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div style="border-left:4px solid #2563EB;padding-left:1rem;margin-bottom:1rem">
        <h3 style="margin:0;color:#1E293B">{icon} {title}</h3>
        {sub}
    </div>""", unsafe_allow_html=True)


def warning_missing_module(name, origin):
    """Muestra aviso cuando falta un módulo."""
    st.warning(f"""
    **Módulo `{name}` no encontrado.**  
    Copia `{name}` desde la carpeta **{origin}** a `modules/` y reinicia la app.
    """)


def warning_missing_data(filename, origin):
    """Muestra aviso cuando falta un archivo de datos."""
    st.warning(f"""
    **Archivo `{filename}` no encontrado.**  
    Copia el archivo desde **{origin}** a `data/` y reinicia la app.
    """)