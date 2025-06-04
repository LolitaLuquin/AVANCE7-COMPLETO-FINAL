# AVANCE7 Eyetracking-FEA con carrusel, filtros y botones de descarga

import streamlit as st
import zipfile
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from io import StringIO, BytesIO
from collections import Counter
from tempfile import TemporaryDirectory
import numpy as np

# --- FUNCIONES DE CARGA ---
def make_unique(headers):
    counts = Counter()
    new_headers = []
    for h in headers:
        counts[h] += 1
        if counts[h] > 1:
            new_headers.append(f"{h}_{counts[h]-1}")
        else:
            new_headers.append(h)
    return new_headers

def read_imotions_csv(file, participant_name, header_index, tipo):
    content = file.read().decode('utf-8')
    lines = content.splitlines()
    headers = make_unique(lines[header_index].strip().split(","))
    data = "\n".join(lines[header_index + 1:])
    df = pd.read_csv(StringIO(data), names=headers)
    df["Participant"] = participant_name
    df["Tipo"] = tipo
    return df

def upload_and_concat(tipo, header_index):
    uploaded_files = st.file_uploader(f"Sube archivos de {tipo} (CSV)", accept_multiple_files=True, type="csv")
    dfs = []
    if uploaded_files:
        for file in uploaded_files:
            participant = file.name.replace(".csv", "").strip()
            df = read_imotions_csv(file, participant, header_index, tipo)
            dfs.append(df)
        df_merged = pd.concat(dfs, ignore_index=True)
        st.success(f"{tipo} fusionado con {len(dfs)} archivo(s).")
        st.download_button(f"Descargar {tipo} mergeado", df_merged.to_csv(index=False).encode(), file_name=f"{tipo.lower()}_merged.csv", mime='text/csv')
        return df_merged
    return pd.DataFrame()

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(layout="wide")
st.title("AVANCE7 - Eyetracking y FEA con carrusel")

with st.sidebar:
    st.header("Carga de archivos")
    df_et = upload_and_concat("Eyetracking", 25)
    df_fea = upload_and_concat("FEA", 25)

# --- ANÁLISIS DE EYETRACKING ---
with st.expander("Análisis de Eyetracking", expanded=False):
    if not df_et.empty:
        st.subheader("Estadísticas generales")
        df_et["ET_TimeSignal"] = pd.to_numeric(df_et["ET_TimeSignal"], errors="coerce")
        df_et = df_et.dropna(subset=["ET_TimeSignal", "SourceStimuliName"])

        tabla_et = df_et.groupby("SourceStimuliName").agg(
            Tiempo_Medio=("ET_TimeSignal", "mean"),
            Desviacion_Estandar=("ET_TimeSignal", "std"),
            Conteo=("ET_TimeSignal", "count")
        ).reset_index()

        st.dataframe(tabla_et)
        st.download_button("Descargar tabla Eyetracking", tabla_et.to_csv(index=False).encode(), file_name="tabla_eyetracking.csv", mime='text/csv')

        estimulos = df_et["SourceStimuliName"].unique()
        data_por_estimulo = [df_et[df_et["SourceStimuliName"] == stim]["ET_TimeSignal"] for stim in estimulos]
        anova_result = stats.f_oneway(*data_por_estimulo)
        f_stat = anova_result.statistic
        p_value = anova_result.pvalue
        f_squared = (f_stat * (len(estimulos) - 1)) / (len(df_et) - len(estimulos)) if len(estimulos) > 1 else None

        estad_txt = f"ANOVA F-statistic: {f_stat:.4f}\n"
        estad_txt += f"p-value: {p_value:.4e}\n"
        if f_squared:
            estad_txt += f"F-squared: {f_squared:.4f}\n"

        st.text_area("Estadísticos", estad_txt, height=100)
        st.download_button("Descargar estadísticos", estad_txt, file_name="estadisticos_eyetracking.txt")

        # Filtro por estímulo
        st.subheader("Gráficas por estímulo")
        estimulo_seleccionado = st.selectbox("Selecciona un estímulo:", sorted(estimulos))
        subset_est = df_et[df_et["SourceStimuliName"] == estimulo_seleccionado]
        fig_est, ax = plt.subplots(figsize=(8,5))
        sns.histplot(subset_est["ET_TimeSignal"], kde=True, ax=ax)
        ax.set_title(f"Histograma de Tiempo - {estimulo_seleccionado}")
        st.pyplot(fig_est)

        with TemporaryDirectory() as tmpdir:
            zip_path = f"{tmpdir}/graficas_estimulo_eyetracking.zip"
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for stim in estimulos:
                    fig, ax = plt.subplots(figsize=(8,5))
                    sns.histplot(df_et[df_et["SourceStimuliName"] == stim]["ET_TimeSignal"], kde=True, ax=ax)
                    ax.set_title(f"Histograma - {stim}")
                    path = f"{tmpdir}/{stim}.png"
                    fig.savefig(path)
                    zipf.write(path, arcname=os.path.basename(path))
            with open(zip_path, "rb") as f:
                st.download_button("Descargar todas las gráficas por estímulo (ZIP)", f.read(), file_name="graficas_estimulo_eyetracking.zip")

# --- ANÁLISIS DE FEA ---
with st.expander("Análisis de FEA (emociones y valencia)", expanded=False):
    if not df_fea.empty:
        st.subheader("Cálculos base de FEA")

        emociones = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise']
        df_fea["Engagement_Promedio"] = df_fea[emociones].mean(axis=1)
        df_fea["Valence_Class"] = df_fea["Valence"].apply(lambda x: "Positiva" if x > 0 else ("Negativa" if x < 0 else "Neutra"))

        tabla_fea = df_fea.groupby("SourceStimuliName").agg({
            "Valence": ["mean", "std"],
            "Engagement_Promedio": ["mean", "std"]
        }).reset_index()
        tabla_fea.columns = ["Estímulo", "Valencia_Media", "Valencia_SD", "Engagement_Media", "Engagement_SD"]

        st.dataframe(tabla_fea)
        st.download_button("Descargar tabla FEA", tabla_fea.to_csv(index=False).encode(), file_name="tabla_resumen_fea.csv", mime="text/csv")

        # Estadísticos
        def f_squared(anova_result):
            return anova_result.statistic / (anova_result.statistic + df_fea.shape[0] - 1) if not np.isnan(anova_result.statistic) else np.nan

        stimuli_groups_val = [g["Valence"].dropna() for _, g in df_fea.groupby("SourceStimuliName") if len(g["Valence"].dropna()) > 1]
        stimuli_groups_eng = [g["Engagement_Promedio"].dropna() for _, g in df_fea.groupby("SourceStimuliName") if len(g["Engagement_Promedio"].dropna()) > 1]

        anova_valencia = stats.f_oneway(*stimuli_groups_val) if len(stimuli_groups_val) > 1 else None
        anova_engagement = stats.f_oneway(*stimuli_groups_eng) if len(stimuli_groups_eng) > 1 else None

        est_txt = ""
        for var, stats_ in {
            "Valencia": {
                "F": anova_valencia.statistic if anova_valencia else "No aplica",
                "p-value": anova_valencia.pvalue if anova_valencia else "No aplica",
                "F²": f_squared(anova_valencia) if anova_valencia else "No aplica"
            },
            "Engagement": {
                "F": anova_engagement.statistic if anova_engagement else "No aplica",
                "p-value": anova_engagement.pvalue if anova_engagement else "No aplica",
                "F²": f_squared(anova_engagement) if anova_engagement else "No aplica"
            }
        }.items():
            est_txt += f"{var}:\n"
            for k, v in stats_.items():
                est_txt += f"  {k}: {v}\n"
            est_txt += "\n"

        st.text_area("Estadísticos FEA", est_txt, height=140)
        st.download_button("Descargar estadísticos FEA", est_txt, file_name="estadisticos_fea.txt")

        # Descarga por estímulo
        with TemporaryDirectory() as tmpdir:
            zip_path = f"{tmpdir}/stimuli_fea_data.zip"
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for stim in df_fea["SourceStimuliName"].unique():
                    df_stim = df_fea[df_fea["SourceStimuliName"] == stim]
                    stim_path = f"{tmpdir}/{stim}.csv"
                    df_stim.to_csv(stim_path, index=False)
                    zipf.write(stim_path, arcname=os.path.basename(stim_path))
            with open(zip_path, "rb") as f:
                st.download_button("Descargar CSVs por estímulo FEA (ZIP)", f.read(), file_name="stimuli_fea_data.zip")
