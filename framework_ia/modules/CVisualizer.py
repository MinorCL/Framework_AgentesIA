
import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
from sklearn.exceptions import ConvergenceWarning
import warnings
import threading
import seaborn as sns
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, URL
from sqlalchemy_utils import database_exists, create_database
import pymssql
import time
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import CGenerator as cG
import CEvaluator as cE
import CPredictor as cP
import CVisualizer as cV
import streamlit as st
from streamlit_option_menu import option_menu

@st.cache_data
def load_and_process_data(csv_files, use_sql, db_params):
    data_gen = cG.DataGenerator(csv_files, use_sql=use_sql, db_params=db_params)
    data_gen.process()
    X_trains, X_tests, y_trains, y_tests = data_gen.get_data()
    return X_trains, X_tests, y_trains, y_tests, data_gen.dfs

@st.cache_data
def train_models(X_trains, X_tests, y_trains, y_tests):
    results_genetic = {}
    results_exhaustive = {}

    for dataset_name in X_trains.keys():
        evaluator = cE.ModelEvaluator(X_trains[dataset_name], X_tests[dataset_name], 
                                   y_trains[dataset_name], y_tests[dataset_name])
        results_genetic[dataset_name] = evaluator.genetic_search()
        X_tests[dataset_name] = evaluator.X_train
        X_tests[dataset_name] = evaluator.X_test      
        results_exhaustive[dataset_name] = evaluator.exhaustive_search()

    genetic_evaluation = {}
    exhaustive_evaluation = {}
    for dataset_name in X_trains.keys():
        predictor = cP.Predictor(X_tests[dataset_name], y_tests[dataset_name])
        genetic_evaluation[dataset_name] = predictor.evaluate_results(results_genetic[dataset_name])
        exhaustive_evaluation[dataset_name] = predictor.evaluate_results(results_exhaustive[dataset_name])

    return results_genetic, results_exhaustive, genetic_evaluation, exhaustive_evaluation


class Visualizer:
    """Componente visualizador: Se encarga de crear el dashboard."""
    def __init__(self, results_genetic, results_exhaustive, dfs):
        self.results_genetic = results_genetic
        self.results_exhaustive = results_exhaustive
        self.dfs = dfs

    def create_results_dataframe(self):
        data = []
        for dataset_name in self.results_genetic.keys():
            for alg in self.results_genetic[dataset_name].keys():
                data.append({
                    'Dataset': dataset_name,
                    'Algoritmo': alg,
                    'RMSE (Genético)': self.results_genetic[dataset_name][alg]['rmse'],
                    'RMSE (Exhaustivo)': self.results_exhaustive[dataset_name][alg]['rmse']
                })
        df = pd.DataFrame(data)
        df.sort_values(by=['RMSE (Genético)', 'RMSE (Exhaustivo)'], ascending=[True, True], inplace=True)
        return df


    def create_correlation_heatmap(self, dataset_name):
        """Crea un heatmap de correlación para un dataset específico."""
        if dataset_name not in self.dfs:
            return go.Figure()  
        
        corr = self.dfs[dataset_name].corr()
        fig = px.imshow(corr, 
                        labels=dict(color="Correlation"),
                        x=corr.columns,
                        y=corr.columns,
                        color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1)
        fig.update_layout(title=f"Heatmap de Correlación - {dataset_name}")
        st.plotly_chart(fig)
        
    def create_dashboard(self):        
        with st.sidebar:
            selected = option_menu(
                menu_title="Menú Principal",
                options=["EDA", "Resultados"],
                menu_icon="cast",
                default_index=0,
            )

        if selected == "EDA":
            self.page_eda()
        elif selected == "Resultados":
            self.page_model_results()
            
    def page_eda(self):
        st.title("Exploratory Data Analysis (EDA)")

        selected_dataset = st.selectbox("Selecciona el dataset:", list(self.dfs.keys()))

        if selected_dataset:
            data = self.dfs[selected_dataset]
            melted_data = data.melt()

            st.subheader("Descripción de los Datos")
            st.write("Tipos de Datos:")
            st.write(data.dtypes)
            st.write("Valores Nulos:")
            st.write(data.isnull().sum())
            st.write("Descripción Estadística:")
            st.write(data.describe())


            st.subheader("Histograma")
            fig = px.histogram(melted_data, x="value", color="variable", marginal="box", hover_data=['variable'])
            st.plotly_chart(fig)

            st.subheader("Boxplot")
            fig = px.box(melted_data, y="value", color="variable", points="all", hover_data=['variable'])
            st.plotly_chart(fig)

            st.subheader("Mapa de Correlacion")
            self.create_correlation_heatmap(selected_dataset)
            
    def page_model_results(self):
        st.title("Resultados")

        df_results = self.create_results_dataframe()
        all_datasets = list(self.dfs.keys())
        all_algorithms = list(self.results_genetic[all_datasets[0]].keys())
        selected_dataset = st.selectbox("Selecciona el dataset:", all_datasets)
        selected_algorithm = st.selectbox("Select el algoritmo", all_algorithms)

        # Display RMSE comparison bar chart
        df_results_filtered = df_results[df_results['Dataset'] == selected_dataset]
        comparison_fig = go.Figure()
        comparison_fig.add_trace(go.Bar(
            x=df_results_filtered['Algoritmo'],
            y=df_results_filtered['RMSE (Genético)'],
            name='Genético',
            marker_color='blue'
        ))
        comparison_fig.add_trace(go.Bar(
            x=df_results_filtered['Algoritmo'],
            y=df_results_filtered['RMSE (Exhaustivo)'],
            name='Exhaustivo',
            marker_color='red'
        ))
        comparison_fig.update_layout(
            title=f'Comparación de RMSE: Genético vs Exhaustivo ({selected_dataset})',
            xaxis_title='Algoritmo',
            yaxis_title='RMSE',
            barmode='group'
        )
        st.plotly_chart(comparison_fig)

        st.subheader(f"Mejores parámetros para {selected_algorithm} ({selected_dataset}):")
        st.markdown("**Método Genético:**")
        st.json(self.results_genetic[selected_dataset][selected_algorithm]['best_params'])
        st.markdown("**Método Exhaustivo:**")
        st.json(self.results_exhaustive[selected_dataset][selected_algorithm]['best_params'])

        st.subheader("Tabla de Resultados")
        st.dataframe(df_results)

