# 🤖 Framework AgentesIA — Sistema Multiagente para Machine Learning

Sistema Multiagente (MAS) desarrollado en **Python + Streamlit** para el curso de Inteligencia Artificial (EIF-420O, UNA). Un **Agente Coordinador** orquesta cuatro agentes especializados —EDA, Clustering, Clasificación y Regresión— que analizan datasets, entrenan modelos, comparan algoritmos mediante benchmarking automático y presentan resultados en un dashboard interactivo.

## 🏗️ Arquitectura del sistema

```
                        Usuario
                           │
                           ▼
              ┌─────────────────────────┐
              │  Agente Coordinador MAS  │
              └─────────────────────────┘
                 │       │       │      │
                 ▼       ▼       ▼      ▼
               EDA   Clustering  Clasif  Regres
                 \      │        │      /
                  \     │        │     /
                   ▼    ▼        ▼    ▼
                  Dashboard Streamlit
```

### Agente Coordinador
Cerebro del sistema: carga y valida los datasets, activa y coordina a los demás agentes, consolida métricas, compara modelos entre sí y genera el reporte final con recomendaciones automáticas.

### Agente EDA (Análisis Exploratorio)
- Calidad de datos: nulos, duplicados, inconsistencias.
- Estadística descriptiva: media, mediana, moda, varianza, desviación estándar.
- Correlación (Pearson, Spearman) y heatmaps.
- Detección de outliers (IQR, Z-Score).
- Visualización: histogramas, boxplots, scatterplots, pairplots.

### Agente No Supervisado — Clustering
- **K-Means** (número óptimo de clusters, método del codo).
- **Clustering Jerárquico** (dendrogramas).
- **DBSCAN** (regiones densas, detección de ruido).
- Métricas: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index.

### Agente Supervisado — Clasificación
- Algoritmos: KNN, Decision Tree, Random Forest, AdaBoost, XGBoost, Naive Bayes, Logistic Regression, SVM.
- Métricas: Accuracy, Precision, Recall, F1, ROC-AUC, matriz de confusión.
- Benchmarking dentro de la misma familia (p. ej. Árbol vs. Random Forest) y entre familias distintas (p. ej. Random Forest vs. SVM).

### Agente Supervisado — Regresión
- Algoritmos: Linear Regression, Ridge/RidgeCV, Lasso/LassoCV, SVR, Decision Tree Regressor, Random Forest Regressor, XGBoost Regressor.
- Métricas: MAE, MSE, RMSE, R², MAPE.
- Benchmarking entre modelos lineales, basados en árboles y de boosting.

## 📊 Dashboard (Streamlit)

| Página | Contenido |
|---|---|
| 1. Inicio | Proyecto, integrantes, objetivos |
| 2. Carga de Datos | Subida y visualización del CSV |
| 3. EDA | Estadísticas, correlaciones, histogramas, nulos |
| 4. Clustering | Método del codo, Silhouette Score, clusters |
| 5. Clasificación | Accuracy, Precision, Recall, F1, AUC |
| 6. Regresión | MAE, RMSE, R² |
| 7. Dashboard Ejecutivo | Mejor algoritmo, comparación general, recomendaciones |

## 🗂️ Estructura del proyecto

```
framework_ia/
├── Inicio.py                       # Punto de entrada (streamlit run Inicio.py)
├── pages/
│   ├── 1_EC1_No_Supervisado.py
│   ├── 2_EC2_Clasificacion.py
│   ├── 3_EC3_Regresion.py
│   └── 4_EC4_IA.py                 # Sistema Multiagente integrado
├── modules/                        # Lógica de los agentes especializados
├── data/                           # Datasets de ejemplo (diabetes, potabilidad, etc.)
├── utils/
│   └── helpers.py
└── requirements.txt
```

## ▶️ Instalación y ejecución

```bash
git clone https://github.com/MinorCL/Framework_AgentesIA.git
cd Framework_AgentesIA/framework_ia
pip install -r requirements.txt
streamlit run Inicio.py
```

> El sistema multiagente integrado (EC4) es autónomo: solo necesita el dataset `diabetes_V2.csv` incluido en `data/`.

## 🎓 Contexto académico

Proyecto para el curso **EIF-420O — Inteligencia Artificial**, Universidad Nacional de Costa Rica (UNA), I Ciclo 2026. Basado en los fundamentos de Sistemas Multiagente (MAS) y Agentic AI aplicados a la automatización del ciclo completo de análisis de datos y Machine Learning.

## 👥 Integrantes

- Gabriel Jesús Bello Escalona
- Minor Castillo Loria