# Framework IA — EIF420O (UNA)

Framework Streamlit que unifica los 4 casos de estudio en una sola app multipágina:

| Página | Caso de Estudio | Contenido |
|---|---|---|
| `pages/1_EC1_No_Supervisado.py` | EC1 | ACP, HAC, KMeans/KMedoids |
| `pages/2_EC2_Clasificacion.py`  | EC2 | KNN, Árbol DT, Random Forest, XGBoost, ADABoost |
| `pages/3_EC3_Regresion.py`      | EC3 | Lineal, Lasso, Ridge, SVR, Árbol, RF, Gradient Boosting |
| `pages/4_EC4_IA.py`             | EC4 | Sistema Multiagente integrado (EDA, Clustering, Clasificación, Regresión, Dashboard) |

EC4 es **autónomo**: reimplementa todo con `scikit-learn` directamente y no depende de los
módulos `PaqNOSup.py`, `GuiaClaseSupervisada.py`, `M_Caso.py` ni `CVisualizer.py`. Solo necesita
el dataset `diabetes_V2.csv`.

EC1, EC2 y EC3 sí dependen de los módulos de tus carpetas originales. Esos archivos no se
incluyen aquí porque solo viste sus nombres en capturas de pantalla; debes copiarlos tú mismo
(ver paso 2).

## 1. Descomprime el proyecto

Descomprime `framework_ia.zip` donde quieras, por ejemplo en
`Documentos/Universidad/Inteligencia Artificial/framework_ia/`.

## 2. Copia los módulos y los datasets que ya tienes

Desde tus carpetas originales (`Caso de Estudio 1`, `2`, `3`), copia estos archivos:

**A `modules/`:**
- `PaqNOSup.py` ← desde **Caso de Estudio 1**
- `GuiaClaseSupervisada.py` ← desde **Caso de Estudio 2**
- `M_Caso.py` ← desde **Caso de Estudio 3**
- `CVisualizer.py` ← desde **Caso de Estudio 3**

**A `data/`:**
- `hotel_bookings_muestra.csv` ← desde **Caso de Estudio 1**
- `BankChurners.csv` ← desde **Caso de Estudio 1**
- `potabilidad_V2.csv` ← desde **Caso de Estudio 2**
- `diabetes_V2.csv` ← desde **Caso de Estudio 2** (o **Caso de Estudio 4**, es el mismo archivo)
- `Toyota_Price.csv` ← desde **Caso de Estudio 3**
- `kc_house_data.csv` ← desde **Caso de Estudio 3**

No necesitas copiar `ModuloACP.py`, `No_Supervisados.py`, `Ec3_main.py` ni `prueba.py`: por lo que
se ve en las capturas, son versiones previas/sueltas que ya no usan las páginas finales.

## 3. Instala dependencias

Desde la carpeta `framework_ia/`, con un entorno virtual activado (recomendado):

```bash
pip install -r requirements.txt
```

Si al ejecutar EC1, EC2 o EC3 ves un `ModuleNotFoundError` (por ejemplo `xgboost` o
`scikit-learn-extra`), es porque tus módulos originales importan algo que no está en
`requirements.txt` (no pude ver su código fuente, solo el listado de carpetas). Instálalo con
`pip install <paquete>` y vuelve a intentar. Si me compartes esos 4 archivos `.py`, te dejo el
`requirements.txt` exacto.

## 4. Ejecuta el framework

```bash
streamlit run Inicio.py
```

Esto abre la página de inicio con las 4 tarjetas de los casos de estudio. Navega entre ellos
con el menú de la izquierda. **No** ejecutes `streamlit run pages/1_EC1...py` directamente: el
punto de entrada siempre es `Inicio.py`.

## 5. Notas por caso de estudio

- **EC1**: si activas HAC con un dataset grande, se toma automáticamente una muestra de 500 filas
  para que no se cuelgue.
- **EC2**: el target de "Diabetes" está mapeado a la columna `class`; revisa que tu CSV use ese
  nombre o ajústalo en `TARGET_MAP` dentro de `pages/2_EC2_Clasificacion.py`.
- **EC3**: requiere `M_Caso.py` con una clase `Regresion` (o `Ec3_main.py` con la misma clase
  como alternativa). Si el botón de ejecución aparece deshabilitado, es porque ese módulo no se
  encontró en `modules/`.
- **EC4**: en la pestaña "Carga de Datos" hay un botón para usar `diabetes_V2.csv` directamente
  desde `data/`, sin necesidad de subirlo manualmente.
