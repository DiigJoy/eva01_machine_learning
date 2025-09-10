# 🚢 Titanic Survival Prediction - Machine Learning

Proyecto de clasificación binaria para predecir la supervivencia de pasajeros del Titanic usando técnicas de Machine Learning.

## 📊 Resultados Principales

- **Mejor Modelo:** XGBoost
- **Accuracy en Test:** 85.4%
- **AUC-ROC:** 0.891
- **F1-Score:** 0.83

## 🎯 Objetivo del Proyecto

Desarrollar un modelo de Machine Learning que pueda predecir si un pasajero del Titanic sobrevivió o no, basándose en características como edad, sexo, clase, familia, etc.

## 📋 Contenido del Repositorio

### 📁 Estructura de Carpetas:
```
eva01_machine_learning/
├── notebooks/
│   └── titanic_survival_analysis.ipynb
├── data/
│   └── titanic.csv
├── results/
│   └── (aquí van las capturas)
├── docs/
│   └── informe_final_titanic.md
├── requirements.txt
└── README.md
```

### 📊 Archivos Principales:
- `notebooks/titanic_survival_analysis.ipynb` - Análisis completo del proyecto
- `data/titanic.csv` - Dataset del Titanic
- `requirements.txt` - Lista de librerías necesarias

## 🚀 Instrucciones de Ejecución

### **Opción 1: Google Colab (Recomendado)**
1. Abre el archivo `notebooks/titanic_survival_analysis.ipynb` en Google Colab
2. Ejecuta todas las celdas en orden
3. Los datos se cargan automáticamente desde URL

### **Opción 2: Jupyter Local**
1. Clona este repositorio:
```bash
   git clone https://github.com/DiigJoy/eva01_machine_learning.git
   cd titanic-ml-classification
```


2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Ejecuta Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Abre `notebooks/titanic_survival_analysis.ipynb`

5. Ejecuta todas las celdas

## 🔧 Tecnologías Utilizadas

- **Python 3.8+**
- **Pandas** - Manipulación de datos
- **NumPy** - Operaciones numéricas
- **Scikit-learn** - Machine Learning
- **XGBoost** - Algoritmo de gradient boosting
- **Matplotlib/Seaborn** - Visualización de datos
- **Jupyter Notebook** - Entorno de desarrollo

## 📈 Metodología

### 1. **Análisis Exploratorio de Datos (EDA)**
- Exploración de variables y distribuciones
- Análisis de valores faltantes
- Correlaciones entre variables
- Visualizaciones descriptivas

### 2. **Preparación de Datos**
- Limpieza de datos y manejo de valores faltantes
- Feature engineering (creación de nuevas variables)
- Encoding de variables categóricas
- División en conjuntos de entrenamiento, validación y test

### 3. **Modelado**
- Implementación de múltiples algoritmos:
  - Logistic Regression (baseline)
  - Random Forest
  - XGBoost (mejor modelo)
- Validación cruzada
- Optimización de hiperparámetros

### 4. **Evaluación**
- Métricas múltiples: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Matriz de confusión
- Análisis de importancia de features
- Interpretación de resultados

## 📊 Principales Hallazgos

1. **Género:** Las mujeres tuvieron 3x más probabilidad de sobrevivir que los hombres
2. **Clase:** Los pasajeros de primera clase tuvieron 63% de supervivencia vs 24% de tercera clase
3. **Edad:** Los niños y jóvenes tuvieron mayor supervivencia
4. **Familia:** Viajar solo redujo las probabilidades de supervivencia
5. **Variables más importantes:** Sexo, Clase, Edad, Tarifa

## 👨‍💻 Autores

**Jorge Barrios y Matias Pavez**
- Curso: Machine Learning (TEL26)
- Fecha: 10 Septiembre 2025

## 📄 Licencia

Este proyecto es para fines educativos.

## 🔗 Enlaces Adicionales

- [Dataset Original](https://www.kaggle.com/c/titanic)
- [Documentación de XGBoost](https://xgboost.readthedocs.io/)
- [Scikit-learn](https://scikit-learn.org/)

---