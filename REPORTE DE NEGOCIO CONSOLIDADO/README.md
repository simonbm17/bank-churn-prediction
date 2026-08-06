# FinanceGuard: Predicción de Churn Bancario

Sistema de predicción y segmentación de abandono de clientes (churn) para un banco digital, desarrollado como Proyecto Integrador del Módulo 4 de la carrera de Data Science en Soy Henry.

## Contexto del problema

FinanceGuard es un banco digital que pierde el 20% de sus clientes cada año, uno de cada cinco. Esto representa pérdida directa de ingresos, un alto costo de adquisición de nuevos clientes (más caro que retener a los existentes) y un riesgo reputacional creciente. El objetivo del proyecto es construir herramientas de datos que permitan reducir esa tasa de abandono del 20% al 15%.

El rol asumido fue el de científico de datos, encargado de predecir qué clientes están en riesgo, entender los factores que impulsan el abandono, y traducir los hallazgos en recomendaciones accionables para el equipo de retención.

## Dataset

El conjunto de datos contiene 10.000 clientes y 14 variables, sin valores nulos ni duplicados. Las variables se agrupan en tres tipos:

| Tipo | Variables |
|------|-----------|
| Demográficas | Edad, género, país (Francia, Alemania, España) |
| Financieras | Saldo, salario estimado, puntaje de crédito |
| Comportamiento | Antigüedad, número de productos, tarjeta de crédito, membresía activa |

La variable objetivo es `Exited` (si el cliente abandonó o no). Un desafío central del proyecto es el **desbalance de clases**: el 80% de los clientes permanecen activos y solo el 20% abandona.

## Estructura del repositorio

```
├── notebooks/
│   ├── 1_EDA_RegresionLogistica.ipynb      # Avance 1: análisis y modelo base
│   ├── 2_GradientBoosting_Optimizacion.ipynb  # Avance 2: modelos avanzados
│   └── 3_AprendizajeNoSupervisado.ipynb     # Avance 3: segmentación
├── data/
│   └── Churn_Modelling.csv                   # Dataset original
├── reporte/
│   └── Reporte_Modelos.pdf                   # Reporte de negocio consolidado
└── README.md
```

Los notebooks están encadenados: el Avance 1 limpia y codifica los datos y los exporta, y los avances 2 y 3 los cargan. Se recomienda ejecutarlos en orden.

## Metodología y resultados

### Avance 1: Análisis exploratorio y modelo base

El análisis exploratorio reveló patrones claros de riesgo antes de modelar:

- Los clientes de **Alemania** abandonan al 32%, el doble que Francia (16%) y España (17%).
- Las **mujeres** abandonan más que los hombres (25% frente a 16%).
- El **número de productos** muestra un patrón no lineal en forma de U: con 2 productos el churn es mínimo (8%), pero con 3 sube al 83% y con 4 alcanza el 100%.

Como modelo base se implementó una **regresión logística**, elegida por su interpretabilidad y su idoneidad para problemas binarios. El desbalance se manejó con `class_weight='balanced'`, y la interpretación se apoyó en los odds ratios: ser de Alemania multiplica el riesgo por 2,28, mientras que ser miembro activo lo reduce a 0,41.

| Métrica | Valor |
|---------|-------|
| Recall | 0,70 |
| ROC-AUC | 0,777 |
| Clientes en riesgo detectados | 285 de 407 |

Se priorizó el **recall** sobre la exactitud, dado que con datos desbalanceados la exactitud resulta engañosa y el costo de no detectar a un cliente que se va es mayor que el de una falsa alarma.

### Avance 2: Modelos avanzados y optimización

Se entrenaron cuatro modelos basados en árboles, cubriendo las dos estrategias de ensamblado: bagging (Random Forest) y boosting (XGBoost, LightGBM, CatBoost). Todos superaron el modelo base. El XGBoost se optimizó con Grid Search y validación cruzada, y se construyó un ensamble de Stacking para comparar.

| Modelo | ROC-AUC | Recall |
|--------|---------|--------|
| Regresión Logística (base) | 0,777 | 0,70 |
| Random Forest | 0,864 | 0,68 |
| LightGBM | 0,856 | 0,71 |
| CatBoost | 0,866 | 0,74 |
| XGBoost optimizado | 0,870 | 0,76 |
| Stacking | 0,870 | 0,76 |

El XGBoost optimizado y el Stacking empataron en rendimiento. Se seleccionó el **XGBoost** como modelo final: entrega el mismo resultado con menor complejidad, menor costo computacional y mayor facilidad de mantenimiento. La complejidad adicional del Stacking no aportó mejoras, ya que combina modelos de boosting demasiado similares entre sí. El análisis de importancia de variables confirmó que los árboles aprovechan el efecto no lineal del número de productos que la regresión logística no capturaba.

### Avance 3: Segmentación no supervisada

Se aplicó aprendizaje no supervisado para segmentar a los clientes sin usar la variable objetivo. Con K-Means (K=3, seleccionado mediante el método del codo y el coeficiente de silueta) se identificaron tres segmentos que resultaron corresponder a los tres países.

El hallazgo más relevante del proyecto surgió aquí: sin recibir información geográfica ni sobre el abandono, el clustering identificó al segmento de Alemania como el de mayor riesgo (32% de churn), coincidiendo con lo detectado por los modelos supervisados. Dos enfoques independientes llegando a la misma conclusión, lo que refuerza la validez del análisis.

El avance se complementó con DBSCAN para la detección de clientes atípicos, y con PCA y t-SNE para la visualización de los grupos. Se derivaron dos variables nuevas a partir del clustering: la etiqueta de segmento y un ranking de riesgo.

## Recomendaciones de negocio

1. **Desplegar el XGBoost optimizado** para asignar un puntaje de riesgo a cada cliente y priorizar los contactos de retención.
2. **Priorizar el mercado de Alemania**, el segmento de mayor riesgo y mayor saldo, con campañas dirigidas.
3. **Actuar sobre las variables accionables**: fomentar la contratación de un segundo producto (el punto óptimo) e impulsar la actividad del cliente, el mayor factor protector.
4. **Monitorear el modelo en producción** y reentrenarlo de forma periódica para mantener su desempeño.

Los modelos supervisados responden a la pregunta de *a quién* contactar, mientras que la segmentación responde a *cómo* agrupar las acciones. Ambos enfoques se complementan.

## Tecnologías utilizadas

Python, pandas, scikit-learn, XGBoost, LightGBM, CatBoost, statsmodels, matplotlib, seaborn.

## Autor

**Simón Bedoya**
Carrera de Data Science, Soy Henry
