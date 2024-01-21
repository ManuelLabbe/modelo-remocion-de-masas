# Proyecto remocion de masas  
**Bayesian Model**

## Notebook train_test_models
El notebook se encuentra documentado sobre las funciones que llama.  
El notebook funciona llamando a la librería utils.py del documento el cual consta de dos funciones de métricas, y una clase llamada "model_training_data" el cual tiene los métodos para obtener el conjunto de entrenamiento y test, los modelos, entrenamiento de los modelos, leer modelos preentrandos y realizar la inferencia con los modelos.

## Descripción de los archivos y carpetas
**data**: Datos raw y datos preprocesados con 2 o 3 variables las cuales son "slope", "PP"(precipitación) y "valor_humedad_suelo1"  

**modelos**: 3 archivos pkl en donde se encuentran dos modelos bayesianos uno simple y otro con factores al cuadrado y al cubo junto on un modelo el cual es un modelo mixto de una red neuronal junto con redes bayesianas.  
  
**remocion_de_masas.yml**: archivo para la creación del entorno de python  
  
**train_test_models.ipynb**: notebook donde se pueden utilizar y/o reentrenar estos modelos  
  
**utils.py**: funciones utilizadas en el notebook train_test_models  
  
## Modelos
Simple Bayesian Model:  
<img src="/otros/imagenes/output.svg" alt="Diagrama simple bayesian model" width="600" height="600">



## Orden de los archivos y carpetas
```
proyecto-remociones-en-masa/  
│  
├── data/  
│   ├── eda.ipynb  
│   ├── raw/  
│   │   ├── New_DB.xlsx  
│   ├── processed/  
│       ├── X_slope_PP_factor08.csv
│       ├── X_slope_PP_factor09.csv  
│       ├── X_slope_PP_vhs1_factor08.csv
│       ├── X_slope_PP_vhs1_factor09.csv  
│  
├── modelos/  
│   ├── trace_bayesian.pkl (bayesian_model_w_factors)  
│   ├── trace_cov.pkl (simple_bayesian_model)  
│   ├── trace_nn.pkl (neural_network)  
│  
├── otro/  
│   ├── fundamentos_red_bayesiana.pdf  
│   ├── bibliografia.txt  
│   ├── imagenes/  
│  
├── remocion_de_masas.yml  
├── README.md  
├── train_test_models.ipynb  
├── utils.py  
```