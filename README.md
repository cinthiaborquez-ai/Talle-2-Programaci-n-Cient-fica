Taller 2 Programación Científica
Integrantes: Cinthia Bórquez y Martina Torres
Fecha de entrega: 29 de Junio de 2026
Descripción: Sistema de análisis computacional de texto sobre el corpus bíblico (King James Version), desarrollado en Python con orientación a objetos. El proyecto aplica técnicas de procesamiento de lenguaje natural, representación vectorial TF-IDF, análisis estadístico y modelos probabilísticos para explorar patrones lingüísticos, similitud semántica y generación automática de texto.

Estructura del proyecto:
taller2_bible/
├── data/
│   ├── t_kjv.csv               
│   └── key_english.csv         
├── notebooks/
│   └── analisis.ipynb         
├── outputs/
│   ├── viz1_longitud_versiculos.png
│   ├── viz2_top_palabras.png
│   ├── viz3_versiculos_por_libro.png
│   ├── viz4_heatmap_similitud.png
│   ├── viz5_pca_versiculos.png
│   ├── viz6_confusion_matrix.png
│   ├── viz7_sentimiento_libros.png
│   └── viz8_sentimiento_capitulos.png
├── src/
└── README.md

Requisitos e instalación
1. Clonar el repositorio
git clone https://github.com/tuusuario/taller2_bible.git
cd taller2_bible
2. Crear entorno visual
python -m venv .venv
.venv\Scripts\activate        
source .venv/bin/activate
3. Instalar dependencias
pip install pandas numpy matplotlib seaborn nltk scikit-learn textblob networkx
4. Descargar recursos NLTK
import nltk
nltk.download('stopwords')

Ejecución
1. Abre VS Code y carga la carpeta taller2_bible.
2. Abre el archivo notebooks/analisis.ipynb.
3. Selecciona el kernel Python del entorno virtual .venv.
4. Ejecuta todas las celdas en orden con Run All.

Clases implementadas
1. ProcesadorTexto: 
Pipeline de preprocesamiento: limpieza, tokenización, eliminación de stopwords, vocabulario y frecuencias.
2. VectorizadorTFIDF:
Implementación manual de TF-IDF sin librerías externas.
3. MotorBusquedaSemantica:
Motor de búsqueda basado en TF-IDF + similitud del coseno implementada manualmente.
4. ClasificadorVersiculos:
Clasificador de versículos por libro usando Regresión Logística.
5. ModeloNgrama:
Modelos de lenguaje estadísticos: unigrama, bigrama, trigrama y n-grama.
6. AnalizadorSentimiento
Análisis de sentimiento por versículo, capítulo y libro usando TextBlob.
