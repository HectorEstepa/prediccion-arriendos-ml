import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import networkx as nx
import joblib
import json
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

#----------- Cargar Datos ------------------
dt = pl.read_csv("DatosRevisadosLimpios.csv")
data = pl.DataFrame(dt)

#----------- Filtro para acotar los datos ------------------
data = data.filter(
    (pl.col("Precio ($)") >= 100000) &
    (pl.col("Precio ($)") <= 2000000) &
    (pl.col("Metros (m²)") >= 15) &
    (pl.col("Metros (m²)") <= 200)
)

#----------- Crear columnas para todas las comunas ------------------
data = data.to_dummies(columns=["Comuna"]) 
comuna_cols = [col for col in data.columns if col.startswith("Comuna_")]

#----------- SELECCIÓN MEJORADA DE VARIABLES ------------------
# Verificar correlaciones para seleccionar mejores features
correlation_matrix = data.select([
    "Precio ($)", "Gastos_Comunes ($)", "Metros (m²)", 
    "Habitaciones", "Baños"
]).corr()

print("Matriz de correlación:")
print(correlation_matrix)

#----------- Preparar datos ------------------
xx = data.select([
    "Gastos_Comunes ($)",
    "Metros (m²)", 
    "Habitaciones",
    "Baños",
    *comuna_cols
]).to_numpy()

yy = data.select(pl.col("Precio ($)")).to_numpy().ravel()

#----------- Separar datos ------------------
x_train, x_test, y_train, y_test = train_test_split(xx, yy, test_size=0.2, random_state=42, stratify=None)

#----------- Escalado MEJORADO ------------------
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#----------- OPCIÓN 1: OPTIMIZACIÓN DE RED NEURONAL CON GRID SEARCH ------------------
def entrenar_red_neuronal_optimizada(x_train, y_train):
    """Entrena una red neuronal con búsqueda de mejores parámetros"""
    
    # Definir parámetros para búsqueda
    param_grid = {
        'hidden_layer_sizes': [(50, 30, 20), (60, 40), (100, 50), (80, 40, 20)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.001, 0.0001, 0.00001],  # Regularización L2
        'learning_rate_init': [0.001, 0.0005],
        'max_iter': [2000, 3000],
        'early_stopping': [True],
        'n_iter_no_change': [50]
    }
    
    modelo = MLPRegressor(random_state=42, solver='adam')
    
    # Búsqueda grid con validación cruzada
    grid_search = GridSearchCV(
        modelo, param_grid, cv=5, scoring='r2', 
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(x_train, y_train)
    
    print(f"\nMejores parámetros: {grid_search.best_params_}")
    print(f"Mejor score CV: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

#----------- OPCIÓN 2: RED NEURONAL MANUAL MEJORADA ------------------
def crear_red_neuronal_avanzada():
    """Crea una red neuronal con arquitectura mejorada"""
    
    modelo = MLPRegressor(
        hidden_layer_sizes=(100, 80, 40),  # Más capas y neuronas
        activation='relu',
        solver='adam',
        alpha=0.0001,           # Regularización para evitar overfitting
        learning_rate_init=0.001,
        max_iter=3000,
        early_stopping=True,    # Parada temprana
        n_iter_no_change=50,
        random_state=42,
        verbose=True
    )
    return modelo

#----------- OPCIÓN 3: ENSAMBLE CON RANDOM FOREST ------------------
def crear_ensamble(x_train, y_train, x_test, y_test):
    """Crea un modelo ensemble combinando redes neuronales y random forest"""
    
    # Red neuronal
    nn_model = MLPRegressor(
        hidden_layer_sizes=(80, 40, 20),
        activation='relu',
        alpha=0.0001,
        max_iter=2000,
        early_stopping=True,
        random_state=42
    )
    
    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    # Entrenar ambos modelos
    nn_model.fit(x_train_scaled, y_train)
    rf_model.fit(x_train, y_train)  # RF no necesita escalado
    
    # Predicciones
    nn_pred = nn_model.predict(x_test_scaled)
    rf_pred = rf_model.predict(x_test)
    
    # Combinar predicciones (promedio ponderado)
    nn_r2 = r2_score(y_test, nn_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    
    # Pesos basados en performance
    total_r2 = nn_r2 + rf_r2
    nn_weight = nn_r2 / total_r2
    rf_weight = rf_r2 / total_r2
    
    print(f"\nPesos del ensemble - NN: {nn_weight:.3f}, RF: {rf_weight:.3f}")
    
    return nn_model, rf_model, nn_weight, rf_weight

#----------- ENTRENAR MODELO SELECCIONADO ------------------
print("Entrenando modelo optimizado...")

# Probar diferentes opciones:
opcion = 2  # Cambiar entre 1, 2, 3 para probar diferentes enfoques

if opcion == 1:
    modeloUsar = entrenar_red_neuronal_optimizada(x_train_scaled, y_train)
elif opcion == 2:
    modeloUsar = crear_red_neuronal_avanzada()
    modeloUsar.fit(x_train_scaled, y_train)
else:
    nn_model, rf_model, nn_weight, rf_weight = crear_ensamble(x_train, y_train, x_test, y_test)
    modeloUsar = nn_model  # Usaremos solo la red neuronal para predicciones simples

#----------- EVALUACIÓN MEJORADA ------------------
if opcion != 3:
    predic = modeloUsar.predict(x_test_scaled)
else:
    # Para ensemble
    nn_pred = nn_model.predict(x_test_scaled)
    rf_pred = rf_model.predict(x_test)
    predic = (nn_pred * nn_weight) + (rf_pred * rf_weight)

r2 = r2_score(y_test, predic)
mse = mean_squared_error(y_test, predic)
mae = mean_absolute_error(y_test, predic)
rmse = np.sqrt(mse)

print(f"\n=== MÉTRICAS FINALES ===")
print(f"R² -> {r2:.4f}")
print(f"MSE -> {mse:,.0f}")
print(f"MAE -> {mae:,.0f}")
print(f"RMSE -> {rmse:,.0f}")

#----------- ANÁLISIS DE RESIDUALES ------------------
residuales = y_test - predic
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(y_test, predic, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("Precio Real")
plt.ylabel("Precio Predicho")
plt.title(f"Predicciones vs Real (R² = {r2:.3f})")

plt.subplot(1, 3, 2)
plt.scatter(predic, residuales, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicciones")
plt.ylabel("Residuales")
plt.title("Análisis de Residuales")

plt.subplot(1, 3, 3)
plt.hist(residuales, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel("Error de Predicción")
plt.ylabel("Frecuencia")
plt.title("Distribución de Errores")

plt.tight_layout()
plt.show()

#----------- GUARDAR MODELO ENTRENADO ------------------
def guardar_modelo(modelo, scaler, comuna_cols, metricas):
    """Guarda el modelo entrenado y metadata"""
    
    # Guardar modelo con joblib (más eficiente para scikit-learn)
    joblib.dump({
        'model': modelo,
        'scaler': scaler,
        'comuna_columns': comuna_cols,
        'metrics': metricas
    }, 'modelo_entrenado.pkl')
    
    # Guardar metadata en JSON
    metadata = {
        'r2_score': float(metricas['r2']),
        'mse': float(metricas['mse']),
        'mae': float(metricas['mae']),
        'rmse': float(metricas['rmse']),
        'comuna_columns': comuna_cols,
        'feature_names': ['Gastos_Comunes ($)', 'Metros (m²)', 'Habitaciones', 'Baños'] + comuna_cols,
        'model_type': str(type(modelo).__name__),
        'training_date': np.datetime64('today').astype(str)
    }
    
    with open('modelo_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Modelo guardado exitosamente:")
    print("   - modelo_entrenado.pkl (modelo y preprocesador)")
    print("   - modelo_metadata.json (información del modelo)")

# Guardar el modelo
metricas = {
    'r2': r2,
    'mse': mse,
    'mae': mae,
    'rmse': rmse
}

guardar_modelo(modeloUsar, scaler, comuna_cols, metricas)

def graficar_red_neuronal(modelo, nombre_archivo="arquitectura_red.png"):
    """
    Genera una imagen PNG que muestra la arquitectura del modelo MLPRegressor entrenado.
    """
    if not hasattr(modelo, "coefs_"):
        print("⚠️ El modelo aún no ha sido entrenado. Entrena antes de graficar.")
        return
    
    # Obtener la estructura de la red
    capas = [modelo.coefs_[0].shape[0]] + [c.shape[1] for c in modelo.coefs_]
    print(f"Estructura detectada: {capas}")

    G = nx.DiGraph()
    posiciones = {}
    y_offset = 0
    capa_anterior = None

    for i, num_neuronas in enumerate(capas):
        x_offset = np.linspace(-1, 1, num_neuronas)
        for j, x in enumerate(x_offset):
            nodo = f"C{i}_N{j}"
            G.add_node(nodo)
            posiciones[nodo] = (x, -y_offset)

            if capa_anterior:
                for prev_nodo in capa_anterior:
                    G.add_edge(prev_nodo, nodo)

        capa_anterior = [f"C{i}_N{j}" for j in range(num_neuronas)]
        y_offset += 1

    # Dibujar
    plt.figure(figsize=(10, 6))
    nx.draw(
        G, posiciones, 
        with_labels=False,
        node_size=400, 
        node_color="skyblue", 
        edge_color="gray",
        arrows=False
    )
    plt.title("Arquitectura de la Red Neuronal (MLPRegressor)")
    plt.axis("off")

    # Guardar PNG
    plt.savefig(nombre_archivo, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Imagen guardada: {nombre_archivo}")

#----------- FUNCIÓN PARA CARGAR Y USAR MODELO GUARDADO ------------------
def cargar_y_predecir(gastos_comun, metrosC, habitaciones, baños, comuna):
    """Carga el modelo guardado y hace predicciones"""
    
    try:
        # Cargar modelo
        modelo_cargado = joblib.load('modelo_entrenado.pkl')
        modelo = modelo_cargado['model']
        scaler = modelo_cargado['scaler']
        comuna_cols = modelo_cargado['comuna_columns']
        
        # Preparar datos de comuna
        comuna_dummies = [1 if col == f"Comuna_{comuna}" else 0 for col in comuna_cols]
        
        if sum(comuna_dummies) == 0:
            print(f"⚠️  Comuna '{comuna}' no encontrada. Usando valores base.")
        
        # Crear array de predicción
        pred_data = [gastos_comun, metrosC, habitaciones, baños] + comuna_dummies
        pred_data_scaled = scaler.transform([pred_data])
        
        # Predecir
        prediccion = modelo.predict(pred_data_scaled)
        
        return prediccion[0]
        
    except FileNotFoundError:
        print("❌ Modelo no encontrado. Ejecuta el entrenamiento primero.")
        return None






#----------- INFORMACIÓN ADICIONAL ------------------
print(f"\n💡 Información del modelo:")
print(f"   - R² alcanzado: {r2:.4f}")
print(f"   - Error promedio: ±${mae:,.0f}")
if r2 >= 0.90:
    print("   ✅ Excelente poder predictivo")
elif r2 >= 0.85:
    print("   👍 Buen poder predictivo")
else:
    print("   🔄 Considera ajustar más el modelo")



