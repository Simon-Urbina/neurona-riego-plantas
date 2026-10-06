import numpy as np

print("Neurona artificial para decidir si una planta necesita riego")

# Humedad del suelo (%) y Temperatura ambiental (°C)
X = np.array([
    [80, 18], [70, 22], [65, 28], [55, 25], [50, 32],
    [40, 30], [35, 25], [30, 32], [20, 35], [10, 38]
], dtype=float)

# 1 = regar, 0 = no regar
y = np.array([
    [0], [0], [0], [0], [0],
    [1], [1], [1], [1], [1]
], dtype=float)

# La humedad se encuentra aproximadamente entre 0 y 100.
# La temperatura se encuentra aproximadamente entre 0 y 50.
# Dividimos cada columna por su valor máximo esperado para dejar los datos en una escala cercana a 0 y 1.
escala = np.array([100, 50])
X_normalizado = X / escala

print("\nDatos normalizados:")
print(X_normalizado)

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

# Devuelve la probabilidad de regar para cada fila de entradas normalizadas
def predecir(entradas_normalizadas, pesos, sesgo):
    return sigmoide(entradas_normalizadas @ pesos + sesgo)

# Entrenamiento de la neurona y devuelve pesos, sesgo y error final
def entrenar(tasa_aprendizaje, epocas, mostrar=False):
    rng = np.random.default_rng(7)
    pesos = rng.normal(size=(2, 1))  # uno para humedad, otro para temperatura
    sesgo = 0.0

    for epoca in range(epocas):
        # Suma ponderada y activación
        z = X_normalizado @ pesos + sesgo
        predicciones = sigmoide(z)

        # Error cuadrático medio
        error = predicciones - y
        gradiente_z = 2 * error * predicciones * (1 - predicciones) / len(X)
        gradiente_pesos = X_normalizado.T @ gradiente_z
        gradiente_sesgo = np.sum(gradiente_z)

        pesos -= tasa_aprendizaje * gradiente_pesos
        sesgo -= tasa_aprendizaje * gradiente_sesgo

        if mostrar and epoca % 2000 == 0:
            perdida = np.mean(error ** 2)
            print(f"Epoca {epoca:5} | error: {perdida:.4f}")

    error_final = np.mean((predecir(X_normalizado, pesos, sesgo) - y) ** 2)
    return pesos, sesgo, error_final


# 1. Entrenamiento básico
tasa_aprendizaje = 0.5
epocas = 10000

print(f"\n1. Entrenamiento base con (tasa={tasa_aprendizaje}, epocas={epocas})")
pesos, sesgo, error_final = entrenar(tasa_aprendizaje, epocas, mostrar=True)

print(f"\nPeso de la humedad: {pesos[0, 0]:.4f}")
print(f"Peso de la temperatura: {pesos[1, 0]:.4f}")
print(f"Sesgo: {sesgo:.4f}")
print(f"Error final (ECM): {error_final:.6f}")

probabilidades = predecir(X_normalizado, pesos, sesgo)
respuestas = (probabilidades >= 0.5).astype(int)

print("\nResultados con los datos de entrenamiento:")
for entrada, probabilidad, respuesta in zip(X, probabilidades.ravel(), respuestas.ravel()):
    print(f"Humedad {entrada[0]}% | Temp {entrada[1]}°C -> probabilidad {probabilidad:.3f} -> respuesta: {respuesta}")


# 2. Pruebas con condiciones nuevas
X_nuevos = np.array([
    [75, 30], [45, 34], [25, 22], [50, 25], [30, 40]
], dtype=float)
X_nuevos_normalizado = X_nuevos / escala

print("\n2. Pruebas con condiciones nuevas:")
prob_nuevos = predecir(X_nuevos_normalizado, pesos, sesgo)
resp_nuevos = (prob_nuevos >= 0.5).astype(int)

for entrada, probabilidad, respuesta in zip(X_nuevos, prob_nuevos.ravel(), resp_nuevos.ravel()):
    if respuesta == 1:
        decision = "Regar"
    else:
        decision = "No regar"
    print(f"Humedad {entrada[0]}% | Temp {entrada[1]}°C -> probabilidad {probabilidad:.3f} -> decisión: {respuesta} ({decision})")


# 3. Experimentos con épocas y tasas de aprendizaje
experimentos = [
    ("Prueba base", 10000, 0.5),
    ("Pocas épocas", 100, 0.5),
    ("Cantidad intermedia", 1000, 0.5),
    ("Más épocas", 20000, 0.5),
    ("Tasa pequeña", 10000, 0.01),
    ("Tasa moderada", 10000, 0.1),
    ("Tasa alta", 10000, 1.0),
    ("Tasa muy alta", 10000, 2.0),
]

caso_prueba = np.array([[45, 34]], dtype=float) / escala

print("\n3. Experimentos con los parámetros")
print(f"{'Experimento':<20} | {'Épocas':>6} | {'Tasa':>5} | {'Error final':>11.6} | {'Correctas':>6} | {'P(45%, 34°C)':>13}")
for nombre, exp_epocas, tasa in experimentos:
    pred, ses, err = entrenar(tasa, exp_epocas)
    correctas = int(np.sum((predecir(X_normalizado, pred, ses) >= 0.5).astype(int) == y.astype(int)))
    prob_caso = predecir(caso_prueba, pred, ses)[0, 0]
    print(f"{nombre:<20} | {exp_epocas:>6} | {tasa:>5} | {err:>11.6} | {correctas:>6}/10 | {prob_caso:>12.3}")


# 4. Prueba adicional con el umbral (sin volver a entrenar)
print("\n4. Prueba con distintos umbrales (mismos pesos del entrenamiento base)")
todos_X = np.vstack([X, X_nuevos])
todas_prob = np.vstack([probabilidades, prob_nuevos]).ravel()

print(f"{'Humedad':>7} | {'Temp':>4} | {'Prob':>5} | u=0.4 | u=0.5 | u=0.6")
for entrada, prob in zip(todos_X, todas_prob):
    respues = [int(prob >= u) for u in (0.4, 0.5, 0.6)]
    marca = "  <- cambia" if len(set(respues)) > 1 else ""
    print(f"{entrada[0]:>5.0f} % | {entrada[1]:>2.0f}°C | {prob:.3f} | "
          f"{respues[0]:^5} | {respues[1]:^5} | {respues[2]:^5}{marca}")