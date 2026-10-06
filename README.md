# Neurona artificial para el riego de plantas

Neurona artificial de una sola unidad, con activación **sigmoide**, que decide si una planta debe regarse a partir de dos variables: la **humedad del suelo** y la **temperatura ambiental**. Está desarrollada en Python con NumPy y se gestiona con `uv`.

> Los datos son didácticos y no representan una recomendación agronómica para una especie real.

## Descripción del problema

| Variable | Significado | Rango aproximado |
|---|---|---|
| `x1` | Humedad del suelo (%) | 0 – 100 |
| `x2` | Temperatura ambiental (°C) | 0 – 50 |
| `y`  | Decisión: `1` = regar, `0` = no regar | {0, 1} |

### Datos de entrenamiento

| Caso | Humedad | Temperatura | Salida |
|:---:|:---:|:---:|:---:|
| 1 | 80 % | 18 °C | 0 |
| 2 | 70 % | 22 °C | 0 |
| 3 | 65 % | 28 °C | 0 |
| 4 | 55 % | 25 °C | 0 |
| 5 | 50 % | 32 °C | 0 |
| 6 | 40 % | 30 °C | 1 |
| 7 | 35 % | 25 °C | 1 |
| 8 | 30 % | 32 °C | 1 |
| 9 | 20 % | 35 °C | 1 |
| 10 | 10 % | 38 °C | 1 |

### Funcionamiento de la neurona

1. **Normalización:** cada columna se divide por su valor máximo esperado, `escala = [100, 50]`, y así ambas entradas quedan cerca del intervalo [0, 1].
2. **Suma ponderada:** `z = X_normalizado @ pesos + sesgo` (dos pesos y un sesgo).
3. **Activación:** `sigmoide(z) = 1 / (1 + e^(-z))` produce una probabilidad entre 0 y 1.
4. **Aprendizaje:** descenso por gradiente sobre el error cuadrático medio (ECM).
   - `gradiente_z = 2 · (p − y) · p · (1 − p) / n`
   - `gradiente_pesos = X_normalizado.T @ gradiente_z`
   - `gradiente_sesgo = Σ gradiente_z`
5. **Decisión binaria:** `respuesta = (probabilidad >= 0.5).astype(int)`.

Los pesos iniciales se generan con una semilla fija (`np.random.default_rng(7)`), así que los resultados se pueden reproducir.

## Instrucciones de ejecución

Requisitos: [uv](https://docs.astral.sh/uv/) y Python ≥ 3.12.

```bash
git clone <url-del-repositorio>
cd neurona-riego-plantas
uv sync
cd src/neurona_riego_plantas
uv run main.py
```

El programa muestra, en orden:

1. Los datos normalizados.
2. El entrenamiento base, con el ECM cada 2000 épocas, los pesos, el sesgo y la probabilidad de cada caso.
3. Las predicciones para las cinco condiciones nuevas.
4. La tabla de experimentos con épocas y tasas de aprendizaje.
5. La comparación de umbrales 0.4, 0.5 y 0.6.

## Resultados

### 1. Entrenamiento base (`tasa_aprendizaje = 0.5`, `epocas = 10000`)

Evolución del error:

| Época | ECM |
|---:|---:|
| 0 | 0.2468 |
| 2000 | 0.0497 |
| 4000 | 0.0350 |
| 6000 | 0.0275 |
| 8000 | 0.0229 |
| 10000 (final) | **0.019645** |

Parámetros aprendidos:

| Parámetro | Valor |
|---|---:|
| Peso de la humedad | **−19.5032** |
| Peso de la temperatura | **+2.3101** |
| Sesgo | **+7.4022** |
| Error final (ECM) | **0.019645** |

Probabilidad calculada para cada caso de entrenamiento:

| Caso | Humedad | Temperatura | Probabilidad | Respuesta | Esperada |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 80 % | 18 °C | 0.001 | 0 | 0 |
| 2 | 70 % | 22 °C | 0.005 | 0 | 0 |
| 3 | 65 % | 28 °C | 0.018 | 0 | 0 |
| 4 | 55 % | 25 °C | 0.103 | 0 | 0 |
| 5 | 50 % | 32 °C | 0.295 | 0 | 0 |
| 6 | 40 % | 30 °C | 0.728 | 1 | 1 |
| 7 | 35 % | 25 °C | 0.850 | 1 | 1 |
| 8 | 30 % | 32 °C | 0.954 | 1 | 1 |
| 9 | 20 % | 35 °C | 0.994 | 1 | 1 |
| 10 | 10 % | 38 °C | 0.999 | 1 | 1 |

La neurona clasifica correctamente **10 de 10** casos.

#### Significado del signo de cada peso

- **Humedad (−19.50, negativo):** cuando la humedad del suelo sube, `z` baja y la probabilidad de regar disminuye. Un suelo húmedo no necesita agua. El valor absoluto es grande, así que la humedad es el factor que más pesa en la decisión.
- **Temperatura (+2.31, positivo):** cuando la temperatura sube, la probabilidad de regar aumenta un poco, porque el calor seca el suelo. El valor es unas 8 veces menor que el de la humedad, así que la temperatura solo inclina la decisión en los casos límite.
- **Sesgo (+7.40):** desplaza la frontera de decisión. Sin él, la frontera tendría que pasar por el origen de las entradas normalizadas. Con estos valores, la frontera (`z = 0`) queda entre el **42 % y el 47 % de humedad**, según la temperatura (18 °C y 38 °C).

### 2. Pruebas con condiciones nuevas

Estos datos se normalizaron con la **misma** `escala = [100, 50]` usada en el entrenamiento.

| Humedad | Temperatura | Probabilidad | Decisión |
|:---:|:---:|:---:|:---:|
| 75 % | 30 °C | 0.003 | 0 — No regar |
| 45 % | 34 °C | 0.549 | 1 — Regar |
| 25 % | 22 °C | 0.972 | 1 — Regar |
| 50 % | 25 °C | 0.232 | 0 — No regar |
| 30 % | 40 °C | 0.968 | 1 — Regar |

Interpretación:

- **75 % / 30 °C:** el suelo está muy húmedo y, aunque hace calor, no hace falta regar.
- **45 % / 34 °C:** es el caso más dudoso (0.549). La humedad está en la frontera y la temperatura alta la empuja apenas hacia "regar".
- **25 % / 22 °C:** el suelo está seco. Aunque la temperatura es baja, la humedad domina y hay que regar.
- **50 % / 25 °C:** tiene una humedad parecida al caso 5 del entrenamiento (salida 0) y una temperatura menor, así que la probabilidad baja más. No se riega.
- **30 % / 40 °C:** el suelo está seco y hace mucho calor, así que hay que regar.

### 3. Experimentos con los parámetros

En cada prueba solo cambia el parámetro indicado. Todas parten de los mismos pesos iniciales (semilla 7).

| Experimento | Épocas | Tasa | Error final (ECM) | Correctas | P(45 %, 34 °C) | Aprendizaje |
|---|---:|---:|---:|:---:|:---:|---|
| Prueba base | 10000 | 0.5 | 0.019645 | 10/10 | 0.549 | Rápido y estable |
| Pocas épocas | 100 | 0.5 | 0.165623 | 10/10 | 0.522 | Insuficiente |
| Cantidad intermedia | 1000 | 0.5 | 0.067481 | 10/10 | 0.579 | Aceptable, aún sin terminar |
| Más épocas | 20000 | 0.5 | 0.011705 | 10/10 | 0.526 | Estable, mejora marginal |
| Tasa pequeña | 10000 | 0.01 | 0.130048 | 10/10 | 0.539 | Lento |
| Tasa moderada | 10000 | 0.1 | 0.049728 | 10/10 | 0.585 | Algo lento |
| Tasa alta | 10000 | 1.0 | 0.011705 | 10/10 | 0.526 | Rápido y estable |
| Tasa muy alta | 10000 | 2.0 | 0.006566 | 10/10 | 0.508 | Muy rápido, sin inestabilidad |

Para clasificar la velocidad de aprendizaje se registró el ECM en épocas intermedias de cada tasa:

| Tasa | Época 100 | Época 1000 | Época 10000 | ¿El error siempre baja? |
|---:|---:|---:|---:|:---:|
| 0.01 | 0.2435 | 0.2226 | 0.1301 | Sí |
| 0.1 | 0.2226 | 0.1300 | 0.0497 | Sí |
| 0.5 | 0.1656 | 0.0675 | 0.0196 | Sí |
| 1.0 | 0.1299 | 0.0497 | 0.0117 | Sí |
| 2.0 | 0.0983 | 0.0349 | 0.0066 | Sí |

En el descenso por gradiente, el avance depende del producto `tasa × épocas`. Por eso "tasa 0.1 con 10000 épocas" llega al mismo error (0.0497) que "tasa 0.5 con 2000 épocas" o "tasa 1.0 con 1000 épocas". Del mismo modo, "Más épocas" (0.5 × 20000) y "Tasa alta" (1.0 × 10000) llegan al mismo error y a la misma probabilidad, 0.011705 y 0.526.

### 4. Prueba adicional con el umbral

Se usan los pesos del entrenamiento base y **no** se vuelve a entrenar.

| Humedad | Temp. | Prob. | u = 0.4 | u = 0.5 | u = 0.6 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 80 % | 18 °C | 0.001 | 0 | 0 | 0 |
| 70 % | 22 °C | 0.005 | 0 | 0 | 0 |
| 65 % | 28 °C | 0.018 | 0 | 0 | 0 |
| 55 % | 25 °C | 0.103 | 0 | 0 | 0 |
| 50 % | 32 °C | 0.295 | 0 | 0 | 0 |
| 40 % | 30 °C | 0.728 | 1 | 1 | 1 |
| 35 % | 25 °C | 0.850 | 1 | 1 | 1 |
| 30 % | 32 °C | 0.954 | 1 | 1 | 1 |
| 20 % | 35 °C | 0.994 | 1 | 1 | 1 |
| 10 % | 38 °C | 0.999 | 1 | 1 | 1 |
| 75 % | 30 °C | 0.003 | 0 | 0 | 0 |
| **45 %** | **34 °C** | **0.549** | **1** | **1** | **0** ← cambia |
| 25 % | 22 °C | 0.972 | 1 | 1 | 1 |
| 50 % | 25 °C | 0.232 | 0 | 0 | 0 |
| 30 % | 40 °C | 0.968 | 1 | 1 | 1 |

Solo cambia el caso **45 % / 34 °C**, porque es el único con probabilidad entre 0.4 y 0.6. Los demás están lejos de 0.5, así que la neurona está muy segura de ellos. El umbral se aplica **después** de calcular la probabilidad: decide cómo leer la salida, pero no interviene en el cálculo de `z` ni en el descenso por gradiente. Por eso los pesos y el sesgo aprendidos no cambian. Un umbral más bajo (0.4) riega en más casos dudosos, y uno más alto (0.6) exige más certeza antes de regar.

## Estructura del proyecto

```
neurona-riego-plantas/
├── pyproject.toml
├── uv.lock
├── README.md
└── src/
    └── neurona_riego_plantas/
        ├── main.py                         # Desarrollo de la neurona
        └── README.md                       # Este documento
```
