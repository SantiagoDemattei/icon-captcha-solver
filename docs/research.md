# Registro de investigación

Caminos que se probaron, se midieron y se descartaron. Ya no están en el código: cada entrada
cuenta qué se probó, el número medido, por qué se descartó y qué habría que agregar sobre el código
actual para volver a probarlo.

Nota metodológica que vale para todo lo que sigue: el set de validación es chico (61 challenges,
~160-180 crops), así que toda comparación se hizo con al menos 3-5 corridas por configuración.
"Mejor val_acc" es el máximo sobre 200 evaluaciones y está inflado por selección; por eso, cuando
existe, la métrica que decide es la tasa de challenges resueltos del solver completo
(`scripts/evaluate_solver.py`, todo o nada, igual que `/icon/verify`).

## Síntesis de crops y entrenamiento en dos etapas

**Qué se probó.** Generar crops sintéticos a partir de la forma limpia de cada ícono de la leyenda
(`dataset/synthesize.py`) y sumarlos al entrenamiento. Cuatro configuraciones:

1. rotación + estiramiento + warp elástico agresivo, 510 reales : 2100 sintéticos;
2. la misma geometría a 510 : 280;
3. geometría corregida (solo rotación + escala uniforme) a 633 : 700;
4. corrupción calibrada contra tasas medidas en los crops reales (pérdida de resolución, fusión de
   regiones, ruido de borde) y entrenamiento en dos etapas: pretrain sintético y fine-tune solo real
   (`train_two_stage()`), a 633 : 700.

**Resultado.** Las cuatro empeoraron la val_acc sobre crops reales: 20.6%, 27.5%, 23.8% y 36.8%,
siempre por debajo del baseline solo-real equivalente de cada momento, y con train_acc subiendo
mucho más rápido que lo normal en las tres primeras.

**Por qué se descartó.** La cuarta prueba descarta que alcance con calibrar mejor el nivel de
corrupción o con separar las fases: el problema parece ser del propio dominio sintético (textura,
borde, estructura del ruido), no de cuánto se corrompe. Después la segmentación por relleno mejoró
tanto los crops reales que la motivación original (pocos datos limpios) dejó de existir.

**Cómo volver a probarlo.** Generar crops a partir de las siluetas de `legend_templates/` (rotación
y escala uniforme, más la corrupción que se quiera simular), agregarlos al manifest con
`source: "synthetic"`, mandarlos siempre a entrenamiento y comparar con `scripts/evaluate_solver.py`
contra el mismo dataset sin ellos.

## Balanceo de clases

**Qué se probó.** `--balance loss` (cross-entropy ponderada por la inversa de la frecuencia de cada
clase) y `--balance sampler` (`WeightedRandomSampler` con muestreo balanceado).

**Resultado.** Medido con la segmentación por color cuantizado, 5 corridas cada uno: `loss` queda
dentro del ruido (accuracy balanceada +2 puntos, accuracy global −4); `sampler` es peor en todo.

**Por qué se descartó.** No mejora la métrica que importa y agrega una opción más a cada corrida.

**Cómo volver a probarlo.** En `icon_solver/training.py`: pesos por clase en `nn.CrossEntropyLoss`
(inversa de la frecuencia) o un `WeightedRandomSampler` en el `DataLoader` de positivos.

## TTA por rotaciones

**Qué se probó.** Promediar las probabilidades del clasificador sobre varias rotaciones de la
entrada (en polar, un `roll` exacto del eje angular): `predict_proba(model, x, rotations)`.

**Resultado.** 83.1% contra 82.4% de val_acc sin TTA, dentro del ruido.

**Por qué se descartó.** Con padding circular en el eje angular la red ya es casi invariante a la
rotación por construcción; promediar rotaciones no agrega información.

**Cómo volver a probarlo.** En `IconModel.class_probs`, promediar el softmax sobre
`torch.roll(x, k * 48 // n, dims=2)` para `k` en `0..n-1`: en polar, cada roll es una rotación exacta.

## Selección de la región verdadera por blob más cercano

**Qué se probó.** Segmentar todo el fondo por color cuantizado (`segment_blobs()`, cajas fijas de 16
por canal + componentes conexas) y tomar como región verdadera de cada click el blob que lo
contiene con menor área, o si ninguno lo contiene, el de centroide más cercano (`_nearest_blob()`).
Se probaron además `color_tolerance=8` (divergencia media 26.2% contra 23.2% con 16) y apertura
morfológica antes de las componentes (23.6% y 23.9%): ambas peor.

**Resultado.** La cuantización parte el relleno del ícono cuando el ruido cruza un borde de caja y
fusiona vecinos de color parecido: solo 11% de los crops era limpio. Con el relleno desde el pixel
del click (`selection="flood"`) la val_acc pasó de ~36% a ~74% sobre el mismo dataset.

**Por qué se descartó.** El relleno sembrado en el click es estrictamente mejor, y además es el
mismo algoritmo que genera los candidatos en inferencia (ver ADR 0002).

**Cómo volver a probarlo.** En `derive_labels` (`icon_solver/dataset/build.py`), reemplazar
`truth_region` por el blob de `segment_blobs()` que contiene el click con menor área.

`segment_blobs()` sigue existiendo en el código vivo, pero solo dentro del etiquetador automático
por HTTP (`scripts/collect_dataset.py`), que la usa junto con el ranking por Hu-moments. Ese camino
tiene un rendimiento medido de 0/5 y se conserva como alternativa documentada, no como vía principal.

## Filtro de crops por similitud con la silueta de la leyenda

**Qué se probó.** Guardar en el manifest la similitud entre cada crop y la silueta de su ícono en la
leyenda (`template_similarity`) y descartar al entrenar los crops por debajo de un umbral
(`--min-template-similarity`).

**Resultado.** Con umbral 0.5 el clasificador solo mejora mucho (66.7% → 74.3% de challenges
resueltos sin plantilla), pero el solver completo queda igual: 84.7% contra 85.8% (barrido del peso
de la plantilla entre 0.5 y 3: 83.1-85.8%).

**Por qué se descartó.** El score por silueta de leyenda del solver ya compensa exactamente esos
errores; filtrar no mueve la métrica final. Se retiró también la clave `template_similarity` del
manifest, que solo existía para este filtro.

**Cómo volver a probarlo.** Guardar en cada entrada del manifest la similitud de su crop con la
silueta de su ícono (`shapes.silhouette_similarity`) y filtrar los positivos por umbral en
`icon_solver/training.py`.

## Ablaciones de la representación y del entrenamiento

Polar, padding circular y schedule coseno quedaron fijos en el código. Estas son las alternativas
que perdieron (volver a probarlas es cambiar la codificación en `shapes.py` o la red en `model.py`
y reentrenar):

- **Cartesiano en vez de polar.** 5 corridas cada uno sobre 633 crops: cartesiano 31.6% de media
  (30.2-32.5) contra polar 39.7% (38.1-42.1), rangos sin superposición. En polar la rotación es un
  shift circular exacto de un eje, que una CNN aprende con muchos menos datos.
- **Padding con ceros en vez de circular en el eje angular.** 81.2% → 82.7% de val_acc con circular,
  y menos varianza entre corridas: con padding circular la convolución es equivariante a la rotación
  exacta y el pooling global la vuelve invariante.
- **Learning rate constante en vez de schedule coseno.** El pico de val_acc no cambia (82.4%), pero
  la media de los últimos 20 epochs pasa de ~72% a ~80%: el modelo final ya no depende de pescar el
  mejor epoch, y por eso se guarda el último.

## Otros descartes que no dejaban código

Registrados en su momento y sin código que retirar: crops con agujeros (máscara real del relleno en
vez del contorno externo relleno; el solver con plantilla empeora de 74.3% a 67.2%), silueta de
leyenda con agujeros (top-1 de 78.0% a 47.7%) y mover el click al "punto interior más lejano del
borde" en íconos cóncavos (56.8% contra 85.8% clickeando el centroide).

## Refactor de arquitectura (2026-09-29)

No es un experimento sino una reorganización del código (paquete único `icon_solver`, segmentación y codificación de forma en un solo lugar, modelo autocontenido `models/icon_model.pt`), hecha contra una referencia de caracterización capturada antes de empezar: los puntos de click del solver para los 318 challenges, los candidatos, las regiones verdaderas, el encoding, el build del dataset y las métricas salen idénticos.

Para confirmar que tampoco cambió el resultado estadístico del entrenamiento, se reentrenó 3 veces con el código nuevo (200 epochs, sobre el mismo `data/processed/`, sin semilla) y se evaluó cada modelo con `scripts/evaluate_solver.py`:

| Corrida | Challenges resueltos | Íconos | Candidato correcto presente |
|---|---|---|---|
| 1 | 91.8% (56/61) | 97.3% | 99.5% |
| 2 | 88.5% (54/61) | 96.2% | 99.5% |
| 3 | 90.2% (55/61) | 96.7% | 99.5% |
| **Media** | **90.2%** | **96.7%** | **99.5%** |

Rango histórico antes del refactor: 85.2-93.4% de challenges resueltos, media 90.7% en 3 corridas. El modelo desplegado sigue siendo el convertido (93.4%, el mejor de sus 3 corridas originales).

Tiempos en CPU (AMD, 12 hilos lógicos, torch con 6 hilos): un `solve` medio sobre validación pasó de 0.269-0.274 s a 0.262 s, y `evaluate_solver` de 19.7-20.3 s a 16.5-16.7 s (el recall de candidatos ahora reutiliza los candidatos del propio solver en vez de volver a segmentar el fondo). Cada corrida de entrenamiento tarda ~4 minutos en una RTX 4070 Super.

`scripts/train.py --seed N` hace el entrenamiento reproducible en CPU (pesos idénticos entre corridas con la misma semilla). En CUDA no lo es: con la misma semilla los pesos difieren por el no determinismo de cuDNN.
