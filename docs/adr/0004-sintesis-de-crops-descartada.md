# Síntesis de crops descartada

Se probó en cuatro configuraciones generar crops sintéticos a partir de la forma limpia de la leyenda (incluida una corrupción calibrada contra los crops reales y un entrenamiento en dos etapas). Las cuatro empeoraron la accuracy sobre crops reales, así que el dataset se arma solo con challenges reales etiquetados y la síntesis se retiró del código.

El problema parece ser el propio dominio sintético, no el nivel de corrupción. Si en algún momento falta volumen, la vía elegida es etiquetar más challenges reales (a mano o con el modelo desplegado resolviendo en vivo), no volver a sintetizar. El detalle de las cuatro pruebas y cómo reproducirlas está en `docs/research.md`.
