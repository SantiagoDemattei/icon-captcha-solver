# El mismo relleno genera los crops de entrenamiento y los candidatos de inferencia

Las regiones verdaderas del entrenamiento salen de un relleno de rango fijo sembrado en el punto de click humano. En inferencia no hay click, así que los candidatos se generan sembrando exactamente ese mismo relleno (mismas tolerancias, misma área máxima, misma codificación a crop polar) sobre una grilla del fondo, en vez de usar una segmentación distinta como la de color cuantizado.

Si el modelo se entrena con un algoritmo de segmentación y se evalúa con otro, la accuracy offline deja de reflejar lo que pasa al resolver: el dominio de inferencia no es el de entrenamiento. Por eso los parámetros de segmentación y la codificación de forma viven en un único lugar que usan ambos caminos, y cambiar cualquiera de ellos obliga a reconstruir el dataset y reentrenar.
