# Clasificación de conjunto cerrado, no embeddings

El plan original era un modelo siamés que comparara la forma del ícono de la leyenda con cada región del fondo. Al etiquetar datos reales la librería de íconos del vendor resultó chica y estable (20 íconos distintos, cada uno renderizado a unos pocos tamaños fijos), así que el problema se trata como clasificación de conjunto cerrado: un clasificador chico sobre la máscara de forma de cada región, más una salida extra "fondo", y la clase de cada ícono de la leyenda se asigna por prototipo más cercano.

Con pocos cientos de challenges etiquetados, aprender una clase por ícono necesita muchos menos datos que aprender una métrica de similitud general. La contrapartida es que un ícono nuevo del vendor requiere agregar la clase y reentrenar; la comparación directa con la silueta de la leyenda del propio challenge, que el solver suma al score, cubre en parte ese caso sin entrenar nada.
