# El set de validación solo contiene challenges etiquetados por humanos

Además de las etiquetas humanas, el modelo desplegado etiqueta challenges nuevos resolviéndolos en vivo y guardando los que `/icon/verify` acepta. Esos challenges van siempre a entrenamiento, nunca a validación: la validación se elige por hash del id del challenge, pero solo entre los etiquetados por humanos.

Un challenge etiquetado por el modelo existe solo porque el modelo ya lo resolvió bien. Ponerlo en validación la llenaría de casos fáciles (sesgo de selección) y dejaría de detectar errores. Las métricas en vivo sin sesgo salen del registro de todos los intentos, incluidos los rechazados.
