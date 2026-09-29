# visual-captcha-solver

Resolver el captcha de íconos de BotDeflector a partir de sus dos imágenes: ubicar en el fondo cada ícono que pide la leyenda y devolver dónde clickearlo.

## El captcha

**Challenge** (*challenge*):
Un par leyenda + fondo servido por el vendor, de un solo uso: una respuesta equivocada lo consume sin decir qué punto falló.
_Avoid_: captcha (para una instancia), imagen

**Leyenda** (*legend*):
La imagen chica con los íconos pedidos, en el orden en que hay que clickearlos.

**Fondo** (*background*):
La imagen grande donde aparecen los íconos pedidos, rotados y escalados, entre formas de fondo y señuelos.

**Señuelo** (*decoy*):
Un ícono de la librería presente en el fondo que no es ninguno de los pedidos.

**Punto de click** (*click point*):
La coordenada sobre el fondo que se envía como respuesta para un ícono pedido; el vendor la acepta si cae cerca del centro del ícono.
_Avoid_: solución (para un punto suelto)

## Etiquetas

**Challenge etiquetado** (*labeled challenge*):
Un challenge guardado junto con los puntos de click que el vendor aceptó.

**Etiquetador** (*labeler*):
Quién produjo los puntos de click de un challenge etiquetado: un humano o el modelo.

**Set de validación** (*validation set*):
Los challenges etiquetados por humanos que nunca se usan para entrenar.

## Reconocimiento

**Clase de ícono** (*icon class*):
Uno de los íconos distintos de la librería del vendor, identificado por un número estable.
_Avoid_: tipo de ícono, categoría

**Prototipo** (*prototype*):
Una máscara de ícono de leyenda que define a qué clase de ícono pertenece un ícono de leyenda nuevo; una clase puede tener varios, uno por tamaño de render.
_Avoid_: template, plantilla canónica

**Silueta de leyenda** (*legend silhouette*):
La forma rellena de un ícono de la leyenda de ese mismo challenge, usada para comparar contra las regiones del fondo.
_Avoid_: template, plantilla

**Región** (*region*):
Una forma segmentada del fondo.
_Avoid_: blob, componente

**Región verdadera** (*truth region*):
La región del fondo que corresponde a un punto de click aceptado.

**Candidato** (*candidate*):
Una región del fondo propuesta como posible ícono pedido al resolver un challenge.

**Crop** (*crop*):
La máscara de forma de una región, normalizada en tamaño, con la que se entrena el clasificador.

**Negativo** (*negative*):
Un crop de un candidato que con certeza no es ninguna de las clases pedidas en su challenge.

**Modelo** (*icon model*):
El artefacto entrenado que usa el solver: los pesos del clasificador junto con los prototipos con los que se entrenó, guardados juntos para que nunca se desincronicen.
_Avoid_: checkpoint (para el artefacto completo)

**Solver** (*solver*):
Lo que, dado un challenge, devuelve un punto de click por ícono de la leyenda.

**Veredicto** (*verdict*):
La respuesta del vendor a los puntos de click enviados para un challenge: verificado o rechazado, sin detalle por punto.
