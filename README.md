# Pruebas de vision sin carrito

Puedes usar videos del celular; no necesitas conectar el telefono como webcam.

1. Graba en horizontal, con la camara baja y apuntando hacia delante y un poco
   hacia el piso. Comienza con piso despejado en la parte inferior central.
2. Manteniendo la camara quieta, coloca un objeto al centro, luego a un lado y
   finalmente retiralo. Otro clip puede incluir cambios de iluminacion.
3. Copia el video a la laptop y abre `probar_video.cmd` con doble clic.
4. Selecciona el archivo. El video comienza pausado en el primer cuadro.
5. Haz clic en una ventana de OpenCV. Cuando el recuadro amarillo contenga
   solamente piso libre, pulsa **C** para tomar una muestra. Puedes usar **N**
   para buscar otro cuadro antes de calibrar.
6. Pulsa **Espacio** para reproducir o pausar, **N** para avanzar un cuadro,
   **R** para borrar las muestras y **Q** o **Esc** para cerrar.
   El deslizador de calibracion permite ajustar la tolerancia del color.

Al llegar al final conserva el ultimo cuadro y muestra FIN. Abre de nuevo el
acceso para repetirlo. Cada ejecucion comienza sin calibracion; no toma muestras
automaticamente.

## Uso desde una terminal con el entorno de Python activado

```powershell
python main.py --select-video
python main.py --video "C:\Videos\prueba_piso.mp4"
python main.py
python main.py --camera 0
```

Sin argumentos conserva la camara de la laptop. El acceso de doble clic busca
primero `.venv`, despues `venv` y luego el entorno existente en la carpeta
vecina `CBS_VISION_PROJECT/cbs-venv`; si no encuentra ninguno usa `python`.

## Que observar

- Sin calibracion debe mostrar STOP.
- Con piso libre suficiente en el centro puede proponer FORWARD.
- Con un objeto central, comprobar si propone girar hacia el lado libre.
- Con los tres sectores bloqueados debe proponer STOP.

Anota cuando no ocurra lo esperado. Los porcentajes representan pixeles que
parecen piso y no probabilidades de seguridad. La variable `depth` no mide metros.
El procesamiento conserva todos los cuadros: si la laptop tarda, el video se
reproduce mas lento. Los FPS mostrados describen lectura y deteccion del ultimo
cuadro nuevo, no una medicion del ciclo completo.

Este programa muestra propuestas de movimiento. No publica ordenes a ROS ni
controla motores. Estos cambios no incluyen despliegue en Jetson.
