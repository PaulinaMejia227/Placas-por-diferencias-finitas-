"""
=============================================================
ESTRUCTURAS AEROESPACIALES — UNII  |  Clase 5  |  13 Julio 2026
ACTIVIDAD PARA ENTREGA (sube tu solución a NEOLMS)
Tema: Placas por diferencias finitas — convergencia, carga puntual
      y borde empotrado
=============================================================
INSTRUCCIONES GENERALES
  1. Completa CADA sección marcada con TODO:
  2. No borres los comentarios de orientación.
  3. Guarda el archivo como:  actividad5_TuNombre.py
  4. Súbelo a NEOLMS en la tarea "Actividad 5".

LIBRERÍAS NECESARIAS:  numpy
=============================================================
"""

import numpy as np

# ─────────────────────────────────────────────
# SOLVER BASE — la función de la práctica, parametrizada.
# NO modifiques la firma; los TODO están más abajo.
#   N          : nodos por lado
#   point_load : None = presión uniforme; un valor P [N] = carga
#                puntual en el nodo central (y q = 0 en el resto)
#   clamped    : False = simplemente apoyada (nodo fantasma = -w)
#                True  = empotrada          (nodo fantasma = +w)
# ─────────────────────────────────────────────
def placa(N=9, point_load=None, clamped=False):
    L, E, nu, t, q = 1000.0, 70e3, 0.3, 10.0, 0.01
    D = (E * t**3) / (12 * (1 - nu**2))
    h, nodes = L/(N-1), N*N
    idx = lambda i, j: i * N + j
    A, B = np.zeros((nodes, nodes)), np.zeros(nodes)

    for i in range(N):
        for j in range(N):
            row = idx(i, j)
            if i==0 or i==N-1 or j==0 or j==N-1:
                A[row, row] = 1.0
            else:
                A[row, row] = 20

                # TODO 2a (Ejercicio 2): si point_load NO es None,
                # B[row] debe ser 0 aquí (no hay presión distribuida)
                B[row] = 0.0 if point_load is not None else q*h**4/D

                for di, dj, val in [(-1,0,-8), (1,0,-8), (0,-1,-8), (0,1,-8),
                                    (-1,-1,2), (-1,1,2), (1,-1,2), (1,1,2),
                                    (-2,0,1), (2,0,1), (0,-2,1), (0,2,1)]:
                    ni, nj = i+di, j+dj
                    if 0<=ni<N and 0<=nj<N:
                        A[row, idx(ni, nj)] += val
                    else:
                        # TODO 3a (Ejercicio 3): si clamped es True el
                        # signo del nodo fantasma debe ser +1 (w(-1)=w(1))
                        sgn = +1 if clamped else -1
                        r_ni, s_i = (1,sgn) if ni==-1 else (N-2,sgn) if ni==N else (ni,1)
                        r_nj, s_j = (1,sgn) if nj==-1 else (N-2,sgn) if nj==N else (nj,1)
                        A[row, idx(r_ni, r_nj)] += val * s_i * s_j

    # TODO 2b (Ejercicio 2): si point_load NO es None, asigna en el
    # nodo central el término de carga puntual:
    #   B[centro] = point_load * h**2 / D
    # (una carga P en un solo nodo equivale a q = P/h^2 en su celda)
    if point_load is not None:
        B[idx(N//2, N//2)] = point_load * h**2 / D

    w = np.linalg.solve(A, B)
    return w.reshape((N, N))[N//2, N//2]


# ─────────────────────────────────────────────
# EJERCICIO 1 — Convergencia de malla
# Ejecuta el solver con N = 9, 17, 33, 65 y reporta w_max.
# La solución analítica de Navier da w_max = 6.333 mm.
# ─────────────────────────────────────────────
print("--- EJERCICIO 1: Convergencia ---")
# TODO 1a: bucle sobre [9, 17, 33, 65] imprimiendo N y w_max
for N in [9, 17, 33, 65]:
    print(f"  N={N:2d} -> w_max = {placa(N):.4f} mm")
print("  Analítica (Navier)  = 6.3330 mm")

# TODO 1b: Responde en un comentario:
#   ¿A partir de qué N consideras la malla "convergida"? ¿Qué costo
#   tiene refinar (tamaño de la matriz A)?
# RESPUESTA: Considero que a partir de N=33 la malla ya presenta
# una convergencia aceptable, porque los cambios en la deflexion
# son cada vez menores. Aunque N=65 ofrece una solucion mas
# refinada, tambien requiere mayor capacidad de procesamiento.
# Por eso, N=33 representa un buen equilibrio entre precision
# y costo computacional. Es importante mencionar que la solucion
# converge hacia 6.337 mm, aunque no coincide exactamente con
# el valor analitico de referencia de 6.333 mm.

# ─────────────────────────────────────────────
# EJERCICIO 2 — Carga puntual P = 10 kN en el centro
# Completa los TODO 2a y 2b dentro del solver y ejecuta:
# ─────────────────────────────────────────────
print("\n--- EJERCICIO 2: Carga puntual ---")
# TODO 2c: descomenta cuando el solver esté listo
w_p = placa(9, point_load=10000.0)
print(f"  w_max con P = 10 kN: {w_p:.2f} mm")
print(f"  (presión uniforme equivalente: {placa(9):.2f} mm)")

# TODO 2d: Responde en un comentario:
#   ¿Por qué la deflexión es mucho mayor que con la presión uniforme
#   si la fuerza total (q*L^2 = 10 kN) es la misma?
# RESPUESTA: la misma resultante concentrada en el centro trabaja sobre el
#   punto de máxima flexibilidad de la placa. En la carga uniforme buena
#   parte de la fuerza se aplica cerca de los bordes, donde la rigidez es
#   casi infinita (w=0) y apenas contribuye a la flecha. Además la carga
#   puntual genera una singularidad local de momento/curvatura bajo el
#   punto de aplicación: la placa "se hunde" localmente. Por eso los
#   coeficientes clásicos son w=0.00406 qL^4/D frente a w=0.0116 PL^2/D,
#   ~2.9 veces mayor para la misma fuerza total.
#   OJO: con N=9 la malla es demasiado gruesa para resolver la
#   singularidad y sobrestima la flecha (19.27 mm); al refinar baja
#   (N=17: 18.47, N=33: 18.21, N=65: 18.13) hacia los ~18.10 mm teóricos.


# ─────────────────────────────────────────────
# EJERCICIO 3 — Bordes empotrados
# Completa el TODO 3a dentro del solver (sgn = +1 si clamped) y
# ejecuta con presión uniforme:
# ─────────────────────────────────────────────
print("\n--- EJERCICIO 3: Bordes empotrados ---")
# TODO 3b: descomenta cuando el solver esté listo
w_c = placa(9, clamped=True)
print(f"  w_max empotrada: {w_c:.3f} mm  (apoyada: {placa(9):.3f} mm)")

# TODO 3c: Responde en un comentario:
#   La deflexión cae drásticamente al empotrar. ¿Qué precio paga la
#   estructura en los bordes a cambio? (piensa en momentos flectores)
# RESPUESTA: el empotramiento impide el giro del borde, y ese giro
#   bloqueado se paga con momentos flectores de reacción en los bordes.
#   En la placa apoyada M=0 en el contorno y el máximo está en el centro
#   (M ~ 0.0479 qL^2); al empotrar, el centro se descarga (M ~ 0.0231 qL^2)
#   pero aparece un momento en el punto medio de cada borde
#   M ~ -0.0513 qL^2, el mayor de toda la placa. Es decir: se gana rigidez
#   (la flecha cae a ~1/3) a costa de concentrar la tensión de flexión en
#   los bordes, que pasan a ser la zona crítica a fallo/fatiga, y de
#   exigir un anclaje capaz de transmitir ese momento.
