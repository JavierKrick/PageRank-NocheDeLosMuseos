# Carga de paquetes necesarios para graficar
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Para leer archivos
import geopandas as gpd # Para hacer cosas geográficas
import seaborn as sns # Para hacer plots lindos
import networkx as nx # Construcción de la red en NetworkX
import scipy

import matplotlib.patches as mpatche

#funciones.py

def construye_adyacencia(D,m):
    # Función que construye la matriz de adyacencia del grafo de museos
    # D matriz de distancias, m cantidad de links por nodo
    # Retorna la matriz de adyacencia como un numpy.
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # recorriendo las filas, anexamos vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # En realidad, elegimos todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int) # Convertimos a entero
    np.fill_diagonal(A,0) # Borramos diagonal para eliminar autolinks
    return(A)

def calculaLU(matriz):
    n = matriz.shape[0]
    U = matriz.copy()
    L = np.eye(n)

    for col in range(n-1):
        for fil in range(col+1, n):
            factor = U[fil][col] / U[col][col]
            L[fil][col] = factor
            U[fil] -= U[col] * factor

    return L, U




def calcula_matriz_C(A):
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C
    nodosSalientes = np.sum(A, axis=1)
    Kinv = np.diag(1 / nodosSalientes)   # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de A
    C = A.T @ Kinv  # Calcula C multiplicando Kinv y A
    return C


def calcula_pagerank(A,alfa):
    # Función para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # d: coeficientes de damping
    # Retorna: Un vector p con los coeficientes de page rank de cada museo

    C = calcula_matriz_C(A)
    N = A.shape[0] # Obtenemos el número de museos N a partir de la estructura de la matriz A
    I = np.eye(N)
    M = (N / alfa) * (I - (1 - alfa) * C)
    L, U = calculaLU(M) # Calculamos descomposición LU a partir de C y d
    b = np.ones(N) # Vector de 1s, multiplicado por el coeficiente correspondiente usando d y N.
    Up = scipy.linalg.solve_triangular(L,b,lower=True) # Primera inversión usando L
    p = scipy.linalg.solve_triangular(U,Up) # Segunda inversión usando U
    return p



def calcula_matriz_C_continua(D):
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C en versión continua
    D = D.copy()
    np.fill_diagonal(D,1) # relleno con 1 la diagonal  para poder hacer 1/D
    F = 1/D
    np.fill_diagonal(F,0)
    n=F[0].size # Obtengo el tamaño para K
    K=np.eye(n) #Creo la matriz K
    Kinv = np.eye(n) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de F

    for x in range(0,n,1):
        K[x][x]=np.sum(F[x]) # Sumo las filas de F (aprovecho que F[x][x] = 0)
        Kinv[x][x]=1/K[x][x] #Como K es una matriz diagonal, le inversa de K es el recíproco de los elementos de K

    C = F.T @ Kinv  # Correción del Profe
    return C

def calcula_B(C,cantidad_de_visitas):
    # Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matirz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    B = np.eye(C.shape[0])
    for i in range(cantidad_de_visitas):   ###### for i in range(cantidad_de_visitas-1): es el original, lo cambié porque no se cumplía la sumatoria
        # Sumamos las matrices de transición para cada cantidad de pasos
        if i>0:
            B=B+potencia_de_matrices(C,i)

    return B

# M : Matriz
# e: Exponente>=0
def potencia_de_matrices(M,e:int):
    res=M.copy()
    if e==0:
        res=(np.eye(M.shape[0]))
        return res
    elif e==1:
        return res
    while e>1:
        res=res@M
        e-=1
    return res

# B: Matriz de la ecuacion 5
# w:vector con número total de visitantes
def calcula_Pto_de_entrada(B,w):
    L, U = calculaLU(B) # Calculamos descomposición LU a partir de C y d
    Up = scipy.linalg.solve_triangular(L,w,lower=True) # Primera inversión usando L
    v = scipy.linalg.solve_triangular(U,Up) # Segunda inversión usando U
    return v

# Recibe un vector y calcula norma 1
def norma_Uno(v):
    res=0
    for x in range(0,len(v),1):
        res=res+abs(v[x])
    return res


#Inversa de una matriz
def calcular_Inversa(M):
    #Obtengo dimensión de la matriz original
    n=M.shape[0]
    #creo matriz de 0
    res=np.zeros((n,n))
    I=np.eye(n)
    L, U= calculaLU(M)

    for x in range(0,len(I),1):
        w=I[:,x]
        Up = scipy.linalg.solve_triangular(L,w,lower=True) # Primera inversión usando L
        v = scipy.linalg.solve_triangular(U,Up)
        res[:,x]=v

    return res

def norma_Uno_Matriz(M):
	#Acá tenemos que buscar la columna cuya suma de valores absolutos de sus elementos es más grande
    n=M.shape[0]
    #creo matriz de 0
    base_res=np.zeros((n))
    for x in range(0,len(M),1):
        v=M[:,x] #v: vector de M
        v=abs(v) #paso cada columna a valor absoluto
        base_res[x]=np.sum(v) # sumo por columna


    res=base_res.max()  #nos quedamos con la columna que dió mayor suma (de valores absolutos de sus elementos)

    return res

# Recibe una matriz invertible, calcula inversa y norma matricial
def calcular_condicion(M):
    inv=M.copy()
    A=norma_Uno_Matriz(M)     # Calculamos la norma 1 de la matriz M
    inv=calcular_Inversa(inv)
    A_inv=norma_Uno_Matriz(inv) #la norma 1 de la matriz inversa
    # κ₁(M) = ||M||₁ * ||M⁻¹||₁
    return A*A_inv
