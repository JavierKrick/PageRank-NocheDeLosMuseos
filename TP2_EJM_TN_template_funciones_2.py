import numpy as np
import scipy

def calcula_L(A):
    grados = np.sum(A, axis=1)
    K = np.diag(grados)
    L = K - A
    return L

def calcula_R(A):
    grados = np.sum(A, axis=1)
    E = np.sum(A) #contabiliza 2m ya que es simetrica y vale 1 tanto para (i,j) como (j,i)
    P = np.outer(grados, grados) / E
    R = A - P
    return R

def calcula_lambda(L,v):
    # Recibe L y v y retorna el corte asociado
    s = np.sign(v)
    s[s == 0] = 1
    lambdon = s.T @ L @ s / 4
    return lambdon

def calcula_Q(R,v):
    # La funcion recibe R y s y retorna la modularidad (a menos de un factor 2E)
    s = np.sign(v)
    s[s == 0] = 1
    Q = s.T @ R @ s / 2
    return Q

def metpot1(A,tol=1e-8,maxrep=np.inf):
   # Recibe una matriz A y calcula su autovalor de mayor módulo, con un error relativo menor a tol y-o haciendo como mucho maxrep repeticiones

   
   n = A.shape[0]
   v = np.random.uniform(-1, 1, size=n) # Generamos un vector de partida aleatorio, entre -1 y 1
   v = v / np.linalg.norm(v, 2) # Lo normalizamos
   v1 = A@v # Aplicamos la matriz una vez
   v1 = v1 / np.linalg.norm(v1, 2) # Lo normalizamos
   l = np.dot(v,A@v)  # Calculamos el autovalor estimado
   l1 = np.dot(v1,A@v1) # Y el estimado en el siguiente paso
   nrep = 0 # Contador
   while np.abs(l1-l)/np.abs(l) > tol and nrep < maxrep: # Si estamos por debajo de la tolerancia buscada 
      v = v1 # actualizamos v y repetimos
      l = l1
      v1 = A@v # Calculo nuevo v1
      v1 = v1 / np.linalg.norm(v1, 2) # Normalizo
      l1 = np.dot(v1,A@v1)  # Calculo autovalor
      nrep += 1 # Un pasito mas
   if not nrep < maxrep:
      print('MaxRep alcanzado')
   l = np.dot(v,A@v) # Calculamos el autovalor
   return v,l,nrep<maxrep

def deflaciona(A,tol=1e-8,maxrep=np.inf):
    # Recibe la matriz A, una tolerancia para el método de la potencia, y un número máximo de repeticiones
    v1,l1,_ = metpot1(A,tol,maxrep) # Buscamos primer autovector con método de la potencia
    P_v1 = np.outer(v1,v1) # v1 ya sale normalizado de metpot1
    deflA = A - l1*P_v1 # Sugerencia, usar la funcion outer de numpy
    return deflA

##### Hasta aca tengo hecho ########
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

def metpotI(A,mu,tol=1e-8,maxrep=np.inf):
    # Retorna el primer autovalor de la inversa de A + mu * I, junto a su autovector y si el método convergió.
    n = A.shape[0]
    Imu = np.eye(n)*mu
    M = A + Imu
    Minv = calcular_Inversa(M)
    return metpot1(Minv,tol=tol,maxrep=maxrep)


def metpotI2(A,mu,tol=1e-8,maxrep=np.inf):
   # Recibe la matriz A, y un valor mu y retorna el segundo autovalor y autovector de la matriz A, 
   # suponiendo que sus autovalores son positivos excepto por el menor que es igual a 0
   # Retorna el segundo autovector, su autovalor, y si el metodo llegó a converger.
   n = A.shape[0]
   I = np.eye(n)
   X = A + I*mu # Calculamos la matriz A shifteada en mu
   iX = calcular_Inversa(X)
   defliX = deflaciona(iX) # La deflacionamos
   v,l,_ =  metpot1(defliX) # Buscamos su segundo autovector
   l = 1/l # Reobtenemos el autovalor correcto 
   #########   ES NECESARIO? iX ya tiene 1 / lambda  #########
   l -= mu
   return v,l,_


def laplaciano_iterativo(A,niveles,nombres_s=None):
    # Recibe una matriz A, una cantidad de niveles sobre los que hacer cortes, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.
    # La función debe, recursivamente, ir realizando cortes y reduciendo en 1 el número de niveles hasta llegar a 0 y retornar.
    if nombres_s is None: # Si no se proveyeron nombres, los asignamos poniendo del 0 al N-1
        nombres_s = range(A.shape[0])
    if A.shape[0] == 1 or niveles == 0: # Si llegamos al último paso, retornamos los nombres en una lista
        return([nombres_s])
    else: # Sino:
        L = calcula_L(A) # Recalculamos el L
        v,l,_ = metpotI2(L, 0.1) # Encontramos el segundo autovector de L
        # Recortamos A en dos partes, la que está asociada a el signo positivo de v y la que está asociada al negativo
        nodos_positivos = v >= 0  
        nodos_negativos = v < 0  
        Ap = A[np.ix_(nodos_positivos, nodos_positivos)] # Asociado al signo positivo
        Am = A[np.ix_(nodos_negativos, nodos_negativos)] # Asociado al signo negativo
        
        return(
                laplaciano_iterativo(Ap,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi>=0]) +
                laplaciano_iterativo(Am,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi<0])
                )        

def modularidad_iterativo(A=None,R=None,nombres_s=None):
    # Recibe una matriz A, una matriz R de modularidad, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.

    if A is None and R is None:
        print('Dame una matriz')
        return(np.nan)
    if R is None:
        R = calcula_R(A)
    if nombres_s is None:
        nombres_s = range(R.shape[0])
    # Acá empieza lo bueno
    if R.shape[0] == 1: # Si llegamos al último nivel
        return([nombres_s])
    else:
        v,l,_ = metpot1(R) # Primer autovector y autovalor de R
        # Modularidad Actual:
        Q0 = np.sum(R[v>0,:][:,v>0]) + np.sum(R[v<0,:][:,v<0])
        if Q0<=0 or all(v>0) or all(v<0): # Si la modularidad actual es menor a cero, o no se propone una partición, terminamos
            return(nombres_s)
        else:
            ## Hacemos como con L, pero usando directamente R para poder mantener siempre la misma matriz de modularidad
            nodos_positivos = v >= 0  
            nodos_negativos = v < 0  
            Rp = R[np.ix_(nodos_positivos, nodos_positivos)] # Parte de R asociada a los valores positivos de v
            Rm = R[np.ix_(nodos_negativos, nodos_negativos)] # Parte asociada a los valores negativos de v
            vp,lp,_ = metpot1(Rp)  # autovector principal de Rp
            vm,lm,_ = metpot1(Rm) # autovector principal de Rm
        
            # Calculamos el cambio en Q que se produciría al hacer esta partición
            Q1 = 0
            if not all(vp>0) or all(vp<0):
               Q1 = np.sum(Rp[vp>0,:][:,vp>0]) + np.sum(Rp[vp<0,:][:,vp<0])
            if not all(vm>0) or all(vm<0):
                Q1 += np.sum(Rm[vm>0,:][:,vm>0]) + np.sum(Rm[vm<0,:][:,vm<0])
            if Q0 >= Q1: # Si al partir obtuvimos un Q menor, devolvemos la última partición que hicimos
                return([[ni for ni,vi in zip(nombres_s,v) if vi>0],[ni for ni,vi in zip(nombres_s,v) if vi<0]])
            else:
                # Sino, repetimos para los subniveles  
                #tenemos que raelizar llamadas recursivas
                # y se concatenan los resultados
                return(modularidad_iterativo(R=Rp, nombres_s=[ni for ni, vi in zip(nombres_s, v) if vi >= 0]) + 
                       modularidad_iterativo(R=Rm, nombres_s=[ni for ni, vi in zip(nombres_s, v) if vi < 0]))
            
if __name__ == "__main__":
    
    A = np.array([
    [0, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 1, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1],
    [0, 0, 1, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 0]
])
    

