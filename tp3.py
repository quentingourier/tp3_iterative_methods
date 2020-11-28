from math import *
from random import *
import numpy as np
import matplotlib.pyplot as pp
import time

#----------------------------------------------------------------------------
#-----------DEFINITIONS---------------------------------------------------
#----------------------------------------------------------------------------

def genererA(n):
    A = np.random.random_sample((n, n))
    '''print("\nA=\n", A)'''
    return A

def MIGenerale ( M , N , b , x0 , epsilon , Nitermax ):
    erreur = 1
    x = x0
    compteur = 0
    A = M - N
    while erreur > epsilon and compteur < Nitermax :
        x_new = np.dot(np.linalg.inv(M), np.dot(N,x)) + np.dot(np.linalg.inv(M), b )
        erreur = np.linalg.norm(np.dot(A, x_new)-b)
        x = x_new
        compteur += 1
    return x_new , compteur , erreur

def MIJacobi ( A , b , x0 , epsilon , Nitermax):
    M = np.diag(np.diag(A))
    N = M - A
    print("\nDécomposition de Jacobi\n")
    print("M =\n", M)
    print("N =\n", N)
    return MIGenerale(M, N, b, x0, epsilon, Nitermax)

def MIGaussSeidel ( A , b , x0 , epsilon , Nitermax):
    M = np.tril(A)
    N = M - A
    print("\nDécomposition de Gauss-Seidel\n")
    print("M =\n", M)
    print("N =\n", N)
    return MIGenerale(M, N, b, x0, epsilon, Nitermax)

def decompo_Relaxation(A, w):
    D = np.diag(np.diag(A))
    E = D - np.tril(A)
    F = D - E - A
    M = (1/w)*D - E
    N = ((1/w)-1)*D + F #M-A
    return M, N

def MIRelaxation (A , b , w , x0 , epsilon , Nitermax):
    D = np.diag(np.diag(A))
    E = D - np.tril(A)
    F = D - E - A
    M = (1/w)*D - E
    N = ((1/w)-1)*D + F #M-A
    print("\nDécomposition de la relaxation\n")
    print("M =\n", M)
    print("N =\n", N)
    return MIGenerale(M, N, b, x0, epsilon, Nitermax)

def converge(M, N):
    B = np.linalg.inv(M) @ N #matrice d'itérat
    vp = np.linalg.eigvals(B)
    ray_spec = max(np.absolute(vp)) #faire attention à ça pour les complexes 
    return ray_spec

def verif_conv(A):
    #initialisation
    '''A = genererA(3)'''
    print("\nA=\n", A)
    #par jacobi
    M = np.diag(np.diag(A))
    N = M - A
    ray_spec = converge(M, N)
    print("\np(J) = ", ray_spec)
    #par gauss-seidel
    M = np.tril(A)
    N = M - A
    ray_spec = converge(M, N)
    print("\np(G) = ", ray_spec)

    
def w_etude(n):
    A = np.array([[1, 2, -2], [1, 1, 1], [2, 2, 1]])
    b = np.random.random_sample((n, 1))
    b = np.asarray(b).reshape(-1)#remettre b sous forme de vecteur
    print(b.shape)
    w = np.linspace(1, 2, 10)
    x0 = np.zeros(b.shape)
    epsilon = 10**-5
    Nitermax = 100
    liste_compteur = []
    for i in range(len(w)):
        x_new , compteur , erreur = MIRelaxation (A , b , w[i] , x0 , epsilon , Nitermax)
        liste_compteur.append(compteur)
    print(liste_compteur)
    print(w)
    pp.plot(w, liste_compteur, label = "nombre d'itération pour w donné", color = 'g')
    pp.xlim(1, 2)
    pp.xlabel('w')
    pp.ylabel("nb d'iteration(s)")
    pp.title("Nombre d'itération(s) nécessaires en fonction du paramètre w")
    pp.legend()
    pp.show()

def courbe_iter_w(liste_w, liste_nbiter):
    pp.plot(liste_w, liste_nbiter, label = 'par MI_Relaxation')
    pp.xlim(0, 2)
    pp.xlabel('paramètre w')
    pp.ylabel("nb iter")
    pp.title("nb iter = f(w)")
    pp.legend()
    pp.show()

def courbe_temps_w(liste_w, liste_tps):
    pp.plot(liste_w, liste_tps, label = 'par MI_Relaxation')
    pp.xlim(0, 2)
    pp.xlabel('paramètre w')
    pp.ylabel("temps (s)")
    pp.title("temps = f(w)")
    pp.legend()
    pp.show()
    
    
#----------------------------------------------------------------------------
#-----------PROGRAMME---------------------------------------------------
#----------------------------------------------------------------------------
    

#question 1
#voir la def

'''
A = np.array([[10, 1], [-1, 10]])
b = np.array([1, 2])
x0 = np.zeros(b.shape)
w = 1.5
epsilon = 10**-5
Nitermax = 1000

print("\nx = ", MIJacobi ( A , b , x0 , epsilon , Nitermax))
print("\nx = ", MIGaussSeidel ( A , b , x0 , epsilon , Nitermax), "\n")
print("\nx = ", MIRelaxation ( A , b, w, x0 , epsilon , Nitermax), "\n")


#question 4
w = 1.5
print(MIRelaxation ( A , b , w , x0 , epsilon , Nitermax))


# Partie 2 Expérimentation des méthodes

# question 1
A = np.zeros((100,100))
b = np.zeros((100, 1))
for i in range (0,100):
    for j in range (0,100):
        if i == j :
            A[i,i] = 3
            b[i,0] = cos((i+1)/8)
        else:
            A[i,j] = 1/(12+(3*(i+1)-5*(j+1))**2)

verif_conv(A)   
b = np.asarray(b).reshape(-1)#remettre b sous forme de vecteur
x0 = np.zeros(b.shape)
Nitermax = 100
liste_epsilon = [10**-1, 10**-2, 10**-3, 10**-4, 10**-5, 10**-6, 10**-7, 10**-8, 10**-9, 10**-10, 10**-11, 10**-12, 10**-13]
liste_nb_iteration_J = []
liste_nb_iteration_GS = []

for i in range(len (liste_epsilon)):
    x, nbiter, err = MIJacobi(A, b, x0, liste_epsilon[i], Nitermax)
    liste_nb_iteration_J.append(nbiter)
    x, nbiter, err = MIGaussSeidel(A, b, x0, liste_epsilon[i], Nitermax)
    liste_nb_iteration_GS.append(nbiter)
print(liste_nb_iteration_J)
print(liste_nb_iteration_GS)

pp.gca().invert_xaxis()
pp.plot(liste_epsilon, liste_nb_iteration_J, label = 'Méthode Jacobi')
pp.plot(liste_epsilon,liste_nb_iteration_GS, label ='Méthode de Gauss-Seidel')
pp.xlim(10**-13 ,0.1)
pp.xscale('log')
pp.xlabel('précision souhaitée')
pp.ylabel("nombre d'itérations nécessaires")
pp.title("Nombre d'itération nécessaires pour plusieurs précisions souhaitées")
pp.legend()
pp.show()

'''
# question 2 
A = np.zeros((100,100))
b = np.zeros((100,1))
for i in range (0,100):
    for j in range (0,100):
        if i == j:
            A[i, j] = 3
            b[i,0] = cos((i+1)/8)
        else:
            A[i,j] = 1/(1+3*abs((i+1)-(j+1)))
verif_conv(A)
x0 = np.zeros((100,1))
liste_epsilon = [10**-1, 10**-2, 10**-3, 10**-4, 10**-5, 10**-6, 10**-7, 10**-8, 10**-9, 10**-10, 10**-11, 10**-12, 10**-13]
Nitermax = 100
liste_nb_iteration_J = []
liste_nb_iteration_GS = []
for i in range (0, len (liste_epsilon)):
    x , nbiter , err = MIJacobi(A, b, x0, liste_epsilon[i], Nitermax)
    liste_nb_iteration_J.append(nbiter)
    x , nbiter , err = MIGaussSeidel(A, b, x0, liste_epsilon[i], Nitermax)
    liste_nb_iteration_GS.append(nbiter)
pp.plot(liste_epsilon,liste_nb_iteration_J,label = 'Méthode Jacobi')
pp.plot(liste_epsilon,liste_nb_iteration_GS,label ='Méthode de Gauss-Seidel')
pp.xlim(10**-13 ,0.1)
pp.xscale('log')
pp.xlabel('précision souhaitée')
pp.ylabel("nombre d'itérations nécessaires")
pp.title("Nombre d'itération nécessaires pour plusieurs précisions souhaitées")
pp.legend()
pp.show()

'''
#question 3
n= 100
A = np.zeros((n,n)) #SYSTEME 1
b = np.zeros((n, 1))
for i in range (0,n):
    for j in range (0,n):
        if i == j :
            A[i,i] = 3
            b[i,0] = cos((i+1)/8)
        else:
            A[i,j] = 1/(12+(3*(i+1)-5*(j+1))**2)   
b = np.asarray(b).reshape(-1)#remettre b sous forme de vecteur

x0 = np.zeros(b.shape)
epsilon = 10**-5
Nitermax = 1000
liste_w = np.linspace(0.1, 1.9, 19)
liste_nbiter = []
liste_tps = []
for i in range(len (liste_w)):
    t0 = time.time()
    x, nbiter, err = MIRelaxation (A , b , liste_w[i] , x0 , epsilon , Nitermax)
    t1 = time.time()
    liste_tps.append(t1-t0)
    liste_nbiter.append(nbiter)
courbe_iter_w(liste_w, liste_nbiter)
courbe_temps_w(liste_w, liste_tps)
'''








