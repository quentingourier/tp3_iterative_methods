#code par Abdelli Nicolas et Gourier Quentin
#AERO 3B & 3E

#----------------------------------------------------------------------------
#-----------IMPORTATIONS---------------------------------------------------
#----------------------------------------------------------------------------

from math import cos
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
    '''print("\nDécomposition de Jacobi\n")
    print("M =\n", M)
    print("N =\n", N)'''
    return MIGenerale(M, N, b, x0, epsilon, Nitermax)

def MIGaussSeidel ( A , b , x0 , epsilon , Nitermax):
    M = np.tril(A)
    N = M - A
    '''print("\nDécomposition de Gauss-Seidel\n")
    print("M =\n", M)
    print("N =\n", N)'''
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
    B = np.dot(np.linalg.inv(M), N) #matrice d'itérat
    print(B)
    vp = np.linalg.eigvals(B)
    ray_spec = np.abs(vp).max()
    print(ray_spec)
    return ray_spec

def verif_conv(A, w):
    #par relaxation
    D = np.diag(np.diag(A))
    E = D - np.tril(A)
    F = D - E - A
    B = np.dot(np.linalg.inv(D-w*E), ((1-w)*D+w*F))
    vp = np.linalg.eigvals(B)
    ray_spec = max(np.absolute(vp)) 
    #print(ray_spec)
    return ray_spec
    
def w_etude(A):
    w = np.linspace(0.1, 1.9, 10000)
    liste_vconv = []
    for i in range(len(w)):
        liste_vconv.append(verif_conv(A, w[i]))
    courbe_conv_x(w, liste_vconv)
    

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
    
def courbe_conv_x(liste_w, liste_vconv):
    pp.plot(liste_w, liste_vconv, label = "rayon de convergence pour w", color = 'g')
    pp.xlim(0.7, 1.5)
    pp.xlabel('w')
    pp.ylabel("rayon de convergence")
    pp.title("rayon de convergence en fonction du paramètre w")
    pp.legend()
    pp.show()

#----------FONCTIONS UTILES-----------------------------------------------------

def gauss(A, B):
    Aaug = np.concatenate((A, B), axis = 1)
    Baug = rg(Aaug)
    compteur = res(Baug)    
    return compteur

def rg(Aaug):
    n,m = np.shape(Aaug) #retourne les lignes et colonnes (la dimension)
    for k in range(0, n-1):
        for i in range(k+1, n):
            gik = Aaug[i, k]/Aaug[k,k]
            Aaug[i, :] = Aaug[i, :] - gik*Aaug[k, :]
    return Aaug

def res(Aaug):
    n, m = np.shape(Aaug)
    X = np.zeros(n)
    compteur = 0
    for i in range(n-1, -1, -1): 
        somme = 0
        for k in range(i,n):
            somme = somme + X[k]*Aaug[i,k]
        X[i] = (Aaug[i, n] - somme) / Aaug[i, i]
        compteur += 1
    return compteur


#----------------------------------------------------------------------------
#-----------PROGRAMME---------------------------------------------------
#----------------------------------------------------------------------------
  

#question 1
#voir la def


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

   
b = np.asarray(b).reshape(-1)#remettre b sous forme de vecteur
x0 = np.zeros(b.shape)
Nitermax = 100
liste_epsilon = [10**-4, 10**-5, 10**-6, 10**-7, 10**-8, 10**-9, 10**-10, 10**-11, 10**-12, 10**-13, 10**-14]
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
pp.xlim(10**-14 ,10**-4)
pp.xscale('log')
pp.xlabel('précision souhaitée')
pp.ylabel("nombre d'itérations nécessaires")
pp.title("Nombre d'itération nécessaires pour plusieurs précisions souhaitées")
pp.legend()
pp.show()


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
x0 = np.zeros((100,1))
liste_epsilon = [10**-4, 10**-5, 10**-6, 10**-7, 10**-8, 10**-9, 10**-10, 10**-11, 10**-12, 10**-13, 10**-14]
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
pp.xlim(10**-14, 10**-4)
pp.xscale('log')
pp.xlabel('précision souhaitée')
pp.ylabel("nombre d'itérations nécessaires")
pp.title("Nombre d'itération nécessaires pour plusieurs précisions souhaitées")
pp.legend()
pp.show()


#question 3


#----SYSTEME 1-------------------------------------------------
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
Nitermax = 100

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
w_etude(A)

#----SYSTEME 2-------------------------------------------------
A = np.zeros((n,n)) #SYSTEME 2
b = np.zeros((n, 1))
for i in range (0,n):
    for j in range (0,n):
        if i == j :
            A[i,i] = 3
            b[i,0] = cos((i+1)/8)
        else:
            A[i,j] = 1/(1+3*abs((i+1)-(j+1)))  

b = np.asarray(b).reshape(-1)#remettre b sous forme de vecteur
x0 = np.zeros(b.shape)
epsilon = 10**-5
Nitermax = 100

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
w_etude(A)


#ETUDE W OPTIMAL
A = np.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]])
w_etude(A)


#comparaison méthode iteratives / directes
taille = [100]
liste_nb_iteration_J = []
liste_nb_iteration_GS = []
liste_nb_iteration_Gauss = []
epsilon = 10**-13
Nitermax = 1000
for i in range(len(taille)):
    n = taille[i]
    A = np.zeros((n,n))
    b = np.zeros((n,1))
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = 3
                b[i,0] = cos((i+1)/8)
            else:
                    A[i,j] = 1/(1+3*abs((i+1)-(j+1)))
    x0 = np.zeros((n,1))    
    x , nbiter , err = MIJacobi(A, b, x0, epsilon, Nitermax)
    liste_nb_iteration_J.append(nbiter)
    x , nbiter , err = MIGaussSeidel(A, b, x0, epsilon, Nitermax)
    liste_nb_iteration_GS.append(nbiter)
    nbiter = gauss(A, b)
    liste_nb_iteration_Gauss.append(nbiter)

print("tailles étudiées : ", taille)
print("nombre d'itérations Jacobi ", liste_nb_iteration_J)
print("nombre d'itérations Gauss-Seidel ", liste_nb_iteration_GS)
print("nombre d'itérations Gauss ", liste_nb_iteration_Gauss)

#
#FIN DU PROGRAMME
#
