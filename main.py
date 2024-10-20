from mpmath import *
import math
import scipy.integrate as integrate
import numpy as np
import cypari2

pari = cypari2.Pari()


### COMPLEX INTEGRATION IMPLEMENTATION
def complex_quadrature(func, a, b):
    '''Computes complex integrals'''
    def real_func(x):
        return np.real(func(x))
    def imag_func(x):
        return np.imag(func(x))
    real_integral = integrate.quad(real_func, a, b)
    imag_integral = integrate.quad(imag_func, a, b)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])


### J-FUNCTION IMPLEMENTATION
def comp_jtau(tau):
    '''Returns j(tau)'''
    small_j = 1728 * kleinj(tau) # big J into little j
    return small_j

def j_coef_asymp(n):
    return np.exp(4*np.pi*np.sqrt(n)) / (np.sqrt(2) * n**(3/4))

def j_coef(n):
    coefs_lst = []
    coefs = pari.laurentseries(lambda x: pari.ellj(x), n)
    for i in coefs:
        coefs_lst.append(i)
    return coefs_lst


### KANEKO RELATED CALCULATIONS
def epsilon(D):
    '''Finds the value of epsilon based on discriminant D'''
    if D % 4 == 1:
        c = 2
    elif D % 4 == 2 or D % 4 == 3:
        c = 1
    
    b_lst = []
    # positive iteration
    b = 1
    while math.sqrt(D*(b**2)+(c**2)) != math.isqrt(D*(b**2)+(c**2)):
        b += 1
    b_lst.append(b)

    # negative iteration
    b = 1
    while math.sqrt(D*(b**2)-(c**2)) != math.isqrt(D*(b**2)-(c**2)):
        b += 1
    b_lst.append(b)

    b = min(b_lst) # gives the lowest value of b that satisfies the perfect square

    a1 = math.sqrt(D*(b**2)+(c**2))
    a2 = math.sqrt(D*(b**2)-(c**2))

    if a1.is_integer() and a2.is_integer():
        a = min(a1,a2)
    else:
        if a1.is_integer():
            a = a1
        elif a2.is_integer():
            a = a2

    epsilon = (1/c)*(a+(b*math.sqrt(D)))
    return epsilon

def log_epsilon(epsilon):
    '''Returns log(epsilon)'''
    return math.log(epsilon)

def val(W,ln_epsilon): # we assume input w is a list e.g. [1/2, sqrt(5)/2] if w=(1+sqrt(5))/2
    '''Returns val(w)'''
    # assume w is in the form a + sqrt(b) where a and b are abitrary constants
    w = W[0] + W[1]
    w_prime = W[0] - W[1]
    w_paren = w - w_prime
    return (1 / (2 * ln_epsilon)) * complex_quadrature(lambda x: comp_jtau((w-(w_paren * w_prime * 1j * math.e**x)) / (1-(w_paren * 1j * math.e**x))), -ln_epsilon, ln_epsilon)[0]


### ANDERSON RELATED CALCULATIONS
def gamma_alpha(s,r):
    '''Returns gamma_alpha from Anderson's paper'''
    if math.gcd(s,r) == 1:
        a = 1
        # ar+1=-bs
        # plug in a=1,2,3
        # till ar+1 is div by s
        while (r * a + 1) % s != 0:
            a += 1
        return int(-(1/s)*(a * r + 1)), a

def quadratic_solver(a,b,c):
    '''Returns the roots of a quadratic form ax**2+bxy+cy**2'''
    return (-b + math.sqrt(b**2-4*a*c)), (2 * a), (-b - math.sqrt(b**2-4*a*c)), (2 * a)

def e_func(x):
    '''e function for use in j1Q'''
    return np.exp(2*math.pi*1j*x)

def j1Q(tau,a,b,c):
    '''Computes j_{1Q} from Anderson's j-function modification'''
    j1 = np.real(comp_jtau(tau))-744
    if a == 0:
        s1, r1 = 0, 1 # r1 is arbitrary
        s2, r2 = 1, 0 # s2 is arbitrary
    else:
        alpha1_num, alpha1_denom, alpha2_num, alpha2_denom = [int(x) for x in list(quadratic_solver(a,b,c))]
        alpha1_div = math.gcd(alpha1_num,alpha1_denom)
        alpha2_div = math.gcd(alpha2_num,alpha2_denom)
        s1, r1 = int(alpha1_num / alpha1_div), int(alpha1_denom / alpha1_denom)
        s2, r2 = int(alpha2_num / alpha2_div), int(alpha2_denom / alpha2_denom)
    if s1 == 0:
        b = 1 # b is arbitrary
        a = -(1/r1)
        sinh1 = 2*math.pi*np.imag((a*tau+b) / (s1 * tau + -r1))
        sinh2 = 2*math.pi*np.imag((gamma_alpha(s2,r2)[1]*tau+gamma_alpha(s2,r2)[0]) / (s2 * tau + -r2))
        e1 = e_func(np.real((a*tau+b) / (s1 * tau + -r1)))
        e2 = e_func(np.real((gamma_alpha(s2,r2)[1]*tau+gamma_alpha(s2,r2)[0]) / (s2 * tau + -r2)))
    else:
        sinh1 = 2*math.pi*np.imag((gamma_alpha(s1,r1)[1]*tau+gamma_alpha(s1,r1)[0]) / (s1 * tau + -r1))
        sinh2 = 2*math.pi*np.imag((gamma_alpha(s2,r2)[1]*tau+gamma_alpha(2,r2)[0]) / (s2 * tau + -r2))
        e1 = e_func(np.real((gamma_alpha(s1,r1)[1]*tau+gamma_alpha(s1,r1)[0]) / (s1 * tau + -r1)))
        e2 = e_func(np.real((gamma_alpha(s2,r2)[1]*tau+gamma_alpha(s2,r2)[0]) / (s2 * tau + -r2)))
    return np.real(j1 - 2 * (sinh1 * e1 + sinh2 * e2))

def anderson_int(a,b,c):
    '''Computes the anderson integral over C_Q geodesics using parameterization (line integral)'''
    sq = math.sqrt(b**2-(4*a*c))/(2*a) # the sqrt
    return complex_quadrature(lambda t: (j1Q(sq*np.cos(t)+(b/(2*a))+1j*sq*np.sin(t),a,b,c))*(-sq*np.sin(t)+1j*sq*np.cos(t))/(a*(sq*np.cos(t)+(b/(2*a))+1j*sq*np.sin(t))**2+b*(sq*np.cos(t)+(b/(2*a))+1j*sq*np.sin(t))+c**2),0.001,np.pi-0.001)


### BFI RELATED CALCULATIONS
def Ei_int(x):
    '''Computes the real function Ei(x) (mpmath documentation)'''
    return ei(x)

def Ei_scratch(x):
    '''Computes the real function Ei(x) (my code from scratch)'''
    return integrate.quad(lambda t: -e**(-t)/t, -x, np.inf)

def nge0(n):
    '''Computes the summation for when n>0'''
    sum = 0
    coefs = j_coef(n+2)[2:]
    terms = []
    for i in range(n):
        indice = coefs[i] * Ei_int(-2*np.pi*(i+1))
        sum += indice
        terms.append(indice)
    return sum

def nle0():
    '''Computes the summation for when n<0 (just n = -1 term)'''
    coefs = j_coef(1)[0]
    return coefs * Ei_int(-2*np.pi*(-1))

def BFI(n): # this integral converges at n = 9, giving value approx. -100.709724331336
    '''The BFI regularized integral'''
    return -2*(nge0(n)+nle0())
