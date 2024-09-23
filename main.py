from mpmath import *
from cauchy_principle_value import CPV
import math
import scipy.integrate as integrate
import numpy as np
import cypari2

pari = cypari2.Pari()

def complex_quadrature(func, a, b):
    '''Computes complex integrals'''
    def real_func(x):
        return np.real(func(x))
    def imag_func(x):
        return np.imag(func(x))
    real_integral = integrate.quad(real_func, a, b)
    imag_integral = integrate.quad(imag_func, a, b)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

def comp_jtau(tau):
    '''Returns j(tau)'''
    small_j = 1728 * kleinj(tau) # big J into little j
    return small_j

def epsilon(D): # how does it work for n = 0 (mod 4)
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

# def jmQ(m, Q, tau):
#     j_m = np.exp(2*math.pi*1j*tau)**(-m)+np.sum()

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
    sinh1 = 2*math.pi*np.imag((gamma_alpha(s1,r1)[1]*tau+gamma_alpha(s1,r1)[0]) / (s1 * tau + -r1))
    sinh2 = 2*math.pi*np.imag((gamma_alpha(s2,r2)[1]*tau+gamma_alpha(2,r2)[0]) / (s2 * tau + -r2))
    e1 = e_func(np.real((gamma_alpha(s1,r1)[1]*tau+gamma_alpha(s1,r1)[0]) / (s1 * tau + -r1)))
    e2 = e_func(np.real((gamma_alpha(s2,r2)[1]*tau+gamma_alpha(s2,r2)[0]) / (s2 * tau + -r2)))
    return np.real(j1 - 2 * (sinh1 * e1 + sinh2 * e2))

def j1Q_test(tau,a,b,c):
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

def anderson_int_og(a,b,c):
    '''Computes the anderson integral over C_Q geodesics using parameterization (line integral)'''
    # line integral review: https://tutorial.math.lamar.edu/classes/calciii/lineintegralspti.aspx
    # quad = np.sqrt(b**2-(4*a*c)) / (2 * a)
    # return complex_quadrature(lambda t: (j1Q(tau,a,b,c))*(a*(quad*np.cos(t)+(b/(2*a)))**2+b*(quad*np.cos(t)+(b/(2*a)))+c)*quad*math.sqrt((np.sin(t))**2+(np.cos(t))**2),0,np.pi)
    nsq = (b**2-(4*a*c))/(4*a) # the non sqrt
    sq = (b*math.sqrt(b**2-(4*a*c)))/a # the sqrt
    return complex_quadrature(lambda t: j1Q(t,a,b,c)/(nsq*np.cos(2*t)+sq*np.cos(t)+(3*b**2)/(4*a)+c+1j*(2*nsq*np.cos(t)*np.sin(t)+sq*np.sin(t))),0,np.pi)

def anderson_int(a,b,c):
    '''Computes the anderson integral over C_Q geodesics using parameterization (line integral)'''
    # line integral review: https://tutorial.math.lamar.edu/classes/calciii/lineintegralspti.aspx
    sq = math.sqrt(b**2-(4*a*c))/(2*a) # the sqrt
    return complex_quadrature(lambda t: (j1Q_test(sq*np.cos(t)+(b/(2*a))+1j*sq*np.sin(t),a,b,c))*(-sq*np.sin(t)+1j*sq*np.cos(t))/(a*(sq*np.cos(t)+(b/(2*a))+1j*sq*np.sin(t))**2+b*(sq*np.cos(t)+(b/(2*a))+1j*sq*np.sin(t))+c**2),0.001,np.pi-0.001)
    # problem with Q(z,1)

def anderson_int_lyapunov(a,b,c,alpha_min,alpha_plus):
    '''Computes the anderson integral over C_Q geodesics using parameterization (line integral)'''
    # line integral review: https://tutorial.math.lamar.edu/classes/calciii/lineintegralspti.aspx
    sq = math.sqrt(b**2-(4*a*c))/(2*a) # the sqrt
    return complex_quadrature(lambda t: (j1Q_test(sq*np.cos(t)+(b/(2*a))+1j*sq*np.sin(t),a,b,c))*(-sq*np.sin(t)+1j*sq*np.cos(t))/(a*(sq*np.cos(t)+(b/(2*a))+1j*sq*np.sin(t))**2+b*(sq*np.cos(t)+(b/(2*a))+1j*sq*np.sin(t))+c**2),alpha_min,alpha_plus)
    # problem with Q(z,1)

def Ei_int(x):
    '''Computes the real function Ei(x)'''
    return integrate.quad(lambda t: e**(-t)/t, -x, np.inf)

def Ei_summ(tau):
    sum = 0
    for i in range(len(pari.ellfromj(tau))):
        sum += Ei_int(-2*np.pi*i) * pari.ellfromj(tau)[i]
    return sum

def cauchy_principal_value ( f, a, b, n ):
    cpv = CPV()
    if ( ( n % 2 ) != 0 ):
        print ( '' )
        print ( 'cauchy_principal_value - Fatal error!' )
        print ( '  N must be even.' )
        raise Exception ( 'cauchy_principal_value - Fatal error.' )
    #
    #  Get the Gauss-Legendre rule.
    #
    [ x, w ] = cpv.legendre_set (n)
    #
    #  Estimate the integral.
    #
    value = 0.0
    for i in range ( 0, n ):
        x2 = ( ( 1.0 - x[i] ) * a   \
            + ( 1.0 + x[i] ) * b ) \
            /   2.0
        value = value + w[i] * ( f ( x2 ) ) / x[i]
    return value

def PV_summ(tau,a,b,n):
    sum = 0
    for i in range(len(pari.ellfromj(tau))):
        sum += cauchy_principal_value(Ei_int(-2*np.pi*i),a,b,n) * pari.ellfromj(tau)[i]
    return sum

# Testing below:

print(np.real(val([1/2,math.sqrt(5)/2],log_epsilon(epsilon(5)))))
print(np.real(val([-3/2,math.sqrt(5)/2],log_epsilon(epsilon(5))))) # different cntd frac, same discriminant gives same val
print(Ei_int(-2))

# print(val([1,math.sqrt(3)],log_epsilon(epsilon(12)))) # does not work for multiples of 4
