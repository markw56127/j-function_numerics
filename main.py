from mpmath import *
import math

def comp_jtau(tau):
    big_J = kleinj(tau) # imports the J-func
    small_j = 1728 * big_J # big J into little j
    return small_j

def discriminant(B): # modulo arithmetic for discriminant (for values of B) and for constant c
    if B % 4 == 1:
        disc = B
    else:
        disc = 4*B
    if disc % 4 == 1:
        c = 2
    elif disc % 4 == 2 or disc % 4 == 3:
        c = 1
    return disc, c

def u_sol(n,c): # how does it work for n = 0 (mod 4)
    '''Iterating to find the lowest b value that gives a perfect square'''
    b_lst = []
    b = 1
    while math.sqrt(n*(b**2)+(c**2)) != math.isqrt(n*(b**2)+(c**2)):
        b += 1
    b_lst.append(b) # adds the b found in the + version to a list

    b = 1
    while math.sqrt(n*(b**2)-(c**2)) != math.isqrt(n*(b**2)-(c**2)):
        b += 1
    b_lst.append(b) # adds the b found in the - version to a list

    b = min(b_lst) # gives the lowest value of b that satisfies the perfect square

    a_1 = math.sqrt(n*(b**2)+(c**2)) # solves for a_1
    a_2 = math.sqrt(n*(b**2)-(c**2))

    if a_1.is_integer(): # gives the positive integer a_1
        a = a_1
    elif a_2.is_integer():
        a = a_2

    u = (1/c)*(a+(b*math.sqrt(n))) # u_1 formula
    return u

def epsilon(u): # calculates the log(u_1)/log(epsilon)
    return math.log(u)

print(epsilon(u_sol(53,2))) # works for D=5, but is off by a factor of 2 for otherwise

# maybe issue with the 1/c in the u formula?