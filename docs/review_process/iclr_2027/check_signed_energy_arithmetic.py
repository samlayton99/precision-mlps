"""Exact-rational checks for the illustrative theorem constants; no experiments."""
from fractions import Fraction as F

def A_squared(p, alpha):
    return alpha**2 * F(2**p * p**p, (p+1)**(p+1))

assert A_squared(3,F(1,3)) < F(307,1000)**2
assert A_squared(5,F(2,15)) < F(196,1000)**2
assert A_squared(7,F(17,315)) < F(136,1000)**2
assert F(3,2) < F(1225,1000)**2

W=705;M=12;eta=F(2,1000)
A3=F(307,1000);A5=F(196,1000);A7=F(136,1000);C3=F(1225,1000)
S4=F(3541,10000);S6=F(1044,100000);S8=F(3551,10000000)
assert 3*F(M**4,W**2)<S4**2
assert F(3*M**3,W**2)<S6
assert F(6*M**4,W**3)<S8
assert W*M<92**2
assert S6<F(1022,10000)**2

EH=F(101,100);EC=F(1,10**7);tau=F(1,10000)
U=A5**2*S6**2/2+8*A3*tau*S4+12*A5*tau*S6+16*A7*S8+3*EC*92
G=C3*EH*F(1022,10000)+EC*5
assert U<F(892,10**6)
assert G<F(1265,10000)
increment=eta*U+eta**2*G**2
assert increment<F(185,10**8)
assert 3_200_000*increment<6
assert F(12,705)<F(131,1000)**2

# Conservative arithmetic for the local descent argument before first exit.
assert (F(3465,1000)+F(101,10000))**2<13
assert F(101,100)+F(5197,1000)*F(101,10000)<F(1063,1000)
assert 27+F(1063,1000)*(4*F(3606,1000)+F(1415,1000))<44
assert eta*44<1
print('All exact-rational constant and 3.2-million-update arithmetic checks passed.')
