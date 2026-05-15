"""
Stub audioop pour Python 3.13 — compatibilité pydub
audioop a été supprimé de Python 3.13, ce stub évite l'erreur d'import.
"""

class error(Exception):
    pass

def add(fragment1, fragment2, width): return b''
def adpcm2lin(fragment, width, state): return b'', state
def alaw2lin(fragment, width): return b''
def avg(fragment, width): return 0
def avgpp(fragment, width): return 0
def bias(fragment, width, bias): return b''
def byteswap(fragment, width): return b''
def cross(fragment, width): return 0
def findfactor(fragment, reference): return 0.0
def findfit(fragment, reference): return (0, 0.0)
def findmax(fragment, length): return 0
def getsample(fragment, width, index): return 0
def lin2adpcm(fragment, width, state): return b'', state
def lin2alaw(fragment, width): return b''
def lin2lin(fragment, width, newwidth): return b''
def lin2ulaw(fragment, width): return b''
def max(fragment, width): return 0
def maxpp(fragment, width): return 0
def minmax(fragment, width): return (0, 0)
def mul(fragment, width, factor): return b''
def ratecv(fragment, width, nchannels, inrate, outrate, state, weightA=1, weightB=0):
    return b'', state
def reverse(fragment, width): return b''
def rms(fragment, width): return 0
def tomono(fragment, width, lfactor, rfactor): return b''
def tostereo(fragment, width, lfactor, rfactor): return b''
def ulaw2lin(fragment, width): return b''
