import numpy as np
import warnings


def clarke_risk_index(BG):
    bg = max(BG, 1)
    fBG = 1.509 * (np.log(bg)**1.084 - 5.381)
    if fBG < 0: 
        LBGI = 10 * (fBG)**2
        HBGI = 0
    if fBG > 0:
        LBGI = 0
        HBGI = 10 * (fBG)**2
    RI = LBGI + HBGI
    return (LBGI, HBGI, RI)


def magni_risk_index(BG):
    bg = max(BG, 1)
    fBG = 3.5506*(np.log(bg)**0.8353-3.7932)
    if fBG < 0: 
        LBGI = 10 * (fBG)**2
        HBGI = 0
    if fBG > 0:
        LBGI = 0
        HBGI = 10 * (fBG)**2
    RI = LBGI + HBGI
    return (LBGI, HBGI, RI)