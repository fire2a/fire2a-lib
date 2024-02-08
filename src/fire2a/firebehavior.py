import numpy as np
import math
import pandas as pd
import datetime

def acceleration(ftype, cfb): 
    """
    Calcula la aceleración de la propagación del fuego.

    Args:
        ftype (str): Tipo de combustible.
        cfb (float): Fracción de la corona quemada.

    Returns:
        float: Aceleración de la propagación del fuego.
    """
    open_list = ["O1a", "O1b", "C1", "S1", "S2", "S3"]  # combustible abierto

    if ftype not in open_list:
        # para combustibles cerrados
        accn = 0.115 - 18.8 * cfb**2.5 * math.exp(-8.0 * cfb)  # Eq.72
    else:
        # para combustibles abiertos
        accn = 0.115

    return accn

def area(dt, df):
    """
    Calcula el área de un elipse.

    Args:
        dt (float): Diámetro mayor.
        df (float): Diámetro menor.

    Returns:
        float: Área de la elipse (en hectáreas).
    """
    a = dt / 2.0
    b = df
    areavar = a * b * math.pi / 10000.0
    return areavar

def back_fire_behaviour(ftype, sfc, brss, csi, rso, fmc, bisi, CFL):
    """
    Calcula el comportamiento del fuego en retroceso.

    Args:
        ftype (str): Tipo de combustible.
        sfc (float): Velocidad de propagación del fuego en superficie.
        brss (float): Tasa de propagación del fuego en superficie.
        csi (float): Índice de intensidad del fuego crítico.
        rso (float): Tasa de propagación del fuego en superficie de referencia.
        fmc (float): Contenido de humedad del combustible.
        bisi (float): Índice de propagación del fuego en retroceso.
        CFL (dict): Diccionario que contiene los factores de corrección de la velocidad de propagación del fuego.

    Returns:
        tuple: Una tupla que contiene la tasa de propagación final del fuego en retroceso,
               la intensidad final del fuego en retroceso,
               el consumo final de combustible en retroceso,
               y el tipo de fuego en retroceso (superficial o de copa).
    """
    bsfi = fire_intensity(sfc, brss)
    back_firetype = "superficial"

    if bsfi > csi:
        back_firetype = "de copa"

    if back_firetype == "de copa":
        bcfb = max(1 - math.exp(-0.23 * (brss - rso)), 0.0)  # fracción quemada de copa
        bcfc = CFL[ftype] * bcfb
        bros = final_ros(ftype, fmc, bisi, bcfb, brss)

        bfc = bcfc + sfc
        bfi = fire_intensity(bfc, bros)
        return bros, bfi, bfc, back_firetype
    else:
        bros = brss
        bfi = bsfi
        bfc = sfc

    return bros, bfi, bfc, back_firetype
def backfire_isi(wsv, ff):
    """
    Calcula el Índice de Propagación Inicial (ISI) para fuegos de retroceso.

    Args:
        wsv (float): Velocidad del viento sostenido [km/h].
        ff (float): Índice FFMC (Código de Humedad de Combustible Fino).

    Returns:
        float: ISI calculado.
    """
    bfw = math.exp(-0.05039 * wsv)  # Eq.75
    bisi = 0.208 * ff * bfw  # Eq.76
    return bisi

def backfire_ros(ftype, bisi, wdfh, a, b, c, FuelConst2, bui0, q):
    """
    Calcula la tasa de propagación del fuego de retroceso.

    Args:
        ftype (str): Tipo de combustible.
        bisi (float): Índice de Propagación Inicial del Fuego de Retroceso (ISI).
        wdfh (dict): Diccionario que contiene los datos meteorológicos y de combustible.
        a (float): Parámetro de la ecuación de propagación del fuego.
        b (float): Parámetro de la ecuación de propagación del fuego.
        c (float): Parámetro de la ecuación de propagación del fuego.
        FuelConst2 (dict): Diccionario que contiene constantes de combustible.
        bui0 (dict): Diccionario que contiene los índices de acumulación inicial del combustible.
        q (dict): Diccionario que contiene los factores de corrección de la velocidad de propagación del fuego.

    Returns:
        float: Tasa de propagación del fuego de retroceso.
    """
    bros = ros_base(ftype, bisi, wdfh['BUI'], a, b, c, FuelConst2)
    bros *= bui_effect(wdfh['BUI'], bui0[ftype], q[ftype])
    return bros

def bui_effect(bui, bui0, q):
    """
    Calcula el efecto del Índice de Acumulación (BUI) en la propagación del fuego.

    Args:
        bui (float): Índice de Acumulación de Combustible (BUI).
        bui0 (float): Índice de Acumulación de Combustible inicial.
        q (float): Factor de corrección de la velocidad de propagación del fuego.

    Returns:
        float: Efecto del BUI en la propagación del fuego.
    """
    bui_avg = 50.0

    if bui == 0:
        bui = 1.0
    be = np.exp(bui_avg * np.log(q) * ((1.0 / bui) - (1.0 / bui0)))
    return be

def crit_surf_intensity(cbh, fmc):
    """
    Calcula la intensidad crítica de la superficie para la transición de fuego superficial a fuego de corona.

    Args:
        cbh (float): Altura base de la copa [m].
        fmc (float): Contenido de humedad foliar [%].

    Returns:
        float: Intensidad crítica de la superficie.
    """
    csi = 0.001 * cbh**1.5 * (460.0 + 25.9 * fmc)**1.5
    return csi
def ffmc_effect(ffmc):
    """
    Calcula el efecto del Código de Humedad de Combustible Fino (FFMC) en la propagación del fuego.

    Args:
        ffmc (float): Índice FFMC (Código de Humedad de Combustible Fino).

    Returns:
        float: Efecto del FFMC en la propagación del fuego.
    """
    mc = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)  # Eq.46
    ff = 91.9 * math.exp(-0.1386 * mc) * (1 + mc**5.31 / 4.93e7)  # Eq.45
    return ff

def final_ros(ftype, fmc, isi, cfb, rss):
    """
    Calcula la tasa final de propagación del fuego, teniendo en cuenta si es un fuego de corona.

    Args:
        ftype (str): Tipo de combustible.
        fmc (float): Contenido de humedad foliar [%].
        isi (float): Índice de Propagación Inicial (ISI) del fuego.
        cfb (float): Fracción de la corona quemada.
        rss (float): Tasa de propagación del fuego en superficie.

    Returns:
        float: Tasa final de propagación del fuego.
    """
    if ftype == "C6":
        rsc = foliar_mois_effect(isi, fmc)
        ros = rss + cfb * (rsc - rss)
    else:
        ros = rss
    return ros

def fire_intensity(fc, ros):
    """
    Calcula la intensidad del fuego basada en el consumo de combustible y la tasa de propagación.

    Args:
        fc (float): Consumo de combustible predicho [kg/m2].
        ros (float): Tasa de propagación predicha [m/min].

    Returns:
        float: Intensidad del fuego [kW/m].
    """
    fi = 300.0 * fc * ros  # Eq.69 [kW/m]
    return fi

def flank_fire_behaviour(ftype, sfc, frss, csi, rso, CFL):
    """
    Determina el comportamiento del fuego en los flancos, incluyendo la intensidad y el consumo de combustible.

    Args:
        ftype (str): Tipo de combustible.
        sfc (float): Velocidad de propagación del fuego en superficie.
        frss (float): Tasa de propagación del fuego en los flancos.
        csi (float): Índice de intensidad del fuego crítico.
        rso (float): Tasa de propagación del fuego en superficie de referencia.
        CFL (dict): Diccionario que contiene los factores de corrección de la velocidad de propagación del fuego.

    Returns:
        tuple: Una tupla que contiene la intensidad final del fuego en los flancos,
               el consumo final de combustible en los flancos,
               y el tipo de fuego en los flancos (superficial o de copa).
    """
    flank_firetype = "surface"
    sfi = fire_intensity(sfc, frss)
    if sfi > csi:
        flank_firetype = "crown"

    if flank_firetype == "crown":
        fcfb = max(1 - math.exp(-0.23 * (frss - rso)), 0.0)  # crown fraction burned
        fcfc = CFL[ftype] * fcfb

        ffc = fcfc + sfc
        ffi = fire_intensity(ffc, frss)
    else:
        ffi = sfi
        ffc = sfc

    return ffi, ffc, flank_firetype

def flank_spread_distance(hrost, brost, hdist, bdist, lb, a, time):
    """
    Calcula la distancia de propagación del fuego y la tasa de propagación a lo largo del tiempo en los flancos.

    Args:
        hrost (float): Tasa de propagación del fuego en la corona.
        brost (float): Tasa de propagación del fuego en superficie de referencia.
        hdist (float): Distancia de propagación del fuego en la corona.
        bdist (float): Distancia de propagación del fuego en superficie de referencia.
        lb (float): Relación longitud-ancho de la zona quemada.
        a (float): Parámetro de la ecuación de propagación del fuego.
        time (float): Tiempo de propagación.

    Returns:
        tuple: Una tupla que contiene la distancia de propagación del fuego en los flancos,
               la tasa de propagación del fuego en los flancos,
               y la relación longitud-ancho ajustada por el tiempo.
    """
    lbt = (lb - 1.0) * (1.0 - math.exp(-a * time)) + 1.0
    rost = (hrost + brost) / (lbt * 2.0)
    fsd = (hdist + bdist) / (2.0 * lbt)
    return fsd, rost, lbt

def flankfire_ros(ros, bros, lb):
    """
    Calcula la tasa de propagación del fuego en los flancos.

    Args:
        ros (float): Tasa de propagación del fuego.
        bros (float): Tasa de propagación del fuego en la corona.
        lb (float): Relación longitud-ancho de la zona quemada.

    Returns:
        float: Tasa de propagación del fuego en los flancos.
    """
    fros = (ros + bros) / (lb * 2.0)
    return fros

def foliar_mois_effect(isi, fmc):
    """
    Calcula el efecto de la humedad foliar en la propagación del fuego.

    Args:
        isi (float): Índice de propagación inicial (ISI) del fuego.
        fmc (float): Contenido de humedad foliar [%].

    Returns:
        float: Efecto de la humedad foliar en la propagación del fuego.
    """
    fme_avg = 0.778
    fme = 1000.0 * (1.5 - 0.00275 * fmc) ** 4.0 / (460.0 + 25.9 * fmc)
    rsc = 60.0 * (1.0 - math.exp(-0.0497 * isi)) * fme / fme_avg
    return rsc

def foliar_moisture(lat, long, elev, jd):
    """
    Estima la humedad foliar basada en la ubicación geográfica y el día juliano.

    Args:
        lat (float): Latitud.
        long (float): Longitud.
        elev (float): Elevación.
        jd (int): Día juliano.

    Returns:
        float: Humedad foliar estimada.
    """
    jd_min = 0

    if jd_min <= 0:  # dispositivo cuando no hay D0
        if elev < 0:
            latn = 23.4 * math.exp(-0.0360 * (150 - long)) + 46.0  # Eq.1
            jd_min = 0.5 + 151.0 * lat / latn
        else:
            latn = 33.7 * math.exp(-0.0351 * (150 - long)) + 43.0
            jd_min = 0.5 + 142.1 * lat / latn + (0.0172 * elev)

    nd = round(abs(jd - jd_min))
    if 30 <= nd < 50:
        fm = 32.9 + 3.17 * nd - 0.0288 * nd ** 2
    elif nd >= 50:
        fm = 120
    else:
        fm = 85.0 + 0.0189 * nd ** 2

    return fm

def get_fueltype_number(ftype):
    """
    Devuelve el número de tipo de combustible basado en el tipo de combustible.

    Args:
        ftype (str): Tipo de combustible.

    Returns:
        str: Número de tipo de combustible (c = cerrado, n = abierto).
    """
    cftype = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "M1", "M2", "M3", "M4", "D1", "D2"]
    cover = "c" if ftype in cftype else "n"  # S1, S2, S3, O1a, O1b
    return cover

def ISF_deadfir(ft, a, b, c, isz, pdf, sf):
    """
    Calcula el Índice de Severidad del Fuego (ISF) para fuegos de madera muerta.

    Args:
        ft (str): Tipo de combustible.
        a (float): Parámetro de la ecuación de propagación del fuego.
        b (float): Parámetro de la ecuación de propagación del fuego.
        c (float): Parámetro de la ecuación de propagación del fuego.
        isz (float): Tamaño inicial del fuego.
        pdf (float): Porcentaje de madera muerta fina.
        sf (float): Velocidad del viento sostenido.

    Returns:
        float: Índice de Severidad del Fuego (ISF) para fuegos de madera muerta.
    """
    slopelimit_isi = 0.01
    rsf_max = sf * a[ft] * (1.0 - math.exp(-1.0 * b[ft] * isz)) ** c[ft]
    check = 1.0 - (rsf_max / a[ft]) ** (1.0 / c[ft]) if rsf_max > 0.0 else 1.0
    check = max(check, slopelimit_isi)
    isf_max = (1.0 / (-1.0 * b[ft])) * math.log(check)

    mult = 0.2 if ft == "M4" else 1.0
    rsf_d1 = sf * (mult * a["D1"]) * (1.0 - math.exp(-1.0 * b[ft] * isz)) ** c[ft]
    check = 1.0 - (rsf_d1 / (mult * a[ft])) ** (1.0 / c[ft]) if rsf_d1 > 0.0 else 1.0
    check = max(check, slopelimit_isi)
    isf_d1 = (1.0 / (-1.0 * b[ft])) * math.log(check)

    isf = pdf / 100.0 * isf_max + (100.0 - pdf) / 100.0 * isf_d1
    return isf

def ISF_mixedwood(ft, a, b, c, isz, pc, sf):
    """
    Calcula el Índice de Severidad del Fuego (ISF) para fuegos de madera mixta.

    Args:
        ft (str): Tipo de combustible.
        a (float): Parámetro de la ecuación de propagación del fuego.
        b (float): Parámetro de la ecuación de propagación del fuego.
        c (float): Parámetro de la ecuación de propagación del fuego.
        isz (float): Tamaño inicial del fuego.
        pc (float): Porcentaje de madera mixta.
        sf (float): Velocidad del viento sostenido.

    Returns:
        float: Índice de Severidad del Fuego (ISF) para fuegos de madera mixta.
    """
    slopelimit_isi = 0.01
    rsf_c2 = sf * a["C2"] * (1.0 - math.exp(-1.0 * b["C2"] * isz)) ** c["C2"]
    check = 1.0 - (rsf_c2 / a["C2"]) ** (1.0 / c["C2"]) if rsf_c2 > 0.0 else 1.0
    check = max(check, slopelimit_isi)
    isf_c2 = (1.0 / (-1.0 * b[ft])) * math.log(check)

    mult = 0.2 if ft == "M2" else 1.0
    rsf_d1 = sf * (mult * a[ft]) * (1.0 - math.exp(-1.0 * b[ft] * isz)) ** c[ft]
    check = 1.0 - (rsf_d1 / (mult * a[ft])) ** (1.0 / c[ft]) if rsf_d1 > 0.0 else 1.0
    check = max(check, slopelimit_isi)
    isf_d1 = (1.0 / (-1.0 * b[ft])) * math.log(check)

    isf = pc / 100.0 * isf_c2 + (100.0 - pc) / 100.0 * isf_d1
    return isf

def l2bAlexander1985(ws):
    """
    Calcula la relación longitud-ancho de la zona quemada según la ecuación propuesta por Alexander (1985).

    Args:
        ws (float): Velocidad del viento sostenido.

    Returns:
        float: Relación longitud-ancho de la zona quemada.
    """
    lb = 0.5 + 0.5 * math.exp(0.05039 * ws)
    return lb

def l2bAnderson1983(typefire, ws):
    """
    Calcula la relación longitud-ancho de la zona quemada según la ecuación propuesta por Anderson (1983).

    Args:
        typefire (str): Tipo de incendio.
        ws (float): Velocidad del viento sostenido.

    Returns:
        float: Relación longitud-ancho de la zona quemada.
    """
    if typefire == "dense-forest-stand":
        lb = 0.936 * math.exp(0.01240 * ws) + 0.461 * math.exp(-0.00748 * ws) - 0.397
    elif typefire == "open-forest-stand":
        lb = 0.936 * math.exp(0.01859 * ws) + 0.461 * math.exp(-0.0112 * ws) - 0.397
    elif typefire == "grass-slash":
        lb = 0.936 * math.exp(0.02479 * ws) + 0.461 * math.exp(-0.0149 * ws) - 0.397
    elif typefire == "heavy-slash":
        lb = 0.936 * math.exp(0.03099 * ws) + 0.461 * math.exp(-0.0187 * ws) - 0.397
    else:  # crown-fire forest stand
        lb = 0.936 * math.exp(0.071278 * ws) + 0.461 * math.exp(-0.043 * ws) - 0.397
    return lb

def l2bFBP(ft, ws):
    """
    Calcula la relación longitud-ancho de la zona quemada según la ecuación propuesta por el modelo FBP.

    Args:
        ft (str): Tipo de combustible.
        ws (float): Velocidad del viento sostenido.

    Returns:
        float: Relación longitud-ancho de la zona quemada.
    """
    if ft in ["O1a", "O1b"]:
        if ws < 1.0:
            lb = 1.0
        else:
            lb = 1.1 * ws ** 0.464
    else:
        lb = 1.0 + 8.729 * (1.0 - math.exp(-0.030 * ws)) ** 2.155
    return lb

def length2breadth(ftype, ws):
    """
    Calcula la relación longitud-ancho de la zona quemada según el tipo de combustible.

    Args:
        ftype (str): Tipo de combustible.
        ws (float): Velocidad del viento sostenido.

    Returns:
        float: Relación longitud-ancho de la zona quemada.
    """
    if ftype in ["O1a", "O1b"]:  # grass fuel
        if ws < 1.0:
            lb = 1.0
        else:
            lb = 1.1 + ws ** 0.464  # Eq.80
    else:
        lb = 1.0 + 8.729 * (1.0 - math.exp(-0.030 * ws)) ** 2.155
    return lb

def perimeter(hdist, bdist, lb):
    """
    Calcula el perímetro de la zona quemada.

    Args:
        hdist (float): Distancia de propagación del fuego en la corona.
        bdist (float): Distancia de propagación del fuego en superficie de referencia.
        lb (float): Relación longitud-ancho de la zona quemada.

    Returns:
        float: Perímetro de la zona quemada.
    """
    pi = 3.1416
    aux = pi * (1.0 + 1.0 / lb) * (1.0 + ((lb - 1.0) / (2.0 * (lb + 1.0))) ** 2.0)
    p = (hdist + bdist) / 2 * aux
    return p

def rate_of_spread(ftype, wdfh, a, b, c, ps, saz, FuelConst2, bui0, q):
    """
    Calcula la tasa de propagación del fuego.

    Args:
        ftype (str): Tipo de combustible.
        wdfh (dict): Diccionario que contiene variables meteorológicas y de combustible.
        a (dict): Parámetros de la ecuación de propagación del fuego.
        b (dict): Parámetros de la ecuación de propagación del fuego.
        c (dict): Parámetros de la ecuación de propagación del fuego.
        ps (float): Pendiente del terreno.
        saz (float): Ángulo de la pendiente del terreno.
        FuelConst2 (dict): Constantes del combustible.
        bui0 (dict): Índice de acumulación basal (BUI) inicial.
        q (dict): Coeficientes de propagación del fuego.

    Returns:
        float: Tasa de propagación del fuego.
    """
    ffmc = wdfh['FFMC']
    ws = wdfh['WS']
    waz = wdfh['WD'] + 180.0
    waz = waz - 360.0 if waz >= 360.0 else waz
    isz = 0.208 * ffmc_effect(ffmc)

    if ps > 0:
        wsv, raz = slope_effect(ftype, wdfh, a, b, c, saz, ps, FuelConst2, isz, ffmc, waz)
    else:
        wsv = ws
        raz = waz

    fw = math.exp(0.05039 * wsv) if wsv < 40.0 else 12.0 * (1.0 - math.exp(-0.0818 * (wsv - 28)))
    isi = isz * fw

    rsi = ros_base(ftype, isi, wdfh['BUI'], a, b, c, FuelConst2)
    rss = rsi * bui_effect(wdfh['BUI'], bui0[ftype], q[ftype])
    return rss, wsv, raz, isi


def ros_base(ftype, isi, bui, a, b, c, FuelConst2):
    """
    Calcula la tasa de propagación base del fuego (RSI) para un tipo de combustible dado.

    Args:
        ftype (str): Tipo de combustible.
        isi (float): Índice de Propagación Inicial (ISI).
        bui (float): Índice de Acumulación Basal (BUI).
        a (dict): Parámetros de la ecuación de propagación del fuego.
        b (dict): Parámetros de la ecuación de propagación del fuego.
        c (dict): Parámetros de la ecuación de propagación del fuego.
        FuelConst2 (dict): Constantes del combustible.

    Returns:
        float: Tasa de propagación base del fuego (RSI).
    """
    pdf = FuelConst2['pdf']
    cur = FuelConst2['cur']
    pc = FuelConst2['pc']

    if ftype in ["O1a", "O1b"]:
        if cur >= 58.8:
            mu1 = 0.176 + 0.02 * (cur - 58.8)
        else:
            mu1 = 0.005 * (math.exp(0.061 * cur) - 1.0)
        mu1 = max(mu1, 0.001)
        rsi = mu1 * (a[ftype] * (1.0 - math.exp(-b[ftype] * isi)) ** c[ftype])
    elif ftype in ["M1", "M2"]:
        mu1 = pc / 100.0
        mu2_1 = (100 - pc) / 100.0
        mu2_2 = 2 * (100 - pc) / 100.0
        ros_C1 = a["C2"] * (1.0 - math.exp(-b["C2"] * isi)) ** c["C2"]
        ros_D1 = a["D1"] * (1.0 - math.exp(-b["D1"] * isi)) ** c["D1"]
        rsi = mu1 * ros_C1 + mu2_1 * ros_D1 if ftype == "M1" else mu1 * ros_C1 + mu2_2 * ros_D1
    elif ftype in ["M3", "M4"]:
        if ftype == "M3":
            a3 = 170 * math.exp(-35.0 / pdf)
            b3 = 0.082 * math.exp(-36.0 / pdf)
            c3 = 1.698 - 0.00303 * pdf
            rsi = a3 * (1.0 - math.exp(-b3 * isi)) ** c3
        else:
            a4 = 140 * math.exp(-35.5 / pdf)
            b4 = 0.0404
            c4 = 3.03 * math.exp(-0.00714 * pdf)
            rsi = a4 * (1.0 - math.exp(-b4 * isi)) ** c4
    elif ftype == "D2":
        rsi = a[ftype] * (1.0 - math.exp(-b[ftype] * isi)) ** c[ftype] if bui >= 80 else 0.0
    else:
        rsi = a[ftype] * (1.0 - math.exp(-b[ftype] * isi)) ** c[ftype]

    return rsi

def slope_effect(ft, wdfh, a, b, c, saz, ps, FuelConst2, isi, ff, waz):
    """
    Calcula el efecto de la pendiente en la tasa de propagación del fuego.

    Args:
        ft (str): Tipo de combustible.
        wdfh (dict): Diccionario que contiene variables meteorológicas y de combustible.
        a (dict): Parámetros de la ecuación de propagación del fuego.
        b (dict): Parámetros de la ecuación de propagación del fuego.
        c (dict): Parámetros de la ecuación de propagación del fuego.
        saz (float): Ángulo de la pendiente del terreno.
        ps (float): Pendiente del terreno.
        FuelConst2 (dict): Constantes del combustible.
        isi (float): Índice de Propagación Inicial (ISI).
        ff (float): Factor de propagación del fuego.
        waz (float): Ángulo del viento.

    Returns:
        tuple: Tasa de propagación del fuego ajustada a la pendiente y ángulo del viento, y el ángulo de propagación ajustado.
    """
    pi = 3.1415
    slopelimit_isi = 0.01
    pc = FuelConst2["pc"]
    pdf = FuelConst2["pdf"]
    ps = min(ps, 70.0)
    sf = min(math.exp(3.533 * (ps / 100.0) ** 1.2), 10.0)
    ws = wdfh['WS']  # Acceso a la columna WS

    if saz >= 360.0:
        saz -= 360.0

    if ft in ["M1", "M2"]:
        isf = ISF_mixedwood(ft, a, b, c, isi, pc, sf)
    elif ft in ["M3", "M4"]:
        isf = ISF_deadfir(ft, a, b, c, isi, pdf, sf)
    else:
        rsz = ros_base(ft, isi, wdfh, a, b, c, FuelConst2)
        rsf = rsz * sf
        check = 1.0 - (rsf / a[ft]) ** (1.0 / c[ft]) if rsf > 0.0 else 1.0
        check = max(check, slopelimit_isi)
        isf = -1.0 / b[ft] * math.log(check)

    isf = isi if isf == 0.0 else isf

    wse1 = math.log(isf / (0.208 * ff)) / 0.05039

    if wse1 <= 40.0:
        wse = wse1
    else:
        isf = min(isf, 0.999 * 2.496 * ff)
        wse2 = 28.0 - math.log(1.0 - isf / (2.496 * ff)) / 0.0818
        wse = wse2

    wrad = waz / 180.0 * pi
    wsx = ws * math.sin(wrad)
    wsy = ws * math.cos(wrad)
    srad = saz / 180.0 * pi
    wsex = wse * math.sin(srad)
    wsey = wse * math.cos(srad)
    wsvx = wsx + wsex
    wsvy = wsy + wsey
    wsv = math.sqrt(wsvx ** 2 + wsvy ** 2)
    raz = math.acos(wsvy / wsv) / pi * 180.0
    raz = 360 - raz if wsvx < 0 else raz

    return wsv, raz
def spread_distance(ros, time, a):
    """
    Calcula la distancia de propagación del fuego y la tasa de propagación a lo largo del tiempo.

    Args:
        ros (float): Tasa de propagación del fuego inicial.
        time (float): Tiempo transcurrido.
        a (float): Parámetro de la ecuación de propagación del fuego.

    Returns:
        tuple: Distancia de propagación del fuego y la tasa de propagación actualizada.
    """
    rost = ros * (1.0 - math.exp(-a * time))
    sd = ros * (time + (math.exp(-a * time) / a) - 1.0 / a)
    return sd, rost


def surf_fuel_consump(ft, wdfh, FuelConst2):
    """
    Estima el consumo de combustible superficial basado en el tipo de combustible y condiciones meteorológicas.

    Args:
        ft (str): Tipo de combustible.
        wdfh (dict): Diccionario que contiene variables meteorológicas y de combustible.
        FuelConst2 (dict): Constantes del combustible.

    Returns:
        float: Consumo de combustible superficial estimado.
    """
    bui = wdfh['BUI']  # Acceso a la columna BUI
    ffmc = wdfh['FFMC']  # Acceso a la columna FFMC

    gfl = FuelConst2["gfl"]
    pc = FuelConst2["pc"]

    if ft == "C1":
        sfc = 0.75 + 0.75 * math.sqrt(1 - math.exp(-0.23 * (ffmc - 84))) if ffmc > 84 else 0.75 - 0.75 * math.sqrt(1 - math.exp(0.23 * (ffmc - 84)))
        sfc = max(sfc, 0)
    elif ft in ["C2", "M3", "M4"]:
        sfc = 5.0 * (1.0 - math.exp(-0.0115 * bui))
    elif ft in ["C3", "C4"]:
        sfc = 5.0 * (1.0 - math.exp(-0.0164 * bui)) ** 2.24
    elif ft in ["C5", "C6"]:
        sfc = 5.0 * (1.0 - math.exp(-0.0149 * bui)) ** 2.48
    elif ft == "C7":
        ffc = 2.0 * (1.0 - math.exp(-0.104 * (ffmc - 70.0)))
        wfc = 1.5 * (1.0 - math.exp(-0.0201 * bui))
        sfc = max(ffc, 0) + wfc
    elif ft in ["O1a", "O1b"]:
        sfc = gfl
    elif ft in ["M1", "M2"]:
        sfc_c2 = 5.0 * (1.0 - math.exp(-0.0115 * bui))
        sfc_d1 = 1.5 * (1.0 - math.exp(-0.0183 * bui))
        sfc = pc / 100.0 * sfc_c2 + (100.0 - pc) / 100.0 * sfc_d1
    elif ft == "S1":
        ffc = 4.0 * (1.0 - math.exp(-0.025 * bui))
        wfc = 4.0 * (1.0 - math.exp(-0.034 * bui))
        sfc = ffc + wfc
    elif ft == "S2":
        ffc = 10.0 * (1.0 - math.exp(-0.013 * bui))
        wfc = 6.0 * (1.0 - math.exp(-0.060 * bui))
        sfc = ffc + wfc
    elif ft == "S3":
        ffc = 12.0 * (1.0 - math.exp(-0.0166 * bui))
        wfc = 20.0 * (1.0 - math.exp(-0.0210 * bui))
        sfc = ffc + wfc
    elif ft == "D1":
        sfc = 1.5 * (1.0 - math.exp(-0.0183 * bui))
    elif ft == "D2":
        sfc = 1.5 * (1.0 - math.exp(-0.0183 * bui)) if bui >= 80 else 0
    else:
        sfc = 0  # Default case for unrecognized fuel type

    return sfc

# 18 Fuel Types
FBPfuelTypes = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'M1', 'M2', 'M3',
                'M4', 'D1', 'D2', 'S1', 'S2', 'S3', 'O1a', 'O1b']

# Crown fuel load [Kg/m2]
CFLvalues = [0.75, 0.8, 1.15, 1.2, 1.2, 1.8, 0.5, 0.8,
             0.8, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Canopy base Height [m]
CBHvalues = [2.0, 3.0, 8.0, 4.0, 18.0, 7.0, 10.0, 6.0, 6.0, 6.0,
             6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Parameters for Basic rate of spread (ISI-equation)
a_values = [90, 110, 110, 110, 30, 30, 45, 110, 110,
            120, 100, 30, 6, 75, 40, 55, 190, 250]

b_values = [0.0649, 0.0282, 0.0444, 0.0293, 0.0697, 0.08, 0.0305, 0.0282,
            0.0282, 0.0572, 0.0404, 0.0232, 0.0232, 0.0297, 0.0438, 0.0829, 0.031, 0.031]

c_values = [4.5, 1.5, 3.0, 1.5, 4.0, 3.0, 2.0, 1.5, 1.5, 1.4, 1.48,
            1.6, 1.6, 1.3, 1.7, 3.2, 1.4, 1.7]

# Parameters for Buildup Effect (BE)
qvalues = [0.9, 0.7, 0.75, 0.8, 0.8, 0.8, 0.85, 0.8, 0.8, 0.8, 0.8,
           0.75, 0.9, 0.75, 0.75, 0.75, 1.0, 1.0]

bui0values = [72, 64, 62, 66, 56, 62, 106, 50, 50, 50, 50,
              32, 32, 38, 63, 31, 1, 1]

# Building dictionaries for parameters
CFL = dict(zip(FBPfuelTypes, CFLvalues))
CBH = dict(zip(FBPfuelTypes, CBHvalues))
q = dict(zip(FBPfuelTypes, qvalues))
bui0 = dict(zip(FBPfuelTypes, bui0values))
a = dict(zip(FBPfuelTypes, a_values))
b = dict(zip(FBPfuelTypes, b_values))
c = dict(zip(FBPfuelTypes, c_values))

FuelConst2 = {
    "pc": 50,  # Percent Conifer for M1/M2 [percent]
    "pdf": 35, # Percent Dead Fir for M3/M4 [percent]
    "gfl": 0.35, # Grass Fuel Load [kg/m^2]
    "cur": 60  # Percent Cured for O1a/O1b [percent]
}
