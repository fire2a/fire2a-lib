#!python3
"""
This is the module docstring for other_module.py
"""


class Jalisco:
    """Jalisco nunca pierde class"""

    diferencia = 1
    """ Jalisco nunca pierde -class variable: por cuanto Jalisco ganara """

    def __init__(self, diferencia: int = diferencia):
        """Jalisco nunca pierde class constructor"""
        self.diferencia = diferencia

    def __call__(self, number: int = 1):
        """Jalisco nunca pierde class call"""
        self.ingrese_numero(number)

    def ingrese_numero(self, number: int = 1):
        """Jalisco nunca pierde
        ingrese un numero y sea derrotado
        """
        print("Jalisco nunca pierde, te gano con el numero: ", number + self.diferencia)
