import numpy as np
import math

ZERNIKE_NAMES = [
    "Piston",               # Z0
    "Tilt X",               # Z1
    "Tilt Y",               # Z2
    "Oblique Astigmatism",  # Z3
    "Defocus",              # Z4
    "Vertical Astigmatism", # Z5
    "Vertical Trefoil",     # Z6
    "Vertical Coma",        # Z7
    "Horizontal Coma",      # Z8
    "Horizontal Trefoil",   # Z9
    "Oblique Quadrafoil",   # Z10
    "Oblique 2nd Astigmatism",  # Z11
    "Primary Spherical",    # Z12
    "Vertical 2nd Astigmatism",  # Z13
    "Vertical Quadrafoil",  # Z14
]

class Zernike:
    """
    Class for generating Zernike polynomials and computing phase for a given set of coefficients.
    """
    def __init__(self, slm_shape: tuple, num_zernikes: int = 15):
        """
        Initialize the Zernike polynomial generator.

        Parameters
        ----------
        slm_shape : tuple
            The shape of the SLM (height, width).
        num_zernikes : int, optional
            The number of Zernike polynomials to generate. Default is 15.
        """
        self.shape = slm_shape
        self.num_zernikes = num_zernikes
        
        # Find the parameters used for calculation of the polynomials
        y, x = np.indices(slm_shape)
        half_h = slm_shape[0] // 2
        half_w = slm_shape[1] // 2
        y = y - half_h
        x = x - half_w
        self.r = np.sqrt(x**2 + y**2) / np.sqrt(half_w**2 + half_h**2)
        self.theta = np.arctan2(y, x)
        self.r[self.r > 1] = 0  # Mask out values outside the unit circle

        # Pre-calculate the polynomials
        self.poly_array = [None] * num_zernikes
        for k in range(num_zernikes):
            n, m = self.zernike_index(k)
            self.poly_array[k] = self.zernike_polynomial(n, m)

        # Store the names of the Zernike polynomials
        self.zernike_names = ZERNIKE_NAMES[:num_zernikes] if num_zernikes <= len(ZERNIKE_NAMES) else ZERNIKE_NAMES + [f"Z{k}" for k in range(len(ZERNIKE_NAMES), num_zernikes)]

    def zernike_index(self, k):
        """
        Map an index k to the Zernike polynomial indices (n, m).

        Parameters
        ----------
        k : int
            The index of the Zernike polynomial.

        Returns
        -------
        n : int
            The radial order of the Zernike polynomial.
        m : int
            The azimuthal order of the Zernike polynomial.
        """ 
        n = 0
        while k >= (n + 1):
            k -= (n + 1)
            n += 1
        m = -n + 2 * k
        return n, m

    def zernike_radial(self, n, m, r):
        """
        Calculate the Zernike radial polynomial R^m_n(r).
        
        Parameters
        ----------
        n : int
            Radial order.
        m : int
            Azimuthal order (|m| <= n, and (n - m) is even).
        r : np.ndarray
            Radial coordinates.

        Returns
        -------
        R_nm : np.ndarray
            Zernike radial polynomial on the given shape.
        """
        R_nm = np.zeros_like(r)
        for k in range((n - m) // 2 + 1):
            R_nm += r**(n - 2*k) * (-1)**k * math.factorial(n - k) / \
                    (math.factorial(k) * math.factorial((n + m) // 2 - k) * math.factorial((n - m) // 2 - k))
        return R_nm

    def zernike_polynomial(self, n, m):
        """
        Generate the phase of the Zernike polynomial Z^m_n on a given shape.

        Parameters
        ----------
        n : int
            Radial order.
        m : int
            Azimuthal order (|m| <= n, and (n - m) is even).

        Returns
        -------
        Z_nm : np.ndarray 
            Zernike polynomial phase on the given shape.
        """

        if m >= 0:
            Z_nm = self.zernike_radial(n, m, self.r) * np.cos(m * self.theta)
        else:
            Z_nm = self.zernike_radial(n, -m, self.r) * np.sin(-m * self.theta)

        return Z_nm

    def get_phase(self, zernike_coeffs):
        """
        Generate the phase of Zernike polynomials from the coefficients.

        Returns
        -------
        phase : np.ndarray
            The phase of the Zernike polynomials on the given shape.
        """
        # Generate the phase of zernike polynomials from the coefficients
        num_terms = len(zernike_coeffs) if self.num_zernikes > len(zernike_coeffs) else self.num_zernikes
        phase = np.sum([self.poly_array[k] * zernike_coeffs[k] for k in range(num_terms)], axis=0)

        return phase
    