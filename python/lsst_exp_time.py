import numpy as np
import inspect
import os


def build_coeff_tables():
    """
    Use the latest throughput values to build Table 2. Requires LSST stack installed and
    syseng_throughputs setup
    """
    from calcM5 import calcM5
    import bandpassUtils as bu
    from lsst.utils import getPackageDir
    defaultDirs = bu.setDefaultDirs(rootDir = getPackageDir('syseng_throughputs'))
    addLosses = True
    atmosphere = bu.readAtmosphere(defaultDirs['atmosphere'], atmosFile='atmos_10_aerosol.dat')
    hardware, system = bu.buildHardwareAndSystem(defaultDirs, addLosses, atmosphereOverride=atmosphere)
    m5_vals = calcM5(hardware, system, atmosphere, title='', return_t2_values=True)
    np.savez('m5_vals.npz', m5_vals=m5_vals)


def conditions2m5(FWHMeff_zenith=0.8, airmass=1., t_vis=30., filtername='r', sky_brightness=None):
    """
    Take the observing conditions and return the 5-sigma limiting depth

    Parameters
    ----------
    FWHMeff_zenith : float (0.8)
        The seeing at zenith (arcsec)
    sky_brightness : float or np.array
        Sky background brightness (mag / sq arcsec)
    airmass : float or np.array (1.)
        The airmass(es) (unitless). Should be same length as sky_brightness
    t_vis : float (30.)
        The total exposure time of a visit (seconds)
    filtername : str
        The filter of the observation (ugrizy)

    Returns
    -------
    m5 : float or np.array
        The 5-sigma limiting depth of a point source taken in the conditions
    """

    # Only load the info once if needed
    if not hasattr(conditions2m5, 'm5_vals'):
        dirname = os.path.dirname(inspect.getfile(conditions2m5))
        filename = os.path.join(dirname, 'm5_vals.npz')
        temp = np.load(filename)
        conditions2m5.m5_vals = temp['m5_vals'][()]
        temp.close

    median_sky_brightnesses = {'u': 22.9, 'g': 22.3, 'r': 21.2, 'i': 20.5,
                               'z': 19.6, 'y': 18.6}
    if sky_brightness is None:
        sky_brightness = median_sky_brightnesses[filtername]

    # The FWHM at the airmass(es)
    FWHMeff = airmass**(0.6) * FWHMeff_zenith

    # Equation 6 from overview paper
    m5 = conditions2m5.m5_vals['Cm'][filtername] + 0.5 * (sky_brightness - 21.)
    m5 += 2.5 * np.log10(0.7 / FWHMeff)
    m5 += 1.25 * np.log10(t_vis / 30.) - conditions2m5.m5_vals['kAtm'][filtername] * (airmass - 1.)

    # Equation 7 from the overview paper
    if t_vis > 30.:
        tau = t_vis/30.
        numerator = 10.**(0.8 * conditions2m5.m5_vals['dCm_infinity'][filtername]) - 1.
        dcm = conditions2m5.m5_vals['dCm_infinity'][filtername]-1.25*np.log10(1. + numerator/tau)
        m5 += dcm

    return m5


def m52snr(m, m5):
    """
    Calculate the SNR for a star of magnitude m in an
    observation with 5-sigma limiting magnitude depth m5.
    Assumes gaussian distribution of photons and might not be
    strictly due in bluer filters. See table 2 and equation 5
    in astroph/0805.2366.

    Parameters
    ----------
    m : float or numpy.ndarray
        The magnitude of the star
    m5 : float or numpy.ndarray
        The m5 limiting magnitude of the observation

    Returns
    -------
    float or numpy.ndarray
        The SNR
    """
    snr = 5.*10.**(-0.4*(m-m5))
    return snr


def lsst_snr(inmag, FWHMeff_zenith=0.8, airmass=1., t_vis=30., filtername='r', sky_brightness=None):
    """
    Given the magnitude and observing conditions, return the expected signal-to-noise ratio
    LSST will achieve.

    Parameters
    ----------
    FWHMeff_zenith : float (0.8)
        The seeing at zenith (arcsec)
    sky_brightness : float or np.array
        Sky background brightness (mag / sq arcsec).
    airmass : float or np.array (1.)
        The airmass(es) (unitless). Should be same length as sky_brightness
    t_vis : float (30.)
        The total exposure time of a visit (seconds)
    filtername : str
        The filter of the observation (ugrizy)

    Returns
    -------
    snr : float or np.array
        The signal to noise of a point source observed by LSST in the specified conditions
    """
    m5 = conditions2m5(FWHMeff_zenith=FWHMeff_zenith, airmass=airmass,
                       t_vis=t_vis, filtername=filtername, sky_brightness=sky_brightness)
    snr = m52snr(inmag, m5)
    return snr



