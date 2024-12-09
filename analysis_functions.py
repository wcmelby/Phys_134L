import numpy as np
# import matplotlib.pyplot as plt
from astropy.io import fits
# import sys
import os
import pandas as pd


filters = ['g', 'r', 'i']
color_index_types = ['g-r', 'r-i', 'g-i']


def calculate_flux(file_root, star_name, x_position, y_position, radius):
    """Calculate flux within an aperture of (radius) pixels, then subtract the median background counts. 
    star_name is a string that is the start of the file name (excluding the filter, like 'van_Maanen_')"""

    gain = 0.7 # e-/ADU
    pixel_sums = []
    fluxes = []
    filter_zeropoints = []

    for filter in filters:
        file = os.path.join(file_root, star_name + f"{filter}" + '.fits')

        hdul = fits.open(file)
        data = hdul[1].data
        header = hdul[1].header
        zeropoint = header.get('L1ZP', 'Keyword not found')
        exptime = header.get('EXPTIME', 'Keyword not found')
        filter_zeropoints.append(zeropoint)

        y, x = np.indices(data.shape)  # Create coordinate grids for the image
        distance = np.sqrt((x - x_position)**2 + (y - y_position)**2)  # Distance of each pixel from the star center

        aperture_mask = distance <= radius

        background_mask = (distance > radius + 1) & (distance <= radius + 10)
        background_values = data[background_mask]  # Pixels outside the aperture
        background_median = np.median(background_values)  # Median background level

        aperture_flux = np.sum(data[aperture_mask] - background_median)

        pixel_sums.append(aperture_flux)
        flux = gain*aperture_flux/exptime
        fluxes.append(flux)

    return fluxes, filter_zeropoints, pixel_sums


def apparent_mag(flux, zeropoint):
    """Calculate the apparent magnitude."""
    return -2.5 * np.log10(flux) + zeropoint


def load_color_index_data(file_path):
    """Load info from the color index csv file."""
    return pd.read_csv(file_path)


csv_data = load_color_index_data('LCO_bb_color_indices.csv')


def calculate_temperature(color_index, color_index_type, data):
    """
    Calculate the temperature for a given color index and color index type using the data from the CSV file. color_index could be g-r.
    """
    # Ensure the color index type is valid
    if color_index_type not in data.columns:
        raise ValueError(f"Invalid color index type: {color_index_type}. Must be one of {data.columns[1:]}")
    
    # Find the closest color index in the data
    closest_row = data.iloc[(data[color_index_type] - color_index).abs().argsort()[:1]]
    return closest_row['Temperature (K)'].values[0]


def relative_to_absolute_magnitude(apparent_mag, distance_pc):
    """
    Convert apparent magnitude to absolute magnitude.
    """
    if distance_pc <= 0:
        raise ValueError("Distance must be positive.")
    # absolute_mag = apparent_mag - 5*np.log10(distance_pc/10) # equivalent
    absolute_mag = apparent_mag - 5 * np.log10(distance_pc) + 5
    return absolute_mag


def calculate_luminosity(absolute_mag, solar_luminosity=3.846e26, solar_absolute_mag=4.83):
    """
    Calculate the luminosity of a star given its absolute magnitude.
    
    Parameters:
        absolute_mag (float): The absolute magnitude of the star.
        solar_luminosity (float): Luminosity of the Sun in watts (default: 3.828e26 W).
        solar_absolute_mag (float): Absolute magnitude of the Sun (default: 4.83).
    
    Returns:
        float: Luminosity of the star in watts.
    """
    luminosity = solar_luminosity * 10**(0.4 * (solar_absolute_mag - absolute_mag))

    return luminosity


def calculate_all(file_root, star_name, x_position, y_position, radius, distance):
    star_fluxes = calculate_flux(file_root, star_name, x_position, y_position, radius)[0]
    filter_zeropoints = calculate_flux(file_root, star_name, x_position, y_position, radius)[1]
    apparent_mags = np.zeros(3)
    absolute_mags = np.zeros(3)
    luminosities = np.zeros(3)

    for i in range(3):
        apparent_mags[i] = apparent_mag(star_fluxes[i], filter_zeropoints[i])
        absolute_mags[i] = relative_to_absolute_magnitude(apparent_mags[i], distance)
        luminosities[i] = calculate_luminosity(absolute_mags[i])

    index_gr = apparent_mags[0] - apparent_mags[1]
    index_ri = apparent_mags[1] - apparent_mags[2]
    index_gi = apparent_mags[0] - apparent_mags[2]
    index_values = [index_gr, index_ri, index_gi]

    temperatures = np.zeros(3)
    for i in range(3):
        temperatures[i] = calculate_temperature(index_values[i], color_index_types[i], csv_data)

    average_temp = np.mean(temperatures)
    temp_std = np.std(temperatures)

    return star_fluxes, apparent_mags, absolute_mags, luminosities, temperatures, average_temp, temp_std
