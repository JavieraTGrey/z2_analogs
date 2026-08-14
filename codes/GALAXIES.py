"""
Central storage for spectra info of all galaxies in the sample.
Import this module and use get_galaxy('gal_ID') to retrieve a
galaxy's dictionary.
"""

import pandas


DATA_DIR = '/Users/javieratoro/Desktop/thesis/BAADE_DATA/testing/'


GALAXIES = {
    'J0020': {
        'DIR': f'{DATA_DIR}10-J0020/',
        'FILES': ['no_very_flats/J0020_NO_VERY_FLATS_tellcorr.fits'],
        'redshift': 0.106,
        'names': ['J0020+0030'],
        'mass': 9.6
    },

    'J0203': {
        'DIR': f'{DATA_DIR}25-J0203/',
        'FILES': ['no_very_flats/J0203_NO_VERY_FLATS_tellcorr.fits',
                  'twilight/J0203_TWILIGHT_tellcorr.fits'],
        'redshift': 0.156,
        'names': ['J0203+0035'],
        'mass': 9.96
    },

    'J0243': {
        'DIR': f'{DATA_DIR}28-J0243/',
        'FILES': ['no_very_flats/J0243_NO_VERY_FLATS_tellcorr.fits',
                  'twilight/J0243_TWILIGHT_tellcorr.fits'],
        'redshift': 0.134,
        'names': ['J0243+0111'],
        'mass': 9.7
    },

    'J0328': {
        'DIR': f'{DATA_DIR}35-J0328/',
        'FILES': ['no_very_flats/J0328_NO_VERY_FLATS_tellcorr.fits',
                  'twilight/J0328_TWILIGHT_tellcorr.fits'],
        'redshift': 0.086,
        'names': ['J0328+0031'],
        'mass': 9.8
    },

    'J0333': {
        'DIR': f'{DATA_DIR}36-J0033/',
        'FILES': ['no_very_flats/J0033_NO_VERY_FLATS_tellcorr.fits',
                  'twilight/J0033_TWILIGHT_tellcorr.fits'],
        'redshift': 0.194,
        'names': ['J0333+0017'],
        'mass': 9.9
        },

    'J0404': {
        'DIR': f'{DATA_DIR}38-J0404/',
        'FILES': ['no_very_flats/J0404_NO_VERY_FLATS_tellcorr.fits',
                  'twilight/J0404_TWILIGHT_tellcorr.fits'],
        'redshift': 0.066,
        'names': ['J0404+0538'],
        'mass': 10.2
        },

    'J2204': {
        'DIR': f'{DATA_DIR}2-J2204/',
        'FILES': ['no_very_flats/J2204_NO_VERY_tellcorr.fits',
                  'twilight/J2204_TWILIGHT_tellcorr.fits'],
        'redshift': 0.185,
        'names': ['J2204+0058'],
        'mass': 10.16
        },

    'J2258': {
        'DIR': f'{DATA_DIR}6-J2258/',
        'FILES': ['twilight/J2258_TWILIGHT_tellcorr.fits',
                  'no_very_flats/J2258_NO_VERY_FLATS_tellcorr.fits'],
        'redshift': 0.094,
        'names': ['J2258+0056'],
        'mass': 9.6
        },

    'J2336': {
        'DIR': f'{DATA_DIR}7-J2336/',
        'FILES': ['no_very_flats/J2336_NO_VERY_BLUE_tellcorr.fits',
                  'twilight/J2336_TWILIGHT_tellcorr.fits'],
        'redshift': 0.17047114835326904,
        'names': ['J2336-0042'],
        'mass': 9.9
    }
}


def get_galaxy(key: str):
    """
    Retrieve the dictionary for a given galaxy.

    Parameters
    ----------
    key : str
        Galaxy identifier as used in the GALAXIES dict (e.g. 'J0328').

    Returns
    -------
    dict
        The galaxy's data dictionary.

    Raises
    ------
    KeyError
        If the galaxy key is not found.
    """
    try:
        return GALAXIES[key]
    except KeyError:
        raise KeyError(f"Galaxy '{key}' not found.")


def load_galaxies(dict: pandas.DataFrame):
    """
    Add galaxies from a pandas DataFrame to the GALAXIES dictionary.
    Has to have this format:
    key | DIRECTIONARY | Name of FILES | redshift | names for object | mass
    """
    for _, row in dict.iterrows():
        key = row['key']
        GALAXIES[key] = {
            'DIR': row['DIR'],
            'FILES': row['FILES'].split(','),
            'redshift': row['redshift'],
            'names': row['names'].split(','),
            'mass': row['mass']
        }
    return


def list_galaxies():
    """Return a list of all galaxy keys currently stored."""
    return list(GALAXIES.keys())


if __name__ == '__main__':
    # Quick sanity check when running this file directly
    print("Available galaxies:", list_galaxies())
    print(get_galaxy('J0328'))
