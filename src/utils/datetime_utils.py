# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
    Collection of utility functions for working with datetimes to ensure consistent formatting.

"""
from datetime import datetime

def current_datetime() -> dict:
    dt_str = datetime.now().strftime("%Y-%m-%dT%H:%M")
    day = datetime.now().isoweekday()
    month = datetime.now().strftime("%B")
    year = datetime.now().year
    return {'ISODateTime': dt_str, 'Day': day, 'Month': month, 'Year': year}