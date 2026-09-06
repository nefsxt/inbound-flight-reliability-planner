"""
Derives an airline identity from the OpenSky `callsign` field so we can
model carrier-specific behavior on top of the route-level baseline.

Why this exists: OpenSky gives us `icao24` (the airframe) and `callsign`
(the flight identifier the aircraft is squawking), but never an explicit
"airline" column. Airline callsigns follow the ICAO convention of a
3-letter operator prefix + flight number (e.g. "AEE1421" = Aegean flight
1421, "THY1ZC" = Turkish Airlines). We only need the prefix.

This mapping is NOT exhaustive -- it covers carriers that plausibly serve
the MVP routes (Munich/Istanbul/Heathrow/Frankfurt/Paris -> Athens or
Thessaloniki) plus a handful of common European carriers. An unmapped
prefix isn't treated as an error: it's kept as its raw 3-letter code so it
still works as a modeling feature (XGBoost just sees another category),
and the dashboard labels it "Unmapped carrier code" rather than guessing.

"""
import re

AIRLINE_NAMES = {
    "AEE": "Aegean Airlines",
    "OAL": "Olympic Air",
    "THY": "Turkish Airlines",
    "PGT": "Pegasus Airlines",
    "SXS": "SunExpress",
    "DLH": "Lufthansa",
    "CFG": "Condor",
    "EWG": "Eurowings",
    "BAW": "British Airways",
    "SHT": "British Airways (Shuttle)",
    "AFR": "Air France",
    "EZY": "easyJet",
    "EJU": "easyJet Europe",
    "RYR": "Ryanair",
    "WZZ": "Wizz Air",
    "VLG": "Vueling",
    "TAP": "TAP Air Portugal",
    "KLM": "KLM",
    "SWR": "Swiss International Air Lines",
    "AUA": "Austrian Airlines",
    "IBE": "Iberia",
    "AZA": "ITA Airways",
    "TVF": "Transavia France",
    "TRA": "Transavia",
    "VOE": "Volotea",
}

# Matches a leading run of 2-4 letters (ICAO operator prefixes are
# usually 3, occasionally different) before the numeric flight suffix.
_PREFIX_RE = re.compile(r"^([A-Z]{2,4})")


def extract_airline_code(callsign) -> str | None:
    """
    'AEE1421 ' -> 'AEE'. Returns None for missing/unparseable callsigns
    (private/GA aircraft squawking a registration instead of an airline
    callsign, or a blank field) rather than raising, since this runs
    row-by-row over real-world data that will sometimes be messy.
    """
    if callsign is None:
        return None
    code = str(callsign).strip().upper()
    if not code or code == "NAN":
        return None
    match = _PREFIX_RE.match(code)
    if not match:
        return None
    return match.group(1)


def airline_label(code) -> str:
    """Human-readable label for the dashboard; never raises on an unknown/NaN code."""
    if code is None:
        return "Unknown carrier"
    try:
        import math
        if isinstance(code, float) and math.isnan(code):
            return "Unknown carrier"
    except TypeError:
        pass
    return AIRLINE_NAMES.get(code, f"Unmapped carrier ({code})")
