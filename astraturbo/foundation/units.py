"""Physical unit conversion utilities for AstraTurbo.

Provides conversion functions between common engineering units used in
turbomachinery design.
"""

from __future__ import annotations

import math


# --- Angle conversions ---

def deg2rad(degrees: float) -> float:
    """Convert degrees to radians."""
    return degrees * math.pi / 180.0


def rad2deg(radians: float) -> float:
    """Convert radians to degrees."""
    return radians * 180.0 / math.pi


# --- Length conversions ---

def mm2m(mm: float) -> float:
    """Convert millimeters to meters."""
    return mm * 1e-3


def m2mm(m: float) -> float:
    """Convert meters to millimeters."""
    return m * 1e3


def inch2m(inches: float) -> float:
    """Convert inches to meters."""
    return inches * 0.0254


def m2inch(m: float) -> float:
    """Convert meters to inches."""
    return m / 0.0254


# --- Pressure conversions ---

def bar2pa(bar: float) -> float:
    """Convert bar to Pascal."""
    return bar * 1e5


def pa2bar(pa: float) -> float:
    """Convert Pascal to bar."""
    return pa * 1e-5


def atm2pa(atm: float) -> float:
    """Convert standard atmospheres to Pascal."""
    return atm * 101325.0


def pa2atm(pa: float) -> float:
    """Convert Pascal to standard atmospheres."""
    return pa / 101325.0


# --- Angular velocity conversions ---

def rpm2rads(rpm: float) -> float:
    """Convert revolutions per minute to radians per second."""
    return rpm * 2.0 * math.pi / 60.0


def rads2rpm(rads: float) -> float:
    """Convert radians per second to revolutions per minute."""
    return rads * 60.0 / (2.0 * math.pi)
