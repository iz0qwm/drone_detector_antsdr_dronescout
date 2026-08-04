"""
Tests for proximity distance calculation and coordinate validation.
"""
import math
import pytest
from services.proximity.calc import (
    haversine_meters,
    is_valid_position,
    bounding_box_filter,
    evaluate_pairs,
)


class TestHaversine:
    """Known-distance haversine validation."""

    def test_rome_to_milan(self):
        # Rome (41.9028, 12.4964) to Milan (45.4642, 9.1900) ≈ 477 km
        d = haversine_meters(41.9028, 12.4964, 45.4642, 9.1900)
        assert 475_000 < d < 480_000

    def test_same_point(self):
        d = haversine_meters(45.0, 12.0, 45.0, 12.0)
        assert d == 0.0

    def test_short_distance(self):
        # ~111m per 0.001 degrees latitude at equator
        d = haversine_meters(0.0, 0.0, 0.001, 0.0)
        assert 110 < d < 112

    def test_known_1km(self):
        # 1 degree latitude ≈ 111.32 km
        d = haversine_meters(45.0, 12.0, 45.009, 12.0)
        assert 990 < d < 1010

    def test_antipodal(self):
        # Half Earth circumference ≈ 20015 km
        d = haversine_meters(0.0, 0.0, 0.0, 180.0)
        assert 20_000_000 < d < 20_100_000


class TestIsValidPosition:
    """Coordinate validation tests."""

    def test_valid_coordinates(self):
        assert is_valid_position(41.9, 12.5) is True

    def test_none_lat(self):
        assert is_valid_position(None, 12.5) is False

    def test_none_lon(self):
        assert is_valid_position(41.9, None) is False

    def test_nan(self):
        assert is_valid_position(float("nan"), 12.5) is False

    def test_inf(self):
        assert is_valid_position(float("inf"), 12.5) is False

    def test_lat_out_of_range(self):
        assert is_valid_position(91.0, 12.5) is False
        assert is_valid_position(-91.0, 12.5) is False

    def test_lon_out_of_range(self):
        assert is_valid_position(41.9, 181.0) is False
        assert is_valid_position(41.9, -181.0) is False

    def test_boundary_values(self):
        assert is_valid_position(90.0, 180.0) is True
        assert is_valid_position(-90.0, -180.0) is True

    def test_zero_zero_drone_rejected(self):
        """ODID sentinel: (0,0) means no GPS fix for drones."""
        assert is_valid_position(0.0, 0.0, source_type="drone") is False

    def test_zero_zero_aircraft_accepted(self):
        """Aircraft sources don't use (0,0) as sentinel."""
        assert is_valid_position(0.0, 0.0, source_type="aircraft") is True

    def test_zero_lat_nonzero_lon_drone_accepted(self):
        """Only exact (0,0) is rejected for drones."""
        assert is_valid_position(0.0, 10.0, source_type="drone") is True

    def test_string_conversion(self):
        assert is_valid_position("41.9", "12.5") is True

    def test_invalid_string(self):
        assert is_valid_position("abc", "12.5") is False


class TestBoundingBoxFilter:
    """Bounding-box pre-filter tests."""

    def test_inside(self):
        targets = [{"latitude": 41.91, "longitude": 12.50}]
        result = bounding_box_filter(41.90, 12.50, 5000, targets)
        assert len(result) == 1

    def test_outside(self):
        targets = [{"latitude": 42.90, "longitude": 12.50}]
        result = bounding_box_filter(41.90, 12.50, 5000, targets)
        assert len(result) == 0

    def test_missing_coords_skipped(self):
        targets = [{"latitude": None, "longitude": 12.50}]
        result = bounding_box_filter(41.90, 12.50, 5000, targets)
        assert len(result) == 0

    def test_multiple_mixed(self):
        targets = [
            {"latitude": 41.91, "longitude": 12.50},  # close
            {"latitude": 50.00, "longitude": 12.50},  # far
            {"latitude": 41.895, "longitude": 12.505},  # close
        ]
        result = bounding_box_filter(41.90, 12.50, 5000, targets)
        assert len(result) == 2


class TestEvaluatePairs:
    """Pair evaluation tests."""

    def test_basic_pair(self):
        drones = [{"serial": "DRN1", "lat": 41.90, "lon": 12.50}]
        targets = [{"track_id": "AC1", "latitude": 41.91, "longitude": 12.50}]
        pairs = evaluate_pairs(drones, targets, 10000)
        assert len(pairs) == 1
        assert pairs[0]["drone_id"] == "DRN1"
        assert pairs[0]["target_id"] == "AC1"
        assert pairs[0]["distance_m"] > 0

    def test_out_of_radius(self):
        drones = [{"serial": "DRN1", "lat": 41.90, "lon": 12.50}]
        targets = [{"track_id": "AC1", "latitude": 42.90, "longitude": 12.50}]
        pairs = evaluate_pairs(drones, targets, 10000)
        assert len(pairs) == 0

    def test_invalid_drone_skipped(self):
        drones = [{"serial": "DRN1", "lat": 0.0, "lon": 0.0}]  # ODID sentinel
        targets = [{"track_id": "AC1", "latitude": 41.90, "longitude": 12.50}]
        pairs = evaluate_pairs(drones, targets, 10000)
        assert len(pairs) == 0

    def test_sorted_by_distance(self):
        drones = [{"serial": "DRN1", "lat": 41.90, "lon": 12.50}]
        targets = [
            {"track_id": "FAR", "latitude": 41.95, "longitude": 12.50},
            {"track_id": "NEAR", "latitude": 41.905, "longitude": 12.50},
        ]
        pairs = evaluate_pairs(drones, targets, 10000)
        assert len(pairs) == 2
        assert pairs[0]["target_id"] == "NEAR"
        assert pairs[1]["target_id"] == "FAR"

    def test_empty_drones(self):
        pairs = evaluate_pairs([], [{"track_id": "AC1", "latitude": 41.9, "longitude": 12.5}], 10000)
        assert pairs == []

    def test_empty_targets(self):
        pairs = evaluate_pairs([{"serial": "D1", "lat": 41.9, "lon": 12.5}], [], 10000)
        assert pairs == []

    def test_multiple_drones(self):
        drones = [
            {"serial": "D1", "lat": 41.90, "lon": 12.50},
            {"serial": "D2", "lat": 41.91, "lon": 12.50},
        ]
        targets = [{"track_id": "AC1", "latitude": 41.905, "longitude": 12.50}]
        pairs = evaluate_pairs(drones, targets, 10000)
        assert len(pairs) == 2
        # Both drones should have pairs with the target
        drone_ids = {p["drone_id"] for p in pairs}
        assert drone_ids == {"D1", "D2"}
