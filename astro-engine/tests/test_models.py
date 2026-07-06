"""Domain-model tests, ported from the original notebook unittest cells.

The original suite contained a duplicated ``test_from_iso_with_aware``; the
second copy is replaced here with ``test_from_iso_naive_uses_default`` so the
naive-string localisation path is actually covered.
"""
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from astro_engine.models.location import Location
from astro_engine.models.date import Date
from astro_engine.models.date_range import DateRange

KOLKATA = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")
NEW_YORK = ZoneInfo("America/New_York")


class TestLocation(unittest.TestCase):
    def test_valid_location(self):
        loc = Location(latitude=12.97, longitude=77.59, elevation=900.0, timezone=ZoneInfo("Asia/Kolkata"))
        self.assertEqual(loc.latitude, 12.97)
        self.assertEqual(loc.longitude, 77.59)
        self.assertEqual(loc.elevation, 900.0)
        self.assertEqual(loc.timezone.key, "Asia/Kolkata")

    def test_default_timezone(self):
        loc = Location(latitude=0, longitude=0)
        self.assertEqual(loc.timezone.key, "Asia/Kolkata")

    def test_invalid_latitude(self):
        with self.assertRaises(ValueError):
            Location(latitude=91.0, longitude=77.0)

    def test_invalid_longitude(self):
        with self.assertRaises(ValueError):
            Location(latitude=12.0, longitude=200.0)

    def test_invalid_timezone_type(self):
        with self.assertRaises(TypeError):
            Location(latitude=10.0, longitude=10.0, timezone="Asia/Kolkata")  # should be ZoneInfo

    def test_str_repr(self):
        loc = Location(latitude=15.0, longitude=30.0)
        s = str(loc)
        r = repr(loc)
        self.assertIn("Asia/Kolkata", s)
        self.assertIn("Location(latitude=15.0", r)


class TestDate(unittest.TestCase):
    def test_init_with_tz(self):
        dt = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
        d = Date(dt)
        self.assertEqual(d.dt, dt)

    def test_init_without_tz_uses_default(self):
        naive_dt = datetime(2024, 1, 1, 10, 0)
        d = Date(naive_dt)
        self.assertEqual(d.dt.tzinfo, KOLKATA)

    def test_from_iso_with_aware(self):
        d = Date.from_iso("2024-01-01T10:00:00+00:00")
        self.assertEqual(d.dt.tzinfo, UTC)

    def test_from_iso_naive_uses_default(self):
        # Replaces the notebook's duplicated aware test with real coverage of
        # the naive-string branch.
        d = Date.from_iso("2024-01-01T10:00:00")
        self.assertEqual(d.dt.tzinfo, KOLKATA)

    def test_in_timezone(self):
        dt = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
        d = Date(dt)
        d_ny = d.in_timezone(NEW_YORK)
        self.assertEqual(d_ny.dt.tzinfo, NEW_YORK)
        self.assertEqual(d.timestamp(), d_ny.timestamp())  # timestamps match

    def test_eq_and_comparison(self):
        dt1 = Date(datetime(2024, 1, 1, 10, 0, tzinfo=UTC))
        dt2 = Date(datetime(2024, 1, 1, 15, 30, tzinfo=KOLKATA))  # same instant
        dt3 = Date(datetime(2024, 1, 1, 11, 0, tzinfo=UTC))
        self.assertEqual(dt1, dt2)
        self.assertNotEqual(dt1, dt3)
        self.assertLess(dt1, dt3)
        self.assertGreaterEqual(dt2, dt1)

    def test_hash_consistent_with_eq(self):
        dt1 = Date(datetime(2024, 1, 1, 10, 0, tzinfo=UTC))
        dt2 = Date(datetime(2024, 1, 1, 15, 30, tzinfo=KOLKATA))  # same instant
        self.assertEqual(hash(dt1), hash(dt2))
        self.assertEqual(len({dt1, dt2}), 1)

    def test_is_same_wall_time(self):
        dt1 = Date(datetime(2024, 1, 1, 10, 0, tzinfo=KOLKATA))
        dt2 = Date(datetime(2024, 1, 1, 10, 0, tzinfo=NEW_YORK))
        self.assertTrue(dt1.is_same_wall_time(dt2))  # same wall time
        dt3 = Date(datetime(2024, 1, 1, 11, 0, tzinfo=NEW_YORK))
        self.assertFalse(dt1.is_same_wall_time(dt3))

    def test_str_and_repr(self):
        dt = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
        d = Date(dt)
        s = str(d)
        r = repr(d)
        self.assertIn("UTC", s)
        self.assertTrue(r.startswith("Date("))
        self.assertIn("2024-01-01T10:00:00+00:00", r)

    def test_helpers(self):
        dt = datetime(2024, 1, 1, 10, 20, 30, tzinfo=UTC)
        d = Date(dt)
        self.assertEqual(d.date(), dt.date())
        self.assertEqual(d.time(), dt.time())
        self.assertAlmostEqual(d.timestamp(), dt.timestamp())


class TestDateRange(unittest.TestCase):
    def setUp(self):
        self.d1 = Date(datetime(2024, 1, 1, tzinfo=UTC))
        self.d2 = Date(datetime(2024, 1, 5, tzinfo=UTC))

    def test_valid_init(self):
        dr = DateRange(self.d1, self.d2)
        self.assertEqual(dr.start, self.d1)
        self.assertEqual(dr.end, self.d2)

    def test_invalid_init_raises(self):
        with self.assertRaises(ValueError):
            DateRange(self.d2, self.d1)  # end before start

    def test_days(self):
        dr = DateRange(self.d1, self.d2)
        self.assertEqual(dr.days(), 5)

    def test_contains(self):
        dr = DateRange(self.d1, self.d2)
        inside = Date(datetime(2024, 1, 3, tzinfo=UTC))
        outside = Date(datetime(2024, 1, 6, tzinfo=UTC))
        self.assertTrue(dr.contains(inside))
        self.assertFalse(dr.contains(outside))

    def test_overlaps_true(self):
        dr1 = DateRange(self.d1, self.d2)
        dr2 = DateRange(Date(datetime(2024, 1, 3, tzinfo=UTC)),
                        Date(datetime(2024, 1, 7, tzinfo=UTC)))
        self.assertTrue(dr1.overlaps(dr2))
        self.assertTrue(dr2.overlaps(dr1))

    def test_overlaps_false(self):
        dr1 = DateRange(self.d1, self.d2)
        dr2 = DateRange(Date(datetime(2024, 1, 6, tzinfo=UTC)),
                        Date(datetime(2024, 1, 10, tzinfo=UTC)))
        self.assertFalse(dr1.overlaps(dr2))

    def test_iteration(self):
        dr = DateRange(self.d1, self.d2)
        days = list(dr)
        self.assertEqual(len(days), 5)
        self.assertTrue(all(isinstance(d, Date) for d in days))
        self.assertEqual(days[0], self.d1)
        self.assertEqual(days[-1], self.d2)

    def test_str_repr(self):
        dr = DateRange(self.d1, self.d2)
        s = str(dr)
        r = repr(dr)
        self.assertIn("\u2192", s)
        self.assertIn("DateRange(", r)
        self.assertIn("start=", r)
        self.assertIn("end=", r)


if __name__ == "__main__":
    unittest.main()
