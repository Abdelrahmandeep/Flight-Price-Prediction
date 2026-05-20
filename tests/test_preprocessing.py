"""
test_preprocessing.py — اختبارات وحدة لدوال المعالجة المشتركة
=============================================================
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# إضافة المجلد الأصلي للمسار حتى يمكن استيراد preprocessing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocessing import (
    extract_date_features,
    convert_duration,
    clean_and_extract_features,
    STOPS_MAP,
)


# ──────────────────────────────────────────────
# اختبارات extract_date_features
# ──────────────────────────────────────────────
class TestExtractDateFeatures:
    def test_basic_date_extraction(self):
        """يستخرج يوم وشهر وسنة من تاريخ الرحلة."""
        df = pd.DataFrame({"Date_of_Journey": ["24/03/2019"]})
        result = extract_date_features(df)
        assert result["Journey_Day"].iloc[0] == 24
        assert result["Journey_Month"].iloc[0] == 3
        assert result["Journey_Year"].iloc[0] == 2019

    def test_date_column_dropped(self):
        """يتأكد من حذف العمود الأصلي بعد الاستخراج."""
        df = pd.DataFrame({"Date_of_Journey": ["01/01/2020"]})
        result = extract_date_features(df)
        assert "Date_of_Journey" not in result.columns

    def test_no_date_column(self):
        """لا يحدث خطأ إذا لم يكن عمود التاريخ موجوداً."""
        df = pd.DataFrame({"Other": [1, 2, 3]})
        result = extract_date_features(df)
        assert "Other" in result.columns
        assert len(result) == 3

    def test_does_not_modify_original(self):
        """يتأكد من أن الدالة لا تعدّل DataFrame الأصلي."""
        df = pd.DataFrame({"Date_of_Journey": ["15/06/2021"]})
        _ = extract_date_features(df)
        assert "Date_of_Journey" in df.columns


# ──────────────────────────────────────────────
# اختبارات convert_duration
# ──────────────────────────────────────────────
class TestConvertDuration:
    def test_hours_and_minutes(self):
        assert convert_duration("2h 30m") == 150

    def test_hours_only(self):
        assert convert_duration("5h") == 300

    def test_minutes_only(self):
        assert convert_duration("45m") == 45

    def test_zero_duration(self):
        assert convert_duration("0h") == 0

    def test_whitespace_handling(self):
        assert convert_duration("  3h  15m  ") == 195

    def test_single_hour(self):
        assert convert_duration("1h") == 60


# ──────────────────────────────────────────────
# اختبارات clean_and_extract_features
# ──────────────────────────────────────────────
class TestCleanAndExtractFeatures:
    def _make_sample_df(self):
        return pd.DataFrame({
            "Dep_Time": ["14:30"],
            "Duration": ["2h 30m"],
            "Total_Stops": ["1 stop"],
            "Airline": ["IndiGo"],
            "Price": [5000],
        })

    def test_dep_hour_minute(self):
        """يستخرج ساعة ودقيقة الإقلاع."""
        df = self._make_sample_df()
        result = clean_and_extract_features(df)
        assert result["Dep_Hour"].iloc[0] == 14
        assert result["Dep_Minute"].iloc[0] == 30

    def test_duration_minutes(self):
        """يحوّل Duration إلى دقائق."""
        df = self._make_sample_df()
        result = clean_and_extract_features(df)
        assert result["Duration_Minutes"].iloc[0] == 150

    def test_total_stops_mapping(self):
        """يحوّل Total_Stops من نص إلى رقم."""
        df = self._make_sample_df()
        result = clean_and_extract_features(df)
        assert result["Total_Stops"].iloc[0] == 1

    def test_non_stop_mapping(self):
        df = pd.DataFrame({
            "Total_Stops": ["non-stop"],
            "Dep_Time": ["08:00"],
            "Duration": ["1h"],
        })
        result = clean_and_extract_features(df)
        assert result["Total_Stops"].iloc[0] == 0

    def test_already_numeric_stops(self):
        """لا يعدّل Total_Stops إذا كان رقمياً بالفعل."""
        df = pd.DataFrame({
            "Total_Stops": [2],
            "Dep_Time": ["10:00"],
            "Duration": ["3h"],
        })
        result = clean_and_extract_features(df)
        assert result["Total_Stops"].iloc[0] == 2

    def test_does_not_modify_original(self):
        """يتأكد من أن الدالة لا تعدّل DataFrame الأصلي."""
        df = self._make_sample_df()
        _ = clean_and_extract_features(df)
        assert "Duration_Minutes" not in df.columns


# ──────────────────────────────────────────────
# اختبارات STOPS_MAP
# ──────────────────────────────────────────────
class TestStopsMap:
    def test_all_stops_present(self):
        expected_keys = {"non-stop", "1 stop", "2 stops", "3 stops", "4 stops"}
        assert set(STOPS_MAP.keys()) == expected_keys

    def test_values_are_integers(self):
        for v in STOPS_MAP.values():
            assert isinstance(v, int)

    def test_values_order(self):
        assert STOPS_MAP["non-stop"] == 0
        assert STOPS_MAP["4 stops"] == 4
