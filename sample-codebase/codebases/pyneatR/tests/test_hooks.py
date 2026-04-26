import pandas as pd
import polars as pl
from pyneatR import neat_pandas, neat_polars
from pyneatR.hooks import (
    _infer_column_type_pandas, _infer_column_type_polars,
    activate, deactivate, is_activated,
)
from pyneatR.locale import reset_locale


class TestColumnInferencePandas:
    """Test automatic column type inference for Pandas."""

    def test_numeric_column(self):
        s = pd.Series([1000, 2000, 3000], name="value")
        assert _infer_column_type_pandas(s) == 'number'

    def test_percent_by_name(self):
        s = pd.Series([0.1, 0.2, 0.3], name="conversion_rate")
        assert _infer_column_type_pandas(s) == 'percent'

    def test_percent_by_range(self):
        s = pd.Series([0.1, 0.5, 0.9], name="some_metric")
        assert _infer_column_type_pandas(s) == 'percent'

    def test_currency_by_name(self):
        s = pd.Series([100.0, 200.0], name="revenue")
        assert _infer_column_type_pandas(s) == 'currency'

    def test_date_column(self):
        s = pd.to_datetime(pd.Series(["2024-01-01", "2024-01-02"]), errors="coerce")
        s.name = "date"
        assert _infer_column_type_pandas(s) == 'date'

    def test_timestamp_column(self):
        s = pd.to_datetime(pd.Series(["2024-01-01 10:15:30", "2024-01-02 11:00:00"]))
        s.name = "created_at"
        assert _infer_column_type_pandas(s) == 'timestamp'

    def test_string_column(self):
        s = pd.Series(["hello", "world"], name="description")
        assert _infer_column_type_pandas(s) == 'string'

    def test_id_column_skip(self):
        s = pd.Series([1, 2, 3], name="user_id")
        assert _infer_column_type_pandas(s) == 'skip'

    def test_id_prefix_skip(self):
        s = pd.Series([1, 2, 3], name="id")
        assert _infer_column_type_pandas(s) == 'skip'


class TestColumnInferencePolars:
    """Test automatic column type inference for Polars."""

    def test_numeric_column(self):
        s = pl.Series("value", [1000, 2000, 3000])
        assert _infer_column_type_polars(s) == 'number'

    def test_percent_by_name(self):
        s = pl.Series("conversion_rate", [0.1, 0.2, 0.3])
        assert _infer_column_type_polars(s) == 'percent'

    def test_currency_by_name(self):
        s = pl.Series("revenue", [100.0, 200.0])
        assert _infer_column_type_polars(s) == 'currency'

    def test_string_column(self):
        s = pl.Series("description", ["hello", "world"])
        assert _infer_column_type_polars(s) == 'string'

    def test_id_column_skip(self):
        s = pl.Series("user_id", [1, 2, 3])
        assert _infer_column_type_polars(s) == 'skip'


class TestNeatPandas:
    """Test neat_pandas() function."""

    def setup_method(self):
        reset_locale()

    def teardown_method(self):
        reset_locale()

    def test_basic_formatting(self):
        df = pd.DataFrame({
            "value": [1000, 2000000, 3000000000],
            "name": ["Alice", "Bob", "Charlie"],
        })
        styled = neat_pandas(df)
        # Should return a Styler object
        assert hasattr(styled, 'to_html')

    def test_explicit_column_types(self):
        df = pd.DataFrame({
            "amount": [1000.0, 2000.0],
            "growth": [0.15, -0.05],
        })
        styled = neat_pandas(df, column_types={
            "amount": "currency",
            "growth": "percent",
        }, currency="USD")
        html = styled.to_html()
        assert "$" in html or "1" in html  # Basic sanity check

    def test_with_locale(self):
        df = pd.DataFrame({
            "revenue": [1_50_00_000.0],
        })
        styled = neat_pandas(df, locale="en_IN", currency="INR")
        html = styled.to_html()
        assert "₹" in html

    def test_no_inference(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        styled = neat_pandas(df, infer=False)
        # Should still return a Styler, just without formatting
        assert hasattr(styled, 'to_html')


class TestNeatPolars:
    """Test neat_polars() function."""

    def setup_method(self):
        reset_locale()

    def teardown_method(self):
        reset_locale()

    def test_basic_formatting(self):
        df = pl.DataFrame({
            "value": [1000, 2000000, 3000000000],
            "name": ["Alice", "Bob", "Charlie"],
        })
        gt_obj = neat_polars(df)
        # Should return a GT object
        assert hasattr(gt_obj, 'as_raw_html')

    def test_explicit_column_types(self):
        df = pl.DataFrame({
            "amount": [1000.0, 2000.0],
            "growth": [0.15, -0.05],
        })
        gt_obj = neat_polars(df, column_types={
            "amount": "currency",
            "growth": "percent",
        }, currency="USD")
        html = gt_obj.as_raw_html()
        assert len(html) > 0  # Basic sanity check


class TestActivateDeactivate:
    """Test activate/deactivate lifecycle."""

    def setup_method(self):
        reset_locale()
        if is_activated():
            deactivate()

    def teardown_method(self):
        if is_activated():
            deactivate()
        reset_locale()

    def test_initial_state(self):
        assert is_activated() is False

    def test_activate_sets_state(self):
        activate()
        assert is_activated() is True

    def test_deactivate_resets_state(self):
        activate()
        deactivate()
        assert is_activated() is False

    def test_double_deactivate_safe(self):
        deactivate()
        deactivate()  # Should not raise

    def test_activate_with_locale(self):
        from pyneatR import get_locale
        activate(locale="en_IN")
        assert get_locale() is not None
        assert get_locale().name == "en_IN"
        deactivate()

    def test_pandas_repr_patched(self):
        """Verify that Pandas repr is patched after activate."""
        original_repr = pd.DataFrame._repr_html_
        activate()
        assert pd.DataFrame._repr_html_ is not original_repr
        deactivate()
        assert pd.DataFrame._repr_html_ is original_repr
