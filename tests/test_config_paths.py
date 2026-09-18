import os
import config


def test_route_key():
    assert config.route_key("EDDF", "LGTS") == "EDDF_LGTS"


def test_route_raw_dir():
    assert config.route_raw_dir("EDDF", "LGTS") == os.path.join("data/raw", "EDDF_LGTS")


def test_route_processed_dir():
    assert config.route_processed_dir("EDDF", "LGTS") == os.path.join(
        "data/processed", "EDDF_LGTS"
    )

def test_route_arrivals_dir():
    assert config.route_arrivals_dir("EDDF", "LGTS") == os.path.join(
        "data/raw", "EDDF_LGTS", "arrivals"
    )


def test_route_weather_path():
    assert config.route_weather_path("EDDF", "LGTS") == os.path.join(
        "data/processed", "EDDF_LGTS", "weather.parquet"
    )


def test_route_processed_and_flights_paths():
    assert config.route_processed_dir("EDDF", "LGTS") == os.path.join("data/processed", "EDDF_LGTS")
    assert config.route_flights_path("EDDF", "LGTS") == os.path.join(
        "data/processed", "EDDF_LGTS", "flights.parquet"
    )


def test_features_path():
    assert config.features_path() == os.path.join("data/processed", "features.parquet")


def test_routes_is_a_list_of_two_tuples():
    assert isinstance(config.ROUTES, list)
    assert len(config.ROUTES) >= 1
    for pair in config.ROUTES:
        assert len(pair) == 2
        origin, dest = pair
        assert origin in config.ORIGIN_AIRPORTS
        assert dest in config.DEST_AIRPORTS
