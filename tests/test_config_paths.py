import os
import config


def test_route_key():
    assert config.route_key("EDDF", "LGTS") == "EDDF_LGTS"


def test_route_raw_dir():
    assert config.route_raw_dir("EDDF", "LGTS") == os.path.join("data/raw", "EDDF_LGTS")


def test_route_arrivals_dir_nested_under_raw_dir():
    arrivals_dir = config.route_arrivals_dir("EDDF", "LGTS")
    assert arrivals_dir == os.path.join("data/raw", "EDDF_LGTS", "arrivals")
    assert arrivals_dir.startswith(config.route_raw_dir("EDDF", "LGTS"))


def test_route_weather_path_lives_next_to_arrivals():
    weather_path = config.route_weather_path("EDDF", "LGTS")
    arrivals_dir = config.route_arrivals_dir("EDDF", "LGTS")
    # Both should share the same per-route parent directory -- this is the
    # "cohesive paths" property: one route = one raw folder holding both
    # arrivals/ and weather.parquet, not scattered across the tree.
    assert os.path.dirname(weather_path) == os.path.dirname(arrivals_dir)
    assert weather_path.endswith("weather.parquet")


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
