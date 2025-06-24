from connectoviz.config import DATA_DIR, PROJ_ROOT


def test_project_root_exists():
    """
    Test if the project root directory exists.
    """
    assert PROJ_ROOT.exists(), f"Project root directory {PROJ_ROOT} does not exist."


def test_data_directory_exists():
    """
    Test if the data directory exists.
    """
    assert DATA_DIR.exists(), f"Data directory {DATA_DIR} does not exist."


def test_data_directory_is_directory():
    """Test if the data directory is indeed a directory."""
    assert DATA_DIR.is_dir(), f"Data directory {DATA_DIR} is not a directory."


def test_data_directory_not_empty():
    """
    Test if the data directory is not empty.
    """
    assert any(DATA_DIR.iterdir()), f"Data directory {DATA_DIR} is empty."


def test_data_directory_contains_expected_files():
    """Test if the data directory contains expected files."""
    expected_files = [
        "atlases",
    ]
    for file in expected_files:
        assert (
            DATA_DIR / file
        ).exists(), f"Expected file {file} not found in {DATA_DIR}."
