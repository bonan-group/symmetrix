from pathlib import Path


def test_cached_model_path_uses_external_cache_and_reuses_download(monkeypatch, tmp_path):
    from model_downloads import cached_model_path

    cache_dir = tmp_path / "cache"
    repo_cache = Path(__file__).parent / "model-cache"
    calls = []

    def fake_urlretrieve(url, destination):
        calls.append((url, Path(destination)))
        Path(destination).write_text("model")
        return str(destination), None

    monkeypatch.setenv("SYMMETRIX_TEST_MODEL_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr("model_downloads.urlretrieve", fake_urlretrieve)

    first = cached_model_path("example.model", "https://example.invalid/example.model")
    second = cached_model_path("example.model", "https://example.invalid/example.model")

    assert first == cache_dir / "example.model"
    assert second == first
    assert first.read_text() == "model"
    assert calls == [("https://example.invalid/example.model", cache_dir / "example.model")]
    assert repo_cache not in first.parents


def test_cached_model_path_accepts_existing_env_override(monkeypatch, tmp_path):
    from model_downloads import cached_model_path

    override = tmp_path / "local.model"
    override.write_text("local")
    monkeypatch.setenv("SYMMETRIX_MACEFIELD_MODEL", str(override))

    path = cached_model_path(
        "MACEField-MH-0-omat-dielectric.model",
        "https://example.invalid/MACEField-MH-0-omat-dielectric.model",
        env_var="SYMMETRIX_MACEFIELD_MODEL",
    )

    assert path == override
