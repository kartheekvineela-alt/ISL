from signvaani import check_install


class FakeModule:
    def __init__(self, version):
        self.__version__ = version


def fake_import(versions):
    def import_module(name):
        if versions.get(name, "1.0") is None:
            raise ImportError(f"No module named '{name}'")
        return FakeModule(versions.get(name, "1.0"))
    return import_module


def test_python_version():
    assert check_install.check_python((3, 11, 9))[0] == "OK"
    assert check_install.check_python((3, 13, 0))[0] == "WARN"


def test_missing_package_fails_and_unpinned_version_warns():
    results, loaded = check_install.check_packages(
        fake_import({"mediapipe": None, "cv2": "4.10.0", "numpy": "2.1.0"}))
    by_name = {msg.split()[0].rstrip(":"): (status, msg) for status, msg in results}
    assert by_name["mediapipe"][0] == "FAIL" and "pip install mediapipe==0.10.14" in by_name["mediapipe"][1]
    assert by_name["numpy"][0] == "WARN"
    assert by_name["cv2"][0] == by_name["torch"][0] == "OK"
    assert "mediapipe" not in loaded and "torch" in loaded


def test_verdict(monkeypatch, capsys):
    monkeypatch.setattr(check_install, "check_packages", lambda: ([("OK", "torch 2.3.0")], {}))
    assert check_install.main([]) == 0 and capsys.readouterr().out.rstrip().endswith("OK")
    monkeypatch.setattr(check_install, "check_packages", lambda: ([("FAIL", "mediapipe: ImportError")], {}))
    assert check_install.main([]) == 1 and "NOT OK" in capsys.readouterr().out
