import importlib
import tempfile


def load_file(mname, fname):
    """Loads a Python source file into a module."""
    loader = importlib.machinery.SourceFileLoader(mname, fname)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def load_text(mname, text):
    """Loads Python source text into a module."""
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py") as stream:
        stream.write(text)
        stream.flush()
        return load_file(mname, stream.name)
