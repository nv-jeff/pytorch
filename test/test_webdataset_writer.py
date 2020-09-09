import os
import shutil
import tempfile
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils.data.webdataset import dataset as wds
from torch.utils.data.webdataset import writer



def test_writer(tmpdir):
    with writer.TarWriter(f"{tmpdir}/writer.tar") as sink:
        sink.write(dict(__key__="a", txt="hello", cls="3"))
    os.system(f"ls -l {tmpdir}")
    ftype = os.popen(f"file {tmpdir}/writer.tar").read()
    assert "compress" not in ftype, ftype

    ds = wds.Dataset(f"{tmpdir}/writer.tar")
    for sample in ds:
        assert set(sample.keys()) == set("__key__ txt cls".split())
        break


def test_writer2(tmpdir):
    with writer.TarWriter(f"{tmpdir}/writer2.tgz") as sink:
        sink.write(dict(__key__="a", txt="hello", cls="3"))
    os.system(f"ls -l {tmpdir}")
    ftype = os.popen(f"file {tmpdir}/writer2.tgz").read()
    assert "compress" in ftype, ftype

    ds = wds.Dataset(f"{tmpdir}/writer2.tgz")
    for sample in ds:
        assert set(sample.keys()) == set("__key__ txt cls".split())
        break


def test_writer3(tmpdir):
    with writer.TarWriter(f"{tmpdir}/writer3.tar") as sink:
        sink.write(dict(__key__="a", pth=["abc"], pyd=dict(x=0)))
    os.system(f"ls -l {tmpdir}")
    os.system(f"tar tvf {tmpdir}/writer3.tar")
    ftype = os.popen(f"file {tmpdir}/writer3.tar").read()
    assert "compress" not in ftype, ftype

    ds = wds.Dataset(f"{tmpdir}/writer3.tar").decode()
    for sample in ds:
        assert set(sample.keys()) == set("__key__ pth pyd".split())
        assert isinstance(sample["pyd"], dict)
        assert sample["pyd"] == dict(x=0)
        assert isinstance(sample["pth"], list)
        assert sample["pth"] == ["abc"]


def test_writer_pipe(tmpdir):
    with writer.TarWriter(f"pipe:cat > {tmpdir}/writer3.tar") as sink:
        sink.write(dict(__key__="a", txt="hello", cls="3"))
    os.system(f"ls -l {tmpdir}")
    ds = wds.Dataset(f"pipe:cat {tmpdir}/writer3.tar")
    for sample in ds:
        assert set(sample.keys()) == set("__key__ txt cls".split())
        break


def test_shardwriter(tmpdir):
    def post(fname):
        assert fname is not None

    with writer.ShardWriter(
        f"{tmpdir}/shards-%04d.tar", maxcount=5, post=post, encoder=False
    ) as sink:
        for i in range(50):
            sink.write(dict(__key__=str(i), txt=b"hello", cls=b"3"))

    os.system(f"ls -l {tmpdir}")
    ftype = os.popen(f"file {tmpdir}/shards-0000.tar").read()
    assert "compress" not in ftype, ftype


class WebdatasetWriterTests(TestCase):

    def test_writer(self):
        tmpdir = tempfile.mkdtemp()
        test_writer(tmpdir)
        shutil.rmtree(tmpdir)

    def test_writer2(self):
        tmpdir = tempfile.mkdtemp()
        test_writer2(tmpdir)
        shutil.rmtree(tmpdir)

    def test_writer3(self):
        tmpdir = tempfile.mkdtemp()
        test_writer3(tmpdir)
        shutil.rmtree(tmpdir)

    def test_writer_pipe(self):
        tmpdir = tempfile.mkdtemp()
        test_writer_pipe(tmpdir)
        shutil.rmtree(tmpdir)

    def test_shardwriter(self):
        tmpdir = tempfile.mkdtemp()
        test_shardwriter(tmpdir)
        shutil.rmtree(tmpdir)


if __name__ == '__main__':
    run_tests()
