import tempfile
from torch.utils.data.webdataset import gopen
from torch.testing._internal.common_utils import TestCase, run_tests


class WebdatasetGopenTests(TestCase):
    def test_file(self):
        with tempfile.TemporaryDirectory() as work:
            with gopen.gopen(f"{work}/temp1", "wb") as stream:
                stream.write(b"abc")
            with gopen.gopen(f"{work}/temp1", "rb") as stream:
                assert stream.read() == b"abc"


    def test_pipe(self):
        with tempfile.TemporaryDirectory() as work:
            with gopen.gopen(f"pipe:cat > {work}/temp2", "wb") as stream:
                stream.write(b"abc")
            with gopen.gopen(f"pipe:cat {work}/temp2", "rb") as stream:
                assert stream.read() == b"abc"


if __name__ == '__main__':
    run_tests()
