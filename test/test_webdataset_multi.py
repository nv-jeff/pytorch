from torch.utils.data.webdataset import dataset as wds
from torch.utils.data.webdataset import multi
from torch.testing._internal.common_utils import TestCase, run_tests

import os.path


local_data = "webdataset_testdata/imagenet-000000.tgz"

def identity(x):
    return x


def count_samples_tuple(source, *args, n=1000):
    count = 0
    for i, sample in enumerate(iter(source)):
        if i >= n:
            break
        assert isinstance(sample, (tuple, dict, list)), (type(sample), sample)
        for f in args:
            assert f(sample)
        count += 1
    return count


def count_samples(source, *args, n=1000):
    count = 0
    for i, sample in enumerate(iter(source)):
        if i >= n:
            break
        for f in args:
            assert f(sample)
        count += 1
    return count


class WebdatasetMultiTests(TestCase):
    def test_multi(self):
        for k in [1, 4, 17]:
            urls = [f"pipe:cat {local_data} # {i}" for i in range(k)]
            ds = wds.Dataset(urls).decode().shuffle(5).to_tuple("png;jpg cls")
            mds = multi.MultiDataset(ds, workers=4)
            assert count_samples_tuple(mds) == 47*k

if __name__ == '__main__':
    run_tests()
