import argparse
import time
from collections import Counter
import itertools as itt
from . import dataset, loader, filters


description = """Check validity of dataset and measure I/O speed.

This command lets you quickly check whether a WebDataset is valid (i.e., has no
repeated keys, which happens if you accidentally store the files unsorted).

It will also compute I/O speeds in terms of samples per second and bytes
per second, letting you check and compare loader speeds for different storage
devices and preprocessing optinos.

For testing I/O pipeline performance, using the `-l` option, you can load
an arbitrary `.py` file that contains a function definition for `make_dataset(url)`.

By default, this only loads up to 1000 samples; you can adjust this number with
the `-c` argument; `-c -1` means loading all samples in every shard.

Examples:

    python -m torch.utils.data.webdataset.bench -c 100 'pipe:gsutil cat gs://nvdata-ytsamples/yt8m-clips-000000.tar'

    python -m torch.utils.data.webdataset.bench -c 100 'pipe:curl -s -L https://storage.googleapis.com/nvdata-ytsamples/yt8m-clips-000000.tar'

"""


class TotalSize:
    """Estimate the total size and count of data records."""

    def __init__(self):
        self.count = 0
        self.total = 0

    def __call__(self, sample):
        self.count += 1
        self.total += sum(len(x) for x in sample.values())
        return sample


def main(args):
    for shard in args.shards:
        print()
        print("===", shard)
        totals = TotalSize()
        if args.load != "":
            dsmod = loader.load_file("dsmod", args.load)
            ds = dsmod.make_dataset(shard)
            ds.pipeline = ds.pipeline[:1] + [filters.map(totals)] + ds.pipeline[1:]
        else:
            ds = dataset.Dataset(shard)
            ds.map(totals)
        keys = set()
        skeys = Counter()
        delta = None
        start = None
        for i, sample in itt.islice(enumerate(ds), 1, 1 + args.count):
            assert sample["__key__"] not in keys, "bad shard: detected duplicate keys"
            if i == 1:
                start = time.time()
            keys = tuple(sorted(set(sample.keys())))
            skeys.update([keys])
        delta = time.time() - start
        print()
        print(f"#samples/sec: {totals.count/delta:15.2f}")
        print(f"#bytes/sec:   {totals.total/delta:15.2f}")
        print()
        print("sample types:")
        stats = list(skeys.most_common())
        for key, count in stats:
            print(f"{count:9d} {key}")

        if len(stats) > 1:
            print()
            print("WARNING: multiple different sample types found")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-c", "--count", type=int, default=1000)
    parser.add_argument("-l", "--load", default="")
    parser.add_argument("shards", nargs="+")
    args = parser.parse_args()
    if args.count < 0:
        args.count = 999999999
    main(args)
