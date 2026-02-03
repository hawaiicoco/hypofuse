# ex02: ROVER confusion

Build synthetic n-best lists, run progressive alignment, fuse with two
policies (majority and score\_weighted), build a confusion network,
and extract the 1-best with `minimal_cut_one_best`.

## Usage

```
python examples/ex02_rover_confusion/run.py <outdir>
```

## Artifacts

- `u0000_cn.json` ... -- one confusion network JSON per utterance
