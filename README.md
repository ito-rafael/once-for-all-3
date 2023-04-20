Steps:

1. Get the neural network architectures in the [OFA](https://arxiv.org/abs/1908.09791) search space using the [OFA²](https://arxiv.org/abs/2303.13683) search.

2. For each neural network (default=100), calculate the probability table by appending a Softmax layer on the architectures. Alternatively, one can download and extract the probabilities tables directly.
```console
$ mkdir -p tables/prob_moo
$ wget 'https://drive.google.com/uc?export=download&id=1zmv1k1zzz9GhD0ep25KvYI2BA8l-Zp1x&confirm=t' -O prob_moo.tar.gz
$ wget 'https://drive.google.com/uc?export=download&id=1XdnjsyOA07TfuNca41GtHBrQboFlsUX8&confirm=t' -O tables.tar.gz
$ tar -xvzf  prob_moo.tar.gz --directory tables/prob_moo
$ tar -xvzf tables.tar.gz --directory tables/
```
3. Run the OFA³ algorithm for find efficient ensembles.
