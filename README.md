A lightweight benchmarking library which:

* uses linear regression to screen off sources of constant error;
* handles benchmarks which must mutate some state;
* has a very simple API!

```rust
use easybench::{bench,bench_env};

// Simple benchmarks are performed with `bench`.
println!("fib 200: {}", bench(|| fib(200) ));
println!("fib 500: {}", bench(|| fib(500) ));

// If a function needs to mutate some state, use `bench_env`.
println!("reverse: {}", bench_env(vec![1,2,3], |xs| xs.reverse()));
println!("sort:    {}", bench_env(vec![1,2,3], |xs| xs.sort()));
```

Running the above yields the following:

```none
fib 200:         38 ns   (R²=1.000, 26053498 iterations in 155 samples)
fib 500:        109 ns   (R²=1.000, 9131585 iterations in 144 samples)
reverse:          3 ns   (R²=0.998, 23684997 iterations in 154 samples)
sort:             3 ns   (R²=0.999, 23684997 iterations in 154 samples)
```

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or
   http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.
