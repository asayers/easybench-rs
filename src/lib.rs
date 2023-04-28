/*!
A lightweight micro-benchmarking library which:

* uses linear regression to screen off constant error;
* handles benchmarks which mutate state;
* is very easy to use!

Easybench is designed for benchmarks with a running time in the range `1 ns <
x < 1 ms` - results may be unreliable if benchmarks are very quick or very
slow. It's inspired by [criterion], but doesn't do as much sophisticated
analysis (no outlier detection, no HTML output).

[criterion]: https://hackage.haskell.org/package/criterion

```
use easybench::{bench,bench_env};

# fn fib(_: usize) -> usize { 0 }
#
// Simple benchmarks are performed with `bench`.
println!("fib 200: {}", bench(|| fib(200) ));
println!("fib 500: {}", bench(|| fib(500) ));

// If a function needs to mutate some state, use `bench_env`.
println!("reverse: {}", bench_env(vec![0;100], |xs| xs.reverse() ));
println!("sort:    {}", bench_env(vec![0;100], |xs| xs.sort()    ));
```

Running the above yields the following results:

```none
fib 200:         38 ns   (R²=1.000, 26053497 iterations in 154 samples)
fib 500:        110 ns   (R²=1.000, 9131584 iterations in 143 samples)
reverse:         54 ns   (R²=0.999, 5669992 iterations in 138 samples)
sort:            93 ns   (R²=1.000, 4685942 iterations in 136 samples)
```

Easy! However, please read the [caveats](#caveats) below before using.

# Benchmarking algorithm

An *iteration* is a single execution of your code. A *sample* is a measurement,
during which your code may be run many times. In other words: taking a sample
means performing some number of iterations and measuring the total time.

The first sample we take performs only 1 iteration, but as we continue taking
samples we increase the number of iterations exponentially. We stop when a
global time limit is reached (currently 1 second).

If a benchmark must mutate some state while running, before taking a sample
`n` copies of the initial state are prepared, where `n` is the number of
iterations in that sample.

Once we have the data, we perform OLS linear regression to find out how
the sample time varies with the number of iterations in the sample. The
gradient of the regression line tells us how long it takes to perform a
single iteration of the benchmark. The R² value is a measure of how much
noise there is in the data.

# Caveats

## Caveat 1: Harness overhead

**TL;DR: Compile with `--release`; the overhead is likely to be within the
noise of your benchmark.**

Any work which easybench does once-per-sample is ignored (this is the purpose of the linear
regression technique described above). However, work which is done once-per-iteration *will* be
counted in the final times.

* In the case of [`bench`] this amounts to incrementing the loop counter.
* In the case of [`bench_env`], we also do a lookup into a big vector in
  order to get the environment for that iteration.
* If you compile your program unoptimised, there may be additional overhead.

[`bench`]: fn.bench.html
[`bench_env`]: fn.bench_env.html

The cost of the above operations depend on the details of your benchmark;
namely: (1) how large is the return value? and (2) does the benchmark evict
the environment vector from the CPU cache? In practice, these criteria are only
satisfied by longer-running benchmarks, making these effects hard to measure.

If you have concerns about the results you're seeing, please take a look at
[the inner loop of `bench_env`][source]. The whole library `cloc`s in at
under 100 lines of code, so it's pretty easy to read.

[source]: ../src/easybench/lib.rs.html#229-237

## Caveat 2: Sufficient data

**TL;DR: Measurements are unreliable when code takes too long (> 1 ms) to run.**

Each benchmark collects data for 1 second. This means that in order to
collect a statistically significant amount of data, your code should run
much faster than this.

When inspecting the results, make sure things look statistically
significant. In particular:

* Make sure the number of samples is big enough. More than 100 is probably OK.
* Make sure the R² isn't suspiciously low. It's easy to achieve a high R²
  value when the number of samples is small, so unfortunately the definition
  of "suspiciously low" depends on how many samples were taken.  As a rule
  of thumb, expect values greater than 0.99.

## Caveat 3: Pure functions

**TL;DR: Return enough information to prevent the optimiser from eliminating
code from your benchmark.**

Benchmarking pure functions involves a nasty gotcha which users should be
aware of. Consider the following benchmarks:

```
# use easybench::{bench,bench_env};
#
# fn fib(_: usize) -> usize { 0 }
#
let fib_1 = bench(|| fib(500) );                     // fine
let fib_2 = bench(|| { fib(500); } );                // spoiler: NOT fine
let fib_3 = bench_env(0, |x| { *x = fib(500); } );   // also fine, but ugly
# let _ = (fib_1, fib_2, fib_3);
```

The results are a little surprising:

```none
fib_1:        110 ns   (R²=1.000, 9131585 iterations in 144 samples)
fib_2:          0 ns   (R²=1.000, 413289203 iterations in 184 samples)
fib_3:        109 ns   (R²=1.000, 9131585 iterations in 144 samples)
```

Oh, `fib_2`, why do you lie? The answer is: `fib(500)` is pure, and its
return value is immediately thrown away, so the optimiser replaces the call
with a no-op (which clocks in at 0 ns).

What about the other two? `fib_1` looks very similar, with one exception:
the closure which we're benchmarking returns the result of the `fib(500)`
call. When it runs your code, easybench takes the return value and tricks the
optimiser into thinking it's going to use it for something, before throwing
it away. This is why `fib_1` is safe from having code accidentally eliminated.

In the case of `fib_3`, we actually *do* use the return value: each
iteration we take the result of `fib(500)` and store it in the iteration's
environment. This has the desired effect, but looks a bit weird.
*/

use std::f64;
use std::fmt::{self, Display, Formatter};
use std::time::*;

// Each time we take a sample we increase the number of iterations:
//      iters = ITER_SCALE_FACTOR ^ sample_no
const ITER_SCALE_FACTOR: f64 = 1.1;

// We try to spend this many seconds (roughly) in total on each benchmark.
const BENCH_TIME_LIMIT: Duration = Duration::from_secs(1);

/// Statistics for a benchmark run.
#[derive(Debug, PartialEq, Clone)]
pub struct Stats {
    /// The time, in nanoseconds, per iteration. If the benchmark generated
    /// fewer than 2 samples in the allotted time then this will be NaN.
    pub ns_per_iter: f64,
    /// The coefficient of determination, R².
    ///
    /// This is an indication of how noisy the benchmark was, where 1 is
    /// good and 0 is bad. Be suspicious of values below 0.9.
    pub goodness_of_fit: f64,
    /// How many times the benchmarked code was actually run.
    pub iterations: usize,
    /// How many samples were taken (ie. how many times we allocated the
    /// environment and measured the time).
    pub samples: usize,
}

impl Display for Stats {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        if self.ns_per_iter.is_nan() {
            write!(
                f,
                "Only generated {} sample(s) - we can't fit a regression line to that! \
                 Try making your benchmark faster.",
                self.samples
            )
        } else {
            let per_iter = Duration::from_nanos(self.ns_per_iter as u64);
            let per_iter = format!("{:?}", per_iter);
            write!(
                f,
                "{:>11} (R²={:.3}, {} iterations in {} samples)",
                per_iter, self.goodness_of_fit, self.iterations, self.samples
            )
        }
    }
}

/// Run a benchmark.
///
/// The return value of `f` is not used, but we trick the optimiser into
/// thinking we're going to use it. Make sure to return enough information
/// to prevent the optimiser from eliminating code from your benchmark! (See
/// the module docs for more.)
pub fn bench<F, O>(mut f: F) -> Stats
where
    F: FnMut() -> O,
{
    bench_env((), |_| f())
}

/// Run a benchmark with an environment.
///
/// The value `env` is a clonable prototype for the "benchmark
/// environment". Each iteration receives a freshly-cloned mutable copy of
/// this environment. The time taken to clone the environment is not included
/// in the results.
///
/// Nb: it's very possible that we will end up allocating many (>10,000)
/// copies of `env` at the same time. Probably best to keep it small.
///
/// See `bench` and the module docs for more.
///
/// ## Overhead
///
/// Every iteration, `bench_env` performs a lookup into a big vector in
/// order to get the environment for that iteration. If your benchmark
/// is memory-intensive then this could, in the worst case, amount to a
/// systematic cache-miss (ie. this vector would have to be fetched from
/// DRAM at the start of every iteration). In this case the results could be
/// affected by a hundred nanoseconds. This is a worst-case scenario however,
/// and I haven't actually been able to trigger it in practice... but it's
/// good to be aware of the possibility.
pub fn bench_env<F, I, O>(env: I, f: F) -> Stats
where
    F: FnMut(&mut I) -> O,
    I: Clone,
{
    bench_gen_env(move || env.clone(), f)
}

/// Run a benchmark with a generated environment.
///
/// The function `gen_env` creates the "benchmark environment" for the
/// computation. Each iteration receives a freshly-created environment. The
/// time taken to create the environment is not included in the results.
///
/// Nb: it's very possible that we will end up generating many (>10,000)
/// copies of `env` at the same time. Probably best to keep it small.
///
/// See `bench` and the module docs for more.
///
/// ## Overhead
///
/// Every iteration, `bench_gen_env` performs a lookup into a big vector
/// in order to get the environment for that iteration. If your benchmark
/// is memory-intensive then this could, in the worst case, amount to a
/// systematic cache-miss (ie. this vector would have to be fetched from
/// DRAM at the start of every iteration). In this case the results could be
/// affected by a hundred nanoseconds. This is a worst-case scenario however,
/// and I haven't actually been able to trigger it in practice... but it's
/// good to be aware of the possibility.
pub fn bench_gen_env<G, F, I, O>(mut gen_env: G, mut f: F) -> Stats
where
    G: FnMut() -> I,
    F: FnMut(&mut I) -> O,
{
    let mut data = Vec::new();
    // The time we started the benchmark (not used in results)
    let bench_start = Instant::now();

    // Collect data until BENCH_TIME_LIMIT is reached.
    while bench_start.elapsed() < BENCH_TIME_LIMIT {
        let iters = ITER_SCALE_FACTOR.powi(data.len() as i32).round() as usize;
        // Prepare the environments - one per iteration
        let mut xs = std::iter::repeat_with(&mut gen_env)
            .take(iters)
            .collect::<Vec<I>>();
        // Start the clock
        let iter_start = Instant::now();
        // We iterate over `&mut xs` rather than draining it, because we
        // don't want to drop the env values until after the clock has stopped.
        for x in &mut xs {
            // Run the code and pretend to use the output
            std::hint::black_box(f(x));
        }
        let time = iter_start.elapsed();
        data.push((iters, time));
    }

    // If the first iter in a sample is consistently slow, that's fine -
    // that's why we do the linear regression. If the first sample is slower
    // than the rest, however, that's not fine.  Therefore, we discard the
    // first sample as a cache-warming exercise.
    data.remove(0);

    // Compute some stats
    let (grad, r2) = regression(&data[..]);
    Stats {
        ns_per_iter: grad,
        goodness_of_fit: r2,
        iterations: data.iter().map(|&(x, _)| x).sum(),
        samples: data.len(),
    }
}

/// Compute the OLS linear regression line for the given data set, returning
/// the line's gradient and R². Requires at least 2 samples.
//
// Overflows:
//
// * sum(x * x): num_samples <= 0.5 * log_k (1 + 2 ^ 64 (FACTOR - 1))
fn regression(data: &[(usize, Duration)]) -> (f64, f64) {
    if data.len() < 2 {
        return (f64::NAN, f64::NAN);
    }
    // Do all the arithmetic using f64, because it can happen that the
    // squared numbers to overflow using integer arithmetic if the
    // tests are too fast (so we run too many iterations).
    let data: Vec<_> = data
        .iter()
        .map(|&(x, y)| (x as f64, y.as_nanos() as f64))
        .collect();
    let n = data.len() as f64;
    let nxbar = data.iter().map(|&(x, _)| x).sum::<f64>(); // iter_time > 5e-11 ns
    let nybar = data.iter().map(|&(_, y)| y).sum::<f64>(); // TIME_LIMIT < 2 ^ 64 ns
    let nxxbar = data.iter().map(|&(x, _)| x * x).sum::<f64>(); // num_iters < 13_000_000_000
    let nyybar = data.iter().map(|&(_, y)| y * y).sum::<f64>(); // TIME_LIMIT < 4.3 e9 ns
    let nxybar = data.iter().map(|&(x, y)| x * y).sum::<f64>();
    let ncovar = nxybar - ((nxbar * nybar) / n);
    let nxvar = nxxbar - ((nxbar * nxbar) / n);
    let nyvar = nyybar - ((nybar * nybar) / n);
    let gradient = ncovar / nxvar;
    let r2 = (ncovar * ncovar) / (nxvar * nyvar);
    assert!(r2.is_nan() || r2 <= 1.0);
    (gradient, r2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    fn fib(n: usize) -> usize {
        let mut i = 0;
        let mut sum = 0;
        let mut last = 0;
        let mut curr = 1usize;
        while i < n - 1 {
            sum = curr.wrapping_add(last);
            last = curr;
            curr = sum;
            i += 1;
        }
        sum
    }

    // This is only here because doctests don't work with `--nocapture`.
    #[test]
    #[ignore]
    fn doctests_again() {
        println!();
        println!("fib 200: {}", bench(|| fib(200)));
        println!("fib 500: {}", bench(|| fib(500)));
        println!("reverse: {}", bench_env(vec![0; 100], |xs| xs.reverse()));
        println!("sort:    {}", bench_env(vec![0; 100], |xs| xs.sort()));

        // This is fine:
        println!("fib 1:   {}", bench(|| fib(500)));
        // This is NOT fine:
        println!(
            "fib 2:   {}",
            bench(|| {
                fib(500);
            })
        );
        // This is also fine, but a bit weird:
        println!(
            "fib 3:   {}",
            bench_env(0, |x| {
                *x = fib(500);
            })
        );
    }

    #[test]
    fn very_quick() {
        println!();
        println!("very quick: {}", bench(|| {}));
    }

    #[test]
    fn very_slow() {
        println!();
        println!(
            "very slow: {}",
            bench(|| thread::sleep(Duration::from_millis(400)))
        );
    }

    #[test]
    fn test_sleep() {
        println!();
        println!(
            "sleep 1 ms: {}",
            bench(|| thread::sleep(Duration::from_millis(1)))
        );
    }

    #[test]
    fn noop() {
        println!();
        println!("noop base: {}", bench(|| {}));
        println!("noop 0:    {}", bench_env(vec![0u64; 0], |_| {}));
        println!("noop 16:   {}", bench_env(vec![0u64; 16], |_| {}));
        println!("noop 64:   {}", bench_env(vec![0u64; 64], |_| {}));
        println!("noop 256:  {}", bench_env(vec![0u64; 256], |_| {}));
        println!("noop 512:  {}", bench_env(vec![0u64; 512], |_| {}));
    }

    #[test]
    fn ret_value() {
        println!();
        println!(
            "no ret 32:    {}",
            bench_env(vec![0u64; 32], |x| { x.clone() })
        );
        println!("return 32:    {}", bench_env(vec![0u64; 32], |x| x.clone()));
        println!(
            "no ret 256:   {}",
            bench_env(vec![0u64; 256], |x| { x.clone() })
        );
        println!(
            "return 256:   {}",
            bench_env(vec![0u64; 256], |x| x.clone())
        );
        println!(
            "no ret 1024:  {}",
            bench_env(vec![0u64; 1024], |x| { x.clone() })
        );
        println!(
            "return 1024:  {}",
            bench_env(vec![0u64; 1024], |x| x.clone())
        );
        println!(
            "no ret 4096:  {}",
            bench_env(vec![0u64; 4096], |x| { x.clone() })
        );
        println!(
            "return 4096:  {}",
            bench_env(vec![0u64; 4096], |x| x.clone())
        );
        println!(
            "no ret 50000: {}",
            bench_env(vec![0u64; 50000], |x| { x.clone() })
        );
        println!(
            "return 50000: {}",
            bench_env(vec![0u64; 50000], |x| x.clone())
        );
    }
}
