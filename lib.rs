/*!
A lightweight micro-benchmarking library which:

* uses linear regression to screen off sources of constant error;
* handles benchmarks which must mutate some state;
* `cloc`s in at under 100 lines of code with no dependencies;
* has an eponymously simple API!

Easybench is optimised for benchmarks with a running time of more than a nanosecond but less than a
millisecond. It's inspired by [criterion], but doesn't do the sophisticated analysis (no HTML
output).

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

An *iteration* is a single execution of your code. A *sample* is a measurement, during which your
code may be run many times. In other words: taking a sample means performing some number of
iterations and measuring the total time.

The first sample we take performs only 1 iteration, but as we continue taking samples we increase
the number of iterations exponentially. We stop when a time limit is reached (currently 1 second).

Next, we perform OLS linear regression on the resulting data. The gradient of the regression line
tells us how long it takes to perform a single iteration of the benchmark. The R² value is a
measure of how much noise there is in the data.

# Caveats

## Caveat 1: Harness overhead

**TL;DR: compile with `--release`, and expect worst-case overhead of a few nanoseconds**

How much overhead does easybench itself introduce? As explained above, we use linear regression to
eliminate any constant error involved with taking a sample. However, this technique doesn't prevent
linear error from showing up - that is, if there's some work which easybench does every iteration,
then it will be included in the results.

If your benchmark takes more than a nanosecond to run, and you compile with `--release`, this work
will typically be negligable:

* In the case of `bench` it amounts to incrementing the loop counter, plus the [cost of
  `black_box`](#bonus-caveat-black-box).
* In the case of `bench_env`, we also do a lookup into a big vector in order to get the environment
  for that iteration. If you're really unlucky, this could amount to a systematic cache miss, which
  would affect the results by a few nanoseconds. This is an unlikely worst-case, but it's best to
  be aware of the possibility.

If you're concerned at all about overhead, please take a look at [the inner loop of
`bench_env`][source]. The code is not scary.

[source]: src/easybench/lib.rs.html#197-208

## Caveat 2: Sufficient data

**TL;DR: measurements of code which takes more than a millisecond to run are not reliable**

Each benchmark collects data for 1 second. This means that in order to collect a statistically
significant amount of data, your code should run much faster than this. In fact, if we don't manage
to take a least 3 samples, easybench will actually panic!

When inspecting the results, make sure things looks statistically significant. In particular:

* Make sure the number of samples is big enough. More than 100 is probably OK.
* Make sure the R² isn't suspiciously low. It's easy to get a high R² value when the number of
  samples is small, so unfortunately the definition of "suspiciously low" depends on how many
  samples were taken.  As a rule of thumb, expect values greater than 0.99.

## Caveat 3: Pure functions

**TL;DR: return enough information to prevent the optimiser from eliminating code from your
benchmark**

Benchmarking pure functions involves a nasty gotcha which users should be aware of. Consider the
following benchmarks:

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

Oh, `fib_2`, why do you lie? The answer is: `fib(500)` is pure, and its return value is immediately
thrown away, so the optimiser replaces the call with a no-op (which clocks in at 0 ns).

What about the other two? `fib_1` looks very similar, with one exception: the closure which we're
benchmarking returns the result of the `fib(500)` call. When it runs your code, easybench takes the
return value and tricks the optimiser into thinking it's going to use it for something, before
throwing it away. This is why `fib_1` is safe from having code accidentally eliminated.

In the case of `fib_3`, we actually *do* use the return value: each iteration we take the result of
`fib(500)` and store it in the iteration's environment. This has the desired effect, but looks a
bit weird.

## Bonus caveat: Black box

The function which easybench uses to trick the optimiser (`black_box`) is stolen from [bencher],
which [states]:

[bencher]: https://docs.rs/bencher/
[states]: https://docs.rs/bencher/0.1.2/bencher/fn.black_box.html

> NOTE: We don't have a proper black box in stable Rust. This is a workaround implementation, that
> may have a too big performance overhead, depending on operation, or it may fail to properly avoid
> having code optimized out. It is good enough that it is used by default.

However, it works well enough in my experience.
*/

use std::f64;
use std::fmt::{self,Display,Formatter};
use std::time::*;

// Each time we take a sample we increase the number of iterations:
//      iters = ITER_SCALE_FACTOR ^ sample_no
const ITER_SCALE_FACTOR: f64 = 1.1;

// We try to spend this many seconds (roughly) in total on each benchmark.
const BENCH_TIME_LIMIT_SECS: u64 = 1;

/// Statistics for a benchmark run.
#[derive(Debug)]
pub struct Stats {
    /// The time, in nanoseconds, per iteration. If the benchmark generated fewer than 2 samples in
    /// the allotted time then this will be NaN.
    pub ns_per_iter: f64,
    /// The coefficient of determination, R².
    ///
    /// This is an indication of how noisy the benchmark was, where 1 is good and 0 is bad. Be
    /// suspicious of values below 0.9.
    pub goodness_of_fit: f64,
    /// How many times the benchmarked code was actually run.
    pub iterations: usize,
    /// How many samples were taken (ie. how many times we allocated the environment and measured
    /// the time).
    pub samples: usize,
}

impl Display for Stats {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        if self.ns_per_iter.is_nan() {
            write!(f, "Only generated {} sample(s) - we can't fit a regression line to that! \
            Try making your benchmark faster.", self.samples)
        } else {
            write!(f, "{:>10.0} ns   (R²={:.3}, {} iterations in {} samples)",
                self.ns_per_iter, self.goodness_of_fit, self.iterations, self.samples)
        }
    }
}

/// Run a benchmark.
///
/// The return value of `f` is not used, but we trick the optimiser into thinking we're going to
/// use it. Make sure to return enough information to prevent the optimiser from eliminating code
/// from your benchmark! (See the module docs for more.)
pub fn bench<F, O>(f: F) -> Stats where F: Fn() -> O {
    bench_env((), |_| f() )
}

/// Run a benchmark with an environment.
///
/// The value `env` is a clonable prototype for the "benchmark environment". Each iteration
/// receives a freshly-cloned mutable copy of this environment. The time taken to clone the
/// environment is not included in the results.
///
/// Nb: it's very possible that we will end up allocating many (>10,000) copies of `env` at the
/// same time. Probably best to keep it small.
///
/// See `bench` and the module docs for more.
pub fn bench_env<F, I, O>(env: I, f: F) -> Stats where F: Fn(&mut I) -> O, I: Clone {
    let mut data = Vec::new();
    let bench_start = Instant::now(); // The time we started the benchmark (not used in results)

    // Collect data until BENCH_TIME_LIMIT_SECS is reached.
    while bench_start.elapsed() < Duration::from_secs(BENCH_TIME_LIMIT_SECS) {
        let iters = ITER_SCALE_FACTOR.powi(data.len() as i32).round() as usize;
        let mut xs = vec![env.clone();iters]; // Prepare the environments - one per iteration
        let iter_start = Instant::now();      // Start the clock
        for i in 0..iters {
            let ref mut x = xs[i];            // Lookup the env for this iteration
            pretend_to_use(f(x));             // Run the code and pretend to use the output
        }
        let time = iter_start.elapsed();
        data.push((iters, time));
    }

    // If the first iter in a sample is consistently slow, that's fine - that's why we do the
    // linear regression. If the first sample is slower than the rest, however, that's not fine.
    // Therefore, we discard the first sample as a cache-warming exercise.
    data.remove(0);

    // Compute some stats
    let (grad, r2) = regression(&data[..]);
    Stats {
        ns_per_iter: grad,
        goodness_of_fit: r2,
        iterations: data.iter().map(|&(x,_)| x).sum(),
        samples: data.len(),
    }
}

/// Compute the OLS linear regression line for the given data set, returning the line's gradient
/// and R². Requires at least 2 samples.
fn regression(data: &[(usize, Duration)]) -> (f64, f64) {
    if data.len() < 2 {
        return (f64::NAN, f64::NAN);
    }
    let data: Vec<(u64, u64)> = data.iter().map(|&(x,y)| (x as u64, as_nanos(y))).collect();
    let n = data.len() as f64;
    let nxbar  = data.iter().map(|&(x,_)| x  ).sum::<u64>(); // iter_time > 5e-11 ns
    let nybar  = data.iter().map(|&(_,y)| y  ).sum::<u64>(); // TIME_LIMIT < 2 ^ 64 ns
    let nxxbar = data.iter().map(|&(x,_)| x*x).sum::<u64>(); // num_iters < 13_000_000_000
    let nyybar = data.iter().map(|&(_,y)| y*y).sum::<u64>(); // TIME_LIMIT < 4.3 e9 ns
    let nxybar = data.iter().map(|&(x,y)| x*y).sum::<u64>();
    let ncovar = nxybar as f64 - ((nxbar * nybar) as f64 / n);
    let nxvar  = nxxbar as f64 - ((nxbar * nxbar) as f64 / n);
    let nyvar  = nyybar as f64 - ((nybar * nybar) as f64 / n);
    let gradient = ncovar / nxvar;
    let r2 = (ncovar * ncovar) / (nxvar * nyvar);
    (gradient, r2)
}

// Panics if x is longer than 584 years (1.8e10 seconds)
fn as_nanos(x: Duration) -> u64 {
    x.as_secs().checked_mul(1_000_000_000).expect("overflow: Duration was longer than 584 years")
        .checked_add(x.subsec_nanos() as u64).unwrap()
}

// Stolen from `bencher`.
//
// NOTE: We don't have a proper black box in stable Rust. This is
// a workaround implementation, that may have a too big performance overhead,
// depending on operation, or it may fail to properly avoid having code
// optimized out. It is good enough that it is used by default.
//
// A function that is opaque to the optimizer, to allow benchmarks to
// pretend to use outputs to assist in avoiding dead-code
// elimination.
fn pretend_to_use<T>(dummy: T) -> T {
    unsafe {
        let ret = ::std::ptr::read_volatile(&dummy);
        ::std::mem::forget(dummy);
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    fn fib(n: usize) -> usize {
        let mut i = 0; let mut sum = 0; let mut last = 0; let mut curr = 1usize;
        while i < n - 1 {
            sum = curr.wrapping_add(last);
            last = curr;
            curr = sum;
            i += 1;
        }
        sum
    }

    // This is only here because doctests don't work with `--nocapture`.
    #[test] #[ignore]
    fn doctests_again() {
        println!("fib 200: {}", bench(|| fib(200) ));
        println!("fib 500: {}", bench(|| fib(500) ));
        println!("reverse: {}", bench_env(vec![0;100], |xs| xs.reverse()));
        println!("sort:    {}", bench_env(vec![0;100], |xs| xs.sort()));

        // This is fine:
        println!("fib 1: {}", bench(|| fib(500) ));
        // This is NOT fine:
        println!("fib 2: {}", bench(|| { fib(500); } ));
        // This is also fine, but a bit weird:
        println!("fib 3: {}", bench_env(0, |x| { *x = fib(500); } ));
    }

    #[test]
    fn very_quick() {
        println!("very quick: {}", bench(|| {}));
    }

    #[test]
    fn very_slow() {
        println!("very slow: {}", bench(|| thread::sleep(Duration::from_millis(400))));
    }
}
