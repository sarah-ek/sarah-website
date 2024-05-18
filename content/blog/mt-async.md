+++
title = "speeding up data parallel code sections with... tokio?"
date = 2024-05-18
template = "blog-page.html"
+++

consider a scenario where we have an algorithm requiring two nested loops, where the outer one must be sequential but the inner one can be parallelized.  
such algorithms are common in linear algebra decompositions, so it's a use case worth investigating for my personal work on [faer](https://github.com/sarah-ek/faer-rs).

to keep things simple, we'll take as an example the following problem:

```rust
fn algo(x: &mut [f64]) {
    let n = x.len();
    for i in 0..n {
        // take the i-th element
        let head = x[i];

        // and add it to all the elements that come after it
        for elem in &mut x[i + 1..] {
            *elem += head;
        }
    }
}
```

the first solution that one could consider when working with such problems is using rayon parallel iterators to parallelize the inner loop.

```rust
use rayon::prelude::*;

fn algo_rayon(x: &mut [f64]) {
    let n = x.len();
    for i in 0..n {
        // take the i-th element
        let head = x[i];

        // and add it to all the elements that come after it
        x[i + 1..].par_iter_mut().for_each(|elem| {
            *elem += head;
        });
    }
}
```

let's benchmark it for a few sizes to see if we got a speedup.
```
╭────────────────┬────────┬───────────┬───────────┬───────────┬───────────╮
│ benchmark      │   args │   fastest │    median │      mean │    stddev │
├────────────────┼────────┼───────────┼───────────┼───────────┼───────────┤
│ sequential     │  10000 │   6.47 ms │   6.64 ms │   6.63 ms │ 120.22 µs │
│ rayon          │  10000 │ 148.96 ms │ 153.78 ms │ 153.61 ms │   2.83 ms │
├────────────────┼────────┼───────────┼───────────┼───────────┼───────────┤
│ sequential     │ 100000 │ 831.04 ms │ 835.84 ms │ 834.19 ms │   3.54 ms │
│ rayon          │ 100000 │   2.38  s │   2.43  s │   2.41  s │  30.99 ms │
├────────────────┼────────┼───────────┼───────────┼───────────┼───────────┤
│ sequential     │ 400000 │  14.54  s │  14.54  s │  14.54  s │     -     │
│ rayon          │ 400000 │  18.83  s │  18.83  s │  18.83  s │     -     │
╰────────────────┴────────┴───────────┴───────────┴───────────┴───────────╯
```

well, that doesn't look very good. looks like the parallel version is slower for all the sizes we gave.
although it's not that much slower for the largest size, so i'm assuming that at some point it'll start beating the sequential version if we keep going higher.

one of the problems with out code is that it's passing the elements one by one to rayon, which adds quite a bit of overhead per unit.
so let's try making our units larger.

```rust
use rayon::prelude::*;

fn algo_rayon_chunk(x: &mut [f64]) {
    let nthreads = rayon::current_num_threads();

    let n = x.len();
    for i in 0..n {
        // take the i-th element
        let head = x[i];
        let len = x[i + 1..].len();

        // and add it to all the elements that come after it
        if len > 0 {
            x[i + 1..].par_chunks_mut(len.div_ceil(nthreads)).for_each(|chunk| {
                for elem in chunk {
                    *elem += head;
                }
            });
        }
    }
}
```

let's benchmark this again.
```
╭────────────────┬────────┬───────────┬───────────┬───────────┬───────────╮
│ benchmark      │   args │   fastest │    median │      mean │    stddev │
├────────────────┼────────┼───────────┼───────────┼───────────┼───────────┤
│ sequential     │  10000 │   7.11 ms │   7.12 ms │   7.14 ms │  83.28 µs │
│ rayon          │  10000 │ 149.42 ms │ 151.62 ms │ 152.17 ms │   4.17 ms │
│ rayon_chunk    │  10000 │  63.45 ms │  64.82 ms │  65.11 ms │   1.59 ms │
├────────────────┼────────┼───────────┼───────────┼───────────┼───────────┤
│ sequential     │ 100000 │ 789.07 ms │ 790.30 ms │ 790.01 ms │ 921.28 µs │
│ rayon          │ 100000 │   2.35  s │   2.37  s │   2.36  s │  15.08 ms │
│ rayon_chunk    │ 100000 │ 971.65 ms │ 977.52 ms │   1.01  s │  61.94 ms │
├────────────────┼────────┼───────────┼───────────┼───────────┼───────────┤
│ sequential     │ 400000 │  14.17  s │  14.17  s │  14.17  s │     -     │
│ rayon          │ 400000 │  19.29  s │  19.29  s │  19.29  s │     -     │
│ rayon_chunk    │ 400000 │  11.52  s │  11.52  s │  11.52  s │     -     │
╰────────────────┴────────┴───────────┴───────────┴───────────┴───────────╯
```

a little better, but still not the speedup we were hoping for.  
although we're giving decently sized chunks to rayon, there's still a cost of going in and out of the thread pool every time to synchronize all the threads.

what if there was a way we could avoid this cost?  
while looking into possible solutions, i thought of the idea of using a barrier to synchronize the threads.

the one from the standard library was too slow for this use case, but i found [hurdles](https://github.com/jonhoo/hurdles)
by Jon Gjengset that seemed like it could be interesting, so i copied it and made a few modifications to suit my kind of scenario.

the code is about to get unsafe, but we'll think about how to add safe abstractions later.  
we'll also hold off on looking at the benchmarks for a while, but i can promise that the results are worth the suspense.

```rust
use std::cell::UnsafeCell;
use hurdles::Barrier;

/// returns the starting index
fn id_to_chunk(id: usize, nthreads: usize, n: usize) -> usize {
    let div = n / nthreads;
    let rem = n % nthreads;

    if id <= rem {
        (div + 1) * id
    } else {
        (div + 1) * rem + div * (id - rem)
    }
}

fn algo_rayon_barrier(x: &mut [f64]) {
    let nthreads = rayon::current_num_threads();
    let n = x.len();

    // SAFETY: we have exclusive access to `x`, and `UnsafeCell<T>` has the same layout as `T`.
    let x = unsafe { std::slice::from_raw_parts_mut(
        x.as_mut_ptr() as *mut UnsafeCell<f64>,
        n,
    ) };
    let x = &*x;

    let init = Barrier::new(nthreads);

    rayon::in_place_scope(|scope| {
        for id in 0..nthreads {
            scope.spawn(move |_| {
                let mut barrier = init.clone();
                let x = x;

                for i in 0..n {
                    barrier.wait();

                    // SAFETY: none of the threads are writing to x[i] at the i-th iteration.
                    let head = unsafe { *(x[i].get()) };

                    let len = x[i + 1..].len();
                    let start = id_to_chunk(id, nthreads, len);
                    let end = id_to_chunk(id + 1, nthreads, len);
                    let thread_x = &x[i + 1..][start..end];

                    // SAFETY: thread `id` has exclusive access to its part of `x`
                    // since the starting indices are non-decreasing.
                    let thread_x = unsafe { std::slice::from_raw_parts_mut(
                        thread_x.as_ptr() as *const f64 as *mut f64,
                        thread_x.len(),
                    ) };

                    for elem in thread_x {
                        *elem += head;
                    }
                }
            });
        }
    });
}
```

there are two problems with this code. the first one is the unsafety. let's try to do something about it.
from a high level point of view, the code does something like this:
- take ownership of `x` to be shared between the threads in the team (cast to `&[UnsafeCell<f64>]`),
- give each thread the shared part of the data (`unsafe { *(x[i].get()) }`)
- give each thread its own exclusive part of the data (`&x[i + 1..][start..end]`)

we can wrap this in a safe api by making the barrier itself take ownership of the data, similarly to how `std::sync::Mutex` is a container.
then whenever the threads reach the `barrier.wait()` statement, one of the threads is given exclusive access to the data.  
that thread is then responsible for splitting up the data into shared and exclusive sections, which are then passed to the other threads accordingly.

i'll spare you the [implementation details](https://github.com/sarah-ek/syncthreads/).
but the usage of the code looks roughly like this
```rust
use syncthreads::BarrierInit;
use syncthreads::AllocHint;
use syncthreads::sync::BarrierParams;

fn rayon_barrier_safe(x: &mut [f64]) {
    let nthreads = rayon::current_num_threads();
    let init = BarrierInit::new(x, nthreads, AllocHint::default(), BarrierParams::default());

    rayon::in_place_scope(|scope| {
        for _ in 0..nthreads {
            scope.spawn(|_| {
                let mut barrier = init.barrier_ref();

                for i in 0..n {

                    let (head, mine): (&f64, &mut [f64]) = syncthreads::sync!(
                        barrier,
                        |x: &mut [f64]| {
                            // this part is only executed by the leader thread
                            // which takes exclusive access of the data

                            let head = x[i];
                            (
                                // shared section
                                head,
                                // exclusive section
                                syncthreads::iter::split_mut(&mut x[i + 1..], nthreads),
                            )
                        },
                    ).unwrap();

                    let head = *head;
                    for x in mine.iter_mut() {
                        *x += head;
                    }
                }
            });
        }
    });
}
```

the second problem is a bit more subtle.  
suppose this code is executed from two separate threadsj  
each one spawns `nthreads` tasks, but we have no guarantee that rayon won't interleave them in some way.
this means it's possible that each execution only gets a subset of the `nthreads` that it requires to make progress,
leading to a deadlock.

my solution to fix this was to replace the barrier by an async adaptation.
the main difference is that instead of waiting, we instead yield control back to the executor at the `sync!` point,
this way tasks can continue to make progress even if their number exceeds the number of threads in the pool.

```rust
use syncthreads::AsyncBarrierInit;
use syncthreads::AllocHint;
use syncthreads::sync::AsyncBarrierParams;

fn tokio_barrier_safe(x: &mut [f64]) {
    let nthreads = rayon::current_num_threads();
    let init = AsyncBarrierInit::new(x, nthreads, AllocHint::default(), AsyncBarrierParams::default());

    tokio_scoped::scope(|scope| {
        for _ in 0..nthreads {
            scope.spawn(async {
                let mut barrier = init.barrier_ref();

                for i in 0..n {

                    let (head, mine): (&f64, &mut [f64]) = syncthreads::sync!(
                        barrier,
                        |x: &mut [f64]| {
                            // this part is only executed by the leader thread
                            // which takes exclusive access of the data

                            let head = x[i];
                            (
                                // shared section
                                head,
                                // exclusive section
                                syncthreads::iter::split_mut(&mut x[i + 1..], nthreads),
                            )
                        },
                    )
                    .await
                    .unwrap();

                    let head = *head;
                    for x in mine.iter_mut() {
                        *x += head;
                    }
                }
            });
        }
    });
}
```

this solves the deadlock issue, making it more reliable, and surprisingly doesn't add much overhead to our solution.

here are the final benchmarks:
```
╭────────────────┬────────┬───────────┬───────────┬───────────┬───────────╮
│ benchmark      │   args │   fastest │    median │      mean │    stddev │
├────────────────┼────────┼───────────┼───────────┼───────────┼───────────┤
│ sequential     │  10000 │   7.02 ms │   7.12 ms │   7.12 ms │  38.27 µs │
│ rayon          │  10000 │ 148.15 ms │ 150.87 ms │ 153.50 ms │   6.81 ms │
│ rayon_chunk    │  10000 │  62.85 ms │  64.19 ms │  66.47 ms │   4.66 ms │
│ barrier        │  10000 │   7.48 ms │   7.90 ms │  10.04 ms │   5.54 ms │
│ async_barrier  │  10000 │   8.89 ms │   9.25 ms │   9.53 ms │   1.19 ms │
├────────────────┼────────┼───────────┼───────────┼───────────┼───────────┤
│ sequential     │ 100000 │ 828.32 ms │ 829.28 ms │ 829.01 ms │ 485.75 µs │
│ rayon          │ 100000 │   2.38  s │   2.47  s │   2.43  s │  64.79 ms │
│ rayon_chunk    │ 100000 │ 922.56 ms │   1.04  s │   1.03  s │ 124.49 ms │
│ barrier        │ 100000 │ 235.42 ms │ 263.35 ms │ 258.01 ms │  15.59 ms │
│ async_barrier  │ 100000 │ 267.28 ms │ 289.61 ms │ 289.03 ms │  19.08 ms │
├────────────────┼────────┼───────────┼───────────┼───────────┼───────────┤
│ sequential     │ 400000 │  14.29  s │  14.29  s │  14.29  s │     -     │
│ rayon          │ 400000 │  18.25  s │  18.25  s │  18.25  s │     -     │
│ rayon_chunk    │ 400000 │   8.91  s │   8.91  s │   8.91  s │     -     │
│ barrier        │ 400000 │   3.75  s │   3.75  s │   3.75  s │     -     │
│ async_barrier  │ 400000 │   3.89  s │   3.89  s │   3.89  s │     -     │
╰────────────────┴────────┴───────────┴───────────┴───────────┴───────────╯  
```
