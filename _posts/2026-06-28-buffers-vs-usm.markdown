---
layout: single
title:  "Buffer vs USM: Which Model to Choose?"
date:   2026-06-28 18:00:00 +0100
categories: hipsycl adaptivecpp sycl buffer usm
---

# Introduction

SYCL has two memory management models: the old buffer-accessor model, and the newer USM (unified shared memory) model that was introduced in SYCL 2020.

The AdaptiveCpp [performance guide](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/performance.md) clearly states that users should always prefer the USM model. Recent versions of AdaptiveCpp even emit a runtime warning when an application uses SYCL buffers.
Nevertheless, there are also certain publications (e.g. [here](https://arxiv.org/html/2604.16043v1)) that argue that buffers perform better. This has caused confusion in the community: Who is right? What actually to use -- buffers or USM?

*Spoiler: The AdaptiveCpp performance guide is right.*

In the following, I would like to go into a little more depth about what the issues with buffers are, why users should use USM, and why certain publications seemingly conclude otherwise.

# Where buffers come from

Buffers are the original model in SYCL, and it is important to understand that they come from a time when SYCL was designed to exclusively target roughly OpenCL 1.2 (+SPIR/SPIR-V) capabilities. This in particular means: No pointer-based memory management (OpenCL SVM was introduced later), only opaque `cl_mem` memory handles.

In the context of these limitations, the folks in the ancient days of SYCL tried to make the model as convenient possible with what they had available. They figured out that C++ abstractions let them do things like automatic dependency detection and data migration. But still, at the heart of the buffer model lies a restriction: that pointer-based memory management (which C++ otherwise assumes exists) is unavailable. The buffer-accessor model thus has also always been driven by the lack of a modern pointer-based memory management model in older OpenCL versions.

The code snippet below illustrates a simple SYCL program that uses the buffer-accessor model to multiply every value of an array by two.

```c++
sycl::queue q;
std::vector<float> data(1024);
{
  sycl::buffer<float> buf{data.data(), data.size()};
  q.submit([&](sycl::handler& h) { 
    // Create read-write accessor
    // (note: read-write is default when no access mode is explicitly
    //  specified)
    sycl::accessor acc{buf, h};
    h.parallel_for(data.size(), [=](sycl::id<1> i) {
      acc[i] *= 2.0f;
    });
  });
} // buffer destruction synchronizes results back to host
```

The buffer model
- automatically inserts dependencies between kernels based on the access modes of accessors. For example, when two kernels want to access the same buffer in read-write mode, then the runtime would detect that it needs to insert a dependency between the two kernels.
- handles lifetime of memory allocations automatically (memory is released when the last copy of the buffer is destroyed)
- automatically migrates data between host and device
- provides multi-dimensional data access via the accessor in kernels

Sounds pretty nifty and attractive, right?

# The USM model

In SYCL, there are three kinds of USM:

1. Device USM. Allocations are explicitly resident on the device, and obtained with `malloc_device()`. The host cannot access device USM allocation, and explicit data transfers are needed to move data between host and device (e.g. with `queue::memcpy()`).
2. Host USM. Allocations are resident on the host and obtained with `malloc_host()`. These allocations are accessible from the device too, but every memory access must traverse the interconnect between host and device (e.g. PCIe). Host USM allocations are mainly interesting as host staging buffers for data transfers to device USM allocations, or for data that is so rarely accessed on device that invoking a `memcpy()` is slower than just accessing it in most memory over the interconnect.
3. Shared USM. In this USM mode, the data can be accessed on both host and device. Data migrates automatically; no explicit data transfers are needed. For performance optimization, `queue::prefetch(ptr, size)` can be used which hints to the runtime that a certain memory range will soon be needed. Shared USM is typically implemented via hardware support. For example, if a memory page is unavailable on device, it might trigger an interrupt, which causes the GPU driver to migrate the affected page. Similarly, if a page is not currently present on the host, a page fault is triggered, which is resolved by fetching the page from the device.

In AdaptiveCpp, all modes of USM are almost universally supported, with the only exception being the experimental Vulkan backend (Vulkan lacks the capabilities to implement e.g. shared USM). OpenCL devices need to support at least coarse-grained SVM to support device USM. In the OpenCL backend, the full feature set is enabled for devices supporting the Intel USM extensions (e.g. Intel CPUs or GPUs). CUDA, HIP, Metal and the CPU backend support all USM modes.

Here is the same example code using device USM. Note the explicit data transfers. At the same time, the kernel launch itself looks considerably more concise.
```c++
sycl::queue q{sycl::property::queue::in_order{}};

std::vector<float> data(1024);
// Allocate device memory
float* device_data = sycl::malloc_device<float>(data.size(), q);

// Copy input data to the device
q.memcpy(device_data, data.data(), data.size() * sizeof(float));
q.parallel_for(data.size(), [=](sycl::id<1> i) {
  device_data[i] *= 2.0f;
});
// Copy results back to the host
q.memcpy(data.data(), device_data, data.size() * sizeof(float));
q.wait();
// Free the allocation
sycl::free(device_data, q);
```

In shared USM, we can omit the data transfers:

```c++
sycl::queue q{sycl::property::queue::in_order{}};

// Allocate device memory and initialized on host
float* device_data = sycl::malloc_shared<float>(data.size(), q);
for(std::size_t i = 0; i < data.size(); ++i)
  device_data[i] = static_cast<float>(i);

// No data transfer needed. As an optimization, 
// you might want to consider a prefetch:
// q.prefetch(device_data, data.size() * sizeof(float));
q.parallel_for(data.size(), [=](sycl::id<1> i) {
  device_data[i] *= 2.0f;
});
// No explicit copy back to host needed
q.wait();
// (use results here)

// Free the allocation
sycl::free(device_data, q);
```

# Issues with the buffer model compared to USM

While the buffer model sounds pretty great, in practice it does not really work out that way (Note: I'm exclusively talking about the specific buffer-accessor model as currently in SYCL, not about the potential benefits of C++ memory management abstractions in general).

There are many issues around lack of flexibility/control, overheads (at both runtime and code generation level), and complexity, some of which I list below:

- **Hard to learn:** the buffer-accessor model may appear simple, but in order to get good performance out of it, you must understand what it does under the hood. This however is implementation-specific expert knowledge. So it is actually not accessible for beginners. In many years of teaching SYCL, I have consistently made the experience that the buffer-accessor model is way harder for beginners to wrap their heads around than USM. On the other hand, most people immediately understand USM: I need to allocate, then `memcpy` to device if I use device USM and want to have the data there, then I use the pointer in kernels, and in the end, I free the allocation. The API of buffer is much, much more complex and much harder to reason about.
- **Hidden synchronization:** There are performance gotchas. For example, the buffer destructor might block (in some cases, but not in others), which can make multiple buffers being destroyed at the end of the same scope become an anti-pattern of repeated, unnecessary synchronization.
- **No control over allocation lifetime:** Because the buffer destructor might not block in some cases, there's actually also no guarantee when memory will actually be freed, because the SYCL runtime might still use the buffer internally. And there's no way of knowing when the memory is actually freed in SYCL. With USM, you just call `free()` when you need the memory freed. You don't have this kind of control with buffers.
- **Runtime costs:** With buffers, the SYCL runtime must figure out the dependencies at runtime for every single kernel launch. This means introspecting the kernel launch, checking the references allocations for current users, comparing their access modes, and updating the dependency graph. In practice however, most applications know their dependencies already at compile time. But with buffers, this information will be reconstructed again and again, and there's nothing you can do against it. This can add substantial kernel launch latency, which is well-known and has been [documented with both DPC++ and AdaptiveCpp](https://dl.acm.org/doi/10.1145/3648115.3648120).
- **Inflexible dependency tracking:** You cannot actually use the same buffer concurrently by multiple read-write kernels, even if the kernels only access disjoint parts of the data. The SYCL specification explicitly requires that such kernels are serialized. With USM, you have the control to do that.
- **Register pressure:** Accessors in kernels generally cause more register pressure than USM pointers. This is because accessors are not lightweight objects, and haul around a lot of information e.g. about the shape of the data that you might not even need in your kernels. Some of that might be optimized away, but the compiler starts from a signficantly larger live state. A particularly pathological (yet common!) case is the following: Many applications use multiple allocations in kernels, but they often have the same size (e.g. a global problem size). With USM, you would pass in the pointers, and pass in the size once -- and the compiler then won't worry about separate sizes for each allocation. With buffers, you have no way of communicating this to the compiler, so it must assume that every accessor might refer to an allocation with different data shapes. Especially for 2D or 3D data, the additionally required resources can be substantial. These register pressure issues are well-understood and also [documented for both DPC++ and AdaptiveCpp](https://doi.org/10.1016/j.future.2024.03.045).
- **Compile times:** SYCL already is infamous for poor compile times. The buffer-accessor model makes this even worse. For example, I've seen the buffer version of CloverLeaf compile 10-20% slower than the USM version. This is because the buffer-accessor model has a large and complex C++ API surface.
- **Interoperability and code migration:** the pointer-based USM model is much closer to code already written in other models -- be it CUDA, HIP, or even normal host C++. This means that code migration and interoperability is generally much easier with USM. You want to pass some device pointer to GPU-aware MPI? With USM, you just pass the pointer. With buffers, there's no well-defined path to accomplish that at all. Similarly, interoperability with libraries like cuBLAS, cuFFT, rocFFT is much easier with USM.
- **Composability:** The buffer-accessor model has a massive composability issue. Consider a library where the user can pass in function objects that are then called by a kernel that the library launches: 
```c++
mylibrary::run_computation(myoperator);
// inside mylibrary::run_computation():
q.parallel_for(..., [=](auto idx){
  //...
  auto x = myoperator();
  //...
});
```
If `myoperator` now requires access to additional memory, you have a problem if you use the buffer-accessor model: every accessor used by a kernel must be registered with the runtime at the callsite of the kernel. But the callsite is inside the library, and may not be accessible to the user defining `myoperator`. The buffer-accessor model implicitly assumes that if you want to modify the set of data accessed by the kernel, you have access to the kernel callsite. This however might not be the case in more complex applications. 



Sometimes, people believe that a SYCL compiler can make stronger assumptions about optimizations when buffers and accessors are used. This is generally false. For example, AdaptiveCpp can automatically detect (for both buffers and USM pointers) whether they alias at kernel launch, and take that into account when the JIT compiler generates code ([paper](https://dl.acm.org/doi/10.1145/3731125.3731127)). At the same time, buffer semantics are not specific enough to let the compiler make strong assumptions about how it is going to be used -- if the compiler had a `matrix` or `tensor` object for numerical types as first class citizen, that might be different. But for buffers, I am not aware of any compiler optimization in modern SYCL compilers that does not also work with USM.

Now, where to go from here? Virtually all production large-scale SYCL applications like GROMACS or AMReX use USM instead of buffers for both performance and functionality reasons. At IWOCL' 26, it was publicly [disclosed](https://www.youtube.com/watch?v=JqGXHucdhoc) in the SYCL "state of the union" talk that the Khronos SYCL working group considers deprecating the buffer-accessor model.
This does not mean that a higher-level abstraction would go away completely, as some ideas are floating around for a "better" replacement API -- but the current buffer-accessor model increasingly looks like it does not have a future. This, of course, is yet another reason to avoid it.

# Why do some people arrive at different conclusions?

If it is so clear that USM should be preferred, why do some publications arrive at other conclusions then?

I believe it has to do with unclear terminology as well as unfair or flawed benchmark methodology.

Firstly, the SYCL terminology of "unified shared memory" is somewhat less specific than in the rest of the industry. In SYCL, USM per se only refers to a pointer-based memory management model with unified virtual address space -- i.e., there can be no overlap between USM pointer addresses, and any possible pointer from a regular host allocation.
Other programming models, for example OpenMP, use the term USM specifically to refer to allocations that migrate automatically between host and device. In SYCL, this is called shared USM, and is a special case of USM as a whole.

As a consequence of this, people sometimes compare the performance of **shared** USM to buffers, and then conclude that "USM" (without further qualification) performs worse. However, had they used device USM, the observation might have flipped.

Can we then at least conclude that **shared** USM performs poorly? Not necessarily.

Shared USM in SYCL is the same thing as unified memory in CUDA, and indeed, SYCL implementations map SYCL shared USM to CUDA unified memory when running on NVIDIA hardware. There is a [large body of work](https://developer.nvidia.com/blog/maximizing-unified-memory-performance-cuda/) around the performance of unified memory in CUDA that applies in the same way to SYCL.

There are two important points:
1. Be careful about what is measured in benchmarks when interpreting results involving shared USM. In shared USM mode, data will be migrated as part of kernel execution. If your benchmark only measures kernel runtime, then you're not running a fair comparison: the explicit `memcpy()` time for the  device USM baseline will then not be included in the baseline time, whereas the data transfer time will be included in the shared USM measurement.
2. It is well-known that an explicit prefetch (`queue::prefetch()`) can often remedy all or most remaining performance differences. This is a straight-forward, one-line optimization that should be considered best practice when working with shared USM, and there is generally no reason not to do it in places where you would expect a data migration to happen with high certainty. Benchmarks not using prefetches can be assumed to not deliver the best of what is possible with shared USM.

On CPU targets, shared USM might even perform better than device USM, because the shared USM model does not assume that host and device memory are necessarily distinct. With shared USM on CPU, no separate host and device allocations need to be maintained. Consequently, any data transfers between them can also be elided.

The main reason why I generally recommend device USM over shared USM is that, for reasons that are beyond my understanding, AMD's current consumer RDNA GPUs do not have the necessary [hardware functionality](https://niconiconi.neocities.org/tech-notes/xnack-on-amd-gpus/) for automatically migrating data. This is a more specific issue than a broad "shared USM is not performant", and this issue may or may not matter for your use case.

# Conclusion

For new production code, I see little reason to choose buffers over USM in 2026. Buffers should be understood as a solution from the early days of SYCL to work around the limitations of old OpenCL versions, and as a solution from which the ecosystem has since largely shifted away -- not as a production feature for new code investments. USM generally gives users more flexibility and performance. Use device USM if you want broad portability and performance, and shared USM for maximum productivity with some specific portability caveats.


