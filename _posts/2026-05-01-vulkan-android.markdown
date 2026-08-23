---
layout: single
title:  "Bringing SYCL to Android: A Vulkan backend for portable GPU compute"
date:   2026-05-01 09:00:00 +0100
categories: hipsycl adaptivecpp sycl vulkan android
author: Ewan Crawford
---

SYCL's promise is performance portability: write modern C++ once and execute across many different accelerators. But that
promise only goes as far as the available backends. While desktop and HPC platforms capitalize on established OpenCL,
CUDA, HIP, and Level Zero backends for accelerating SYCL applications on the GPU, mobile isn't a domain commonly associated
with SYCL. Yet almost every Android device ships with a capable Vulkan implementation, making it an unfulfilled chapter of
the SYCL performance portability story.

While projects such as [Sylkan](https://dl.acm.org/doi/10.1145/3456669.3456683) have demonstrated that SYCL over Vulkan is
feasible, the space has remained relatively unexplored in terms of feature completeness, Android support, and integration
with the wider SYCL ecosystem. OpenCL has had more success in this area with projects such as
[clvk](https://github.com/kpet/clvk), [pocl](https://github.com/pocl/pocl), and [ANGLE](https://github.com/google/angle)
successfully layering its compute API over Vulkan.

AdaptiveCpp closes this SYCL gap with the recently added Vulkan backend and Android cross compilation support.
This post highlights what's going on under the hood in AdaptiveCpp to layer on top of Vulkan, before showing how to build and
use the Vulkan backend. Starting with an overview of the two halves of bringing up a backend target in SYCL, the runtime
and the compiler.

# Runtime Implementation Overview

Before any kernel code runs on a device, the runtime has to do a lot of heavy lifting to map SYCL concepts like queues,
events, and memory allocations down onto Vulkan's explicit, low-level API. To reduce the implementation verbosity the
Khronos [VulkanHpp](https://github.com/KhronosGroup/Vulkan-Hpp) headers are used in the AdaptiveCpp source code,
which provide useful concepts like RAII wrapped Vulkan object handles.

Each SYCL device corresponds to a logical Vulkan device that meets the key capability criteria to implement SYCL,
namely [Timeline Semaphores](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_timeline_semaphore.html)
and [Buffer Device Address](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_buffer_device_address.html).
Let's elaborate on why those extensions are necessary building blocks of our SYCL implementation.

## Timeline Semaphores

Synchronization in Vulkan is complicated, so we simplified everything down to use a fundamental primitive that is powerful
but also easy to reason about, the timeline semaphore. Rather than being in a binary signaled or not signaled state
it uses a monotonically increasing 64-bit integer value to define synchronization order, eliminating the need to be reset
before reuse. What's more, a timeline semaphore can also be signaled from host or device, which is useful for reasons we'll discuss later.

In our implementation each SYCL queue that is created by a user has its own timeline semaphore, with a value that is
initialized to zero and incremented on each command submission. On a `sycl::queue::submit()` call a `vkCommandBuffer`
submission is made to enqueue that work to the device. Crucially, rather than batching multiple commands into a command-buffer
each command-buffer contains a single command, which is synchronized entirely using timeline semaphores. This allows
each command to be uniquely identified by a queue handle and timeline value pair.

{% comment %}
Mermaid source used to generate diagram below
sequenceDiagram
    participant App as SYCL application
    participant RT  as AdaptiveCpp runtime
    participant CB  as VkCommandBuffer
    participant Q   as VkQueue (shared)
    participant TS  as VkSemaphore (timeline)
    App->>RT: queue.submit(kernel A)
    RT->>CB: vkBeginCommandBuffer
    RT->>CB: vkCmdDispatch (kernel A)
    RT->>CB: vkEndCommandBuffer
    RT->>Q: vkQueueSubmit2
    Q-->>TS: signals counter N+1
    App->>RT: queue.submit(kernel B) [depends on kernel A event]
    RT->>CB: vkBeginCommandBuffer
    RT->>CB: vkCmdDispatch (kernel B)
    RT->>CB: vkEndCommandBuffer
    RT->>Q: vkQueueSubmit2
    TS-->>Q: unblocks when counter ≥ N+1
    Q-->>TS: signals counter N+2
{% endcomment %}
![Vulkan backend sequence diagram](/assets/images/vulkan-backend-sequence-diagram.png)

## Buffer Device Address

Memory management is where SYCL's USM abstraction and Vulkan's explicitness collide most directly. Exposing
USM was a problem we needed to solve that was not achieved in Sylkan or any of the OpenCL-on-Vulkan
layered implementations to date with respect to OpenCL USM.

In AdaptiveCpp SYCL buffers are implemented on top of USM, therefore supporting USM is a crucial requirement for
the backend. There are many types of USM in SYCL but the minimum we need to support is device USM, this gives the
user a pointer to an allocation that can be dereferenced on device, but not dereferenced on host. Unlike host/shared
USM types where a pointer can be dereferenced on host and device.

Vulkan asynchronous commands operate on `VkBuffer` objects, but these themselves are backed by `VkDeviceMemory` objects
that must be allocated then bound to a `VkBuffer`. To implement USM we can't give the user a device dereferencable
pointer to `VkDeviceMemory` because Vulkan only allows the memory to be mapped to a host pointer, not directly addressed
from device. Instead the solution is to implement device USM as a `VkBuffer` backed by `VkDeviceMemory`, and use the
Buffer Device Address API
[vkGetBufferDeviceAddress()](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetBufferDeviceAddress.html) to
get a 64-bit address that can be returned to the user from `sycl::malloc_device()`.

Initially this seems straightforward, but there is a problem, the asynchronous SYCL `memcpy()` command takes pointer
parameters that are either USM allocations **or host pointers**, but `vkCmdCopyBuffer` only takes `VkBuffer` objects.
We already have the `VkBuffer` associated with the USM allocations the user created, but we need a way to map a host
pointer operand to a `VkBuffer`. This also needs to be done asynchronously, if we create a `VkBuffer` internally and copy
the host data into it immediately, then we won't respect previous asynchronous commands writing to the host data
which have not yet completed. Likewise we need a way to copy the data back from the internal `VkBuffer` to
host pointer, if it was a destination memcpy operand, after the `vkCmdCopyBuffer` command has completed.

This is where timeline semaphores' host signaling functionality becomes crucial! Through CPU multi-threading the runtime
does asynchronous host side work to copy data to/from a host pointer to `VkBuffer` while respecting the SYCL command
dependencies. Here a host worker thread is used to wait on and signal the timeline semaphore values of the queue.

{% comment %}
Mermaid source used to generate diagram below
sequenceDiagram
    participant App   as SYCL application
    participant Sched as Runtime scheduler(DAG / inorder_executor)
    participant VkQ   as SYCL queue(vk_queue)
    participant Alloc as allocation(vk_allocator)
    participant WT    as worker_thread(CPU std::thread)

    App->>Sched: queue.submit(memcpy src→dst) [both pointers are plain host ptrs]
    Sched->>VkQ: submit_memcpy(op, node)
    VkQ->>Alloc: Allocate Vulkan memory for src operand
    Alloc-->>VkQ: VkBuffer & VkDeviceMemory
    VkQ->>Alloc: Allocate Vulkan memory for dst operand
    Alloc-->>VkQ: VkBuffer & VkDeviceMemory

    VkQ->>WT: enqueue async host function for src operand copy
    WT-->>VkQ:
    Note over WT: worker_thread wakes when timeline semaphore is expected value
    WT->>WT: memcpy src ptr operand into src Vulkan memory objects
    WT->>WT: increment timeline semaphore
    Note over VkQ:Command-buffer submission wait semaphore value is completion of async worker_thread
    VkQ->>VkQ: Create vkCmdCopyBuffer command-bufer and submit.
    VkQ->>VkQ: Increment timeline semaphore value

    VkQ->>WT: enqueue async host function for dst operand copy
    WT-->>VkQ:
    VkQ-->>Sched:
    Note over WT: worker_thread wakes when timeline semaphore is expected value
    WT->>WT: memcpy dst Vulkan memory objects into dst pointer operand
    WT->>WT: increment timeline semaphore

    App->>Sched: sycl::event::wait() or nqueue.wait()
    Sched->>VkQ: vk_queue::wait()
    Note over VkQ:  Semaphore wait on completion value for async worker_thread for dst copy
    VkQ-->>Sched:
    Sched->>App:
{% endcomment %}
![Vulkan memcpy host operand sequence diagram](/assets/images/vulkan-memcpy-host-sequence-diagram.png)

# Compiler Implementation Overview

Layering the runtime is only half the battle, how to compile SYCL kernels down to Vulkan consumable shader SPIR-V presents
more fundamental challenges. While SYCL kernels can already be compiled to
SPIR-V for consumption by OpenCL and Level Zero backends, this is not the same dialect of SPIR-V as Vulkan drivers
consume. Fortunately there exists an established tool for generating Vulkan SPIR-V from compute kernels that we
can leverage, [clspv](https://github.com/google/clspv), which is used by all the layered OpenCL-on-Vulkan implementations
to compile OpenCL-C into Vulkan SPIR-V.

The clspv tool can accept LLVM IR input as well as OpenCL-C, making it suitable for integration into AdaptiveCpp's
[SSCP compilation flow](https://github.com/EwanC/AdaptiveCpp/blob/develop/doc/compilation.md).
This runtime JIT compiler allows backends to lower LLVM IR for the device using the most appropriate tooling for
that backend. For example, OpenCL/Level-Zero backends call into the
[LLVM-SPIRV translator](https://github.com/khronosgroup/spirv-llvm-translator)
to lower LLVM-IR to kernel capability SPIR-V, and the Vulkan backend calls into `clspv` in a similar way.

clspv is typically used with OpenCL-C input rather than C++ single-source IR, so we had to build a
pipeline of LLVM passes for transforming the IR into a form that's consumable by clspv.
The biggest challenge here is generic pointers which are not part of the OpenCL-C 1.2 language clspv is used
to consuming. In OpenCL-C 1.2 pointers always have an [address space qualifier](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_C.html#address-space-qualifiers)
to tell the compiler if it's a `__global`, `__local`, `__constant`, or `__private` address space pointer.
In SYCL however there are no such qualifiers and the address space of all pointers must be inferred.
This is possible in most cases but breaks down when pointers themselves are loaded from memory, as there
is no way to correctly infer the address space. So the SSCP Vulkan backend does a lot of work to try to
restore the LLVM address space in IR, but ultimately this is a fundamental mismatch between SYCL generic
pointers and SPIR-V.

Once we have generated the SPIR-V for our kernels, the SPIR-V code will advertise through
[capabilities](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#Capabilities)
the specific functionality that a Vulkan driver must support to run it. More advanced SYCL kernels require more capabilities
but the baseline SPIR-V capability support to run any kernel are:

* `physicalStorageBufferAddresses` - Use `PhysicalStorageBuffer64` memory
  model where physical pointers are allowed. This allows USM pointers to
  be used in the kernel which are used in AdaptiveCpp to implement SYCL
  buffers as well as USM.
* `shaderInt64` - Enables 64-bit integers in SPIR-V, allows the SYCL runtime
  to pass pointer parameters to a kernel as `i64` members of a SPIR-V
  push constant struct
* `variablePointers` - Allows a SPIR-V pointer to not statically know what
  object it comes from. Enables use of `OpPtrAccessChain` SPIR-V instruction
  which are required to create a pointer to the physical pointer member inside
  the push constant parameters struct.

# Using the AdaptiveCpp Vulkan Backend

Now we've covered the theory, let's see the Vulkan backend in action.

When it comes to using the Vulkan AdaptiveCpp backend, thanks to the proliferation of Vulkan drivers
there are many platforms which the backend can be tested on. We're proud that AdaptiveCpp GitHub CI now has
all of Linux, Windows, and MacOS operating systems tested on the Vulkan backend for every commit,
using [Mesa llvmpipe](https://docs.mesa3d.org/drivers/llvmpipe.html) for Linux & Windows,
and [MoltenVK](https://github.com/KhronosGroup/MoltenVK) for macOS.

In this article we'll only cover how to build and use the backend on Ubuntu using a native build flow.
Android operating systems require a more complex build process using the Android NDK to
cross compile AdaptiveCpp and other dependencies. You can find the in-depth instructions for how to do that
[here](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/install-android.md).

## Building AdaptiveCpp With Vulkan Backend Enabled

There are four main pieces to assemble before you can compile and run your first SYCL program for Vulkan: the LunarG SDK,
a clspv binary, a Vulkan driver, and AdaptiveCpp itself. The first main build dependency is the [LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home)
for SPIRV Tools, Vulkan layers & loader, and the VulkanHpp headers. The second is a `clspv` binary itself,
which can be built following the GitHub repo instructions. A Vulkan driver is also needed to
run on top of. Finally, we need to build AdaptiveCpp with the Vulkan backend enabled.

### LunarG SDK

The LunarG SDK should be available in your system path after sourcing the `setup-env.sh` script it
ships (we recommend automatically sourcing this as part of your `.bashrc`).

```sh
$ wget https://sdk.lunarg.com/sdk/download/1.4.357.0/linux/vulkansdk-linux-x86_64-1.4.357.0.tar.xz
$ tar -xvf vulkansdk-linux-x86_64-1.4.357.0.tar.xz
$ source 1.4.357.0/setup-env.sh
```

### Vulkan Driver

The Vulkan SDK comes with a `vulkaninfo` tool for printing the Vulkan drivers on your system,
at least one driver is required to use as a SYCL backend device. If you don't have any installed then the
easiest way to reliably get a supported driver is to install the Mesa drivers with
`apt install mesa-vulkan-drivers`. This will provide at least the llvmpipe CPU Vulkan driver
which provides all the necessary capabilities for SYCL. For example:

```sh
$ vulkaninfo --summary
GPU0:
	apiVersion         = 1.4.318
	driverVersion      = 25.2.8
	vendorID           = 0x10005
	deviceID           = 0x0000
	deviceType         = PHYSICAL_DEVICE_TYPE_CPU
	deviceName         = llvmpipe (LLVM 20.1.8, 256 bits)
	driverID           = DRIVER_ID_MESA_LLVMPIPE
	driverName         = llvmpipe
	driverInfo         = Mesa 25.2.8-0ubuntu0.25.10.2 (LLVM 20.1.8)
	conformanceVersion = 1.3.1.1
```

### clspv

A `clspv` executable is required to be invoked at runtime as part of SSCP compilation,
and can be built from source. The exact commit of clspv should be checked in AdaptiveCpp CI
or [doc/install-vulkan.md](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/install-vulkan.md#requirements)
for the SHA.

```sh
$ git clone https://github.com/google/clspv
$ cd clspv
$ git checkout <supported commit>
$ python3 utils/fetch_sources.py
$ mkdir build && cd build
$ cmake .. -GNinja
$ ninja
$ export CLSPV_BIN_DIR=$PWD/bin
```

The path to the directory with the `clspv` tool is exported as an environment variable so we
can reference it in a later build step.

### AdaptiveCpp

Now we can build AdaptiveCpp itself using these dependencies. Combined with the
`-DWITH_VULKAN_BACKEND=ON` option for enabling the Vulkan backend in the build,
the relevant parts of the CMake invocation are:

```sh
$ git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git
$ cd AdaptiveCpp && mkdir build && cd build
$ cmake -GNinja -DWITH_VULKAN_BACKEND=ON -DCMAKE_PROGRAM_PATH=$CLSPV_BIN_DIR -DCMAKE_INSTALL_PREFIX=$PWD/install
$ ninja install
$ export ACPP_BIN_DIR=$PWD/install/bin
```

You can then check that a Vulkan device is indeed available. Note that other devices may be available too,
but the below is the minimum expected number of devices that `acpp-info` should output.

```sh
$ $ACPP_BIN_DIR/acpp-info -l
=================Backend information===================
Loaded backend 0: OpenMP
  Found device: AdaptiveCpp OpenMP host device
Loaded backend 1: Vulkan
  Found device: llvmpipe (LLVM 20.1.8, 256 bits)
```

## Compiling and running AdaptiveCpp applications With Vulkan Backend

Let's build and run a simple SYCL application with AdaptiveCpp to show
the Vulkan backend in action.

```cpp
// sycl_test.cpp
#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>

int main() {
    sycl::device d{sycl::default_selector{}};
    sycl::queue q(d, sycl::property::queue::in_order());

    std::string device = d.get_info<sycl::info::device::name>();
    std::cout << "Default-selected queue runs on device: " << device << std::endl;

    constexpr size_t N = 1024;
    int *devicePtr = sycl::malloc_device<int>(N, q);

    q.parallel_for(N, [=](sycl::id<1> idx) {
      devicePtr[idx] = idx;
    });

    std::vector<int> dataHost(N);
    q.copy(devicePtr, dataHost.data(), N).wait();

    bool success = true;
    for (int i = 0; i < N; i++) {
      success = success && (dataHost[i] == i);
    }

    std::cout << (success ? "SYCL application SUCCESS" : "SYCL application FAILED")
              << std::endl;

    sycl::free(devicePtr, q);
    return 0;
}
```

The default compilation model for the `acpp` compiler is SSCP flow, so we don't need
to specify any extra options to ask for that. Once we have our compiled `sycl_test`
binary we can run it using the `ACPP_VISIBILITY_MASK` environment variable to
ask for the Vulkan backend as `ACPP_VISIBILITY_MASK=vk`. If there is more
than one Vulkan driver on your system then you can ask for a specific one
by name, for example here we ask for llvmpipe with `ACPP_VISIBILITY_MASK=vk:llvmpipe`.

```sh
$ $ACPP_BIN_DIR/acpp sycl_test.cpp -o sycl_test
$ ACPP_VISIBILITY_MASK=vk:llvmpipe ./sycl_test
Default-selected queue runs on device: llvmpipe (LLVM 20.1.8, 256 bits)
SYCL application SUCCESS
```
# Android Benchmarks

With the Ubuntu setup working, let's look at what this enables on Android by illustrating
the benefits of SYCL acceleration on one of the benchmarks from
[HecBench](https://github.com/ORNL/HeCBench). HecBench provides multiple source code variants of
each benchmark for different backends. We used the mandelbrot benchmark which has among others a OpenMP variant
[mandelbrot-omp](https://github.com/ORNL/HeCBench/tree/master/src/mandelbrot-omp), and SYCL variant
[mandelbrot-sycl](https://github.com/ORNL/HeCBench/tree/master/src/mandelbrot-sycl).

The benchmarks were cross compiled using release 27 of the Android Native Development Kit (NDK). The direct
OpenMP benchmark was compiled as follows:

```sh
$ cd mandelbrot-omp
$ $NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ *.cpp -O3 -o mandelbrot-omp --target=aarch64-linux-android34 -static-libstdc++
```

Note that the SYCL benchmarks in HecBench require `-DUSE_GPU=1` to be set during compilation to enable a GPU
SYCL queue selector, so for each benchmark we create two SYCL executables linked against an android cross compiled build of AdaptiveCpp
`$ACPP_NDK_BUILD`. See the
[install-android](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/install-android.md)
doc for more details on how to achieve this.

```sh
$ cd mandelbrot-sycl
$ $ACPP_BIN_DIR/acpp *.cpp -O3 -o mandelbrot-gpu-ndk -DUSE_GPU=1 --target=aarch64-linux-android34 --sysroot=$NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot --rtlib=compiler-rt -static-libstdc++  -resource-dir=$NDK/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/18/ -L $ACPP_NDK_BUILD/lib
$ $ACPP_BIN_DIR/acpp *.cpp -O3 -o mandelbrot-cpu-ndk --target=aarch64-linux-android34 --sysroot=$NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot --rtlib=compiler-rt -static-libstdc++  -resource-dir=$NDK/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/18/ -L $ACPP_NDK_BUILD/lib
```

The SYCL acceleration results speak for themselves, on an Android Device with a Qualcomm Snapdragon SOC with
Adreno 750 GPU and Arm v8a we achieved a 4.4x speedup over raw OpenMP by using the GPU exposed by Vulkan.
Taking the average parallel time over 100 iterations which is output by the benchmark, and using the median from 5 runs of the benchmark,
we observed the following:

| Benchmark                 | Average parallel time (ms) |
| ------------------------- | -------------------------- |
|`./mandelbrot-omp 100`     | 249                        |
|`./mandelbrot-cpu-ndk 100` | 89                         |
|`./mandelbrot-gpu-ndk 100` | 57                         |

The 2.8x speedup from the SYCL OpenMP backend over the straight OpenMP benchmark is something that's
observable on a desktop x64 platform and not only Android. It is due to the higher parallel execution times
in straight OpenMP than SYCL OpenMP (a statistic output by the benchmark), where the key difference is
that SYCL is runtime JIT compiling kernels with AdaptiveCpp SSCP compilation.

# Conclusion

With Vulkan as a foundation, SYCL can now reach every device class from HPC to Android phones.
This is just the beginning, there's still work left to do. Our roadmap of tasks is tracked as AdaptiveCpp GitHub issues under the
[Vulkan label](https://github.com/AdaptiveCpp/AdaptiveCpp/issues?q=is%3Aissue%20state%3Aopen%20label%3Avulkan).
Including exciting features such as [backend interop](https://github.com/AdaptiveCpp/AdaptiveCpp/issues/2111)
that could bring SYCL and Vulkan graphics APIs together in the same application.

If this work interests you please try it out and share your results. There are many different Vulkan drivers and application
workloads out there, and we'd love to know what works and what needs help.
