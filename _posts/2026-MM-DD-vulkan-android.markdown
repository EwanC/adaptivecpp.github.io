---
layout: single
title:  "Bringing SYCL to Android: A Vulkan backend for portable GPU compute"
date:   2026-MM-DD HH:00:00 +0100
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
[clvk](https://github.com/kpet/clvk), [pocl](https://github.com/pocl/pocl), and [ANCLE](https://github.com/google/angle)
successfully layering its compute API over Vulkan.

AdaptiveCpp has helped SYCL close this gap with the recently added Vulkan backend and Android cross compilation support.
This post highlights what's going on under the hood in AdaptiveCpp to layer onto of Vulkan before showing how to build and
use the Vulkan backend. Starting with an overview of the two halves of bringing up a backend target in SYCL, the runtime
and the compiler.

# Runtime Implementation Overview

The AdaptiveCpp backend runtime implementation maps the SYCL host API and semantics down to underlying backend host API calls.
Vulkan has a powerful, but very verbose API, so the Khronos [VulkanHpp](https://github.com/KhronosGroup/Vulkan-Hpp) headers
are used to reduce implementation verbosity.

Each SYCL device corresponds to a logical Vulkan device that meets the key capability criteria to implement SYCL,
namely [Timeline Semaphores](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_timeline_semaphore.html)
and [Buffer Device Address](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_buffer_device_address.html).

## Timeline Semaphores

To elaborate on why those extensions are necessary building blocks of our SYCL implementations. Each `sycl::queue::submit()`
call becomes a `vkCommandBuffer` submission with a single command, which is synchronized entirely using timeline semaphores.
Each timeline semaphore is owned by a SYCL queue with a value that is initialized to zero and incremented on each command
submission. This allows the same synchronization primitive to be reused many times without resetting, unlike a `vkFence`, and
each command to be uniquely identified by a queue handle and timeline value pair. What's more, a timeline semaphore can
also be signaled from host which is useful for reasons we'll discuss in a moment.

{% comment %}
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
{% endcomment }
![Vulkan backend sequence diagram](/assets/images/vulkan-backend-sequence-diagram.png)

## Buffer Device Address

In AdaptiveCpp SYCL buffers are implemented on top of USM, therefore supporting USM is a crucial requirement for
the backend. There are many types of USM in SYCL but the minimum we need to support is device USM, this gives the
user a pointer to an allocation which can be dereferenced on device, but not dereferenced on host. Unlike host/shared
USM types where a pointer can be dereferenced on host and device.

Vulkan asynchronous commands operate on `VkBuffer` objects, but these themselves are backed by `VkDeviceMemory` objects
that must be allocated then bound to a `VkBuffer`. To implement USM we can't give the user a device dereferencable
pointer to `VkDeviceMemory` because Vulkan only lets the object be mapped/unmapped to a host pointer. Instead the
solution is to implement device USM as a `VkBuffer` backed by `VkDeviceMemory`, and use the Buffer Device Address API
[vkGetBufferDeviceAddress()](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetBufferDeviceAddress.html) to
get a 64-bit address to that can be returned to the user from `sycl::malloc_device()`.

Initially this seems straightforward, but there is a problem, the asynchronous SYCL `memcpy()` command takes pointer
parameters that are either USM allocations **or host pointers**, but `vkCmdCopyBuffer` only takes `VkBuffer` objects.
We already have the `VkBuffer` associated with the USM allocations the user created, but we need a way to map a host
pointer operand to a `VkBuffer`. This also needs to be done asynchronously, if we create a `vkBuffer` internally and copy
the host data into it straightaway, then we won't respect previous asynchronous commands writing to the host data
which have not yet completed. Likewise we need a way to copy the data back from the internal `VkBuffer` to
host pointer, if it was a destination memcpy operand, after the `vkCmdCopyBuffer` command has completed.

This is where timeline semaphores host signaling functionality becomes crucial! Through CPU multi-threading the runtime
does asynchronous host side work to copy data to/from a host pointer to `vkBuffer` while respecting the SYCL command
dependencies. Here a host worker thread is used to wait on and signal the timeline semaphore values of the queue.

# Compiler Implementation Overview

Layering the runtime is actually the less complex half of the problem, how to compile SYCL kernels down to Vulkan
consumable shader SPIR-V presents more fundamental challenges. While SYCL kernels can already be compiled to
SPIR-V for consumption by OpenCL and Level Zero backends, this is not the same dialect of SPIR-V as Vulkan drivers
consume. Fortunately there exists an established tool for generating Vulkan SPIR-V from compute kernels,
[clspv](https://github.com/google/clspv), which is used by all the layered OpenCL-on-Vulkan implementations
to compile OpenCL-C into Vulkan SPIR-V.

The clspv tool can accept LLVM IR input as well as OpenCL-C, making it suitable for integration into AdaptiveCpp's
SSCP compilation flow. This runtime JIT compiler allows backends to lower LLVM IR for the device using the most
appropriate tooling for that backend. For example, OpenCL/Level-Zero backends call into the LLVM-SPIRV translator
to lower LLVM-IR to kernel capability SPIR-V, and the Vulkan backend calls into `clspv` in a similar way.

clspv is typically used with OpenCL-C input rather than C++ single-source IR, so a significant amount of work
went into LLVM passes to transform that IR into a form that's consumable by clspv.
The biggest challenge here is generic pointers which are not part of the OpenCL-C 1.2 language clspv is used
to consuming. In OpenCL-C 1.2 pointers always have an address space qualifier to tell the compiler if it's a
`__global`, `__local`, `__constant`, or `__private` address space pointer. In SYCL however there are no such
qualifiers and the address space of all pointers must be inferred. This is possible in most cases but breaks
down when pointers themselves are loaded from memory, as there is no way to correctly infer the address space.
So the SSCP Vulkan backend does a lot of work to try restore the LLVM address space in IR, but ultimately this
is a fundamental mismatch between SYCL generic pointers and SPIR-V.

Once we have generated the SPIR-V for our kernels, the SPIR-V code will advertise through metadata the specific
capabilities that a Vulkan driver must support to run it. More advanced SYCL kernels require more capabilities
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

When it comes to using the Vulkan AdaptiveCpp backend, thanks to the proliferation of Vulkan drivers
there are many platforms which the backend can be tested on. On AdaptiveCpp GitHub CI all of Linux, Windows, and MacOS
are testing using Mesa llvmpipe for Linux & Windows, and MoltenVK for macOS.

In this article we'll only cover how to build and use the backend on Ubuntu using a native build flow.
Android operating systems require a more complex build process using the Android NDK to
cross compile AdaptiveCpp and other dependencies. You can find the in-depth instructions for how to do that
[here](github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/install-android.md).

## Building AdaptiveCpp With Vulkan Backend Enabled

The first main build dependency is the [LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home)
for SPIRV Tools, Vulkan layers & loader, and the VulkanHpp headers. The second is a `clspv` binary itself,
which can be built following the GitHub repo instructions. Finally, a Vulkan driver itself is also needed to
run on top of.

### LunarG SDK

The LunarG SDK should be available in your system path after sourcing the `setup-env.sh` script it
ships (recommend sourcing this as part of your `.bashrc`).

```sh
$ wget https://sdk.lunarg.com/sdk/download/1.4.357.0/linux/vulkansdk-linux-x86_64-1.4.357.0.tar.xz
$ tar -xvf vulkansdk-linux-x86_64-1.4.357.0.tar.xz
$ source 1.4.357.0/setup-env.sh
```

### Vulkan Driver

The Vulkan SDK comes with a `vulkaninfo` tool for printing the Vulkan drivers on your system,
at least one driver is required to use as a SYCL backend device. If don't have any installed then the
easiest way to reliably get a supported driver is to install the Mesa drivers with
`apt install mesa-vulkan-drivers`. This will provide at least the llvmpipe CPU Vulkan driver
which provides all the necessary capabilities for SYCL.

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
$ export CLSPV_BIN_DIR=$PWD/build/bin
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

Lets build and run a simple SYCL application with AdaptiveCpp to show
the Vulkan backend in action.

```cpp
// sycl_test.cpp
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

```sh
$ $ACPP_BIN_DIR/acpp sycl_test.cpp -o sycl_test
$ ACPP_VISIBILITY_MASK=vk:llvmpipe ./sycl_test
Default-selected queue runs on device: llvmpipe (LLVM 20.1.8, 256 bits)
SYCL application SUCCESS
```

The default compilation model for the `acpp` compiler is SSCP flow, so we don't need
to specify any extra options to ask for that. Once we have our compiled `sycl_test`
binary we can run it using the `ACPP_VISIBILITY_MASK` environment variable to
ask for the Vulkan backend as `ACPP_VISIBILITY_MASK=vk`. If there is more
than one Vulkan driver on your system then you can ask for a specific one
by name, for example here we ask for llvmpipe with `ACPP_VISIBILITY_MASK=vk:llvmpipe`.

# Android Benchmarks

Returning to the Android platform, to illustrate the benefits of the SYCL acceleration
enabled by AdaptiveCpp Android support we cross compiled one of the benchmarks from
[HecBench](https://github.com/ORNL/HeCBench).

HecBench provides multiple source code variants of each benchmark for different backends. We used
the mandlebrot benchmark which has among others a OpenMP variant
[mandelbrot-omp](https://github.com/ORNL/HeCBench/tree/master/src/mandelbrot-omp), and SYCL variant
[mandelbrot-sycl](https://github.com/ORNL/HeCBench/tree/master/src/mandelbrot-sycl).

Using release 27 of the Android Native Development Kit (NDK) to compile these could be compiled as follows:

```sh
$ cd mandelbrot-omp
$ $NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ *.cpp -O3 -o mandlebrot-omp --target=aarch64-linux-android34 -static-libstdc++
```

Note that the SYCL benchmarks in HecBench require `-DUSE_GPU=1` to be set during compilation to enable a GPU
SYCL queue selector, so we create two executables linked against an android cross compiled build of AdaptiveCpp
`$ACPP_NDK_BUILD`, see the
[install-android](github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/install-android.md).
doc for more details on how to achieve this.

```sh
$ cd mandelbrot-sycl
$ $ACPP_BIN_DIR/acpp *.cpp -O3 -o mandelbrot-gpu-ndk -DUSE_GPU=1 --target=aarch64-linux-android34 --sysroot=$NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot --rtlib=compiler-rt -static-libstdc++  -resource-dir=$NDK/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/18/ -L $ACPP_NDK_BUILD/lib

$ $ACPP_BIN_DIR/acpp *.cpp -O3 -o mandelbrot-cpu-ndk --target=aarch64-linux-android34 --sysroot=$NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot --rtlib=compiler-rt -static-libstdc++  -resource-dir=$NDK/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/18/ -L $ACPP_NDK_BUILD/lib
```

On an Android Device with an Qualcomm Snapdragon SOC with Adreno 750 GPU and Arm v8a we observed the following results
from running the benchmarks. Taking the average parallel time over 100 iterations which is output by the benchmark,
and using the median from 5 runs of the benchmark.

| Benchmark                 | Average parallel time (ms) |
| ------------------------- | -------------------------- |
|`./mandelbrot-omp 100`     | 249                        |
|`./mandelbrot-cpu-ndk 100` | 89                         |
|`./mandelbrot-gpu-ndk 100` | 57                         |

We can see that with the SYCL OpenMP backend we get an improvement over the straight OpenMP benchmark, and
with GPU offloading through the SYCL Vulkan backend a further improvement still.

# Conclusion

To conclude we've introduced AdaptiveCpp's support for SYCL over Vulkan, building on the
proliferation of Vulkan drivers as a portable standard across mobile, desktop, and HPC platforms.
In particular enabling SYCL as programming model on Android platforms, exposing both the CPU
through the OpenMP backend and GPU through Vulkan backend.

There still a lot of work to do however. Improvements to implementation quality are tracked in the
AdaptiveCpp GitHub issues under the
[Vulkan label](https://github.com/AdaptiveCpp/AdaptiveCpp/issues?q=is%3Aissue%20state%3Aopen%20label%3Avulkan).
Future features are also tracked here, such as backend interop which will enable the host application to
use the same Vulkan objects across Vulkan graphics APIs and compute objects backing SYCL.
Longer term once the project is more mature, the fundamental limitations of the SYCL-on-Vulkan can be brought to the
relevant SYCL/Vulkan/SPIR-V Khronos working groups to enable explicit specification around support.
