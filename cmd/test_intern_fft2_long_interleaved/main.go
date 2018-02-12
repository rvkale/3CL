package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/opencl/cl"
)

var (
	Flag_platform		= flag.Int("platform", 0, "Specify OpenCL platform")
	Flag_gpu			= flag.Int("gpu", 0, "Specify GPU")
	Flag_count			= flag.Int("count", 1, "Number of data points")
)

func gold_fft2(datain []float32, N int) []float32 {
	arr := datain
	out_arr := make([]float32, 4*N)
	for i := 0; i < N; i++ {
		in0_r, in0_i := arr[2*i], arr[2*i+1]
		in1_r, in1_i := arr[2*(i+N)], arr[2*(i+N)+1]

		out_arr[2*i] = in0_r + in1_r
		out_arr[2*i+1] = in0_i + in1_i
		out_arr[2*(i+N)] = in0_r - in1_r
		out_arr[2*(i+N)+1] = in0_i - in1_i
	}
	return out_arr
}

func gold_ifft2(datain []float32, N int) []float32 {
	arr := datain
	out_arr := make([]float32, 4*N)
	for i := 0; i < N; i++ {
		in0_r, in0_i := 0.5 * arr[2*i], 0.5 * arr[2*i+1]
		in1_r, in1_i := 0.5 * arr[2*(i+N)], 0.5 * arr[2*(i+N)+1]

		out_arr[2*i] = in0_r + in1_r
		out_arr[2*i+1] = in0_i + in1_i
		out_arr[2*(i+N)] = in0_r - in1_r
		out_arr[2*(i+N)+1] = in0_i - in1_i
	}
	return out_arr
}

func main() {
	flag.Parse()

	nElem := *Flag_count
	testSz := nElem * (int)(4)
	data := make([]float32, testSz)
	for i := 0; i < len(data); i++ {
		data[i] = rand.Float32()
	}
	gold_res := gold_fft2(data[:], nElem)
	gold_ret := gold_ifft2(gold_res[:], nElem)

	opencl.Init(*Flag_platform, *Flag_gpu)
	platforms := opencl.ClPlatforms
	fmt.Printf("Discovered platforms: \n")
	for i, p := range platforms {
		fmt.Printf("Platform %d: \n", i)
		fmt.Printf("  Name: %s \n", p.Name())
		fmt.Printf("  Vendor: %s \n", p.Vendor())
		fmt.Printf("  Profile: %s \n", p.Profile())
		fmt.Printf("  Version: %s \n", p.Version())
		fmt.Printf("  Extensions: %s \n", p.Extensions())
	}
	platform := opencl.ClPlatform
	fmt.Printf("In use: \n")
	fmt.Printf("  Vendor: %s \n", platform.Vendor())
	fmt.Printf("  Profile: %s \n", platform.Profile())
	fmt.Printf("  Version: %s \n", platform.Version())
	fmt.Printf("  Extensions: %s \n", platform.Extensions())

	fmt.Printf("Discovered devices: \n")
	devices := opencl.ClDevices
	deviceIndex := -1
	for i, d := range devices {
		if deviceIndex < 0 && d.Type() == cl.DeviceTypeGPU {
			deviceIndex = i
		}
		fmt.Printf("Device %d (%s): %s \n", i, d.Type(), d.Name())
		fmt.Printf("  Address Bits: %d \n", d.AddressBits())
		fmt.Printf("  Available: %+v \n", d.Available())
		fmt.Printf("  Compiler Available: %+v \n", d.CompilerAvailable())
		fmt.Printf("  Double FP Config: %s \n", d.DoubleFPConfig())
		fmt.Printf("  Driver Version: %s \n", d.DriverVersion())
		fmt.Printf("  Error Correction Supported: %+v \n", d.ErrorCorrectionSupport())
		fmt.Printf("  Execution Capabilities: %s \n", d.ExecutionCapabilities())
		fmt.Printf("  Extensions: %s \n", d.Extensions())
		fmt.Printf("  Global Memory Cache Type: %s \n", d.GlobalMemCacheType())
		fmt.Printf("  Global Memory Cacheline Size: %d KB \n", d.GlobalMemCachelineSize()/1024)
		fmt.Printf("  Global Memory Size: %d MB \n", d.GlobalMemSize()/(1024*1024))
		fmt.Printf("  Half FP Config: %s \n", d.HalfFPConfig())
		fmt.Printf("  Host Unified Memory: %+v \n", d.HostUnifiedMemory())
		fmt.Printf("  Image Support: %+v \n", d.ImageSupport())
		fmt.Printf("  Image2D Max Dimensions: %d x %d \n", d.Image2DMaxWidth(), d.Image2DMaxHeight())
		fmt.Printf("  Image3D Max Dimensions: %d x %d x %d \n", d.Image3DMaxWidth(), d.Image3DMaxHeight(), d.Image3DMaxDepth())
		fmt.Printf("  Little Endian: %+v \n", d.EndianLittle())
		fmt.Printf("  Local Mem Size Size: %d KB \n", d.LocalMemSize()/1024)
		fmt.Printf("  Local Mem Type: %s \n", d.LocalMemType())
		fmt.Printf("  Max Clock Frequency: %d \n", d.MaxClockFrequency())
		fmt.Printf("  Max Compute Units: %d \n", d.MaxComputeUnits())
		fmt.Printf("  Max Constant Args: %d \n", d.MaxConstantArgs())
		fmt.Printf("  Max Constant Buffer Size: %d KB \n", d.MaxConstantBufferSize()/1024)
		fmt.Printf("  Max Mem Alloc Size: %d KB \n", d.MaxMemAllocSize()/1024)
		fmt.Printf("  Max Parameter Size: %d \n", d.MaxParameterSize())
		fmt.Printf("  Max Read-Image Args: %d \n", d.MaxReadImageArgs())
		fmt.Printf("  Max Samplers: %d \n", d.MaxSamplers())
		fmt.Printf("  Max Work Group Size: %d \n", d.MaxWorkGroupSize())
		fmt.Printf("  Preferred Work Group Size: %d \n", opencl.ClPrefWGSz)
		fmt.Printf("  Max Work Item Dimensions: %d \n", d.MaxWorkItemDimensions())
		fmt.Printf("  Max Work Item Sizes: %d \n", d.MaxWorkItemSizes())
		fmt.Printf("  Max Write-Image Args: %d \n", d.MaxWriteImageArgs())
		fmt.Printf("  Memory Base Address Alignment: %d \n", d.MemBaseAddrAlign())
		fmt.Printf("  Native Vector Width Char: %d \n", d.NativeVectorWidthChar())
		fmt.Printf("  Native Vector Width Short: %d \n", d.NativeVectorWidthShort())
		fmt.Printf("  Native Vector Width Int: %d \n", d.NativeVectorWidthInt())
		fmt.Printf("  Native Vector Width Long: %d \n", d.NativeVectorWidthLong())
		fmt.Printf("  Native Vector Width Float: %d \n", d.NativeVectorWidthFloat())
		fmt.Printf("  Native Vector Width Double: %d \n", d.NativeVectorWidthDouble())
		fmt.Printf("  Native Vector Width Half: %d \n", d.NativeVectorWidthHalf())
		fmt.Printf("  OpenCL C Version: %s \n", d.OpenCLCVersion())
		fmt.Printf("  Profile: %s \n", d.Profile())
		fmt.Printf("  Profiling Timer Resolution: %d \n", d.ProfilingTimerResolution())
		fmt.Printf("  Vendor: %s \n", d.Vendor())
		fmt.Printf("  Version: %s \n", d.Version())
	}
	device, context, queue := opencl.ClDevice, opencl.ClCtx, opencl.ClCmdQueue
	kernels := opencl.KernList

	kernelObj := kernels["fft2_c2c_long_interleaved_oop"]
	totalArgs, err := kernelObj.NumArgs()
	if err != nil {
		fmt.Printf("Failed to get number of arguments of kernel: $+v \n", err)
	} else {
		fmt.Printf("Number of arguments in kernel : %d \n", totalArgs)
	}
	for i := 0; i < totalArgs; i++ {
		name, err := kernelObj.ArgName(i)
		if err == cl.ErrUnsupported {
			break
		} else if err != nil {
			fmt.Printf("GetKernelArgInfo for name failed: %+v \n", err)
			break
		} else {
			fmt.Printf("Kernel arg %d: %s \n", i, name)
		}
	}

	fmt.Printf("Begin first run of fft2_c2c_long_interleaved_oop kernel... \n");

	input, err := context.CreateEmptyBuffer(cl.MemReadOnly, 4*len(data))
	if err != nil {
		fmt.Printf("CreateBuffer failed for input: %+v \n", err)
		return
	}
	output, err := context.CreateEmptyBuffer(cl.MemReadOnly, 4*len(data))
	if err != nil {
		fmt.Printf("CreateBuffer failed for output: %+v \n", err)
		return
	}
	rev_in, err := context.CreateEmptyBuffer(cl.MemReadOnly, 4*len(data))
	if err != nil {
		fmt.Printf("CreateBuffer failed for input: %+v \n", err)
		return
	}
	if _, err := queue.EnqueueWriteBufferFloat32(input, true, 0, data[:], nil); err != nil {
		fmt.Printf("EnqueueWriteBufferFloat32 failed: %+v \n", err)
		return
	}
	if err := kernelObj.SetArgs(input, output, uint32(nElem), uint32(1), uint32(2)); err != nil {
		fmt.Printf("SetKernelArgs failed: %+v \n", err)
		return
	}

	local, err := kernelObj.WorkGroupSize(device)
	if err != nil {
		fmt.Printf("WorkGroupSize failed: %+v \n", err)
		return
	}
	fmt.Printf("Work group size: %d \n", local)
	size, _ := kernelObj.PreferredWorkGroupSizeMultiple(nil)
	fmt.Printf("Preferred Work Group Size Multiple: %d \n", size)

	global := len(data)
	d := len(data) % local
	if d != 0 {
		global += local - d
	}
	if _, err := queue.EnqueueNDRangeKernel(kernelObj, nil, []int{global}, []int{local}, nil); err != nil {
		fmt.Printf("EnqueueNDRangeKernel failed: %+v \n", err)
		return
	}

	if err := queue.Finish(); err != nil {
		fmt.Printf("Finish failed: %+v \n", err)
		return
	}

	results := make([]float32, len(data))
	if _, err := queue.EnqueueReadBufferFloat32(output, true, 0, results, nil); err != nil {
		fmt.Printf("EnqueueReadBufferFloat32 failed: %+v \n", err)
		return
	}

	fmt.Printf("First run of fft2_c2c_long_interleaved_oop kernel completed and starting inverse to check... \n");

	kernelObj1 := kernels["ifft2_c2c_long_interleaved_oop"]
	totalArgs, err = kernelObj1.NumArgs()
	if err != nil {
		fmt.Printf("Failed to get number of arguments of kernel: $+v \n", err)
	} else {
		fmt.Printf("Number of arguments in kernel : %d \n", totalArgs)
	}
	for i := 0; i < totalArgs; i++ {
		name, err := kernelObj1.ArgName(i)
		if err == cl.ErrUnsupported {
			break
		} else if err != nil {
			fmt.Printf("GetKernelArgInfo for name failed: %+v \n", err)
			break
		} else {
			fmt.Printf("Kernel arg %d: %s \n", i, name)
		}
	}

	if err = kernelObj1.SetArgs(output, rev_in, uint32(nElem), uint32(1), uint32(2)); err != nil {
		fmt.Printf("SetKernelArgs failed: %+v \n", err)
		return
	}

	local, err = kernelObj1.WorkGroupSize(device)
	if err != nil {
		fmt.Printf("WorkGroupSize failed: %+v \n", err)
		return
	}
	fmt.Printf("Work group size: %d \n", local)
	size, _ = kernelObj1.PreferredWorkGroupSizeMultiple(nil)
	fmt.Printf("Preferred Work Group Size Multiple: %d \n", size)

	fmt.Printf("Begin first run of ifft2_c2c_long_interleaved_oop kernel... \n");

	if _, err := queue.EnqueueNDRangeKernel(kernelObj1, nil, []int{global}, []int{local}, nil); err != nil {
		fmt.Printf("EnqueueNDRangeKernel failed: %+v \n", err)
		return
	}

	if err := queue.Finish(); err != nil {
		fmt.Printf("Finish failed: %+v \n", err)
		return
	}

	inverse_input := make([]float32, len(data))

	if _, err := queue.EnqueueReadBufferFloat32(rev_in, true, 0, inverse_input, nil); err != nil {
		fmt.Printf("EnqueueReadBufferFloat32 failed: %+v \n", err)
		return
	}

	// Print data input
	for i := 0; i < nElem; i++ {
		fmt.Printf("Data In: (")
		for j := 0; j < 2; j++ {
			fmt.Printf("%f + i*%f", data[2*i+j*2*nElem], data[2*i+j*2*nElem+1])
			if j == 1 {
			} else {
				fmt.Printf("; ")
			}
		}
		fmt.Printf(")\n")
	}
	
	// Print golden result
	for i := 0; i < nElem; i++ {
		fmt.Printf("Golden Result: (")
		for j := 0; j < 2; j++ {
			fmt.Printf("%f + i*%f", gold_res[2*i+j*2*nElem], gold_res[2*i+j*2*nElem+1])
			if j == 1 {
			} else {
				fmt.Printf("; ")
			}
		}
		fmt.Printf(")\n")
	}
	
	// Print returned result
	for i := 0; i < nElem; i++ {
		fmt.Printf("Returned FFT2: (")
		for j := 0; j < 2; j++ {
			fmt.Printf("%f + i*%f", results[2*i+j*2*nElem], results[2*i+j*2*nElem+1])
			if j == 1 {
			} else {
				fmt.Printf("; ")
			}
		}
		fmt.Printf(")\n")
	}
	
	// Print inversed FFT
	for i := 0; i < nElem; i++ {
		fmt.Printf("Inverse FFT2: (")
		for j := 0; j < 2; j++ {
			fmt.Printf("%f + i*%f", inverse_input[2*i+j*2*nElem], inverse_input[2*i+j*2*nElem+1])
			if j == 1 {
			} else {
				fmt.Printf("; ")
			}
		}
		fmt.Printf(")\n")
	}
	
	correct := 0
	max_relerr, max_abserr := float64(-1e-6), float64(-1e-6)
	rel_num, abs_num := float32(-1e-12), float32(-1e-12)
	for i, v := range inverse_input {
		if gold_ret[i] == v {
			correct++
		} else {
			if gold_ret[i] != 0 {
				tmp1 := math.Abs(float64(v - gold_ret[i]));
				if tmp1 > max_abserr {
					max_abserr = tmp1
					abs_num = gold_ret[i]
				}
				tmp := (v - gold_ret[i]) / gold_ret[i]
				tmp1 = math.Abs(float64(tmp))
				if tmp1 < 1e-6 {
					correct++
				} else {
					if tmp1 > max_relerr {
						max_relerr = tmp1
						rel_num = gold_ret[i]
					}
				}
			} else {
				tmp2 := math.Abs(float64(v))
				if tmp2 < 1e-6 {
					correct++
				} else {
					if tmp2 > max_abserr {
						max_abserr = tmp2
						abs_num = v
					}
				}
			}
		}
	}

	if correct != len(data) {
		fmt.Printf("%d/%d correct values \n", correct, len(data))
		fmt.Printf("Max. rel. error: %g; Number: %g\n", max_relerr, rel_num)
		fmt.Printf("Max. abs. error: %g; Number: %g\n", max_abserr, abs_num)
		return
	}
	
	fmt.Printf("Finished tests on fft2_c2c_long_interleaved_oop\n")

	fmt.Printf("freeing resources \n")
	input.Release()
	output.Release()
	for _, krn := range kernels {
		krn.Release()
	}

	opencl.ReleaseAndClean()
}