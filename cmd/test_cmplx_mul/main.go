package main

import (
	"flag"
	"fmt"
	//"math"
	"math/rand"
	"os"

	"github.com/mumax/3cl/data"
	//"github.com/mumax/3cl/engine"
	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/opencl/cl"
)

var (
	Flag_gpu = flag.Int("gpu", 0, "Specify GPU")
	Flag_size  = flag.Int("length", 4, "length of data to test")
	Flag_print = flag.Bool("print", false, "Print out result")
	Flag_comp  = flag.Int("components", 1, "Number of components to test")
	Flag_conj  = flag.Bool("conjugate", false, "Conjugate B in multiplication")
)

func Complex_multi(plier []float32, plicant []float32,dataSize int, NComponents int) []float32 {
	if dataSize < 4 {
		fmt.Println("argument to -length must be 4 or greater!")
		os.Exit(-1)
	}
	if (NComponents < 1) || (NComponents > 3) {
		fmt.Println("argument to -components must be 1, 2 or 3!")
		os.Exit(-1)
	}

	queue := opencl.ClCmdQueue
	//	device, context, queue := opencl.ClDevice, opencl.ClCtx, opencl.ClCmdQueue
	kernels := opencl.KernList

	kernelObj := kernels["cmplx_mul"]
	totalArgs, err := kernelObj.NumArgs()
	if err != nil {
		//fmt.Printf("Failed to get number of arguments of kernel: $+v \n", err)
		fmt.Printf("\n Failed to get number of arguments of kernel: %+v ", err)

	} else {
		fmt.Printf("\n Number of arguments in kernel : %d", totalArgs)
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

	fmt.Printf("Begin first run of cmplx_mul kernel... \n")

	// Creating inputs
	fmt.Println("Generating input data...")
	dataSize *= 2
	size := [3]int{dataSize, 1, 1}
	inputs0 := make([][]float32, NComponents)
	for i := 0; i < NComponents; i++ {
		inputs0[i] = make([]float32, size[0])
		for j := 0; j < len(inputs0[i]); j++ {
			inputs0[i][j] = plier[j]
		}
	}
	inputs1 := make([][]float32, NComponents)
	for i := 0; i < NComponents; i++ {
		inputs1[i] = make([]float32, size[0])
		for j := 0; j < len(inputs1[i]); j++ {
			inputs1[i][j] = plicant[j]
		}
	}

	fmt.Println("Done. Transferring input data from CPU to GPU...")
	cpuArray0 := data.SliceFromArray(inputs0, size)
	cpuArray1 := data.SliceFromArray(inputs1, size)
	gpuBuffer0 := opencl.Buffer(NComponents, size)
	gpuBuffer1 := opencl.Buffer(NComponents, size)
	outBuffer := opencl.Buffer(NComponents, [3]int{dataSize, 1, 1})
	outArray := data.NewSlice(NComponents, [3]int{dataSize, 1, 1})

	data.Copy(gpuBuffer0, cpuArray0)
	data.Copy(gpuBuffer1, cpuArray1)

	fmt.Println("Waiting for data transfer to complete...")
	queue.Finish()
	fmt.Println("Input data transfer completed.")

	fmt.Println("Executing kernel...")
	if *Flag_conj {
		opencl.ComplexArrayMul(outBuffer, gpuBuffer0, gpuBuffer1, 1, dataSize/2, 0)
	} else {
		opencl.ComplexArrayMul(outBuffer, gpuBuffer0, gpuBuffer1, 0, dataSize/2, 0)
	}
	fmt.Println("Waiting for kernel to finish execution...")
	queue.Finish()
	fmt.Println("Execution finished.")

	fmt.Println("Retrieving results...")
	data.Copy(outArray, outBuffer)
	queue.Finish()
	fmt.Println("Done.")
	results := outArray.Host()
	plication := make([]float32, dataSize)
	for i := 0; i < NComponents; i++ {
		for j := 0; j < len(inputs1[i]); j++ {
			plication[j] = results[i][j]
		}
	}
	fmt.Printf("Finished tests on cmplx_run\n")

	fmt.Printf("freeing resources \n")
	// opencl.Recycle(gpuBuffer0)
	// opencl.Recycle(gpuBuffer1)
	// opencl.Recycle(outBuffer)
	// for _, krn := range kernels {
	// 	krn.Release()
	// }
	return plication
}

func main() {
	flag.Parse()
	N := int(*Flag_size)
	ReqComponents := int(*Flag_comp)
	//tol := 5e-7
	rand.Seed(54)
	PartA := make([]float32, 2*N)
	PartB := make([]float32, 2*N)
	print_iter := 0
	for print_iter < N {
		PartA[2*print_iter] = rand.Float32()
		PartA[2*print_iter+1] = rand.Float32()
		PartB[2*print_iter] = rand.Float32()
		PartB[2*print_iter+1] = rand.Float32()
		print_iter++
	}

	opencl.Init(*Flag_gpu)



	// opencl.Init(*engine.Flag_gpu)

	platform := opencl.ClPlatform
	fmt.Printf("Platform in use: \n")
	fmt.Printf("  Vendor: %s \n", platform.Vendor())
	fmt.Printf("  Profile: %s \n", platform.Profile())
	fmt.Printf("  Version: %s \n", platform.Version())
	fmt.Printf("  Extensions: %s \n", platform.Extensions())

	fmt.Printf("Device in use: \n")

	d := opencl.ClDevice
	//fmt.Printf("Device %d (%s): %s \n", *engine.Flag_gpu, d.Type(), d.Name())
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

	

	// for ii := 0; ii < NComponents; ii++ {
	// 	correct := 0
	// 	var tmpR0, tmpR1 float32
	// 	for i := 0; i < dataSize/2; i++ {
	// 		idx := 2*i
	// 		AX, AY := inputs0[ii][idx], inputs0[ii][idx+1]
	// 		BX, BY := inputs1[ii][idx], inputs1[ii][idx+1]
	// 		if *Flag_conj {
	// 			tmpR0 = AX*BX + AY*BY
	// 			tmpR1 = AY*BX - AX*BY
	// 		} else {
	// 			tmpR0 = AX*BX - AY*BY
	// 			tmpR1 = AX*BY + AY*BX
	// 		}
	// 		if (math.Abs(float64(results[ii][idx]-tmpR0)) < tol) && (math.Abs(float64(results[ii][idx+1]-tmpR1)) < tol) {
	// 			correct++
	// 		} else {
	// 			if *Flag_print {
	// 				fmt.Printf("Expecting [%d][%d]: (%f + i*%f)*(%f + i*%f) = (%f + i*(%f)) ; have: (%f + i*(%f)) \n", ii, idx, inputs0[ii][idx], inputs0[ii][idx+1], inputs1[ii][idx], inputs1[ii][idx+1], tmpR0, tmpR1, results[ii][idx], results[ii][idx+1])
	// 			}
	// 		}
	// 	}

	// 	if correct != dataSize/2 {
	// 		fmt.Printf("%d/%d correct values \n", correct, dataSize/2)
	// 		return
	// 	}
	// }

	fmt.Printf("\n Calling FFT function for Part A: \n	")
	PartAFFT := Complex_multi(PartA, PartB, N, ReqComponents)

	fmt.Printf("\n Size of Part A FFT is %d \n", len(PartAFFT))
	fmt.Printf("\n Printing array A \n")

	print_iter = 0
	for print_iter < N {
		fmt.Printf("(%f, %f) ", PartA[2*print_iter], PartA[2*print_iter+1])
		print_iter++
	}
	fmt.Printf("\n Printing Array B \n")
	print_iter = 0
	for print_iter < N {
		fmt.Printf("(%f, %f) ", PartB[2*print_iter], PartB[2*print_iter+1])
		print_iter++
	}

	fmt.Printf("\n Printing Result \n")
	print_iter = 0
	for print_iter < N {
		fmt.Printf("(%f, %f) ", PartAFFT[2*print_iter], PartAFFT[2*print_iter+1])
		print_iter++
	}
	fmt.Printf("\n")


	fmt.Printf("\n Calling FFT function for Part B:")

	PartBFFT := Complex_multi(PartA, PartB, N, ReqComponents)

	fmt.Printf("\n Size of Part B FFT is %d \n", len(PartBFFT))
	print_iter = 0
	for print_iter < N {
		fmt.Printf("(%f, %f) ", PartBFFT[2*print_iter], PartBFFT[2*print_iter+1])
		print_iter++
	}
	fmt.Printf("\n")

	opencl.ReleaseAndClean()
}
