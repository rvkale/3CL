package main

import (
	"fmt"
	"math"
	"math/cmplx"
	"encoding/json"
	"os"
	"io/ioutil"
	"strconv"
	"flag"
	"math/rand"
	"unsafe"
	"github.com/mumax/3cl/data"
	//"github.com/mumax/3cl/engine"
	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/opencl/cl"
)

// json filenames are hardcoded and are required to be in the same directory currently
var (
	Flag_gpu = flag.Int("gpu", 0, "Specify GPU")
	Flag_size  = flag.Int("length", 4, "length of data to test")
	Flag_print = flag.Bool("print", false, "Print out result")
	Flag_comp  = flag.Int("components", 1, "Number of components to test")
	Flag_conj  = flag.Bool("conjugate", false, "Conjugate B in multiplication")
)

/*******************************Function for finding blusteins length begins here*******************************/
func findLength(tempLength int, fileName string) int {
	
	var j int
	m := make(map[string]int)
	strLength := strconv.Itoa(tempLength)

	jsonFile, _ := os.Open(fileName)
	defer jsonFile.Close()
	byteValue, _ := ioutil.ReadAll(jsonFile)
	json.Unmarshal([]byte(byteValue), &m)

	j = m[strLength]

	fmt.Printf("The value of the required length is: %v", j)

	m = nil	
	
	return j

}

func blusteinCase(length int) (int,int) {
	switch {
	case length > 128000000:
		return length,-1
	case length > 115200000:
		return findLength(length, "new_length_lookup_10.json"),1 //Hardcoded filenames
	case length > 102400000:
		return findLength(length, "new_length_lookup_9.json"),1
	case length > 89600000:
		return findLength(length, "new_length_lookup_8.json"),1
	case length > 76800000:
		return findLength(length, "new_length_lookup_7.json"),1
	case length > 64000000:
		return findLength(length, "new_length_lookup_6.json"),1
	case length > 51200000:
		return findLength(length, "new_length_lookup_5.json"),1
	case length > 38400000:
		return findLength(length, "new_length_lookup_4.json"),1
	case length > 25600000:
		return findLength(length, "new_length_lookup_3.json"),1
	case length > 12800000:
		return findLength(length, "new_length_lookup_2.json"),1
	case length > 1:
		return findLength(length, "new_length_lookup_1.json"),1
	case length < 2:
		return length,-2
	}
	return length,-3
}

/**++++++++++++++++++++++++++++++++++++++++++++++Function for finding blusteins length ends here++++++++++++++++++++++++++++****/

//Function to add zeros
func AddZero(x []float32, zeroLength int) []float32 {
	if len(x) >= 2*zeroLength {
		return x
	}

	r := make([]float32, 2 * zeroLength)
	copy(r, x)
	return r
}


//Functions to remove zeros
func RemoveZero(x []float32, origLength int) []float32 {
	if len(x) <= 2*origLength {
		return x
	}

	r := make([]float32, 2*origLength)
	copy(r,x)
	return r
}

//Function to find A part for the forward FFT
func PreProcessA(x []float32, origLength int) []float32 {
	processedA := make([]float32, len(x))
	var tempVal complex128
	//fmt.Printf("\n Length of array %d", len(x))
	for iter := 0; iter < int(len(x)/2);iter++ {
		if iter<origLength {
			tempVal = complex(float64(x[2*iter]),float64(x[2*iter + 1])) * cmplx.Exp(math.Pi * cmplx.Pow(complex(float64(iter),float64(0)),complex128(2)) * (-1/complex(float64(origLength),float64(0))) * (1i))
			processedA[2*iter] = float32(real(tempVal))
			processedA[2*iter+1] = float32(imag(tempVal))
		}
		 
	}
	return processedA
}

//Function to find B part for the forward FFT
func PreProcessB(newLength int, origLength int) []float32 {
	processedB := make([]float32, 2*newLength)
	var tempVal complex128
	for iter := 0; iter < int(newLength);iter++ {
		//fmt.Printf("\n Executing")
		if iter<origLength {
			tempVal = cmplx.Exp(math.Pi * cmplx.Pow(complex(float64(iter),float64(0)),complex128(2)) * (1/complex(float64(origLength),float64(0))) * (1i))
		} else if iter<newLength {
			tempVal = cmplx.Exp(math.Pi * cmplx.Pow(complex(float64(newLength-iter),float64(0)),complex128(2)) * (1/complex(float64(origLength),float64(0))) * (1i))
		}
		processedB[2*iter] = float32(real(tempVal))
		processedB[2*iter+1] = float32(imag(tempVal))
		 
	}
	return processedB
}
//Function to find twiddle factor to multiply after A*B for forward FFT
func ForwFftTwid(newLength int,origLength int) []float32 {
	ForwTwid := make([]float32, 2*newLength)
	var tempVal complex128
	for iter := 0; iter < newLength;iter++ {
		tempVal = cmplx.Exp(math.Pi * cmplx.Pow(complex(float64(iter),0),complex(float64(2),0)) * (1/complex(float64(origLength),float64(0))) * (-1i))
		ForwTwid[2*iter] = float32(real(tempVal))
		ForwTwid[2*iter + 1] = float32(imag(tempVal))

	}	
	return ForwTwid
}

//Function to find A part for the Inverse FFT
func InvProcessA(x []float32, origLength int) []float32 {
	FilteredA := make([]float32, len(x))
	var tempVal complex128
	//fmt.Printf("\n Length of array %d", len(x))
	for iter := 0; iter < int(len(x)/2);iter++ {
		if iter<origLength {
			tempVal = complex(float64(x[2*iter]),float64(x[2*iter + 1])) * cmplx.Exp(math.Pi * cmplx.Pow(complex(float64(iter),float64(0)),complex128(2)) * (1/complex(float64(origLength),float64(0))) * (1i))
			FilteredA[2*iter] = float32(real(tempVal))
			FilteredA[2*iter+1] = float32(imag(tempVal))
		}
		 
	}
	return FilteredA
}

//Function to find B part for the Inverse FFT
func InvProcessB(newLength int, origLength int) []float32 {
	FilteredB := make([]float32, 2*newLength)
	var tempVal complex128
	for iter := 0; iter < int(newLength);iter++ {
		//fmt.Printf("\n Executing")
		if iter<origLength {
			tempVal = cmplx.Exp(math.Pi * cmplx.Pow(complex(float64(iter),float64(0)),complex128(2)) * (-1/complex(float64(origLength),float64(0))) * (1i))
		} else if iter<newLength {
			tempVal = cmplx.Exp(math.Pi * cmplx.Pow(complex(float64(newLength-iter),float64(0)),complex128(2)) * (-1/complex(float64(origLength),float64(0))) * (1i))
		}
		FilteredB[2*iter] = float32(real(tempVal))
		FilteredB[2*iter+1] = float32(imag(tempVal))
		 
	}
	return FilteredB
}


//Function to find twiddle factor to multiply after a*b for Inverse FFT
func InvFftTwid(newLength int, origLength int) []float32 {
	InvTwid := make([]float32, 2*newLength)
	var tempVal complex128
	for iter := 0; iter < newLength;iter++ {
		//tempVal = cmplx.Exp(math.Pi * cmplx.Pow(complex(float64(iter),0),complex(float64(2),0)) * (1i))
		tempVal = cmplx.Exp(math.Pi * complex(float64(iter),0) * complex(float64(iter),0) * (1/complex(float64(origLength),float64(0))) * (1i))
		InvTwid[2*iter] = float32(real(tempVal))
		InvTwid[2*iter + 1] = float32(imag(tempVal))

	}	
	return InvTwid
}

// Function to find ClFFT of the given []float32

func FindClfft(InputData []float32,N int,Direction string) []float32 {
	context := opencl.ClCtx
	queue := opencl.ClCmdQueue
	//kernels := opencl.KernList

	/* print input array */
	fmt.Printf("\n Performing fft on an one dimensional array of size N = %d \n", N)
	// print_iter := 0
	// for print_iter < N {
	// 	fmt.Printf("(%f, %f) ", X[2*print_iter], X[2*print_iter + 1])
	// 	print_iter++
	// }
	// fmt.Printf("\n\nfft result: \n")

	/* Prepare OpenCL memory objects and place data inside them. */
	bufX, errC := context.CreateEmptyBuffer(cl.MemWriteOnly, N*2*int(unsafe.Sizeof(InputData[0])))
	bufOut, errCO := context.CreateEmptyBuffer(cl.MemReadOnly, N*2*int(unsafe.Sizeof(InputData[0])))

	if errC != nil {
		fmt.Printf("unable to create input buffer: %+v \n ", errC)
	}
	if errCO != nil {
		fmt.Printf("unable to create output buffer: %+v \n ", errCO)
	}

	if _, err := queue.EnqueueWriteBufferFloat32(bufX, true, 0, InputData[:], nil); err != nil {
		fmt.Printf("failed to write data into buffer \n")
	}

	flag := cl.CLFFTDim1D
	fftPlanHandle, errF := cl.NewCLFFTPlan(context, flag, []int{N}) //Don't change this to 2*N
	if errF != nil {
		fmt.Printf("unable to create new fft plan \n")
	}
	errF = fftPlanHandle.SetSinglePrecision()
	if errF != nil {
		fmt.Printf("unable to set fft precision \n")
	}
	ArrLayout := cl.NewArrayLayout()
	ArrLayout.SetInputLayout(cl.CLFFTLayoutComplexInterleaved)
	ArrLayout.SetOutputLayout(cl.CLFFTLayoutComplexInterleaved)
	errF = fftPlanHandle.SetResultOutOfPlace()
	if errF != nil {
		fmt.Printf("unable to set fft result location \n")
	}

	/* Bake the plan. */
	errF = fftPlanHandle.BakePlanSimple([]*cl.CommandQueue{queue})
	if errF != nil {
		fmt.Printf("unable to bake fft plan: %+v \n", errF)
	}

	/* Execute the plan. */
	if Direction == "frw" {
		_, errF = fftPlanHandle.EnqueueForwardTransform([]*cl.CommandQueue{queue}, nil, []*cl.MemObject{bufX}, []*cl.MemObject{bufOut}, nil)
		if errF != nil {
			fmt.Printf("unable to enqueue transform: %+v \n", errF)
		} else {
			fmt.Printf("\n Executing forward transform...\n")
		}
	} else if Direction == "inv" {
		_, errF = fftPlanHandle.EnqueueBackwardTransform([]*cl.CommandQueue{queue}, nil, []*cl.MemObject{bufX}, []*cl.MemObject{bufOut}, nil)
		if errF != nil {
			fmt.Printf("unable to enqueue transform: %+v \n", errF)
		} else {
			fmt.Printf("\n Executing inverse transform... \n ")
		}
	} else {
		panic("\n Wrong direction entered for FFT. It is neither forward not inverse.")
	}
	// _, errF = fftPlanHandle.EnqueueForwardTransform([]*cl.CommandQueue{queue}, nil, []*cl.MemObject{bufX}, []*cl.MemObject{bufOut}, nil)
	// if errF != nil {
	// 	fmt.Printf("unable to enqueue transform: %+v \n", errF)
	// }

	errF = queue.Flush()
	if errF != nil {
		fmt.Printf("unable to flush queue: %+v \n", errF)
	}

	/* Fetch results of calculations. */
	_, errF = queue.EnqueueReadBufferFloat32(bufOut, true, 0, InputData, nil)
	errF = queue.Flush()
	if errF != nil {
		fmt.Printf("unable to read output buffer: %+v \n", errF)
	}

	/* print output array */
	// print_iter = 0
	// for print_iter < N {
	// 	fmt.Printf("(%f, %f) ", X[2*print_iter], X[2*print_iter+1])
	// 	print_iter++
	// }
	// fmt.Printf("\n")

	fmt.Printf("Finished tests on clFFT\n")
	fftPlanHandle.Destroy()
	// cl.TeardownCLFFT()

	// fmt.Printf("Begin releasing resources\n")
	// for _, krn := range kernels {
	// 	krn.Release()
	// }

	//bufX.Release()
	//bufOut.Release()
	//context.Release()
	//queue.Release()

	//opencl.ReleaseAndClean()

	return InputData
}



/**********Function for Complex Multiplication******************/
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

	fmt.Printf("\n Begin first run of cmplx_mul kernel... \n")

	// Creating inputs
	fmt.Println("\n Generating input data...")
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
	var Desci int //Descision variable
	N := int(*Flag_size)
	ReqComponents := int(*Flag_comp)
	rand.Seed(91)
	// fmt.Printf("Enter the length as 67 for now: ")
	// _, err := fmt.Scanf("%d", &N)
	// if err!= nil {panic("Serious Error!")}
	X := make([]float32, 2*N)


	/* Print input array */

	print_iter := 0
	for print_iter < N {
		x := rand.Float32()
		y := rand.Float32()
		X[2*print_iter] = x
		X[2*print_iter+1] = y
		fmt.Printf("(%f, %f) ", x, y)
		print_iter++
	}

	var BluN int

	BluN = 2*N //Minimum condition for Blustein's M>=2*N

	
	//Check if new length is valid and if Blusteins Algorithm is required

	FinalN,Desci := blusteinCase(BluN)

	switch Desci {
	case -1:
		panic("\n Error! Length too large to handle! Terminating immidiately...")
	case -2:
		panic("\n Error! Length too small/negative to handle! Terminating immidiately...")
	case -3:
		panic("\n Something is weird! Terminating... Check immidiately...")
	case 1:
		if FinalN == 0 {
			fmt.Printf("\n Bluestein is not required. Executing clFFT with length %v...", BluN)
			FinalN = BluN
		} else {
			fmt.Printf("\n Bluestein is required. The value of bluestein length is %v ", FinalN)
		}
	}

	
	/* Prepare OpenCL memory objects and place data inside them for . */
	flag.Parse()
	opencl.Init(*Flag_gpu) //Initialize GPU with a flag to pick the desired gpu
	//opencl.Init(*engine.Flag_gpu)


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

	/* Zero Padding for adjusting the length if necessary*/
	ZeroForwPadX := AddZero(X, FinalN) //Padding zeros to extend lenth

	/********************************************************Forward FFT Part A begins***************************************************/


	ForwFftA := PreProcessA(ZeroForwPadX, N) //Part A for Forward FFT

	fmt.Printf("\n Finished adding zeros \n")

	fmt.Printf("\n Calculating FFT of part A... \n")

	PartAForwFFT := FindClfft(ForwFftA, FinalN, "frw")

	fmt.Printf("\n Finished calculating FFT of part A...\n")
	
	/***+++++++++++++++++++++++++++++++++++++++++++++++++++++++Forward FFT Part A ends++++++++++++++++++++++++++++++++++++++++++++++*****/

	/**********************************************************Forward FFT Part B begins*************************************************/

	ForwFftB := PreProcessB(FinalN, N)

	fmt.Printf("\n Calculating FFT of part B...\n")

	PartBForwFFT := FindClfft(ForwFftB, FinalN, "frw")

	fmt.Printf("\n Finished calculating FFT of part B...\n ")
	/*++++++++++++++++++++++++++++++++++++++++++++++++++++Forward FFT Part B ends here++++++++++++++++++++++++++++++++++++++++++++++++***/


	/*********************Bitwise multiplication for Forward FFT of Part A and Part B begins here*******************************************/
	// queue := opencl.ClCmdQueue
	// //	device, context, queue := opencl.ClDevice, opencl.ClCtx, opencl.ClCmdQueue

	fmt.Printf("\n Calculating multiplication of forward  A*B...\n")

	DftMulAB := Complex_multi(PartAForwFFT, PartBForwFFT, FinalN, ReqComponents)

	fmt.Printf("\n Finished calculating multiplication of forward  A*B...\n")
	
	/***++++++++++++++++++Bitwise multiplication for Forward FFT of Part A and Part B ends here++++++++++++++++++++++++++++++++++++++++***/



	/***********************************Forward DFT by taking iverse of A* B begins here*************************************************************/
	fmt.Printf("\n Calculating Inverse FFT of  A*B...\n")

	InvAxB := FindClfft(DftMulAB, FinalN, "inv")

	fmt.Printf("\n Finished calculating Inverse FFT of A*B...\n ")
	/*++++++++++++++++++++++++++++++++++Forward DFT by taking iverse of A* B ends here++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++***/

	/*********************Bitwise multiplication for Forward FFT with Twiddle Factor begins here****************************************/
	
	
	ForwTwiddle := ForwFftTwid(FinalN, N)

	fmt.Printf("\n Calculating multiplication with forw Twiddle...\n")

	ForwFFTfinal := Complex_multi(ForwTwiddle,InvAxB, FinalN, ReqComponents)

	fmt.Printf("\n Finished calculating multiplication with forw Twiddle...\n ")
	
		
	/***++++++++++++++++++Bitwise multiplication for Forward FFT with Twiddle Factor ends here++++++++++++++++++++++++++++++++++++++++***/



	/***Remove padded zeros to get answer for the correct length*********/

	FinalDftX := RemoveZero(ForwFFTfinal, N) //Removing zeros to extend lenth

	fmt.Printf("\n Checking FFT......\n")
	print_iter = 0
	for print_iter < N {
		fmt.Printf("(%f, %f) ", FinalDftX[2*print_iter], FinalDftX[2*print_iter+1])
		print_iter++
	}
	fmt.Printf("\n")

	




	/**************************************Inverse DFT**********************************/

	/* Zero Padding for adjusting the length if necessary*/
	ZeroInvPadX := AddZero(FinalDftX, FinalN) //Padding zeros to extend lenth

	fmt.Printf("\n Finished adding zeros \n")

	/********************************************************Inverse FFT Part A begins***************************************************/

	InvFftA := InvProcessA(ZeroInvPadX, N) //Part A for Inverse FFT

	
	
	fmt.Printf("\n Calculating FFT of Inverse part A... \n")

	PartAInvFFT := FindClfft(InvFftA, FinalN, "frw")

	fmt.Printf("\n Finished calculating FFT of Inverse part A...\n")
	

	/***+++++++++++++++++++++++++++++++++++++++++++++++++++++++Inverse FFT Part A ends++++++++++++++++++++++++++++++++++++++++++++++*****/

	/**********************************************************Inverse FFT Part B begins*************************************************/

	InvFftB := InvProcessB(FinalN, N)

	fmt.Printf("\n Calculating FFT of Inverse part B...\n")

	PartBInvFFT := FindClfft(InvFftB, FinalN, "frw")

	fmt.Printf("\n Finished calculating FFT of Inverse part B...\n ")

	/*++++++++++++++++++++++++++++++++++++++++++++++++++++Inverse FFT Part B ends here++++++++++++++++++++++++++++++++++++++++++++++++***/

	/*********************Bitwise multiplication for Inverse FFT Part a and Part b begins here****************************************/
	fmt.Printf("\n Calculating multiplication of Inverse  A*B...\n")

	DftInvaxb := Complex_multi(PartAInvFFT, PartBInvFFT, FinalN, ReqComponents)

	fmt.Printf("\n Finished calculating multiplication of Inverse  A*B...\n")
	/***++++++++++++++++++Bitwise multiplication for Inverse FFT Part a and Part b ends here++++++++++++++++++++++++++++++++++++++++***/
	
	
	/***********************************Inverse DFT by taking iverse of A* B begins here*************************************************************/
	fmt.Printf("\n Calculating Inverse FFT of inverse A*B...\n")

	InvOfaxb := FindClfft(DftInvaxb, FinalN, "inv")

	fmt.Printf("\n Finished calculating Inverse FFT of inverse A*B...\n ")
	/*++++++++++++++++++++++++++++++++++Inverse DFT by taking iverse of A* B ends here++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++***/

	/*********************Bitwise multiplication for Inverse FFT with Twiddle Factor begins here****************************************/
	
	InvTwiddle := InvFftTwid(FinalN, N)

	fmt.Printf("\n Calculating multiplication with Inv Twiddle......\n")

	InvFFTfinal := Complex_multi(InvTwiddle,InvOfaxb, FinalN, ReqComponents)

	fmt.Printf("\n Finished calculating multiplication with Inv Twiddle......\n ")
	
	/***++++++++++++++++++Bitwise multiplication for Inverse FFT with Twiddle Factor ends here++++++++++++++++++++++++++++++++++++++++***/

	FinalVarx := RemoveZero(InvFFTfinal, N) //Removing zeros for final result

	fmt.Printf("\n Size of Part B FFT is %d \n", len(FinalVarx))
	print_iter = 0
	for print_iter < N {
		fmt.Printf("(%f, %f) ", FinalVarx[2*print_iter], FinalVarx[2*print_iter+1])
		print_iter++
	}
	fmt.Printf("\n")

	opencl.ReleaseAndClean()


	// //Testing results
	// testArr0 := make([]float64, N)
	// // testArr1 := make([]float64, N)
	// //for ii := 0; ii < size[0]; ii++ {
	// for ii := 0; ii < N; ii++ {
	// 	testArr0[ii] = float64(FinalVarx[ii] - X[ii])
		// testArr1[ii] = float64(FinalVarx[ii])
	// for ii := N / 2; ii > 0; ii /= 2 {
	// 	for jj := 0; jj < ii; jj++ {
	// 		aVal := testArr0[jj]
	// 		bVal := testArr0[jj+ii]
	// 		tsum := aVal + bVal
	// 		aEr := tsum - bVal
	// 		bEr := tsum - aVal
	// 		aErr := aEr - aVal
	// 		bErr := bEr - bVal
	// 		//testArr1[jj] += aErr + bErr
	// 		testArr0[jj] = tsum
	// 	}
	// }
	//golden := testArr0[0] - testArr1[0]

	//tol := float64(golden * 1e-5)
	//engine.Expect("Result", float64(results), float64(golden), tol)
	// if float64(results) == golden {
	// 	fmt.Println("Results match!")
	// } else {
	// 	fmt.Println("Results do not match! golden: ", golden, "; result: ", results)
	// }

	// fmt.Printf("Finished tests on sum\n")

	// fmt.Printf("freeing resources \n")
	// //	gpuBuffer.Free()
	// opencl.Recycle(gpuBuffer)

	// opencl.ReleaseAndClean()
	
	// //var k = []complex128 {complex(4,1), complex(10,2), complex(4,1), complex(10,2), complex(0,0)}
	// //var z = []complex128
	// var k = []float32 {4,1,10,2,4,1,10,2,0,0}
	// var z []float32


	// //z = AddZero( k, 5);
	// z = PreProcessB(len(k),4)
	// fmt.Printf( "\n Value is: %f + %fi", z[2],z[3]);
	// fmt.Printf( "\n Value is: %f + %fi ", z[6],z[7]);
	// fmt.Printf( "\n Value is: %f + %fi ", z[8],z[9]);
	// fmt.Printf("\n Length of array %d", len(k))

	
}