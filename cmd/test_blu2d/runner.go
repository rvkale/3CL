package main

import (
	"fmt"
	//"math"
	//"math/cmplx"
	//"encoding/json"
	//"os"
	//"io/ioutil"
	//"strconv"
	"flag"
	"math/rand"
	//"unsafe"
	//"github.com/mumax/3cl/data"
	//"github.com/mumax/3cl/engine"
	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/cmd/test_blu2d/fftwrapper"
	"github.com/mumax/3cl/cmd/test_blu2d/purefft"

	//"github.com/mumax/3cl/opencl/cl"
)


var (
	Flag_gpu = flag.Int("gpu", 0, "Specify GPU")
	Flag_size  = flag.Int("length", 359, "length of data to test")
	Flag_print = flag.Bool("print", false, "Print out result")
	Flag_comp  = flag.Int("components", 1, "Number of components to test")
	//Flag_conj  = flag.Bool("conjugate", false, "Conjugate B in multiplication")
)

func main() {
	
	flag.Parse()
	var Desci int //Descision variable
	N := int(*Flag_size)
	ReqComponents := int(*Flag_comp)
	opencl.Init(*Flag_gpu) 
	rand.Seed(178)
	X := make([]float32, 2*N)

	/* Print input array */

	print_iter := 0
	for print_iter < N {
		x := rand.Float32()
		y := rand.Float32()
		// x := float32(1)
		// y := float32(1)
		X[2*print_iter] = x
		X[2*print_iter+1] = y
		fmt.Printf("(%f, %f) ", x, y)
		print_iter++
	}

	
	
	// fmt.Printf("Enter the length as 67 for now: ")
	// _, err := fmt.Scanf("%d", &N)
	// if err!= nil {panic("Serious Error!")}


	
	/* Prepare OpenCL memory objects and place data inside them for . */
	//Initialize GPU with a flag to pick the desired gpu
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
	

	fmt.Printf("\n Checking FFT......\n")
	print_iter = 0
	for print_iter < N {
		fmt.Printf("(%f, %f) ", FinalDftX[2*print_iter], FinalDftX[2*print_iter+1])
		print_iter++
	}
	fmt.Printf("\n")

	




	/********************************************************************Inverse DFT**********************************************************************/

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

	IntermVarx := RemoveZero(InvFFTfinal, N) //Removing zeros for final result

	FinalVarx := make([]float32, len(IntermVarx))

	fmt.Printf("\n Size of Part B FFT is %d \n", len(FinalVarx))
	print_iter = 0
	for print_iter < N {
		FinalVarx[2*print_iter] = IntermVarx[2*print_iter]/float32(N)
		FinalVarx[2*print_iter+1] = IntermVarx[2*print_iter+1]/float32(N)
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