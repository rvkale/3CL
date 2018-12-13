package wrap1dreal

import (
	"flag"
	"fmt"
	"math"
	"math/cmplx"

	//"math/rand"

	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/opencl/cl"
)

var Flag_conj = flag.Bool("conjugate", false, "Conjugate B in multiplication")

/**++++++++++++++++++++++++++++++++++++++++++++++Function for finding blusteins length ends here++++++++++++++++++++++++++++****/

//Function to add zeros
func AddZero(x []float32, zeroLength int) []float32 {
	if len(x) >= 2*zeroLength {
		return x
	}
	r := make([]float32, 2*zeroLength)
	copy(r, x)
	return r
}

//Functions to remove zeros
func RemoveZero(x []float32, origLength int) []float32 {
	if len(x) <= 2*origLength {
		return x
	}
	r := make([]float32, 2*origLength)
	copy(r, x)
	return r
}

//Function to find A part for the forward FFT
func PreProcessA(x []float32, origLength int) []float32 {
	processedA := make([]float32, len(x))
	var tempVal complex128
	//fmt.Printf("\n Length of array %d", len(x))
	for iter := 0; iter < origLength; iter++ {
		tempVal = complex(float64(x[2*iter]), float64(x[2*iter+1])) * cmplx.Exp(math.Pi*cmplx.Pow(complex(float64(iter), float64(0)), complex128(2))*(-1/complex(float64(origLength), float64(0)))*(1i))
		processedA[2*iter] = float32(real(tempVal))
		processedA[2*iter+1] = float32(imag(tempVal))
	}
	return processedA
}

//Function to find B part for the forward FFT
func PreProcessB(newLength int, origLength int) []float32 {
	processedB := make([]float32, 2*newLength)
	var tempVal complex128
	for iter := 0; iter < int(newLength); iter++ {
		//fmt.Printf("\n Executing")
		if iter < origLength {
			tempVal = cmplx.Exp(math.Pi * cmplx.Pow(complex(float64(iter), float64(0)), complex128(2)) * (1 / complex(float64(origLength), float64(0))) * (1i))
		} else if iter < newLength {
			tempVal = cmplx.Exp(math.Pi * cmplx.Pow(complex(float64(newLength-iter), float64(0)), complex128(2)) * (1 / complex(float64(origLength), float64(0))) * (1i))
		}
		processedB[2*iter] = float32(real(tempVal))
		processedB[2*iter+1] = float32(imag(tempVal))

	}
	return processedB
}

//Function to find twiddle factor to multiply after A*B for forward FFT
func ForwFftTwid(newLength int, origLength int) []float32 {
	ForwTwid := make([]float32, 2*newLength)
	var tempVal complex128
	for iter := 0; iter < newLength; iter++ {
		tempVal = cmplx.Exp(math.Pi * cmplx.Pow(complex(float64(iter), 0), complex(float64(2), 0)) * (1 / complex(float64(origLength), float64(0))) * (-1i))
		ForwTwid[2*iter] = float32(real(tempVal))
		ForwTwid[2*iter+1] = float32(imag(tempVal))

	}
	return ForwTwid
}

//Function to find A part for the Inverse FFT
func InvProcessA(x []float32, origLength int) []float32 {
	FilteredA := make([]float32, len(x))
	var tempVal complex128
	//fmt.Printf("\n Length of array %d", len(x))
	for iter := 0; iter < origLength; iter++ {
		tempVal = complex(float64(x[2*iter]), float64(x[2*iter+1])) * cmplx.Exp(math.Pi*cmplx.Pow(complex(float64(iter), float64(0)), complex128(2))*(1/complex(float64(origLength), float64(0)))*(1i))
		FilteredA[2*iter] = float32(real(tempVal))
		FilteredA[2*iter+1] = float32(imag(tempVal))
	}
	return FilteredA
}

//Function to find B part for the Inverse FFT
func InvProcessB(newLength int, origLength int) []float32 {
	FilteredB := make([]float32, 2*newLength)
	var tempVal complex128
	for iter := 0; iter < int(newLength); iter++ {
		//fmt.Printf("\n Executing")
		if iter < origLength {
			tempVal = cmplx.Exp(math.Pi * cmplx.Pow(complex(float64(iter), float64(0)), complex128(2)) * (-1 / complex(float64(origLength), float64(0))) * (1i))
		} else if iter < newLength {
			tempVal = cmplx.Exp(math.Pi * cmplx.Pow(complex(float64(newLength-iter), float64(0)), complex128(2)) * (-1 / complex(float64(origLength), float64(0))) * (1i))
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
	for iter := 0; iter < newLength; iter++ {
		//tempVal = cmplx.Exp(math.Pi * cmplx.Pow(complex(float64(iter),0),complex(float64(2),0)) * (1i))
		tempVal = cmplx.Exp(math.Pi * cmplx.Pow(complex(float64(iter), 0), complex(float64(2), 0)) * (1 / complex(float64(origLength), float64(0))) * (1i))
		InvTwid[2*iter] = float32(real(tempVal))
		InvTwid[2*iter+1] = float32(imag(tempVal))

	}
	return InvTwid
}

//ComplexMulti Multiplication of complex numbers
func ComplexMulti(outArray, cpuArray0, cpuArray1 *data.Slice, dataSize int, NComponents int) {
	queue := opencl.ClCmdQueue
	//	device, context, queue := opencl.ClDevice, opencl.ClCtx, opencl.ClCmdQueue
	kernels := opencl.KernList

	kernelObj := kernels["cmplx_mul"]
	totalArgs, err := kernelObj.NumArgs()
	if err != nil {
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

	fmt.Println("Done. Transferring input data from CPU to GPU...")
	size := [3]int{dataSize * 2, 1, 1}
	//cpuArray0 := data.SliceFromArray(inputs0, size)
	//cpuArray1 := data.SliceFromArray(inputs1, size)
	gpuBuffer0 := opencl.Buffer(NComponents, size)
	gpuBuffer1 := opencl.Buffer(NComponents, size)
	outBuffer := opencl.Buffer(NComponents, size)
	//outArray := data.NewSlice(NComponents, [3]int{dataSize, 1, 1})

	data.Copy(gpuBuffer0, cpuArray0)
	data.Copy(gpuBuffer1, cpuArray1)

	fmt.Println("Waiting for data transfer to complete...")
	queue.Finish()
	fmt.Println("Input data transfer completed.")

	fmt.Println("Executing kernel...")
	if *Flag_conj {
		opencl.ComplexArrayMul(outBuffer, gpuBuffer0, gpuBuffer1, 1, dataSize, 0)
	} else {
		opencl.ComplexArrayMul(outBuffer, gpuBuffer0, gpuBuffer1, 0, dataSize, 0)
	}
	fmt.Println("Waiting for kernel to finish execution...")
	queue.Finish()
	fmt.Println("Execution finished.")

	fmt.Println("Retrieving results...")
	data.Copy(outArray, outBuffer)
	queue.Finish()
	fmt.Println("Done.")
	// results := outArray.Host()
	// plication := make([]float32, dataSize)
	// for i := 0; i < NComponents; i++ {
	// 	for j := 0; j < len(inputs1[i]); j++ {
	// 		plication[j] = results[i][j]
	// 	}
	// }
	fmt.Printf("Finished tests on cmplx_run\n")

	fmt.Printf("freeing resources \n")
}

//Function for 1D Forward FFT
func ForwFft1D(X []float32, ReqComponents int) []float32 {

	ZeroForwPadX := AddZero(X, FinalN) //Padding zeros to extend lenth

	/********************************************************Forward FFT Part A begins***************************************************/

	ForwFftA := PreProcessA(ZeroForwPadX, len(X)) //Part A for Forward FFT

	fmt.Printf("\n Finished adding zeros \n")

	fmt.Printf("\n Calculating FFT of part A... \n")

	PartAForwFFT := FindClfft(ForwFftA, FinalN, "frw")

	fmt.Printf("\n Finished calculating FFT of part A...\n")

	/***+++++++++++++++++++++++++++++++++++++++++++++++++++++++Forward FFT Part A ends++++++++++++++++++++++++++++++++++++++++++++++*****/

	/**********************************************************Forward FFT Part B begins*************************************************/

	ForwFftB := PreProcessB(FinalN, len(X))

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

	ForwTwiddle := ForwFftTwid(FinalN, len(X))

	fmt.Printf("\n Calculating multiplication with forw Twiddle...\n")

	ForwFFTfinal := Complex_multi(ForwTwiddle, InvAxB, FinalN, ReqComponents)

	fmt.Printf("\n Finished calculating multiplication with forw Twiddle...\n ")

	/***++++++++++++++++++Bitwise multiplication for Forward FFT with Twiddle Factor ends here++++++++++++++++++++++++++++++++++++++++***/

	/***Remove padded zeros to get answer for the correct length*********/

	FinalDftX := RemoveZero(ForwFFTfinal, len(X)) //Removing zeros to extend lenth
	return FinalDftX
}
