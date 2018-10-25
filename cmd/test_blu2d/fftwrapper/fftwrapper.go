package fftwrapper

import (
	"fmt"
	"math"
	"math/cmplx"
	"encoding/json"
	"os"
	"io/ioutil"
	"strconv"
	"flag"
	//"math/rand"
	"unsafe"
	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/cmd/test_blu2d/purefft"

)

var Flag_conj  = flag.Bool("conjugate", false, "Conjugate B in multiplication")

func findLength(tempLength int, fileName string) int {
	
	var j int
	m := make(map[string]int)
	strLength := strconv.Itoa(tempLength)

	jsonFile, _ := os.Open(fileName)
	defer jsonFile.Close()
	byteValue, _ := ioutil.ReadAll(jsonFile)
	json.Unmarshal([]byte(byteValue), &m)

	j = m[strLength]

	// fmt.Printf("The value of the required length is: %v", j)

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
	for iter := 0; iter < origLength;iter++ {
		tempVal = complex(float64(x[2*iter]),float64(x[2*iter + 1])) * cmplx.Exp(math.Pi * cmplx.Pow(complex(float64(iter),float64(0)),complex128(2)) * (-1/complex(float64(origLength),float64(0))) * (1i))
		processedA[2*iter] = float32(real(tempVal))
		processedA[2*iter+1] = float32(imag(tempVal))
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
	for iter := 0; iter < origLength;iter++ {
		tempVal = complex(float64(x[2*iter]),float64(x[2*iter + 1])) * cmplx.Exp(math.Pi * cmplx.Pow(complex(float64(iter),float64(0)),complex128(2)) * (1/complex(float64(origLength),float64(0))) * (1i))
		FilteredA[2*iter] = float32(real(tempVal))
		FilteredA[2*iter+1] = float32(imag(tempVal))
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
		tempVal = cmplx.Exp(math.Pi * cmplx.Pow(complex(float64(iter),0),complex(float64(2),0)) * (1/complex(float64(origLength),float64(0))) * (1i))
		InvTwid[2*iter] = float32(real(tempVal))
		InvTwid[2*iter + 1] = float32(imag(tempVal))

	}	
	return InvTwid
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
	return plication
}


//Function for 1D Forward FFT
func ForwFft1D(X []float32, ReqComponents int) []float32 {

	//Check if Blusteins is required
	Check1,Finder := blusteinCase(len(X))

	if Check1 == 0 && Finder == 1 {
		fmt.Printf("\n Bluesteins Implementation not necessary. Finding FFT directly...\n")
		FinalResults := FindClfft(X,len(X),"frw")
		return FinalResults

	}
	var BluN int

	BluN = 2*(len(X)+1) //Minimum condition for Blustein's M>=2*N
	
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
			// fmt.Printf("\n Bluestein is not required. Executing clFFT with length %v...", BluN)
			FinalN = BluN
		}
		fmt.Printf("\n Adjusting length and finding FFT using Blusteins Algorithm with Legnth = %d...\n", FinalN)
	}

	
	
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

	ForwFFTfinal := Complex_multi(ForwTwiddle,InvAxB, FinalN, ReqComponents)

	fmt.Printf("\n Finished calculating multiplication with forw Twiddle...\n ")
			
	/***++++++++++++++++++Bitwise multiplication for Forward FFT with Twiddle Factor ends here++++++++++++++++++++++++++++++++++++++++***/

	
	/***Remove padded zeros to get answer for the correct length*********/

	FinalDftX := RemoveZero(ForwFFTfinal, len(X)) //Removing zeros to extend lenth
	return FinalDftX
}

//Function for 1D IDFT

//Function for Transpose

//Function for 2D Forward FFT
func ForwFft2D(BiggerX [][]float32, ReqComponents int) [][]float32 {
	
	ToBeTransposed := make([][]float32, len(BiggerX))
	for i := 0; i<len(BiggerX); i++ {
		ToBeTransposed[i] = ForwFft1D(BiggerX[i],ReqComponents)
	}

	//Is tranpose supposed to happen on GPU??


	Final2dDftX := make([][]float32, len(BiggerX))

	return Final2dDftX
}