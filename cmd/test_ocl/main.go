package main

import (
	//"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"

	//"github.com/mumax/3cl/cmd/test_blu2d/purefft"

	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/opencl/cl"
	//"github.com/mumax/3cl/cmd/test_blu_2d"
)

var (
	Flag_gpu   = flag.Int("gpu", 0, "Specify GPU")
	Flag_size  = flag.Int("length", 30, "length of data to test")
	Flag_print = flag.Bool("print", false, "Print out result")
	Flag_comp  = flag.Int("components", 1, "Number of components to test")
	//Flag_conj  = flag.Bool("conjugate", false, "Conjugate B in multiplication")
)

//////// Radices and maximum length supported by clFFT ////////
//var supported_radices = []int{17, 13, 11, 8, 7, 5, 4, 3, 2}
var supported_radices = []int{13, 11, 8, 7, 5, 4, 3, 2}

const maxLen int = 128000000

//HermitianWarning Issue a warning if complex conjugates of hermitian are not closely matching

//PrintArray Prints the input array for debugging
func PrintArray(InpArr *data.Slice, ArrLength int) {
	// queue := opencl.ClCmdQueue
	outArray := data.NewSlice(1, [3]int{ArrLength, 1, 1})
	data.Copy(outArray, InpArr)
	fmt.Printf("\n Printing the requested array \n")
	// queue.Finish()
	// queue.Release()
	//fmt.Println("\n Output data transfer completed. Printing ")
	result2 := outArray.Host()
	//results := make([][]float32, 1)
	for k := 0; k < 1; k++ {
		// results[i] = make([]float32, 2*5*2)
		for j := 0; j < int(ArrLength); j++ {
			// fmt.Printf(" ( %f , %f ) ", result2[k][2*j], result2[k][2*j+1])
			fmt.Printf(" ( %f ) ", result2[k][j])
		}
	}
}

func main() {

	var fft_length int
	fft_length = 68

	flag.Parse()
	//var Desci int //Descision variable
	N := int(*Flag_size)
	opencl.Init(*Flag_gpu)
	//rand.Seed(time.Now().Unix())
	rand.Seed(24)
	//X := make([]float32, 2*N)
	NComponents := int(*Flag_comp)
	if N < 4 {
		fmt.Println("argument to -fft must be 4 or greater!")
		os.Exit(-1)
	}
	if (NComponents < 1) || (NComponents > 3) {
		fmt.Println("argument to -components must be 1, 2 or 3!")
		os.Exit(-1)
	}

	//opencl.Init(*engine.Flag_gpu)

	/* Print input array */

	//plan1d := FftPlanValue{false, true, true, false, N, 1, 1, 0}

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

	queue := opencl.ClCmdQueue

	fmt.Printf("\n Printing device: %v", d)

	// fmt.Printf("\n Executing Forward 2D FFT. Printing input array \n")
	// plan2d := FftPlan2DValue{false, true, true, true, false, int(*Flag_size), 2, 1, int(*Flag_size), 2}
	inputs2d := make([][]float32, NComponents)

	var size2d [3]int

	// size2d = [3]int{2 * fft_length, 1, 1}

	// size2d = [3]int{13, 1, 1}
	// for i := 0; i < NComponents; i++ {
	// 	inputs2d[i] = make([]float32, size2d[0])
	// 	for j := 0; j < 1; j++ {
	// 		for k := 0; k < 13; k++ {
	// 			inputs2d[i][j*13+k] = float32(j*13 + k) //* float32(0.1) //float32(0.1)
	// 			fmt.Printf("( %f ) ", inputs2d[i][j*13+k])
	// 		}
	// 		fmt.Printf("\n")
	// 	}
	// }

	// size2d = [3]int{136, 1, 1}
	// inputs2d[0] = []float32{0.000000, 0.000000, 1.000000, 0.000000, 2.000000, 0.000000,
	// 	3.000000, 0.000000, 4.000000, 0.000000, 5.000000, 0.000000, 6.000000, 0.000000,
	// 	7.000000, 0.000000, 8.000000, 0.000000,
	// 	9.000000, 0.000000, 10.000000, 0.000000, 11.000000, 0.000000,
	// 	12.000000, 0.000000, 13.000000, 0.000000, 14.000000, 0.000000,
	// 	15.000000, 0.000000, 16.000000, 0.000000,
	// 	17.000000, 0.000000, 18.000000, 0.000000, 19.000000, 0.000000,
	// 	20.000000, 0.000000, 21.000000, 0.000000, 22.000000, 0.000000, 23.000000, 0.000000,
	// 	24.000000, 0.000000, 25.000000, 0.000000,
	// 	26.000000, 0.000000, 27.000000, 0.000000, 28.000000, 0.000000,
	// 	29.000000, 0.000000, 30.000000, 0.000000, 31.000000, 0.000000,
	// 	32.000000, 0.000000, 33.000000, 0.000000,
	// 	34.000000, 0.000000, 35.000000, 0.000000, 36.000000, 0.000000,
	// 	37.000000, 0.000000, 38.000000, 0.000000, 39.000000, 0.000000, 40.000000, 0.000000,
	// 	41.000000, 0.000000, 42.000000, 0.000000,
	// 	43.000000, 0.000000, 44.000000, 0.000000, 45.000000, 0.000000,
	// 	46.000000, 0.000000, 47.000000, 0.000000, 48.000000, 0.000000,
	// 	49.000000, 0.000000, 50.000000, 0.000000,
	// 	51.000000, 0.000000, 52.000000, 0.000000, 53.000000, 0.000000,
	// 	54.000000, 0.000000, 55.000000, 0.000000, 56.000000, 0.000000, 57.000000, 0.000000,
	// 	58.000000, 0.000000, 59.000000, 0.000000,
	// 	60.000000, 0.000000, 61.000000, 0.000000, 62.000000, 0.000000,
	// 	63.000000, 0.000000, 64.000000, 0.000000, 65.000000, 0.000000,
	// 	66.000000, 0.000000, 67.000000, 0.000000}

	size2d = [3]int{68, 1, 1}
	inputs2d[0] = []float32{0.000000, 1.000000, 2.000000,
		3.000000, 4.000000, 5.000000, 6.000000,
		7.000000, 8.000000,
		9.000000, 10.000000, 11.000000,
		12.000000, 13.000000, 14.000000,
		15.000000, 16.000000,
		17.000000, 18.000000, 19.000000,
		20.000000, 21.000000, 22.000000, 23.000000,
		24.000000, 25.000000,
		26.000000, 27.000000, 28.000000,
		29.000000, 30.000000, 31.000000,
		32.000000, 33.000000,
		34.000000, 35.000000, 36.000000,
		37.000000, 38.000000, 39.000000, 40.000000,
		41.000000, 42.000000,
		43.000000, 44.000000, 45.000000,
		46.000000, 47.000000, 48.000000,
		49.000000, 50.000000,
		51.000000, 52.000000, 53.000000,
		54.000000, 55.000000, 56.000000, 57.000000,
		58.000000, 59.000000,
		60.000000, 61.000000, 62.000000,
		63.000000, 64.000000, 65.000000,
		66.000000, 67.000000}

	// size2d = [3]int{18, 1, 1}
	// inputs2d[0] = []float32{136.000000, 0.000000, -8.500000, 45.4709838, -8.500000, 21.94102921,
	// 	-8.500000, 13.72797136, -8.500000, 9.32405583, -8.500000, 6.41890204, -8.500000, 4.23249709,
	// 	-8.500000, 2.41845851, -8.500000, 0.78764099} //,
	// -8.500000, -0.78764099, -8.500000, -2.41845851, -8.500000, -4.23249709,
	// -8.500000, -6.41890204, -8.500000, -9.32405583, -8.500000, -13.72797136,
	// -8.500000, -21.94102921, -8.500000, -45.4709838}

	// size2d = [3]int{34, 1, 1}
	// for i := 0; i < NComponents; i++ {
	// 	inputs2d[i] = make([]float32, size2d[0])
	// 	for z := 0; z < 1; z++ {
	// 		for j := 0; j < 17; j++ {
	// 			for k := 0; k < 2; k++ {
	// 				inputs2d[i][z*34+j*2+k] = float32(z*34+j*2+k) * float32(0.01) //float32(0.1)
	// 				fmt.Printf("( %f ) ", inputs2d[i][z*34+j*2+k])
	// 			}
	// 			fmt.Printf("\n")
	// 		}
	// 	}
	// }

	// if !plan2d.IsForw && plan2d.IsRealHerm {
	// fmt.Printf("\n Printing default value of hermitian matrix of 17*2 of FFT of 0,1,2,...,16;17,18,...,33")
	// size2d = [3]int{2 * int(1+plan2d.RowDim/2) * plan2d.ColDim, 1, 1}
	// size2d = [3]int{2 * 34, 1, 1}
	// for i := 0; i < NComponents; i++ {
	// 	inputs2d[i] = make([]float32, size2d[0])
	// 	for j := 0; j < plan2d.ColDim; j++ {
	// 		for k := 0; k < int(1+plan2d.RowDim/2); k++ {
	// 			inputs2d[i][2*(j*plan2d.RowDim+k)] = float32(j*int(1+plan2d.RowDim/2) + k)   //float32(0.1)
	// 			inputs2d[i][2*(j*plan2d.RowDim+k)+1] = float32(j*int(1+plan2d.RowDim/2) + k) //float32(0.1)
	// 			//fmt.Printf(" (%f, %f) ", inputs2d[i][2*(j*int(1+plan2d.RowDim/2)+k)], inputs2d[i][2*(j*int(1+plan2d.RowDim/2)+k)+1])
	// 		}
	// 		//fmt.Printf("\n")
	// 	}
	// }

	// size2d = [3]int{36, 1, 1}
	// inputs2d[0] = []float32{56.099998, 0.000001, -1.700004, 9.094196, -1.700002, 4.388192, -1.699986, 2.745589,
	// 	-1.699995, 1.864812, -1.700000, 1.283777, -1.699999, 0.846498, -1.700004, 0.483691,
	// 	-1.700001, 0.157525,
	// 	-28.900002, 0.000000, 0.000001, -0.000000, 0.000001, 0.000006, -0.000006, 0.000001,
	// 	-0.000002, 0.000000, -0.000000, 0.000002, -0.000001, 0.000001, 0.000002, 0.000000,
	// 	0.000000, 0.000000}

	// inputs2d[0] = []float32{56.099998, 0.000001, -1.700004, 9.094196, -1.700002, 4.388192, -1.699986, 2.745589,
	// 	-1.699995, 1.864812, -1.700000, 1.283777, -1.699999, 0.846498, -1.700004, 0.483691,
	// 	-1.700001, 0.157525, -1.700001, -0.157525, -1.700004, -0.483691, -1.699999, -0.846498,
	// 	-1.700000, -1.283777, -1.699995, -1.864812, -1.699986, -2.745589, -1.700002, -4.388192,
	// 	-1.700004, -9.094196,
	// 	-28.900002, 0.000000, 0.000001, -0.000000, 0.000001, 0.000006, -0.000006, 0.000001,
	// 	-0.000002, 0.000000, -0.000000, 0.000002, -0.000001, 0.000001, 0.000002, 0.000000,
	// 	0.000000, 0.000000, 0.000000, 0.000000, 0.000002, 0.000000, -0.000001, -0.000001,
	// 	-0.000000, -0.000002, -0.000002, 0.000000, -0.000006, -0.000001, 0.000001, -0.000006,
	// 	0.000001, 0.000000}

	// panic("\n this is the input \n")

	fmt.Println("\n Done. Transferring input data from CPU to GPU...")
	cpuArray2d := data.SliceFromArray(inputs2d, size2d)
	gpu2dBuffer := opencl.Buffer(NComponents, size2d)
	gpu2destBuf := opencl.Buffer(NComponents, [3]int{136, 1, 1})
	gpu3newBuf := opencl.Buffer(NComponents, size2d)
	// gpu3newBuf := opencl.Buffer(NComponents, [3]int{136, 1, 1})
	// //outBuffer := opencl.Buffer(NComponents, [3]int{2 * N, 1, 1})

	data.Copy(gpu2dBuffer, cpuArray2d)

	fmt.Println("Waiting for data transfer to complete...")
	queue.Finish()
	fmt.Println("Input data transfer completed.")

	// PrintArray(cpuArray2d, 17)

	// //opencl.Recycle(gpu2dBuffer)
	tmpptr := gpu2dBuffer.DevPtr(0)
	srcmemobj := (*cl.MemObject)(tmpptr)

	dstpt := gpu2destBuf.DevPtr(0)
	dstmemobj := (*cl.MemObject)(dstpt)

	thirdest := gpu3newBuf.DevPtr(0)
	thirdestobj := (*cl.MemObject)(thirdest)

	effort, _ := cl.CreateDefaultOclFFTPlan()
	effort.SetDevice(d)
	effort.SetContext(opencl.ClCtx)
	// effort.SetQueue(queue)
	effort.SetProgram()
	// fmt.Printf("\n \n %v ", effort.GetDevice())
	effort.SetQueue(queue)
	effort.SetLayout(cl.CLFFTLayoutReal)
	// effort.SetLayout(cl.CLFFTLayoutHermitianInterleaved)
	// effort.SetLayout(cl.CLFFTLayoutComplexInterleaved)
	effort.SetDirection(cl.ClFFTDirectionForward)
	// effort.SetDirection(cl.ClFFTDirectionBackward)
	effort.SetPrecision(cl.CLFFTPrecisionSingle)

	// effort.SetLengths([3]int{17, 2, 2})
	effort.SetLengths([3]int{2, 2, 17})
	effort.SetInStride([3]int{1, 0, 0})
	effort.SetOutStride([3]int{1, 0, 0})

	effort.SetSource(srcmemobj)
	effort.SetDest(dstmemobj)

	fmt.Printf("\n Printing array \n")
	fmt.Printf("%v \n", effort.GetLengths())

	effort.Bake()

	err := effort.ExecTransform(dstmemobj, srcmemobj)

	// effort.Hermit2Full(dstmemobj, srcmemobj, 17, 9)

	if err != nil {
		fmt.Printf("\n This is not working as intended %v ", err)
	}

	PrintArray(gpu2destBuf, 136)

	// effort.Destroy()

	fmt.Printf("\n \n \n \n \n \n \n Executed the forward FFT Correctly \n \n \n \n \n \n \n ")

	// effort.SetLayout(cl.CLFFTLayoutReal)
	effort.SetLayout(cl.CLFFTLayoutHermitianInterleaved)
	// effort.SetLayout(cl.CLFFTLayoutComplexInterleaved)
	// effort.SetDirection(cl.ClFFTDirectionForward)
	effort.SetDirection(cl.ClFFTDirectionBackward)
	effort.SetPrecision(cl.CLFFTPrecisionSingle)
	effort.SetInStride([3]int{1, 0, 0})
	effort.SetOutStride([3]int{1, 0, 0})
	effort.SetSource(dstmemobj)
	effort.SetDest(thirdestobj)
	effort.Bake()
	// effort.ExecTransform(dstmemobj, srcmemobj)
	err = effort.ExecTransform(thirdestobj, dstmemobj)

	// effort.Hermit2Full(dstmemobj, srcmemobj, 17, 9)

	if err != nil {
		fmt.Printf("\n This is not working as intended %v %d ", err, fft_length)
	}
	PrintArray(gpu3newBuf, 68)

	effort.Destroy()

	// Parse2D(gpu2dBuffer, plan2d)

	// fmt.Printf("\n Executing 3D FFT \n")

	// if plan2d.IsForw && plan2d.IsRealHerm {
	// 	size2d = [3]int{plan2d.RowDim * plan2d.ColDim, 1, 1}
	// 	for i := 0; i < NComponents; i++ {
	// 		inputs2d[i] = make([]float32, size2d[0])
	// 		for j := 0; j < plan2d.ColDim; j++ {
	// 			for k := 0; k < plan2d.RowDim; k++ {
	// 				inputs2d[i][j*plan2d.RowDim+k] = float32(j*plan2d.RowDim+k) * float32(0.1) //float32(0.1)
	// 				fmt.Printf("( %f ) ", inputs2d[i][j*plan2d.RowDim+k])
	// 			}
	// 			fmt.Printf("\n")
	// 		}
	// 	}
	// }

	// if plan2d.IsForw && !plan2d.IsRealHerm {
	// 	fmt.Printf("\n Printing default value of hermitian matrix of 17*2 of FFT of 0,1,2,...,16;17,18,...,33")
	// 	size2d = [3]int{2 * plan2d.RowDim * plan2d.ColDim, 1, 1}
	// 	for i := 0; i < NComponents; i++ {
	// 		inputs2d[i] = make([]float32, size2d[0])
	// 		for j := 0; j < plan2d.ColDim; j++ {
	// 			for k := 0; k < plan2d.RowDim; k++ {
	// 				inputs2d[i][2*(j*plan2d.RowDim+k)] = float32(j*plan2d.RowDim+k) * float32(0.1)
	// 				inputs2d[i][2*(j*plan2d.RowDim+k)+1] = float32(j*plan2d.RowDim+k) * float32(0.1)
	// 				fmt.Printf(" (%f, %f) ", inputs2d[i][2*(j*plan2d.RowDim+k)], inputs2d[i][2*(j*plan2d.RowDim+k)+1])
	// 			}
	// 			//fmt.Printf("\n")
	// 		}
	// 	}
	// }

	// if !plan2d.IsForw && plan2d.IsRealHerm {
	// 	fmt.Printf("\n Printing default value of hermitian matrix of 17*2 of FFT of 0,1,2,...,16;17,18,...,33")
	// 	size2d = [3]int{2 * int(1+plan2d.RowDim/2) * plan2d.ColDim, 1, 1}
	// 	// for i := 0; i < NComponents; i++ {
	// 	// 	inputs2d[i] = make([]float32, size2d[0])
	// 	// 	for j := 0; j < plan2d.ColDim; j++ {
	// 	// 		for k := 0; k < int(1+plan2d.RowDim/2); k++ {
	// 	// 			inputs2d[i][2*(j*plan2d.RowDim+k)] = float32(j*int(1+plan2d.RowDim/2) + k)   //float32(0.1)
	// 	// 			inputs2d[i][2*(j*plan2d.RowDim+k)+1] = float32(j*int(1+plan2d.RowDim/2) + k) //float32(0.1)
	// 	// 			//fmt.Printf(" (%f, %f) ", inputs2d[i][2*(j*int(1+plan2d.RowDim/2)+k)], inputs2d[i][2*(j*int(1+plan2d.RowDim/2)+k)+1])
	// 	// 		}
	// 	// 		//fmt.Printf("\n")
	// 	// 	}
	// 	// }
	// 	inputs2d[0] = []float32{56.099998, 0.000001, -1.700004, 9.094196, -1.700002, 4.388192, -1.699986, 2.745589,
	// 		-1.699995, 1.864812, -1.700000, 1.283777, -1.699999, 0.846498, -1.700004, 0.483691,
	// 		-1.700001, 0.157525,
	// 		-28.900002, 0.000000, 0.000001, -0.000000, 0.000001, 0.000006, -0.000006, 0.000001,
	// 		-0.000002, 0.000000, -0.000000, 0.000002, -0.000001, 0.000001, 0.000002, 0.000000,
	// 		0.000000, 0.000000}
	// }

	// if !plan2d.IsRealHerm && !plan2d.IsForw {
	// 	size2d = [3]int{2 * plan2d.RowDim * plan2d.ColDim, 1, 1}
	// 	inputs2d[0] = []float32{56.099998, 0.000001, -1.700004, 9.094196, -1.700002, 4.388192, -1.699986, 2.745589,
	// 		-1.699995, 1.864812, -1.700000, 1.283777, -1.699999, 0.846498, -1.700004, 0.483691,
	// 		-1.700001, 0.157525, -1.700001, -0.157525, -1.700004, -0.483691, -1.699999, -0.846498,
	// 		-1.700000, -1.283777, -1.699995, -1.864812, -1.699986, -2.745589, -1.700002, -4.388192,
	// 		-1.700004, -9.094196,
	// 		-28.900002, 0.000000, 0.000001, -0.000000, 0.000001, 0.000006, -0.000006, 0.000001,
	// 		-0.000002, 0.000000, -0.000000, 0.000002, -0.000001, 0.000001, 0.000002, 0.000000,
	// 		0.000000, 0.000000, 0.000000, 0.000000, 0.000002, 0.000000, -0.000001, -0.000001,
	// 		-0.000000, -0.000002, -0.000002, 0.000000, -0.000006, -0.000001, 0.000001, -0.000006,
	// 		0.000001, 0.000000}
	// 	// for i := 0; i < NComponents; i++ {
	// 	// 	inputs2d[i] = make([]float32, size2d[0])
	// 	// 	for j := 0; j < plan2d.ColDim; j++ {
	// 	// 		for k := 0; k < plan2d.RowDim; k++ {
	// 	// 			inputs2d[i][2*(j*plan2d.RowDim+k)] = float32(j*plan2d.RowDim + k)   //float32(0.1)
	// 	// 			inputs2d[i][2*(j*plan2d.RowDim+k)+1] = float32(j*plan2d.RowDim + k) //float32(0.1)
	// 	// 			fmt.Printf(" (%f, %f) ", inputs2d[i][2*(j*plan2d.RowDim+k)], inputs2d[i][2*(j*plan2d.RowDim+k)+1])
	// 	// 		}
	// 	// 		fmt.Printf("\n")
	// 	// 	}
	// 	// }
	// }

	// fmt.Println("\n Done. Transferring input data from CPU to GPU...")
	// cpuArray2d := data.SliceFromArray(inputs2d, size2d)
	// gpu2dBuffer := opencl.Buffer(NComponents, size2d)
	// //outBuffer := opencl.Buffer(NComponents, [3]int{2 * N, 1, 1})

	// data.Copy(gpu2dBuffer, cpuArray2d)

	// fmt.Println("Waiting for data transfer to complete...")
	// queue.Finish()
	// fmt.Println("Input data transfer completed.")

	// //opencl.Recycle(gpu2dBuffer)

	// Parse2D(gpu2dBuffer, plan2d)

	// fmt.Printf("\n Executing 3D FFT \n")

	// plan3d := FftPlan3DValue{false, true, true, true, false, true, int(*Flag_size), 2, 2, int(*Flag_size), 2, 2}
	// inputs3d := make([][]float32, NComponents)
	// var size3d [3]int

	// if plan3d.IsForw && plan3d.IsRealHerm {
	// 	size3d = [3]int{plan3d.RowDim * plan3d.ColDim * plan3d.DepthDim, 1, 1}
	// 	for i := 0; i < NComponents; i++ {
	// 		inputs3d[i] = make([]float32, size3d[0])
	// 		for z := 0; z < plan3d.DepthDim; z++ {
	// 			for j := 0; j < plan3d.ColDim; j++ {
	// 				for k := 0; k < plan3d.RowDim; k++ {
	// 					inputs3d[i][z*plan3d.RowDim*plan3d.ColDim+j*plan3d.RowDim+k] = float32(z*plan3d.RowDim*plan3d.ColDim+j*plan3d.RowDim+k) * float32(0.01) //float32(0.1)
	// 					fmt.Printf("( %f ) ", inputs3d[i][z*plan3d.RowDim*plan3d.ColDim+j*plan3d.RowDim+k])
	// 				}
	// 				fmt.Printf("\n")
	// 			}
	// 		}
	// 	}
	// }

	// if plan3d.IsForw && !plan3d.IsRealHerm {
	// 	size3d = [3]int{2 * plan3d.RowDim * plan3d.ColDim * plan3d.DepthDim, 1, 1}
	// 	for i := 0; i < NComponents; i++ {
	// 		inputs3d[i] = make([]float32, size3d[0])
	// 		for z := 0; z < plan3d.DepthDim; z++ {
	// 			for j := 0; j < plan3d.ColDim; j++ {
	// 				for k := 0; k < plan3d.RowDim; k++ {
	// 					inputs3d[i][2*(z*plan3d.RowDim*plan3d.ColDim+j*plan3d.RowDim+k)] = float32(z*plan3d.RowDim*plan3d.ColDim+j*plan3d.RowDim+k) * float32(0.01) //float32(0.1)
	// 					inputs3d[i][2*(z*plan3d.RowDim*plan3d.ColDim+j*plan3d.RowDim+k)+1] = float32(z*plan3d.RowDim*plan3d.ColDim+j*plan3d.RowDim+k) * float32(0.01)
	// 					fmt.Printf("( %f , %f ) ", inputs3d[i][2*(z*plan3d.RowDim*plan3d.ColDim+j*plan3d.RowDim+k)], inputs3d[i][2*(z*plan3d.RowDim*plan3d.ColDim+j*plan3d.RowDim+k)+1])
	// 				}
	// 				fmt.Printf("\n")
	// 			}
	// 		}
	// 	}
	// }

	// if !plan3d.IsForw && plan3d.IsRealHerm {
	// 	fmt.Printf("\n Printing default value of hermitian matrix of 17*2 of FFT of 0,1,2,...,16;17,18,...,33")
	// 	size3d = [3]int{2 * int(1+plan3d.RowDim/2) * plan3d.ColDim * plan3d.DepthDim, 1, 1}
	// 	// for i := 0; i < NComponents; i++ {
	// 	// 	inputs2d[i] = make([]float32, size2d[0])
	// 	// 	for j := 0; j < plan2d.ColDim; j++ {
	// 	// 		for k := 0; k < int(1+plan2d.RowDim/2); k++ {
	// 	// 			inputs2d[i][2*(j*plan2d.RowDim+k)] = float32(j*int(1+plan2d.RowDim/2) + k)   //float32(0.1)
	// 	// 			inputs2d[i][2*(j*plan2d.RowDim+k)+1] = float32(j*int(1+plan2d.RowDim/2) + k) //float32(0.1)
	// 	// 			//fmt.Printf(" (%f, %f) ", inputs2d[i][2*(j*int(1+plan2d.RowDim/2)+k)], inputs2d[i][2*(j*int(1+plan2d.RowDim/2)+k)+1])
	// 	// 		}
	// 	// 		//fmt.Printf("\n")
	// 	// 	}
	// 	// }
	// 	inputs3d[0] = []float32{22.7800, 0.0000, -0.3400, 1.8188, -0.3400, 0.8776, -0.3400, 0.5491, -0.3400, 0.3730, -0.3400, 0.2568, -0.3400, 0.1693, -0.3400, 0.0967, -0.3400, 0.0315,
	// 		-5.7800, 0.0000, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000, -0.0000, -0.0000, -0.0000,
	// 		-11.5600, 0.0000, -0.0000, -0.0000, -0.0000, 0.0000, 0.0000, 0.0000, -0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0000, -0.0000, 0.0000, -0.0000, 0.0000,
	// 		0.0000, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000, -0.0000, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0000, 0.0000, -0.0000, 0.0000, -0.0000, 0.0000}
	// }

	// if !plan3d.IsRealHerm && !plan3d.IsForw {
	// 	size3d = [3]int{2 * plan3d.RowDim * plan3d.ColDim * plan3d.DepthDim, 1, 1}
	// 	inputs3d[0] = []float32{22.7800, 22.7800, -2.1588, 1.4788, -1.2176, 0.5376, -0.8891, 0.2091, -0.7130, 0.0330, -0.5968, -0.0832, -0.5093, -0.1707, -0.4367, -0.2433,
	// 		-0.3715, -0.3085, -0.3085, -0.3715, -0.2433, -0.4367, -0.1707, -0.5093, -0.0832, -0.5968, 0.0330, -0.7130, 0.2091, -0.8891, 0.5376, -1.2176,
	// 		1.4788, -2.1588, -5.7800, -5.7800, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0000,
	// 		-0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0000,
	// 		-11.5600, -11.5600, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0000,
	// 		-0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0000,
	// 		0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0000,
	// 		-0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0000,
	// 		0.0000, 0.0000}

	// }
	// // for i := 0; i < NComponents; i++ {
	// // 	inputs2d[i] = make([]float32, size2d[0])
	// // 	for j := 0; j < plan2d.ColDim; j++ {
	// // 		for k := 0; k < plan2d.RowDim; k++ {
	// // 			inputs2d[i][2*(j*plan2d.RowDim+k)] = float32(j*plan2d.RowDim + k)   //float32(0.1)
	// // 			inputs2d[i][2*(j*plan2d.RowDim+k)+1] = float32(j*plan2d.RowDim + k) //float32(0.1)
	// // 			fmt.Printf(" (%f, %f) ", inputs2d[i][2*(j*plan2d.RowDim+k)], inputs2d[i][2*(j*plan2d.RowDim+k)+1])
	// // 		}
	// // 		fmt.Printf("\n")
	// // 	}
	// // }
	// // }

	// fmt.Println("\n Done. Transferring input data from CPU to GPU...")
	// cpuArray3d := data.SliceFromArray(inputs3d, size3d)
	// gpu3dBuffer := opencl.Buffer(NComponents, size3d)
	// //outBuffer := opencl.Buffer(NComponents, [3]int{2 * N, 1, 1})

	// data.Copy(gpu3dBuffer, cpuArray3d)

	// fmt.Println("Waiting for data transfer to complete...")
	// queue.Finish()
	// fmt.Println("Input data transfer completed.")

	// //Parse3D(gpu3dBuffer, plan3d)

	fmt.Printf("\n Finishing FFT......\n")
	opencl.ReleaseAndClean()
}
