package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"
	"unsafe"

	"github.com/mumax/3cl/cmd/test_blu2d/purefft"
	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/opencl/cl"
	//"github.com/mumax/3cl/cmd/test_blu_2d"
)

var (
	Flag_gpu   = flag.Int("gpu", 0, "Specify GPU")
	Flag_size  = flag.Int("length", 17, "length of data to test")
	Flag_print = flag.Bool("print", false, "Print out result")
	Flag_comp  = flag.Int("components", 1, "Number of components to test")
	//Flag_conj  = flag.Bool("conjugate", false, "Conjugate B in multiplication")
)

//HermitianWarning Issue a warning if complex conjugates of hermitian are not closely matching
func HermitianWarning(InpArr *data.Slice, TotLeng, BluLength int) {

	scanFlag := TotLeng % 2
	cntPt := 1
	if scanFlag > 0 {
		cntPt = 1 + TotLeng/2
	} else {
		cntPt = TotLeng / 2
	}

	queue := opencl.ClCmdQueue
	outArray := data.NewSlice(1, [3]int{2 * BluLength, 1, 1})
	data.Copy(outArray, InpArr)
	queue.Finish()
	//fmt.Println("\n Output data transfer completed. Printing ")
	results := outArray.Host()
	incorrect := 0
	var testVarR0, testVarR1, testVarR2, testVarR3 float32

	// Check the pivots
	for i := 0; i < int(*Flag_comp); i++ {

		// Check the rest of the array
		for ii := 1; ii < cntPt; ii++ {
			reflectedIdx := 2 * (TotLeng - ii)
			testVarR0, testVarR1, testVarR2, testVarR3 = results[i][2*ii], results[i][2*ii+1], results[i][reflectedIdx], results[i][reflectedIdx+1]
			if (testVarR0 == testVarR2) && (testVarR1 == -1.0*testVarR3) {
			} else {
				fmt.Printf("Error at idx[%d]: expect (%f - i*(%f)) but have (%f + i*(%f)) \n", ii, testVarR0, testVarR1, testVarR2, testVarR3)
				incorrect++
			}
		}

		if incorrect == 0 {
			fmt.Println("All points correct!")
		} else {
			fmt.Println("Errors were found!")
		}
	}
}

//MemOffsetCpyFloat32 Memory Copy Function with offsets. For reference see MemCpy from /opencl/cl/Slice.go
// func MemOffsetCpy(dst, src unsafe.Pointer, offsetDst, OffsetSrc, bytes int) []*cl.Event {
func MemOffsetCpyFloat32(dst, src unsafe.Pointer, offsetDst, offsetSrc, bytes int) {

	queue := opencl.ClCmdQueue
	_, err := queue.EnqueueCopyBufferFloat32((*cl.MemObject)(src), (*cl.MemObject)(dst), offsetSrc, offsetDst, bytes, nil)

	//eventList[0], err = queue.EnqueueCopyBuffer(srcMemObj, dstMemObj, offsetSrc, offsetDst, bytes, nil)
	if err != nil {
		fmt.Printf("\n EnqueueCopyBuffer failed: %+v \n", err)
		panic("\n Stopping execution \n")
		//return nil
	}
	queue.Finish()
}

func findLength(tempLength int, fileName string) int {

	fmt.Printf("\n Print the filename: %s ", fileName)

	var j int
	j = 73
	m := make(map[string]int)
	strLength := strconv.Itoa(tempLength)
	fmt.Printf("\n Print the filename: %s ", strLength)
	pwd, _ := os.Getwd()

	jsonFile, err := os.Open(pwd + fileName)

	if err != nil {
		log.Fatal(err)
	}

	defer jsonFile.Close()
	byteValue, _ := ioutil.ReadAll(jsonFile)
	json.Unmarshal([]byte(byteValue), &m)

	j = m[strLength]

	fmt.Printf("\n The value of the required length is: %v", j)

	//m = nil

	return j

}

//PrintArray Prints the input array for debugging
func PrintArray(InpArr *data.Slice, ArrLength int) {
	queue := opencl.ClCmdQueue
	outArray := data.NewSlice(1, [3]int{2 * ArrLength, 1, 1})
	data.Copy(outArray, InpArr)
	fmt.Printf("\n Printing the requested array \n")
	queue.Finish()
	//fmt.Println("\n Output data transfer completed. Printing ")
	result2 := outArray.Host()
	//results := make([][]float32, 1)
	for k := 0; k < 1; k++ {
		//results[i] = make([]float32, 2*5*2)
		for j := 0; j < ArrLength; j++ {
			fmt.Printf(" ( %f , %f ) ", result2[k][2*j], result2[k][2*j+1])
		}
	}
}

//PrintRealArray Prints the input array for debugging
func PrintRealArray(InpArr *data.Slice, ArrLength int) {
	queue := opencl.ClCmdQueue
	outArray := data.NewSlice(1, [3]int{ArrLength, 1, 1})
	data.Copy(outArray, InpArr)
	fmt.Printf("\n Printing the requested array \n")
	queue.Finish()
	//fmt.Println("\n Output data transfer completed. Printing ")
	result2 := outArray.Host()
	//results := make([][]float32, 1)
	for k := 0; k < 1; k++ {
		//results[i] = make([]float32, 2*5*2)
		for j := 0; j < ArrLength; j++ {
			fmt.Printf(" ( %f )", result2[k][j])
		}
	}
}

func blusteinCase(length int) (int, int) {
	switch {
	case length > 128000000:
		return length, -1
	case length > 115200000:
		return findLength(length, "/go/bin/new_length_lookup_10.json"), 1 //Hardcoded filenames
	case length > 102400000:
		return findLength(length, "new_length_lookup_9.json"), 1
	case length > 89600000:
		return findLength(length, "new_length_lookup_8.json"), 1
	case length > 76800000:
		return findLength(length, "new_length_lookup_7.json"), 1
	case length > 64000000:
		return findLength(length, "new_length_lookup_6.json"), 1
	case length > 51200000:
		return findLength(length, "new_length_lookup_5.json"), 1
	case length > 38400000:
		return findLength(length, "new_length_lookup_4.json"), 1
	case length > 25600000:
		return findLength(length, "new_length_lookup_3.json"), 1
	case length > 12800000:
		return findLength(length, "new_length_lookup_2.json"), 1
	case length > 1:
		return findLength(length, "/new_length_lookup_1.json"), 1
	case length < 2:
		return length, -2
	}
	return length, -3
}

//FftPlanValue Structure to identify the plan for processing
type FftPlanValue struct {
	IsForw, IsRealHerm, IsSinglePreci, IsBlusteinsReq bool
	RowDim, ColDim, DepthDim, FinalN                  int
}

// FftPlan2DValue Structure to represent plan for FFT processing of 2D array
type FftPlan2DValue struct {
	IsForw, IsRealHerm, IsSinglePreci, IsBlusteinsReqRow, IsBlusteinsReqCol bool
	RowDim, ColDim, DepthDim, RowBluLeng, ColBluLeng                        int
}

//BoolGen to generate random plan values
func BoolGen() bool {
	var src = rand.NewSource(time.Now().UnixNano())
	var r = rand.New(src)
	return r.Int63n(2) == 0
}

//Parse1D Parsing the 1D Input
func Parse1D(InpBuf *data.Slice, class interface{}) {
	//queue := opencl.ClCmdQueue
	fmt.Printf("\n Parsing the input to execute appropriate FFT function...\n")
	inp1d, ok := class.(FftPlanValue)
	if !ok {
		panic("\n Wrong Input given... Terminating...\n")
	}
	var Desci int
	inp1d.FinalN, Desci = blusteinCase(inp1d.RowDim) //Desci is decision variable

	fmt.Printf("\n Value of new length : %d", inp1d.FinalN)

	switch Desci {
	case -1:
		panic("\n Error! Length too large to handle! Terminating immidiately...")
	case -2:
		panic("\n Error! Length too small/negative to handle! Terminating immidiately...")
	case -3:
		panic("\n Something is weird! Terminating... Check immidiately...")
	case 1:
		if inp1d.FinalN == 0 {
			// fmt.Printf("\n Bluestein is not required. Executing clFFT with length %v...", BluN)
			inp1d.FinalN = inp1d.RowDim
			inp1d.IsBlusteinsReq = false
			fmt.Printf("\n Blusteins Algorithm not required as Legnth = %d...\n", inp1d.FinalN)
		} else {
			inp1d.FinalN = 2 * inp1d.FinalN
			inp1d.IsBlusteinsReq = true
			fmt.Printf("\n Adjusting length and finding FFT using Blusteins Algorithm with Legnth = %d...\n", inp1d.FinalN)
		}
	}
	var OutputBuf *data.Slice
	if !inp1d.IsForw && inp1d.IsRealHerm {
		OutputBuf = opencl.Buffer(int(*Flag_comp), [3]int{inp1d.RowDim, 1, 1})
	} else if inp1d.IsForw && inp1d.IsRealHerm {
		OutputBuf = opencl.Buffer(int(*Flag_comp), [3]int{2 * (1 + inp1d.RowDim/2), 1, 1})
	} else {
		OutputBuf = opencl.Buffer(int(*Flag_comp), [3]int{2 * inp1d.RowDim, 1, 1})
	}

	//OutputBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * inp1d.RowDim, 1, 1})
	defer opencl.Recycle(OutputBuf)

	FFT1D(OutputBuf, InpBuf, inp1d)

	fmt.Print("\n Finished calculating 1D FFT. Output will be \n")

	if !inp1d.IsForw && inp1d.IsRealHerm {
		PrintRealArray(OutputBuf, inp1d.RowDim)
	} else if inp1d.IsForw && inp1d.IsRealHerm {
		PrintArray(OutputBuf, 1+int(inp1d.RowDim/2))
	} else {
		PrintArray(OutputBuf, inp1d.RowDim)
	}
	// PrintArray(OutputBuf, inp1d.RowDim)
}

//Parse2D Parsing the 2D Input
func Parse2D(InpBuf *data.Slice, class interface{}) {

	fmt.Printf("\n Parsing the input to execute appropriate FFT function...\n")
	c, ok := class.(FftPlan2DValue)
	if !ok {
		panic("\n Wrong 2D Input given... Terminating...\n")
	}
	//context := opencl.ClCtx
	//queue := opencl.ClCmdQueue

	var PrintSize int

	fmt.Printf("\n Calculating 2D FFT for the given input \n")

	fmt.Printf("\n Generating the correct size Output matrix \n")

	var FinalBuf *data.Slice
	if !c.IsForw && c.IsRealHerm {
		FinalBuf = opencl.Buffer(int(*Flag_comp), [3]int{c.RowDim * c.ColDim, 1, 1})
	}
	if c.IsForw && c.IsRealHerm {
		FinalBuf = opencl.Buffer(int(*Flag_comp), [3]int{2 * c.ColDim * (1 + c.RowDim/2), 1, 1})
	}
	if !c.IsRealHerm {
		FinalBuf = opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim * c.ColDim, 1, 1})
	}

	//var IsBlusteinRow, IsBlusteinCol bool
	ValRow, DecideRow := blusteinCase(c.RowDim)
	ValCol, DecideCol := blusteinCase(c.ColDim)
	if (DecideRow == 1) && (DecideCol == 1) && (ValRow == 0) && (ValCol == 0) {
		fmt.Printf("\n No need to execute Blusteins for any dimension. Executing CLFFT directly \n")
		if c.IsForw {
			if c.IsRealHerm {
				PrintSize = c.ColDim * int(1+c.RowDim/2)
				// TempBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.ColDim * (1 + c.RowDim/2), 1, 1})
				// defer opencl.Recycle(TempBuf)
				fmt.Printf("\n Executing Forward Real FFT without Bluestein's ...\n")
				purefft.Clfft2D(InpBuf, FinalBuf, c.RowDim, c.ColDim, true, true, c.IsSinglePreci)
				//purefft.Clfft2D(InpBuf, TempBuf, c.RowDim, c.ColDim, true, true, c.IsSinglePreci)

				// fmt.Printf("\n Running Hermitian to Full \n")
				// //FinalBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.ColDim * c.RowDim, 1, 1})
				// for j := 0; j < c.ColDim; j++ {
				// 	SmallBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * (1 + c.RowDim/2), 1, 1})
				// 	MemOffsetCpyFloat32(SmallBuff.DevPtr(0), TempBuf.DevPtr(0), 0, 2*j*c.RowDim, 2*c.RowDim)
				// 	HermBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
				// 	opencl.Hermitian2Full(HermBuff, SmallBuff)
				// 	MemOffsetCpyFloat32(FinalBuf.DevPtr(0), HermBuff.DevPtr(0), 2*j*c.RowDim, 0, 2*c.RowDim)
				// 	opencl.Recycle(SmallBuff)
				// 	opencl.Recycle(HermBuff)
				// }
				//opencl.Hermitian2Full(FinalBuf, TempBuf)
				//fmt.Printf("\n Finished running Hermitian to Full. Final Output is ready \n")
			} else {
				//FinalBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.ColDim * c.RowDim, 1, 1})
				fmt.Printf("\n Executing Forward Complex FFT without Bluestein's ...\n")
				purefft.Clfft2D(InpBuf, FinalBuf, c.RowDim, c.ColDim, false, true, c.IsSinglePreci)
			}

		} else {
			if c.IsRealHerm {

				//FinalBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.ColDim * c.RowDim, 1, 1})
				// TempBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.ColDim * c.RowDim, 1, 1})
				// defer opencl.Recycle(TempBuf)
				// fmt.Printf("\n Executing Forward Real FFT without Bluestein's ...\n")
				// //purefft.Clfft2D(InpBuf, TempBuf, c.RowDim, c.ColDim, c.IsRealHerm, c.IsForw, c.IsSinglePreci)

				// fmt.Printf("\n Running Hermitian to Full \n")
				// //FinalBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.ColDim * c.RowDim, 1, 1})
				// for j := 0; j < c.ColDim; j++ {
				// 	SmallBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * (1 + c.RowDim/2), 1, 1})
				// 	MemOffsetCpyFloat32(SmallBuff.DevPtr(0), InpBuf.DevPtr(0), 0, 2*j*(1+c.RowDim/2), 2*(1+c.RowDim/2))
				// 	HermBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
				// 	opencl.Hermitian2Full(HermBuff, SmallBuff)
				// 	MemOffsetCpyFloat32(TempBuf.DevPtr(0), HermBuff.DevPtr(0), 2*j*c.RowDim, 0, 2*c.RowDim)
				// 	opencl.Recycle(SmallBuff)
				// 	opencl.Recycle(HermBuff)
				// }
				fmt.Printf("\n Executing Inverse Hermitian FFT without Bluestein's...\n")
				purefft.Clfft2D(InpBuf, FinalBuf, c.RowDim, c.ColDim, c.IsRealHerm, c.IsForw, c.IsSinglePreci)

			} else {
				//FinalBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim * c.RowDim, 1, 1})
				fmt.Printf("\n Executing Inverse Complex FFT without Bluestein's...\n")
				purefft.Clfft2D(InpBuf, FinalBuf, c.RowDim, c.ColDim, c.IsRealHerm, c.IsForw, c.IsSinglePreci)
			}
		}
	}
	if (DecideRow == 1) && (DecideCol == 1) && ((ValRow != 0) || (ValCol != 0)) {
		fmt.Printf("\n Blusteins required for at least one dimension \n")
		if ValRow == 0 {
			// fmt.Printf("\n Bluestein is not required. Executing clFFT with length %v...", BluN)
			c.RowBluLeng = c.RowDim
			c.IsBlusteinsReqRow = false
			fmt.Printf("\n Blusteins Algorithm not required for Row as Legnth = %d...\n", c.RowBluLeng)
		} else {
			c.RowBluLeng = 2 * ValRow
			c.IsBlusteinsReqRow = true
			fmt.Printf("\n Blusteins Algorithm required for Rows with New Legnth = %d...\n", c.RowBluLeng)
		}
		if ValCol == 0 {
			// fmt.Printf("\n Bluestein is not required. Executing clFFT with length %v...", BluN)
			c.ColBluLeng = c.ColDim
			c.IsBlusteinsReqCol = false
			fmt.Printf("\n Blusteins Algorithm not required for Columns as Legnth = %d...\n", c.ColBluLeng)
		} else {
			c.ColBluLeng = 2 * ValCol
			c.IsBlusteinsReqCol = true
			fmt.Printf("\n Blusteins Algorithm required for Columns with New Legnth = %d...\n", c.ColBluLeng)
		}

		fmt.Printf("\n Blusteins assignments finished. Now non blustein cases begin \n")

		if c.IsForw {
			if c.IsRealHerm {
				for i := 0; i < int(*Flag_comp); i++ {
					PrintSize = int(1+c.RowDim/2) * c.ColDim
					TempOutBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * int(1+c.RowDim/2) * c.ColDim, 1, 1})
					TranpoBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.ColDim * int(1+c.RowDim/2), 1, 1})
					//defer opencl.Recycle(TempOutBuf)
					for j := 0; j < c.ColDim; j++ {
						//SmallBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
						SmallBuff := opencl.Buffer(int(*Flag_comp), [3]int{c.RowDim, 1, 1})
						MemOffsetCpyFloat32(SmallBuff.DevPtr(0), InpBuf.DevPtr(0), 0, j*c.RowDim, c.RowDim)
						FftBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * int(1+c.RowDim/2), 1, 1})
						TempPlan := FftPlanValue{true, true, true, c.IsBlusteinsReqRow, c.RowDim, 1, 1, c.RowBluLeng}
						FFT1D(FftBuff, SmallBuff, TempPlan)
						MemOffsetCpyFloat32(TempOutBuf.DevPtr(0), FftBuff.DevPtr(0), 2*j*int(1+c.RowDim/2), 0, 2*int(1+c.RowDim/2))
						opencl.Recycle(SmallBuff)
						opencl.Recycle(FftBuff)
					}

					fmt.Printf("\n Implementing Transpose \n")
					opencl.ComplexMatrixTranspose(TranpoBuf, TempOutBuf, 0, int(1+c.RowDim/2), c.ColDim)
					fmt.Printf("\n Finished transposing \n")

					SecOutBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * int(1+c.RowDim/2) * c.ColDim, 1, 1})

					//FinalTempBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim * c.ColDim, 1, 1})
					//defer opencl.Recycle(SecOutBuf)
					for j := 0; j < int(1+c.RowDim/2); j++ {
						SmallBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.ColDim, 1, 1})
						MemOffsetCpyFloat32(SmallBuff.DevPtr(0), TranpoBuf.DevPtr(0), 0, 2*j*c.ColDim, 2*c.ColDim)
						FftBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.ColDim, 1, 1})
						TempPlan := FftPlanValue{true, false, true, c.IsBlusteinsReqCol, c.ColDim, 1, 1, c.ColBluLeng}
						FFT1D(FftBuff, SmallBuff, TempPlan)
						MemOffsetCpyFloat32(SecOutBuf.DevPtr(0), FftBuff.DevPtr(0), 2*j*c.ColDim, 0, 2*c.ColDim)
						opencl.Recycle(SmallBuff)
						opencl.Recycle(FftBuff)
					}
					fmt.Printf("\n Implementing Transpose \n")
					opencl.ComplexMatrixTranspose(FinalBuf, SecOutBuf, 0, c.ColDim, int(1+c.RowDim/2))
					fmt.Printf("\n Printing 2d individual output array \n")
					//PrintArray(FinalBuf, c.RowDim*c.ColDim)
					opencl.Recycle(TempOutBuf)
					opencl.Recycle(SecOutBuf)
					opencl.Recycle(TranpoBuf)
					//opencl.Recycle(FinalTempBuf)
				}
			} else {
				for i := 0; i < int(*Flag_comp); i++ {
					PrintSize = c.RowDim * c.ColDim
					TempOutBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim * c.ColDim, 1, 1})
					TranpoBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim * c.ColDim, 1, 1})
					//defer opencl.Recycle(TempOutBuf)
					for j := 0; j < c.ColDim; j++ {
						SmallBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
						MemOffsetCpyFloat32(SmallBuff.DevPtr(0), InpBuf.DevPtr(0), 0, 2*j*c.RowDim, 2*c.RowDim)
						FftBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
						TempPlan := FftPlanValue{true, false, true, c.IsBlusteinsReqRow, c.RowDim, 1, 1, c.RowBluLeng}
						FFT1D(FftBuff, SmallBuff, TempPlan)
						MemOffsetCpyFloat32(TempOutBuf.DevPtr(0), FftBuff.DevPtr(0), 2*j*c.RowDim, 0, 2*c.RowDim)
						opencl.Recycle(SmallBuff)
						opencl.Recycle(FftBuff)
					}

					fmt.Printf("\n Implementing Transpose \n")
					opencl.ComplexMatrixTranspose(TranpoBuf, TempOutBuf, 0, c.RowDim, c.ColDim)

					SecOutBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim * c.ColDim, 1, 1})

					//TranpoBuf := opencl.Buffer(NComponents, [3]int{2 * plan2d.RowDim * plan2d.ColDim, 1, 1})
					//defer opencl.Recycle(SecOutBuf)
					for j := 0; j < c.RowDim; j++ {
						SmallBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.ColDim, 1, 1})
						MemOffsetCpyFloat32(SmallBuff.DevPtr(0), TranpoBuf.DevPtr(0), 0, 2*j*c.ColDim, 2*c.ColDim)
						FftBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.ColDim, 1, 1})
						TempPlan := FftPlanValue{true, false, true, c.IsBlusteinsReqCol, c.ColDim, 1, 1, c.ColBluLeng}
						FFT1D(FftBuff, SmallBuff, TempPlan)
						MemOffsetCpyFloat32(SecOutBuf.DevPtr(0), FftBuff.DevPtr(0), 2*j*c.ColDim, 0, 2*c.ColDim)
						opencl.Recycle(SmallBuff)
						opencl.Recycle(FftBuff)
					}
					fmt.Printf("\n Implementing Transpose \n")
					opencl.ComplexMatrixTranspose(FinalBuf, SecOutBuf, 0, c.ColDim, c.RowDim)
					fmt.Printf("\n Printing 2d individual output array \n")
					//PrintArray(FinalBuf, c.RowDim*c.ColDim)
					opencl.Recycle(TempOutBuf)
					opencl.Recycle(SecOutBuf)
					opencl.Recycle(TranpoBuf)
				}
			}
		} else if c.IsRealHerm {
			for i := 0; i < int(*Flag_comp); i++ {
				PrintSize = c.RowDim * c.ColDim
				TempOutBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim * c.ColDim, 1, 1})
				TranpoBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim * c.ColDim, 1, 1})
				//defer opencl.Recycle(TempOutBuf)
				for j := 0; j < c.ColDim; j++ {
					SmallBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * int(1+c.RowDim/2), 1, 1})
					MemOffsetCpyFloat32(SmallBuff.DevPtr(0), InpBuf.DevPtr(0), 0, 2*j*int(1+c.RowDim/2), 2*int(1+c.RowDim/2))
					TempFftBuff := opencl.Buffer(int(*Flag_comp), [3]int{c.RowDim, 1, 1})
					TempPlan := FftPlanValue{false, true, true, c.IsBlusteinsReqRow, c.RowDim, 1, 1, c.RowBluLeng}
					FFT1D(TempFftBuff, SmallBuff, TempPlan)
					FftBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
					opencl.PackComplexArray(FftBuff, TempFftBuff, c.RowDim, 0, 0)
					MemOffsetCpyFloat32(TempOutBuf.DevPtr(0), FftBuff.DevPtr(0), 2*j*c.RowDim, 0, 2*c.RowDim)
					opencl.Recycle(SmallBuff)
					opencl.Recycle(FftBuff)
					opencl.Recycle(TempFftBuff)
				}
				fmt.Printf("\n Implementing Transpose \n")
				opencl.ComplexMatrixTranspose(TranpoBuf, TempOutBuf, 0, c.RowDim, c.ColDim)

				SecOutBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim * c.ColDim, 1, 1})
				//TranpoBuf := opencl.Buffer(NComponents, [3]int{2 * plan2d.RowDim * plan2d.ColDim, 1, 1})
				//defer opencl.Recycle(SecOutBuf)
				for j := 0; j < c.RowDim; j++ {
					SmallBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.ColDim, 1, 1})
					MemOffsetCpyFloat32(SmallBuff.DevPtr(0), TranpoBuf.DevPtr(0), 0, 2*j*c.ColDim, 2*c.ColDim)
					FftBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.ColDim, 1, 1})
					TempPlan := FftPlanValue{false, false, true, c.IsBlusteinsReqCol, c.ColDim, 1, 1, c.ColBluLeng}
					FFT1D(FftBuff, SmallBuff, TempPlan)
					MemOffsetCpyFloat32(SecOutBuf.DevPtr(0), FftBuff.DevPtr(0), 2*j*c.ColDim, 0, 2*c.ColDim)
					opencl.Recycle(SmallBuff)
					opencl.Recycle(FftBuff)
				}
				fmt.Printf("\n Implementing Transpose \n")
				OutTransTempBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim * c.ColDim, 1, 1})
				opencl.ComplexMatrixTranspose(OutTransTempBuff, SecOutBuf, 0, c.ColDim, c.RowDim)
				fmt.Printf("\n Printing 2d individual output array \n")
				for j := 0; j < c.ColDim; j++ {
					SmallBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
					MemOffsetCpyFloat32(SmallBuff.DevPtr(0), OutTransTempBuff.DevPtr(0), 0, 2*j*c.RowDim, 2*c.RowDim)
					fmt.Printf("\n Finished first copy \n")
					TempCompressBuff := opencl.Buffer(int(*Flag_comp), [3]int{c.RowDim, 1, 1})
					opencl.CompressCmplxtoReal(TempCompressBuff, SmallBuff, c.RowDim, 0, 0)
					MemOffsetCpyFloat32(FinalBuf.DevPtr(0), TempCompressBuff.DevPtr(0), j*c.RowDim, 0, c.RowDim)
					opencl.Recycle(SmallBuff)
					opencl.Recycle(TempCompressBuff)
				}
				//PrintArray(FinalBuf, c.RowDim*c.ColDim)
				opencl.Recycle(TempOutBuf)
				opencl.Recycle(SecOutBuf)
				opencl.Recycle(TranpoBuf)
				opencl.Recycle(OutTransTempBuff)
			}
		} else {
			for i := 0; i < int(*Flag_comp); i++ {
				PrintSize = c.RowDim * c.ColDim
				TempOutBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim * c.ColDim, 1, 1})
				TranpoBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim * c.ColDim, 1, 1})
				//defer opencl.Recycle(TempOutBuf)
				for j := 0; j < c.ColDim; j++ {
					SmallBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
					MemOffsetCpyFloat32(SmallBuff.DevPtr(0), InpBuf.DevPtr(0), 0, 2*j*c.RowDim, 2*c.RowDim)
					FftBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
					TempPlan := FftPlanValue{false, false, true, c.IsBlusteinsReqRow, c.RowDim, 1, 1, c.RowBluLeng}
					FFT1D(FftBuff, SmallBuff, TempPlan)
					MemOffsetCpyFloat32(TempOutBuf.DevPtr(0), FftBuff.DevPtr(0), 2*j*c.RowDim, 0, 2*c.RowDim)
					opencl.Recycle(SmallBuff)
					opencl.Recycle(FftBuff)
				}
				fmt.Printf("\n Implementing Transpose \n")
				opencl.ComplexMatrixTranspose(TranpoBuf, TempOutBuf, 0, c.RowDim, c.ColDim)

				SecOutBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim * c.ColDim, 1, 1})
				//TranpoBuf := opencl.Buffer(NComponents, [3]int{2 * plan2d.RowDim * plan2d.ColDim, 1, 1})
				//defer opencl.Recycle(SecOutBuf)
				for j := 0; j < c.RowDim; j++ {
					SmallBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.ColDim, 1, 1})
					MemOffsetCpyFloat32(SmallBuff.DevPtr(0), TranpoBuf.DevPtr(0), 0, 2*j*c.ColDim, 2*c.ColDim)
					FftBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.ColDim, 1, 1})
					TempPlan := FftPlanValue{false, false, true, c.IsBlusteinsReqCol, c.ColDim, 1, 1, c.ColBluLeng}
					FFT1D(FftBuff, SmallBuff, TempPlan)
					MemOffsetCpyFloat32(SecOutBuf.DevPtr(0), FftBuff.DevPtr(0), 2*j*c.ColDim, 0, 2*c.ColDim)
					opencl.Recycle(SmallBuff)
					opencl.Recycle(FftBuff)
				}
				fmt.Printf("\n Implementing Transpose \n")
				opencl.ComplexMatrixTranspose(FinalBuf, SecOutBuf, 0, c.ColDim, c.RowDim)
				fmt.Printf("\n Printing 2d individual output array \n")
				//PrintArray(FinalBuf, c.RowDim*c.ColDim)
				opencl.Recycle(TempOutBuf)
				opencl.Recycle(SecOutBuf)
				opencl.Recycle(TranpoBuf)
			}
		}

	}

	if c.IsRealHerm && !c.IsForw {
		PrintRealArray(FinalBuf, PrintSize)
	} else {
		PrintArray(FinalBuf, PrintSize)
	}
}

//FFT1D to identify the details about the FFT
func FFT1D(FinalBuf, InpBuf *data.Slice, class interface{}) {
	queue := opencl.ClCmdQueue
	fmt.Printf("\n Parsing the input to execute appropriate FFT function...\n")
	c, ok := class.(FftPlanValue)
	if !ok {
		panic("\n Wrong Input given... Terminating...\n")
	}

	if !c.IsBlusteinsReq {
		if c.IsForw {
			if c.IsRealHerm {
				// TempBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
				// defer opencl.Recycle(TempBuf)
				// fmt.Printf("\n Converting Real Array to Packed Complex \n")
				// opencl.PackComplexArray(TempBuf, InpBuf, c.RowDim, 0, 0)
				// fmt.Println("\n Waiting for kernel to finish execution...")
				// queue.Finish()
				// fmt.Println("\n Execution finished.")
				//OpBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * (1 + c.RowDim/2), 1, 1})
				//defer opencl.Recycle(OpBuf)
				fmt.Printf("\n Executing Forward Real FFT without Bluestein's ...\n")
				//purefft.Clfft1D(TempBuf, OpBuf, c.RowDim, true, true, c.IsSinglePreci)
				purefft.Clfft1D(InpBuf, FinalBuf, c.RowDim, c.FinalN, true, true, c.IsSinglePreci, false)

				// fmt.Printf("\n Printing final array...\n")
				// PrintArray(FinalBuf, int(1+c.RowDim/2))

				//fmt.Printf("\n Running Hermitian to Full \n")
				//FinalBuf := data.NewSlice(int(*Flag_comp), [3]int{2 * (1 + c.RowDim), 1, 1})
				//opencl.Hermitian2Full(FinalBuf, OpBuf)
				//fmt.Printf("\n Finished running Hermitian to Full. Final Output is ready \n")
				//return FinalBuf
			} else {
				//FinalBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
				fmt.Printf("\n Executing Forward Complex FFT without Bluestein's ...\n")
				purefft.Clfft1D(InpBuf, FinalBuf, c.RowDim, c.FinalN, false, true, c.IsSinglePreci, false)
				//return FinalBuf
			}

		} else {
			if c.IsRealHerm {
				//TempBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
				//defer opencl.Recycle(TempBuf)
				//opencl.Hermitian2Full(TempBuf, InpBuf)
				fmt.Printf("\n Executing Inverse Hermitian FFT without Bluestein's...\n")
				//purefft.Clfft1D(InpBuf, FinalBuf, c.RowDim, c.IsRealHerm, c.IsForw, c.IsSinglePreci)
				purefft.Clfft1D(InpBuf, FinalBuf, c.RowDim, c.FinalN, true, false, c.IsSinglePreci, false)
				//return FinalBuf

			} else {
				//FinalBuf := data.NewSlice(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
				fmt.Printf("\n Executing Inverse Complex FFT without Bluestein's...\n")
				purefft.Clfft1D(InpBuf, FinalBuf, c.RowDim, c.FinalN, false, false, c.IsSinglePreci, false)
				//return FinalBuf
			}
		}

	} else {
		if c.IsForw {
			if c.IsRealHerm {
				fmt.Printf("\n Executing Forward Real FFT with Bluesteins...\n")

				PartATempBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
				defer opencl.Recycle(PartATempBuf)
				fmt.Printf("\n Processing Part A with the twiddle factor \n")
				opencl.PackComplexArray(PartATempBuf, InpBuf, c.RowDim, 0, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				PartAProcBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(PartAProcBuf)
				fmt.Printf("\n Processing Part A with the twiddle factor \n")
				opencl.PartAProcess(PartAProcBuf, PartATempBuf, c.RowDim, c.FinalN, 1, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				PartBBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(PartBBuf)
				fmt.Printf("\n Generating part B for Bluesteins")
				opencl.PartBTwidFac(PartBBuf, c.RowDim, c.FinalN, 1, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Executing forward FFT for Part A \n")
				PartAFFT := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(PartAFFT)
				purefft.Clfft1D(PartAProcBuf, PartAFFT, c.FinalN, c.FinalN, false, true, c.IsSinglePreci, false)
				fmt.Printf("\n Executing forward FFT for Part B \n")
				PartBFFT := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(PartBFFT)
				purefft.Clfft1D(PartBBuf, PartBFFT, c.FinalN, c.FinalN, false, true, c.IsSinglePreci, false)

				fmt.Printf("\n Multiplying Part A and Part B FFT \n")
				MulBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(MulBuff)
				opencl.ComplexArrayMul(MulBuff, PartAFFT, PartBFFT, 0, c.FinalN, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Taking inverse FFT of multiplication \n")
				InvBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(InvBuff)
				purefft.Clfft1D(MulBuff, InvBuff, c.FinalN, c.FinalN, false, false, c.IsSinglePreci, false)

				fmt.Printf("\n Preparing final twiddle factor")
				FinTwid := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(FinTwid)
				opencl.FinalMulTwid(FinTwid, c.RowDim, c.FinalN, 1, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Multiplying with Final Twiddle Factor")
				// opencl.ComplexArrayMul(FinalBuf, FinTwid, InvBuff, 0, c.FinalN, 0)
				// fmt.Println("\n Waiting for kernel to finish execution...")
				// queue.Finish()
				// fmt.Println("\n Execution finished.")

				FinTempBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(FinTempBuff)
				//opencl.ComplexArrayMul(FinalBuf, FinTwid, InvBuff, 0, c.FinalN, 0)
				opencl.ComplexArrayMul(FinTempBuff, FinTwid, InvBuff, 0, c.FinalN, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				// Execute the special harm check function here
				fmt.Printf("\n Checking Hermitian Warning \n")
				HermitianWarning(FinTempBuff, c.RowDim, c.FinalN)

				MemOffsetCpyFloat32(FinalBuf.DevPtr(0), FinTempBuff.DevPtr(0), 0, 0, 2*int(1+c.RowDim/2))
				fmt.Printf("\n Finished calculating Foward FFT using Blusteins Method \n")

			} else {

				fmt.Printf("\n Executing Forward Complex FFT with Bluesteins...\n")

				PartAProcBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(PartAProcBuf)
				fmt.Printf("\n Processing Part A with the twiddle factor \n")
				opencl.PartAProcess(PartAProcBuf, InpBuf, c.RowDim, c.FinalN, 1, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				PartBBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(PartBBuf)
				fmt.Printf("\n Generating part B for Bluesteins")
				opencl.PartBTwidFac(PartBBuf, c.RowDim, c.FinalN, 1, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Executing forward FFT for Part A \n")
				PartAFFT := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(PartAFFT)
				purefft.Clfft1D(PartAProcBuf, PartAFFT, c.FinalN, c.FinalN, false, true, c.IsSinglePreci, false)

				fmt.Printf("\n Executing forward FFT for Part B \n")

				PartBFFT := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(PartBFFT)
				purefft.Clfft1D(PartBBuf, PartBFFT, c.FinalN, c.FinalN, false, true, c.IsSinglePreci, false)

				fmt.Printf("\n Multiplying Part A and Part B FFT \n")
				MulBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(MulBuff)
				opencl.ComplexArrayMul(MulBuff, PartAFFT, PartBFFT, 0, c.FinalN, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Taking inverse FFT of multiplication \n")
				InvBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(InvBuff)
				purefft.Clfft1D(MulBuff, InvBuff, c.FinalN, c.FinalN, false, false, c.IsSinglePreci, false)

				fmt.Printf("\n Preparing final twiddle factor")
				FinTwid := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(FinTwid)
				opencl.FinalMulTwid(FinTwid, c.RowDim, c.FinalN, 1, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Multiplying with Final Twiddle Factor")
				FinTempBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(FinTempBuff)
				//opencl.ComplexArrayMul(FinalBuf, FinTwid, InvBuff, 0, c.FinalN, 0)
				opencl.ComplexArrayMul(FinTempBuff, FinTwid, InvBuff, 0, c.FinalN, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				MemOffsetCpyFloat32(FinalBuf.DevPtr(0), FinTempBuff.DevPtr(0), 0, 0, 2*c.RowDim)
				fmt.Printf("\n Finished calculating Foward FFT using Blusteins Method \n")

			}
		} else {
			if c.IsRealHerm {
				fmt.Printf("\n Executing Inverse Real FFT with Bluesteins...\n")

				PartABuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
				defer opencl.Recycle(PartABuf)

				fmt.Printf("\n Converting Hermitian to Full Complex of Part A to complex for multiplication with twiddle factor\n")
				opencl.Hermitian2Full(PartABuf, InpBuf)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				// fmt.Println("\n Execution finished.")
				// PrintArray(PartABuf, c.RowDim)

				PartAProcBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(PartAProcBuf)
				fmt.Printf("\n Processing Part A with the twiddle factor \n")
				opencl.PartAProcess(PartAProcBuf, PartABuf, c.RowDim, c.FinalN, -1, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				PartBBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(PartBBuf)
				fmt.Printf("\n Generating part B for Bluesteins")
				opencl.PartBTwidFac(PartBBuf, c.RowDim, c.FinalN, -1, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Executing forward FFT for Part A \n")
				PartAFFT := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(PartAFFT)
				purefft.Clfft1D(PartAProcBuf, PartAFFT, c.FinalN, c.FinalN, false, true, c.IsSinglePreci, false)
				fmt.Printf("\n Executing forward FFT for Part B \n")
				PartBFFT := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(PartBFFT)
				purefft.Clfft1D(PartBBuf, PartBFFT, c.FinalN, c.FinalN, false, true, c.IsSinglePreci, false)
				fmt.Printf("\n Multiplying Part A and Part B FFT \n")
				MulBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(MulBuff)
				opencl.ComplexArrayMul(MulBuff, PartAFFT, PartBFFT, 0, c.FinalN, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Taking inverse FFT of multiplication \n")
				InvBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(InvBuff)
				purefft.Clfft1D(MulBuff, InvBuff, c.FinalN, c.RowDim, false, false, c.IsSinglePreci, true)

				fmt.Printf("\n Preparing final twiddle factor")
				FinTwid := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(FinTwid)
				opencl.FinalMulTwid(FinTwid, c.RowDim, c.FinalN, -1, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Multiplying with Final Twiddle Factor")
				FinTempBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(FinTempBuff)
				opencl.ComplexArrayMul(FinTempBuff, FinTwid, InvBuff, 0, c.FinalN, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				// ScaleBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				// defer opencl.Recycle(ScaleBuff)
				// fmt.Printf("\n Scaling for final FFT")
				// opencl.ScaleDown(ScaleBuff, FinTempBuff, c.RowDim, c.FinalN, 0)
				// fmt.Println("\n Waiting for kernel to finish execution...")
				// queue.Finish()
				// fmt.Println("\n Execution finished.")
				fmt.Printf("\n Converting the array to required length \n")
				opencl.CompressCmplxtoReal(FinalBuf, FinTempBuff, c.RowDim, 0, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")
				//MemOffsetCpyFloat32(FinalBuf.DevPtr(0), FinTempBuff.DevPtr(0), 0, 0, c.RowDim)
				fmt.Printf("\n Finished calculating Inverse FFT using Blusteins Method \n")

			} else {
				fmt.Printf("\n Executing Inverse Complex FFT ...\n")

				PartAProcBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(PartAProcBuf)
				fmt.Printf("\n Processing Part A with the twiddle factor \n")
				opencl.PartAProcess(PartAProcBuf, InpBuf, c.RowDim, c.FinalN, -1, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				PartBBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(PartBBuf)
				fmt.Printf("\n Generating part B for Bluesteins")
				opencl.PartBTwidFac(PartBBuf, c.RowDim, c.FinalN, -1, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Executing forward FFT for Part A \n")
				PartAFFT := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(PartAFFT)
				purefft.Clfft1D(PartAProcBuf, PartAFFT, c.FinalN, c.FinalN, false, true, c.IsSinglePreci, false)

				fmt.Printf("\n Executing forward FFT for Part B \n")
				PartBFFT := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(PartBFFT)
				purefft.Clfft1D(PartBBuf, PartBFFT, c.FinalN, c.FinalN, false, true, c.IsSinglePreci, false)

				fmt.Printf("\n Multiplying Part A and Part B FFT \n")
				MulBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(MulBuff)
				opencl.ComplexArrayMul(MulBuff, PartAFFT, PartBFFT, 0, c.FinalN, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Taking inverse FFT of multiplication \n")
				InvBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(InvBuff)
				purefft.Clfft1D(MulBuff, InvBuff, c.FinalN, c.RowDim, false, false, c.IsSinglePreci, true)

				fmt.Printf("\n Preparing final twiddle factor")
				FinTwid := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				opencl.FinalMulTwid(FinTwid, c.RowDim, c.FinalN, -1, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Multiplying with Final Twiddle Factor")
				FinTempBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				defer opencl.Recycle(FinTempBuff)
				opencl.ComplexArrayMul(FinTempBuff, FinTwid, InvBuff, 0, c.FinalN, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				// ScaleBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.FinalN, 1, 1})
				// defer opencl.Recycle(ScaleBuff)
				// fmt.Printf("\n Scaling for final FFT")
				// opencl.ScaleDown(ScaleBuff, FinTempBuff, c.RowDim, c.FinalN, 0)
				// fmt.Println("\n Waiting for kernel to finish execution...")
				// queue.Finish()
				// fmt.Println("\n Execution finished.")

				MemOffsetCpyFloat32(FinalBuf.DevPtr(0), FinTempBuff.DevPtr(0), 0, 0, 2*c.RowDim)
				fmt.Printf("\n Finished calculating Inverse FFT using Blusteins Method \n")
				//Alternatively use opencl/cl/clFFT.go => SetScale()

			}
		}
	}
}

func main() {

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

	plan1d := FftPlanValue{false, true, true, false, N, 1, 1, 0}

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
	// fmt.Printf("  Available: %+v \n", d.Available())
	// fmt.Printf("  Compiler Available: %+v \n", d.CompilerAvailable())
	// fmt.Printf("  Double FP Config: %s \n", d.DoubleFPConfig())
	// fmt.Printf("  Driver Version: %s \n", d.DriverVersion())
	// fmt.Printf("  Error Correction Supported: %+v \n", d.ErrorCorrectionSupport())
	// fmt.Printf("  Execution Capabilities: %s \n", d.ExecutionCapabilities())
	// fmt.Printf("  Extensions: %s \n", d.Extensions())
	// fmt.Printf("  Global Memory Cache Type: %s \n", d.GlobalMemCacheType())
	// fmt.Printf("  Global Memory Cacheline Size: %d KB \n", d.GlobalMemCachelineSize()/1024)
	// fmt.Printf("  Global Memory Size: %d MB \n", d.GlobalMemSize()/(1024*1024))
	// fmt.Printf("  Half FP Config: %s \n", d.HalfFPConfig())
	// fmt.Printf("  Host Unified Memory: %+v \n", d.HostUnifiedMemory())
	// fmt.Printf("  Image Support: %+v \n", d.ImageSupport())
	// fmt.Printf("  Image2D Max Dimensions: %d x %d \n", d.Image2DMaxWidth(), d.Image2DMaxHeight())
	// fmt.Printf("  Image3D Max Dimensions: %d x %d x %d \n", d.Image3DMaxWidth(), d.Image3DMaxHeight(), d.Image3DMaxDepth())
	// fmt.Printf("  Little Endian: %+v \n", d.EndianLittle())
	// fmt.Printf("  Local Mem Size Size: %d KB \n", d.LocalMemSize()/1024)
	// fmt.Printf("  Local Mem Type: %s \n", d.LocalMemType())
	// fmt.Printf("  Max Clock Frequency: %d \n", d.MaxClockFrequency())
	// fmt.Printf("  Max Compute Units: %d \n", d.MaxComputeUnits())
	// fmt.Printf("  Max Constant Args: %d \n", d.MaxConstantArgs())
	// fmt.Printf("  Max Constant Buffer Size: %d KB \n", d.MaxConstantBufferSize()/1024)
	// fmt.Printf("  Max Mem Alloc Size: %d KB \n", d.MaxMemAllocSize()/1024)
	// fmt.Printf("  Max Parameter Size: %d \n", d.MaxParameterSize())
	// fmt.Printf("  Max Read-Image Args: %d \n", d.MaxReadImageArgs())
	// fmt.Printf("  Max Samplers: %d \n", d.MaxSamplers())
	// fmt.Printf("  Max Work Group Size: %d \n", d.MaxWorkGroupSize())
	// fmt.Printf("  Preferred Work Group Size: %d \n", opencl.ClPrefWGSz)
	// fmt.Printf("  Max Work Item Dimensions: %d \n", d.MaxWorkItemDimensions())
	// fmt.Printf("  Max Work Item Sizes: %d \n", d.MaxWorkItemSizes())
	// fmt.Printf("  Max Write-Image Args: %d \n", d.MaxWriteImageArgs())
	// fmt.Printf("  Memory Base Address Alignment: %d \n", d.MemBaseAddrAlign())
	// fmt.Printf("  Native Vector Width Char: %d \n", d.NativeVectorWidthChar())
	// fmt.Printf("  Native Vector Width Short: %d \n", d.NativeVectorWidthShort())
	// fmt.Printf("  Native Vector Width Int: %d \n", d.NativeVectorWidthInt())
	// fmt.Printf("  Native Vector Width Long: %d \n", d.NativeVectorWidthLong())
	// fmt.Printf("  Native Vector Width Float: %d \n", d.NativeVectorWidthFloat())
	// fmt.Printf("  Native Vector Width Double: %d \n", d.NativeVectorWidthDouble())
	// fmt.Printf("  Native Vector Width Half: %d \n", d.NativeVectorWidthHalf())
	// fmt.Printf("  OpenCL C Version: %s \n", d.OpenCLCVersion())
	// fmt.Printf("  Profile: %s \n", d.Profile())
	// fmt.Printf("  Profiling Timer Resolution: %d \n", d.ProfilingTimerResolution())
	// fmt.Printf("  Vendor: %s \n", d.Vendor())
	// fmt.Printf("  Version: %s \n", d.Version())

	queue := opencl.ClCmdQueue

	fmt.Println("Generating input data...")
	inputs := make([][]float32, NComponents)
	var size [3]int

	if plan1d.IsRealHerm {
		if plan1d.IsForw {
			size = [3]int{plan1d.RowDim, 1, 1}
			for i := 0; i < NComponents; i++ {
				inputs[i] = make([]float32, size[0])
				for j := 0; j < plan1d.RowDim; j++ {
					inputs[i][j] = float32(j)
					fmt.Printf("( %f ) ", inputs[i][j])

					// inputs[i][2*j] = float32(0.1)
					// inputs[i][2*j+1] = 0
					// fmt.Printf(" (%f, %f) ", inputs[i][2*j], inputs[i][2*j+1])

				}
			}
		} else {
			size = [3]int{2 * (1 + int(math.Floor(float64(plan1d.RowDim/2)))), 1, 1}
			for i := 0; i < NComponents; i++ {
				inputs[i] = make([]float32, size[0])
				for j := 0; j < 1+int(math.Floor(float64(plan1d.RowDim/2))); j++ {
					inputs[i][2*j] = 0.1
					if j == 0 {
						inputs[i][2*j+1] = 0
					} else {
						inputs[i][2*j+1] = 0.1 * float32(math.Pow(-1, float64(j)))
					}

					fmt.Printf(" (%f, %f) ", inputs[i][2*j], inputs[i][2*j+1])
				}
			}
		}
	} else {
		size = [3]int{2 * plan1d.RowDim, 1, 1}
		for i := 0; i < NComponents; i++ {
			inputs[i] = make([]float32, size[0])
			for j := 0; j < plan1d.RowDim; j++ {
				inputs[i][2*j] = float32(0.1)
				inputs[i][2*j+1] = float32(0.1)
				fmt.Printf(" (%f, %f) ", inputs[i][2*j], inputs[i][2*j+1])
			}
		}
	}

	//size = [3]int{10, 1, 1}
	//inputs[0] = []float32{9.00, 0, -2.1180, -1.5388, 0.1180, 0.3633, 0.1180, -0.3633, -2.1180, 1.5388}
	//size = [3]int{6, 1, 1}
	//inputs[0] = []float32{9.00, 0, -2.1180, -1.5388, 0.1180, 0.3633}

	fmt.Println("Done. Transferring input data from CPU to GPU...")
	cpuArray1d := data.SliceFromArray(inputs, size)
	gpuBuffer := opencl.Buffer(NComponents, size)
	//outBuffer := opencl.Buffer(NComponents, [3]int{2 * N, 1, 1})

	data.Copy(gpuBuffer, cpuArray1d)

	fmt.Println("Waiting for data transfer to complete...")
	queue.Finish()
	fmt.Println("Input data transfer completed.")

	Parse1D(gpuBuffer, plan1d)
	//panic("\n 1D finished...")

	//bufX, errC := context.CreateEmptyBuffer(cl.MemWriteOnly, c.RowDim*2*int(unsafe.Sizeof(X[0])))

	//context := opencl.ClCtx
	// var OutputBuf *data.Slice
	// if !plan1d.IsBlusteinsReq {
	// 	if plan1d.IsForw {
	// 		OutputBuf = opencl.Buffer(int(*Flag_comp), [3]int{2 * plan1d.FinalN, 1, 1})
	// 	} else {
	// 		if plan1d.IsRealHerm {
	// 			OutputBuf = opencl.Buffer(int(*Flag_comp), [3]int{plan1d.FinalN, 1, 1})
	// 		} else {
	// 			OutputBuf = opencl.Buffer(int(*Flag_comp), [3]int{2 * plan1d.FinalN, 1, 1})
	// 		}
	// 	}
	// } else {
	// 	if plan1d.IsForw {
	// 		OutputBuf = opencl.Buffer(int(*Flag_comp), [3]int{2 * plan1d.FinalN, 1, 1})
	// 	}
	// }

	fmt.Printf("\n Executing Forward 2D FFT. Printing input array \n")
	plan2d := FftPlan2DValue{false, false, true, true, false, int(*Flag_size), 2, 1, int(*Flag_size), 2}
	inputs2d := make([][]float32, NComponents)
	var size2d [3]int

	if plan2d.IsForw && plan2d.IsRealHerm {
		size2d = [3]int{plan2d.RowDim * plan2d.ColDim, 1, 1}
		for i := 0; i < NComponents; i++ {
			inputs2d[i] = make([]float32, size2d[0])
			for j := 0; j < plan2d.ColDim; j++ {
				for k := 0; k < plan2d.RowDim; k++ {
					inputs2d[i][j*plan2d.RowDim+k] = float32(j*plan2d.RowDim+k) * float32(0.1) //float32(0.1)
					fmt.Printf("( %f ) ", inputs2d[i][j*plan2d.RowDim+k])
				}
				fmt.Printf("\n")
			}
		}
	}

	if !plan2d.IsForw && plan2d.IsRealHerm {
		fmt.Printf("\n Printing default value of hermitian matrix of 17*2 of FFT of 0,1,2,...,16;17,18,...,33")
		size2d = [3]int{2 * int(1+plan2d.RowDim/2) * plan2d.ColDim, 1, 1}
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
		inputs2d[0] = []float32{56.099998, 0.000001, -1.700004, 9.094196, -1.700002, 4.388192, -1.699986, 2.745589,
			-1.699995, 1.864812, -1.700000, 1.283777, -1.699999, 0.846498, -1.700004, 0.483691,
			-1.700001, 0.157525,
			-28.900002, 0.000000, 0.000001, -0.000000, 0.000001, 0.000006, -0.000006, 0.000001,
			-0.000002, 0.000000, -0.000000, 0.000002, -0.000001, 0.000001, 0.000002, 0.000000,
			0.000000, 0.000000}
	}

	if !plan2d.IsRealHerm {
		size2d = [3]int{2 * plan2d.RowDim * plan2d.ColDim, 1, 1}
		inputs2d[0] = []float32{56.099998, 0.000001, -1.700004, 9.094196, -1.700002, 4.388192, -1.699986, 2.745589,
			-1.699995, 1.864812, -1.700000, 1.283777, -1.699999, 0.846498, -1.700004, 0.483691,
			-1.700001, 0.157525, -1.700001, -0.157525, -1.700004, -0.483691, -1.699999, -0.846498,
			-1.700000, -1.283777, -1.699995, -1.864812, -1.699986, -2.745589, -1.700002, -4.388192,
			-1.700004, -9.094196,
			-28.900002, 0.000000, 0.000001, -0.000000, 0.000001, 0.000006, -0.000006, 0.000001,
			-0.000002, 0.000000, -0.000000, 0.000002, -0.000001, 0.000001, 0.000002, 0.000000,
			0.000000, 0.000000, 0.000000, 0.000000, 0.000002, 0.000000, -0.000001, -0.000001,
			-0.000000, -0.000002, -0.000002, 0.000000, -0.000006, -0.000001, 0.000001, -0.000006,
			0.000001, 0.000000}
		// for i := 0; i < NComponents; i++ {
		// 	inputs2d[i] = make([]float32, size2d[0])
		// 	for j := 0; j < plan2d.ColDim; j++ {
		// 		for k := 0; k < plan2d.RowDim; k++ {
		// 			inputs2d[i][2*(j*plan2d.RowDim+k)] = float32(j*plan2d.RowDim + k)   //float32(0.1)
		// 			inputs2d[i][2*(j*plan2d.RowDim+k)+1] = float32(j*plan2d.RowDim + k) //float32(0.1)
		// 			fmt.Printf(" (%f, %f) ", inputs2d[i][2*(j*plan2d.RowDim+k)], inputs2d[i][2*(j*plan2d.RowDim+k)+1])
		// 		}
		// 		fmt.Printf("\n")
		// 	}
		// }
	}

	fmt.Println("\n Done. Transferring input data from CPU to GPU...")
	cpuArray2d := data.SliceFromArray(inputs2d, size2d)
	gpu2dBuffer := opencl.Buffer(NComponents, size2d)
	//outBuffer := opencl.Buffer(NComponents, [3]int{2 * N, 1, 1})

	data.Copy(gpu2dBuffer, cpuArray2d)

	fmt.Println("Waiting for data transfer to complete...")
	queue.Finish()
	fmt.Println("Input data transfer completed.")

	// Final2Buf := opencl.Buffer(NComponents, [3]int{2 * plan2d.RowDim * plan2d.ColDim, 1, 1})

	// for i := 0; i < NComponents; i++ {
	// 	TempOutBuf := opencl.Buffer(NComponents, [3]int{2 * plan2d.RowDim * plan2d.ColDim, 1, 1})
	// 	TranpoBuf := opencl.Buffer(NComponents, [3]int{2 * plan2d.RowDim * plan2d.ColDim, 1, 1})
	// 	defer opencl.Recycle(TempOutBuf)
	// 	for j := 0; j < plan2d.ColDim; j++ {
	// 		SmallBuff := opencl.Buffer(NComponents, [3]int{2 * plan2d.RowDim, 1, 1})
	// 		MemOffsetCpyFloat32(SmallBuff.DevPtr(0), gpu2dBuffer.DevPtr(0), 0, 2*j*plan2d.RowDim, 2*plan2d.RowDim)
	// 		FftBuff := opencl.Buffer(NComponents, [3]int{2 * plan2d.RowDim, 1, 1})
	// 		TempPlan := FftPlanValue{plan2d.IsForw, plan2d.IsRealHerm, true, false, plan2d.RowDim, 1, 1, plan2d.RowDim}
	// 		FFT1D(FftBuff, SmallBuff, TempPlan)
	// 		MemOffsetCpyFloat32(TempOutBuf.DevPtr(0), FftBuff.DevPtr(0), 2*j*plan2d.RowDim, 0, 2*plan2d.RowDim)
	// 		opencl.Recycle(SmallBuff)
	// 		opencl.Recycle(FftBuff)
	// 	}
	// 	fmt.Printf("\n Implementing Transpose \n")
	// 	opencl.ComplexMatrixTranspose(TranpoBuf, TempOutBuf, 0, plan2d.RowDim, plan2d.ColDim)

	// 	SecOutBuf := opencl.Buffer(NComponents, [3]int{2 * plan2d.RowDim * plan2d.ColDim, 1, 1})
	// 	//TranpoBuf := opencl.Buffer(NComponents, [3]int{2 * plan2d.RowDim * plan2d.ColDim, 1, 1})
	// 	defer opencl.Recycle(SecOutBuf)
	// 	for j := 0; j < plan2d.RowDim; j++ {
	// 		SmallBuff := opencl.Buffer(NComponents, [3]int{2 * plan2d.ColDim, 1, 1})
	// 		MemOffsetCpyFloat32(SmallBuff.DevPtr(0), TranpoBuf.DevPtr(0), 0, 2*j*plan2d.ColDim, 2*plan2d.ColDim)
	// 		FftBuff := opencl.Buffer(NComponents, [3]int{2 * plan2d.ColDim, 1, 1})
	// 		TempPlan := FftPlanValue{plan2d.IsForw, false, true, false, plan2d.ColDim, 1, 1, plan2d.ColDim}
	// 		FFT1D(FftBuff, SmallBuff, TempPlan)
	// 		MemOffsetCpyFloat32(SecOutBuf.DevPtr(0), FftBuff.DevPtr(0), 2*j*plan2d.ColDim, 0, 2*plan2d.ColDim)
	// 		opencl.Recycle(SmallBuff)
	// 		opencl.Recycle(FftBuff)
	// 	}
	// 	fmt.Printf("\n Implementing Transpose \n")
	// 	opencl.ComplexMatrixTranspose(Final2Buf, SecOutBuf, 0, plan2d.RowDim, plan2d.ColDim)
	// 	fmt.Printf("\n Printing 2d individual output array \n")
	// 	PrintArray(Final2Buf, plan2d.RowDim*plan2d.ColDim)
	// 	opencl.Recycle(TempOutBuf)
	// 	opencl.Recycle(SecOutBuf)
	// 	opencl.Recycle(TranpoBuf)
	// }

	Parse2D(gpu2dBuffer, plan2d)

	fmt.Printf("\n Checking FFT......\n")
	opencl.ReleaseAndClean()
}
