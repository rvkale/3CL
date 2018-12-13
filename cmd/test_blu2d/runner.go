package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"strconv"
	"time"
	"unsafe"

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
	"github.com/mumax/3cl/cmd/test_blu2d/purefft"
	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/opencl"
	//"github.com/mumax/3cl/opencl/cl"
)

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

func blusteinCase(length int) (int, int) {
	switch {
	case length > 128000000:
		return length, -1
	case length > 115200000:
		return findLength(length, "new_length_lookup_10.json"), 1 //Hardcoded filenames
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
		return findLength(length, "new_length_lookup_1.json"), 1
	case length < 2:
		return length, -2
	}
	return length, -3
}

var (
	Flag_gpu   = flag.Int("gpu", 0, "Specify GPU")
	Flag_size  = flag.Int("length", 359, "length of data to test")
	Flag_print = flag.Bool("print", false, "Print out result")
	Flag_comp  = flag.Int("components", 1, "Number of components to test")
	//Flag_conj  = flag.Bool("conjugate", false, "Conjugate B in multiplication")
)

//FftPlanValue Structure to identify the plan for processing
type FftPlanValue struct {
	IsForw, IsRealHerm, IsSinglePreci bool
	RowDim, ColDim, DepthDim          int
}

//BoolGen to generate random plan values
func BoolGen() bool {
	var src = rand.NewSource(time.Now().UnixNano())
	var r = rand.New(src)
	return r.Int63n(2) == 0
}

//Big2SmallSlice To obtain small slice from big slice
func Big2SmallSlice(OutSlice, InSlice *data.Slice, size_info [3]int, OffsetRow, OffsetCol int) {

	ptrs := make([]unsafe.Pointer, 1)
	ptrs[0] = unsafe.Pointer(InSlice.DevPtr(0))
	OutSlice = data.SliceFromPtrs(size_info, (int8)(InSlice.MemType()), ptrs)
}

//Small2BigSlice To obtain big slice from small slice
func Small2BigSlice(OutSlice, InSlice *data.Slice, size_info [3]int, OffsetRow, OffsetCol int) {
	ptrs := make([]unsafe.Pointer, size_info[0])
	for i := range ptrs {
		//util.Argument(size_info[0] == size_info[0]*size_info[1]*size_info[2])
		ptrs[i] = unsafe.Pointer(InSlice.DevPtr(0))
	}
	OutSlice = data.SliceFromPtrs(size_info, (int8)(InSlice.MemType()), ptrs)

}

//Parse1DInput to identify the details about the FFT
func Parse1DInput(InpBuf *data.Slice, class interface{}) *data.Slice {
	fmt.Printf("\n Parsing the input to execute appropriate FFT function...\n")
	c, ok := class.(FftPlanValue)
	if !ok {
		panic("\n Wrong Input given... Terminating...\n")
	}

	//bufX, errC := context.CreateEmptyBuffer(cl.MemWriteOnly, c.RowDim*2*int(unsafe.Sizeof(X[0])))

	var IsBlusteinsReq bool

	//Check if new length is valid and if Blusteins Algorithm is required

	FinalN, Desci := blusteinCase(c.RowDim) //Desci is decision variable

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
			FinalN = c.RowDim
			IsBlusteinsReq = false
		} else {
			FinalN = 2 * FinalN
			IsBlusteinsReq = true
		}
		fmt.Printf("\n Adjusting length and finding FFT using Blusteins Algorithm with Legnth = %d...\n", FinalN)
	}
	//context := opencl.ClCtx
	queue := opencl.ClCmdQueue

	if !IsBlusteinsReq {
		if c.IsForw {
			if c.IsRealHerm {
				OpBuf := data.NewSlice(int(*Flag_comp), [3]int{2 * (1 + c.RowDim/2), 1, 1})
				fmt.Printf("\n Executing Forward Real FFT without Bluestein's ...\n")
				purefft.Clfft1D(InpBuf, OpBuf, c.RowDim, c.IsRealHerm, c.IsForw, c.IsSinglePreci)

				fmt.Printf("\n Running Hermitian to Full \n")
				FinalBuf := data.NewSlice(int(*Flag_comp), [3]int{2 * (1 + c.RowDim), 1, 1})
				opencl.Hermitian2Full(FinalBuf, OpBuf)
				fmt.Printf("\n Finished running Hermitian to Full. Final Output is ready \n")
				return FinalBuf
			} else {
				FinalBuf := data.NewSlice(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
				fmt.Printf("\n Executing Forward Complex FFT without Bluestein's ...\n")
				purefft.Clfft1D(InpBuf, FinalBuf, c.RowDim, c.IsRealHerm, c.IsForw, c.IsSinglePreci)
				return FinalBuf
			}

		} else {
			if c.IsRealHerm {
				FinalBuf := data.NewSlice(int(*Flag_comp), [3]int{c.RowDim, 1, 1})
				fmt.Printf("\n Executing Inverse Hermitian FFT without Bluestein's...\n")
				purefft.Clfft1D(InpBuf, FinalBuf, c.RowDim, c.IsRealHerm, c.IsForw, c.IsSinglePreci)
				return FinalBuf

			} else {
				FinalBuf := data.NewSlice(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
				fmt.Printf("\n Executing Inverse Complex FFT without Bluestein's...\n")
				purefft.Clfft1D(InpBuf, FinalBuf, c.RowDim, c.IsRealHerm, c.IsForw, c.IsSinglePreci)
				return FinalBuf
			}
		}

	} else {
		if c.IsForw {
			if c.IsRealHerm {
				fmt.Printf("\n Executing Forward Real FFT with Bluesteins...\n")
				GpuBuff := opencl.Buffer(int(*Flag_comp), [3]int{c.RowDim, 1, 1})
				defer opencl.Recycle(GpuBuff)
				data.Copy(GpuBuff, InpBuf)
				fmt.Println("Waiting for data transfer to complete...")
				queue.Finish()
				fmt.Println("Input data transfer completed.")
				PartABuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
				defer opencl.Recycle(PartABuf)

				fmt.Printf("\n Converting Real Part A to complex for multiplication with twiddle factor\n")
				opencl.PackComplexArray(PartABuf, GpuBuff, c.RowDim, 0, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				PartAProcBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				defer opencl.Recycle(PartAProcBuf)
				fmt.Printf("\n Processing Part A with the twiddle factor \n")
				opencl.PartAProcess(PartAProcBuf, PartABuf, c.RowDim, FinalN, 1, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				PartBBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				defer opencl.Recycle(PartBBuf)
				fmt.Printf("\n Generating part B for Bluesteins")
				opencl.PartBTwidFac(PartBBuf, c.RowDim, FinalN, 1, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Executing forward FFT for Part A \n")
				PartAFFT := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				defer opencl.Recycle(PartAFFT)
				purefft.Clfft1D(PartAProcBuf, PartAFFT, FinalN, false, true, c.IsSinglePreci)
				fmt.Printf("\n Executing forward FFT for Part B \n")
				PartBFFT := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				defer opencl.Recycle(PartBFFT)
				purefft.Clfft1D(PartBBuf, PartBFFT, FinalN, false, true, c.IsSinglePreci)

				fmt.Printf("\n Multiplying Part A and Part B FFT \n")
				MulBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				defer opencl.Recycle(MulBuff)
				opencl.ComplexArrayMul(MulBuff, PartAFFT, PartBFFT, 0, FinalN, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Taking inverse FFT of multiplication \n")
				InvBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				defer opencl.Recycle(InvBuff)
				purefft.Clfft1D(MulBuff, InvBuff, FinalN, false, false, c.IsSinglePreci)

				fmt.Printf("\n Preparing final twiddle factor")
				FinTwid := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				opencl.FinalMulTwid(FinTwid, c.RowDim, FinalN, 1, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Multiplying with Final Twiddle Factor")
				FinTempBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				opencl.ComplexArrayMul(FinTempBuff, FinTwid, InvBuff, 0, FinalN, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")
				return FinTempBuff

			} else {
				//fmt.Printf("\n Executing Forward Complex FFT ...\n")
				fmt.Printf("\n Executing Forward Complex FFT with Bluesteins...\n")
				PartABuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
				defer opencl.Recycle(PartABuf)
				data.Copy(PartABuf, InpBuf)
				fmt.Println("Waiting for data transfer to complete...")
				queue.Finish()
				fmt.Println("Input data transfer completed.")
				// PartABuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
				// defer opencl.Recycle(PartABuf)

				// fmt.Printf("\n Converting Real Part A to complex for multiplication with twiddle factor\n")
				// opencl.PackComplexArray(PartABuf, GpuBuff, c.RowDim, 0, 0)
				// fmt.Println("\n Waiting for kernel to finish execution...")
				// queue.Finish()
				// fmt.Println("\n Execution finished.")

				PartAProcBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				defer opencl.Recycle(PartAProcBuf)
				fmt.Printf("\n Processing Part A with the twiddle factor \n")
				opencl.PartAProcess(PartAProcBuf, PartABuf, c.RowDim, FinalN, 1, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				PartBBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				defer opencl.Recycle(PartBBuf)
				fmt.Printf("\n Generating part B for Bluesteins")
				opencl.PartBTwidFac(PartBBuf, c.RowDim, FinalN, 1, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Executing forward FFT for Part A \n")
				PartAFFT := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				defer opencl.Recycle(PartAFFT)
				purefft.Clfft1D(PartAProcBuf, PartAFFT, FinalN, false, true, c.IsSinglePreci)
				fmt.Printf("\n Executing forward FFT for Part B \n")
				PartBFFT := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				defer opencl.Recycle(PartBFFT)
				purefft.Clfft1D(PartBBuf, PartBFFT, FinalN, false, true, c.IsSinglePreci)

				fmt.Printf("\n Multiplying Part A and Part B FFT \n")
				MulBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				defer opencl.Recycle(MulBuff)
				opencl.ComplexArrayMul(MulBuff, PartAFFT, PartBFFT, 0, FinalN, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Taking inverse FFT of multiplication \n")
				InvBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				defer opencl.Recycle(InvBuff)
				purefft.Clfft1D(MulBuff, InvBuff, FinalN, false, false, c.IsSinglePreci)

				fmt.Printf("\n Preparing final twiddle factor")
				FinTwid := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				opencl.FinalMulTwid(FinTwid, c.RowDim, FinalN, 1, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Multiplying with Final Twiddle Factor")
				FinTempBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				opencl.ComplexArrayMul(FinTempBuff, FinTwid, InvBuff, 0, FinalN, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")
				return FinTempBuff

			}
		} else {
			if c.IsRealHerm {
				//fmt.Printf("\n Executing Inverse Hermitian FFT ...\n")
				fmt.Printf("\n Executing Inverse Real FFT with Bluesteins...\n")
				GpuBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * (1 + c.RowDim/2), 1, 1})
				defer opencl.Recycle(GpuBuff)
				data.Copy(GpuBuff, InpBuf)
				fmt.Println("Waiting for data transfer to complete...")
				queue.Finish()
				fmt.Println("Input data transfer completed.")
				PartABuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
				defer opencl.Recycle(PartABuf)

				fmt.Printf("\n Converting Hermitian to Full Complex of Part A to complex for multiplication with twiddle factor\n")
				opencl.Hermitian2Full(PartABuf, GpuBuff)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				PartAProcBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				defer opencl.Recycle(PartAProcBuf)
				fmt.Printf("\n Processing Part A with the twiddle factor \n")
				opencl.PartAProcess(PartAProcBuf, PartABuf, c.RowDim, FinalN, -1, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				PartBBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				defer opencl.Recycle(PartBBuf)
				fmt.Printf("\n Generating part B for Bluesteins")
				opencl.PartBTwidFac(PartBBuf, c.RowDim, FinalN, -1, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Executing forward FFT for Part A \n")
				PartAFFT := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				defer opencl.Recycle(PartAFFT)
				purefft.Clfft1D(PartAProcBuf, PartAFFT, FinalN, false, true, c.IsSinglePreci)
				fmt.Printf("\n Executing forward FFT for Part B \n")
				PartBFFT := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				defer opencl.Recycle(PartBFFT)
				purefft.Clfft1D(PartBBuf, PartBFFT, FinalN, false, true, c.IsSinglePreci)
				fmt.Printf("\n Multiplying Part A and Part B FFT \n")
				MulBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				defer opencl.Recycle(MulBuff)
				opencl.ComplexArrayMul(MulBuff, PartAFFT, PartBFFT, 0, FinalN, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Taking inverse FFT of multiplication \n")
				InvBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				defer opencl.Recycle(InvBuff)
				purefft.Clfft1D(MulBuff, InvBuff, FinalN, false, false, c.IsSinglePreci)

				fmt.Printf("\n Preparing final twiddle factor")
				FinTwid := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				opencl.FinalMulTwid(FinTwid, c.RowDim, FinalN, -1, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Multiplying with Final Twiddle Factor")
				FinTempBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				opencl.ComplexArrayMul(FinTempBuff, FinTwid, InvBuff, 0, FinalN, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")
				return FinTempBuff

			} else {
				fmt.Printf("\n Executing Inverse Complex FFT ...\n")
				fmt.Printf("\n Executing Forward Complex FFT with Bluesteins...\n")
				PartABuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
				defer opencl.Recycle(PartABuf)
				data.Copy(PartABuf, InpBuf)
				fmt.Println("Waiting for data transfer to complete...")
				queue.Finish()
				fmt.Println("Input data transfer completed.")
				// PartABuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
				// defer opencl.Recycle(PartABuf)

				// fmt.Printf("\n Converting Real Part A to complex for multiplication with twiddle factor\n")
				// opencl.PackComplexArray(PartABuf, GpuBuff, c.RowDim, 0, 0)
				// fmt.Println("\n Waiting for kernel to finish execution...")
				// queue.Finish()
				// fmt.Println("\n Execution finished.")

				PartAProcBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				defer opencl.Recycle(PartAProcBuf)
				fmt.Printf("\n Processing Part A with the twiddle factor \n")
				opencl.PartAProcess(PartAProcBuf, PartABuf, c.RowDim, FinalN, -1, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				PartBBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				defer opencl.Recycle(PartBBuf)
				fmt.Printf("\n Generating part B for Bluesteins")
				opencl.PartBTwidFac(PartBBuf, c.RowDim, FinalN, -1, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Executing forward FFT for Part A \n")
				PartAFFT := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				defer opencl.Recycle(PartAFFT)
				purefft.Clfft1D(PartAProcBuf, PartAFFT, FinalN, false, true, c.IsSinglePreci)
				fmt.Printf("\n Executing forward FFT for Part B \n")
				PartBFFT := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				defer opencl.Recycle(PartBFFT)
				purefft.Clfft1D(PartBBuf, PartBFFT, FinalN, false, true, c.IsSinglePreci)

				fmt.Printf("\n Multiplying Part A and Part B FFT \n")
				MulBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				defer opencl.Recycle(MulBuff)
				opencl.ComplexArrayMul(MulBuff, PartAFFT, PartBFFT, 0, FinalN, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Taking inverse FFT of multiplication \n")
				InvBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				defer opencl.Recycle(InvBuff)
				purefft.Clfft1D(MulBuff, InvBuff, FinalN, false, false, c.IsSinglePreci)

				fmt.Printf("\n Preparing final twiddle factor")
				FinTwid := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				opencl.FinalMulTwid(FinTwid, c.RowDim, FinalN, -1, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Multiplying with Final Twiddle Factor")
				FinTempBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * FinalN, 1, 1})
				opencl.ComplexArrayMul(FinTempBuff, FinTwid, InvBuff, 0, FinalN, 0)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")
				return FinTempBuff

			}
		}
	}
}

//Parse2DInput Function to calculate FFT of 2D data. Either directly or Bluesteins)
func Parse2DInput(InpBuf *data.Slice, class interface{}) {

	fmt.Printf("\n Parsing the input to execute appropriate FFT function...\n")
	c, ok := class.(FftPlanValue)
	if !ok {
		panic("\n Wrong Input given... Terminating...\n")
	}
	//context := opencl.ClCtx
	queue := opencl.ClCmdQueue

	fmt.Printf("\n Calculating 2D FFT for the given input \n")
	//var IsBlusteinRow, IsBlusteinCol bool
	ValRow, DecideRow := blusteinCase(c.RowDim)
	ValCol, DecideCol := blusteinCase(c.ColDim)
	if (DecideRow == 1) && (DecideCol == 1) {
		if (ValRow == 0) && (ValCol == 0) {
			fmt.Printf("\n No need to execute Blusteins for any dimension. Executing CLFFT directly \n")
			if c.IsForw {
				if c.IsRealHerm {
					OpBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * (1 + c.ColDim*c.RowDim/2), 1, 1})
					fmt.Printf("\n Executing Forward Real FFT without Bluestein's ...\n")
					purefft.Clfft2D(InpBuf, OpBuf, c.RowDim, c.ColDim, c.IsRealHerm, c.IsForw, c.IsSinglePreci)

					fmt.Printf("\n Running Hermitian to Full \n")
					FinalBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * (1 + c.ColDim*c.RowDim), 1, 1})
					opencl.Hermitian2Full(FinalBuf, OpBuf)
					fmt.Printf("\n Finished running Hermitian to Full. Final Output is ready \n")
				} else {
					FinalBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.ColDim * c.RowDim, 1, 1})
					fmt.Printf("\n Executing Forward Complex FFT without Bluestein's ...\n")
					purefft.Clfft2D(InpBuf, FinalBuf, c.RowDim, c.ColDim, c.IsRealHerm, c.IsForw, c.IsSinglePreci)
				}

			} else {
				if c.IsRealHerm {
					FinalBuf := opencl.Buffer(int(*Flag_comp), [3]int{c.ColDim * c.RowDim, 1, 1})
					fmt.Printf("\n Executing Inverse Hermitian FFT without Bluestein's...\n")
					purefft.Clfft2D(InpBuf, FinalBuf, c.RowDim, c.ColDim, c.IsRealHerm, c.IsForw, c.IsSinglePreci)

				} else {
					FinalBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim * c.RowDim, 1, 1})
					fmt.Printf("\n Executing Inverse Complex FFT without Bluestein's...\n")
					purefft.Clfft2D(InpBuf, FinalBuf, c.RowDim, c.ColDim, c.IsRealHerm, c.IsForw, c.IsSinglePreci)
				}
			}
		}
	}

	if c.IsForw {
		if c.IsRealHerm {
			fmt.Printf("\n Executing Forward 2D FFT using Blusteins \n")
			MainDestBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * ValRow * ValCol, 1, 1})
			TempDestBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * ValCol, 1, 1})
			for i := 0; i < int(*Flag_comp); i++ {
				Big2SmallSlice(TempDestBuff, InpBuf, [3]int{2 * ValCol, 1, 1}, i, 1)
				RowBuf := Parse1DInput(TempDestBuff, class)
				Small2BigSlice(MainDestBuff, RowBuf, [3]int{2 * ValCol * ValRow, 1, 1}, i, 1)
			}

			TransDestBuff := data.NewSlice(int(*Flag_comp), [3]int{2 * ValRow * ValCol, 1, 1})
			opencl.ComplexMatrixTranspose(TransDestBuff, MainDestBuff, 0, ValRow, ValCol)
			fmt.Println("Waiting for kernel to finish execution...")
			queue.Finish()
			fmt.Println("Execution finished.")

			//OpBuf := data.NewSlice(int(*Flag_comp), [3]int{2 * (1 + c.ColDim*c.RowDim/2), 1, 1})
			MainFftBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * ValRow * ValCol, 1, 1})
			TempFftBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * ValCol, 1, 1})
			for i := 0; i < int(*Flag_comp); i++ {
				Big2SmallSlice(TempFftBuff, TransDestBuff, [3]int{2 * ValCol, 1, 1}, i, 1)
				RowBuf := Parse1DInput(TempFftBuff, class)
				Small2BigSlice(MainFftBuff, RowBuf, [3]int{2 * ValCol * ValRow, 1, 1}, i, 1)
			}
			// fmt.Printf("\n Executing Forward Real FFT without Bluestein's ...\n")
			// purefft.Clfft2D(InpBuf, OpBuf, c.RowDim, c.ColDim, c.IsRealHerm, c.IsForw, c.IsSinglePreci)

			FinalTranBuff := data.NewSlice(int(*Flag_comp), [3]int{2 * ValRow * ValCol, 1, 1})
			opencl.ComplexMatrixTranspose(FinalTranBuff, MainFftBuff, 0, ValRow, ValCol)
			fmt.Println("Waiting for kernel to finish execution...")
			queue.Finish()
			fmt.Println("Execution finished.")
			fmt.Printf("\n Finished running Hermitian to Full. Final Output is ready \n")
		} else {
			fmt.Printf("\n Executing Forward 2D FFT using Blusteins \n")
			MainDestBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * ValRow * ValCol, 1, 1})
			TempDestBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * ValCol, 1, 1})
			for i := 0; i < int(*Flag_comp); i++ {
				Big2SmallSlice(TempDestBuff, InpBuf, [3]int{2 * ValCol, 1, 1}, i, 1)
				RowBuf := Parse1DInput(TempDestBuff, class)
				Small2BigSlice(MainDestBuff, RowBuf, [3]int{2 * ValCol * ValRow, 1, 1}, i, 1)
			}

			TransDestBuff := data.NewSlice(int(*Flag_comp), [3]int{2 * ValRow * ValCol, 1, 1})
			opencl.ComplexMatrixTranspose(TransDestBuff, MainDestBuff, 0, ValRow, ValCol)
			fmt.Println("Waiting for kernel to finish execution...")
			queue.Finish()
			fmt.Println("Execution finished.")

			//OpBuf := data.NewSlice(int(*Flag_comp), [3]int{2 * (1 + c.ColDim*c.RowDim/2), 1, 1})
			MainFftBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * ValRow * ValCol, 1, 1})
			TempFftBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * ValCol, 1, 1})
			for i := 0; i < int(*Flag_comp); i++ {
				Big2SmallSlice(TempFftBuff, TransDestBuff, [3]int{2 * ValCol, 1, 1}, i, 1)
				RowBuf := Parse1DInput(TempFftBuff, class)
				Small2BigSlice(MainFftBuff, RowBuf, [3]int{2 * ValCol * ValRow, 1, 1}, i, 1)
			}
			// fmt.Printf("\n Executing Forward Real FFT without Bluestein's ...\n")
			// purefft.Clfft2D(InpBuf, OpBuf, c.RowDim, c.ColDim, c.IsRealHerm, c.IsForw, c.IsSinglePreci)

			FinalTranBuff := data.NewSlice(int(*Flag_comp), [3]int{2 * ValRow * ValCol, 1, 1})
			opencl.ComplexMatrixTranspose(FinalTranBuff, MainFftBuff, 0, ValRow, ValCol)
			fmt.Println("Waiting for kernel to finish execution...")
			queue.Finish()
			fmt.Println("Execution finished.")
			fmt.Printf("\n Finished running Hermitian to Full. Final Output is ready \n")
			// FinalBuf := data.NewSlice(int(*Flag_comp), [3]int{2 * c.ColDim * c.RowDim, 1, 1})
			// fmt.Printf("\n Executing Forward Complex FFT without Bluestein's ...\n")
			// purefft.Clfft2D(InpBuf, FinalBuf, c.RowDim, c.ColDim, c.IsRealHerm, c.IsForw, c.IsSinglePreci)
		}

	} else {
		if c.IsRealHerm {
			fmt.Printf("\n Executing Forward 2D FFT using Blusteins \n")
			MainDestBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * ValRow * ValCol, 1, 1})
			TempDestBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * ValCol, 1, 1})
			for i := 0; i < int(*Flag_comp); i++ {
				Big2SmallSlice(TempDestBuff, InpBuf, [3]int{2 * ValCol, 1, 1}, i, 1)
				RowBuf := Parse1DInput(TempDestBuff, class)
				Small2BigSlice(MainDestBuff, RowBuf, [3]int{2 * ValCol * ValRow, 1, 1}, i, 1)
			}

			TransDestBuff := data.NewSlice(int(*Flag_comp), [3]int{2 * ValRow * ValCol, 1, 1})
			opencl.ComplexMatrixTranspose(TransDestBuff, MainDestBuff, 0, ValRow, ValCol)
			fmt.Println("Waiting for kernel to finish execution...")
			queue.Finish()
			fmt.Println("Execution finished.")

			//OpBuf := data.NewSlice(int(*Flag_comp), [3]int{2 * (1 + c.ColDim*c.RowDim/2), 1, 1})
			MainFftBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * ValRow * ValCol, 1, 1})
			TempFftBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * ValCol, 1, 1})
			for i := 0; i < int(*Flag_comp); i++ {
				Big2SmallSlice(TempFftBuff, TransDestBuff, [3]int{2 * ValCol, 1, 1}, i, 1)
				RowBuf := Parse1DInput(TempFftBuff, class)
				Small2BigSlice(MainFftBuff, RowBuf, [3]int{2 * ValCol * ValRow, 1, 1}, i, 1)
			}
			// fmt.Printf("\n Executing Forward Real FFT without Bluestein's ...\n")
			// purefft.Clfft2D(InpBuf, OpBuf, c.RowDim, c.ColDim, c.IsRealHerm, c.IsForw, c.IsSinglePreci)

			FinalTranBuff := data.NewSlice(int(*Flag_comp), [3]int{2 * ValRow * ValCol, 1, 1})
			opencl.ComplexMatrixTranspose(FinalTranBuff, MainFftBuff, 0, ValRow, ValCol)
			fmt.Println("Waiting for kernel to finish execution...")
			queue.Finish()
			fmt.Println("Execution finished.")
			fmt.Printf("\n Finished running Hermitian to Full. Final Output is ready \n")

		} else {
			fmt.Printf("\n Executing Forward 2D FFT using Blusteins \n")
			MainDestBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * ValRow * ValCol, 1, 1})
			TempDestBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * ValCol, 1, 1})
			for i := 0; i < int(*Flag_comp); i++ {
				Big2SmallSlice(TempDestBuff, InpBuf, [3]int{2 * ValCol, 1, 1}, i, 1)
				RowBuf := Parse1DInput(TempDestBuff, class)
				Small2BigSlice(MainDestBuff, RowBuf, [3]int{2 * ValCol * ValRow, 1, 1}, i, 1)
			}

			TransDestBuff := data.NewSlice(int(*Flag_comp), [3]int{2 * ValRow * ValCol, 1, 1})
			opencl.ComplexMatrixTranspose(TransDestBuff, MainDestBuff, 0, ValRow, ValCol)
			fmt.Println("Waiting for kernel to finish execution...")
			queue.Finish()
			fmt.Println("Execution finished.")

			//OpBuf := data.NewSlice(int(*Flag_comp), [3]int{2 * (1 + c.ColDim*c.RowDim/2), 1, 1})
			MainFftBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * ValRow * ValCol, 1, 1})
			TempFftBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * ValCol, 1, 1})
			for i := 0; i < int(*Flag_comp); i++ {
				Big2SmallSlice(TempFftBuff, TransDestBuff, [3]int{2 * ValCol, 1, 1}, i, 1)
				RowBuf := Parse1DInput(TempFftBuff, class)
				Small2BigSlice(MainFftBuff, RowBuf, [3]int{2 * ValCol * ValRow, 1, 1}, i, 1)
			}
			// fmt.Printf("\n Executing Forward Real FFT without Bluestein's ...\n")
			// purefft.Clfft2D(InpBuf, OpBuf, c.RowDim, c.ColDim, c.IsRealHerm, c.IsForw, c.IsSinglePreci)

			FinalTranBuff := data.NewSlice(int(*Flag_comp), [3]int{2 * ValRow * ValCol, 1, 1})
			opencl.ComplexMatrixTranspose(FinalTranBuff, MainFftBuff, 0, ValRow, ValCol)
			fmt.Println("Waiting for kernel to finish execution...")
			queue.Finish()
			fmt.Println("Execution finished.")
			fmt.Printf("\n Finished running Hermitian to Full. Final Output is ready \n")
		}
	}

}
func main() {

	flag.Parse()
	//var Desci int //Descision variable
	N := int(*Flag_size)
	//ReqComponents := int(*Flag_comp)
	opencl.Init(*Flag_gpu)
	rand.Seed(time.Now().Unix())
	X := make([]float32, 2*N)
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

	plan1d := FftPlanValue{BoolGen(), BoolGen(), BoolGen(), N, 1, 1}

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

	queue := opencl.ClCmdQueue

	/* Zero Padding for adjusting the length if necessary*/
	fmt.Println("Generating input data...")
	dataSize := N / 2
	dataSize += 1
	size := [3]int{2 * dataSize, 1, 1}
	inputs := make([][]float32, NComponents)
	for i := 0; i < NComponents; i++ {
		inputs[i] = make([]float32, size[0])
		for j := 0; j < len(inputs[i]); j++ {
			inputs[i][j] = rand.Float32()
		}
	}
	fmt.Println("Done. Transferring input data from CPU to GPU...")
	cpuArray1d := data.SliceFromArray(inputs, size)
	gpuBuffer := opencl.Buffer(NComponents, size)
	// outBuffer := opencl.Buffer(NComponents, [3]int{2 * N, 1, 1})
	// outArray := data.NewSlice(NComponents, [3]int{2 * N, 1, 1})

	data.Copy(gpuBuffer, cpuArray1d)

	fmt.Println("Waiting for data transfer to complete...")
	queue.Finish()
	fmt.Println("Input data transfer completed.")

	Parse1DInput(cpuArray1d, plan1d)

	size2d := [3]int{8, 2, 1}
	inputs2d := make([][]float32, NComponents)
	for i := 0; i < NComponents; i++ {
		inputs2d[i] = make([]float32, size2d[0]*size2d[1])
		for j := 0; j < size2d[0]; j++ {
			for k := 0; k < size2d[1]; k++ {
				x := rand.Float32()
				y := rand.Float32()
				idx := int(2 * (k + j*size2d[0]))
				inputs2d[i][idx] = x
				inputs2d[i][idx+1] = y
				fmt.Printf("(%f, %f) ", x, y)
			}
			fmt.Printf("\n")
		}
	}

	cpuArray2d := data.SliceFromArray(inputs2d, size2d)

	plan2d := FftPlanValue{BoolGen(), BoolGen(), BoolGen(), N, 1, 1}
	Parse2DInput(cpuArray2d, plan2d)

	fmt.Printf("\n Checking FFT......\n")

}
