package purefft

import (
	"fmt"
	//"math"
	//"math/cmplx"
	//"encoding/json"
	//"os"
	//"io/ioutil"
	//"strconv"
	//"flag"
	//"math/rand"

	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/opencl/cl"
)

//Clfft1D Function to find 1D ClFFT of the given []float32..
//func Clfft1D(bufX, bufOut *cl.MemObject, InputData []float32, N int, IsReal, IsForw, IsSinglePrecision bool) []float32 {
//func Clfft1D(bufOut, bufX *cl.MemObject, N int, IsReal, IsForw, IsSinglePrecision bool) {
func Clfft1D(InBuf, OutBuf *data.Slice, N, ScaleLength int, IsReal, IsForw, IsSinglePrecision, IsScalingReq bool) {

	context := opencl.ClCtx
	queue := opencl.ClCmdQueue

	fmt.Printf("\n Performing fft on an one dimensional array of size N = %d \n", N)

	/* Prepare OpenCL memory objects and place data inside them. */
	//bufX, errC := context.CreateEmptyBuffer(cl.MemWriteOnly, N*2*int(unsafe.Sizeof(InputData[0])))
	//bufOut, errCO := context.CreateEmptyBuffer(cl.MemReadOnly, N*2*int(unsafe.Sizeof(InputData[0])))

	// if errC != nil {
	// 	fmt.Printf("unable to create input buffer: %+v \n ", errC)
	// }
	// if errCO != nil {
	// 	fmt.Printf("unable to create output buffer: %+v \n ", errCO)
	// }

	// if _, err := queue.EnqueueWriteBufferFloat32(bufX, true, 0, InputData[:], nil); err != nil {
	// 	fmt.Printf("failed to write data into buffer \n")
	// }

	tmpPtr := InBuf.DevPtr(0)
	srcMemObj := *(*cl.MemObject)(tmpPtr)
	tmpPtr = OutBuf.DevPtr(0)
	dstMemObj := *(*cl.MemObject)(tmpPtr)

	flag := cl.CLFFTDim1D
	fftPlanHandle, errF := cl.NewCLFFTPlan(context, flag, []int{N}) //Don't change this to 2*N
	if errF != nil {
		fmt.Printf("unable to create new fft plan \n")
	}

	if IsSinglePrecision == true {
		errF = fftPlanHandle.SetSinglePrecision()
		if errF != nil {
			fmt.Printf("unable to set fft precision \n")
		}
	} else {
		errF = fftPlanHandle.SetDoublePrecision()
		if errF != nil {
			fmt.Printf("unable to set fft precision \n")
		}
	}

	ArrLayout := cl.NewArrayLayout()

	if IsForw == true {
		if IsReal == false {
			ArrLayout.SetInputLayout(cl.CLFFTLayoutComplexInterleaved)
			ArrLayout.SetOutputLayout(cl.CLFFTLayoutComplexInterleaved)
		} else {
			ArrLayout.SetInputLayout(cl.CLFFTLayoutReal)
			ArrLayout.SetOutputLayout(cl.CLFFTLayoutHermitianInterleaved)
		}
	} else {
		if IsReal == false {
			ArrLayout.SetInputLayout(cl.CLFFTLayoutComplexInterleaved)
			ArrLayout.SetOutputLayout(cl.CLFFTLayoutComplexInterleaved)
		} else {
			ArrLayout.SetInputLayout(cl.CLFFTLayoutHermitianInterleaved)
			ArrLayout.SetOutputLayout(cl.CLFFTLayoutReal)
		}
	}
	// ArrLayout.SetInputLayout(cl.CLFFTLayoutComplexInterleaved)
	// ArrLayout.SetOutputLayout(cl.CLFFTLayoutComplexInterleaved)

	errD := fftPlanHandle.SetLayouts(ArrLayout)
	if errD != nil {
		fmt.Printf("unable to set Array Layout \n")
	}
	errF = fftPlanHandle.SetResultOutOfPlace()
	if errF != nil {
		fmt.Printf("unable to set fft result location \n")
	}

	if IsScalingReq {
		errF = fftPlanHandle.SetScale(cl.ClFFTDirectionBackward, float32(float64(1.0)/float64(ScaleLength*N)))
		if errF != nil {
			fmt.Printf("unable to set fft result scaling \n")
		}
	}

	/* Bake the plan. */
	errF = fftPlanHandle.BakePlanSimple([]*cl.CommandQueue{queue})
	if errF != nil {
		fmt.Printf("unable to bake fft plan: %+v \n", errF)
	}

	/* Execute the plan. */
	if IsForw {
		_, errF = fftPlanHandle.EnqueueForwardTransform([]*cl.CommandQueue{queue}, nil, []*cl.MemObject{&srcMemObj}, []*cl.MemObject{&dstMemObj}, nil)
		if errF != nil {
			fmt.Printf("\n Unable to enqueue forward transform: %+v \n", errF)
		} else {
			fmt.Printf("\n Executing forward transform...\n")
		}
	} else {
		_, errF = fftPlanHandle.EnqueueBackwardTransform([]*cl.CommandQueue{queue}, nil, []*cl.MemObject{&srcMemObj}, []*cl.MemObject{&dstMemObj}, nil)
		if errF != nil {
			fmt.Printf("Unable to enqueue inverse transform: %+v \n", errF)
		} else {
			fmt.Printf("\n Executing inverse transform... \n ")
		}
	}

	errF = queue.Flush()
	if errF != nil {
		fmt.Printf("unable to flush queue: %+v \n", errF)
	}

	/* Fetch results of calculations. */
	// _, errF = queue.EnqueueReadBufferFloat32(bufOut, true, 0, InputData, nil)
	// errF = queue.Flush()
	// if errF != nil {
	// 	fmt.Printf("unable to read output buffer: %+v \n", errF)
	// }
	fmt.Printf("Finished tests on clFFT\n")
	fftPlanHandle.Destroy()
	//return InputData
}

//Clfft2D Function to find 2D FFT
func Clfft2D(InBuf, OutBuf *data.Slice, N0 int, N1 int, IsReal, IsForw, IsSinglePrecision bool) {

	context := opencl.ClCtx
	queue := opencl.ClCmdQueue

	fmt.Printf("\n Performing fft on an one dimensional array of size N = %d \n", N0)

	// /* Prepare OpenCL memory objects and place data inside them. */
	// bufX, errC := context.CreateEmptyBuffer(cl.MemWriteOnly, N0*N1*2*int(unsafe.Sizeof(InputData[0])))
	// bufOut, errCO := context.CreateEmptyBuffer(cl.MemReadOnly, N0*N1*2*int(unsafe.Sizeof(InputData[0])))

	// if errC != nil {
	// 	fmt.Printf("unable to create input buffer: %+v \n ", errC)
	// }
	// if errCO != nil {
	// 	fmt.Printf("unable to create output buffer: %+v \n ", errCO)
	// }

	// if _, err := queue.EnqueueWriteBufferFloat32(bufX, true, 0, InputData[:], nil); err != nil {
	// 	fmt.Printf("failed to write data into buffer \n")
	// }

	tmpPtr := InBuf.DevPtr(0)
	srcMemObj := *(*cl.MemObject)(tmpPtr)
	tmpPtr = OutBuf.DevPtr(0)
	dstMemObj := *(*cl.MemObject)(tmpPtr)

	flag := cl.CLFFTDim2D
	fftPlanHandle, errF := cl.NewCLFFTPlan(context, flag, []int{N0, N1})
	if errF != nil {
		fmt.Printf("unable to create new fft plan \n")
	}

	if IsSinglePrecision == true {
		errF = fftPlanHandle.SetSinglePrecision()
		if errF != nil {
			fmt.Printf("unable to set fft precision \n")
		}
	} else {
		errF = fftPlanHandle.SetDoublePrecision()
		if errF != nil {
			fmt.Printf("unable to set fft precision \n")
		}
	}

	ArrLayout := cl.NewArrayLayout()
	if IsForw == true {
		if IsReal == false {
			ArrLayout.SetInputLayout(cl.CLFFTLayoutComplexInterleaved)
			ArrLayout.SetOutputLayout(cl.CLFFTLayoutComplexInterleaved)
		} else {
			ArrLayout.SetInputLayout(cl.CLFFTLayoutReal)
			ArrLayout.SetOutputLayout(cl.CLFFTLayoutHermitianInterleaved)
		}
	} else {
		if IsReal == false {
			ArrLayout.SetInputLayout(cl.CLFFTLayoutComplexInterleaved)
			ArrLayout.SetOutputLayout(cl.CLFFTLayoutComplexInterleaved)
		} else {
			ArrLayout.SetInputLayout(cl.CLFFTLayoutHermitianInterleaved)
			ArrLayout.SetOutputLayout(cl.CLFFTLayoutReal)
		}
	}

	errD := fftPlanHandle.SetLayouts(ArrLayout)
	if errD != nil {
		fmt.Printf("unable to set Array Layout \n")
	}

	// ArrLayout.SetInputLayout(cl.CLFFTLayoutComplexInterleaved)
	// ArrLayout.SetOutputLayout(cl.CLFFTLayoutComplexInterleaved)
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
	if IsForw {
		_, errF = fftPlanHandle.EnqueueForwardTransform([]*cl.CommandQueue{queue}, nil, []*cl.MemObject{&srcMemObj}, []*cl.MemObject{&dstMemObj}, nil)
		if errF != nil {
			fmt.Printf("unable to enqueue transform: %+v \n", errF)
		} else {
			fmt.Printf("\n Executing forward transform...\n")
		}
	} else {
		_, errF = fftPlanHandle.EnqueueBackwardTransform([]*cl.CommandQueue{queue}, nil, []*cl.MemObject{&srcMemObj}, []*cl.MemObject{&dstMemObj}, nil)
		if errF != nil {
			fmt.Printf("unable to enqueue transform: %+v \n", errF)
		} else {
			fmt.Printf("\n Executing inverse transform... \n ")
		}
	}

	errF = queue.Flush()
	if errF != nil {
		fmt.Printf("unable to flush queue: %+v \n", errF)
	}

	// /* Fetch results of calculations. */
	// _, errF = queue.EnqueueReadBufferFloat32(bufOut, true, 0, InputData, nil)
	// errF = queue.Flush()
	// if errF != nil {
	// 	fmt.Printf("unable to read output buffer: %+v \n", errF)
	// }
	fmt.Printf("Finished tests on clFFT\n")
	fftPlanHandle.Destroy()
	//return InputData
}
