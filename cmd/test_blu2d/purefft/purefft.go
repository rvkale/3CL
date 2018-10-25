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
	"unsafe"
	//"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/opencl/cl"

)

// Function to find 1D ClFFT of the given []float32

func FindClfft(InputData []float32,N int,Direction string) []float32 {
	context := opencl.ClCtx
	queue := opencl.ClCmdQueue

	fmt.Printf("\n Performing fft on an one dimensional array of size N = %d \n", N)


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
	fmt.Printf("Finished tests on clFFT\n")
	fftPlanHandle.Destroy()
	return InputData
}


//Function to create 2D FFT array
func Find2DClfft(InputData []float32, N0 int, N1 int, Direction string) []float32 {
	context := opencl.ClCtx
	queue := opencl.ClCmdQueue

	fmt.Printf("\n Performing fft on an one dimensional array of size N = %d \n", N0)


	/* Prepare OpenCL memory objects and place data inside them. */
	bufX, errC := context.CreateEmptyBuffer(cl.MemWriteOnly, N0*N1*2*int(unsafe.Sizeof(InputData[0])))
	bufOut, errCO := context.CreateEmptyBuffer(cl.MemReadOnly, N0*N1*2*int(unsafe.Sizeof(InputData[0])))

	if errC != nil {
		fmt.Printf("unable to create input buffer: %+v \n ", errC)
	}
	if errCO != nil {
		fmt.Printf("unable to create output buffer: %+v \n ", errCO)
	}

	if _, err := queue.EnqueueWriteBufferFloat32(bufX, true, 0, InputData[:], nil); err != nil {
		fmt.Printf("failed to write data into buffer \n")
	}

	flag := cl.CLFFTDim2D
	fftPlanHandle, errF := cl.NewCLFFTPlan(context, flag, []int{N0, N1})
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

	fmt.Printf("\n")

	fmt.Printf("Finished tests on clFFT\n")
	fftPlanHandle.Destroy()
	return InputData

}
