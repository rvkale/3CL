package opencl

import (
	"fmt"
	"log"

	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/timer"
)

// 3D single-precision real-to-complex FFT plan.
type fft3DC2RPlan struct {
	fftplan
	// handle *cl.OclFFTPlan
	size [3]int
}

// 3D single-precision real-to-complex FFT plan.
func newFFT3DC2R(Nx, Ny, Nz int) fft3DC2RPlan {

	// handle, err := cl.NewCLFFTPlan(ClCtx, cl.CLFFTDim3D, []int{Nx, Ny, Nz}) // new xyz swap
	effort, err := cl.CreateDefaultOclFFTPlan()
	if err != nil {
		log.Printf("Unable to create fft3dc2r plan \n")
	}

	effort.SetDevice(ClDevice)
	effort.SetContext(ClCtx)
	effort.SetQueue(ClCmdQueue)
	effort.SetProgram()
	effort.SetDimension(cl.CLFFTDim3D)

	// arrLayout := cl.NewArrayLayout()
	// arrLayout.SetInputLayout(cl.CLFFTLayoutHermitianInterleaved)
	// arrLayout.SetOutputLayout(cl.CLFFTLayoutReal)
	// err = handle.SetLayouts(arrLayout)
	// if err != nil {
	// 	log.Printf("Unable to set buffer layouts of fft3dc2r plan \n")
	// }

	effort.SetLayout(cl.CLFFTLayoutHermitianInterleaved)
	InStrideArr := [3]int{1, Nx/2 + 1, Ny * (Nx/2 + 1)}
	// err = handle.SetInStride(InStrideArr)
	// if err != nil {
	// 	log.Printf("Unable to set input stride of fft3dc2r plan \n")
	// }
	effort.SetInStride(InStrideArr)

	// err = handle.SetResultOutOfPlace()
	// if err != nil {
	// 	log.Printf("Unable to set placeness of fft3dc2r result \n")
	// }
	effort.SetResultLocation(cl.ClFFTResultLocationOutOfPlace)

	// err = handle.SetSinglePrecision()
	// if err != nil {
	// 	log.Printf("Unable to set precision of fft3dc2r plan \n")
	// }
	effort.SetPrecision(cl.CLFFTPrecisionSingle)

	effort.SetDirection(cl.ClFFTDirectionBackward)

	effort.SetLengths([3]int{Nx, Ny, Nz})

	// err = handle.SetResultNoTranspose()
	// if err != nil {
	// 	log.Printf("Unable to set transpose of fft3dc2r result \n")
	// }

	// err = handle.SetScale(cl.ClFFTDirectionBackward, float32(1.0))
	// if err != nil {
	// 	log.Printf("Unable to set scaling factor of fft3dc2r result \n")
	// }

	effort.Bake()

	// err = handle.BakePlanSimple([]*cl.CommandQueue{ClCmdQueue})
	// if err != nil {
	// 	log.Printf("Unable to bake fft3dc2r plan \n")
	// }

	return fft3DC2RPlan{fftplan{effort}, [3]int{Nx, Ny, Nz}}
}

// Execute the FFT plan, asynchronous.
// src and dst are 3D arrays stored 1D arrays.
func (p *fft3DC2RPlan) ExecAsync(src, dst *data.Slice) ([]*cl.Event, error) {
	if Synchronous {
		ClCmdQueue.Finish()
		timer.Start("fft")
	}
	oksrclen := p.InputLenFloats()
	if src.Len() != oksrclen {
		panic(fmt.Errorf("fft size mismatch: expecting src len %v, got %v", oksrclen, src.Len()))
	}
	okdstlen := p.OutputLenFloats()
	if dst.Len() != okdstlen {
		panic(fmt.Errorf("fft size mismatch: expecting dst len %v, got %v", okdstlen, dst.Len()))
	}
	tmpPtr := src.DevPtr(0)
	srcMemObj := (*cl.MemObject)(tmpPtr)
	tmpPtr = dst.DevPtr(0)
	dstMemObj := (*cl.MemObject)(tmpPtr)
	// eventsList, err := p.handle.EnqueueBackwardTransform([]*cl.CommandQueue{ClCmdQueue}, []*cl.Event{src.GetEvent(0), dst.GetEvent(0)},
	// 	[]*cl.MemObject{&srcMemObj}, []*cl.MemObject{&dstMemObj}, nil)

	p.handle.ExecTransform(dstMemObj, srcMemObj)

	if Synchronous {
		ClCmdQueue.Finish()
		timer.Stop("fft")
	}

	ev1, erre := p.handle.GetContext().CreateUserEvent()
	if erre != nil {
		panic("\n Failed to create event \n")
	}

	erre = ev1.SetUserEventStatus(cl.CommandExecStatusComplete)
	// var evelist []*Event

	evelist := []*cl.Event{ev1}
	return evelist, erre
	// return eventsList, err
	// return nil, nil
}

// 3D size of the input array.
func (p *fft3DC2RPlan) InputSizeFloats() (Nx, Ny, Nz int) {
	return p.size[X] + 2, p.size[Y], p.size[Z]
}

// 3D size of the output array.
func (p *fft3DC2RPlan) OutputSizeFloats() (Nx, Ny, Nz int) {
	return p.size[X], p.size[Y], p.size[Z]
}

// Required length of the (1D) input array.
func (p *fft3DC2RPlan) InputLenFloats() int {
	return prod3(p.InputSizeFloats())
}

// Required length of the (1D) output array.
func (p *fft3DC2RPlan) OutputLenFloats() int {
	return prod3(p.OutputSizeFloats())
}
