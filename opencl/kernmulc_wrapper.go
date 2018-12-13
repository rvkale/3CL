package opencl

/*
 THIS FILE IS AUTO-GENERATED BY OCL2GO.
 EDITING IS FUTILE.
*/

import(
	"unsafe"
	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/timer"
	"sync"
)


// Stores the arguments for kernmulC kernel invocation
type kernmulC_args_t struct{
	 arg_fftM unsafe.Pointer
	 arg_fftK unsafe.Pointer
	 arg_Nx int
	 arg_Ny int
	 argptr [4]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for kernmulC kernel invocation
var kernmulC_args kernmulC_args_t

func init(){
	// OpenCL driver kernel call wants pointers to arguments, set them up once.
	 kernmulC_args.argptr[0] = unsafe.Pointer(&kernmulC_args.arg_fftM)
	 kernmulC_args.argptr[1] = unsafe.Pointer(&kernmulC_args.arg_fftK)
	 kernmulC_args.argptr[2] = unsafe.Pointer(&kernmulC_args.arg_Nx)
	 kernmulC_args.argptr[3] = unsafe.Pointer(&kernmulC_args.arg_Ny)
	 }

// Wrapper for kernmulC OpenCL kernel, asynchronous.
func k_kernmulC_async ( fftM unsafe.Pointer, fftK unsafe.Pointer, Nx int, Ny int,  cfg *config, events []*cl.Event) *cl.Event {
	if Synchronous{ // debug
		ClCmdQueue.Finish()
		timer.Start("kernmulC")
	}

	kernmulC_args.Lock()
	defer kernmulC_args.Unlock()

	 kernmulC_args.arg_fftM = fftM
	 kernmulC_args.arg_fftK = fftK
	 kernmulC_args.arg_Nx = Nx
	 kernmulC_args.arg_Ny = Ny
	

	SetKernelArgWrapper("kernmulC", 0, fftM)
	SetKernelArgWrapper("kernmulC", 1, fftK)
	SetKernelArgWrapper("kernmulC", 2, Nx)
	SetKernelArgWrapper("kernmulC", 3, Ny)
	

//	args := kernmulC_args.argptr[:]
	event := LaunchKernel("kernmulC", cfg.Grid, cfg.Block, events)

	if Synchronous{ // debug
		ClCmdQueue.Finish()
		timer.Stop("kernmulC")
	}

	return event
}

