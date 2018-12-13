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


// Stores the arguments for multwiddlefact kernel invocation
type multwiddlefact_args_t struct{
	 arg_dataOut unsafe.Pointer
	 arg_origlength int
	 arg_extenlength int
	 arg_fftdirec int
	 arg_offset int
	 argptr [5]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for multwiddlefact kernel invocation
var multwiddlefact_args multwiddlefact_args_t

func init(){
	// OpenCL driver kernel call wants pointers to arguments, set them up once.
	 multwiddlefact_args.argptr[0] = unsafe.Pointer(&multwiddlefact_args.arg_dataOut)
	 multwiddlefact_args.argptr[1] = unsafe.Pointer(&multwiddlefact_args.arg_origlength)
	 multwiddlefact_args.argptr[2] = unsafe.Pointer(&multwiddlefact_args.arg_extenlength)
	 multwiddlefact_args.argptr[3] = unsafe.Pointer(&multwiddlefact_args.arg_fftdirec)
	 multwiddlefact_args.argptr[4] = unsafe.Pointer(&multwiddlefact_args.arg_offset)
	 }

// Wrapper for multwiddlefact OpenCL kernel, asynchronous.
func k_multwiddlefact_async ( dataOut unsafe.Pointer, origlength int, extenlength int, fftdirec int, offset int,  cfg *config, events []*cl.Event) *cl.Event {
	if Synchronous{ // debug
		ClCmdQueue.Finish()
		timer.Start("multwiddlefact")
	}

	multwiddlefact_args.Lock()
	defer multwiddlefact_args.Unlock()

	 multwiddlefact_args.arg_dataOut = dataOut
	 multwiddlefact_args.arg_origlength = origlength
	 multwiddlefact_args.arg_extenlength = extenlength
	 multwiddlefact_args.arg_fftdirec = fftdirec
	 multwiddlefact_args.arg_offset = offset
	

	SetKernelArgWrapper("multwiddlefact", 0, dataOut)
	SetKernelArgWrapper("multwiddlefact", 1, origlength)
	SetKernelArgWrapper("multwiddlefact", 2, extenlength)
	SetKernelArgWrapper("multwiddlefact", 3, fftdirec)
	SetKernelArgWrapper("multwiddlefact", 4, offset)
	

//	args := multwiddlefact_args.argptr[:]
	event := LaunchKernel("multwiddlefact", cfg.Grid, cfg.Block, events)

	if Synchronous{ // debug
		ClCmdQueue.Finish()
		timer.Stop("multwiddlefact")
	}

	return event
}
