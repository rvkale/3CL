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


// Stores the arguments for divide kernel invocation
type divide_args_t struct{
	 arg_dst unsafe.Pointer
	 arg_a unsafe.Pointer
	 arg_b unsafe.Pointer
	 arg_N int
	 argptr [4]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for divide kernel invocation
var divide_args divide_args_t

func init(){
	// OpenCL driver kernel call wants pointers to arguments, set them up once.
	 divide_args.argptr[0] = unsafe.Pointer(&divide_args.arg_dst)
	 divide_args.argptr[1] = unsafe.Pointer(&divide_args.arg_a)
	 divide_args.argptr[2] = unsafe.Pointer(&divide_args.arg_b)
	 divide_args.argptr[3] = unsafe.Pointer(&divide_args.arg_N)
	 }

// Wrapper for divide OpenCL kernel, asynchronous.
func k_divide_async ( dst unsafe.Pointer, a unsafe.Pointer, b unsafe.Pointer, N int,  cfg *config, events []*cl.Event) *cl.Event {
	if Synchronous{ // debug
		ClCmdQueue.Finish()
		timer.Start("divide")
	}

	divide_args.Lock()
	defer divide_args.Unlock()

	 divide_args.arg_dst = dst
	 divide_args.arg_a = a
	 divide_args.arg_b = b
	 divide_args.arg_N = N
	

	SetKernelArgWrapper("divide", 0, dst)
	SetKernelArgWrapper("divide", 1, a)
	SetKernelArgWrapper("divide", 2, b)
	SetKernelArgWrapper("divide", 3, N)
	

//	args := divide_args.argptr[:]
	event := LaunchKernel("divide", cfg.Grid, cfg.Block, events)

	if Synchronous{ // debug
		ClCmdQueue.Finish()
		timer.Stop("divide")
	}

	return event
}

