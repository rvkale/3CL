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


// Stores the arguments for compress_cmplx kernel invocation
type compress_cmplx_args_t struct{
	 arg_dst unsafe.Pointer
	 arg_src unsafe.Pointer
	 arg_count int
	 arg_iOffset int
	 arg_oOffset int
	 argptr [5]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for compress_cmplx kernel invocation
var compress_cmplx_args compress_cmplx_args_t

func init(){
	// OpenCL driver kernel call wants pointers to arguments, set them up once.
	 compress_cmplx_args.argptr[0] = unsafe.Pointer(&compress_cmplx_args.arg_dst)
	 compress_cmplx_args.argptr[1] = unsafe.Pointer(&compress_cmplx_args.arg_src)
	 compress_cmplx_args.argptr[2] = unsafe.Pointer(&compress_cmplx_args.arg_count)
	 compress_cmplx_args.argptr[3] = unsafe.Pointer(&compress_cmplx_args.arg_iOffset)
	 compress_cmplx_args.argptr[4] = unsafe.Pointer(&compress_cmplx_args.arg_oOffset)
	 }

// Wrapper for compress_cmplx OpenCL kernel, asynchronous.
func k_compress_cmplx_async ( dst unsafe.Pointer, src unsafe.Pointer, count int, iOffset int, oOffset int,  cfg *config, events []*cl.Event) *cl.Event {
	if Synchronous{ // debug
		ClCmdQueue.Finish()
		timer.Start("compress_cmplx")
	}

	compress_cmplx_args.Lock()
	defer compress_cmplx_args.Unlock()

	 compress_cmplx_args.arg_dst = dst
	 compress_cmplx_args.arg_src = src
	 compress_cmplx_args.arg_count = count
	 compress_cmplx_args.arg_iOffset = iOffset
	 compress_cmplx_args.arg_oOffset = oOffset
	

	SetKernelArgWrapper("compress_cmplx", 0, dst)
	SetKernelArgWrapper("compress_cmplx", 1, src)
	SetKernelArgWrapper("compress_cmplx", 2, count)
	SetKernelArgWrapper("compress_cmplx", 3, iOffset)
	SetKernelArgWrapper("compress_cmplx", 4, oOffset)
	

//	args := compress_cmplx_args.argptr[:]
	event := LaunchKernel("compress_cmplx", cfg.Grid, cfg.Block, events)

	if Synchronous{ // debug
		ClCmdQueue.Finish()
		timer.Stop("compress_cmplx")
	}

	return event
}

