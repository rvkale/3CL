/*
Package extends clFFT with Bluesteins algorithm to enable FFT of any radix
*/

package cl

import (
	"C"
	"fmt"
)
import (
	"log"
)

// var (
// 	Flag_gpu   = flag.Int("gpu", 0, "Specify GPU")
// 	Flag_size  = flag.Int("length", 17, "length of data to test")
// 	Flag_print = flag.Bool("print", false, "Print out result")
// 	Flag_comp  = flag.Int("components", 1, "Number of components to test")
// 	//Flag_conj  = flag.Bool("conjugate", false, "Conjugate B in multiplication")
// )

//////// Wrapper to allow function pointers ////////
type OclFFTFuncs struct {
	ExecFunc func(dst, src *MemObject) error
}

func (p *OclFFTFuncs) Exec(dst, src *MemObject) error {
	return p.ExecFunc(dst, src)
}

//////// Map data structure for storing kernels in OclFFTPlan struct ////////
type KernelMap map[string]*Kernel

//////// Data structure for twiddle buffers ////////
type chirpArray []int

//////// Map data structure for storing twiddle buffers for Bluesteins ////////
type forwardChirpTwiddles map[int]*chirpArray
type forwardChirpTwiddlesFFT map[int]*chirpArray

//////// Map data structure for storing twiddle buffers for Bluesteins ////////
type backwardChirpTwiddles map[int]*chirpArray
type backwardChirpTwiddlesFFT map[int]*chirpArray

//////// Radices and maximum length supported by clFFT ////////
var supported_radices = []int{17, 13, 11, 8, 7, 5, 4, 3, 2}

const maxLen int = 128000000

//FftPlanValue Structure to identify the plan for processing
type FftPlan1DValue struct {
	IsForw, IsRealHerm, IsSinglePreci, IsBlusteinsReq bool
	RowDim, ColDim, DepthDim, FinalN                  int
}

// FftPlan2DValue Structure to represent plan for FFT processing of 2D array
type FftPlan2DValue struct {
	IsForw, IsRealHerm, IsSinglePreci, IsBlusteinsReqRow, IsBlusteinsReqCol bool
	RowDim, ColDim, DepthDim, RowBluLeng, ColBluLeng                        int
}

// FftPlan3DValue Structure to represent plan for FFT processing of 3D array
type FftPlan3DValue struct {
	IsForw, IsRealHerm, IsSinglePreci, IsBlusteinsReqRow, IsBlusteinsReqCol, IsBlusteinReqDep bool
	RowDim, ColDim, DepthDim, RowBluLeng, ColBluLeng, DepBluLeng                              int
}

//////// Wrapper plan to interface with clFFT ////////
type OclFFTPlan struct {
	bake          bool
	direction     ClFFTDirection
	precision     ClFFTPrecision
	layout        ClFFTLayout
	dimension     ClFFTDim
	placeness     ClFFTResultLocation
	chirpz        [3]bool
	chirp_lengths [3]int
	fftLengths    [3]int
	batches       int
	inStride      [3]int
	outStride     [3]int
	inDist        int
	outDist       int
	clfftplans    [3]*ClFFTPlan
	buffers       [3]*MemObject
	exec_sequence []*OclFFTFuncs
	clCtx         *Context
	clDevice      *Device
	clKernels     KernelMap
	clProg        *Program
}

//////// private library functions ////////
func createOclFFTPlan() *OclFFTPlan {
	newPlan := new(OclFFTPlan)
	return newPlan
}

//determineLengthCompatibility Finding Bluesteins length
func determineLengthCompatibility(in int) bool {
	var outFlag bool
	outFlag = false
	if in < 1 || in > maxLen {
		return outFlag
	}

	result := in
	result_length := 0
	for _, rad := range supported_radices {
		result_length = result % rad
		for result_length == 0 {
			result /= rad
			result_length = result % rad
		}
	}
	if result == 1 {
		return true
	}
	return false
}

//determineChirpLength Finding Bluesteins length
func determineChirpLength(in int) (int, int) {
	outLength := -1
	if in < 1 || in > maxLen/2 {
		return outLength, -3
	}

	outLength = 2 * in
	for determineLengthCompatibility(outLength) == false {
		outLength++
	}

	if outLength == 2*in {
		return 0, 1
	}

	return outLength, 1

}

// func determineLengthCompatibility(in int) bool {
// 	var outFlag bool
// 	outFlag = false
// 	if in < 1 || in > maxLen {
// 		return outFlag
// 	}

// 	result := in
// 	result_length := 0
// 	for _, rad := range supported_radices {
// 		result_length = result % rad
// 		for result_length == 0 {
// 			result /= rad
// 			result_length = result % rad
// 		}
// 	}
// 	if result == 1 {
// 		return true
// 	}
// 	return false
// }

// func determineChirpLength(in int) int {
// 	outLength := -1
// 	if in < 1 || in > maxLen/2 {
// 		return outLength
// 	}

// 	outLength = 2 * in
// 	for determineLengthCompatibility(outLength) == false {
// 		outLength += 1
// 	}
// 	return outLength
// }

func (p *OclFFTPlan) setChirp() {
	for ind, dimLen := range p.fftLengths {
		if dimLen > 1 {
			chkFlag := determineLengthCompatibility(dimLen)
			if chkFlag {
				p.chirpz[ind] = false
				p.chirp_lengths[ind] = -1
			} else {
				p.chirpz[ind] = true
				p.chirp_lengths[ind], _ = determineChirpLength(dimLen)
			}
		}
	}
}

//////// public library functions ////////
func CreateDefaultOclFFTPlan() (*OclFFTPlan, error) {
	newPlan := createOclFFTPlan()
	newPlan.SetContext(nil)
	return newPlan, nil
}

func (p *OclFFTPlan) GetContext() *Context {
	return p.clCtx
}

func (p *OclFFTPlan) GetDevice() *Device {
	return p.clDevice
}

func (p *OclFFTPlan) GetProgram() *Program {
	return p.clProg
}

func (p *OclFFTPlan) GetKernel(in string) *Kernel {
	return p.clKernels[in]
}

func (p *OclFFTPlan) GetDirection() ClFFTDirection {
	return p.direction
}

func (p *OclFFTPlan) GetPrecision() ClFFTPrecision {
	return p.precision
}

func (p *OclFFTPlan) GetLayout() ClFFTLayout {
	return p.layout
}

func (p *OclFFTPlan) GetDimension() ClFFTDim {
	return p.dimension
}

func (p *OclFFTPlan) GetResultLocation() ClFFTResultLocation {
	return p.placeness
}

func (p *OclFFTPlan) GetLengths() [3]int {
	return p.fftLengths
}

func (p *OclFFTPlan) GetInStride() [3]int {
	return p.inStride
}

func (p *OclFFTPlan) GetOutStride() [3]int {
	return p.outStride
}

func (p *OclFFTPlan) GetInDistance() int {
	return p.inDist
}

func (p *OclFFTPlan) GetOutDistance() int {
	return p.outDist
}

func (p *OclFFTPlan) GetBatchCount() int {
	return p.batches
}

func (p *OclFFTPlan) SetContext(in *Context) {
	if p.clCtx == nil || p.clCtx != in {
		p.clCtx = in
		p.bake = false
	}
}

func (p *OclFFTPlan) SetDevice(in *Device) {
	if p.clDevice == nil || p.clDevice != in {
		p.clDevice = in
		p.bake = false
	}
}

func (p *OclFFTPlan) setProgram() error {
	if p.clCtx == nil || p.clDevice == nil {
		return toError(nil)
	}
	var err error
	p.clProg, err = createOclFFTProgram(p.clCtx)
	if err != nil {
		fmt.Printf("CreateProgramWithSource failed: %+v \n", err)
		return err
	}
	if err = p.clProg.BuildProgram([]*Device{p.clDevice}, "-cl-std=CL1.2 -cl-fp32-correctly-rounded-divide-sqrt -cl-kernel-arg-info"); err != nil {
		fmt.Printf("BuildProgram failed: %+v \n", err)
		return err
	}
	for _, kernname := range KernelNames {
		p.clKernels[kernname], err = p.clProg.CreateKernel(kernname)
		if err != nil {
			fmt.Printf("CreateKernel failed: %+v \n", err)
			return err
		}
	}
	return toError(nil)
}

func (p *OclFFTPlan) SetDirection(in ClFFTDirection) {
	if p.direction != in {
		p.direction = in
		p.bake = false
	}
}

func (p *OclFFTPlan) SetPrecision(in ClFFTPrecision) {
	if p.precision != in {
		p.precision = in
		p.bake = false
	}
}

func (p *OclFFTPlan) SetLayout(in ClFFTLayout) {
	if p.layout != in {
		p.layout = in
		p.bake = false
	}
}

func (p *OclFFTPlan) SetDimension(in ClFFTDim) {
	if p.dimension != in {
		p.dimension = in
		p.bake = false
	}
}

func (p *OclFFTPlan) SetResultLocation(in ClFFTResultLocation) {
	if p.placeness != in {
		p.placeness = in
		p.bake = false
	}
}

func (p *OclFFTPlan) SetLengths(in [3]int) {
	if p.fftLengths != in && in[0] > 0 && in[1] > 0 && in[2] > 0 {
		p.fftLengths = in
		p.bake = false
		p.setChirp()
	}
}

func (p *OclFFTPlan) SetInStride(in [3]int) {
	if p.inStride != in && in[0] > 0 && in[1] > 0 && in[2] > 0 {
		p.inStride = in
		p.bake = false
	}
}

func (p *OclFFTPlan) SetOutStride(in [3]int) {
	if p.outStride != in && in[0] > 0 && in[1] > 0 && in[2] > 0 {
		p.outStride = in
		p.bake = false
	}
}

func (p *OclFFTPlan) SetInDist(in int) {
	if p.inDist != in && in > 0 {
		p.inDist = in
		p.bake = false
	}
}

func (p *OclFFTPlan) SetOutDist(in int) {
	if p.outDist != in && in > 0 {
		p.outDist = in
		p.bake = false
	}
}

func (p *OclFFTPlan) SetBatchCount(in int) {
	if p.batches != in && in > 0 {
		p.batches = in
		p.bake = false
	}
}

func (p *OclFFTPlan) Destroy() {
}

func (p *OclFFTPlan) Bake() error {
	p.bake = true
	return toError(nil)
}

func (p *OclFFTPlan) ExecTransform(dst, src *MemObject) error {
	if p.bake != true {
		test := p.Bake()
		if test != nil {
			return toError(test)
		}
	}

	for _, funcs := range p.exec_sequence {
		test := funcs.Exec(dst, src)
		if test != nil {
			return toError(test)
		}
	}

	return toError(nil)
}

func OclFFTTearDown() error {
	return toError(nil)
}

// var KernelNames = []string{"cmplx_mul", "compress_cmplx", "cmplx_transpose", "finaltwiddlefact", "hermitian2full", "multwiddlefact", "pack_cmplx"}

// func createOclFFTProgram(ctx *Context) (*Program, error) {
// 	return ctx.CreateProgramWithSource([]string{oclFFTKernelSrc})
// }

////////////////////////     Required Kernel Functions //////////////////////////

//config kernel launch configuration
type config struct {
	Grid, Block []int
}

func divUp(x, y int) int {
	return ((x - 1) / y) + 1
}

//PackComplexArray Convert real array to full complex array
func PackComplexArray(dst, src *MemObject, cnt, iOff, oOff int, pl *OclFFTPlan) {
	var queue *CommandQueue
	var cfg = &config{Grid: []int{8, 1, 1}, Block: []int{1, 1, 1}}
	var tmpEventList, tmpEventList1 []*Event

	if pl.GetKernel("pack_cmplx") == nil {
		log.Panic("Kernel " + "pack_cmplx" + " does not exist!")
	}

	pl.GetKernel("pack_cmplx").SetArg(0, dst)
	pl.GetKernel("pack_cmplx").SetArg(1, src)
	pl.GetKernel("pack_cmplx").SetArg(2, cnt)
	pl.GetKernel("pack_cmplx").SetArg(3, iOff)
	pl.GetKernel("pack_cmplx").SetArg(4, oOff)

	KernEvent, err := queue.EnqueueNDRangeKernel(pl.GetKernel("pack_cmplx"), nil, cfg.Grid, cfg.Block, tmpEventList)
	if err != nil {
		log.Fatal(err)
	}
	tmpEventList1 = append(tmpEventList1, KernEvent)

	if err := WaitForEvents(tmpEventList1); err != nil {
		fmt.Printf("WaitForEvents failed in PackComplexArray: %+v \n", err)
	}

}

//Hermitian2Full Wrapper for hermitian2full OpenCL kernel, asynchronous.
func Hermitian2Full(dst, src *MemObject, sz, count int, pl *OclFFTPlan) {
	var queue *CommandQueue
	var cfg = &config{Grid: []int{8, 1, 1}, Block: []int{1, 1, 1}}
	var tmpEventList, tmpEventList1 []*Event

	//ClPrefWGSz, err := pl.GetKernel("hermitian2full").PreferredWorkGroupSizeMultiple(pl.GetDevice())

	if pl.GetKernel("hermitian2full") == nil {
		log.Panic("Kernel " + "hermitian2full" + " does not exist!")
	}

	pl.GetKernel("hermitian2full").SetArg(0, dst)
	pl.GetKernel("hermitian2full").SetArg(1, src)
	pl.GetKernel("hermitian2full").SetArg(2, sz)
	pl.GetKernel("hermitian2full").SetArg(3, count)
	pl.GetKernel("hermitian2full").SetArgUnsafe(4, cfg.Block[0]*cfg.Block[1]*cfg.Block[2]*4, nil)
	pl.GetKernel("hermitian2full").SetArgUnsafe(5, cfg.Block[0]*cfg.Block[1]*cfg.Block[2]*4, nil)

	KernEvent, err := queue.EnqueueNDRangeKernel(pl.GetKernel("hermitian2full"), nil, cfg.Grid, cfg.Block, tmpEventList)
	if err != nil {
		log.Fatal(err)
	}
	tmpEventList1 = append(tmpEventList1, KernEvent)

	if err := WaitForEvents(tmpEventList1); err != nil {
		fmt.Printf("WaitForEvents failed in hermitian2full: %+v \n", err)
	}
}

//PartAProcess To preprocess the input data and extend it
func PartAProcess(dst, src *MemObject, originalLeng, extendedLeng, fftDirection, offset int, pl *OclFFTPlan) {
	//util.Argument(dst.NComp() == src.NComp())
	var queue *CommandQueue
	var cfg *config
	//var cfg *config
	// ClPrefWGSz, err := pl.GetKernel("hermitian2full").PreferredWorkGroupSizeMultiple(pl.GetDevice())
	// ClCUnits := pl.GetDevice().MaxComputeUnits()
	bl := make([]int, 1)
	bl[0], _ = pl.GetKernel("vartwiddlefa").PreferredWorkGroupSizeMultiple(pl.GetDevice())
	gr := make([]int, 1)
	gr[0] = bl[0] * pl.GetDevice().MaxComputeUnits()
	cfg = &config{Grid: gr, Block: bl}
	//cfg := make1DConf(extendedLeng)

	var tmpEventList, tmpEventList1 []*Event

	if pl.GetKernel("vartwiddlefa") == nil {
		log.Panic("Kernel " + "vartwiddlefa" + " does not exist!")
	}

	pl.GetKernel("vartwiddlefa").SetArg(0, dst)
	pl.GetKernel("vartwiddlefa").SetArg(1, src)
	pl.GetKernel("vartwiddlefa").SetArg(2, originalLeng)
	pl.GetKernel("vartwiddlefa").SetArg(3, extendedLeng)
	pl.GetKernel("vartwiddlefa").SetArg(4, fftDirection)
	pl.GetKernel("vartwiddlefa").SetArg(5, offset)

	KernEvent, _ := queue.EnqueueNDRangeKernel(pl.GetKernel("vartwiddlefa"), nil, cfg.Grid, cfg.Block, tmpEventList)
	tmpEventList1 = append(tmpEventList1, KernEvent)

	if err := WaitForEvents(tmpEventList1); err != nil {
		fmt.Printf("WaitForEvents failed in PartAProcess: %+v \n", err)
	}
}

//PartBTwidFac Calculating the twiddle factor for multiplication
func PartBTwidFac(dst *MemObject, originalLeng, extendedLeng, fftDirection, offset int, pl *OclFFTPlan) {

	var queue *CommandQueue
	var cfg *config
	//var cfg *config
	// ClPrefWGSz, err := pl.GetKernel("hermitian2full").PreferredWorkGroupSizeMultiple(pl.GetDevice())
	// ClCUnits := pl.GetDevice().MaxComputeUnits()
	bl := make([]int, 1)
	bl[0], _ = pl.GetKernel("multwiddlefact").PreferredWorkGroupSizeMultiple(pl.GetDevice())
	gr := make([]int, 1)
	gr[0] = bl[0] * pl.GetDevice().MaxComputeUnits()
	cfg = &config{Grid: gr, Block: bl}
	//cfg := make1DConf(extendedLeng)

	var tmpEventList, tmpEventList1 []*Event

	if pl.GetKernel("multwiddlefact") == nil {
		log.Panic("Kernel " + "multwiddlefact" + " does not exist!")
	}

	pl.GetKernel("multwiddlefact").SetArg(0, dst)
	pl.GetKernel("multwiddlefact").SetArg(1, originalLeng)
	pl.GetKernel("multwiddlefact").SetArg(2, extendedLeng)
	pl.GetKernel("multwiddlefact").SetArg(3, fftDirection)
	pl.GetKernel("multwiddlefact").SetArg(4, offset)

	KernEvent, _ := queue.EnqueueNDRangeKernel(pl.GetKernel("multwiddlefact"), nil, cfg.Grid, cfg.Block, tmpEventList)
	tmpEventList1 = append(tmpEventList1, KernEvent)

	if err := WaitForEvents(tmpEventList1); err != nil {
		fmt.Printf("WaitForEvents failed in PartBTwidFact: %+v \n", err)
	}
}

//FinalMulTwid Final Twiddle Factor to multiply the
func FinalMulTwid(dst *MemObject, originalLeng, extendedLeng, fftDirection, offset int, pl *OclFFTPlan) {
	var queue *CommandQueue

	var cfg *config
	bl := make([]int, 1)
	bl[0], _ = pl.GetKernel("finaltwiddlefact").PreferredWorkGroupSizeMultiple(pl.GetDevice())
	gr := make([]int, 1)
	gr[0] = bl[0] * pl.GetDevice().MaxComputeUnits()
	cfg = &config{Grid: gr, Block: bl}

	var tmpEventList, tmpEventList1 []*Event

	if pl.GetKernel("finaltwiddlefact") == nil {
		log.Panic("Kernel " + "finaltwiddlefact" + " does not exist!")
	}

	pl.GetKernel("finaltwiddlefact").SetArg(0, dst)
	pl.GetKernel("finaltwiddlefact").SetArg(1, originalLeng)
	pl.GetKernel("finaltwiddlefact").SetArg(2, extendedLeng)
	pl.GetKernel("finaltwiddlefact").SetArg(3, fftDirection)
	pl.GetKernel("finaltwiddlefact").SetArg(4, offset)

	KernEvent, _ := queue.EnqueueNDRangeKernel(pl.GetKernel("finaltwiddlefact"), nil, cfg.Grid, cfg.Block, tmpEventList)
	tmpEventList1 = append(tmpEventList1, KernEvent)

	if err := WaitForEvents(tmpEventList1); err != nil {
		fmt.Printf("WaitForEvents failed in FinalMulTwid: %+v \n", err)
	}
}

//ComplexMatrixTranspose Tranpose Complex matrix transpose
func ComplexMatrixTranspose(dst, src *MemObject, offset, width, height int, pl *OclFFTPlan) {
	var queue *CommandQueue

	var cfg *config
	//cfg := make3DConf([3]int{width, height, 1})
	bl := make([]int, 3)
	bl[0], bl[1], bl[2] = 16, 16, 1 //hardcoded values

	nx := divUp(width, 16)
	ny := divUp(height, 16)
	gr := make([]int, 3)
	gr[0], gr[1], gr[2] = (nx * bl[0]), (ny * bl[1]), (1 * bl[2])
	cfg = &config{Grid: gr, Block: bl}

	var tmpEventList, tmpEventList1 []*Event

	if pl.GetKernel("cmplx_transpose") == nil {
		log.Panic("Kernel " + "cmplx_transpose" + " does not exist!")
	}

	pl.GetKernel("cmplx_transpose").SetArg(0, dst)
	pl.GetKernel("cmplx_transpose").SetArg(1, src)
	pl.GetKernel("cmplx_transpose").SetArg(2, offset)
	pl.GetKernel("cmplx_transpose").SetArg(3, width)
	pl.GetKernel("cmplx_transpose").SetArg(4, height)

	KernEvent, _ := queue.EnqueueNDRangeKernel(pl.GetKernel("cmplx_transpose"), nil, cfg.Grid, cfg.Block, tmpEventList)
	tmpEventList1 = append(tmpEventList1, KernEvent)

	if err := WaitForEvents(tmpEventList1); err != nil {
		fmt.Printf("WaitForEvents failed in complexmatrixtranspose: %+v \n", err)
	}
}

//CompressCmplxtoReal Convert complex output of iverse hermitian to real
func CompressCmplxtoReal(dst, src *MemObject, cnt, iOff, oOff int, pl *OclFFTPlan) {

	var queue *CommandQueue
	var cfg = &config{Grid: []int{8, 1, 1}, Block: []int{1, 1, 1}}
	var tmpEventList, tmpEventList1 []*Event

	if pl.GetKernel("compress_cmplx") == nil {
		log.Panic("Kernel " + "compress_cmplx" + " does not exist!")
	}

	pl.GetKernel("compress_cmplx").SetArg(0, dst)
	pl.GetKernel("compress_cmplx").SetArg(1, src)
	pl.GetKernel("compress_cmplx").SetArg(2, cnt)
	pl.GetKernel("compress_cmplx").SetArg(3, iOff)
	pl.GetKernel("compress_cmplx").SetArg(4, oOff)

	KernEvent, _ := queue.EnqueueNDRangeKernel(pl.GetKernel("compress_cmplx"), nil, cfg.Grid, cfg.Block, tmpEventList)
	tmpEventList1 = append(tmpEventList1, KernEvent)

	if err := WaitForEvents(tmpEventList1); err != nil {
		fmt.Printf("WaitForEvents failed in packcmplxarray: %+v \n", err)
	}
}

//ComplexArrayMul Complex array mul
func ComplexArrayMul(dst, a, b *MemObject, conjB, cnt, offset int, pl *OclFFTPlan) {
	var queue *CommandQueue

	var cfg *config
	bl := make([]int, 1)
	bl[0], _ = pl.GetKernel("finaltwiddlefact").PreferredWorkGroupSizeMultiple(pl.GetDevice())
	gr := make([]int, 1)
	gr[0] = bl[0] * pl.GetDevice().MaxComputeUnits()
	cfg = &config{Grid: gr, Block: bl}

	var tmpEventList, tmpEventList1 []*Event

	if pl.GetKernel("cmplx_mul") == nil {
		log.Panic("Kernel " + "cmplx_mul" + " does not exist!")
	}

	pl.GetKernel("cmplx_mul").SetArg(0, dst)
	pl.GetKernel("cmplx_mul").SetArg(1, a)
	pl.GetKernel("cmplx_mul").SetArg(2, b)
	pl.GetKernel("cmplx_mul").SetArg(3, conjB)
	pl.GetKernel("cmplx_mul").SetArg(4, cnt)
	pl.GetKernel("cmplx_mul").SetArg(4, offset)

	KernEvent, _ := queue.EnqueueNDRangeKernel(pl.GetKernel("cmplx_mul"), nil, cfg.Grid, cfg.Block, tmpEventList)
	tmpEventList1 = append(tmpEventList1, KernEvent)

	if err := WaitForEvents(tmpEventList1); err != nil {
		fmt.Printf("WaitForEvents failed in complexarraymul: %+v \n", err)
	}
}

//MemInputCpyFloat32 Memory Copy cl object to local
// func MemInputCpyFloat32(dst unsafe.Pointer, src *MemObject, offsetDst, offsetSrc, bytes int) {
func MemInputCpyFloat32(dst, src *MemObject, offsetDst, offsetSrc, bytes int) {

	var queue *CommandQueue
	_, err := queue.EnqueueCopyBufferFloat32(src, dst, offsetSrc, offsetDst, bytes, nil)

	//eventList[0], err = queue.EnqueueCopyBuffer(srcMemObj, dstMemObj, offsetSrc, offsetDst, bytes, nil)
	if err != nil {
		fmt.Printf("\n EnqueueCopyBuffer failed: %+v \n", err)
		panic("\n Stopping execution \n")
		//return nil
	}
	queue.Finish()
}

//Clfft3D to caluclate 3d fft directly
func Clfft3D(OutBuf, InBuf *MemObject, N0, N1, N2 int, IsReal, IsForw, IsSinglePrecision bool, context *Context) {

	// var context *Context
	var queue *CommandQueue

	// tmpPtr := InBuf.DevPtr(0)
	// srcMemObj := *(*cl.MemObject)(tmpPtr)
	// srcMem
	// tmpPtr = OutBuf.DevPtr(0)
	// dstMemObj := *(*cl.MemObject)(tmpPtr)

	flag := CLFFTDim3D
	fftPlanHandle, errF := NewCLFFTPlan(context, flag, []int{N0, N1, N2})
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

	ArrLayout := NewArrayLayout()
	if IsForw == true {
		if IsReal == false {
			ArrLayout.SetInputLayout(CLFFTLayoutComplexInterleaved)
			ArrLayout.SetOutputLayout(CLFFTLayoutComplexInterleaved)
		} else {
			ArrLayout.SetInputLayout(CLFFTLayoutReal)
			ArrLayout.SetOutputLayout(CLFFTLayoutHermitianInterleaved)
		}
	} else {
		if IsReal == false {
			ArrLayout.SetInputLayout(CLFFTLayoutComplexInterleaved)
			ArrLayout.SetOutputLayout(CLFFTLayoutComplexInterleaved)
		} else {
			ArrLayout.SetInputLayout(CLFFTLayoutHermitianInterleaved)
			ArrLayout.SetOutputLayout(CLFFTLayoutReal)
		}
	}

	errD := fftPlanHandle.SetLayouts(ArrLayout)
	if errD != nil {
		fmt.Printf("unable to set Array Layout \n")
	}

	errF = fftPlanHandle.SetResultOutOfPlace()
	if errF != nil {
		fmt.Printf("unable to set fft result location \n")
	}

	/* Bake the plan. */
	errF = fftPlanHandle.BakePlanSimple([]*CommandQueue{queue})
	if errF != nil {
		fmt.Printf("unable to bake fft plan: %+v \n", errF)
	}

	/* Execute the plan. */

	if IsForw {
		_, errF = fftPlanHandle.EnqueueForwardTransform([]*CommandQueue{queue}, nil, []*MemObject{InBuf}, []*MemObject{OutBuf}, nil)
		if errF != nil {
			fmt.Printf("unable to enqueue transform: %+v \n", errF)
		} else {
			fmt.Printf("\n Executing forward transform...\n")
		}
	} else {
		_, errF = fftPlanHandle.EnqueueBackwardTransform([]*CommandQueue{queue}, nil, []*MemObject{InBuf}, []*MemObject{OutBuf}, nil)
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

	fmt.Printf("Finished tests on clFFT\n")
	fftPlanHandle.Destroy()
}

//Clfft2D to caluclate 2d fft directly
func Clfft2D(OutBuf, InBuf *MemObject, N0 int, N1 int, IsReal, IsForw, IsSinglePrecision bool, context *Context) {

	// var context *Context
	var queue *CommandQueue

	// tmpPtr := InBuf.DevPtr(0)
	// srcMemObj := *(*cl.MemObject)(tmpPtr)
	// tmpPtr = OutBuf.DevPtr(0)
	// dstMemObj := *(*cl.MemObject)(tmpPtr)

	flag := CLFFTDim2D
	fftPlanHandle, errF := NewCLFFTPlan(context, flag, []int{N0, N1})
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

	ArrLayout := NewArrayLayout()
	if IsForw == true {
		if IsReal == false {
			ArrLayout.SetInputLayout(CLFFTLayoutComplexInterleaved)
			ArrLayout.SetOutputLayout(CLFFTLayoutComplexInterleaved)
		} else {
			ArrLayout.SetInputLayout(CLFFTLayoutReal)
			ArrLayout.SetOutputLayout(CLFFTLayoutHermitianInterleaved)
		}
	} else {
		if IsReal == false {
			ArrLayout.SetInputLayout(CLFFTLayoutComplexInterleaved)
			ArrLayout.SetOutputLayout(CLFFTLayoutComplexInterleaved)
		} else {
			ArrLayout.SetInputLayout(CLFFTLayoutHermitianInterleaved)
			ArrLayout.SetOutputLayout(CLFFTLayoutReal)
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
	errF = fftPlanHandle.BakePlanSimple([]*CommandQueue{queue})
	if errF != nil {
		fmt.Printf("unable to bake fft plan: %+v \n", errF)
	}

	/* Execute the plan. */
	if IsForw {
		_, errF = fftPlanHandle.EnqueueForwardTransform([]*CommandQueue{queue}, nil, []*MemObject{InBuf}, []*MemObject{OutBuf}, nil)
		if errF != nil {
			fmt.Printf("unable to enqueue transform: %+v \n", errF)
		} else {
			fmt.Printf("\n Executing forward transform...\n")
		}
	} else {
		_, errF = fftPlanHandle.EnqueueBackwardTransform([]*CommandQueue{queue}, nil, []*MemObject{InBuf}, []*MemObject{OutBuf}, nil)
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

	fmt.Printf("Finished tests on clFFT\n")
	fftPlanHandle.Destroy()
	//return InputData
}

//Clfft1D to caluclate 3d fft directly
func Clfft1D(OutBuf, InBuf *MemObject, N, ScaleLength int, IsReal, IsForw, IsSinglePrecision, IsScalingReq bool, context *Context) {

	// var context *Context
	var queue *CommandQueue

	flag := CLFFTDim1D
	fftPlanHandle, errF := NewCLFFTPlan(context, flag, []int{N}) //Don't change this to 2*N
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

	ArrLayout := NewArrayLayout()

	if IsForw == true {
		if IsReal == false {
			ArrLayout.SetInputLayout(CLFFTLayoutComplexInterleaved)
			ArrLayout.SetOutputLayout(CLFFTLayoutComplexInterleaved)
		} else {
			ArrLayout.SetInputLayout(CLFFTLayoutReal)
			ArrLayout.SetOutputLayout(CLFFTLayoutHermitianInterleaved)
		}
	} else {
		if IsReal == false {
			ArrLayout.SetInputLayout(CLFFTLayoutComplexInterleaved)
			ArrLayout.SetOutputLayout(CLFFTLayoutComplexInterleaved)
		} else {
			ArrLayout.SetInputLayout(CLFFTLayoutHermitianInterleaved)
			ArrLayout.SetOutputLayout(CLFFTLayoutReal)
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
		errF = fftPlanHandle.SetScale(ClFFTDirectionBackward, float32(float64(1.0)/float64(ScaleLength*N)))
		if errF != nil {
			fmt.Printf("unable to set fft result scaling \n")
		}
	}

	/* Bake the plan. */
	errF = fftPlanHandle.BakePlanSimple([]*CommandQueue{queue})
	if errF != nil {
		fmt.Printf("unable to bake fft plan: %+v \n", errF)
	}

	/* Execute the plan. */
	if IsForw {
		_, errF = fftPlanHandle.EnqueueForwardTransform([]*CommandQueue{queue}, nil, []*MemObject{InBuf}, []*MemObject{OutBuf}, nil)
		if errF != nil {
			fmt.Printf("\n Unable to enqueue forward transform: %+v \n", errF)
		} else {
			fmt.Printf("\n Executing forward transform...\n")
		}
	} else {
		_, errF = fftPlanHandle.EnqueueBackwardTransform([]*CommandQueue{queue}, nil, []*MemObject{InBuf}, []*MemObject{OutBuf}, nil)
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

	fmt.Printf("Finished tests on clFFT\n")
	fftPlanHandle.Destroy()
	//return InputData
}

//Parse1D Parsing the 1D Input
//func Parse1D(FinalBuf, InpBuf *MemObject, class interface{}) {
func Parse1D(InpBuf *MemObject, p *OclFFTPlan) {

	context := p.GetContext()
	//queue := opencl.ClCmdQueue
	fmt.Printf("\n Parsing the 1D input to execute appropriate FFT function...\n")
	var inp1d FftPlan1DValue
	if p.GetDirection() == ClFFTDirectionForward {
		inp1d.IsForw = true
	} else {
		inp1d.IsForw = false
	}

	if p.GetPrecision() == CLFFTPrecisionSingle {
		inp1d.IsSinglePreci = true
	} else {
		inp1d.IsSinglePreci = false
	}

	inp1d.RowDim = p.GetLengths()[0] * p.GetLengths()[1] * p.GetLengths()[2]
	inp1d.ColDim = 1
	inp1d.DepthDim = 1

	var Desci int
	inp1d.FinalN, Desci = determineChirpLength(inp1d.RowDim) //Desci is decision variable

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
			//inp1d.FinalN = inp1d.FinalN
			inp1d.IsBlusteinsReq = true
			fmt.Printf("\n Adjusting length and finding FFT using Blusteins Algorithm with Legnth = %d...\n", inp1d.FinalN)
		}
	}
	var FinalBuf *MemObject
	if !inp1d.IsForw && inp1d.IsRealHerm {
		FinalBuf, _ = context.CreateEmptyBufferFloat32(MemReadWrite, inp1d.RowDim)
	} else if inp1d.IsForw && inp1d.IsRealHerm {
		FinalBuf, _ = context.CreateEmptyBufferFloat32(MemReadWrite, 2*(1+inp1d.RowDim/2))
	} else {
		FinalBuf, _ = context.CreateEmptyBufferFloat32(MemReadWrite, 2*inp1d.RowDim)
	}

	//OutputBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * inp1d.RowDim, 1, 1})
	//defer opencl.Recycle(OutputBuf)

	FFT1D(FinalBuf, InpBuf, inp1d, context, p)

	fmt.Print("\n Finished calculating 1D FFT. Output will be \n")

	// if !inp1d.IsForw && inp1d.IsRealHerm {
	// 	PrintRealArray(OutputBuf.DevPtr(0), inp1d.RowDim)
	// } else if inp1d.IsForw && inp1d.IsRealHerm {
	// 	PrintArray(OutputBuf, 1+int(inp1d.RowDim/2))
	// } else {
	// 	PrintArray(OutputBuf, inp1d.RowDim)
	// }
	// PrintArray(OutputBuf, inp1d.RowDim)
}

//Parse2D Parsing the 2D Input
func Parse2D(InpBuf *MemObject, p *OclFFTPlan) {

	fmt.Printf("\n Parsing the 2D input to execute appropriate FFT function...\n")
	var c FftPlan2DValue
	if p.GetDirection() == ClFFTDirectionForward {
		c.IsForw = true
	} else {
		c.IsForw = false
	}

	if p.GetPrecision() == CLFFTPrecisionSingle {
		c.IsSinglePreci = true
	} else {
		c.IsSinglePreci = false
	}

	c.RowDim = p.GetLengths()[1]
	c.ColDim = p.GetLengths()[0]
	c.DepthDim = p.GetLengths()[2]

	context := p.GetContext()
	//queue := opencl.ClCmdQueue

	//var PrintSize int

	fmt.Printf("\n Calculating 2D FFT for the given input \n")

	fmt.Printf("\n Generating the correct size Output matrix \n")

	var FinalBuf *MemObject
	if !c.IsForw && c.IsRealHerm {
		FinalBuf, _ = context.CreateEmptyBufferFloat32(MemReadWrite, c.RowDim*c.ColDim)
	}
	if c.IsForw && c.IsRealHerm {
		FinalBuf, _ = context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.ColDim*(1+c.RowDim/2))
	}
	if !c.IsRealHerm {
		FinalBuf, _ = context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim*c.ColDim)
	}

	//var IsBlusteinRow, IsBlusteinCol bool
	ValRow, DecideRow := determineChirpLength(c.RowDim)
	ValCol, DecideCol := determineChirpLength(c.ColDim)
	if (DecideRow == 1) && (DecideCol == 1) && (ValRow == 0) && (ValCol == 0) {
		fmt.Printf("\n No need to execute Blusteins for any dimension. Executing CLFFT directly \n")
		if c.IsForw {
			if c.IsRealHerm {
				//PrintSize = c.ColDim * int(1+c.RowDim/2)
				// TempBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.ColDim * (1 + c.RowDim/2), 1, 1})
				// defer opencl.Recycle(TempBuf)
				fmt.Printf("\n Executing Forward Real FFT without Bluestein's ...\n")
				Clfft2D(FinalBuf, InpBuf, c.RowDim, c.ColDim, true, true, c.IsSinglePreci, context)

			} else {
				//FinalBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.ColDim * c.RowDim, 1, 1})
				fmt.Printf("\n Executing Forward Complex FFT without Bluestein's ...\n")
				Clfft2D(FinalBuf, InpBuf, c.RowDim, c.ColDim, false, true, c.IsSinglePreci, context)
			}

		} else {
			if c.IsRealHerm {
				fmt.Printf("\n Executing Inverse Hermitian FFT without Bluestein's...\n")
				Clfft2D(FinalBuf, InpBuf, c.RowDim, c.ColDim, c.IsRealHerm, c.IsForw, c.IsSinglePreci, context)

			} else {
				//FinalBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim * c.RowDim, 1, 1})
				fmt.Printf("\n Executing Inverse Complex FFT without Bluestein's...\n")
				Clfft2D(FinalBuf, InpBuf, c.RowDim, c.ColDim, c.IsRealHerm, c.IsForw, c.IsSinglePreci, context)
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
			c.RowBluLeng = ValRow
			c.IsBlusteinsReqRow = true
			fmt.Printf("\n Blusteins Algorithm required for Rows with New Legnth = %d...\n", c.RowBluLeng)
		}
		if ValCol == 0 {
			// fmt.Printf("\n Bluestein is not required. Executing clFFT with length %v...", BluN)
			c.ColBluLeng = c.ColDim
			c.IsBlusteinsReqCol = false
			fmt.Printf("\n Blusteins Algorithm not required for Columns as Legnth = %d...\n", c.ColBluLeng)
		} else {
			c.ColBluLeng = ValCol
			c.IsBlusteinsReqCol = true
			fmt.Printf("\n Blusteins Algorithm required for Columns with New Legnth = %d...\n", c.ColBluLeng)
		}

		// 	fmt.Printf("\n Blusteins assignments finished. Now non blustein cases begin \n")

		if c.IsForw {
			if c.IsRealHerm {
				// for i := 0; i < int(*Flag_comp); i++ {
				// 				PrintSize = int(1+c.RowDim/2) * c.ColDim
				TempOutBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*int(1+c.RowDim/2)*c.ColDim)
				TranpoBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.ColDim*int(1+c.RowDim/2))
				//
				for j := 0; j < c.ColDim; j++ {
					//SmallBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
					SmallBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, c.RowDim)
					MemInputCpyFloat32(SmallBuff, InpBuf, 0, j*c.RowDim, c.RowDim)
					FftBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*int(1+c.RowDim/2))
					TempPlan := FftPlan1DValue{true, true, true, c.IsBlusteinsReqRow, c.RowDim, 1, 1, c.RowBluLeng}
					FFT1D(FftBuff, SmallBuff, TempPlan, context, p)
					MemInputCpyFloat32(TempOutBuf, FftBuff, 2*j*int(1+c.RowDim/2), 0, 2*int(1+c.RowDim/2))
					SmallBuff.Release()
					FftBuff.Release()
				}

				fmt.Printf("\n Implementing Transpose \n")
				ComplexMatrixTranspose(TranpoBuf, TempOutBuf, 0, int(1+c.RowDim/2), c.ColDim, p)
				fmt.Printf("\n Finished transposing \n")

				SecOutBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*int(1+c.RowDim/2)*c.ColDim)

				//FinalTempBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim * c.ColDim, 1, 1})
				//defer opencl.Recycle(SecOutBuf)
				for j := 0; j < int(1+c.RowDim/2); j++ {
					SmallBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.ColDim)
					MemInputCpyFloat32(SmallBuff, TranpoBuf, 0, 2*j*c.ColDim, 2*c.ColDim)
					FftBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.ColDim)
					TempPlan := FftPlan1DValue{true, false, true, c.IsBlusteinsReqCol, c.ColDim, 1, 1, c.ColBluLeng}
					FFT1D(FftBuff, SmallBuff, TempPlan, context, p)
					MemInputCpyFloat32(SecOutBuf, FftBuff, 2*j*c.ColDim, 0, 2*c.ColDim)
					SmallBuff.Release()
					FftBuff.Release()
				}
				fmt.Printf("\n Implementing Transpose \n")
				ComplexMatrixTranspose(FinalBuf, SecOutBuf, 0, c.ColDim, int(1+c.RowDim/2), p)
				fmt.Printf("\n Printing 2d individual output array \n")
				//PrintArray(FinalBuf, c.RowDim*c.ColDim)
				TempOutBuf.Release()
				SecOutBuf.Release()
				TranpoBuf.Release()
				//opencl.Recycle(FinalTempBuf)
				// }
			} else {
				// for i := 0; i < int(*Flag_comp); i++ {
				// 				PrintSize = c.RowDim * c.ColDim
				TempOutBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim*c.ColDim)
				TranpoBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim*c.ColDim)
				//defer opencl.Recycle(TempOutBuf)
				for j := 0; j < c.ColDim; j++ {
					SmallBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim)
					MemInputCpyFloat32(SmallBuff, InpBuf, 0, 2*j*c.RowDim, 2*c.RowDim)
					FftBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim)
					TempPlan := FftPlan1DValue{true, false, true, c.IsBlusteinsReqRow, c.RowDim, 1, 1, c.RowBluLeng}
					FFT1D(FftBuff, SmallBuff, TempPlan, context, p)
					MemInputCpyFloat32(TempOutBuf, FftBuff, 2*j*c.RowDim, 0, 2*c.RowDim)
					SmallBuff.Release()
					FftBuff.Release()
				}

				fmt.Printf("\n Implementing Transpose \n")
				ComplexMatrixTranspose(TranpoBuf, TempOutBuf, 0, c.RowDim, c.ColDim, p)

				SecOutBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim*c.ColDim)

				//TranpoBuf := opencl.Buffer(NComponents, [3]int{2 * plan2d.RowDim * plan2d.ColDim, 1, 1})
				//defer opencl.Recycle(SecOutBuf)
				for j := 0; j < c.RowDim; j++ {
					SmallBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.ColDim)
					MemInputCpyFloat32(SmallBuff, TranpoBuf, 0, 2*j*c.ColDim, 2*c.ColDim)
					FftBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.ColDim)
					TempPlan := FftPlan1DValue{true, false, true, c.IsBlusteinsReqCol, c.ColDim, 1, 1, c.ColBluLeng}
					FFT1D(FftBuff, SmallBuff, TempPlan, context, p)
					MemInputCpyFloat32(SecOutBuf, FftBuff, 2*j*c.ColDim, 0, 2*c.ColDim)
					SmallBuff.Release()
					FftBuff.Release()
				}
				fmt.Printf("\n Implementing Transpose \n")
				ComplexMatrixTranspose(FinalBuf, SecOutBuf, 0, c.ColDim, c.RowDim, p)
				fmt.Printf("\n Printing 2d individual output array \n")
				//PrintArray(FinalBuf, c.RowDim*c.ColDim)
				TempOutBuf.Release()
				SecOutBuf.Release()
				TranpoBuf.Release()
				// }
			}
		} else if c.IsRealHerm {
			// for i := 0; i < int(*Flag_comp); i++ {
			fmt.Printf("\n Coming here correctly \n")
			// 			PrintSize = c.RowDim * c.ColDim
			TempOutBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim*c.ColDim)
			TranpoBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim*c.ColDim)
			//defer opencl.Recycle(TempOutBuf)
			for j := 0; j < c.ColDim; j++ {
				SmallBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*int(1+c.RowDim/2))
				fmt.Printf("\n Coming here correctly \n")
				MemInputCpyFloat32(SmallBuff, InpBuf, 0, 2*j*int(1+c.RowDim/2), 2*int(1+c.RowDim/2))
				fmt.Printf("\n \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ First Copy for the %d time ///////////////////", j)
				TempFftBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, c.RowDim)
				TempPlan := FftPlan1DValue{false, true, true, c.IsBlusteinsReqRow, c.RowDim, 1, 1, c.RowBluLeng}
				FFT1D(TempFftBuff, SmallBuff, TempPlan, context, p)
				FftBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim)
				fmt.Printf("\n \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ FFt for the %d time ///////////////////", j)
				PackComplexArray(FftBuff, TempFftBuff, c.RowDim, 0, 0, p)
				MemInputCpyFloat32(TempOutBuf, FftBuff, 2*j*c.RowDim, 0, 2*c.RowDim)
				SmallBuff.Release()
				FftBuff.Release()
				TempFftBuff.Release()
			}
			fmt.Printf("\n Implementing Transpose \n")
			ComplexMatrixTranspose(TranpoBuf, TempOutBuf, 0, c.RowDim, c.ColDim, p)

			SecOutBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim*c.ColDim)
			//TranpoBuf := opencl.Buffer(NComponents, [3]int{2 * plan2d.RowDim * plan2d.ColDim, 1, 1})
			//defer opencl.Recycle(SecOutBuf)
			for j := 0; j < c.RowDim; j++ {
				SmallBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.ColDim)
				MemInputCpyFloat32(SmallBuff, TranpoBuf, 0, 2*j*c.ColDim, 2*c.ColDim)
				FftBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.ColDim)
				TempPlan := FftPlan1DValue{false, false, true, c.IsBlusteinsReqCol, c.ColDim, 1, 1, c.ColBluLeng}
				FFT1D(FftBuff, SmallBuff, TempPlan, context, p)
				MemInputCpyFloat32(SecOutBuf, FftBuff, 2*j*c.ColDim, 0, 2*c.ColDim)
				SmallBuff.Release()
				FftBuff.Release()
			}
			fmt.Printf("\n Implementing Transpose \n")
			OutTransTempBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim*c.ColDim)
			ComplexMatrixTranspose(OutTransTempBuff, SecOutBuf, 0, c.ColDim, c.RowDim, p)
			fmt.Printf("\n Printing 2d individual output array \n")
			for j := 0; j < c.ColDim; j++ {
				SmallBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim)
				MemInputCpyFloat32(SmallBuff, OutTransTempBuff, 0, 2*j*c.RowDim, 2*c.RowDim)
				fmt.Printf("\n Finished first copy \n")
				TempCompressBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, c.RowDim)
				CompressCmplxtoReal(TempCompressBuff, SmallBuff, c.RowDim, 0, 0, p)
				MemInputCpyFloat32(FinalBuf, TempCompressBuff, j*c.RowDim, 0, c.RowDim)
				SmallBuff.Release()
				TempCompressBuff.Release()
			}
			//PrintArray(FinalBuf, c.RowDim*c.ColDim)
			TempOutBuf.Release()
			SecOutBuf.Release()
			TranpoBuf.Release()
			OutTransTempBuff.Release()
			// }
		} else {
			// for i := 0; i < int(*Flag_comp); i++ {
			// PrintSize = c.RowDim * c.ColDim
			TempOutBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim*c.ColDim)
			TranpoBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim*c.ColDim)
			//defer opencl.Recycle(TempOutBuf)
			for j := 0; j < c.ColDim; j++ {
				SmallBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim)
				MemInputCpyFloat32(SmallBuff, InpBuf, 0, 2*j*c.RowDim, 2*c.RowDim)
				FftBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim)
				TempPlan := FftPlan1DValue{false, false, true, c.IsBlusteinsReqRow, c.RowDim, 1, 1, c.RowBluLeng}
				FFT1D(FftBuff, SmallBuff, TempPlan, context, p)
				MemInputCpyFloat32(TempOutBuf, FftBuff, 2*j*c.RowDim, 0, 2*c.RowDim)
				SmallBuff.Release()
				FftBuff.Release()
			}
			fmt.Printf("\n Implementing Transpose \n")
			ComplexMatrixTranspose(TranpoBuf, TempOutBuf, 0, c.RowDim, c.ColDim, p)

			SecOutBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim*c.ColDim)
			//TranpoBuf := opencl.Buffer(NComponents, [3]int{2 * plan2d.RowDim * plan2d.ColDim, 1, 1})
			//defer opencl.Recycle(SecOutBuf)
			for j := 0; j < c.RowDim; j++ {
				SmallBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.ColDim)
				MemInputCpyFloat32(SmallBuff, TranpoBuf, 0, 2*j*c.ColDim, 2*c.ColDim)
				FftBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.ColDim)
				TempPlan := FftPlan1DValue{false, false, true, c.IsBlusteinsReqCol, c.ColDim, 1, 1, c.ColBluLeng}
				FFT1D(FftBuff, SmallBuff, TempPlan, context, p)
				MemInputCpyFloat32(SecOutBuf, FftBuff, 2*j*c.ColDim, 0, 2*c.ColDim)
				SmallBuff.Release()
				FftBuff.Release()
			}
			fmt.Printf("\n Implementing Transpose \n")
			ComplexMatrixTranspose(FinalBuf, SecOutBuf, 0, c.ColDim, c.RowDim, p)
			fmt.Printf("\n Printing 2d individual output array \n")
			//PrintArray(FinalBuf, c.RowDim*c.ColDim)
			TempOutBuf.Release()
			SecOutBuf.Release()
			TranpoBuf.Release()
			// }
		}

	}

	// if c.IsRealHerm && !c.IsForw {
	// 	PrintRealArray(FinalBuf.DevPtr(0), PrintSize)
	// } else {
	// 	PrintArray(FinalBuf, PrintSize)
	// }
}

//FFT1D to identify the details about the FFT
func FFT1D(FinalBuf, InpBuf *MemObject, class interface{}, context *Context, pl *OclFFTPlan) {
	var queue *CommandQueue
	fmt.Printf("\n Parsing the input to execute appropriate FFT function...\n")
	c, ok := class.(FftPlan1DValue)

	if !ok {
		panic("\n Wrong Input given... Terminating...\n")
	}

	if !c.IsBlusteinsReq {
		if c.IsForw {
			if c.IsRealHerm {

				fmt.Printf("\n Executing Forward Real FFT without Bluestein's ...\n")
				//purefft.Clfft1D(TempBuf, OpBuf, c.RowDim, true, true, c.IsSinglePreci)
				Clfft1D(FinalBuf, InpBuf, c.RowDim, c.FinalN, true, true, c.IsSinglePreci, false, context)

			} else {
				//FinalBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
				fmt.Printf("\n Executing Forward Complex FFT without Bluestein's ...\n")
				Clfft1D(FinalBuf, InpBuf, c.RowDim, c.FinalN, false, true, c.IsSinglePreci, false, context)
				//return FinalBuf
			}

		} else {
			if c.IsRealHerm {
				//TempBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
				//defer opencl.Recycle(TempBuf)
				//opencl.Hermitian2Full(TempBuf, InpBuf)
				fmt.Printf("\n Executing Inverse Hermitian FFT without Bluestein's...\n")
				//purefft.Clfft1D(InpBuf, FinalBuf, c.RowDim, c.IsRealHerm, c.IsForw, c.IsSinglePreci)
				Clfft1D(FinalBuf, InpBuf, c.RowDim, c.FinalN, true, false, c.IsSinglePreci, false, context)
				//return FinalBuf

			} else {
				//FinalBuf := data.NewSlice(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
				fmt.Printf("\n Executing Inverse Complex FFT without Bluestein's...\n")
				Clfft1D(FinalBuf, InpBuf, c.RowDim, c.FinalN, false, false, c.IsSinglePreci, false, context)
				//return FinalBuf
			}
		}

	} else {
		if c.IsForw {
			if c.IsRealHerm {
				fmt.Printf("\n Executing Forward Real FFT with Bluesteins...\n")
				PartATempBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim)
				defer PartATempBuf.Release()
				fmt.Printf("\n Processing Part A with the twiddle factor \n")
				PackComplexArray(PartATempBuf, InpBuf, c.RowDim, 0, 0, pl)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				PartAProcBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer PartAProcBuf.Release()
				fmt.Printf("\n Processing Part A with the twiddle factor \n")
				PartAProcess(PartAProcBuf, PartATempBuf, c.RowDim, c.FinalN, 1, 0, pl)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				PartBBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer PartBBuf.Release()
				fmt.Printf("\n Generating part B for Bluesteins")
				PartBTwidFac(PartBBuf, c.RowDim, c.FinalN, 1, 0, pl)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Executing forward FFT for Part A \n")
				PartAFFT, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer PartAFFT.Release()
				Clfft1D(PartAFFT, PartAProcBuf, c.FinalN, c.FinalN, false, true, c.IsSinglePreci, false, context)
				fmt.Printf("\n Executing forward FFT for Part B \n")
				PartBFFT, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer PartBFFT.Release()
				Clfft1D(PartBFFT, PartBBuf, c.FinalN, c.FinalN, false, true, c.IsSinglePreci, false, context)

				fmt.Printf("\n Multiplying Part A and Part B FFT \n")
				MulBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer MulBuff.Release()
				ComplexArrayMul(MulBuff, PartAFFT, PartBFFT, 0, c.FinalN, 0, pl)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Taking inverse FFT of multiplication \n")
				InvBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer InvBuff.Release()
				Clfft1D(InvBuff, MulBuff, c.FinalN, c.FinalN, false, false, c.IsSinglePreci, false, context)

				fmt.Printf("\n Preparing final twiddle factor")
				FinTwid, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer FinTwid.Release()
				FinalMulTwid(FinTwid, c.RowDim, c.FinalN, 1, 0, pl)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Multiplying with Final Twiddle Factor")

				FinTempBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer FinTempBuff.Release()
				ComplexArrayMul(FinTempBuff, FinTwid, InvBuff, 0, c.FinalN, 0, pl)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				// Execute the special harm check function here
				fmt.Printf("\n Checking Hermitian Warning \n")
				//HermitianWarning(FinTempBuff, c.RowDim, c.FinalN)

				MemInputCpyFloat32(FinalBuf, FinTempBuff, 0, 0, 2*int(1+c.RowDim/2))
				fmt.Printf("\n Finished calculating Foward FFT using Blusteins Method \n")

			} else {

				fmt.Printf("\n Executing Forward Complex FFT with Bluesteins...\n")

				PartAProcBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer PartAProcBuf.Release()
				fmt.Printf("\n Processing Part A with the twiddle factor \n")
				PartAProcess(PartAProcBuf, InpBuf, c.RowDim, c.FinalN, 1, 0, pl)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				PartBBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer PartBBuf.Release()
				fmt.Printf("\n Generating part B for Bluesteins")
				PartBTwidFac(PartBBuf, c.RowDim, c.FinalN, 1, 0, pl)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Executing forward FFT for Part A \n")
				PartAFFT, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer PartAFFT.Release()
				Clfft1D(PartAFFT, PartAProcBuf, c.FinalN, c.FinalN, false, true, c.IsSinglePreci, false, context)

				fmt.Printf("\n Executing forward FFT for Part B \n")

				PartBFFT, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer PartBFFT.Release()
				Clfft1D(PartBFFT, PartBBuf, c.FinalN, c.FinalN, false, true, c.IsSinglePreci, false, context)

				fmt.Printf("\n Multiplying Part A and Part B FFT \n")
				MulBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer MulBuff.Release()
				ComplexArrayMul(MulBuff, PartAFFT, PartBFFT, 0, c.FinalN, 0, pl)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Taking inverse FFT of multiplication \n")
				InvBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer InvBuff.Release()
				Clfft1D(InvBuff, MulBuff, c.FinalN, c.FinalN, false, false, c.IsSinglePreci, false, context)

				fmt.Printf("\n Preparing final twiddle factor")
				FinTwid, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer FinTwid.Release()
				FinalMulTwid(FinTwid, c.RowDim, c.FinalN, 1, 0, pl)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Multiplying with Final Twiddle Factor")
				FinTempBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer FinTempBuff.Release()
				ComplexArrayMul(FinTempBuff, FinTwid, InvBuff, 0, c.FinalN, 0, pl)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				MemInputCpyFloat32(FinalBuf, FinTempBuff, 0, 0, 2*c.RowDim)
				fmt.Printf("\n Finished calculating Foward FFT using Blusteins Method \n")

			}
		} else {
			if c.IsRealHerm {
				fmt.Printf("\n Executing Inverse Real FFT with Bluesteins...\n")

				PartABuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim)
				defer PartABuf.Release()

				fmt.Printf("\n Converting Hermitian to Full Complex of Part A to complex for multiplication with twiddle factor\n")
				Hermitian2Full(PartABuf, InpBuf, c.FinalN/2, c.RowDim/2, pl)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()

				PartAProcBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer PartAProcBuf.Release()
				fmt.Printf("\n Processing Part A with the twiddle factor \n")
				PartAProcess(PartAProcBuf, PartABuf, c.RowDim, c.FinalN, -1, 0, pl)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				PartBBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer PartBBuf.Release()
				fmt.Printf("\n Generating part B for Bluesteins")
				PartBTwidFac(PartBBuf, c.RowDim, c.FinalN, -1, 0, pl)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Executing forward FFT for Part A \n")
				PartAFFT, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer PartAFFT.Release()
				Clfft1D(PartAFFT, PartAProcBuf, c.FinalN, c.FinalN, false, true, c.IsSinglePreci, false, context)
				fmt.Printf("\n Executing forward FFT for Part B \n")
				PartBFFT, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer PartBFFT.Release()
				Clfft1D(PartBFFT, PartBBuf, c.FinalN, c.FinalN, false, true, c.IsSinglePreci, false, context)
				fmt.Printf("\n Multiplying Part A and Part B FFT \n")
				MulBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer MulBuff.Release()
				ComplexArrayMul(MulBuff, PartAFFT, PartBFFT, 0, c.FinalN, 0, pl)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Taking inverse FFT of multiplication \n")
				InvBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer InvBuff.Release()
				Clfft1D(InvBuff, MulBuff, c.FinalN, c.RowDim, false, false, c.IsSinglePreci, true, context)

				fmt.Printf("\n Preparing final twiddle factor")
				FinTwid, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer FinTwid.Release()
				FinalMulTwid(FinTwid, c.RowDim, c.FinalN, -1, 0, pl)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Multiplying with Final Twiddle Factor")
				FinTempBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer FinTempBuff.Release()
				ComplexArrayMul(FinTempBuff, FinTwid, InvBuff, 0, c.FinalN, 0, pl)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Converting the array to required length \n")
				CompressCmplxtoReal(FinalBuf, FinTempBuff, c.RowDim, 0, 0, pl)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Finished calculating Inverse FFT using Blusteins Method \n")

			} else {
				fmt.Printf("\n Executing Inverse Complex FFT ...\n")

				PartAProcBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer PartAProcBuf.Release()
				fmt.Printf("\n Processing Part A with the twiddle factor \n")
				PartAProcess(PartAProcBuf, InpBuf, c.RowDim, c.FinalN, -1, 0, pl)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				PartBBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer PartBBuf.Release()
				fmt.Printf("\n Generating part B for Bluesteins")
				PartBTwidFac(PartBBuf, c.RowDim, c.FinalN, -1, 0, pl)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Executing forward FFT for Part A \n")
				PartAFFT, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer PartAFFT.Release()
				Clfft1D(PartAFFT, PartAProcBuf, c.FinalN, c.FinalN, false, true, c.IsSinglePreci, false, context)

				fmt.Printf("\n Executing forward FFT for Part B \n")
				PartBFFT, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer PartBFFT.Release()
				Clfft1D(PartBFFT, PartBBuf, c.FinalN, c.FinalN, false, true, c.IsSinglePreci, false, context)

				fmt.Printf("\n Multiplying Part A and Part B FFT \n")
				MulBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer MulBuff.Release()
				ComplexArrayMul(MulBuff, PartAFFT, PartBFFT, 0, c.FinalN, 0, pl)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Taking inverse FFT of multiplication \n")
				InvBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer InvBuff.Release()
				Clfft1D(InvBuff, MulBuff, c.FinalN, c.RowDim, false, false, c.IsSinglePreci, true, context)

				fmt.Printf("\n Preparing final twiddle factor")
				FinTwid, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				FinalMulTwid(FinTwid, c.RowDim, c.FinalN, -1, 0, pl)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				fmt.Printf("\n Multiplying with Final Twiddle Factor")
				FinTempBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.FinalN)
				defer FinTempBuff.Release()
				ComplexArrayMul(FinTempBuff, FinTwid, InvBuff, 0, c.FinalN, 0, pl)
				fmt.Println("\n Waiting for kernel to finish execution...")
				queue.Finish()
				fmt.Println("\n Execution finished.")

				MemInputCpyFloat32(FinalBuf, FinTempBuff, 0, 0, 2*c.RowDim)
				fmt.Printf("\n Finished calculating Inverse FFT using Blusteins Method \n")
				// 			//Alternatively use opencl/cl/clFFT.go => SetScale()

			}
		}
	}
}

//Parse3D Parsing the 3D Input
func Parse3D(InpBuf *MemObject, class interface{}, p *OclFFTPlan) {

	fmt.Printf("\n Parsing the input to execute appropriate FFT function...\n")

	var c FftPlan3DValue
	if p.GetDirection() == ClFFTDirectionForward {
		c.IsForw = true
	} else {
		c.IsForw = false
	}

	if p.GetPrecision() == CLFFTPrecisionSingle {
		c.IsSinglePreci = true
	} else {
		c.IsSinglePreci = false
	}

	c.RowDim = p.GetLengths()[1]
	c.ColDim = p.GetLengths()[0]
	c.DepthDim = p.GetLengths()[2]

	context := p.GetContext()
	// 	c, ok := class.(FftPlan3DValue)
	// 	if !ok {
	// 		panic("\n Wrong 3D Input given... Terminating...\n")
	// 	}
	// 	//context := opencl.ClCtx
	// 	//queue := opencl.ClCmdQueue

	// 	//var PrintSize int

	// 	fmt.Printf("\n Calculating 3D FFT for the given input \n")

	// 	fmt.Printf("\n Generating the correct size Output matrix \n")

	var FinalBuf *MemObject
	if !c.IsForw && c.IsRealHerm {
		FinalBuf, _ = context.CreateEmptyBufferFloat32(MemReadWrite, c.RowDim*c.ColDim*c.DepthDim)
	}
	if c.IsForw && c.IsRealHerm {
		FinalBuf, _ = context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.DepthDim*c.ColDim*(1+c.RowDim/2))
	}
	if !c.IsRealHerm {
		FinalBuf, _ = context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.DepthDim*c.RowDim*c.ColDim)
	}

	// 	//var IsBlusteinRow, IsBlusteinCol bool
	ValRow, DecideRow := determineChirpLength(c.RowDim)
	ValCol, DecideCol := determineChirpLength(c.ColDim)
	ValDep, DecideDep := determineChirpLength(c.DepthDim)
	if (DecideRow != 1) || (DecideCol != 1) || (DecideDep != 1) {
		panic("\n Something is wrong with Length of the given array. Please check...\n")
	}
	if (ValRow == 0) && (ValCol == 0) && (ValDep == 0) {
		fmt.Printf("\n No need to execute Blusteins for any dimension. Executing CLFFT directly \n")
		if c.IsForw {
			if c.IsRealHerm {
				//PrintSize = c.DepthDim * c.ColDim * int(1+c.RowDim/2)
				// TempBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.ColDim * (1 + c.RowDim/2), 1, 1})
				// defer opencl.Recycle(TempBuf)
				fmt.Printf("\n Executing Forward Real FFT without Bluestein's ...\n")
				Clfft3D(FinalBuf, InpBuf, c.RowDim, c.ColDim, c.DepthDim, true, true, c.IsSinglePreci, context)

			} else {
				//PrintSize = c.DepthDim * c.ColDim * int(1+c.RowDim/2)
				//FinalBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.ColDim * c.RowDim, 1, 1})
				fmt.Printf("\n Executing Forward Complex FFT without Bluestein's ...\n")
				Clfft3D(FinalBuf, InpBuf, c.RowDim, c.ColDim, c.DepthDim, false, true, c.IsSinglePreci, context)
			}

		} else {
			if c.IsRealHerm {
				//PrintSize = c.DepthDim * c.ColDim * int(1+c.RowDim/2)
				fmt.Printf("\n Executing Inverse Hermitian FFT without Bluestein's...\n")
				Clfft3D(FinalBuf, InpBuf, c.RowDim, c.ColDim, c.DepthDim, true, false, c.IsSinglePreci, context)

			} else {
				//PrintSize = c.DepthDim * c.ColDim * int(1+c.RowDim/2)
				//FinalBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim * c.RowDim, 1, 1})
				fmt.Printf("\n Executing Inverse Complex FFT without Bluestein's...\n")
				Clfft3D(FinalBuf, InpBuf, c.RowDim, c.ColDim, c.DepthDim, false, false, c.IsSinglePreci, context)
			}
		}
	}

	if (ValRow != 0) || (ValCol != 0) || (ValDep != 0) {
		fmt.Printf("\n Executing FFT with Blusteins Algorithm for at least one dimension \n")
		if ValRow == 0 {
			// fmt.Printf("\n Bluestein is not required. Executing clFFT with length %v...", BluN)
			c.RowBluLeng = c.RowDim
			c.IsBlusteinsReqRow = false
			fmt.Printf("\n Blusteins Algorithm not required for Row as Legnth = %d...\n", c.RowBluLeng)
		} else {
			c.RowBluLeng = ValRow
			c.IsBlusteinsReqRow = true
			fmt.Printf("\n Blusteins Algorithm required for Rows with New Legnth = %d...\n", c.RowBluLeng)
		}
		if ValCol == 0 {
			// fmt.Printf("\n Bluestein is not required. Executing clFFT with length %v...", BluN)
			c.ColBluLeng = c.ColDim
			c.IsBlusteinsReqCol = false
			fmt.Printf("\n Blusteins Algorithm not required for Columns as Legnth = %d...\n", c.ColBluLeng)
		} else {
			c.ColBluLeng = ValCol
			c.IsBlusteinsReqCol = true
			fmt.Printf("\n Blusteins Algorithm required for Columns with New Legnth = %d...\n", c.ColBluLeng)
		}
		if ValDep == 0 {
			// fmt.Printf("\n Bluestein is not required. Executing clFFT with length %v...", BluN)
			c.DepBluLeng = c.DepthDim
			c.IsBlusteinReqDep = false
			fmt.Printf("\n Blusteins Algorithm not required for Columns as Legnth = %d...\n", c.DepBluLeng)
		} else {
			c.DepBluLeng = ValDep
			c.IsBlusteinReqDep = true
			fmt.Printf("\n Blusteins Algorithm required for Columns with New Legnth = %d...\n", c.DepBluLeng)
		}

		fmt.Printf("\n Blusteins assignments finished. Now blustein cases begin \n")

		if c.IsForw {
			if c.IsRealHerm {
				// for i := 0; i < int(*Flag_comp); i++ {
				TempOutBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*int(1+c.RowDim/2)*c.ColDim*c.DepthDim)
				TranpoBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.ColDim*int(1+c.RowDim/2)*c.DepthDim)
				//defer opencl.Recycle(TempOutBuf)
				for j := 0; j < c.ColDim*c.DepthDim; j++ {
					//SmallBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
					SmallBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, c.RowDim)
					MemInputCpyFloat32(SmallBuff, InpBuf, 0, j*c.RowDim, c.RowDim)
					FftBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*int(1+c.RowDim/2))
					TempPlan := FftPlan1DValue{true, true, true, c.IsBlusteinsReqRow, c.RowDim, 1, 1, c.RowBluLeng}
					FFT1D(FftBuff, SmallBuff, TempPlan, context, p)
					MemInputCpyFloat32(TempOutBuf, FftBuff, 2*j*int(1+c.RowDim/2), 0, 2*int(1+c.RowDim/2))
					SmallBuff.Release()
					FftBuff.Release()
				}

				fmt.Printf("\n Implementing Transpose \n")
				ComplexMatrixTranspose(TranpoBuf, TempOutBuf, 0, int(1+c.RowDim/2), c.ColDim*c.DepthDim, p)
				fmt.Printf("\n +++++++++++++++++++ Finished transposing for first dimension +++++++++++++++++ \n")

				SecOutBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*int(1+c.RowDim/2)*c.ColDim*c.DepthDim)
				SecTrnBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*int(1+c.RowDim/2)*c.ColDim*c.DepthDim)

				//FinalTempBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim * c.ColDim, 1, 1})
				//defer opencl.Recycle(SecOutBuf)
				for j := 0; j < int(1+c.RowDim/2)*c.DepthDim; j++ {
					SmallBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.ColDim)
					MemInputCpyFloat32(SmallBuff, TranpoBuf, 0, 2*j*c.ColDim, 2*c.ColDim)
					FftBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.ColDim)
					TempPlan := FftPlan1DValue{true, false, true, c.IsBlusteinsReqCol, c.ColDim, 1, 1, c.ColBluLeng}
					FFT1D(FftBuff, SmallBuff, TempPlan, context, p)
					MemInputCpyFloat32(SecOutBuf, FftBuff, 2*j*c.ColDim, 0, 2*c.ColDim)
					SmallBuff.Release()
					FftBuff.Release()
				}
				fmt.Printf("\n Implementing Transpose \n")
				ComplexMatrixTranspose(SecTrnBuf, SecOutBuf, 0, c.ColDim, int(1+c.RowDim/2)*c.DepthDim, p)
				fmt.Printf("\n +++++++++++++++++++++++++++ Finished transpose for second dimension +++++++++++++++++ \n")

				TerOutBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*int(1+c.RowDim/2)*c.ColDim*c.DepthDim)

				//FinalTempBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim * c.ColDim, 1, 1})
				//defer opencl.Recycle(SecOutBuf)
				for j := 0; j < int(1+c.RowDim/2)*c.ColDim; j++ {
					SmallBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.DepthDim)
					MemInputCpyFloat32(SmallBuff, SecTrnBuf, 0, 2*j*c.DepthDim, 2*c.DepthDim)
					FftBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.DepthDim)
					TempPlan := FftPlan1DValue{true, false, true, c.IsBlusteinReqDep, c.DepthDim, 1, 1, c.DepBluLeng}
					FFT1D(FftBuff, SmallBuff, TempPlan, context, p)
					MemInputCpyFloat32(TerOutBuf, FftBuff, 2*j*c.DepthDim, 0, 2*c.DepthDim)
					SmallBuff.Release()
					FftBuff.Release()
				}
				fmt.Printf("\n Implementing Transpose \n")
				ComplexMatrixTranspose(FinalBuf, TerOutBuf, 0, c.DepthDim, int(1+c.RowDim/2)*c.ColDim, p)
				fmt.Printf("\n +++++++++++++++++++++++++++ Finished transpose for third dimension +++++++++++++++++ \n")

				//PrintArray(FinalBuf, c.RowDim*c.ColDim)
				TempOutBuf.Release()
				SecOutBuf.Release()
				TranpoBuf.Release()
				SecTrnBuf.Release()
				TerOutBuf.Release()
				//opencl.Recycle(FinalTempBuf)
				// }
			} else {
				fmt.Printf("\n Executing Blusteins and Forw and Real")
				// for i := 0; i < int(*Flag_comp); i++ {
				TempOutBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim*c.ColDim*c.DepthDim)
				TranpoBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim*c.ColDim*c.DepthDim)
				//defer opencl.Recycle(TempOutBuf)
				for j := 0; j < c.ColDim*c.DepthDim; j++ {
					//SmallBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
					SmallBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim)
					MemInputCpyFloat32(SmallBuff, InpBuf, 0, 2*j*c.RowDim, 2*c.RowDim)
					FftBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim)
					TempPlan := FftPlan1DValue{true, false, true, c.IsBlusteinsReqRow, c.RowDim, 1, 1, c.RowBluLeng}
					FFT1D(FftBuff, SmallBuff, TempPlan, context, p)
					MemInputCpyFloat32(TempOutBuf, FftBuff, 2*j*c.RowDim, 0, 2*c.RowDim)
					SmallBuff.Release()
					FftBuff.Release()
				}

				fmt.Printf("\n Implementing Transpose \n")
				ComplexMatrixTranspose(TranpoBuf, TempOutBuf, 0, c.RowDim, c.ColDim*c.DepthDim, p)
				fmt.Printf("\n +++++++++++++++++++++++++++ Finished transpose for first dimension +++++++++++++++++ \n")

				SecOutBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim*c.ColDim*c.DepthDim)
				SecTrnBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim*c.ColDim*c.DepthDim)

				//FinalTempBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim * c.ColDim, 1, 1})
				//defer opencl.Recycle(SecOutBuf)
				for j := 0; j < c.RowDim*c.DepthDim; j++ {
					SmallBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.ColDim)
					MemInputCpyFloat32(SmallBuff, TranpoBuf, 0, 2*j*c.ColDim, 2*c.ColDim)
					FftBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.ColDim)
					TempPlan := FftPlan1DValue{true, false, true, c.IsBlusteinsReqCol, c.ColDim, 1, 1, c.ColBluLeng}
					FFT1D(FftBuff, SmallBuff, TempPlan, context, p)
					MemInputCpyFloat32(SecOutBuf, FftBuff, 2*j*c.ColDim, 0, 2*c.ColDim)
					SmallBuff.Release()
					FftBuff.Release()
				}
				fmt.Printf("\n Implementing Transpose \n")
				ComplexMatrixTranspose(SecTrnBuf, SecOutBuf, 0, c.ColDim, c.RowDim*c.DepthDim, p)
				fmt.Printf("\n +++++++++++++++++++++++++++ Finished transpose for second dimension +++++++++++++++++ \n")

				TerOutBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim*c.ColDim*c.DepthDim)

				//FinalTempBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim * c.ColDim, 1, 1})
				//defer opencl.Recycle(SecOutBuf)
				for j := 0; j < c.RowDim*c.ColDim; j++ {
					SmallBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.DepthDim)
					MemInputCpyFloat32(SmallBuff, SecTrnBuf, 0, 2*j*c.DepthDim, 2*c.DepthDim)
					FftBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.DepthDim)
					TempPlan := FftPlan1DValue{true, false, true, c.IsBlusteinReqDep, c.DepthDim, 1, 1, c.DepBluLeng}
					FFT1D(FftBuff, SmallBuff, TempPlan, context, p)
					MemInputCpyFloat32(TerOutBuf, FftBuff, 2*j*c.DepthDim, 0, 2*c.DepthDim)
					SmallBuff.Release()
					FftBuff.Release()
				}
				fmt.Printf("\n Implementing Transpose \n")
				ComplexMatrixTranspose(FinalBuf, TerOutBuf, 0, c.DepthDim, c.RowDim*c.ColDim, p)
				fmt.Printf("\n +++++++++++++++++++++++++++ Finished transpose for third dimension +++++++++++++++++ \n")

				//PrintArray(FinalBuf, c.RowDim*c.ColDim)
				TempOutBuf.Release()
				SecOutBuf.Release()
				TranpoBuf.Release()
				SecTrnBuf.Release()
				TerOutBuf.Release()
				//opencl.Recycle(FinalTempBuf)
				// }
			}

		} else {
			if c.IsRealHerm {
				// for i := 0; i < int(*Flag_comp); i++ {
				//PrintSize = c.RowDim * c.ColDim
				TempOutBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim*c.ColDim*c.DepthDim)
				TranpoBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim*c.ColDim*c.DepthDim)
				//defer opencl.Recycle(TempOutBuf)
				for j := 0; j < c.ColDim*c.DepthDim; j++ {
					SmallBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*int(1+c.RowDim/2))
					MemInputCpyFloat32(SmallBuff, InpBuf, 0, 2*j*int(1+c.RowDim/2), 2*int(1+c.RowDim/2))
					TempFftBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, c.RowDim)
					TempPlan := FftPlan1DValue{false, true, true, c.IsBlusteinsReqRow, c.RowDim, 1, 1, c.RowBluLeng}
					FFT1D(TempFftBuff, SmallBuff, TempPlan, context, p)
					FftBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim)
					PackComplexArray(FftBuff, TempFftBuff, c.RowDim, 0, 0, p)
					MemInputCpyFloat32(TempOutBuf, FftBuff, 2*j*c.RowDim, 0, 2*c.RowDim)
					SmallBuff.Release()
					FftBuff.Release()
					TempFftBuff.Release()
				}

				ComplexMatrixTranspose(TranpoBuf, TempOutBuf, 0, c.RowDim, c.ColDim*c.DepthDim, p)
				fmt.Printf("\n +++++++++++++++++++++++++++ Finished transpose for first dimension +++++++++++++++++ \n")

				SecOutBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim*c.ColDim*c.DepthDim)
				SecTrnBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim*c.ColDim*c.DepthDim)
				//TranpoBuf := opencl.Buffer(NComponents, [3]int{2 * plan2d.RowDim * plan2d.ColDim, 1, 1})
				//defer opencl.Recycle(SecOutBuf)
				for j := 0; j < c.RowDim*c.DepthDim; j++ {
					SmallBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.ColDim)
					MemInputCpyFloat32(SmallBuff, TranpoBuf, 0, 2*j*c.ColDim, 2*c.ColDim)
					FftBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.ColDim)
					TempPlan := FftPlan1DValue{false, false, true, c.IsBlusteinsReqCol, c.ColDim, 1, 1, c.ColBluLeng}
					FFT1D(FftBuff, SmallBuff, TempPlan, context, p)
					MemInputCpyFloat32(SecOutBuf, FftBuff, 2*j*c.ColDim, 0, 2*c.ColDim)
					SmallBuff.Release()
					FftBuff.Release()
				}

				ComplexMatrixTranspose(SecTrnBuf, SecOutBuf, 0, c.ColDim, c.RowDim*c.DepthDim, p)
				fmt.Printf("\n +++++++++++++++++++++++++++ Finished transpose for second dimension +++++++++++++++++ \n")

				//PrintArray(FinalBuf, c.RowDim*c.ColDim)

				TerOutBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim*c.ColDim*c.DepthDim)

				//FinalTempBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim * c.ColDim, 1, 1})
				//defer opencl.Recycle(SecOutBuf)
				for j := 0; j < c.RowDim*c.ColDim; j++ {
					SmallBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.DepthDim)
					MemInputCpyFloat32(SmallBuff, SecTrnBuf, 0, 2*j*c.DepthDim, 2*c.DepthDim)
					FftBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.DepthDim)
					TempPlan := FftPlan1DValue{false, false, true, c.IsBlusteinReqDep, c.DepthDim, 1, 1, c.DepBluLeng}
					FFT1D(FftBuff, SmallBuff, TempPlan, context, p)
					MemInputCpyFloat32(TerOutBuf, FftBuff, 2*j*c.DepthDim, 0, 2*c.DepthDim)
					SmallBuff.Release()
					FftBuff.Release()
				}

				OutTransTempBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim*c.ColDim*c.DepthDim)
				ComplexMatrixTranspose(OutTransTempBuff, TerOutBuf, 0, c.DepthDim, c.RowDim*c.DepthDim, p)
				fmt.Printf("\n +++++++++++++++++++++++++++ Finished transpose for third dimension +++++++++++++++++ \n")

				fmt.Printf("\n Printing 3d individual output array \n")
				for j := 0; j < c.ColDim*c.DepthDim; j++ {
					SmallBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim)
					MemInputCpyFloat32(SmallBuff, OutTransTempBuff, 0, 2*j*c.RowDim, 2*c.RowDim)
					//fmt.Printf("\n Finished first copy \n")
					TempCompressBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, c.RowDim)
					CompressCmplxtoReal(TempCompressBuff, SmallBuff, c.RowDim, 0, 0, p)
					MemInputCpyFloat32(FinalBuf, TempCompressBuff, j*c.RowDim, 0, c.RowDim)
					SmallBuff.Release()
					TempCompressBuff.Release()
				}

				//PrintArray(FinalBuf, c.RowDim*c.ColDim)
				TempOutBuf.Release()
				SecOutBuf.Release()
				TranpoBuf.Release()
				TerOutBuf.Release()
				SecTrnBuf.Release()
				OutTransTempBuff.Release()
				// }
			} else {
				// for i := 0; i < int(*Flag_comp); i++ {
				TempOutBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim*c.ColDim*c.DepthDim)
				TranpoBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim*c.ColDim*c.DepthDim)
				//defer opencl.Recycle(TempOutBuf)
				for j := 0; j < c.ColDim*c.DepthDim; j++ {
					//SmallBuff := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim, 1, 1})
					SmallBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim)
					MemInputCpyFloat32(SmallBuff, InpBuf, 0, 2*j*c.RowDim, 2*c.RowDim)
					FftBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim)
					TempPlan := FftPlan1DValue{false, false, true, c.IsBlusteinsReqRow, c.RowDim, 1, 1, c.RowBluLeng}
					FFT1D(FftBuff, SmallBuff, TempPlan, context, p)
					MemInputCpyFloat32(TempOutBuf, FftBuff, 2*j*c.RowDim, 0, 2*c.RowDim)
					SmallBuff.Release()
					FftBuff.Release()
				}

				fmt.Printf("\n Implementing Transpose \n")
				ComplexMatrixTranspose(TranpoBuf, TempOutBuf, 0, c.RowDim, c.ColDim*c.DepthDim, p)
				fmt.Printf("\n Finished transposing \n")

				SecOutBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim*c.ColDim*c.DepthDim)
				SecTrnBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim*c.ColDim*c.DepthDim)

				//FinalTempBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim * c.ColDim, 1, 1})
				//defer opencl.Recycle(SecOutBuf)
				for j := 0; j < c.RowDim*c.DepthDim; j++ {
					SmallBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.ColDim)
					MemInputCpyFloat32(SmallBuff, TranpoBuf, 0, 2*j*c.ColDim, 2*c.ColDim)
					FftBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.ColDim)
					TempPlan := FftPlan1DValue{false, false, true, c.IsBlusteinsReqCol, c.ColDim, 1, 1, c.ColBluLeng}
					FFT1D(FftBuff, SmallBuff, TempPlan, context, p)
					MemInputCpyFloat32(SecOutBuf, FftBuff, 2*j*c.ColDim, 0, 2*c.ColDim)
					SmallBuff.Release()
					FftBuff.Release()
				}
				fmt.Printf("\n Implementing Transpose \n")
				ComplexMatrixTranspose(SecTrnBuf, SecOutBuf, 0, c.ColDim, c.RowDim*c.DepthDim, p)
				fmt.Printf("\n Printing 2d individual output array \n")

				TerOutBuf, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.RowDim*c.ColDim*c.DepthDim)

				//FinalTempBuf := opencl.Buffer(int(*Flag_comp), [3]int{2 * c.RowDim * c.ColDim, 1, 1})
				//defer opencl.Recycle(SecOutBuf)
				for j := 0; j < c.RowDim*c.ColDim; j++ {
					SmallBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.DepthDim)
					MemInputCpyFloat32(SmallBuff, SecTrnBuf, 0, 2*j*c.DepthDim, 2*c.DepthDim)
					FftBuff, _ := context.CreateEmptyBufferFloat32(MemReadWrite, 2*c.DepthDim)
					TempPlan := FftPlan1DValue{false, false, true, c.IsBlusteinReqDep, c.DepthDim, 1, 1, c.DepBluLeng}
					FFT1D(FftBuff, SmallBuff, TempPlan, context, p)
					MemInputCpyFloat32(TerOutBuf, FftBuff, 2*j*c.DepthDim, 0, 2*c.DepthDim)
					SmallBuff.Release()
					FftBuff.Release()
				}
				fmt.Printf("\n Implementing Transpose \n")
				ComplexMatrixTranspose(FinalBuf, TerOutBuf, 0, c.DepthDim, c.RowDim*c.ColDim, p)
				fmt.Printf("\n Printing 2d individual output array \n")

				//PrintArray(FinalBuf, c.RowDim*c.ColDim)
				TempOutBuf.Release()
				SecOutBuf.Release()
				TranpoBuf.Release()
				SecTrnBuf.Release()
				TerOutBuf.Release()
				//opencl.Recycle(FinalTempBuf)
				// }

			}
		}

		// 		if c.IsRealHerm && c.IsForw {
		// 			Print3dArray(FinalBuf, int(1+c.RowDim/2), c.ColDim, c.DepthDim, 2)
		// 		}
		// 		if c.IsRealHerm && !c.IsForw {
		// 			Print3dArray(FinalBuf, c.RowDim, c.ColDim, c.DepthDim, 1)
		// 		}
		// 		if !c.IsRealHerm {
		// 			Print3dArray(FinalBuf, c.RowDim, c.ColDim, c.DepthDim, 2)
		// 		}
	}
}
