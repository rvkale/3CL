package opencl

import (
	"fmt"

	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/util"
)

// Hermitian2Full Convert Hermitian FFT array to full complex array
func Hermitian2Full(dst, src *data.Slice) {
	util.Argument(src.NComp() == dst.NComp())
	var tmpEventList, tmpEventList1 []*cl.Event
	for ii := 0; ii < src.NComp(); ii++ {
		tmpEvent := src.GetEvent(ii)
		if tmpEvent != nil {
			tmpEventList = append(tmpEventList, tmpEvent)
		}
	}
	for ii := 0; ii < src.NComp(); ii++ {
		event := k_hermitian2full_async(dst.DevPtr(ii), src.DevPtr(ii), dst.Len()/2, src.Len()/2, reduceintcfg, tmpEventList)
		dst.SetEvent(ii, event)
		src.SetEvent(ii, event)
		tmpEventList1 = append(tmpEventList1, event)
	}

	if err := cl.WaitForEvents(tmpEventList1); err != nil {
		fmt.Printf("WaitForEvents failed in hermitian2full: %+v \n", err)
	}
}

//PackComplexArray Convert real array to full complex array
func PackComplexArray(dst, src *data.Slice, cnt, iOff, oOff int) {
	util.Argument(src.NComp() == dst.NComp())
	util.Argument(src.Len() >= cnt)
	util.Argument(dst.Len() >= 2*cnt)
	util.Argument(cnt >= 0)
	util.Argument(iOff >= 0)
	util.Argument(oOff >= 0)
	util.Argument(cnt+iOff <= src.Len())
	util.Argument(cnt+oOff <= dst.Len())
	var tmpEventList, tmpEventList1 []*cl.Event
	for ii := 0; ii < src.NComp(); ii++ {
		tmpEvent := src.GetEvent(ii)
		if tmpEvent != nil {
			tmpEventList = append(tmpEventList, tmpEvent)
		}
	}
	for ii := 0; ii < src.NComp(); ii++ {
		event := k_pack_cmplx_async(dst.DevPtr(ii), src.DevPtr(ii), cnt, iOff, oOff, reduceintcfg, tmpEventList)
		dst.SetEvent(ii, event)
		src.SetEvent(ii, event)
		tmpEventList1 = append(tmpEventList1, event)
	}

	if err := cl.WaitForEvents(tmpEventList1); err != nil {
		fmt.Printf("WaitForEvents failed in packcmplxarray: %+v \n", err)
	}
}

//ComplexArrayMul Complex array mul
func ComplexArrayMul(dst, a, b *data.Slice, conjB, cnt, offset int) {
	util.Argument(a.NComp() == b.NComp())
	util.Argument(dst.NComp() == b.NComp())
	util.Argument(cnt >= 0)
	util.Argument(offset >= 0)
	util.Argument(a.Len() >= 2*(cnt+offset))
	util.Argument(b.Len() >= 2*(cnt+offset))
	util.Argument(dst.Len() >= 2*(cnt+offset))
	cfg := make1DConf(cnt)
	var tmpEventList, tmpEventList1 []*cl.Event
	for ii := 0; ii < a.NComp(); ii++ {
		tmpEvent := a.GetEvent(ii)
		if tmpEvent != nil {
			tmpEventList = append(tmpEventList, tmpEvent)
		}
	}
	for ii := 0; ii < b.NComp(); ii++ {
		tmpEvent := b.GetEvent(ii)
		if tmpEvent != nil {
			tmpEventList = append(tmpEventList, tmpEvent)
		}
	}
	for ii := 0; ii < dst.NComp(); ii++ {
		tmpEvent := dst.GetEvent(ii)
		if tmpEvent != nil {
			tmpEventList = append(tmpEventList, tmpEvent)
		}
	}
	for ii := 0; ii < a.NComp(); ii++ {
		event := k_cmplx_mul_async(dst.DevPtr(ii), a.DevPtr(ii), b.DevPtr(ii), conjB, cnt, offset, cfg, tmpEventList)
		dst.SetEvent(ii, event)
		a.SetEvent(ii, event)
		b.SetEvent(ii, event)
		tmpEventList1 = append(tmpEventList1, event)
	}

	if err := cl.WaitForEvents(tmpEventList1); err != nil {
		fmt.Printf("WaitForEvents failed in complexarraymul: %+v \n", err)
	}
}

// //Trans_pose Transpose Array
// func Trans_pose(dst, src *data.Slice, src_rows, src_col int) {
// 	util.Argument(src.NComp() == dst.NComp())
// 	util.Argument(src_rows >= 0)
// 	util.Argument(src_col >= 0)
// 	util.Argument(dst.Len() >= src_rows*src_col)
// 	//cfg := make1DConf(int(src_rows * src_col / 2))
// 	var tmpEventList, tmpEventList1 []*cl.Event
// 	for ii := 0; ii < src.NComp(); ii++ {
// 		tmpEvent := src.GetEvent(ii)
// 		if tmpEvent != nil {
// 			tmpEventList = append(tmpEventList, tmpEvent)
// 		}
// 	}
// 	for ii := 0; ii < src.NComp(); ii++ {
// 		fmt.Printf("\n Calling the actual function now \n")
// 		//event := k_trans_pose_async(dst.DevPtr(ii), src.DevPtr(ii), 0, src_col, src_rows, reduceintcfg, tmpEventList)
// 		event := k_trans_pose_async(dst.DevPtr(ii), src.DevPtr(ii), 0, src_col, src_rows, reducecfg, tmpEventList)
// 		dst.SetEvent(ii, event)
// 		src.SetEvent(ii, event)
// 		tmpEventList1 = append(tmpEventList1, event)
// 	}

// 	if err := cl.WaitForEvents(tmpEventList1); err != nil {
// 		fmt.Printf("WaitForEvents failed in transpose: %+v \n", err)
// 	}
// }

//ComplexMatrixTranspose Tranpose Complex matrix transpose
func ComplexMatrixTranspose(dst, src *data.Slice, offset, width, height int) {
	util.Argument(dst.NComp() == src.NComp())
	util.Argument(dst.Len() == src.Len())
	util.Argument(width > 0)
	util.Argument(height > 0)
	cfg := make3DConf([3]int{width, height, 1})
	var tmpEventList, tmpEventList1 []*cl.Event
	for ii := 0; ii < src.NComp(); ii++ {
		tmpEvent := src.GetEvent(ii)
		if tmpEvent != nil {
			tmpEventList = append(tmpEventList, tmpEvent)
		}
	}
	for ii := 0; ii < dst.NComp(); ii++ {
		tmpEvent := dst.GetEvent(ii)
		if tmpEvent != nil {
			tmpEventList = append(tmpEventList, tmpEvent)
		}
	}
	for ii := 0; ii < src.NComp(); ii++ {
		event := k_cmplx_transpose_async(dst.DevPtr(ii), src.DevPtr(ii), offset, width, height, cfg, tmpEventList)
		dst.SetEvent(ii, event)
		src.SetEvent(ii, event)
		tmpEventList1 = append(tmpEventList1, event)
	}
	if err := cl.WaitForEvents(tmpEventList1); err != nil {
		fmt.Printf("WaitForEvents failed in complexmatrixtranspose: %+v \n", err)
	}
}

//PartBTwidFac Calculating the twiddle factor for multiplication
func PartBTwidFac(dst *data.Slice, originalLeng, extendedLeng, fftDirection, offset int) {
	//util.Argument(dst.NComp() == src.NComp())
	util.Argument(originalLeng >= 0)
	util.Argument(extendedLeng >= 0)
	util.Argument(dst.Len() >= 2*(extendedLeng+offset))
	cfg := make1DConf(extendedLeng)
	var tmpEventList, tmpEventList1 []*cl.Event
	// for ii := 0; ii < b.NComp(); ii++ {
	// 	tmpEvent := b.GetEvent(ii)
	// 	if tmpEvent != nil {
	// 		tmpEventList = append(tmpEventList, tmpEvent)
	// 	}
	// }
	for ii := 0; ii < dst.NComp(); ii++ {
		tmpEvent := dst.GetEvent(ii)
		if tmpEvent != nil {
			tmpEventList = append(tmpEventList, tmpEvent)
		}
	}
	for ii := 0; ii < dst.NComp(); ii++ {
		event := k_multwiddlefact_async(dst.DevPtr(ii) /*a.DevPtr(ii)*/, originalLeng, extendedLeng, fftDirection, offset, cfg, tmpEventList)
		dst.SetEvent(ii, event)
		//a.SetEvent(ii, event)
		//b.SetEvent(ii, event)
		tmpEventList1 = append(tmpEventList1, event)
	}

	if err := cl.WaitForEvents(tmpEventList1); err != nil {
		fmt.Printf("WaitForEvents failed in PartBTwidFact: %+v \n", err)
	}
}

//FinalMulTwid Final Twiddle Factor to multiply the
func FinalMulTwid(dst *data.Slice, originalLeng, extendedLeng, fftDirection, offset int) {
	//util.Argument(dst.NComp() == src.NComp())
	util.Argument(originalLeng >= 0)
	util.Argument(extendedLeng >= 0)
	util.Argument(dst.Len() >= 2*(extendedLeng+offset))
	cfg := make1DConf(extendedLeng)
	var tmpEventList, tmpEventList1 []*cl.Event
	// for ii := 0; ii < b.NComp(); ii++ {
	// 	tmpEvent := b.GetEvent(ii)
	// 	if tmpEvent != nil {
	// 		tmpEventList = append(tmpEventList, tmpEvent)
	// 	}
	// }
	for ii := 0; ii < dst.NComp(); ii++ {
		tmpEvent := dst.GetEvent(ii)
		if tmpEvent != nil {
			tmpEventList = append(tmpEventList, tmpEvent)
		}
	}
	for ii := 0; ii < dst.NComp(); ii++ {
		event := k_finaltwiddlefact_async(dst.DevPtr(ii) /*a.DevPtr(ii)*/, originalLeng, extendedLeng, fftDirection, offset, cfg, tmpEventList)
		dst.SetEvent(ii, event)
		//a.SetEvent(ii, event)
		//b.SetEvent(ii, event)
		tmpEventList1 = append(tmpEventList1, event)
	}

	if err := cl.WaitForEvents(tmpEventList1); err != nil {
		fmt.Printf("WaitForEvents failed in PartBTwidFact: %+v \n", err)
	}
}

//PartAProcess To preprocess the input data and extend it
func PartAProcess(dst, src *data.Slice, originalLeng, extendedLeng, fftDirection, offset int) {
	util.Argument(dst.NComp() == src.NComp())
	util.Argument(originalLeng >= 0)
	util.Argument(extendedLeng >= 0)
	util.Argument(dst.Len() >= 2*(extendedLeng+offset))
	cfg := make1DConf(extendedLeng)
	var tmpEventList, tmpEventList1 []*cl.Event
	for ii := 0; ii < src.NComp(); ii++ {
		tmpEvent := src.GetEvent(ii)
		if tmpEvent != nil {
			tmpEventList = append(tmpEventList, tmpEvent)
		}
	}
	for ii := 0; ii < dst.NComp(); ii++ {
		tmpEvent := dst.GetEvent(ii)
		if tmpEvent != nil {
			tmpEventList = append(tmpEventList, tmpEvent)
		}
	}
	for ii := 0; ii < src.NComp(); ii++ {
		event := k_vartwiddlefa_async(dst.DevPtr(ii), src.DevPtr(ii), originalLeng, extendedLeng, fftDirection, offset, cfg, tmpEventList)
		dst.SetEvent(ii, event)
		//a.SetEvent(ii, event)
		//b.SetEvent(ii, event)
		tmpEventList1 = append(tmpEventList1, event)
	}

	if err := cl.WaitForEvents(tmpEventList1); err != nil {
		fmt.Printf("WaitForEvents failed in PartAProcess: %+v \n", err)
	}
}

//ScaleDown to scale the inverse fft
func ScaleDown(dst, src *data.Slice, leng, bluleng, offset int) {
	util.Argument(dst.NComp() == src.NComp())
	util.Argument(leng > 0)
	util.Argument(bluleng > 0)
	cfg := make1DConf(bluleng)
	var tmpEventList, tmpEventList1 []*cl.Event
	for ii := 0; ii < src.NComp(); ii++ {
		tmpEvent := src.GetEvent(ii)
		if tmpEvent != nil {
			tmpEventList = append(tmpEventList, tmpEvent)
		}
	}
	for ii := 0; ii < dst.NComp(); ii++ {
		tmpEvent := dst.GetEvent(ii)
		if tmpEvent != nil {
			tmpEventList = append(tmpEventList, tmpEvent)
		}
	}
	for ii := 0; ii < src.NComp(); ii++ {
		event := k_scaledown_async(dst.DevPtr(ii), src.DevPtr(ii), leng, bluleng, offset, cfg, tmpEventList)
		dst.SetEvent(ii, event)
		//a.SetEvent(ii, event)
		//b.SetEvent(ii, event)
		tmpEventList1 = append(tmpEventList1, event)
	}

	if err := cl.WaitForEvents(tmpEventList1); err != nil {
		fmt.Printf("WaitForEvents failed in PartAProcess: %+v \n", err)
	}
}

//CompressCmplxtoReal Convert complex output of iverse hermitian to real
func CompressCmplxtoReal(dst, src *data.Slice, cnt, iOff, oOff int) {
	util.Argument(src.NComp() == dst.NComp())
	util.Argument(dst.Len() >= cnt)
	util.Argument(src.Len() >= 2*cnt)
	util.Argument(cnt >= 0)
	util.Argument(iOff >= 0)
	util.Argument(oOff >= 0)
	util.Argument(cnt+iOff <= src.Len())
	util.Argument(cnt+oOff <= dst.Len())
	var tmpEventList, tmpEventList1 []*cl.Event
	for ii := 0; ii < src.NComp(); ii++ {
		tmpEvent := src.GetEvent(ii)
		if tmpEvent != nil {
			tmpEventList = append(tmpEventList, tmpEvent)
		}
	}
	for ii := 0; ii < src.NComp(); ii++ {
		event := k_compress_cmplx_async(dst.DevPtr(ii), src.DevPtr(ii), cnt, iOff, oOff, reduceintcfg, tmpEventList)
		dst.SetEvent(ii, event)
		src.SetEvent(ii, event)
		tmpEventList1 = append(tmpEventList1, event)
	}

	if err := cl.WaitForEvents(tmpEventList1); err != nil {
		fmt.Printf("WaitForEvents failed in packcmplxarray: %+v \n", err)
	}
}
