package opencl

import (
	"fmt"

	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/util"
)

// Convert Hermitian FFT array to full complex array
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

// Convert real array to full complex array
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

//Trans_pose Transpose Array
func Trans_pose(dst, src *data.Slice, src_rows, src_col int) {
	util.Argument(src.NComp() == dst.NComp())
	util.Argument(src_rows >= 0)
	util.Argument(src_col >= 0)
	util.Argument(dst.Len() >= src_rows*src_col)
	//cfg := make1DConf(int(src_rows * src_col / 2))
	var tmpEventList, tmpEventList1 []*cl.Event
	for ii := 0; ii < src.NComp(); ii++ {
		tmpEvent := src.GetEvent(ii)
		if tmpEvent != nil {
			tmpEventList = append(tmpEventList, tmpEvent)
		}
	}
	for ii := 0; ii < src.NComp(); ii++ {
		fmt.Printf("\n Calling the actual function now \n")
		//event := k_trans_pose_async(dst.DevPtr(ii), src.DevPtr(ii), 0, src_col, src_rows, reduceintcfg, tmpEventList)
		event := k_trans_pose_async(dst.DevPtr(ii), src.DevPtr(ii), 0, src_col, src_rows, reducecfg, tmpEventList)
		dst.SetEvent(ii, event)
		src.SetEvent(ii, event)
		tmpEventList1 = append(tmpEventList1, event)
	}

	if err := cl.WaitForEvents(tmpEventList1); err != nil {
		fmt.Printf("WaitForEvents failed in transpose: %+v \n", err)
	}
}
