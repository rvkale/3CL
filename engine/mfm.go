package engine

import (
	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/data"
)

var (
	MFM        = NewScalarField("MFM", "arb.", "MFM image", SetMFM)
	MFMLift    inputValue
	MFMTipSize inputValue
	mfmconv_   *opencl.MFMConvolution
)

func init() {
	MFMLift = numParam(50e-9, "MFMLift", "m", reinitmfmconv)
	MFMTipSize = numParam(1e-3, "MFMDipole", "m", reinitmfmconv)
	DeclLValue("MFMLift", &MFMLift, "MFM lift height")
	DeclLValue("MFMDipole", &MFMTipSize, "Height of vertically magnetized part of MFM tip")
}

func SetMFM(dst *data.Slice) {
	buf := opencl.Buffer(3, Mesh().Size())
	defer opencl.Recycle(buf)
	if mfmconv_ == nil {
		reinitmfmconv()
	}

	mfmconv_.Exec(buf, M.Buffer(), geometry.Gpu(), Bsat.gpuLUT1(), regions.Gpu())
	opencl.Madd3(dst, buf.Comp(0), buf.Comp(1), buf.Comp(2), 1, 1, 1)
}

func reinitmfmconv() {
	SetBusy(true)
	defer SetBusy(false)
	if mfmconv_ == nil {
		mfmconv_ = opencl.NewMFM(Mesh(), MFMLift.v, MFMTipSize.v)
	} else {
		mfmconv_.Reinit(MFMLift.v, MFMTipSize.v)
	}
}