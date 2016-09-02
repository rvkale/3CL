package engine

import (
	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/data"
)

var (
	MaxAngle  = NewScalarValue("MaxAngle", "rad", "maximum angle between neighboring spins", GetMaxAngle) // TODO: Max(...)
	SpinAngle = NewScalarField("spinAngle", "rad", "Angle between neighboring spins", SetSpinAngle)
)

func SetSpinAngle(dst *data.Slice) {
	opencl.SetMaxAngle(dst, M.Buffer(), lex2.Gpu(), regions.Gpu(), M.Mesh())
}

func GetMaxAngle() float64 {
	s, recycle := SpinAngle.Slice()
	if recycle {
		defer opencl.Recycle(s)
	}
	return float64(opencl.MaxAbs(s)) // just a max would be fine, but not currently implemented
}
