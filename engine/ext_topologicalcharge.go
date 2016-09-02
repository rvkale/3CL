package engine

import (
	"math"

	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/data"
)

var (
	Ext_TopologicalCharge        = NewScalarValue("ext_topologicalcharge", "", "2D topological charge", GetTopologicalCharge)
	Ext_TopologicalChargeDensity = NewScalarField("ext_topologicalchargedensity", "1/m2",
		"2D topological charge density m·(m/∂x ❌ ∂m/∂y)", SetTopologicalChargeDensity)
)

func SetTopologicalChargeDensity(dst *data.Slice) {
	opencl.SetTopologicalCharge(dst, M.Buffer(), M.Mesh())
}

func GetTopologicalCharge() float64 {
	s, recycle := Ext_TopologicalChargeDensity.Slice()
	if recycle {
		defer opencl.Recycle(s)
	}
	c := Mesh().CellSize()
	N := Mesh().Size()
	return (0.25 * c[X] * c[Y] / math.Pi / float64(N[Z])) * float64(opencl.Sum(s))
}
