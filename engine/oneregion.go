package engine

import (
	"fmt"
	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/util"
)

func sInRegion(q outputField, r int) ScalarField {
	return AsScalarField(inRegion(q, r))
}

func vInRegion(q outputField, r int) VectorField {
	return AsVectorField(inRegion(q, r))
}

func sOneRegion(q outputField, r int) *sOneReg {
	util.Argument(q.NComp() == 1)
	return &sOneReg{oneReg{q, r}}
}

func vOneRegion(q outputField, r int) *vOneReg {
	util.Argument(q.NComp() == 3)
	return &vOneReg{oneReg{q, r}}
}

type sOneReg struct{ oneReg }

func (q *sOneReg) Average() float64 { return q.average()[0] }

type vOneReg struct{ oneReg }

func (q *vOneReg) Average() data.Vector { return unslice(q.average()) }

// represents a new quantity equal to q in the given region, 0 outside.
type oneReg struct {
	parent outputField
	region int
}

func inRegion(q outputField, region int) outputField {
	return &oneReg{q, region}
}

func (q *oneReg) NComp() int       { return q.parent.NComp() }
func (q *oneReg) Name() string     { return fmt.Sprint(q.parent.Name(), ".region", q.region) }
func (q *oneReg) Unit() string     { return q.parent.Unit() }
func (q *oneReg) Mesh() *data.Mesh { return q.parent.Mesh() }

// returns a new slice equal to q in the given region, 0 outside.
func (q *oneReg) Slice() (*data.Slice, bool) {
	src, r := q.parent.Slice()
	if r {
		defer opencl.Recycle(src)
	}
	out := opencl.Buffer(q.NComp(), q.Mesh().Size())
	opencl.RegionSelect(out, src, regions.Gpu(), byte(q.region))
	return out, true
}

func (q *oneReg) average() []float64 {
	slice, r := q.Slice()
	if r {
		defer opencl.Recycle(slice)
	}
	avg := sAverageUniverse(slice)
	sDiv(avg, regions.volume(q.region))
	return avg
}

func (q *oneReg) Average() []float64 { return q.average() }

// slice division
func sDiv(v []float64, x float64) {
	for i := range v {
		v[i] /= x
	}
}
