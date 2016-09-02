package engine

// Total energy calculation

import (
	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/data"
)

// TODO: Integrate(Edens)
// TODO: consistent naming SetEdensTotal, ...

var (
	energyTerms []func() float64        // all contributions to total energy
	edensTerms  []func(dst *data.Slice) // all contributions to total energy density (add to dst)
	Edens_total = NewScalarField("Edens_total", "J/m3", "Total energy density", SetTotalEdens)
	E_total     = NewScalarValue("E_total", "J", "total energy", GetTotalEnergy)
)

// add energy term to global energy
func registerEnergy(term func() float64, dens func(*data.Slice)) {
	energyTerms = append(energyTerms, term)
	edensTerms = append(edensTerms, dens)
}

// Returns the total energy in J.
func GetTotalEnergy() float64 {
	E := 0.
	for _, f := range energyTerms {
		E += f()
	}
	checkNaN1(E)
	return E
}

// Set dst to total energy density in J/m3
func SetTotalEdens(dst *data.Slice) {
	opencl.Zero(dst)
	for _, addTerm := range edensTerms {
		addTerm(dst)
	}
}

// volume of one cell in m3
func cellVolume() float64 {
	c := Mesh().CellSize()
	return c[0] * c[1] * c[2]
}

// returns a function that adds to dst the energy density:
// 	prefactor * dot (M_full, field)
func makeEdensAdder(field outputField, prefactor float64) func(*data.Slice) {
	return func(dst *data.Slice) {
		B, r1 := field.Slice()
		if r1 {
			defer opencl.Recycle(B)
		}
		m, r2 := M_full.Slice()
		if r2 {
			defer opencl.Recycle(m)
		}
		factor := float32(prefactor)
		opencl.AddDotProduct(dst, factor, B, m)
	}
}

// vector dot product
func dot(a, b outputField) float64 {
	A, recyA := a.Slice()
	if recyA {
		defer opencl.Recycle(A)
	}
	B, recyB := b.Slice()
	if recyB {
		defer opencl.Recycle(B)
	}
	return float64(opencl.Dot(A, B))
}
