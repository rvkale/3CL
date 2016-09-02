package mag

import (
	"fmt"
	d "github.com/mumax/3cl/data"
	"github.com/mumax/3cl/util"
	"math"
)

// Kernel for the vertical derivative of the force on an MFM tip due to mx, my, mz.
// This is the 2nd derivative of the energy w.r.t. z.
func MFMKernel(mesh *d.Mesh, lift, tipsize float64) (kernel [3]*d.Slice) {

	const TipCharge = 1 / Mu0 // tip charge
	const Δ = 1e-9            // tip oscillation, take 2nd derivative over this distance
	util.AssertMsg(lift > 0, "MFM tip crashed into sample, please lift the new one higher")

	{ // Kernel mesh is 2x larger than input, instead in case of PBC
		pbc := mesh.PBC()
		sz := padSize(mesh.Size(), pbc)
		cs := mesh.CellSize()
		mesh = d.NewMesh(sz[X], sz[Y], sz[Z], cs[X], cs[Y], cs[Z], pbc[:]...)
	}

	// Shorthand
	size := mesh.Size()
	pbc := mesh.PBC()
	cellsize := mesh.CellSize()
	volume := cellsize[X] * cellsize[Y] * cellsize[Z]
	fmt.Println("calculating MFM kernel")

	// Sanity check
	{
		util.Assert(size[Z] >= 1 && size[Y] >= 2 && size[X] >= 2)
		util.Assert(cellsize[X] > 0 && cellsize[Y] > 0 && cellsize[Z] > 0)
		util.AssertMsg(size[X]%2 == 0 && size[Y]%2 == 0, "Even kernel size needed")
		if size[Z] > 1 {
			util.AssertMsg(size[Z]%2 == 0, "Even kernel size needed")
		}
	}

	// Allocate only upper diagonal part. The rest is symmetric due to reciprocity.
	var K [3][][][]float32
	for i := 0; i < 3; i++ {
		kernel[i] = d.NewSlice(1, mesh.Size())
		K[i] = kernel[i].Scalars()
	}

	r1, r2 := kernelRanges(size, pbc)
	progress, progmax := 0, (1+r2[Y]-r1[Y])*(1+r2[Z]-r1[Z])

	for iz := r1[Z]; iz <= r2[Z]; iz++ {
		zw := wrap(iz, size[Z])
		z := float64(iz) * cellsize[Z]

		for iy := r1[Y]; iy <= r2[Y]; iy++ {
			yw := wrap(iy, size[Y])
			y := float64(iy) * cellsize[Y]
			progress++
			util.Progress(progress, progmax, "Calculating MFM kernel")

			for ix := r1[X]; ix <= r2[X]; ix++ {
				x := float64(ix) * cellsize[X]
				xw := wrap(ix, size[X])

				for s := 0; s < 3; s++ { // source index Ksxyz
					m := d.Vector{0, 0, 0}
					m[s] = 1

					var E [3]float64 // 3 energies for 2nd derivative

					for i := -1; i <= 1; i++ {
						I := float64(i)
						R := d.Vector{-x, -y, z - (lift + (I * Δ))}
						r := R.Len()
						B := R.Mul(TipCharge / (4 * math.Pi * r * r * r))

						R = d.Vector{-x, -y, z - (lift + tipsize + (I * Δ))}
						r = R.Len()
						B = B.Add(R.Mul(-TipCharge / (4 * math.Pi * r * r * r)))

						E[i+1] = B.Dot(m) * volume // i=-1 stored in  E[0]
					}

					dFdz_tip := ((E[0] - E[1]) + (E[2] - E[1])) / (Δ * Δ) // dFz/dz = d2E/dz2

					K[s][zw][yw][xw] += float32(dFdz_tip) // += needed in case of PBC
				}
			}
		}
	}

	return kernel
}