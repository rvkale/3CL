package opencl

import "github.com/mumax/3cl/opencl/cl"

// OpenCL Launch parameters.
// there might be better choices for recent hardware,
// but it barely makes a difference in the end.
const (
	BlockSize    = 512
	TileX, TileY = 32, 32
	MaxGridSize  = 65535
)

// cuda launch configuration
type config struct {
	Grid, Block cl.Dim3
}

// Make a 1D kernel launch configuration suited for N threads.
func make1DConf(N int) *config {
	bl := cl.Dim3{X: BlockSize, Y: 1, Z: 1}

	n2 := divUp(N, BlockSize) // N2 blocks left
	nx := divUp(n2, MaxGridSize)
	ny := divUp(n2, nx)
	gr := cl.Dim3{X: nx, Y: ny, Z: 1}

	return &config{gr, bl}
}

// Make a 3D kernel launch configuration suited for N threads.
func make3DConf(N [3]int) *config {
	bl := cl.Dim3{X: TileX, Y: TileY, Z: 1}

	nx := divUp(N[X], TileX)
	ny := divUp(N[Y], TileY)
	gr := cl.Dim3{X: nx, Y: ny, Z: N[Z]}

	return &config{gr, bl}
}

// integer minimum
func iMin(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Integer division rounded up.
func divUp(x, y int) int {
	return ((x - 1) / y) + 1
}

const (
	X = 0
	Y = 1
	Z = 2
)
