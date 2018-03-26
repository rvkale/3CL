package engine

import (
	"github.com/mumax/3/opencl"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"math"
)

type RK56 struct {
}

func (rk *RK56) Step() {

	m := M.Buffer()
	size := m.Size()

	if FixDt != 0 {
		Dt_si = FixDt
	}

	t0 := Time
	// backup magnetization
	m0 := opencl.Buffer(3, size)
	defer opencl.Recycle(m0)
	data.Copy(m0, m)

	k1, k2, k3, k4, k5, k6, k7, k8 := opencl.Buffer(3, size), opencl.Buffer(3, size), opencl.Buffer(3, size), opencl.Buffer(3, size), opencl.Buffer(3, size), opencl.Buffer(3, size), opencl.Buffer(3, size), opencl.Buffer(3, size)
	defer opencl.Recycle(k1)
	defer opencl.Recycle(k2)
	defer opencl.Recycle(k3)
	defer opencl.Recycle(k4)
	defer opencl.Recycle(k5)
	defer opencl.Recycle(k6)
	defer opencl.Recycle(k7)
	defer opencl.Recycle(k8)
	//k2 will be recyled as k9

	h := float32(Dt_si * GammaLL) // internal time step = Dt * gammaLL

	// stage 1
	torqueFn(k1)

	// stage 2
	Time = t0 + (1./6.)*Dt_si
	opencl.Madd2(m, m, k1, 1, (1./6.)*h) // m = m*1 + k1*h/6
	M.normalize()
	torqueFn(k2)

	// stage 3
	Time = t0 + (4./15.)*Dt_si
	opencl.Madd3(m, m0, k1, k2, 1, (4./75.)*h, (16./75.)*h)
	M.normalize()
	torqueFn(k3)

	// stage 4
	Time = t0 + (2./3.)*Dt_si
	madd4(m, m0, k1, k2, k3, 1, (5./6.)*h, (-8./3.)*h, (5./2.)*h)
	M.normalize()
	torqueFn(k4)

	// stage 5
	Time = t0 + (4./5.)*Dt_si
	madd5(m, m0, k1, k2, k3, k4, 1, (-8./5.)*h, (144./25.)*h, (-4.)*h, (16./25.)*h)
	M.normalize()
	torqueFn(k5)

	// stage 6
	Time = t0 + (1.)*Dt_si
	madd6(m, m0, k1, k2, k3, k4, k5, 1, (361./320.)*h, (-18./5.)*h, (407./128.)*h, (-11./80.)*h, (55./128.)*h)
	M.normalize()
	torqueFn(k6)

	// stage 7
	Time = t0
	madd5(m, m0, k1, k3, k4, k5, 1, (-11./640.)*h, (11./256.)*h, (-11/160.)*h, (11./256.)*h)
	M.normalize()
	torqueFn(k7)

	// stage 8
	Time = t0 + (1.)*Dt_si
	madd7(m, m0, k1, k2, k3, k4, k5, k7, 1, (93./640.)*h, (-18./5.)*h, (803./256.)*h, (-11./160.)*h, (99./256.)*h, (1.)*h)
	M.normalize()
	torqueFn(k8)


	// stage 9: 6th order solution
	Time = t0 + (1.)*Dt_si
	//madd6(m, m0, k1, k3, k4, k5, k6, 1, (31./384.)*h, (1125./2816.)*h, (9./32.)*h, (125./768.)*h, (5./66.)*h)
	madd7(m, m0, k1, k3, k4, k5, k7,k8, 1, (7./1408.)*h, (1125./2816.)*h, (9./32.)*h, (125./768.)*h, (5./66.)*h, (5./66.)*h)
	M.normalize()
	torqueFn(k2) // re-use k2

	// error estimate
	Err := opencl.Buffer(3, size)
	defer opencl.Recycle(Err)
	madd4(Err, k1, k6, k7, k8, (-5./66.), (-5./66.), (5./66.), (5./66.))

	// determine error
	err := opencl.MaxVecNorm(Err) * float64(h)

	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		setLastErr(err)
		setMaxTorque(k2)
		NSteps++
		Time = t0 + Dt_si
		adaptDt(math.Pow(MaxErr/err, 1./6.))
	} else {
		// undo bad step
		//util.Println("Bad step at t=", t0, ", err=", err)
		util.Assert(FixDt == 0)
		Time = t0
		data.Copy(m, m0)
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./7.))
	}
}

func (rk *RK56) Free() {
}

// TODO: into opencl

func madd7(dst, src1, src2, src3, src4, src5, src6, src7 *data.Slice, w1, w2, w3, w4, w5, w6, w7 float32) {
	65s
	madd6(dst, src1, src2, src3, src4, src5,src6, w1, w2, w3, w4, w5,w6)
	cuda.Madd2(dst, dst, src7, 1, w7)
}
