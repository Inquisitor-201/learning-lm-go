package tensor

import (
	"math"
	"reflect"
	"testing"
)

func TestGather(t *testing.T) {
	// Test case 1: Normal scenario with small matrix
	inputSmall := NewTensor([]float32{
		1, 2, // Row 0
		3, 4, // Row 1
		5, 6, // Row 2
	}, []uint32{3, 2})

	indicesSmall := NewTensor([]uint32{1, 0, 2, 1}, []uint32{4})
	expectedSmall := NewTensor([]float32{
		3, 4, // Row at index 1
		1, 2, // Row at index 0
		5, 6, // Row at index 2
		3, 4, // Row at index 1 (duplicated)
	}, []uint32{4, 2})

	outputSmall := Gather(inputSmall, indicesSmall)
	if !reflect.DeepEqual(outputSmall.Data(), expectedSmall.Data()) {
		t.Errorf("Small matrix gather failed: expected %v, got %v",
			expectedSmall.Data(), outputSmall.Data())
	}

	// Test case 2: Large matrix with sequential indices
	const rows = 100
	const cols = 50
	largeData := make([]float32, rows*cols)
	for i := 0; i < rows*cols; i++ {
		largeData[i] = float32(i)
	}
	inputLarge := NewTensor(largeData, []uint32{uint32(rows), uint32(cols)})

	indicesLarge := EmptyTensor[uint32]([]uint32{uint32(rows)})
	for i := uint32(0); i < rows; i++ {
		indicesLarge.Data()[i] = uint32(rows - 1 - i) // Reverse order indices
	}

	expectedLargeData := make([]float32, rows*cols)
	for i := uint32(0); i < rows; i++ {
		srcIdx := (rows - 1 - i) * cols
		dstIdx := i * cols
		copy(expectedLargeData[dstIdx:dstIdx+cols], largeData[srcIdx:srcIdx+cols])
	}
	expectedLarge := NewTensor(expectedLargeData, []uint32{uint32(rows), uint32(cols)})

	outputLarge := Gather(inputLarge, indicesLarge)
	if !reflect.DeepEqual(outputLarge.Data(), expectedLarge.Data()) {
		t.Errorf("Large matrix gather failed: expected %v...(truncated), got %v...(truncated)",
			expectedLarge.Data()[:10], outputLarge.Data()[:10])
	}

	// Test case 3: Indices with repeated elements
	inputRepeated := NewTensor([]float32{
		10, 20, 30,
		40, 50, 60,
		70, 80, 90,
	}, []uint32{3, 3})

	indicesRepeated := NewTensor([]uint32{0, 0, 2, 2, 1}, []uint32{5})
	expectedRepeated := NewTensor([]float32{
		10, 20, 30,
		10, 20, 30,
		70, 80, 90,
		70, 80, 90,
		40, 50, 60,
	}, []uint32{5, 3})

	outputRepeated := Gather(inputRepeated, indicesRepeated)
	if !reflect.DeepEqual(outputRepeated.Data(), expectedRepeated.Data()) {
		t.Errorf("Repeated indices gather failed: expected %v, got %v",
			expectedRepeated.Data(), outputRepeated.Data())
	}

	// Test case 4: Empty indices
	indicesEmpty := NewTensor([]uint32{}, []uint32{0})
	expectedEmpty := EmptyTensor[float32]([]uint32{0, 3})
	outputEmpty := Gather(inputRepeated, indicesEmpty)
	if !reflect.DeepEqual(outputEmpty.Data(), expectedEmpty.Data()) {
		t.Errorf("Empty indices gather failed: expected %v, got %v",
			expectedEmpty.Data(), outputEmpty.Data())
	}
}

// TestGatherPanic_InputNot2D verifies panic when input is not a 2D tensor
func TestGatherPanic_InputNot2D(t *testing.T) {
	input3D := NewTensor([]float32{1, 2, 3, 4}, []uint32{2, 2, 1}) // 3D tensor (invalid)
	indices := NewTensor([]uint32{1, 0, 2, 1}, []uint32{4})        // Valid 1D indices

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Gather did not panic when input is not 2D")
		}
	}()

	Gather(input3D, indices) // Should trigger panic
}

// TestGatherPanic_IndicesNot1D verifies panic when indices are not a 1D tensor
func TestGatherPanic_IndicesNot1D(t *testing.T) {
	input := NewTensor([]float32{1, 2, 3, 4, 5, 6}, []uint32{3, 2}) // Valid 2D input
	indices2D := NewTensor([]uint32{1, 0}, []uint32{1, 2})          // 2D indices (invalid)

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Gather did not panic when indices are not 1D")
		}
	}()

	Gather(input, indices2D) // Should trigger panic
}

// TestGatherPanic_IndicesOutOfBounds verifies panic when indices are out of bounds
func TestGatherPanic_IndicesOutOfBounds(t *testing.T) {
	input := NewTensor([]float32{1, 2, 3, 4, 5, 6}, []uint32{3, 2}) // 3 rows (indices 0-2)
	indicesOutOfBounds := NewTensor([]uint32{5}, []uint32{1})       // Index 5 is out of range

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Gather did not panic when indices are out of bounds")
		}
	}()

	Gather(input, indicesOutOfBounds) // Should trigger panic
}

func TestMaskedSoftmax_1(t *testing.T) {
	input := NewTensor([]float32{
		1, 1, 1,
		1, -1, 0,
		1, 2, 3}, []uint32{1, 3, 3}) // 3 rows (indices 0-2)
	expected := NewTensor([]float32{
		1, 0, 0,
		0.880797078, 0.119202922, 0,
		0.09003057317, 0.2447284711, 0.6652409558}, []uint32{1, 3, 3}) // 3 rows (indices 0-2)

	MaskedSoftmax(input)

	res, err := input.CloseTo(expected, 1e-6)
	if err != nil {
		t.Errorf("MaskedSoftmax paniced: %v", err)
	}
	if !res {
		t.Errorf("MaskedSoftmax failed: expected %v, got %v", expected, input)
	}
}

func TestMaskedSoftmax_2(t *testing.T) {
	input := NewTensor([]float32{
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
	}, []uint32{1, 3, 5},
	) // 3 rows (indices 0-2)

	expected := NewTensor([]float32{
		1. / 3, 1. / 3, 1. / 3, 0, 0,
		1. / 4, 1. / 4, 1. / 4, 1. / 4, 0,
		1. / 5, 1. / 5, 1. / 5, 1. / 5, 1. / 5,
	}, []uint32{1, 3, 5},
	) // 3 rows (indices 0-2)

	MaskedSoftmax(input)

	res, err := input.CloseTo(expected, 1e-6)
	if err != nil {
		t.Errorf("MaskedSoftmax paniced: %v", err)
	}
	if !res {
		t.Errorf("MaskedSoftmax failed: expected %v, got %v", expected, input)
	}
}

func TestSwiGLu(t *testing.T) {
	x := NewTensor([]float32{1.0, 2.0, 3.0}, []uint32{1, 3})
	y := NewTensor([]float32{2.0, 3.0, 4.0}, []uint32{1, 3})

	SwiGLu(y, x)
	expected := NewTensor([]float32{1.4621172, 5.2847824, 11.43089}, []uint32{1, 3})
	// use CloseTo to compare tensors
	res, err := y.CloseTo(expected, 1e-6)
	if err != nil {
		t.Errorf("SwiGLu panicked: %v", err)
	}
	if !res {
		t.Errorf("SwiGLu failed: expected %v, got %v", expected, y)
	}
}

func TestRMSNorm(t *testing.T) {
	x := NewTensor([]float32{1.0, 2.0, 3.0, 4.0}, []uint32{2, 2})
	w := NewTensor([]float32{1.0, 2.0}, []uint32{2})
	expected := NewTensor([]float32{0.6324554, 2.5298216, 0.8485281, 2.2627416}, []uint32{2, 2})
	y := RMSNorm(x, w, 1e-6)
	res, err := y.CloseTo(expected, 1e-6)
	if err != nil {
		t.Errorf("RMSNorm paniced: %v", err)
	}
	if !res {
		t.Errorf("RMSNorm failed: expected %v, got %v", expected, y)
	}
}

func TestRMSNorm_2(t *testing.T) {
	x := NewTensor(
		[]float32{
			0.0, 0.0, 0.1,
			3.0, 4.0, 5.0,
		},
		[]uint32{2, 3})
	w := NewTensor([]float32{1.0, 2.0, 3.0}, []uint32{3})
	expected := NewTensor(
		[]float32{
			0.0, 0.0, 5.195373175,
			0.7348469008, 1.959591735, 3.674234504,
		},
		[]uint32{2, 3})
	y := RMSNorm(x, w, 1e-6)
	res, err := y.CloseTo(expected, 1e-6)
	if err != nil {
		t.Errorf("RMSNorm paniced: %v", err)
	}
	if !res {
		t.Errorf("RMSNorm failed: expected %v, got %v", expected, y)
	}
}

func TestMatMulTransB(t *testing.T) {
	a := NewTensor(
		[]float32{
			1.0, 2.0, 3.0, 4.0,
			5.0, 6.0, 7.0, 8.0,
		},
		[]uint32{2, 4},
	)

	b := NewTensor(
		[]float32{
			2.0, 3.0, 4.0, 5.0,
			6.0, 7.0, 8.0, 9.0,
			10.0, 11.0, 12.0, 13.0,
		},
		[]uint32{3, 4},
	)

	expected := NewTensor(
		[]float32{
			40, 80, 120,
			96, 200, 304,
		},
		[]uint32{2, 3},
	)
	y := MatMulTransB(a, b)
	res, err := y.CloseTo(expected, 1e-6)
	if err != nil {
		t.Errorf("MatMulTransB paniced: %v", err)
	}
	if !res {
		t.Errorf("MatMulTransB failed: expected %v, got %v", expected, y)
	}
}

func TestGroupAttnQK(t *testing.T) {
	a := NewTensor(
		[]float32{
			1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
			2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5,
		},
		[]uint32{2, 4, 3},
	)

	b := NewTensor(
		[]float32{
			1, 1, 1, 2, 2, 2,
		},
		[]uint32{1, 2, 3},
	)

	expected := NewTensor(
		[]float32{
			3, 6,
			6, 9,
			18, 24,
			24, 30,
		},
		[]uint32{4, 2, 1},
	)
	y, _ := GroupAttnQK(a, b)
	ScalarMul(float32(math.Sqrt(3)), y)

	res, err := y.CloseTo(expected, 1e-4)
	if err != nil {
		t.Errorf("MatMulTransB paniced: %v", err)
	}
	if !res {
		t.Errorf("MatMulTransB failed: expected %v, got %v", expected, y)
	}
}

func TestGroupAttnQK_2(t *testing.T) {
	a := NewTensor(
		[]float32{
			1, 1, 2, 2, 3, 3, 4, 4,
			2, 2, 3, 3, 4, 4, 5, 5,
			3, 3, 4, 4, 5, 5, 6, 6,
		},
		[]uint32{3, 4, 2},
	)

	b := NewTensor(
		[]float32{
			1, 1, 2, 2,
			3, 3, 4, 4,
		},
		[]uint32{2, 2, 2},
	)

	expected := NewTensor(
		[]float32{
			2, 6, 4, 12, 6, 18,
			4, 12, 6, 18, 8, 24,
			12, 24, 16, 32, 20, 40,
			16, 32, 20, 40, 24, 48,
		},
		[]uint32{4, 3, 2},
	)
	y, _ := GroupAttnQK(a, b)
	ScalarMul(float32(math.Sqrt(2)), y)

	res, err := y.CloseTo(expected, 1e-4)
	if err != nil {
		t.Errorf("MatMulTransB paniced: %v", err)
	}
	if !res {
		t.Errorf("MatMulTransB failed: expected %v, got %v", expected, y)
	}
}

func TestGroupAttnScore(t *testing.T) {
	a := NewTensor(
		[]float32{
			1, 2, 3, 4,
			2, 3, 4, 5,
		},
		[]uint32{2, 2, 2},
	)
	b := NewTensor(
		[]float32{
			1, 1, 1, 1,
			2, 2, 2, 2,
			3, 3, 3, 3,
		},
		[]uint32{3, 2, 2},
	)
	expected := NewTensor(
		[]float32{
			0.1070418015, 0.8929581985, 0,
			8.24594052e-4, 0.02829456776, 0.9708808382,
			0.007035351085, 0.9929646489, 0,
			2.96199932e-6, 0.001719563085, 0.9982774749,
		},
		[]uint32{2, 2, 3},
	)
	y, _ := GroupAttnScore(a, b)

	if res, err := y.CloseTo(expected, 1e-4); !res || err != nil {
		t.Errorf("GroupAttn failed: expected %v, got %v", expected, y)
	}
}

func TestGroupAttnV(t *testing.T) {
	attn := NewTensor(
		[]float32{
			0.8, 0.2, 0,
			0.5, 0.4, 0.1,
			0.5, 0.5, 0,
			0.1, 0.1, 0.8,
		},
		[]uint32{2, 2, 3},
	)
	v := NewTensor(
		[]float32{
			1, 2,
			2, 1,
			1, 1,
		},
		[]uint32{3, 1, 2},
	)
	expected := NewTensor(
		[]float32{
			1.2, 1.8, 1.5, 1.5,
			1.4, 1.5, 1.1, 1.1,
		},
		[]uint32{2, 2, 2},
	)
	y, _ := GroupAttnV(attn, v)

	if res, err := y.CloseTo(expected, 1e-4); !res || err != nil {
		t.Errorf("GroupAttn failed: expected %v, got %v", expected, y)
	}
}
