package tensor

import (
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
		indicesLarge.data[i] = uint32(rows - 1 - i) // Reverse order indices
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

	// Test case 5: Input is not a 2D tensor (should panic)
	input3D := NewTensor([]float32{1, 2, 3, 4}, []uint32{2, 2, 1})
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Gather did not panic when input is not 2D")
		}
	}()
	Gather(input3D, indicesSmall)

	// Test case 6: Indices is not a 1D tensor (should panic)
	indices2D := NewTensor([]uint32{1, 0}, []uint32{1, 2})
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Gather did not panic when indices is not 1D")
		}
	}()
	Gather(inputSmall, indices2D)

	// Test case 7: Out-of-bounds indices (should panic)
	indicesOutOfBounds := NewTensor([]uint32{5}, []uint32{1})
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Gather did not panic when indices are out of bounds")
		}
	}()
	Gather(inputSmall, indicesOutOfBounds)
}
