package tensor

import (
	"fmt"
	"reflect"
	"testing"
)

func equalSlice(a, b []uint32) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}

func TestNewTensor(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	shape := []uint32{2, 3, 2}
	tensor := NewTensor(data, shape)

	if !equalSlice(tensor.shape, shape) {
		t.Errorf("Expected shape %v, but got %v", shape, tensor.shape)
	}

	if tensor.length != 12 {
		t.Errorf("Expected length 12, but got %d", tensor.length)
	}

	for i := uint32(0); i < tensor.Size(); i++ {
		if !floatEq(tensor.data[i], data[i], 1e-5) {
			t.Errorf("Expected data %v, but got %v", data, tensor.data)
		}
	}

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("NewTensor should panic when data length not match shape")
		}
	}()

	NewTensor([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}, shape) // Should panic
}

func TestEmptyTensor(t *testing.T) {
	shape := []uint32{2, 3, 4}
	tensor := EmptyTensor[uint32](shape)

	if !equalSlice(tensor.shape, shape) {
		t.Errorf("Expected shape %v, but got %v", shape, tensor.shape)
	}

	if tensor.length != 24 {
		t.Errorf("Expected length 24, but got %d", tensor.length)
	}
}

func TestNewTensor_Boundary(t *testing.T) {
	// Test case 1: Empty tensor with zero elements (empty data and shape)
	dataEmpty := []float32{}
	shapeEmpty := []uint32{}
	tensorEmpty := NewTensor(dataEmpty, shapeEmpty)
	if !equalSlice(tensorEmpty.shape, shapeEmpty) {
		t.Errorf("Empty tensor: expected shape %v, got %v", shapeEmpty, tensorEmpty.shape)
	}
	if tensorEmpty.length != 0 {
		t.Errorf("Empty tensor: expected length 0, got %d", tensorEmpty.length)
	}

	// Test case 2: Single-element tensor (1D shape)
	dataSingle := []float32{100}
	shapeSingle := []uint32{1}
	tensorSingle := NewTensor(dataSingle, shapeSingle)
	if !floatEq(tensorSingle.data[0], 100, 1e-5) {
		t.Errorf("Single element tensor: expected data [100], got %v", tensorSingle.data)
	}
	if tensorSingle.length != 1 {
		t.Errorf("Single element tensor: expected length 1, got %d", tensorSingle.length)
	}

	// Test case 3: High-dimensional tensor (4D) to verify total element count calculation
	shape4D := []uint32{2, 1, 3, 2} // 2*1*3*2 = 12 elements
	data4D := make([]float32, 12)
	for i := 0; i < 12; i++ {
		data4D[i] = float32(i)
	}
	tensor4D := NewTensor(data4D, shape4D)
	if tensor4D.length != 12 {
		t.Errorf("4D tensor: expected length 12, got %d", tensor4D.length)
	}

	// Test case 4: Zero-length data with non-zero shape (should panic)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for 0-length data with non-zero shape, but no panic")
		}
	}()
	NewTensor([]float32{}, []uint32{2}) // Shape requires 2 elements, but data is empty
}

func TestEmptyTensor_Boundary(t *testing.T) {
	// Test case 1: Empty tensor with empty shape (zero elements)
	shapeEmpty := []uint32{}
	tensorEmpty := EmptyTensor[float32](shapeEmpty)
	if uint32(len(tensorEmpty.data)) != 0 {
		t.Errorf("Empty shape: expected data length 0, got %d", len(tensorEmpty.data))
	}

	// Test case 2: Large tensor (1,000,000 elements) to verify memory allocation
	shapeLarge := []uint32{1000, 1000} // 1,000,000 elements
	tensorLarge := EmptyTensor[uint32](shapeLarge)
	expectedLen := uint32(1000 * 1000)
	if tensorLarge.length != expectedLen {
		t.Errorf("Large tensor: expected length %d, got %d", expectedLen, tensorLarge.length)
	}
	if uint32(len(tensorLarge.data)) != expectedLen {
		t.Errorf("Large tensor: expected data length %d, got %d", expectedLen, len(tensorLarge.data))
	}

	// Test case 3: Single-element empty tensor to verify zero value initialization
	shapeSingle := []uint32{1}
	tensorSingle := EmptyTensor[int32](shapeSingle)
	if tensorSingle.data[0] != 0 { // Zero value for int32 is 0
		t.Errorf("Single element empty tensor: expected 0, got %d", tensorSingle.data[0])
	}
}

func TestTensorAccessors(t *testing.T) {
	// Create a 2x3 tensor with data [1, 2, 3, 4, 5, 6]
	data := []float32{1, 2, 3, 4, 5, 6}
	shape := []uint32{2, 3}
	tensor := NewTensor(data, shape)

	// Test Data() Method
	if !reflect.DeepEqual(tensor.Data(), data) {
		t.Errorf("Data() returned %v, expected %v", tensor.Data(), data)
	}

	// Test Shape() Method
	if !equalSlice(tensor.Shape(), shape) {
		t.Errorf("Shape() returned %v, expected %v", tensor.Shape(), shape)
	}

	// Test Size() Method
	expectedSize := uint32(6) // 2x3 = 6
	if tensor.Size() != expectedSize {
		t.Errorf("Size() returned %d, expected %d", tensor.Size(), expectedSize)
	}
}

func TestTensorReshape(t *testing.T) {
	// Test case 1: Valid reshape from 2D to 3D
	tensor := NewTensor([]float32{1, 2, 3, 4, 5, 6}, []uint32{2, 3})
	tensor.Reshape([]uint32{2, 1, 3}) // 2x1x3 = 6 elements

	if !equalSlice(tensor.Shape(), []uint32{2, 1, 3}) {
		t.Errorf("Reshape failed: expected shape [2 1 3], got %v", tensor.Shape())
	}

	// Test case 2: Valid reshape to 1D
	tensor.Reshape([]uint32{6})
	if !equalSlice(tensor.Shape(), []uint32{6}) {
		t.Errorf("Reshape failed: expected shape [6], got %v", tensor.Shape())
	}

	// Test case 3: Invalid reshape (size mismatch)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Reshape did not panic on invalid size")
		}
	}()
	tensor.Reshape([]uint32{2, 2}) // 2x2 = 4 (mismatch with 6 elements)
}

func TestTensorAt(t *testing.T) {
	// 3D tensor: [2, 2, 2] = 8 elements
	tensor := NewTensor([]float32{1, 2, 3, 4, 5, 6, 7, 8}, []uint32{2, 2, 2})

	// Test case 1: Valid indices
	testCases := []struct {
		indices []uint32
		want    float32
	}{
		{[]uint32{0, 0, 0}, 1},
		{[]uint32{0, 0, 1}, 2},
		{[]uint32{0, 1, 0}, 3},
		{[]uint32{0, 1, 1}, 4},
		{[]uint32{1, 0, 0}, 5},
		{[]uint32{1, 0, 1}, 6},
		{[]uint32{1, 1, 0}, 7},
		{[]uint32{1, 1, 1}, 8},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("index_%v", tc.indices), func(t *testing.T) {
			got := *tensor.At(tc.indices...)
			if got != tc.want {
				t.Errorf("At(%v) = %v, want %v", tc.indices, got, tc.want)
			}
		})
	}

	// Test case 2: Index out of range
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("At did not panic on out-of-range index")
		}
	}()

	tensor.At(1, 1, 2) // 3rd dimension is 2 (0-1)
}

func TestTensorAt_WrongIndexCount(t *testing.T) {
	tensor := NewTensor([]float32{1, 2, 3, 4, 5, 6, 7, 8}, []uint32{2, 2, 2})
	// Test case 3: Incorrect number of indices
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("At did not panic on incorrect index count")
		}
	}()
	tensor.At(1, 1) // Requires 3 indices for 3D tensor
}

func TestReshapeAndAt(t *testing.T) {
	// Create a 2x3 tensor
	originalData := []float32{1, 2, 3, 4, 5, 6}
	tensor := NewTensor(originalData, []uint32{2, 3})

	// Reshape to 3x2
	tensor.Reshape([]uint32{3, 2})

	// Verify index access after reshaping
	testCases := []struct {
		indices []uint32 // Indices in the new shape
		want    float32  // Expected value from original data
	}{
		{[]uint32{0, 0}, 1}, // Corresponds to original position [0,0]
		{[]uint32{0, 1}, 2}, // Corresponds to original position [0,1]
		{[]uint32{1, 0}, 3}, // Corresponds to original position [0,2]
		{[]uint32{1, 1}, 4}, // Corresponds to original position [1,0]
		{[]uint32{2, 0}, 5}, // Corresponds to original position [1,1]
		{[]uint32{2, 1}, 6}, // Corresponds to original position [1,2]
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("reshaped_index_%v", tc.indices), func(t *testing.T) {
			got := *tensor.At(tc.indices...)
			if got != tc.want {
				t.Errorf("After reshape, At(%v) = %v, want %v", tc.indices, got, tc.want)
			}
		})
	}

	// Test high-dimensional reshape (2x3 -> 2x1x3)
	tensor.Reshape([]uint32{2, 1, 3})
	testCases3D := []struct {
		indices []uint32
		want    float32
	}{
		{[]uint32{0, 0, 0}, 1},
		{[]uint32{0, 0, 1}, 2},
		{[]uint32{0, 0, 2}, 3},
		{[]uint32{1, 0, 0}, 4},
		{[]uint32{1, 0, 1}, 5},
		{[]uint32{1, 0, 2}, 6},
	}

	for _, tc := range testCases3D {
		t.Run(fmt.Sprintf("3D_reshaped_index_%v", tc.indices), func(t *testing.T) {
			got := *tensor.At(tc.indices...)
			if got != tc.want {
				t.Errorf("After 3D reshape, At(%v) = %v, want %v", tc.indices, got, tc.want)
			}
		})
	}
}

func TestCloseTo(t *testing.T) {
	// Test case 1: Different number of dimensions
	t1 := NewTensor([]float32{1, 2}, []uint32{2})          // 1D tensor
	t2 := NewTensor([]float32{1, 2, 3, 4}, []uint32{2, 2}) // 2D tensor
	if _, err := t1.CloseTo(t2, 0.1); err == nil {
		t.Error("Expected error for different dimension counts, got nil")
	} else if err.Error() != "tensors must have the same number of dimensions,dimensions: 1 != 2" {
		t.Errorf("Unexpected error message: %v", err)
	}

	// Test case 2: Same dimension count but different shape
	t3 := NewTensor([]float32{1, 2, 3}, []uint32{3}) // 1D with length 3
	t4 := NewTensor([]float32{1, 2}, []uint32{2})    // 1D with length 2
	if _, err := t3.CloseTo(t4, 0.1); err == nil {
		t.Error("Expected error for shape mismatch, got nil")
	} else if err.Error() != "tensors must have the same shape" {
		t.Errorf("Unexpected error message: %v", err)
	}

	// Test case 3: Same shape but different data length (theoretical edge case)
	// Manually create tensors to bypass NewTensor's length check
	t5 := &Tensor[float32]{data: []float32{1, 2}, shape: []uint32{2}, length: 2}
	t6 := &Tensor[float32]{data: []float32{1}, shape: []uint32{2}, length: 2} // Invalid data length
	if _, err := t5.CloseTo(t6, 0.1); err == nil {
		t.Error("Expected error for data length mismatch, got nil")
	} else if err.Error() != "tensors must have the same length" {
		t.Errorf("Unexpected error message: %v", err)
	}

	// Test case 4: All elements within tolerance
	t7 := NewTensor([]float32{1.0, 2.5, 3.0}, []uint32{3})
	t8 := NewTensor([]float32{1.05, 2.45, 3.02}, []uint32{3})
	if match, err := t7.CloseTo(t8, 0.05); !match || err != nil {
		t.Errorf("Expected match=true, err=nil, got match=%v, err=%v", match, err)
	}

	// Test case 5: One element outside tolerance
	t9 := NewTensor([]float32{1.0, 2.0, 3.0}, []uint32{3})
	t10 := NewTensor([]float32{1.0, 2.2, 3.0}, []uint32{3})
	if match, err := t9.CloseTo(t10, 0.05); match || err != nil {
		t.Errorf("Expected match=false, err=nil, got match=%v, err=%v", match, err)
	}

	// Test case 6: Integer type tensors (using TensorDataType constraint)
	t11 := NewTensor([]int32{5, 10, 15}, []uint32{3})
	t12 := NewTensor([]int32{5, 10, 15}, []uint32{3})
	if match, err := t11.CloseTo(t12, 0); !match || err != nil {
		t.Errorf("Expected match=true for identical integers, got match=%v, err=%v", match, err)
	}
}
