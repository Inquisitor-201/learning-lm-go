package tensor

import (
	"errors"
	"fmt"
	"math"
)

type Tensor[T TensorDataType] struct {
	data   []T
	shape  []uint32
	length uint32
}

type TensorDataType interface {
	FloatDataType | ~int32 | ~int64 | ~uint32 | ~uint64
}

type FloatDataType interface {
	~float32 | ~float64
}

func NewTensor[T TensorDataType](data []T, shape []uint32) *Tensor[T] {
	total := calculateSize(shape)
	// check that the length of the data is the same as the total number of elements
	if len(data) != int(total) {
		panic("data length does not match the total number of elements in the tensor")
	}
	return &Tensor[T]{data: data, shape: shape, length: total}
}

func EmptyTensor[T TensorDataType](shape []uint32) *Tensor[T] {
	total := calculateSize(shape)
	return &Tensor[T]{data: make([]T, int(total)), shape: shape, length: total}
}

func (t *Tensor[T]) Data() []T {
	return t.data
}

func (t *Tensor[T]) Shape() []uint32 {
	return t.shape
}

func (t *Tensor[T]) Size() uint32 {
	return t.length
}

func (t *Tensor[T]) Reshape(newShape []uint32) {
	newSize := calculateSize(newShape)
	if newSize != t.length {
		panic("new shape does not match the length of the tensor")
	}
	t.shape = newShape
}

func (t *Tensor[T]) At(index ...uint32) *T {
	if len(index) != len(t.shape) {
		panic("index length does not match the number of dimensions in the tensor")
	}
	offset := uint32(0)
	for i := 0; i < len(t.shape); i++ {
		if index[i] >= t.shape[i] {
			panic(fmt.Sprintf("index %d out of range", index[i]))
		}
		offset = offset*t.shape[i] + index[i]
	}

	return &t.data[offset]
}

func (t *Tensor[T]) CloseTo(other *Tensor[T], rel float32) (bool, error) {
	if len(t.shape) != len(other.shape) {
		return false, fmt.Errorf("tensors must have the same number of dimensions,"+
			"dimensions: %d != %d", len(t.shape), len(other.shape))
	}
	for i, dim := range t.shape {
		if dim != other.shape[i] {
			return false, errors.New("tensors must have the same shape")
		}
	}
	if len(t.data) != len(other.data) {
		return false, errors.New("tensors must have the same length")
	}
	// type assertion to check if T is a float type
	for i := range t.data {
		if !floatEq(float32(t.data[i]), float32(other.data[i]), rel) {
			return false, nil
		}
	}
	return true, nil
}

func floatEq(a, b, rel float32) bool {
	absDiff := math.Abs(float64(a - b))
	return absDiff <= float64(rel)*(math.Abs(float64(a))+math.Abs(float64(b)))/2.0
}

func calculateSize(shape []uint32) uint32 {
	if len(shape) == 0 {
		return 0
	}
	length := uint32(1)
	for _, dim := range shape {
		length *= dim
	}
	return length
}
