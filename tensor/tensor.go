package tensor

import (
	"errors"
	"fmt"
	"math"
)

type Tensor[T Float] struct {
	data   []T
	shape  []int
	length int
}

type Float interface {
	float32 | float64
}

func NewTensor[T Float](data []T, shape []int) *Tensor[T] {
	total := calculateSize(shape)
	// check that the length of the data is the same as the total number of elements
	if len(data) != total {
		panic("data length does not match the total number of elements in the tensor")
	}
	return &Tensor[T]{data: data, shape: shape, length: total}
}

func EmptyTensor[T Float](shape []int) *Tensor[T] {
	total := calculateSize(shape)
	return &Tensor[T]{data: make([]T, total), shape: shape, length: total}
}

func (t *Tensor[T]) Data() []T {
	return t.data
}

func (t *Tensor[T]) Shape() []int {
	return t.shape
}

func (t *Tensor[T]) Size() int {
	return t.length
}

func (t *Tensor[T]) Reshape(newShape []int) int {
	newSize := calculateSize(newShape)
	if newSize != t.length {
		panic("new shape does not match the length of the tensor")
	}
	t.shape = newShape
	return newSize
}

func (t *Tensor[float32]) CloseTo(other *Tensor[float32], rel float32) (bool, error) {
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
	for i := range t.data {
		if floatEq(t.data[i], other.data[i], rel) {
			return false, nil
		}
	}
	return true, nil
}

func floatEq[T Float](a, b, rel T) bool {
	absDiff := math.Abs(float64(a - b))
	return absDiff <= float64(rel)*(math.Abs(float64(a))+math.Abs(float64(b)))/2.0
}

func calculateSize(shape []int) int {
	length := 1
	for _, dim := range shape {
		length *= dim
	}
	return length
}
