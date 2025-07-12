package tensor

import (
	"errors"
	"fmt"
	"math"
	"strings"
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

	return &t.Data()[offset]
}

func (t *Tensor[T]) Slice(offset uint32, shape []uint32) *Tensor[T] {
	newSize := calculateSize(shape)
	if offset+newSize > t.length {
		panic("slice out of range")
	}

	return &Tensor[T]{data: t.Data()[offset : offset+newSize], shape: shape, length: newSize}
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
	if len(t.Data()) != len(other.Data()) {
		return false, errors.New("tensors must have the same length")
	}
	// type assertion to check if T is a float type
	for i := range t.Data() {
		if !FloatEq(float32(t.Data()[i]), float32(other.Data()[i]), rel) {
			return false, nil
		}
	}
	return true, nil
}

func FloatEq(a, b, rel float32) bool {
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

// String returns a formatted and human-readable string representation of the Tensor.
// For large tensors, it automatically truncates the output with ellipsis (...) to maintain readability.
func (t *Tensor[T]) String() string {
	if len(t.shape) == 0 {
		return "[]"
	}

	// Thresholds for truncation
	maxElements := 1000   // Maximum elements to print before truncation
	maxLineElements := 20 // Maximum elements per line before truncation
	maxSlices := 5        // Maximum 2D slices to print for ND tensors

	// Check if the tensor is too large to print fully
	totalElements := int(t.length)
	shouldTruncate := totalElements > maxElements

	var sb strings.Builder

	// 0D tensor (scalar)
	if len(t.shape) == 0 {
		if len(t.Data()) == 0 {
			return "[]"
		}
		return fmt.Sprintf("[%v]", t.Data()[0])
	}

	// 1D tensor (vector)
	if len(t.shape) == 1 {
		sb.WriteString("[")
		for i, v := range t.Data() {
			if i > 0 {
				sb.WriteString(" ")
			}

			// Truncate long vectors
			if shouldTruncate && i >= maxLineElements {
				sb.WriteString("...")
				break
			}

			sb.WriteString(fmt.Sprintf("%v", v))
		}
		sb.WriteString("]")
		return sb.String()
	}

	// 2D tensor (matrix)
	if len(t.shape) == 2 {
		rows, cols := int(t.shape[0]), int(t.shape[1])

		// Calculate maximum width for proper alignment
		maxWidth := 0
		elementsToCheck := totalElements
		if shouldTruncate {
			elementsToCheck = int(math.Min(float64(maxElements), float64(totalElements)))
		}

		for i := 0; i < elementsToCheck; i++ {
			s := fmt.Sprintf("%v", t.Data()[i])
			if len(s) > maxWidth {
				maxWidth = len(s)
			}
		}

		// Print column indices (starting from 0)
		sb.WriteString("    ") // Indent for row labels
		printCols := cols
		if shouldTruncate && cols > maxLineElements {
			printCols = maxLineElements
		}

		for c := 0; c < printCols; c++ {
			sb.WriteString(fmt.Sprintf("%*d ", maxWidth, c))
		}
		if shouldTruncate && cols > maxLineElements {
			sb.WriteString(fmt.Sprintf("%*s", maxWidth, "..."))
		}
		sb.WriteString("\n")

		// Print rows with row indices
		printRows := rows
		if shouldTruncate && rows > maxLineElements {
			printRows = maxLineElements
		}

		for r := 0; r < printRows; r++ {
			sb.WriteString(fmt.Sprintf("%2d: [", r))
			for c := 0; c < printCols; c++ {
				idx := r*cols + c
				sb.WriteString(fmt.Sprintf("%*v ", maxWidth, t.Data()[idx]))
			}
			if shouldTruncate && cols > maxLineElements {
				sb.WriteString(fmt.Sprintf("%*s ", maxWidth, "..."))
			}
			sb.WriteString("]\n")
		}

		if shouldTruncate && rows > maxLineElements {
			sb.WriteString(fmt.Sprintf("..: %*s\n", maxWidth*(printCols+2)+2, "..."))
		}

		return sb.String()
	}

	// 3D+ tensors: print each 2D slice with its index
	return t.printND(sb, shouldTruncate, maxSlices)
}

// printND handles printing of tensors with 3 or more dimensions by recursively printing 2D slices.
func (t *Tensor[T]) printND(sb strings.Builder, shouldTruncate bool, maxSlices int) string {
	// For tensors with more than 2 dimensions, print each 2D slice with its indices
	totalDims := len(t.shape)
	firstDim := int(t.shape[0])
	remainingDims := t.shape[1:]

	// Calculate the size of the remaining dimensions
	remainingSize := calculateSize(remainingDims)

	printSlices := firstDim
	if shouldTruncate && firstDim > maxSlices {
		printSlices = maxSlices
	}

	for i := 0; i < printSlices; i++ {
		// Header for this 2D slice
		sb.WriteString(fmt.Sprintf("Tensor slice [%d", i))
		if totalDims > 3 {
			// Show ellipsis for higher dimensions
			sb.WriteString(", ...")
		}
		sb.WriteString("]\n")

		// Extract data for this 2D slice
		startIdx := i * int(remainingSize)
		endIdx := startIdx + int(remainingSize)
		sliceData := t.Data()[startIdx:endIdx]

		// Create a temporary 2D tensor for this slice
		tempTensor := &Tensor[T]{
			data:   sliceData,
			shape:  remainingDims,
			length: remainingSize,
		}

		// Print the 2D slice with truncation
		sb.WriteString(tempTensor.String())
		sb.WriteString("\n")
	}

	if shouldTruncate && firstDim > maxSlices {
		sb.WriteString("...\n")
	}

	return sb.String()
}
