package model

import (
	"learning-lm-go/tensor"
	"testing"
)

// TestMLP verifies the correctness of the MLP (Multi-Layer Perceptron) implementation.
// It initializes input tensors, runs the MLP function, and checks if the output matches expectations.
func TestMLP(t *testing.T) {
	// Define test parameters
	seqLen := 4          // Sequence length
	d := 2               // Dimension of input features
	di := 3              // Intermediate dimension in MLP
	eps := float32(1e-6) // Epsilon for numerical stability (e.g., in RMS normalization)

	// 1. Initialize input tensors
	// residual: shape [seqLen, d] = [4, 2], initialized with all 1.0s (8 elements total)
	residualData := []float32{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
	residual := tensor.NewTensor(residualData, []uint32{uint32(seqLen), uint32(d)})

	// 2. Initialize weight tensors
	// w_up: shape [di, d] = [3, 2], weight matrix for up-projection (3×2=6 elements)
	wUpData := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
	wUp := tensor.NewTensor(wUpData, []uint32{uint32(di), uint32(d)})

	// w_down: shape [d, di] = [2, 3], weight matrix for down-projection (2×3=6 elements)
	wDownData := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
	wDown := tensor.NewTensor(wDownData, []uint32{uint32(d), uint32(di)})

	// w_gate: shape [di, d] = [3, 2], weight matrix for gate activation (same as w_up here)
	wGateData := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
	wGate := tensor.NewTensor(wGateData, []uint32{uint32(di), uint32(d)})

	// rms_w: shape [d] = [2], weights for RMS normalization
	rmsWData := []float32{1.0, 1.0}
	rmsW := tensor.NewTensor(rmsWData, []uint32{uint32(d)})

	// 3. Execute the MLP function
	// Note: The actual mlp function implementation must be ported from Rust logic
	mlpOutput := MLP(
		residual,
		wUp,
		wDown,
		wGate,
		rmsW,
		eps,
	)

	// 4. Define expected output tensor
	expectedData := []float32{
		1.3429964, 1.7290739,
		1.3429964, 1.7290739,
		1.3429964, 1.7290739,
		1.3429964, 1.7290739,
	}
	expected := tensor.NewTensor[float32](expectedData, []uint32{uint32(seqLen), uint32(d)})

	// 5. Validate if the result is within the acceptable error range
	ok, err := mlpOutput.CloseTo(expected, 1e-3)
	if err != nil {
		t.Fatalf("Error comparing tensors: %v", err)
	}
	if !ok {
		t.Errorf("MLP output does not match the expected result: %v vs %v", residual, expected)
	}
}
