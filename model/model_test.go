package model

import (
	"fmt"
	"learning-lm-go/tensor"
	"path/filepath"
	"runtime"
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

func TestLoadSafetensors(t *testing.T) {
	// 1. Get project root directory (similar to Rust's CARGO_MANIFEST_DIR)
	_, filename, _, _ := runtime.Caller(0)
	projectDir := filepath.Dir(filepath.Dir(filename)) // Get project root

	// 2. Build model path (using PathBuf-style path operations)
	modelDir := filepath.Join(projectDir, "models", "story")

	// 3. Load model (assuming FromSafetensors is implemented)
	model, err := FromSafeTensors(modelDir)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	// 4. Validate model dimension parameters
	t.Run("Dimension Validation", func(t *testing.T) {
		assertEqual(t, 2048, model.Config.Vocab, "Vocabulary size mismatch")
		assertEqual(t, 2, model.Config.NLayers, "Layer count mismatch")
		assertEqual(t, 8, model.Config.NQH, "Query head count mismatch")
		assertEqual(t, 4, model.Config.NKVH, "Key-Value head count mismatch")
		assertEqual(t, 128, model.Config.D, "Dimension size mismatch")
		assertEqual(t, 16, model.Config.DQKV, "QKV dimension mismatch")
		assertEqual(t, 384, model.Config.Di, "Intermediate dimension mismatch")
	})

	fmt.Println("model.config: ", model.Config)
	// 5. Validate weight numerical precision (using floatEqual instead of Rust's float_eq)
	t.Run("Weight Value Validation", func(t *testing.T) {
		params := model.Params

		// Embedding layer validation
		if !tensor.FloatEq(params.EmbeddingTable.Data()[50], 0.14453125, 1e-6) {
			t.Error("Embedding layer index 50 value mismatch")
		}

		// Weight sharing validation (lm_head should point to embedding_table)
		if params.LMHead.Data()[10] != params.EmbeddingTable.Data()[10] {
			t.Error("lm_head and embedding_table don't share weights")
		}

		// Layer weight validation
		testCases := []struct {
			name      string
			value     float32
			expected  float32
			tolerance float32
		}{
			{"RMSAttW[0]", params.RMSAttW[0].Data()[10], 0.18652344, 1e-6},
			{"RMSFfnW[1]", params.RMSFfnW[1].Data()[10], 0.32421875, 1e-6},
			{"RMSOutW", params.RMSOutW.Data()[100], 0.73046875, 1e-6},
			{"WDown[0]", params.WDown[0].Data()[100], -0.0625, 1e-6},
			{"WUp[0]", params.WUp[0].Data()[100], 1.46875, 1e-6},
			{"WGate[1]", params.WGate[1].Data()[100], 0.296875, 1e-6},
			{"WQ[1]", params.WQ[1].Data()[100], 0.032226563, 1e-6},
			{"WK[1]", params.WK[1].Data()[100], -0.21386719, 1e-6},
			{"WV[0]", params.WV[0].Data()[100], 0.041015625, 1e-6},
			{"WO[0]", params.WO[0].Data()[100], 0.01965332, 1e-6},
		}

		for _, tc := range testCases {
			if !tensor.FloatEq(tc.value, tc.expected, tc.tolerance) {
				t.Errorf("%s value mismatch: expected %f, got %f", tc.name, tc.expected, tc.value)
			}
		}
	})
}

// assertEqual wraps integer assertions (common Go testing pattern)
func assertEqual(t *testing.T, expected, actual int, msg string) {
	t.Helper()
	if actual != expected {
		t.Errorf("%s: expected %d, got %d", msg, expected, actual)
	}
}
