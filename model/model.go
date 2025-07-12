package model

import (
	"encoding/json"
	"fmt"
	"learning-lm-go/kvcache"
	"learning-lm-go/tensor"
	"os"
	"path/filepath"

	"github.com/sirupsen/logrus"
)

type Tensor[T tensor.TensorDataType] = tensor.Tensor[T]

type LlamaConfig struct {
	Vocab      int `json:"vocab_size"`
	NLayers    int `json:"num_hidden_layers"`
	NQH        int `json:"num_attention_heads"`
	NKVH       int `json:"num_key_value_heads"`
	D          int `json:"hidden_size"`
	Di         int `json:"intermediate_size"`
	DQKV       int
	RMSNormEps float32 `json:"rms_norm_eps"`
	RopeTheta  float32 `json:"rope_theta"`
	MaxSeqLen  int     `json:"max_position_embeddings"`
	BosTokenID uint32  `json:"bos_token_id"`
	EosTokenID uint32  `json:"eos_token_id"`
}

type Llama struct {
	Config *LlamaConfig
	Params *LlamaParams[float32]
}

func FromSafeTensors(modelDir string) (*Llama, error) {
	configPath := filepath.Join(modelDir, "config.json")
	configData, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %v", err)
	}

	var config LlamaConfig
	if err := json.Unmarshal(configData, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %v", err)
	}
	config.DQKV = config.D / config.NQH
	logrus.Debug("Config: ", config)

	modelPath := filepath.Join(modelDir, "model.safetensors")
	params, err := ParamsFromSafeTensors(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to parse model file: %v", err)
	}

	return &Llama{
		Config: &config,
		Params: params,
	}, nil
}

func (l *Llama) Generate(tokens []uint32, maxLen uint32, top_p float32, top_k uint32, temperature float32) []uint32 {
	return []uint32{}
}

func (l *Llama) Forward(input *Tensor[uint32], cache *kvcache.KVCache[float32]) {
	seqLen := input.Size()
	pastSeqLen := cache.Len()
	cache.Increment(seqLen)

	totalSeqLen := seqLen + pastSeqLen
	fmt.Println("totalSeqLen: ", totalSeqLen)
	// nGroups := l.
}

func MLP(residual, wUp, wDown, wGate, rmsW *Tensor[float32], eps float32) *Tensor[float32] {
	hidden := tensor.RMSNorm(residual, rmsW, eps)
	gate := tensor.MatMulTransB(hidden, wGate)

	up := tensor.MatMulTransB(hidden, wUp)
	tensor.SwiGLu(up, gate)
	output := tensor.MatMulTransB(up, wDown)
	return tensor.Add(residual, output)
}
