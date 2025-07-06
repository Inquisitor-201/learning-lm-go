package model

import (
	"encoding/json"
	"fmt"
	"learning-lm-go/safetensors"
	"os"
	"path/filepath"

	"github.com/sirupsen/logrus"
)

type LlamaConfigJson struct {
	VocabSize             int     `json:"vocab_size"`
	NumHiddenLayers       int     `json:"num_hidden_layers"`
	NumAttentionHeads     int     `json:"num_attention_heads"`
	NumKeyValueHeads      int     `json:"num_key_value_heads"`
	HiddenSize            int     `json:"hidden_size"`
	IntermediateSize      int     `json:"intermediate_size"`
	RMSNormEps            float32 `json:"rms_norm_eps"`
	RopeTheta             float32 `json:"rope_theta"`
	MaxPositionEmbeddings int     `json:"max_position_embeddings"`
	BosTokenID            uint32  `json:"bos_token_id"`
	EosTokenID            uint32  `json:"eos_token_id"`
}

type SafeTensors struct {
}

type Llama struct {
	Vocab int
}

func FromSafeTensors(modelDir string) (*Llama, error) {
	configPath := filepath.Join(modelDir, "config.json")
	configData, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %v", err)
	}

	var config LlamaConfigJson
	if err := json.Unmarshal(configData, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %v", err)
	}
	logrus.Debug("Config: ", config)

	modelPath := filepath.Join(modelDir, "model.safetensors")
	SafeTensors, err := safetensors.Parse(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to parse model file: %v", err)
	}
	logrus.Debug("SafeTensors: ", SafeTensors)

	return nil, nil
}
