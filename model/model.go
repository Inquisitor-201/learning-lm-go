package model

import (
	"encoding/json"
	"fmt"
	"learning-lm-go/kvcache"
	"learning-lm-go/tensor"
	"os"
	"path/filepath"
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

	if config.NQH%config.NKVH != 0 {
		return nil, fmt.Errorf("num_attention_heads must be divisible by num_key_value_heads")
	}
	if config.D%config.NQH != 0 {
		return nil, fmt.Errorf("hidden_size must be divisible by num_attention_heads")
	}
	config.DQKV = config.D / config.NQH

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

func (l *Llama) Generate(tokens []uint32, maxLen uint32, top_p float32, top_k uint32, temperature float32) ([]uint32, error) {
	cache, err := kvcache.NewKVCache[float32](
		uint32(l.Config.NLayers),
		uint32(l.Config.MaxSeqLen),
		uint32(l.Config.DQKV*l.Config.NKVH),
		0,
	)
	if err != nil {
		return []uint32{}, fmt.Errorf("failed to create cache: %v", err)
	}

	finalSeq := make([]uint32, len(tokens))
	copy(finalSeq, tokens)

	n := 50
	for i := 0; i < n; i++ {
		logits := l.Forward(tensor.NewTensor(tokens, []uint32{uint32(len(tokens))}), cache)

		maxIdx := uint32(0)
		maxProb := logits.Data()[0]
		for i := 1; i < int(logits.Size()); i++ {
			if logits.Data()[i] > maxProb {
				maxProb = logits.Data()[i]
				maxIdx = uint32(i)
			}
		}

		tokens = []uint32{maxIdx}
		finalSeq = append(finalSeq, maxIdx)
	}
	return finalSeq, nil
}

func (l *Llama) Forward(input *Tensor[uint32], cache *kvcache.KVCache[float32]) *Tensor[float32] {
	seqLen := input.Size()
	pastSeqLen := cache.Len()
	cache.Increment(seqLen)

	totalSeqLen := seqLen + pastSeqLen
	// nGroups := l.Config.NQH / l.Config.NKVH

	residual := tensor.Gather(l.Params.EmbeddingTable, input)

	for i := 0; i < l.Config.NLayers; i++ {
		hidden := tensor.RMSNorm(residual, l.Params.RMSAttW[i], l.Config.RMSNormEps)
		q := tensor.MatMulTransB(hidden, l.Params.WQ[i])
		k := tensor.MatMulTransB(hidden, l.Params.WK[i])
		v := tensor.MatMulTransB(hidden, l.Params.WV[i])
		tensor.Rope(
			q.Reshape([]uint32{seqLen, uint32(l.Config.NQH), uint32(l.Config.DQKV)}),
			pastSeqLen,
			l.Config.RopeTheta,
		)
		tensor.Rope(
			k.Reshape([]uint32{seqLen, uint32(l.Config.NKVH), uint32(l.Config.DQKV)}),
			pastSeqLen,
			l.Config.RopeTheta,
		)
		v.Reshape([]uint32{seqLen, uint32(l.Config.NKVH), uint32(l.Config.DQKV)})

		fullK, err := cache.KCache(uint32(i), 0)
		if err != nil {
			panic(err)
		}
		fullV, err := cache.VCache(uint32(i), 0)
		if err != nil {
			panic(err)
		}

		copy(
			fullK.Data()[int(pastSeqLen)*l.Config.NKVH*l.Config.DQKV:int(totalSeqLen)*l.Config.NKVH*l.Config.DQKV],
			k.Data(),
		)

		copy(
			fullV.Data()[int(pastSeqLen)*l.Config.NKVH*l.Config.DQKV:int(totalSeqLen)*l.Config.NKVH*l.Config.DQKV],
			v.Data(),
		)

		fullK.Reshape([]uint32{totalSeqLen, uint32(l.Config.NKVH), uint32(l.Config.DQKV)})
		fullV.Reshape([]uint32{totalSeqLen, uint32(l.Config.NKVH), uint32(l.Config.DQKV)})

		score, err := tensor.GroupAttnScore(q, fullK)
		if err != nil {
			panic(err)
		}
		attnV, err := tensor.GroupAttnV(score, fullV)
		if err != nil {
			panic(err)
		}
		attnV.Reshape([]uint32{seqLen, uint32(l.Config.NQH * l.Config.DQKV)})

		out := tensor.MatMulTransB(attnV, l.Params.WO[i])
		residual = tensor.Add(residual, out)

		residual = FFN(residual,
			l.Params.WUp[i],
			l.Params.WDown[i],
			l.Params.WGate[i],
			l.Params.RMSFfnW[i],
			l.Config.RMSNormEps,
		)
	}

	residual = residual.Slice((seqLen-1)*uint32(l.Config.D), []uint32{1, uint32(l.Config.D)})
	final_norm := tensor.RMSNorm(
		residual,
		l.Params.RMSOutW, // 最终层的归一化权重
		l.Config.RMSNormEps,
	)
	logits := tensor.MatMulTransB(final_norm, l.Params.LMHead) // 输出投影层
	if logits.Size() != uint32(l.Config.Vocab) {
		panic("invalid logits size")
	}
	return logits
}

func FFN(residual, wUp, wDown, wGate, rmsW *Tensor[float32], eps float32) *Tensor[float32] {
	hidden := tensor.RMSNorm(residual, rmsW, eps)
	gate := tensor.MatMulTransB(hidden, wGate)

	up := tensor.MatMulTransB(hidden, wUp)
	tensor.SwiGLu(up, gate)
	output := tensor.MatMulTransB(up, wDown)
	return tensor.Add(residual, output)
}
