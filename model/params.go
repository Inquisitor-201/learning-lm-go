package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"learning-lm-go/tensor"
	"log"
	"math"
	"os"
	"regexp"
	"strconv"
	"strings"
)

type LlamaParams[T tensor.TensorDataType] struct {
	// token_id到嵌入的查找表
	EmbeddingTable *tensor.Tensor[T] // (vocab_size, dim)

	// 解码器层参数
	RMSAttW []*tensor.Tensor[T] // (hidden_size, ) 每层一个
	WQ      []*tensor.Tensor[T] // (n_heads * head_size, hidden_size) 每层一个
	WK      []*tensor.Tensor[T] // (n_kv_heads * head_size, hidden_size) 每层一个
	WV      []*tensor.Tensor[T] // (n_kv_heads * head_size, hidden_size) 每层一个
	WO      []*tensor.Tensor[T] // (hidden_size, n_heads * head_size) 每层一个

	// FFN层参数
	RMSFfnW []*tensor.Tensor[T] // (hidden_size, ) 每层一个
	WUp     []*tensor.Tensor[T] // (intermediate_size, hidden_size) 每层一个
	WGate   []*tensor.Tensor[T] // (intermediate_size, hidden_size) 每层一个
	WDown   []*tensor.Tensor[T] // (hidden_size, intermediate_size) 每层一个

	// 输出层参数
	RMSOutW *tensor.Tensor[T] // (hidden_size, )
	LMHead  *tensor.Tensor[T] // (vocab_size, dim)
}

func bytesToTypedSlice[T tensor.TensorDataType](dataBytes []byte, dtype string) ([]T, error) {
	var zero T
	switch any(zero).(type) {
	case float32:
		if dtype != "F32" {
			return nil, fmt.Errorf("dtype mismatch: expected F32, got %s", dtype)
		}
		numElements := len(dataBytes) / 4
		data := make([]T, numElements)
		for i := 0; i < numElements; i++ {
			bits := binary.LittleEndian.Uint32(dataBytes[i*4 : (i+1)*4])
			val := float32(math.Float32frombits(bits))
			data[i] = T(val)
		}
		return data, nil
	// 扩展其他类型（如float64/int32等）
	default:
		return nil, fmt.Errorf("unsupported dtype: %s", dtype)
	}
}

func extractLayerIndex(key string) (int, bool) {
	re := regexp.MustCompile(`model\.layers\.(\d+)`)
	matches := re.FindStringSubmatch(key)
	if len(matches) < 2 {
		return 0, false
	}
	layerIndex, _ := strconv.Atoi(matches[1])
	return layerIndex, true
}

func ParamsFromSafeTensors(filePath string) (*LlamaParams[float32], error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	var headerLen uint64
	// read 8 bytes for header length
	if err := binary.Read(file, binary.LittleEndian, &headerLen); err != nil {
		return nil, fmt.Errorf("failed to read header length: %v", err)
	}

	headerData := make([]byte, headerLen)

	if _, err := io.ReadFull(file, headerData); err != nil {
		return nil, fmt.Errorf("failed to read header data: %v", err)
	}

	metaData := make(map[string]interface{})
	if err := json.Unmarshal(headerData, &metaData); err != nil {
		return nil, fmt.Errorf("failed to parse header data: %v", err)
	}

	totalLayers := 0
	for key := range metaData {
		if layerIndex, ok := extractLayerIndex(key); ok && layerIndex+1 > totalLayers {
			totalLayers = layerIndex + 1
		}
	}

	params := &LlamaParams[float32]{
		RMSAttW: make([]*tensor.Tensor[float32], totalLayers),
		WQ:      make([]*tensor.Tensor[float32], totalLayers),
		WK:      make([]*tensor.Tensor[float32], totalLayers),
		WV:      make([]*tensor.Tensor[float32], totalLayers),
		WO:      make([]*tensor.Tensor[float32], totalLayers),
		RMSFfnW: make([]*tensor.Tensor[float32], totalLayers),
		WUp:     make([]*tensor.Tensor[float32], totalLayers),
		WGate:   make([]*tensor.Tensor[float32], totalLayers),
		WDown:   make([]*tensor.Tensor[float32], totalLayers),
	}

	dataStart := int64(8 + headerLen)

	metadataMap := make(map[string]string)
	if meta, ok := metaData["__metadata__"].(map[string]interface{}); ok {
		for k, v := range meta {
			if s, ok := v.(string); ok {
				metadataMap[k] = s
			}
		}
	}

	for key, val := range metaData {
		if key == "__metadata__" {
			continue
		}

		tensorInfo, ok := val.(map[string]interface{})
		if !ok {
			continue
		}

		// 解析张量元信息
		dtype, _ := tensorInfo["dtype"].(string)
		shapeInterface, _ := tensorInfo["shape"].([]interface{})
		offsetsInterface, _ := tensorInfo["data_offsets"].([]interface{})

		shape := make([]uint32, len(shapeInterface))
		for i, s := range shapeInterface {
			shape[i] = uint32(s.(float64))
		}
		startOffset := uint64(offsetsInterface[0].(float64))
		endOffset := uint64(offsetsInterface[1].(float64))
		dataLength := endOffset - startOffset

		// 读取张量数据
		buf := make([]byte, dataLength)
		_, err := file.ReadAt(buf, dataStart+int64(startOffset))
		if err != nil {
			log.Printf("Failed to read tensor %s: %v", key, err)
			continue
		}

		// 字节转泛型切片
		dataSlice, err := bytesToTypedSlice[float32](buf, dtype)
		if err != nil {
			return nil, err
		}

		// 创建张量
		tensor := tensor.NewTensor(dataSlice, shape)

		// 根据键名分配张量
		switch {
		case key == "lm_head.weight":
			params.EmbeddingTable = tensor // 来自元数据映射
			params.LMHead = tensor
		case key == "model.norm.weight":
			params.RMSOutW = tensor
		case strings.HasPrefix(key, "model.layers."):
			layerIndex, ok := extractLayerIndex(key)
			if !ok {
				continue
			}
			parts := strings.Split(key, ".")
			if len(parts) < 4 {
				// 如果分割后少于4部分，说明格式不对，跳过
				continue
			}
			suffix := strings.Join(parts[3:], ".")
			switch suffix {
			case "input_layernorm.weight":
				params.RMSAttW[layerIndex] = tensor
			case "post_attention_layernorm.weight":
				params.RMSFfnW[layerIndex] = tensor
			case "self_attn.q_proj.weight":
				params.WQ[layerIndex] = tensor
			case "self_attn.k_proj.weight":
				params.WK[layerIndex] = tensor
			case "self_attn.v_proj.weight":
				params.WV[layerIndex] = tensor
			case "self_attn.o_proj.weight":
				params.WO[layerIndex] = tensor
			case "mlp.gate_proj.weight":
				params.WGate[layerIndex] = tensor
			case "mlp.up_proj.weight":
				params.WUp[layerIndex] = tensor
			case "mlp.down_proj.weight":
				params.WDown[layerIndex] = tensor
			}
		}
	}
	return params, nil
}
