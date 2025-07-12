package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/sirupsen/logrus"
)

type TensorMetaData struct {
}

type MetaData struct {
	Tensors map[string]TensorMetaData `json:"tensors"`
}

type Safetensors struct {
}

type LlamaParams struct {
}

func ParamsParse(filePath string) (*LlamaParams, error) {
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

	logrus.Debug("header len: ", headerLen)

	headerData := make([]byte, headerLen)

	if _, err := io.ReadFull(file, headerData); err != nil {
		return nil, fmt.Errorf("failed to read header data: %v", err)
	}

	metaData := make(map[string]interface{})
	if err := json.Unmarshal(headerData, &metaData); err != nil {
		return nil, fmt.Errorf("failed to parse header data: %v", err)
	}

	return &LlamaParams{}, nil
}
