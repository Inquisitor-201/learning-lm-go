package kvcache

import (
	"errors"
	"learning-lm-go/tensor"
)

// KVCache stores key-value cache data structure for transformer models
type KVCache[T tensor.TensorDataType] struct {
	kCache    []*tensor.Tensor[T] // Key cache for each layer: (max_seq_len, n_kv_head * dqkv) x layers
	vCache    []*tensor.Tensor[T] // Value cache for each layer: (max_seq_len, n_kv_head * dqkv) x layers
	maxSeqLen uint32              // Maximum sequence length the cache can hold
	dim       uint32              // Dimension of the key/value tensors
	length    uint32              // Current length of the sequence in the cache
}

// NewKVCache creates a new KVCache instance with specified parameters
func NewKVCache[T tensor.TensorDataType](nLayers, maxSeqLen, dim, initLen uint32) (*KVCache[T], error) {
	if nLayers <= 0 || maxSeqLen <= 0 || dim <= 0 {
		return nil, errors.New("invalid parameters: all values must be positive except initLen which can be zero")
	}

	if initLen > maxSeqLen {
		return nil, errors.New("initial length exceeds maximum sequence length")
	}

	kCache := make([]*tensor.Tensor[T], nLayers)
	vCache := make([]*tensor.Tensor[T], nLayers)

	// Initialize key and value caches with empty tensors
	for i := uint32(0); i < nLayers; i++ {
		kTensor := tensor.EmptyTensor[T]([]uint32{maxSeqLen, dim})
		vTensor := tensor.EmptyTensor[T]([]uint32{maxSeqLen, dim})

		kCache[i] = kTensor
		vCache[i] = vTensor
	}

	return &KVCache[T]{
		kCache:    kCache,
		vCache:    vCache,
		maxSeqLen: maxSeqLen,
		dim:       dim,
		length:    initLen,
	}, nil
}

// KCache returns the key cache tensor for the specified layer starting from the given position
func (kc *KVCache[T]) KCache(layer, start uint32) (*tensor.Tensor[T], error) {
	if layer >= uint32(len(kc.kCache)) {
		return nil, errors.New("layer index out of range")
	}

	if start >= uint32(kc.length) {
		return nil, errors.New("start index out of range")
	}

	// Return a view of the key cache from start to current length
	return kc.kCache[layer].Slice(start*uint32(kc.dim), []uint32{uint32(kc.length - start), uint32(kc.dim)}), nil
}

// VCache returns the value cache tensor for the specified layer starting from the given position
func (kc *KVCache[T]) VCache(layer, start uint32) (*tensor.Tensor[T], error) {
	if layer >= uint32(len(kc.vCache)) {
		return nil, errors.New("layer index out of range")
	}

	if start >= uint32(kc.length) {
		return nil, errors.New("start index out of range")
	}

	// Return a view of the value cache from start to current length
	return kc.vCache[layer].Slice(start*uint32(kc.dim), []uint32{uint32(kc.length - start), uint32(kc.dim)}), nil
}

// Increment increases the current sequence length by the given amount
func (kc *KVCache[T]) Increment(seqLen uint32) error {
	if kc.length+seqLen > kc.maxSeqLen {
		return errors.New("increment would exceed maximum cache capacity")
	}

	kc.length += seqLen
	return nil
}

// Len returns the current length of the sequence in the cache
func (kc *KVCache[T]) Len() uint32 {
	return kc.length
}

// MaxSeqLen returns the maximum capacity of the cache
func (kc *KVCache[T]) MaxSeqLen() uint32 {
	return kc.maxSeqLen
}

// Dim returns the dimension of the key/value tensors
func (kc *KVCache[T]) Dim() uint32 {
	return kc.dim
}

// NumLayers returns the number of layers in the cache
func (kc *KVCache[T]) NumLayers() int {
	return len(kc.kCache)
}
