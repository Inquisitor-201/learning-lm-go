package tensor

import "math"

func Gather(input *Tensor[float32], indices *Tensor[uint32]) *Tensor[float32] {
	// TODO: implement Gather
	if len(input.shape) != 2 {
		panic("input must be a 2D tensor")
	}

	if len(indices.shape) != 1 {
		panic("indices must be a 1D tensor")
	}

	output := EmptyTensor[float32]([]uint32{uint32(len(indices.data)), input.shape[1]})
	var i uint32
	for i = 0; i < uint32(len(indices.data)); i++ {
		copy(output.data[i*output.shape[1]:(i+1)*output.shape[1]],
			input.data[indices.data[i]*input.shape[1]:(indices.data[i]+1)*input.shape[1]])
	}
	return output
}

func Rope(y *Tensor[float32], startPos uint32, theta float32) {
	shape := y.Shape()
	if len(shape) != 3 {
		panic("shape must be a 3D tensor")
	}
	seq_len := shape[0]
	n_heads := shape[1]
	d := shape[2]

	if d%2 != 0 {
		panic("d must be even")
	}

	for tok := uint32(0); tok < seq_len; tok++ {
		pos := startPos + tok
		for head := uint32(0); head < n_heads; head++ {
			for i := uint32(0); i < d/2; i++ {
				a := *y.At(tok, head, i)
				b := *y.At(tok, head, d/2+i)
				freq := float32(pos) / float32(math.Pow(float64(theta), 2*float64(i)/float64(d)))
				sin, cos := math.Sincos(float64(freq))
				*y.At(tok, head, i) = a*float32(cos) - b*float32(sin)
				*y.At(tok, head, d/2+i) = a*float32(sin) + b*float32(cos)
			}
		}
	}
}
