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

func MaskedSoftmax(y *Tensor[float32]) {
	shape := y.Shape()
	ndim := len(shape)
	if ndim < 2 {
		panic("masked_softmax: tensor must have at least 2 dimensions")
	}

	seqLen := shape[ndim-2]
	totalSeqLen := shape[ndim-1]
	batch := y.Size() / (seqLen * totalSeqLen)

	data := y.Data()

	for b := uint32(0); b < batch; b++ {
		base := b * seqLen * totalSeqLen

		for i := uint32(0); i < seqLen; i++ {
			offset := base + i*totalSeqLen
			boundary := totalSeqLen - seqLen + i + 1

			maxVal := data[offset]
			for j := uint32(0); j < boundary; j++ {
				current := data[offset+j]
				if current > maxVal {
					maxVal = current
				}
			}

			sumExp := float32(0.0)
			for j := uint32(0); j < boundary; j++ {
				expVal := float32(math.Exp(float64(data[offset+j] - maxVal)))
				data[offset+j] = expVal
				sumExp += expVal
			}

			for j := uint32(0); j < boundary; j++ {
				data[offset+j] /= sumExp
			}
			for j := boundary; j < totalSeqLen; j++ {
				data[offset+j] = 0.0
			}
		}
	}
}

func SwiGLu(y *Tensor[float32], x *Tensor[float32]) {
	if y.Size() != x.Size() {
		panic("SwiGLu: y and x must have the same size")
	}
	for i := range y.Data() {
		x_val := x.Data()[i]
		y_val := y.Data()[i]
		y.Data()[i] = y_val * x_val / float32(1.0+math.Exp(-1.0*float64(x_val)))
	}
}

func RMSNorm(x *Tensor[float32], w *Tensor[float32], epsilon float32) *Tensor[float32] {
	if len(x.Shape()) < 1 {
		panic("RMSNorm: x must have at least 1 dimension")
	}

	if len(w.Shape()) != 1 {
		panic("RMSNorm: w must have exactly 1 dimension")
	}

	ndim := len(x.Shape())
	if x.shape[ndim-1] != w.shape[0] {
		panic("RMSNorm: x and w must have the same size in the first dimension")
	}

	d := x.Shape()[ndim-1]
	batch := x.Size() / d
	y := EmptyTensor[float32](x.Shape())

	for b := uint32(0); b < batch; b++ {
		base := b * d

		res := float32(0.0)

		for i := uint32(0); i < d; i++ {
			offset := base + i
			xi := x.Data()[offset]
			res += xi * xi
		}
		res = float32(math.Sqrt(float64(res/float32(d) + epsilon)))
		for i := uint32(0); i < d; i++ {
			offset := base + i
			y.Data()[offset] = *w.At(i) * x.Data()[offset] / res
		}
	}
	return y
}

// enum operator
const (
	OpAdd = iota
	OpSub
	OpMul
	OpDiv
)

func Add(x *Tensor[float32], y *Tensor[float32]) {
	ApplyOp(OpAdd, x, y)
}

func Sub(x *Tensor[float32], y *Tensor[float32]) {
	ApplyOp(OpSub, x, y)
}

func Dot(x *Tensor[float32], y *Tensor[float32]) {
	ApplyOp(OpMul, x, y)
}

func Div(x *Tensor[float32], y *Tensor[float32]) {
	ApplyOp(OpDiv, x, y)
}

func ApplyOp(op int, x *Tensor[float32], y *Tensor[float32]) {
	if x.Size() != y.Size() {
		panic("Apply: x and y must have the same size")
	}
	for i := range x.Data() {
		x_val := x.Data()[i]
		y_val := y.Data()[i]
		switch op {
		case OpAdd:
			x.Data()[i] = x_val + y_val
		case OpSub:
			x.Data()[i] = x_val - y_val
		case OpMul:
			x.Data()[i] = x_val * y_val
		case OpDiv:
			x.Data()[i] = x_val / y_val
		default:
			panic("unknown operator")
		}
	}
}

func Neg(x *Tensor[float32]) {
	for i := range x.Data() {
		x.Data()[i] = -x.Data()[i]
	}
}
