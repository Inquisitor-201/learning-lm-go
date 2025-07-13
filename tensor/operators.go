package tensor

import (
	"errors"
	"fmt"
	"math"
)

func Gather(input *Tensor[float32], indices *Tensor[uint32]) *Tensor[float32] {
	// TODO: implement Gather
	if len(input.shape) != 2 {
		panic("input must be a 2D tensor")
	}

	if len(indices.shape) != 1 {
		panic("indices must be a 1D tensor")
	}

	output := EmptyTensor[float32]([]uint32{uint32(len(indices.Data())), input.shape[1]})
	var i uint32
	for i = 0; i < uint32(len(indices.Data())); i++ {
		copy(output.Data()[i*output.shape[1]:(i+1)*output.shape[1]],
			input.Data()[indices.Data()[i]*input.shape[1]:(indices.Data()[i]+1)*input.shape[1]])
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

	if seqLen > totalSeqLen {
		panic("masked_softmax: seq_len must be less than or equal to total_seq_len")
	}
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

func Add[T TensorDataType](x *Tensor[T], y *Tensor[T]) *Tensor[T] {
	return ApplyOp(OpAdd, x, y)
}

func Sub[T TensorDataType](x *Tensor[T], y *Tensor[T]) *Tensor[T] {
	return ApplyOp(OpSub, x, y)
}

func Dot[T TensorDataType](x *Tensor[T], y *Tensor[T]) *Tensor[T] {
	return ApplyOp(OpMul, x, y)
}

func Div[T TensorDataType](x *Tensor[T], y *Tensor[T]) *Tensor[T] {
	return ApplyOp(OpDiv, x, y)
}

func ApplyOp[T TensorDataType](op int, x *Tensor[T], y *Tensor[T]) *Tensor[T] {
	if x.Size() != y.Size() {
		panic("Apply: x and y must have the same size")
	}
	res := EmptyTensor[T](x.Shape())
	for i := range x.Data() {
		x_val := x.Data()[i]
		y_val := y.Data()[i]
		switch op {
		case OpAdd:
			res.Data()[i] = x_val + y_val
		case OpSub:
			res.Data()[i] = x_val - y_val
		case OpMul:
			res.Data()[i] = x_val * y_val
		case OpDiv:
			res.Data()[i] = x_val / y_val
		default:
			panic("unknown operator")
		}
	}
	return res
}

func Neg[T TensorDataType](x *Tensor[T]) {
	for i := range x.Data() {
		x.Data()[i] = -x.Data()[i]
	}
}

func ScalarMul[T TensorDataType](a T, x *Tensor[T]) {
	for i := range x.Data() {
		x.Data()[i] = a * x.Data()[i]
	}
}

// Calculate A @ B^T
func MatMulTransB[T TensorDataType](a *Tensor[T], b *Tensor[T]) *Tensor[T] {
	if len(a.Shape()) < 2 {
		panic("MatMul: a must have at least 2 dimensions")
	}
	if len(b.Shape()) != 2 {
		panic("MatMul: b must have exactly 2 dimensions")
	}

	ndimA := len(a.Shape())

	if a.shape[ndimA-1] != b.shape[1] {
		panic("MatMul: a and b must have the same size in the last dimension")
	}

	na := a.Size() / a.shape[ndimA-1]
	nb := b.Size() / b.shape[1]

	data := make([]T, na*nb)

	for i := uint32(0); i < na; i++ {
		for j := uint32(0); j < nb; j++ {
			sum := T(0)
			baseA := i * a.shape[ndimA-1]
			baseB := j * b.shape[1]
			for k := uint32(0); k < a.shape[ndimA-1]; k++ {
				sum += a.Data()[baseA+k] * b.Data()[baseB+k]
			}
			data[i*nb+j] = sum
		}
	}

	shape := make([]uint32, 0, ndimA)
	shape = append(shape, a.shape[:ndimA-1]...)
	shape = append(shape, nb)
	return NewTensor(data, shape)
}

// GroupAttnQK 计算分组注意力中Q和K的转置乘积
// q的形状为 [n1, hq, D]，k的形状为 [n2, hk, D]
// 返回结果形状为 [hq, n1, n2]，其中每个元素 attn[h][i][j] = sum(d=0..D-1) q[i][h][d] * k[j][h'][d]
// 其中 h' = h / (hq/hk)，要求 hq 必须是 hk 的倍数（即 hq = m*hk，m为整数）
func GroupAttnQK(q *Tensor[float32], k *Tensor[float32]) (*Tensor[float32], error) {
	// 检查q和k是否为三维张量
	qShape := q.Shape()
	if len(qShape) != 3 {
		return nil, errors.New("q must be a 3D tensor")
	}
	kShape := k.Shape()
	if len(kShape) != 3 {
		return nil, errors.New("k must be a 3D tensor")
	}

	n1, hq, dq := qShape[0], qShape[1], qShape[2]
	n2, hk, dk := kShape[0], kShape[1], kShape[2]

	factor := 1. / float32(math.Sqrt(float64(dq)))

	if dq != dk {
		return nil, fmt.Errorf("q and k must have the same last dimension, got %d (q) and %d (k)", dq, dk)
	}

	if hq%hk != 0 {
		return nil, fmt.Errorf("q's head count (hq=%d) must be a multiple of k's head count (hk=%d)", hq, hk)
	}
	m := hq / hk

	attnShape := []uint32{hq, n1, n2}
	attn := EmptyTensor[float32](attnShape)

	qData := q.Data()
	kData := k.Data()
	attnData := attn.Data()

	for h := uint32(0); h < hq; h++ {
		for i := uint32(0); i < n1; i++ {
			for j := uint32(0); j < n2; j++ {
				hPrime := h / m
				if hPrime >= hk {
					return nil, fmt.Errorf("invalid hPrime %d (exceeds hk=%d) for h=%d", hPrime, hk, h)
				}

				sum := float32(0.0)
				for d := uint32(0); d < dq; d++ {
					qIdx := i*hq*dq + h*dq + d
					kIdx := j*hk*dk + hPrime*dk + d

					sum += qData[qIdx] * kData[kIdx]
				}

				attnIdx := h*n1*n2 + i*n2 + j
				attnData[attnIdx] = sum * factor
			}
		}
	}

	return attn, nil
}

func GroupAttnScore(q *Tensor[float32], k *Tensor[float32]) (*Tensor[float32], error) {
	attn, err := GroupAttnQK(q, k)
	if err != nil {
		return nil, err
	}
	MaskedSoftmax(attn)
	return attn, nil
}

// GroupAttnV 计算分组注意力机制中的值矩阵乘法
// attn 形状: [h, n1, n2]
// v 形状: [n2, h', D]
// 返回形状: [n1, h, D]
func GroupAttnV(attn *Tensor[float32], v *Tensor[float32]) (*Tensor[float32], error) {
	// 检查维度数量
	if len(attn.Shape()) != 3 {
		return nil, errors.New("attn must be a 3D tensor")
	}
	if len(v.Shape()) != 3 {
		return nil, errors.New("v must be a 3D tensor")
	}

	// 获取各维度大小
	h, n1, n2_attn := attn.Shape()[0], attn.Shape()[1], attn.Shape()[2]
	n2_v, h_prime, d := v.Shape()[0], v.Shape()[1], v.Shape()[2]

	m := h / h_prime

	// 检查n2是否匹配
	if n2_attn != n2_v {
		return nil, fmt.Errorf("n2 dimension mismatch: attn has %d, v has %d", n2_attn, n2_v)
	}

	// 检查h是否是h'的倍数
	if h%h_prime != 0 {
		return nil, fmt.Errorf("h must be a multiple of h', got h=%d and h'=%d", h, h_prime)
	}

	// 创建结果张量
	resultShape := []uint32{n1, h, d}
	result := EmptyTensor[float32](resultShape)

	// 获取底层数据切片以提高访问效率
	attnData := attn.Data()
	vData := v.Data()
	resultData := result.Data()

	// 计算结果张量的每个元素
	for i := uint32(0); i < n1; i++ {
		for h_idx := uint32(0); h_idx < h; h_idx++ {
			// 计算对应的h'索引
			h_prime_idx := h_idx / m

			for d_idx := uint32(0); d_idx < d; d_idx++ {
				// 初始化累加和
				sum := float32(0)

				// 对n2维度求和
				for j := uint32(0); j < n2_attn; j++ {
					// 计算attn中元素的索引
					attnIdx := h_idx*n1*n2_attn + i*n2_attn + j

					// 计算v中元素的索引
					vIdx := j*h_prime*d + h_prime_idx*d + d_idx

					// 累加乘积
					sum += attnData[attnIdx] * vData[vIdx]
				}

				// 计算结果张量中的索引
				resultIdx := i*h*d + h_idx*d + d_idx

				// 存储结果
				resultData[resultIdx] = sum
			}
		}
	}

	return result, nil
}
