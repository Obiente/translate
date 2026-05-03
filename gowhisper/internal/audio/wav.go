package audio

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"

	"github.com/go-audio/wav"
)

// DecodeWAVToFloat32 decodes a small WAV blob into 32-bit float PCM samples.
func DecodeWAVToFloat32(b []byte) ([]float32, int, error) {
	r := bytes.NewReader(b)
	dec := wav.NewDecoder(r)
	if !dec.IsValidFile() {
		return nil, 0, errors.New("invalid wav file")
	}
	buf, err := dec.FullPCMBuffer()
	if err != nil {
		if err == io.EOF {
			err = nil
		} else {
			return nil, 0, err
		}
	}
	if buf == nil {
		return nil, 0, errors.New("empty wav buffer")
	}
	// buf is *audio.IntBuffer; normalize to float32 [-1,1]
	ib := buf
	bitDepth := ib.SourceBitDepth
	if bitDepth <= 0 {
		bitDepth = 16
	}
	maxInt := 1 << (bitDepth - 1)
	if maxInt <= 0 {
		maxInt = 32768
	}
	max := float32(maxInt)
	out := make([]float32, len(ib.Data))
	for i, v := range ib.Data {
		out[i] = float32(v) / max
	}
	sr := int(dec.SampleRate)
	if sr == 0 && ib.Format != nil {
		sr = ib.Format.SampleRate
	}
	if sr == 0 {
		sr = 16000
	}
	return out, sr, nil
}

// DecodePCM16LEToFloat32 converts little-endian PCM16 bytes into float32 samples and returns the given sample rate.
func DecodePCM16LEToFloat32(b []byte, sampleRate int) ([]float32, int, error) {
	if sampleRate <= 0 {
		sampleRate = 16000
	}
	if len(b)%2 != 0 {
		return nil, 0, errors.New("pcm16 length must be even")
	}
	out := make([]float32, len(b)/2)
	for i := 0; i < len(out); i++ {
		v := int16(binary.LittleEndian.Uint16(b[i*2 : i*2+2]))
		out[i] = float32(v) / 32768.0
	}
	return out, sampleRate, nil
}

// SignalStats returns RMS and peak amplitude for normalized PCM32F samples.
func SignalStats(samples []float32) (rms float64, peak float64) {
	if len(samples) == 0 {
		return 0, 0
	}
	var sumSquares float64
	for _, sample := range samples {
		value := math.Abs(float64(sample))
		sumSquares += value * value
		if value > peak {
			peak = value
		}
	}
	rms = math.Sqrt(sumSquares / float64(len(samples)))
	return rms, peak
}

// ResampleLinear resamples PCM32F from inRate to outRate using linear interpolation.
func ResampleLinear(samples []float32, inRate, outRate int) []float32 {
	if inRate <= 0 || outRate <= 0 || inRate == outRate || len(samples) == 0 {
		if inRate == outRate {
			return append([]float32(nil), samples...)
		}
		return samples
	}
	ratio := float64(outRate) / float64(inRate)
	outLen := int(float64(len(samples)) * ratio)
	if outLen <= 1 {
		outLen = 1
	}
	out := make([]float32, outLen)
	for i := 0; i < outLen; i++ {
		srcPos := float64(i) / ratio
		i0 := int(srcPos)
		if i0 >= len(samples)-1 {
			out[i] = samples[len(samples)-1]
			continue
		}
		frac := float32(srcPos - float64(i0))
		s0 := samples[i0]
		s1 := samples[i0+1]
		out[i] = s0 + (s1-s0)*frac
	}
	return out
}

// EncodePCM16MonoWAV encodes normalized mono PCM samples to a 16-bit PCM WAV file.
func EncodePCM16MonoWAV(samples []float32, sampleRate int) ([]byte, error) {
	if sampleRate <= 0 {
		sampleRate = 16000
	}
	dataSize := len(samples) * 2
	buf := bytes.NewBuffer(make([]byte, 0, 44+dataSize))

	write := func(v any) error {
		return binary.Write(buf, binary.LittleEndian, v)
	}

	if _, err := buf.WriteString("RIFF"); err != nil {
		return nil, err
	}
	if err := write(uint32(36 + dataSize)); err != nil {
		return nil, err
	}
	if _, err := buf.WriteString("WAVE"); err != nil {
		return nil, err
	}
	if _, err := buf.WriteString("fmt "); err != nil {
		return nil, err
	}
	if err := write(uint32(16)); err != nil {
		return nil, err
	}
	if err := write(uint16(1)); err != nil {
		return nil, err
	}
	if err := write(uint16(1)); err != nil {
		return nil, err
	}
	if err := write(uint32(sampleRate)); err != nil {
		return nil, err
	}
	byteRate := sampleRate * 2
	if err := write(uint32(byteRate)); err != nil {
		return nil, err
	}
	if err := write(uint16(2)); err != nil {
		return nil, err
	}
	if err := write(uint16(16)); err != nil {
		return nil, err
	}
	if _, err := buf.WriteString("data"); err != nil {
		return nil, err
	}
	if err := write(uint32(dataSize)); err != nil {
		return nil, err
	}

	for _, sample := range samples {
		value := math.Round(math.Max(-1, math.Min(1, float64(sample))) * 32767.0)
		if err := write(int16(value)); err != nil {
			return nil, err
		}
	}

	return buf.Bytes(), nil
}
