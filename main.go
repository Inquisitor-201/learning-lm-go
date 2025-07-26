package main

import (
	"fmt"
	"learning-lm-go/model"
	"path"
	"path/filepath"
	"runtime"

	nested "github.com/antonfisher/nested-logrus-formatter"
	"github.com/daulet/tokenizers"
	"github.com/sirupsen/logrus"
)

func SetUpLogger() {
	logrus.SetLevel(logrus.DebugLevel)

	logrus.SetReportCaller(true)

	logrus.SetFormatter(&nested.Formatter{
		HideKeys:        true,
		FieldsOrder:     []string{"component", "category"},
		TimestampFormat: "2006-01-02 15:04:05.000",
		ShowFullLevel:   true,
		NoColors:        false,

		CallerFirst: true,
		CustomCallerFormatter: func(frame *runtime.Frame) string {
			return fmt.Sprintf(" [%s:%d]", filepath.Base(frame.File), frame.Line)
		},
	})
}

func main() {
	SetUpLogger()
	model_dir := "models/story"
	llama, err := model.FromSafeTensors(model_dir)
	if err != nil {
		logrus.Fatal("Faile to load model: ", err)
		panic("Loading model failed")
	}
	logrus.Debug("Llama: ", llama)
	tk, err := tokenizers.FromFile(path.Join(model_dir, "tokenizer.json"))
	if err != nil {
		logrus.Fatal("Faile to load tokenizer: ", err)
		panic("Loading tokenizer failed")
	}

	input_tokens, _ := tk.Encode("<|start_story|>Bluey", false)
	logrus.Info("Input tokens: ", input_tokens)

	output_tokens, err := llama.Generate(input_tokens, 300, 0.9, 40, 0.6)
	if err != nil {
		logrus.Fatal("Faile to generate tokens: ", err)
		panic("Generating tokens failed")
	}
	logrus.Info("Output tokens: ", output_tokens)
	output_text := tk.Decode(output_tokens, false)
	logrus.Info("Output text: ", output_text)
}
