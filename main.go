package main

import (
	"fmt"
	"learning-lm-go/model"
	"path/filepath"
	"runtime"

	nested "github.com/antonfisher/nested-logrus-formatter"
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
}
