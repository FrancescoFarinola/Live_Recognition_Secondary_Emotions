#ifndef EMOTION_H
#define EMOTION_H

#include <QtWidgets/QMainWindow>
#include "ui_emotion.h"
#include <QApplication>
#include <QFileDialog>
#include <QMessageBox>

class Emotion : public QMainWindow
{
	Q_OBJECT

public:
	Emotion(QWidget *parent = 0);
	~Emotion();
	std::string apriFile();
	std::string salvaFile();
	std::string salvaRisultati();

private:
	Ui::EmotionClass ui;
	private slots:
		void real_time();
		void registra();
		void carica_analizza();

};

#endif // EMOTION_H
