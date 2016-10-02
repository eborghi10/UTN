#pragma once
#include <stdio.h>
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

// Empieza entrenando las caras recolectadas.
// El algoritmo de reconocimiento de caras puede ser uno de estos:
//    "FaceRecognizer.Eigenfaces":  Eigenfaces, también conocida como PCA (Turk and Pentland, 1991).
//    "FaceRecognizer.Fisherfaces": Fisherfaces, también conocida como LDA (Belhumeur et al, 1997).
//    "FaceRecognizer.LBPH":        Local Binary Pattern Histograms (Ahonen et al, 2006).
Ptr<FaceRecognizer> learnCollectedFaces(const vector<Mat> preprocessedFaces, const vector<int> faceLabels);

// Muestra la información interna del reconocimiento de caras, para ayudar en el debugging.
// void showTrainingDebugData(const Ptr<FaceRecognizer> model, const int faceWidth, const int faceHeight);
// Genera una cara reconstruida aproximada proyectando los autovalores y autovectores de la cara (preprocesada).
Mat reconstructFace(const Ptr<FaceRecognizer> model, const Mat preprocessedFace);

// Compara dos imágenes usando la Norma Euclideana (Raiz cuadrada de la suma de los errores cuadrados).
double getSimilarity(const Mat A, const Mat B);
