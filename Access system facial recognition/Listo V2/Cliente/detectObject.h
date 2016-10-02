#pragma once
#include <stdio.h>
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

/// Buscar por sólo un único objeto en la imagen, como la cara más grande, almacenar el resultado en 'largestObject'.
void detectLargestObject(const Mat &img, CascadeClassifier &cascade, Rect &largestObject, int scaledWidth = 320);


/// Búsqueda de muchos objetos en la imagen, como todas las caras, almacenar los resultados en 'objetcs'.
void detectManyObjects(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth = 320);


// Puede utilizar cascadas Haar o cascadas de LBP para Detección de la cara, o incluso los ojos, la boca o la detección auto.
// La entrada se encogió temporalmente a 'scaledWidth' para la detección mucho más rápido, ya que 240 es suficiente para encontrar caras.
// Nota: detectLargestObject () debe ser más rápido que detectManyObjects ().
