#pragma once
#include <stdio.h>
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

//Con esta funcion buscamos los ojos dentro de la imagen del face dado. Retorna los centros de los ojos detectados, o -1 si no los encontro.
void detectBothEyes(const Mat &face, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, Point &leftEye, Point &rightEye, Rect *searchedLeftEye = NULL, Rect *searchedRightEye = NULL);

//Mediante esta función equalizamos ambas partes de la cara, izq y der, de esta manera compensamos las diferencias de iluminacion y podemos obtener mejores resultados
void equalizeLeftAndRightHalves(Mat &faceImg);

//Crea una imagen de la cara detectada en escala de grises, con tamaño, contraste y brillo estandar.
//Con doLeftAndRightSeparately = true, procesamos ambas partes de la cara por separado.
// Realiza el preprocesamiento como una combinación de:
// - Geométrica de escala, rotación y traslación mediante detección de ojos,
// - Suaviza ruido de la imagen utilizando un Filtro Bilateral,
// - Estandarizar el brillo de los dos lados izquierdo y derecho de la cara de forma independiente mediante histograma de Nivelación,
// - Elimina fondo y el cabello usando una máscara elíptica.
// Devuelve un rostro preprocesado o imagen NULL (es decir: no se pudo detectar la cara y 2 ojos).
// Si se encuentra un rostro, se puede almacenar el rect las coordenadas del mismo, posicion de los ojos
Mat getPreprocessedFace(Mat &srcImg, int desiredFaceWidth, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, bool doLeftAndRightSeparately, Rect *storeFaceRect = NULL, Point *storeLeftEye = NULL, Point *storeRightEye = NULL, Rect *searchedLeftEye = NULL, Rect *searchedRightEye = NULL);
