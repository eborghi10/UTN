
#include "detectObject.h"

const double DESIRED_LEFT_EYE_X = 0.16;
const double DESIRED_LEFT_EYE_Y = 0.14;
const double FACE_ELLIPSE_CY = 0.40;
const double FACE_ELLIPSE_W = 0.50;
const double FACE_ELLIPSE_H = 0.80;


void detectBothEyes(const Mat &face, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, Point &leftEye, Point &rightEye, Rect *searchedLeftEye, Rect *searchedRightEye)
{

    // Con eye.xml or eyeglasses.xml: detecta los 2 ojos con un 40% de fiabilidad, pero
    //no sirve para deteccion de ojos cerrados, para eso entrenar con mcs.lefteye.xml (rigth tbm)
    // y lefteye_2splits.xml


    //Tabla para una buena busqueda de ojos segun regiones de busqueda (ver pag 239)
    const float EYE_SX = 0.16f;
    const float EYE_SY = 0.26f;
    const float EYE_SW = 0.30f;
    const float EYE_SH = 0.28f;

    //Aca sacamos las dos regiones de los ojos del rostro detectado por detectLargestObject

    int leftX = cvRound(face.cols * EYE_SX);
    int topY = cvRound(face.rows * EYE_SY);
    int widthX = cvRound(face.cols * EYE_SW);
    int heightY = cvRound(face.rows * EYE_SH);
    int rightX = cvRound(face.cols * (1.0-EYE_SX-EYE_SW) );  //Comienza por la esquina del ojo derecho

    Mat topLeftOfFace = face(Rect(leftX, topY, widthX, heightY));
    Mat topRightOfFace = face(Rect(rightX, topY, widthX, heightY));
    Rect leftEyeRect, rightEyeRect; //Donde vamos a guardar los ojos


    // Devuelve la ventana de búsqueda de la persona que llama, si se desea.
    if (searchedLeftEye)
        *searchedLeftEye = Rect(leftX, topY, widthX, heightY);
    if (searchedRightEye)
        *searchedRightEye = Rect(rightX, topY, widthX, heightY);

    //Buscamos los ojos usando el primer metodo de deteccion
    detectLargestObject(topLeftOfFace, eyeCascade1, leftEyeRect, topLeftOfFace.cols);
    detectLargestObject(topRightOfFace, eyeCascade1, rightEyeRect, topRightOfFace.cols);

    //Si no funciona el primero buscamos con el segundo
    if (leftEyeRect.width <= 0 && !eyeCascade2.empty()) {
        detectLargestObject(topLeftOfFace, eyeCascade2, leftEyeRect, topLeftOfFace.cols);
        }

    if (rightEyeRect.width <= 0 && !eyeCascade2.empty()) {
        detectLargestObject(topRightOfFace, eyeCascade2, rightEyeRect, topRightOfFace.cols);
        }


    //Se realiza un chequeo, para ver si los ojos fueron detectados
    if (leftEyeRect.width > 0) {
        leftEyeRect.x += leftX; // Ajusta el rectángulo del ojo izquierdo porque el borde de la cara fue removido.
        leftEyeRect.y += topY;
        leftEye = Point(leftEyeRect.x + leftEyeRect.width/2, leftEyeRect.y + leftEyeRect.height/2);
        //Sirve para dibujar el circulo en el ojo detectado !!
    }
    else {
        leftEye = Point(-1, -1);   // Retorna un punto inválido
    }

    if (rightEyeRect.width > 0) { // Ajusta el rectángulo del ojo derecho
        rightEyeRect.x += rightX;
        rightEyeRect.y += topY;
        rightEye = Point(rightEyeRect.x + rightEyeRect.width/2, rightEyeRect.y + rightEyeRect.height/2);
    }
    else {
        rightEye = Point(-1, -1);    // Retorna un punto inválido
}

}
// Ecualiza los ambos lados de la cara por separado.
void equalizeLeftAndRightHalves(Mat &faceImg)
{

    int w = faceImg.cols;
    int h = faceImg.rows;

    // 1) equalizamos toda la cara
    Mat wholeFace;
    equalizeHist(faceImg, wholeFace);

    // 2) luego la parte derecha e izquierda por separado
    int midX = w/2;
    Mat leftSide = faceImg(Rect(0,0, midX,h));
    Mat rightSide = faceImg(Rect(midX,0, w-midX,h));
    equalizeHist(leftSide, leftSide);
    equalizeHist(rightSide, rightSide);

    // 3) Combinamos las 3 partes obtenidas anteriormente, de modo de obtener una transicion suave
    for (int y=0; y<h; y++) {
        for (int x=0; x<w; x++) {
            int v;
            if (x < w/4) {          // Left 25%:
                v = leftSide.at<uchar>(y,x);
            }
            else if (x < w*2/4) {   // Mid-left 25%:
                int lv = leftSide.at<uchar>(y,x);
                int wv = wholeFace.at<uchar>(y,x);

                float f = (x - w*1/4) / (float)(w*0.25f);
                v = cvRound((1.0f - f) * lv + (f) * wv);
            }
            else if (x < w*3/4) {   // Mid-right 25%:
                int rv = rightSide.at<uchar>(y,x-midX);
                int wv = wholeFace.at<uchar>(y,x);

                float f = (x - w*2/4) / (float)(w*0.25f);
                v = cvRound((1.0f - f) * wv + (f) * rv);
            }
            else {                  // Right 25%:
                v = rightSide.at<uchar>(y,x-midX);
            }
            faceImg.at<uchar>(y,x) = v;
        }// fin lazo x
    }//fin lazo y
}

Mat getPreprocessedFace(Mat &srcImg, int desiredFaceWidth, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, bool doLeftAndRightSeparately, Rect *storeFaceRect, Point *storeLeftEye, Point *storeRightEye, Rect *searchedLeftEye, Rect *searchedRightEye)
{


    // Usa caras cuadradas
    int desiredFaceHeight = desiredFaceWidth;

    // Marca las regiones de la cara y ojos como inválidas en caso de que no sean detectadas.
    if (storeFaceRect)
        storeFaceRect->width = -1;
    if (storeLeftEye)
        storeLeftEye->x = -1;
    if (storeRightEye)
        storeRightEye->x= -1;
    if (searchedLeftEye)
        searchedLeftEye->width = -1;
    if (searchedRightEye)
        searchedRightEye->width = -1;

    // Busca la cara mas grande
    Rect faceRect;
    detectLargestObject(srcImg, faceCascade, faceRect);

    // Verifica si la cara fue detectada
    if (faceRect.width > 0) {

        if (storeFaceRect)
            *storeFaceRect = faceRect;

        Mat faceImg = srcImg(faceRect);    //Toma la cara detectada

       // Convierte de RGB a escala de grises, si ya esta no hace nada
        Mat gray;
        if (faceImg.channels() == 3) {
            cvtColor(faceImg, gray, CV_BGR2GRAY);
        }
        else if (faceImg.channels() == 4) {
            cvtColor(faceImg, gray, CV_BGRA2GRAY);
        }
        else {

            gray = faceImg;
        }

        // Busca los 2 ojos en la imagen con máxima resolución
        Point leftEye, rightEye;

        //Funcion que detecta ambos ojos
        detectBothEyes(gray, eyeCascade1, eyeCascade2, leftEye, rightEye, searchedLeftEye, searchedRightEye);

        if (storeLeftEye)
            *storeLeftEye = leftEye;
        if (storeRightEye)
            *storeRightEye = rightEye;

        // Veo si los dos ojos fueron detectados
        if (leftEye.x >= 0 && rightEye.x >= 0) {
            // Cambio el tamaño de la imagen al mismo que poseen las del set de entrenamiento
            // Si se encontraron ambos ojos, se procesa la imagen para obtener una posición de ojos ideal.
            // Obtiene el punto central entre los dos ojos

            Point2f eyesCenter = Point2f( (leftEye.x + rightEye.x) * 0.5f, (leftEye.y + rightEye.y) * 0.5f );

            //Obtiene el ángulo entre los dos ojos
            double dy = (rightEye.y - leftEye.y);
            double dx = (rightEye.x - leftEye.x);
            double len = sqrt(dx*dx + dy*dy);
            double angle = atan2(dy, dx) * 180.0/CV_PI; // Convierte de radianes a grados

            const double DESIRED_RIGHT_EYE_X = (1.0f - DESIRED_LEFT_EYE_X);
             // Se calcula el tamaño que es necesario agrandar la imagen.
            double desiredLen = (DESIRED_RIGHT_EYE_X - DESIRED_LEFT_EYE_X) * desiredFaceWidth;
            double scale = desiredLen / len;
            // Obtiene la matriz de transformación para rotar y escalar la cara a un ángulo y tamaño deseado
            Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, scale);
            // Traslada el centro de los ojos a una posición deseada
            rot_mat.at<double>(0, 2) += desiredFaceWidth * 0.5f - eyesCenter.x;
            rot_mat.at<double>(1, 2) += desiredFaceHeight * DESIRED_LEFT_EYE_Y - eyesCenter.y;

            // Aplica la matriz de transformación
            Mat warped = Mat(desiredFaceHeight, desiredFaceWidth, CV_8U, Scalar(128)); // Clear the output image to a default grey.
            warpAffine(gray, warped, rot_mat, warped.size());

            //Funcion equalizadora
            // Se le cambia el brillo y el contraste a la imagen
            if (!doLeftAndRightSeparately) {
                // Se aplica a toda la cara
                equalizeHist(warped, warped);
            }
            else {
                // Se aplica a cada mitad de la cara por separado
                equalizeLeftAndRightHalves(warped);
            }
            // Utiliza un "Filtro Bilateral" para reducir el ruido suavizando la imagen, manteniendo los bordes de la cara
            Mat filtered = Mat(warped.size(), CV_8U);
            bilateralFilter(warped, filtered, 0, 20.0, 2.0);
            // Dibuja una elipse en el medio de la cara
            Mat mask = Mat(warped.size(), CV_8U, Scalar(0)); // Comienza con una máscara vacía
            Point faceCenter = Point( desiredFaceWidth/2, cvRound(desiredFaceHeight * FACE_ELLIPSE_CY) );
            Size size = Size( cvRound(desiredFaceWidth * FACE_ELLIPSE_W), cvRound(desiredFaceHeight * FACE_ELLIPSE_H) );
            ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(255), CV_FILLED);

           // Usa la máscara para remover píxeles externos
            Mat dstImg = Mat(warped.size(), CV_8U, Scalar(128)); // Clear the output image to a default gray.

            // Aplica la máscara elíptica en la cara
            filtered.copyTo(dstImg, mask);  // Copia los pixels no enmascarados desde la imagen filtrada a dstImg.

            return dstImg;
        }

    }
    return Mat();
}
