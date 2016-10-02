#include "detectObject.h"
#include "cv.h"
#include "highgui.h"
#include <stdio.h>

// Búsqueda de objetos tales como caras de la imagen utilizando los parámetros dados,
// almacenando el cv múltiple :: Rect en 'objetos'.
// Puede utilizar cascadas Haar o cascadas de LBP para Detección de la cara, o incluso los ojos, la boca.
// La entrada se encogió temporalmente a 'scaledWidth' para la detección mucho más rápido,
// ya que 200 es suficiente para encontrar caras.
void detectObjectsCustom(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth, int flags, Size minFeatureSize, float searchScaleFactor, int minNeighbors)
{

//************************************************
//Si la imagen de entrada no está en escala de grises
// se la convierte de color a escala de grises.
//************************************************
     Mat gray;

    if (img.channels() == 3) //Para camaras de pc
    {
        cvtColor(img, gray, CV_BGR2GRAY);
    }
    else if (img.channels() == 4) //Para fotos de celulares
    {
        cvtColor(img, gray, CV_BGRA2GRAY);
    }
    else {
        // Accedemos a la imagen de entrada directamente, puesto que ya está en escala de grises.
        gray = img;
    }
//************************************************




//************************************************
//Reducimos la imagen para acelerar la ejecucion !!
//************************************************

    Mat inputImg;

    float scale = img.cols / (float)scaledWidth;
    if (img.cols > scaledWidth) {
        // Reducir la imagen mientras se mantiene la misma relación de aspecto.
        int scaledHeight = cvRound(img.rows / scale);

        //Funcion para reducir la imagen
        resize(gray, inputImg, Size(scaledWidth, scaledHeight));

    }
    else {
        // Accedemos a la imagen de entrada directamente, puesto que ya es pequeño.
        inputImg = gray;
    }

//************************************************
    // Estandarizar el brillo y el contraste para mejorar las imágenes oscuras.
    Mat equalizedImg;
    //Esta funcion nativa de Opencv, en base a un histograma ajusta brillo y contraste de la imagen
    equalizeHist(inputImg, equalizedImg);
//************************************************

// Luego de pasarlo a gris, achicar la imagen y ecualizarla, ya podemos detectar
//un rostro usando detectMultiScale()


    // Detectar objetos en la pequeña imagen en escala de grises.
    cascade.detectMultiScale(equalizedImg, objects, searchScaleFactor, minNeighbors, flags, minFeatureSize);
    //Retorna objetcs

    //Ampliar los resultados si la imagen se contrae temporalmente antes de la detección.
    if (img.cols > scaledWidth) {
    //chequeo si se detecto algun rostro viendo la cantidad de elementos en objects.
        for (int i = 0; i < (int)objects.size(); i++ ) {
            objects[i].x = cvRound(objects[i].x * scale);
            objects[i].y = cvRound(objects[i].y * scale);
            objects[i].width = cvRound(objects[i].width * scale);
            objects[i].height = cvRound(objects[i].height * scale);
        }
    }

    // Asegúrese de que el objeto está completamente dentro de la imagen,
    //en caso de que estaba en una frontera.
    for (int i = 0; i < (int)objects.size(); i++ ) {
        if (objects[i].x < 0)
            objects[i].x = 0;
        if (objects[i].y < 0)
            objects[i].y = 0;
        if (objects[i].x + objects[i].width > img.cols)
            objects[i].x = img.cols - objects[i].width;
        if (objects[i].y + objects[i].height > img.rows)
            objects[i].y = img.rows - objects[i].height;
    }


//*******************************************************************************
// Regreso con los rectángulos de caras detectadas almacenados en "objetos".
//*******************************************************************************


}




// Busca un único objeto en la imagen, como la cara más grande, almacenando el resultado en 'largestObject'.
void detectLargestObject(const Mat &img, CascadeClassifier &cascade, Rect &largestObject, int scaledWidth)
{
    // Solo busca un solo objeto (El más grande en la imagen).
    int flags = CASCADE_FIND_BIGGEST_OBJECT;// | CASCADE_DO_ROUGH_SEARCH;
    // El siguiente código buscará todas las caras en la imagen
    // pero si se desea buscar solo la mas grande debería cambiarse a:
    // int flags = CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH;

    // Tamaño del objeto más pequeño
    Size minFeatureSize = Size(20, 20);

    // El grado de detalle debe ser la búsqueda. Debe ser mayor que 1,0. Tomamos 1,1f
    float searchScaleFactor = 1.1f;

    int minNeighbors = 4;

    //Realizar Objeto o detección de la cara,
    // buscando sólo 1 objeto (el más grande en la imagen).
    vector<Rect> objects;


    detectObjectsCustom(img, cascade, objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors);


    if (objects.size() > 0) {
        //Retorna solo el objeto detectado
        largestObject = (Rect)objects.at(0);
	}

    else {
        //Retorna invlido, es decir una posicion imposible
        largestObject = Rect(-1,-1,-1,-1);
    }
}

// Busca muchos objetos en la imagen almacenándolos en 'objects'.
void detectManyObjects(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth)
{
    //Busca varios objetos en una imagen
    int flags = CASCADE_SCALE_IMAGE;

    // El objeto mas chico en la imagen
    Size minFeatureSize = Size(20, 20);

    // El grado de detalle debe ser la búsqueda. Debe ser mayor que 1,0.
    float searchScaleFactor = 1.1f;

    // Cuanto espacio debe haber entre objetos. Depende de las falsas detecciones del sistema.
    // minNeighbors=2 significa un montón de buenas+malas detecciones, y minNeighbors=6 significa solo buenas
    // detecciones pero algunas se pierden.
    int minNeighbors = 4;


    detectObjectsCustom(img, cascade, objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors);
}
