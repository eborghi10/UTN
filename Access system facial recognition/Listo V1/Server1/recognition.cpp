#include "recognition.h"

Ptr<FaceRecognizer> learnCollectedFaces(const vector<Mat> preprocessedFaces, const vector<int> faceLabels)
{
    Ptr<FaceRecognizer> model;

    bool haveContribModule = initModule_contrib();
    if (!haveContribModule) {
        cerr << "ERROR: The 'contrib' module is needed for FaceRecognizer but has not been loaded into OpenCV!" << endl;
        exit(1);
    }

    //Creamos el vector tipo facerecognizer empleando el metodo de Fisher !!!
    //En esta variable es donde se va a generar el entrenamiento con todos los faces tomados para luego
    //realizar las comparaciones pertinenetes.
    model = Algorithm::create<FaceRecognizer>("FaceRecognizer.Fisherfaces");
    if (model.empty()) {
        cerr << "ERROR: The FaceRecognizer algorithm [" << "FaceRecognizer.Fisherfaces"<< "] is not available in your version of OpenCV. Please update to OpenCV v2.4.1 or newer." << endl;
        exit(1);
    }

    //Aca es donde realizamos el entrenamiento !!
    model->train(preprocessedFaces, faceLabels);

    return model;
}


Mat getImageFrom1DFloatMat(const Mat matrixRow, int height)
{
    Mat rectangularMat = matrixRow.reshape(1, height);
    //8 bits
    Mat dst;
    normalize(rectangularMat, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    return dst;
}

// Generar una cara aproximadamente reconstruido por back-proyección de los vectores propios y valores propios de la (pre-procesado) cara dada.
Mat reconstructFace(const Ptr<FaceRecognizer> model, const Mat preprocessedFace)
{
    //sólo podemos reconstruir el rostro de algunos tipos de modelos FaceRecognizer (es decir: Eigenfaces o Fisherfaces),
    try {
        // Obtener algunos datos requeridos en el modelo FaceRecognizer.
        Mat eigenvectors = model->get<Mat>("eigenvectors");
        Mat averageFaceRow = model->get<Mat>("mean");

        int faceHeight = preprocessedFace.rows;

        // Proyecto de la imagen de entrada sobre el subespacio PCA.
        Mat projection = subspaceProject(eigenvectors, averageFaceRow, preprocessedFace.reshape(1,1));

        // Generar la cara reconstruida de nuevo desde el subespacio PCA.
        Mat reconstructionRow = subspaceReconstruct(eigenvectors, averageFaceRow, projection);

        // Convertir la matriz fila a una imagen regular de 8 bits. Que sea una imagen de forma rectangular en lugar de una sola fila.
        Mat reconstructionMat = reconstructionRow.reshape(1, faceHeight);
        // Convertimos 8bits flotantes a 8 bits uchar
        Mat reconstructedFace = Mat(reconstructionMat.size(), CV_8U);
        reconstructionMat.convertTo(reconstructedFace, CV_8U, 1, 0);

        return reconstructedFace;

    } catch (cv::Exception e) {
        return Mat();
    }
}


// Comparar dos imágenes por conseguir el error (raíz cuadrada de la suma de los errores al cuadrado).
double getSimilarity(const Mat A, const Mat B)
{
    if (A.rows > 0 && A.rows == B.rows && A.cols > 0 && A.cols == B.cols) {
        // Calcula el error de dos imagenes
        double errorL2 = norm(A, B, CV_L2);
        //Se lleva a una escala acorde
        double similarity = errorL2 / (double)(A.rows * A.cols);
        return similarity;
    }
    else {
        return 100000000.0;  // Return un mal numero
    }
}
