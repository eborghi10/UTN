/********************************************************************
*       PROGRAMA PARA CARGAR BASE DE DATOS (DB) VERSION 2
*
*  Nos permite realizar la carga de la base de datos generando para
*  ello un archivo del tipo YML. Nos va a permitir realizar la
*  actualizacion de la misma en tiempo de ejecución.
*
*********************************************************************/

#include <iostream>
#include <opencv2/objdetect/objdetect.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <ctime>
#include <string>
#include <stdlib.h>
#include <errno.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netdb.h>
#include <sys/socket.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/shm.h>
#include "opencv2/opencv.hpp"
#include "preprocessFace.h"
#include "recognition.h"

#define SHM_SIZE 1024

//Shared
key_t key;
int shmid, fds;
char *addr1;

typedef struct {
  int cantidad;
  char Nombres[20][20];
  int Actualizar;
  int prim;
  int nom;
  int exit;
  } areacompartida;

areacompartida *pAC;


//*************************************************

using namespace cv;
using namespace std;

//Para salir tocando la letra ESC
#if !defined VK_ESCAPE
    #define VK_ESCAPE 0x1B      // Escape character (27)
#endif

const float UNKNOWN_PERSON_THRESHOLD = 0.7f;
const int faceWidth = 70;
const int faceHeight = faceWidth;
const double CHANGE_IN_IMAGE_FOR_COLLECTION = 0.3;
const double CHANGE_IN_SECONDS_FOR_COLLECTION = 1.0;

const char *windowName = "Control Facial";
const int BORDER = 8;  //Tamaño de los cuadritos con las intrucciones Puede volar
const bool preprocessLeftAndRightSeparately = true;

enum MODES {MODE_STARTUP=0, MODE_DETECTION, MODE_NEW, MODE_CLOSE, MODE_send};
MODES m_mode = MODE_STARTUP;

int cont; //Cuenta caras
//Para el agregado de personas nuevas a la base de datos
int m_selectedPerson = -1, m_numPersons = 0;
vector<int> m_latestFaces;
//********************************
// Position of GUI buttons:
Rect m_rcBtnAdd;
Rect m_rcBtnClose;
//*********************************

//************************************************
const char *faceCascadeFilename = "lbpcascade_frontalface.xml";     // LBP face detector.
const char *eyeCascadeFilename1 = "haarcascade_eye.xml";               // Basic eye detector for open eyes only.
const char *eyeCascadeFilename2 = "haarcascade_eye_tree_eyeglasses.xml"; // Basic eye detector for open eyes if they might wear glasses.
///************************************************************//

//*************************************************
//           FUNCIONES
//*************************************************
void initDetectors(CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2)
{
    try {
        // Lo cargamos
        faceCascade.load(faceCascadeFilename);
    } catch (cv::Exception &e) {}
    if ( faceCascade.empty() ) {
        cerr << "ERROR: No se pudo cargar el Face" << faceCascadeFilename << "]!" << endl;
        cerr << "Copia el XML que queres cargar en el directorio actual NABO !!!" << endl;
        exit(1);
    }
    cout << "Se cargo con exito la XML deteccion de face [" << faceCascadeFilename << "]." << endl;
    // Load the Eye Detection cascade classifier xml file.
    try {
        eyeCascade1.load(eyeCascadeFilename1);
    } catch (cv::Exception &e) {}
    if ( eyeCascade1.empty() ) {
        cerr << "ERROR: No se pudo cargar el primer ojo [" << eyeCascadeFilename1 << "]!" << endl;
        cerr << "Copia el XML que queres cargar en el directorio actual NABO !!!" << endl;
        exit(1);
    }
    cout << "Se cargo con exito la XML deteccion de 1er ojo [" << eyeCascadeFilename1 << "]." << endl;
    // Load the Eye Detection cascade classifier xml file.
    try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
        eyeCascade2.load(eyeCascadeFilename2);
    } catch (cv::Exception &e) {}
    if ( eyeCascade2.empty() ) {
        cerr << " ERROR: No se pudo cargar el segundo ojo [" << eyeCascadeFilename2 << "]." << endl;
        // Dont exit if the 2nd eye detector did not load, because we have the 1st eye detector at least.
        //exit(1);
    }
    else
        cout << "Se cargaron todos con exitos !!! [" << eyeCascadeFilename2 << "]." << endl;
}
///******************************************************************************************

///******************************************************************************************
///                      Acceso camara, captura
///                      Solo se activa, no se toman datos !!!
///******************************************************************************************
void initWebcam(VideoCapture &videoCapture, int cameraNumber)
{
    try {
        videoCapture.open(0);
    } catch (cv::Exception &e) {}
    if ( !videoCapture.isOpened() ) {
        cerr << "ERROR: No se puede acceder al device!" << endl;
        exit(1);
    }
    cout << "Cargando camara" << cameraNumber << "." << endl;
}


///******************************************************************************************
///                 Para escribir dentro de los rectangulos !!!
///******************************************************************************************

Rect drawString(Mat img, string text, Point coord, Scalar color, float fontScale = 0.6f, int thickness = 1, int fontFace = FONT_HERSHEY_COMPLEX)
{
    // Get the text size & baseline.
    int baseline=0;
    Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;

    // Adjust the coords for left/right-justified or top/bottom-justified.
    if (coord.y >= 0) {
        // Coordinates are for the top-left corner of the text from the top-left of the image, so move down by one row.
        coord.y += textSize.height;
    }
    else {
        // Coordinates are for the bottom-left corner of the text from the bottom-left of the image, so come up from the bottom.
        coord.y += img.rows - baseline + 1;
    }
    // Become right-justified if desired.
    if (coord.x < 0) {
        coord.x += img.cols - textSize.width + 1;
    }

    // Get the bounding box around the text.
    Rect boundingRect = Rect(coord.x, coord.y - textSize.height, textSize.width, baseline + textSize.height);

    // Draw anti-aliased text.
    putText(img, text, coord, fontFace, fontScale, color, thickness, CV_AA);

    // Let the user know how big their text is, in case they want to arrange things.
    return boundingRect;
}

///******************************************************************************************

Rect drawButton(Mat img, string text, Point coord, int minWidth = 0)
{
    int B = BORDER;
    Point textCoord = Point(coord.x + B, coord.y + B);
    // Get the bounding box around the text.
    Rect rcText = drawString(img, text, textCoord, CV_RGB(0,0,0));
    // Draw a filled rectangle around the text.
    Rect rcButton = Rect(rcText.x - B, rcText.y - B, rcText.width + 2*B, rcText.height + 2*B);
    // Set a minimum button width.
    if (rcButton.width < minWidth)
        rcButton.width = minWidth;
    // Make a semi-transparent white rectangle.
    Mat matButton = img(rcButton);
    matButton += CV_RGB(90, 90, 90);
    // Draw a non-transparent white border.
    rectangle(img, rcButton, CV_RGB(2,41,255), 1, CV_AA);

    // Draw the actual text that will be displayed, using anti-aliasing.
    drawString(img, text, textCoord, CV_RGB(0,255,0));

    return rcButton;
}

///*********************************************************************+
///Usada para la deteccion de clicks, la ubicacion del mismo !!!
///*********************************************************************+
bool isPointInRect(const Point pt, const Rect rc)
{
    if (pt.x >= rc.x && pt.x <= (rc.x + rc.width - 1))
        if (pt.y >= rc.y && pt.y <= (rc.y + rc.height - 1))
            return true;

    return false;
}

///******************************************************************************************
/// Mouse event handler. Called automatically by OpenCV when the user clicks in the GUI window.
///******************************************************************************************

void onMouse(int event, int x, int y, int, void*)
{
// Sólo nos preocupamos por los clics del ratón a la izquierda, no la derecha clics del ratón o el movimiento del ratón.
    if (event != CV_EVENT_LBUTTONDOWN)
        return;

    //PAra ver si se hizo click en uno de los botones !!!
    Point pt = Point(x,y);

    if (isPointInRect(pt, m_rcBtnAdd)) {
          m_mode = MODE_CLOSE;
    }

    else if (isPointInRect(pt, m_rcBtnClose)) {


   // if ((m_numPersons == 0) || (m_latestFaces[m_numPersons-1] >= 0)) {
        m_numPersons++;
        m_selectedPerson = m_numPersons - 1;
       //     m_latestFaces.push_back(-1); // Allocate space for an extra person.
        cout << "Numero de personas en la base de datos: " << m_numPersons << endl;

         //   }

        m_mode = MODE_NEW;
   //*****************************************************

}
}//OnMouse

//Hasta aca se cargaron los XML usados para el preentrenamiento para la deteccion de rostros
//Se pueden emplear mas que arriba se detallan, solo añadirlos
//******************************************************************************************


    Rect faceRect;
    Rect searchedLeftEye, searchedRightEye;
    Point leftEye, rightEye;
    //Las variables tipo Rect y Point son de posicion, Rect para dibujar el rectangulo (para el face) y Point para los
    //circulos.

    //Contiene el modelo entrenado
    Ptr<FaceRecognizer> model, model2;
    //Contienen cantidad de personas y caras
    vector<Mat> preprocessedFaces;
    vector<int> faceLabels;

    bool gotFaceAndEyes = false; //Para saber cuando encontramos rostros
    int identity = -1, persona= -1, n_person;
    double porcentaje = 0.3;

    //Todas variables para el tratamiento y captura de los frames
    Mat cameraFrame;
    Mat displayedFrame;
    Mat preprocessedFace;



void preprocessed(){

        identity = -1;

        double old_time = 0;
        Mat old_prepreprocessedFace;

            // COMPROBAMOS SI SE DETECTO UNA CARA
            if (gotFaceAndEyes) {

                //Compruebe si este rostro se ve un poco diferente de la cara recogido previamente.
                double imageDiff = 10000000000.0;

                if (old_prepreprocessedFace.data) {
                    imageDiff = getSimilarity(preprocessedFace, old_prepreprocessedFace);
                }

                // También registro cuando sucedió.
                double current_time = (double)getTickCount();
                double timeDiff_seconds = (current_time - old_time)/getTickFrequency();

                // Sólo procesa la cara si es notablemente diferente de la trama anterior y si no ha habido espacio de tiempo notable.
                if ((imageDiff > CHANGE_IN_IMAGE_FOR_COLLECTION) && (timeDiff_seconds > CHANGE_IN_SECONDS_FOR_COLLECTION)) {

                //También agregue la imagen espejo para el conjunto de entrenamiento,
                //por lo que tenemos más datos de formación, así como para hacer frente a los rostros que miran a la izquierda o a la derecha.

                Mat mirroredFace;
                flip(preprocessedFace, mirroredFace, 1); //me da la mitad de la cara


                // Añadir las imágenes de la cara a la lista de rostros detectados.
                preprocessedFaces.push_back(preprocessedFace);
                preprocessedFaces.push_back(mirroredFace);

                faceLabels.push_back(m_selectedPerson); //m_selectedPerson es un int, da el numero de personas
                faceLabels.push_back(m_selectedPerson);

                cout << "Cara Guardada !!!! " << (preprocessedFaces.size()/2) << " DE PERSONA .." << m_selectedPerson << endl;

                // Guarde una copia de la cara procesado, para comparar en la próxima iteración.
                old_prepreprocessedFace = preprocessedFace;
                old_time = current_time;

                }//if cara distinta a la anterior


            } //Si detecto un rostro */


}//funcion


void Trainning()
{
            cout << "\n\n Entrenando ..." <<endl;

            bool haveEnoughData = true;
            //Compruebo que se hayan cargado mas de 2 personas en la BD
            if (m_numPersons < 1 || preprocessedFaces.size() <= 0 || preprocessedFaces.size() != faceLabels.size()) {
                cout << "Se necesita mas de 2 personas para entrenar !!! ..." << endl;
                haveEnoughData = false;
            }

            if (haveEnoughData) {
                model = learnCollectedFaces(preprocessedFaces, faceLabels);
                //Vamos a guardar el model entrenado con las caras tomadas anteriormente

	            if(!pAC->prim){
	                model->save("/home/emiliano/Desktop/XML/test.yml");
	                pAC->prim=1;
	                cout<<"\n creo archivo \n"<<endl;
	             }
	            else{
	                cout<<"\n abro archivo \n"<<endl;
	                FileStorage fs2("/home/emiliano/Desktop/XML/test.yml", FileStorage::APPEND);
	                model->save(fs2);
	               }
			}

}


//******************************************************************************************
//                         Loop hasta ESC
//******************************************************************************************

void recognizeAndTrainUsingWebcam(VideoCapture &videoCapture, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2)
{

// Carga el modo de arrancar !! en deteccion de rostro !!
    m_mode = MODE_DETECTION;

    string tom1="Recolectando Caras";

  while (true) 
  {


	//******************************************************************
        // Toma los frames de la camara !!
        videoCapture >> cameraFrame;

        if( cameraFrame.empty() ) {
            cerr << "ERROR: No se puedo obtener el proximo frame." << endl;
            exit(1);
        }

        //Obtener una copia de la trama de la cámara que podemos dibujar sobre.
        cameraFrame.copyTo(displayedFrame);
        preprocessedFace = getPreprocessedFace(displayedFrame, faceWidth, faceCascade, eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately, &faceRect, &leftEye, &rightEye, &searchedLeftEye, &searchedRightEye);

	//********************************************************************************+
	//                     RECTANGULO Y CIRCULOS EN LA CARA!!!
	//********************************************************************************+
            if (faceRect.width > 0) {

                rectangle(displayedFrame, faceRect, CV_RGB(204, 0, 0), 2, CV_AA);

            Scalar eyeColor = CV_RGB(102,255,51);
            if (leftEye.x >= 0) {   // Check if the eye was detected
                circle(displayedFrame, Point(faceRect.x + leftEye.x, faceRect.y + leftEye.y), 6, eyeColor, 1, CV_AA);
            }
            if (rightEye.x >= 0) {   // Check if the eye was detected
                circle(displayedFrame, Point(faceRect.x + rightEye.x, faceRect.y + rightEye.y), 6, eyeColor, 1, CV_AA);
            }}
	//********************************************************************************+


        if (m_mode == MODE_DETECTION) {

        }//Deteccion


        else if (m_mode == MODE_CLOSE) {
        cout << "\n\n Cerrando programa Base da Datos..."<<endl;
        //Cerramos todo lo que usamos
        Trainning();
        cout << "\n\n Cerrado con exito...."<<endl;
        //Guardamos la cantidad de personas y los nombres
        pAC->cantidad=m_numPersons;
        videoCapture.release();
        displayedFrame.release();
        cameraFrame.release();
        pAC->exit=1;
        break;
       }//Close


else if (m_mode == MODE_NEW) 
{
        gotFaceAndEyes = false;
        if (preprocessedFace.data){
            gotFaceAndEyes = true;

            }
        else cont--;	//No estoy detectando rostro
        //Preprocesamos y guardamos caras
        //Leyenda en pantalla mientras se este tomando rostros
        float txtSize = 0.9;
        drawString(displayedFrame, tom1, Point(BORDER, -BORDER-2), CV_RGB(0,0,0), txtSize);  // Black shadow.
        drawString(displayedFrame, tom1, Point(BORDER+1, -BORDER-1), CV_RGB(255,0,0), txtSize); //White text.
        //Genera un flash sobre la deteccion, para saber cuando se capturo un rostro
        Mat displayedFaceRegion = displayedFrame(faceRect);
        displayedFaceRegion += CV_RGB(90,90,90);

        preprocessed();
        //Ya tenemos suficiente para entrenar
        cont++;

        if(cont>40){
        //Ya tengo en base de datos 40 caras
        m_mode= MODE_DETECTION;
        cout << "Clickea New para añadir mas personas "<< endl;
        cont=0;
        //Ingresamos Nombre de la persona nueva
        pAC->cantidad=m_numPersons;
        pAC->nom=1;
        }
}

        //******************************//
        //   Dibuja los botones !!!
        //******************************//
        m_rcBtnAdd = drawButton(displayedFrame, "Terminar", Point(BORDER, BORDER));
        m_rcBtnClose = drawButton(displayedFrame, "Agregar",Point(m_rcBtnAdd.x, m_rcBtnAdd.y + m_rcBtnAdd.height), m_rcBtnAdd.width); //Point(m_rcBtnDel.x, m_rcBtnDel.y + m_rcBtnDel.height), m_rcBtnAdd.width);
        //******************************//

        //Mostramos el frame actual, lo que da una idea de tiempo real
        imshow(windowName, displayedFrame);

        //ESC
        char keypress = waitKey(20);
        if (keypress == VK_ESCAPE) {
            break;
        }

    }//end while
}


//*************************************************************************************************+
//*************************************************************************************************+
//                                       MAIN
//*************************************************************************************************+
//*************************************************************************************************+
int main(int argc, char *argv[])
{

//***********************************************
//		    Shared memory
//***********************************************
    //Creo la shared
    if ((key = ftok("/home/emiliano/Desktop/SHM.txt", 'o')) == -1) {
        perror("ftok");
        exit(1);
    }

    //Se conecta a la shared memory y la crea si no existe
    if ((shmid = shmget(key, 64*SHM_SIZE, 0644 | IPC_CREAT)) == -1) {
        perror("shmget");
        exit(1);
    }

   //Obtiene el puntero a memoria
    addr1 = (char *)shmat(shmid, (void *)0, 0);
    if (addr1 == (char *)(-1))
    {
        perror("shmat");
        exit(1);
    }

    //puntero al primer elemento del area compartida
     pAC = (areacompartida *) addr1;

//************************************************
//   Decalaramos las variables XML y las capturas
//     videos para inicializarlos y preentrenar
//************************************************

      CascadeClassifier faceCascade;
      CascadeClassifier eyeCascade1;
      CascadeClassifier eyeCascade2;
      VideoCapture cap;

      initWebcam(cap,0);
      cap.set(CV_CAP_PROP_FRAME_WIDTH,640);
      cap.set(CV_CAP_PROP_FRAME_HEIGHT,480);

      initDetectors(faceCascade, eyeCascade1, eyeCascade2);


       // Creamos la ventana donde vamos a mostrar las cosas
       namedWindow(windowName);

       //Chequeamos la cantidad de personas que vamos
       m_numPersons = pAC->cantidad;

        // Para detectar el click de mouse, llama a onMouse()
       setMouseCallback(windowName, onMouse, 0);

      if(!fork()){

                while(!pAC->exit){
                        if(pAC->nom){
                        cout<<"\n\n Ingrese SOLO su Nombre seguido de ENTER"<<endl;
                        scanf("%s", pAC->Nombres[pAC->cantidad-1]);
                        pAC->nom=0;
                        }
                }
                pAC->exit=0;
                exit(1);

       }

       recognizeAndTrainUsingWebcam(cap, faceCascade, eyeCascade1, eyeCascade2);

       cout << "\n Listo para cerrarse .."<<endl;

        //Actualizar base de datos
        if(!pAC->Actualizar){
            pAC->Actualizar=1;
        }
        //Se toco ESC
    cap.release();
    cvDestroyWindow(windowName);
	shmctl(shmid, IPC_RMID, 0); //Destruye shared memory
  	shmdt(addr1);
	exit(1);
	cout <<""<<endl;

    return 0;
}
