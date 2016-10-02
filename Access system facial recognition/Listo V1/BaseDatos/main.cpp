/*******************************************************************************************************
*                         PROGRAMA PARA CARGAR BASE DE DATOS (DB) VERSION 1
*
*  Nos permite realizar la carga de la base de datos mediante una comunicación TCP con el servidor
*
********************************************************************************************************/
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
#include "opencv2/opencv.hpp"
#include "preprocessFace.h"


//***********************************************+
//              TCP
//*************************************************
int sockfd, sockfd2, numbytes;
struct sockaddr_in their_addr, their_addr2;
struct hostent *he;
//***********************************************+

//*************************************************
//Variables para caso de nueva persona
char Nombre[20];
//***********************************************+

//*************************************************
//C++
using namespace cv;
using namespace std;

//Para salir tocando la letra ESC
#if !defined VK_ESCAPE
    #define VK_ESCAPE 0x1B      // Escape character (27)
#endif

const int faceWidth = 70;
const int faceHeight = faceWidth;

const char *windowName = "Carga Base de Datos (BD)";
const int BORDER = 8;
const bool preprocessLeftAndRightSeparately = true;

enum MODES {MODE_STARTUP=0, MODE_DETECTION, MODE_NEW, MODE_CLOSE, MODE_send};
MODES m_mode = MODE_STARTUP;
//Cuenta caras
int cont;

//********************************
// Position of GUI buttons:
Rect m_rcBtnAdd;
Rect m_rcBtnClose;
//*********************************

//************************************************
const char *faceCascadeFilename = "lbpcascade_frontalface.xml";     // LBP face detector.
const char *eyeCascadeFilename1 = "haarcascade_eye.xml";               // Basic eye detector for open eyes only.
const char *eyeCascadeFilename2 = "haarcascade_eye_tree_eyeglasses.xml"; // Basic eye detector for open eyes if they might wear glasses.
//************************************************************//


//*************************************************
//           FUNCIONES
//*************************************************
void initDetectors(CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2)
{
    try {
         faceCascade.load(faceCascadeFilename);
    } catch (cv::Exception &e) {}
    if ( faceCascade.empty() ) {
        cerr << "ERROR: No se pudo cargar el Face" << faceCascadeFilename << "]!" << endl;
        cerr << "Copia el XML que queres cargar en el directorio actual NABO !!!" << endl;
        exit(1);
    }
    cout << "Se cargo con exito la XML deteccion de face [" << faceCascadeFilename << "]." << endl;
    // Load Eye Detection cascade classifier xml file.
    try {
        eyeCascade1.load(eyeCascadeFilename1);
    } catch (cv::Exception &e) {}
    if ( eyeCascade1.empty() ) {
        cerr << "ERROR: No se pudo cargar el primer ojo [" << eyeCascadeFilename1 << "]!" << endl;
        cerr << "Copia el XML que queres cargar en el directorio actual NABO !!!" << endl;
        exit(1);
    }
    cout << "Se cargo con exito la XML deteccion de 1er ojo [" << eyeCascadeFilename1 << "]." << endl;
    // Load Eye Detection cascade classifier xml file.
    try {
        eyeCascade2.load(eyeCascadeFilename2);
    } catch (cv::Exception &e) {}
    if ( eyeCascade2.empty() ) {
        cerr << " ERROR: No se pudo cargar el segundo ojo [" << eyeCascadeFilename2 << "]." << endl;
        exit(1);
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
        cout << "Se clickeo en Cargar New person" << endl;
        m_mode = MODE_NEW;
    }

}//OnMouse


///******************************************************************************************
///                         Loop hasta ESC
///******************************************************************************************

void recognizeAndTrainUsingWebcam(VideoCapture &videoCapture, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2)
{

    // Carga el modo de arrancar !! en deteccion de rostro !!
    m_mode = MODE_DETECTION;

    string tom1="Recolectando Caras";

  while (true) {


//******************************************************************
        // Toma los frames de la camara !!

        Mat cameraFrame;
        videoCapture >> cameraFrame;

        if( cameraFrame.empty() ) {
            cerr << "ERROR: No se puedo obtener el proximo frame." << endl;
            exit(1);
        }

        //Obtener una copia de la trama de la cámara que podemos dibujar sobre.
        Mat displayedFrame;
        cameraFrame.copyTo(displayedFrame);

//********************************************************************

        /// Find a face and preprocess it to have a standard size and contrast & brightness.
        Rect faceRect;  /// Position of detected face.
        Rect searchedLeftEye, searchedRightEye; /// top-left and top-right regions of the face, where eyes were searched.
        Point leftEye, rightEye;    /// Position of the detected eyes.

        Mat preprocessedFace = getPreprocessedFace(displayedFrame, faceWidth, faceCascade, eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately, &faceRect, &leftEye, &rightEye, &searchedLeftEye, &searchedRightEye);

//********************************************************************************+
//                     RECTANGULO Y CIRCULOS EN LA CARA!!!
//********************************************************************************+
            if (faceRect.width > 0) {

                rectangle(displayedFrame, faceRect, CV_RGB(204, 0, 0), 2, CV_AA);

            Scalar eyeColor = CV_RGB(102,255,51);
            if (leftEye.x >= 0) {   //Chequemos ojo detectado
                circle(displayedFrame, Point(faceRect.x + leftEye.x, faceRect.y + leftEye.y), 6, eyeColor, 1, CV_AA);
            }
            if (rightEye.x >= 0) {   //Chequemos ojo detectado
                circle(displayedFrame, Point(faceRect.x + rightEye.x, faceRect.y + rightEye.y), 6, eyeColor, 1, CV_AA);
            }}
//********************************************************************************+

        if (m_mode == MODE_DETECTION) {
                //No hacemos nada !!!
        }//Deteccion

        else if (m_mode == MODE_CLOSE) {
        //En este modo, lo primero que hacemos es modificar flag = 2, para indicarle al servidor que busque parecidos
        Nombre[0]=2; //Finalizacion en server
        cout << "\n\n Cerrando programa Base da Datos..."<<endl;
        //Cerramos todo lo que usamos
        if (send(sockfd, Nombre, sizeof(Nombre), 0) == -1)
        {      perror("send flag");
               close(sockfd2);
               exit(0);
         }
         break; //Para salir del loop
       }//Mode Close

        else if (m_mode == MODE_NEW) {
                cout<<"\n Modo add person"<<endl;
                  //Enviamos flag que identifica que vamos a hacer
                 Nombre[0]=1;//En 2 es reconocimiento
                 cout << "\n Flag 1"<< endl;
                //Ahora mandamos Nombre al servidor
                 if (send(sockfd, Nombre, sizeof(Nombre), 0) == -1)
                 {      perror("send flag");
                        close(sockfd2);
                        exit(0);
                  }
                cout << "\n Enviamos ...  " << endl;

     Nombre[0]=0;
     m_mode = MODE_send;

     }//mode new

    else if (m_mode == MODE_send) {
        //***********************************************************************************************//
        //Esto me dibuja la leyenda: Recolectando caras !!
        float txtSize = 0.9;
        drawString(displayedFrame, tom1, Point(BORDER, -BORDER-2), CV_RGB(0,0,0), txtSize);  // Black shadow.
        drawString(displayedFrame, tom1, Point(BORDER+1, -BORDER-1), CV_RGB(255,0,0), txtSize); //White text.
        //***********************************************************************************************//

    if(preprocessedFace.data){
        //Chequemos que se haya detectado un rostro, es la condicion para que enviemos el frame
        try //Chequeamos errores en el proceso
        {
        cameraFrame = (cameraFrame.reshape(0,1));//Esto se hace antes de enviar con el fin de asegurarnos un envío contínuo

        //*****************************************************//
        //Dibuja el flash sobre el rectangulo del rostro detectado
        Mat displayedFaceRegion = displayedFrame(faceRect);
        displayedFaceRegion += CV_RGB(90,90,90);
        //*****************************************************//

        //Envíamos el frame detectado
        if(send(sockfd, (void*)cameraFrame.data , 921600 , 0) == -1)
        {   perror("send");
            close(sockfd);
            exit(0);
            }
        cout<<"send image finished"<<endl;
        cont++;
        usleep(100);

              }// try para el control de excepciones
        catch (std::exception& e)
        {
            std::cerr << e.what() << std::endl;
        }

        }//Contador de rostros

    if(cont>39 ){//llegue a 40 caras
            m_mode = MODE_DETECTION;
            cont=0;
            cout << "\n\n Rostro tomado con exito"<<endl;
            }

       }//Mode Send


        //******************************//
        //   Dibuja los botones !!!
        //******************************//
        m_rcBtnAdd = drawButton(displayedFrame, "Terminar", Point(BORDER, BORDER));
        m_rcBtnClose = drawButton(displayedFrame, "Agregar",Point(m_rcBtnAdd.x, m_rcBtnAdd.y + m_rcBtnAdd.height), m_rcBtnAdd.width);
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


int init_TCP( int argc, char *argv[])
{
        struct hostent *he;
        if (argc != 2)
        {
                fprintf(stderr,"usage: client hostname\n");
                exit(1);
        }

        if ((he=gethostbyname(argv[1])) == NULL)
        {
                perror("gethostbyname");
                exit(1);
        }

        if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) == -1)
        {
                perror("socket");
                exit(1);
        }

        their_addr.sin_family = AF_INET;
        their_addr.sin_port = htons(3490); //Puerto a emplear para la conexion el servidor
        their_addr.sin_addr = *((struct in_addr *)he->h_addr);
        bzero(&(their_addr.sin_zero), 8);

return 0;

}



//*************************************************************************************************+
//*************************************************************************************************+
//                                     MAIN
//*************************************************************************************************+
//*************************************************************************************************+
int main(int argc, char *argv[])
{


//*******************************************
// Decalaramos las variables XML y las capturas
// videos para inicializarlos y preentrenar
//*******************************************

      CascadeClassifier faceCascade;
      CascadeClassifier eyeCascade1;
      CascadeClassifier eyeCascade2;
      VideoCapture cap;

      initWebcam(cap,0);
      cap.set(CV_CAP_PROP_FRAME_WIDTH,640);
      cap.set(CV_CAP_PROP_FRAME_HEIGHT,480);

      initDetectors(faceCascade, eyeCascade1, eyeCascade2);

//****************************************************
//           TCP
//****************************************************
       init_TCP(argc, argv);
//****************************************************

       // Creamos la ventana donde vamos a mostrar
       namedWindow(windowName);

       // Para detectar el click de mouse, llama a onMouse()
       setMouseCallback(windowName, onMouse, 0);


       //Conectamos al Servidor !!
       if (connect(sockfd, (struct sockaddr *)&their_addr,
                           sizeof(struct sockaddr)) == -1)
        {
                perror("connect: Problema con el servidor !!");
                exit(1);
        }

        cout << "Conectado !! " << endl;

        // Funcion principal, que realiza TODO .... se ejecuta hasta tocar ESC
        recognizeAndTrainUsingWebcam(cap, faceCascade, eyeCascade1, eyeCascade2);

        cout << "\n Listo para cerrarse .."<<endl;
        //Se toco ESC
        close(sockfd);
        cap.release();

    return 0;
}
