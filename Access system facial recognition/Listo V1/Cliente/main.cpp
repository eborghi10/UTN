/*******************************************************************************************************
*                                    PROGRAMA CLIENTE VERSION 1
*
*  El mismo se va a encargar de detectar y enviar los rostros al servidor para que este le
*  responda si la persona se encuentra o no en la base de datos y de esta manera puede o no acceder a
*  al recurso restringido.
*
********************************************************************************************************/

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
#include "opencv2/opencv.hpp"
#include <unistd.h>
#include <sys/stat.h>
#include "preprocessFace.h"


//***********************************************+
//      Variables para TCP
//**********************************************
struct sockaddr_in their_addr;
struct hostent *he;
int numbytes, imgSize , sockfd;
char Nombre[20];
int sockfd2;
struct sockaddr_in their_addr2;
//**********************************************


//***************************************+
//      Variables y constantes
//***************************************
//C++
using namespace cv;
using namespace std;

//****************************************************************************
//Tamaño de rostro a detectar, es el recatangulo que vamos a obtener
const int faceWidth = 70;
const int faceHeight = faceWidth;

const char *faceCascadeFilename = "lbpcascade_frontalface.xml";     // LBP face detector.
const char *eyeCascadeFilename1 = "haarcascade_eye.xml";               // Basic eye detector for open eyes only.
const char *eyeCascadeFilename2 = "haarcascade_eye_tree_eyeglasses.xml"; // Basic eye detector for open eyes if they might wear glasses.

//****************************************************************************

const char *windowName = "Control Facial";   //Nombre de la ventana main
const bool preprocessLeftAndRightSeparately = true;

int cont; //Cuenta caras

void Reconocido(void);

//************************************************

//***********************************************************************
//                        Funciones
//***********************************************************************

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
//******************************************************************************************

//******************************************************************************************
//                      Acceso camara, captura
//                      Solo se activa, no se toman datos !!!
//******************************************************************************************
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


//******************************************************************************************
//                         Loop hasta ESC
//******************************************************************************************

void recognizeAndTrainUsingWebcam(VideoCapture &videoCapture, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2)
{


  while (true) {


//******************************************************************
        // Toma los frames de la camara !!

        Mat cameraFrame;
        videoCapture >> cameraFrame;

        if( cameraFrame.empty() ) {
            cerr << "ERROR: No se puedo obtener el proximo frame." << endl;
            exit(1);
        }

        //Obtenemos una copia del frame sobre el cual vamos a dibujar
        Mat displayedFrame;
        cameraFrame.copyTo(displayedFrame);

//********************************************************************
        Rect faceRect;
        Rect searchedLeftEye, searchedRightEye;
        Point leftEye, rightEye;

        Mat preprocessedFace = getPreprocessedFace(displayedFrame, faceWidth, faceCascade, eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately, &faceRect, &leftEye, &rightEye, &searchedLeftEye, &searchedRightEye);


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

if(preprocessedFace.data){
        //Con esto nos aseguramos de enviar siempre que se detecte un rostro
        try
        {
        cameraFrame = (cameraFrame.reshape(0,1)); //Para que sea un envio continuo
        //Enviamos los Frames actuales tomados
        if(send(sockfd, (void*)cameraFrame.data , 921600 , 0) == -1)
        {   perror("send");
            close(sockfd);
            exit(0);
            }
        usleep(100);
              }// try para el control de excepciones
        catch (std::exception& e)
        {
            std::cerr << e.what() << std::endl;
        }

}

       //Mostramos la captura actual
       namedWindow(windowName);
       imshow(windowName, displayedFrame);
       waitKey(20);

    }//end while
}


void Reconocido()
{
        //En esta funcion nos encargamos de recibir el nombre de la persona identificada o no y mostrarlo en pantalla

        if ((numbytes=recv(sockfd, Nombre, sizeof(Nombre), 0)) == -1)
        {
                perror("recv");
                exit(1);
        }

        Nombre[numbytes]='\0';

        if(strcmp(Nombre, "Desconocido")){
            if(strcmp(Nombre, ""))
            cout << "\n\n Accedido, Hola: "<<Nombre<<endl;
            }
        else
            cout << "\n\n Persona No identificada .. "<<endl;

}


int init_TCP( int argc, char *argv[])
{       //Inicializamos los puertos a escuchar para las conexiones TCPs empleadas
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
        their_addr.sin_port = htons(3491);
        their_addr.sin_addr = *((struct in_addr *)he->h_addr);
        bzero(&(their_addr.sin_zero), 8);


        //Config TCP2 para el envio del Keep alive

          if ((sockfd2 = socket(AF_INET, SOCK_STREAM, 0)) == -1)
        {
                perror("socket");
                exit(1);
        }
        their_addr2.sin_family = AF_INET;
        their_addr2.sin_port = htons(3490);
        their_addr2.sin_addr = *((struct in_addr *)he->h_addr);
        bzero(&(their_addr2.sin_zero), 8);

return 0;

}



//*************************************************************************************************+
//*************************************************************************************************+
//                           MAIN
//*************************************************************************************************+
//*************************************************************************************************+
int main(int argc, char *argv[])
{

      CascadeClassifier faceCascade;
      CascadeClassifier eyeCascade1;
      CascadeClassifier eyeCascade2;
      VideoCapture cap;

      initWebcam(cap,0);
      cap.set(CV_CAP_PROP_FRAME_WIDTH,640);
      cap.set(CV_CAP_PROP_FRAME_HEIGHT,480);

      initDetectors(faceCascade, eyeCascade1, eyeCascade2);

//****************************************************
//              TCP
//****************************************************
        init_TCP(argc, argv);
//****************************************************

       // Creamos la ventana donde vamos a mostrar las cosas
       namedWindow(windowName);

       //Realizamos una primera conexion al puerto ddedicado a los Frames y respuestas
       if (connect(sockfd, (struct sockaddr *)&their_addr,
                           sizeof(struct sockaddr)) == -1)
        {
                cout << "\n Servidor desconectado.."<<endl;
                exit(1);
        }
       //Esta segunda conexion se realiza para mantener el Keep Alive con el servidor
       if (connect(sockfd2, (struct sockaddr *)&their_addr2,
                           sizeof(struct sockaddr)) == -1)
        {
                cout << "\n Servidor desconectado.."<<endl;
                exit(1);
        }

        cout << "Conectado !! " << endl;

        if(!fork()){
            // hacemos que se encargue un child de esto, solo envia los frames al servidor
            recognizeAndTrainUsingWebcam(cap, faceCascade, eyeCascade1, eyeCascade2);
              }//Hijo 1


        if(!fork())
        {   //Este child se encargará de mantenerse enviando siempre que el cliente ande bien
            //el vivo, cuya funcion es mantener el canal abierto con el servidor
            while(1){
            //Solo se encarga de enviar el sigo vivo cada x min
            sleep(110);
            if(send(sockfd2, "vivo" , 4 , 0) == -1)
            {
                perror("sendadasdds");
                close(sockfd);
                exit(0);
            }
            }//While

        }//Hijo 2


        while(1){
            //Padre: este se encarga de esperar la respuesta del servidor
            Reconocido();

         }

        //Se toco ESC
        close(sockfd);
        cap.release();

    return 0;
}


//*************************************************************************************************+


