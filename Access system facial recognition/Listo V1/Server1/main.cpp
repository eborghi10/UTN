/*******************************************************************************************************
*                                    PROGRAMA SERVIDOR VERSION 1
*
*  Al ejecutarlo se va a quedar esperando por la carga de la base de datos que se realiza mendiante TCP
*  empleando el programa BD. Una vez realizada la carga, se procede a esperar por conexiones entrantes.
*  Al ejecutar los clientes, se creará un child por conexion que se encargar de: Recibir, preprocesar y
*  enviar la respuesta al cliente, ademas de llevar el control del keep alive mediante un select.
*
********************************************************************************************************/
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <sys/ipc.h>
#include "opencv2/opencv.hpp"
#include "preprocessFace.h"
#include "recognition.h"




//*****************************************************
//***********          TCP          *******************
//*****************************************************
#define TCP 3490

fd_set read_select;
struct timeval tv;

char rec[4];

int TCPFD[3], numbytes, newfd[3], MAX_FD, SELVAL, salir;

struct sockaddr_in TCP_1[3];
struct sockaddr_in FD[3]; //their tpc address

unsigned int N_TCP;

socklen_t SOCK_SIZE;
//*****************************************************

void clean(int val);
void TCP_CONFIG(unsigned int);


//*****************************************************
//          Variables y flags
//*****************************************************

    //Matriz de personas
    char Person [10][20];
    //C++
    using namespace cv;
    using namespace std;

    const float UNKNOWN_PERSON_THRESHOLD = 0.7f;//Definimos el nivel de confiabilidad
    //Tamaño de rostro a detectar, es el recatangulo que vamos a obtener
    const int faceWidth = 70;
    const int faceHeight = faceWidth;

    //Constantes definidas para el algoritmo de deteccion de diferencias en los rostros tomados
    const double CHANGE_IN_IMAGE_FOR_COLLECTION = 0.3;
    const double CHANGE_IN_SECONDS_FOR_COLLECTION = 1.0;

    const char *windowName = "Deteccion Facial";   //Nombre de la ventana main
    const bool preprocessLeftAndRightSeparately = true;

    //Variables para el Preprocesamiento y el buscaparecidos !!


    Rect faceRect;  // posicion de la cara detectada
    Rect searchedLeftEye, searchedRightEye;
    Point leftEye, rightEye;    // Posicion de los ojos detectados
    //Las variables tipo Rect y Point son de posicion, Rect para dibujar el rectangulo (para el face) y Point para los
    //circulos.

    //Contiene el modelo entrenado
    Ptr<FaceRecognizer> model;
    //Contienen cantidad de personas y caras
    vector<Mat> preprocessedFaces; //Contendrá los rostros listos para generar el model, es decir, preprocesados
    vector<int> faceLabels, m_latestFaces;    //Para el agregado de personas nuevas a la base de datos
    int m_selectedPerson = -1, m_numPersons, n_person;

    bool gotFaceAndEyes = false; //Para saber cuando encontramos rostros
    Mat preprocessedFace; //Para alamcenar los frames preprocesados

    int identity = -1;
    int cont, persona= -1;
    //Importante para la deteccion, le damos el nivel de porcentaje de similitud para arrancar
    double porcentaje = 0.3;



//************************************************
//                 XMLS
//************************************************
    //Cargamos los XML a emplear en este programa, se pueden usar mas, tarda mas el procesamiento, pero es mas efectivo
    const char *faceCascadeFilename = "lbpcascade_frontalface.xml";     // LBP face detector.
    const char *eyeCascadeFilename1 = "haarcascade_eye.xml";               // Basic eye detector for open eyes only.
    const char *eyeCascadeFilename2 = "haarcascade_eye_tree_eyeglasses.xml"; // Basic eye detector for open eyes if they might wear glasses.
    //Variables empleadas para el load de los XML
    CascadeClassifier faceCascade;
    CascadeClassifier eyeCascade1;
    CascadeClassifier eyeCascade2;
//************************************************************//


//************************************************************//
//          Variables para la captura de frames
//************************************************************//
    //Aca es donde vamos a guardar el frame mat
    Mat  img = Mat::zeros( 480, 640, CV_8UC3);  //Esta si trabajamos con el frame completo, 480*640*3CH(RGB) = 96000 bytes
    int  imgSizes = img.total()*img.elemSize(); //96000 bytes
    VideoCapture cap; //Variable para guardar el frame obtenido de la camara
//*************************************


//**********************************************************************///
//******************      FUNCIONES         ****************************///
//**********************************************************************///


//************************************************
// C++ convierte de enteros a string
//************************************************
template <typename T> string toString(T t)
{
    ostringstream out;
    out << t;
    return out.str();
}



void initDetectors()
{

    //Carga con control de fallos!!!
    try {
    // Load Face Detection cascade classifier xml.
        faceCascade.load(faceCascadeFilename);
    } catch (cv::Exception &e) {}
    if ( faceCascade.empty() ) {
        cerr << "ERROR: No se pudo cargar el Face" << faceCascadeFilename << "]!" << endl;
        cerr << "Copia el XML que queres cargar en el directorio actual NABO !!!" << endl;
        exit(1);
    }
    cout << "Se cargo con exito la XML deteccion de face [" << faceCascadeFilename << "]." << endl;

    // Load Eye Detection cascade classifier xml.
    try {
        eyeCascade1.load(eyeCascadeFilename1);
    } catch (cv::Exception &e) {}
    if ( eyeCascade1.empty() ) {
        cerr << "ERROR: No se pudo cargar el primer ojo [" << eyeCascadeFilename1 << "]!" << endl;
        cerr << "Copia el XML que queres cargar en el directorio actual NABO !!!" << endl;
        exit(1);
    }
    cout << "Se cargo con exito la XML deteccion de 1er ojo [" << eyeCascadeFilename1 << "]." << endl;

    // Load Eye Detection cascade classifier xml.
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

//Hasta aca se cargaron los XML usados para el preentrenamiento para la deteccion de rostros
//Se pueden emplear mas que arriba se detallan, solo añadirlos, aumentamos el tiempo de procesamiento
//pero se mejora en aciertos !!!!
//******************************************************************************************


void preprocessed(){

        // Ejecute el sistema de reconocimiento de rostros en la imagen de la cámara.
        // Se basará algunas cosas sobre la imagen dada, así que asegúrese de que no está de memoria de sólo lectura!
        identity = -1;

        double old_time = 0;
        Mat old_prepreprocessedFace;

            // COMPROBAMOS SI SE DETECTO UNA CARA
            if (gotFaceAndEyes) {

                //*****************************************************************************************//
                //                 Algoritmo usado para evitar deteccion repetida de rostro
                //*****************************************************************************************//
                //Comprobamos si este rostro se ve un poco diferente de la cara recogido previamente.
                double imageDiff = 10000000000.0;

                //Veo si ya pasamos por lo menos una vez, lo que implica que old tenga dato
                if (old_prepreprocessedFace.data) {
                    imageDiff = getSimilarity(preprocessedFace, old_prepreprocessedFace);
                }

                // Para mejorar el preprocesamiento y deteccion de repetidos, tomamos los tiempos cuando ocurren las tomas de rostos.
                double current_time = (double)getTickCount();
                double timeDiff_seconds = (current_time - old_time)/getTickFrequency();


                // Sólo procesa la cara si es notablemente diferente de la anterior y si no ha habido espacio de tiempo notable.
                if ((imageDiff > CHANGE_IN_IMAGE_FOR_COLLECTION) && (timeDiff_seconds > CHANGE_IN_SECONDS_FOR_COLLECTION)) {

                    Mat mirroredFace;
                    //También agregamos la imagen espejo para el conjunto de entrenamiento,
                    //por lo que tenemos más datos de formación, así como para hacer frente a los rostros que miran a la izquierda o a la derecha.
                    flip(preprocessedFace, mirroredFace, 1); //me da la mitad de la cara

                    // Guardamos finalmente las imágenes de la cara en la lista de rostros detectados.
                    preprocessedFaces.push_back(preprocessedFace);
                    preprocessedFaces.push_back(mirroredFace);

                    faceLabels.push_back(m_selectedPerson);
                    faceLabels.push_back(m_selectedPerson);

                     //referencia a la última cara de cada persona.
                     m_latestFaces[m_selectedPerson] = preprocessedFaces.size() - 2;  //cara no reflejado.

                     cout << "Cara Guardada !!!! " << (preprocessedFaces.size()/2) << " DE PERSONA .." << m_selectedPerson << endl;

                    // Guardamos una copia de la cara procesada, para comparar en la próxima iteración.
                    old_prepreprocessedFace = preprocessedFace;
                    old_time = current_time;

                }//if cara distinta a la anterior


            } //Si detecto un rostro


}//preprocessed



void Buscaparecido(){

            //Chequeamos que haya algo en la base de datos
            if ((preprocessedFaces.size() > 0) && (preprocessedFaces.size() == faceLabels.size())) {

                Mat reconstructedFace;
                reconstructedFace = reconstructFace(model, preprocessedFace);

                double similarity = getSimilarity(preprocessedFace, reconstructedFace);

                string outputStr;

                //Comparamos segun sensivilidad preestablecida
                if (similarity < UNKNOWN_PERSON_THRESHOLD) {
                    // Identificamos de quien es el rostro procesado
                    identity = model->predict(preprocessedFace);
                    outputStr = toString(identity);
                    cout << "Identico: " << outputStr << ". Similar: " << similarity << endl;
                }
                else {
                    // Si obtenemos un nivel de confianza bajo, marcams como persona desconocida
                    outputStr = "Unknown";
                    cout << "\n Identico: " << outputStr<<endl;
                    //Mandamos el Nombre de la persona que identificamos o no
                    if(send(newfd[1], "Desconocido" , 20 , 0) == -1)
                    {   perror("send");
                        close(newfd[1]);
                        exit(0);
                    }
                }

                //Vemos cual es el mas parecido
               if(cont < 40 ){
                   if(similarity > porcentaje )
                  { persona  = identity;
                    porcentaje = similarity;
                  }
                  cont++;
                  }
               else{
                       outputStr = toString(persona);
                       printf("\n Sos igual a: %s \n", Person[persona]);
                    //Mandamos el Nombre de la persona que identificamos o no
                    if(send(newfd[1], (void*)Person[persona] , 20 , 0) == -1)
                    {   perror("send");
                        close(newfd[1]);
                        exit(0);
                    }

                cont=0;
                persona=-1;
                porcentaje=0.5;
            }



            }

}

void Trainning()
{

            bool haveEnoughData = true;

            if (m_numPersons < 1 || preprocessedFaces.size() <= 0 || preprocessedFaces.size() != faceLabels.size()) {
                cout << "Se necesita recolectar mas rostros !!!!" << endl;
                haveEnoughData = false;
            }


            if (haveEnoughData) {

                model = learnCollectedFaces(preprocessedFaces, faceLabels);

            }
}


/**********************************************************
**		        Funcion TCP_CONFIG
**********************************************************/
void TCP_CONFIG (unsigned int N_TCP)
{

	if ((TCPFD[N_TCP - TCP] = socket(AF_INET, SOCK_STREAM, 0)) == -1) /*Crea un socket y verifica si hubo algun error*/
    {
        perror("Error al ejecutar socket en el puerto TCP\n");
        exit(1);
    }

    TCP_1[N_TCP - TCP].sin_family = AF_INET;
    TCP_1[N_TCP - TCP].sin_port = htons(N_TCP);
    TCP_1[N_TCP - TCP].sin_addr.s_addr = INADDR_ANY;
    bzero(&(TCP_1[N_TCP - TCP].sin_zero), 8);
	if ( bind(TCPFD[N_TCP - TCP], (struct sockaddr *)&(TCP_1[N_TCP - TCP]),sizeof(struct sockaddr)) == -1)
    {
		perror("Error al ejecutar bind en el puerto TCP\n");
        exit(1);
    }
	if (listen(TCPFD[N_TCP - TCP], 1) == -1)
	{
		perror("listen");
		exit(1);
	}
}



void GetFrames( int fd){

        //Funcion mediante la cual obtenemos un frame de la camara y lo rearmamos en una variable tipo Mat segun los chanels del mismo

        int bytes, ptr;
        uchar sockData[imgSizes];

        //Se encarga de recibir y rearmar la  matriz de frame
        for (int i = 0; i < imgSizes; i += bytes) {
                if ((bytes = recv(fd, sockData + i , imgSizes  - i, 0)) == -1) {
                perror("recv");
                exit(1);
                }
            }

        ptr=0;
        //Rearmamos la matriz del frame (cada pixel por cada canal)
        for (int i = 0;  i < img.rows; i++) {
                for (int j = 0; j < img.cols; j++) {
                img.at<cv::Vec3b>(i,j) = cv::Vec3b(sockData[ptr+ 0],sockData[ptr+1],sockData[ptr+2]);
                ptr=ptr+3;
                }
        }


}




///*************************************************************************************************+
///*************************************************************************************************+
///                                          MAIN
///*************************************************************************************************+
///*************************************************************************************************+
int main(int argc, char *argv[])
{

//***********************************************
//		Señales trapeadas
//***********************************************
    signal(SIGINT, clean); //trapeo ctrl + c
//***********************************************



    TCP_CONFIG(TCP); //3490
    TCP_CONFIG(TCP+1); //3491

    SOCK_SIZE = sizeof(struct sockaddr_in);

    salir=1;
    char buff[20];




// Cargamos los XML classifiers. Esto es necesario, ya que en base a ellos vamos a
// realizar el preprocesamiento, y lograr asi detectar un rostro
    initDetectors();



///********************************************************************************
///*************     Programa que carga la BD        ******************************
///********************************************************************************
    cout << "\n\n Ejecute pregrama BD para generar la base de datos"<<endl;


   if ((newfd[0]=accept(TCPFD[0], (struct sockaddr *)&FD[0],&SOCK_SIZE)) == -1)
	  {	perror("accept_BD");
		exit(1);
	  }

    cout << "\n\n Conectado a Base de Datos.." << endl;
    cout << "\n Esperando por nueva informacion.." << endl;


while(salir){


    cout << "Clickea New para añadir mas personas "<< endl;

    //Ya tenemos una conexion en newfd
    //Vemos que tenemos que hacer, loguin o ingresar new person
    if ((recv(newfd[0], buff , sizeof(buff) , 0)) == -1) {
                perror("recv inicial");
                exit(1);
                }

    if(buff[0] == 1 ){

        if ((m_numPersons == 0) || (m_latestFaces[m_numPersons-1] >= 0)) {

// Agregar una nueva persona
            m_numPersons++;
            m_latestFaces.push_back(-1); // Allocate space for an extra person.
            cout << "Numero de personas en la base de datos: " << m_numPersons << endl;
        }


// Guardamos la cantidad de personas
        m_selectedPerson = m_numPersons - 1;

//En caso de malas detecciones aumentar el numero de rostros tomados para comparar
    for(int cant=0; cant < 40; cant++){

        //Tomamos frames
        GetFrames(newfd[0]);
        preprocessedFace = getPreprocessedFace(img, faceWidth, faceCascade, eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately, &faceRect, &leftEye, &rightEye, &searchedLeftEye, &searchedRightEye);
        gotFaceAndEyes = false;
        if (preprocessedFace.data)
            gotFaceAndEyes = true;
        else cant--;//No estoy detectando rostro
        //Preprocesamos y guardamos caras
        preprocessed();

    }

    //Ingresamos Nombre de la persona nueva

    cout<<"\n\n Ingrese SOLO su Nombre seguido de ENTER" <<endl;
    scanf("%s", Person[m_numPersons-1]);

    buff[0]=0;
    //Cerramos el child
   }


    else if(buff[0]==2){

    cout << "\n\n Cerrando Base de datos...."<<endl;
    close(newfd[0]);
    salir=0;
    cout << "\n\n Cerrado con exito...."<<endl;


    }

    else{
       }


}//while


///**************************************************************************************
///**************************************************************************************
    //Ya tenemos suficiente para entrenar
    //Generamos el model con la Base de Datos, para ello entrenamos con las faces tomadas
    Trainning();
    //Cerramos la conexion TCP, ya que BD no se volverá a ejecutar
    close(newfd[0]);

    //Cargamos el tiempo de espera del keep alive
    tv.tv_sec = 120;  // time out en 2 min
    tv.tv_usec = 0;





///*******************         SERVIDOR       *********************************///


while(1){


    if ((newfd[1]=accept(TCPFD[1], (struct sockaddr *)&FD[1],&SOCK_SIZE)) == -1)
	  {	perror("accept_Client");
		exit(1);
	  }


     cout << "\n\n Esperando por conexion de los clientes..."<<endl;


    //creo child pora cada cliente
    if(!fork())
    {
        cout<<"\n\n Cliente conectado"<<endl;
        cout <<"\n\n Soy Camara N°: "<<getpid()<<endl;
       // printf("\n Soy Hijo: %d", getpid());

        //Conectamos a TCP para uso del keep alive
       if ((newfd[0]=accept(TCPFD[0], (struct sockaddr *)&FD[0],&SOCK_SIZE)) == -1)
        {	perror("accept_Client");
            exit(1);
        }

        //Solo identificamos a la persona, si lo encontramos devolvemos el numero a la persona que se parece
        //sino devolvemos 0

        cout<<"\n Logueandose ..."<<endl;


        while(!salir){

        FD_ZERO(&read_select);
        for(int i=0; i<2 ; i++)
		FD_SET(newfd[i],&read_select);

		MAX_FD = newfd[0];

        for(int i=0; i<2 ; i++)
            if(MAX_FD<newfd[i])
                    MAX_FD = newfd[i];

		MAX_FD++;

        SELVAL = select(MAX_FD,&read_select,NULL,NULL,&tv);

        if(FD_ISSET(newfd[1], &read_select))
        {
        GetFrames(newfd[1]);
        preprocessedFace = getPreprocessedFace(img, faceWidth, faceCascade, eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately, &faceRect, &leftEye, &rightEye, &searchedLeftEye, &searchedRightEye);
        Buscaparecido();

       }//if

        if(FD_ISSET(newfd[0], &read_select))
        {

            if ((numbytes=recv(newfd[0], rec , sizeof(rec) , 0)) == -1) {
                perror("recv inicial");
                exit(1);
                }

            tv.tv_sec = 120; //2 min
            tv.tv_usec = 0;

        }


        if (!SELVAL){ //o fue por time out
	  //  cout <<"\nTime out ... muriendo\n"<<endl;
	    salir = 1;
	    }



}//While salir


    close(newfd[1]);
    close(newfd[0]);
  //  cout << "\n\n Cliente desconectado ..."<<endl;
    exit(0);

    }




}//Padre



    return 0;
}

///****************************************************************************************************///


//****************************************************
//		Señales trapeadas
//****************************************************

void clean(int val)
{
   //Trapeamos la señal ctr+c para eliminar todos los recursos empleados

   int i;

   //Aca TCP
   for(i=0;i<2;i++){
   close(TCPFD[i]);
   close(newfd[i]);
   }

   exit(1);

}



