/*******************************************************************************************************
*                                    PROGRAMA SERVIDOR VERSION 2
*
*  Al ejecutarlo en la inicializacion se atacha a la shared creada por DB, para usar de señalizacion. Luego, buscamos y
*  cargamos la base de datos YML generada con DB. Una vez realizada la carga, se procede a esperar por conexiones
*  entrantes.
*  Al ejecutar los clientes, se creará un child por conexion que se encargar de: Recibir, preprocesar y
*  enviar la respuesta al cliente, ademas de llevar el control del keep alive mediante un select.
*
********************************************************************************************************/
#include <stdlib.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <iostream>
#include <vector>
#include <string>
#include "opencv2/opencv.hpp"
#include <unistd.h>
#include <sys/stat.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/objdetect/objdetect.hpp>
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
//***********        SEM Y SHM         ****************
//*****************************************************
#define num_sem 1 //1 semaforo
#define SHM_SIZE 1024 //1k

//Shared
key_t key;
int shmid;
char *addr1;
//Semaforos
int semid;
struct sembuf psembuf, vsembuf;
short initArray[num_sem],outArray[num_sem];
//Structura que se encarga de señalizar y compartir los nombre en la carga de BD
typedef struct {
  int cantidad;
  char Nombres[20][20];
  int Actualizar;
  } areacompartida;

areacompartida *pAC;


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

//Para el agregado de personas nuevas a la base de datos
int m_selectedPerson = -1, m_numPersons = 0;

Rect faceRect;  // posicion de la cara detectada
Rect searchedLeftEye, searchedRightEye;
Point leftEye, rightEye;    // Posicion de los ojos detectados
//Las variables tipo Rect y Point son de posicion, Rect para dibujar el rectangulo (para el face) y Point para los
//circulos.

//Contiene el modelo entrenado
Ptr<FaceRecognizer> model, model2;
//Contienen cantidad de personas y caras
vector<Mat> preprocessedFaces;//Contendrá los rostros listos para generar el model, es decir, preprocesados
vector<int> faceLabels, m_latestFaces;    //Para el agregado de personas nuevas a la base de datos;

bool gotFaceAndEyes = false; //Para saber cuando encontramos rostros

Mat preprocessedFace; //Para alamcenar los frames preprocesados

int identity = -1;
int cont, persona= -1;
//Importante para la deteccion, le damos el nivel de porcentaje de similitud para arrancar
double porcentaje = 0.5;


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

void clean(int val);

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
//************************************************

/// Carga un rostro y 1 o 2 ojos XML para el entrenamiento
void initDetectors()
{
    // Load the Face Detection cascade classifier xml file.
    //Carga con control de fallos!!!
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

//Hasta aca se cargaron los XML usados para el preentrenamiento para la deteccion de rostros
//Se pueden emplear mas que arriba se detallan, solo añadirlos
//******************************************************************************************


void Buscaparecido(){
    //Chequeamos que haya un rostro detectado
    if(gotFaceAndEyes){

                Mat reconstructedFace;
                reconstructedFace = reconstructFace(model2, preprocessedFace);

                double similarity = getSimilarity(preprocessedFace, reconstructedFace);

                string outputStr;

                //Comparamos segun sensivilidad preestablecida
                if (similarity < UNKNOWN_PERSON_THRESHOLD) {
                    // Identificamos de quien es el rostro procesado
                    identity = model2->predict(preprocessedFace);
                    outputStr = toString(identity);
                    cout << "Identico: " << outputStr << ". Similar: " << similarity << endl;
                }
                else {
                    // Si obtenemos un nivel de confianza bajo, marcams como persona desconocida
                    outputStr = "Unknown";
                    cout << "\n Identico: " << outputStr<<endl;
                    //Mandamos el Nombre de la persona que identificamos o no
                    if(send(newfd[1], "Desconocido" , 20 , 0) == -1)
                    {   perror("send_desconocido");
                        close(newfd[1]);
                        exit(0);
                    }
                }

                //Vemos cual es el mas parecido
               if(cont < 50 ){
                   if(similarity > porcentaje )
                  { persona  = identity;
                    porcentaje = similarity;
                  }
                  cont++;
                  }
               else{
                    outputStr = toString(persona);
                    cout <<"\n Sos igual a: "<< pAC->Nombres[persona]<<endl;
                    //Mandamos el Nombre de la persona que identificamos o no
                    if(send(newfd[1], (void*)pAC->Nombres[persona] , 20 , 0) == -1)
                    {   perror("send_Nombre");
                        close(newfd[1]);
                        exit(0);
                    }

                cont=0;
                persona=-1;
                porcentaje=0.5;
            }//else
           }//Deteccion de rostro
}


//**********************************************************
//      		Funcion TCP_CONFIG
//**********************************************************
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




//*************************************************************************************************+
//*************************************************************************************************+
//                                        MAIN
//*************************************************************************************************+
//*************************************************************************************************+
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
    int  fds[2];


//***********************************************
//		        Shared memory
//***********************************************
    //Creo la shared
    if ((key = ftok("/home/pointbach/Escritorio/m.txt", 'o')) == -1) {
        perror("ftok");
        exit(1);
    }

    //Se conecta a la shared memory y la crea si no existe
    if ((shmid = shmget(key, 64*SHM_SIZE, 0644)) == -1) {
        cout << ("\n\n Base de Datos Vacia !!!! Ejecute Programa BD para inicializarla, y luego el Servidor")<<endl;
        exit(1);
    }


    //Me atacho a la memoria
    //Obtiene el puntero a memoria
    addr1 = (char *)shmat(shmid, (void *)0, 0);
    if (addr1 == (char *)(-1))
    {
        perror("shmat");
        exit(1);
    }

    //puntero al primer elemento del area compartida
    pAC = (areacompartida *) addr1;

//***********************************************


//***********************************************
//		            Semaforos
//***********************************************
    semid = semget(key, 1 , 0666 | IPC_CREAT);
    initArray[0] = 1;
    semctl(semid, 1 , SETALL, initArray);
    semctl(semid, 1 ,GETALL,outArray);
    //Preparar 2 strcuts para operar sobre el semaforo
    //semaforo del array
    psembuf.sem_num = 0; //primer semaforo del array
    psembuf.sem_op=-1; //la operacion "-1" toma el recurso
    psembuf.sem_flg=SEM_UNDO;
    vsembuf.sem_num = 0; //primer semaforo del array
    //la operacion "+1" libera el recurso
    vsembuf.sem_op = 1;
    vsembuf.sem_flg=SEM_UNDO;
//**********************************************


// Cargamos los XML classifiers. Esto es necesario, ya que en base a ellos vamos a
// realizar el preprocesamiento, y lograr asi detectar un rostro
    initDetectors();

    //Abrimos y cargamos la primera vez con la base de datos
 /*   if ((fds[0] =  open("/home/pointbach/Escritorio/test.yml", O_RDONLY, 00700))==-1){
                perror("No se encontró la base de datos");
                exit(0);
    }*/
    fds[0] =  open("/home/pointbach/Escritorio/test.yml", O_RDONLY, 00700);
    model2 = Algorithm::create<FaceRecognizer>("FaceRecognizer.Fisherfaces");
    model2->load("/home/pointbach/Escritorio/test.yml");
    close(fds[0]);

//**************************************************************************************
//**************************************************************************************


//Cargamos el tiempo de espera del keep alive
    tv.tv_sec = 120;  // time out en 2 min
    tv.tv_usec = 0;

    semop(semid, &vsembuf,1);


if(!fork()){

///*******************         SERVIDOR         *****************************///


while(1){

    cout << "\n\n Esperando por conexion de los clientes..."<<endl;

    if ((newfd[1]=accept(TCPFD[1], (struct sockaddr *)&FD[1],&SOCK_SIZE)) == -1)
	  {	perror("accept_Client");
		exit(1);
	  }


//********************************************************************************
//************          Rutina de los clientes              **********************
//********************************************************************************

    //creo child pora cada cliente

    if(!fork())
    {
        cout<<"\n\n Cliente conectado"<<endl;
        cout<<"\n Soy Cliente N° : "<< getpid()<<endl;
         //Conectamos a TCP para uso del keep alive
       if ((newfd[0]=accept(TCPFD[0], (struct sockaddr *)&FD[0],&SOCK_SIZE)) == -1)
        {	perror("accept_Client");
            exit(1);
        }

        //Solo identificamos a la persona, si lo encontramos devolvemos el numero a la persona que se parece
        //sino devolvemos 0
        cout<<"\n Logueandose ..."<<endl;
        //tomo frame actual
        while(1){
        GetFrames(newfd[1]);
        preprocessedFace = getPreprocessedFace(img, faceWidth, faceCascade, eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately, &faceRect, &leftEye, &rightEye, &searchedLeftEye, &searchedRightEye);
        gotFaceAndEyes = false;
        if (preprocessedFace.data)
            gotFaceAndEyes = true;

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
        //Zona critica
        semop(semid, &psembuf,1); //Tomo el recurso
        Buscaparecido();
        semop(semid, &vsembuf,1); //libero el recurso
        cout<<"\n pepe"<<endl;

       }//if

        if(FD_ISSET(newfd[0], &read_select))
        {

            if ((numbytes=recv(newfd[0], rec , sizeof(rec) , 0)) == -1) {
                perror("recv inicial");
                exit(1);
                }

            cout << "\n\n Recibimos"<<endl;

            tv.tv_sec = 120; //2 min
            tv.tv_usec = 0;

        }


        if (!SELVAL){ //o fue por time out
	  //  cout <<"\nTime out ... muriendo\n"<<endl;
	    salir = 1;
	    }




        //Zona critica
        semop(semid, &psembuf,1); //Tomo el recurso

        Buscaparecido();

        semop(semid, &vsembuf,1); //libero el recurso


        } //While salir

    close(newfd[1]);
    close(newfd[0]);
    exit(0);
    }//Fork
}

}



while(1){


    if(pAC->Actualizar==1){

    //Libero recursos para que puedan trabajar
	semop(semid, &psembuf,1);

    fds[0] =  open("/home/pointbach/Escritorio/test.yml", O_RDONLY, 00700);
    model2 = Algorithm::create<FaceRecognizer>("FaceRecognizer.Fisherfaces");
    model2->load("/home/pointbach/Escritorio/test.yml");
    close(fds[0]);

	usleep(2000);//le doy tiempo para resuelvan sus tareas

	cout << "\n\n Base de Datos Actualizada"<<endl;
    pAC->Actualizar=0;

    //Tomo nuevamente el recurso para que me interrumpan en la escritura
	semop(semid, &vsembuf,1);


    }


}

    return 0;
}




//****************************************************
//		Señales trapeadas
//****************************************************

void clean(int val)
{
   int i;
   shmctl(shmid, IPC_RMID, 0); //Destruye shared memory
   semctl(semid, 0, IPC_RMID); //Destruye semaforo
   shmdt(addr1);
   //Aca TCP
   for(i=0;i<2;i++){
   close(TCPFD[i]);
   close(newfd[i]);
   }
   exit(1);

}




