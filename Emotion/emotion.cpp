#include "emotion.h"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <fstream> 
#include <windows.h>
#include <opencv2/ml.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#include <dlib/image_io.h>
#include <dlib/image_transforms.h>
#include <direct.h>
#include "svm.h"
#include "svm.cpp"

#include <direct.h>
#define GetCurrentDir _getcwd

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))


using namespace dlib;
//using namespace cv;
using namespace cv::ml;
using namespace std;


Emotion::Emotion(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	QPixmap pix("Logo.jpg");
    ui.imm->setPixmap(pix);
	connect(ui.button1, SIGNAL(clicked()), this, SLOT(real_time()));
	connect(ui.button2, SIGNAL(clicked()), this, SLOT(registra()));
	connect(ui.button3, SIGNAL(clicked()), this, SLOT(carica_analizza()));
}

Emotion::~Emotion()
{

}

void mostra(cv::Mat original, cv::Mat risultati);
void mostra2(cv::Mat original,cv::Mat risultati,cv::String percentuale11,cv::String percentuale22,cv::String percentuale33,cv::String percentuale44,cv::String percentuale55,cv::String percentuale66,cv::String percentuale77,cv::String percentuale88,cv::String percentuale99,cv::String percentuale1010,cv::String percentuale00,float perc_entusiasmo,float perc_interesse,float perc_sorpresa,float perc_curiosita,float perc_concentrazione,float perc_attenzione,float perc_delusione,float perc_noia,float perc_perplessita,float perc_fastidio,float perc_frustrazione);


const std::string currentDateTime() {
    
	time_t theTime = time(NULL);
	struct tm *aTime = localtime(&theTime);

	int day = aTime->tm_mday;
	int month = aTime->tm_mon + 1; // Month is 0 – 11, add 1 to get a jan-dec 1-12 concept
	int year = aTime->tm_year + 1900; // Year is # years since 1900
	int hour=aTime->tm_hour;
	int min=aTime->tm_min;
	int sec=aTime->tm_sec;

	stringstream ss_year;
    ss_year << year;
    string x_year = ss_year.str();

	stringstream ss_month;
    ss_month << month;
    string x_month = ss_month.str();

	stringstream ss_day;
    ss_day << day;
    string x_day = ss_day.str();

	stringstream ss_hour;
    ss_hour << hour;
    string x_hour = ss_hour.str();

	stringstream ss_min;
    ss_min << min;
    string x_min = ss_min.str();

	stringstream ss_sec;
    ss_sec << sec;
    string x_sec = ss_sec.str();

	string timestamp=x_year+x_month+x_day+x_hour+x_min+x_sec;

    return timestamp;
}



int num_cifre(int numero){

	 long cifre=1,calcola=0;
	 bool finito=false;
	 calcola=numero;
	 while(!finito)
     {
        calcola/=10;
        if(calcola!=0)
            cifre++; 
        else
            finito=true; 
    }
	return cifre;
}


void scrittura(string filepath,string a1,string a2, string a3,string a4,string a5,string a6,string a7,string a8,string a9,string a10,string a11,string sg){
  FILE *fd;
  int x=-32;

  const char *ppp;
  
  /* apre il file in scrittura */
  ppp=filepath.c_str();
  fd=fopen(ppp, "a");
  if( fd==NULL ) {
    perror("Errore in apertura del file");
    exit(1);
  }

  fprintf(fd, "entusiasmo;interesse;sorpresa;curiosita';concentrazione;attenzione;delusione;noia;perplessita';fastidio;frustrazione;\n");
  fprintf(fd, "%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;\n",a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,sg);

		/* chiude il file */
  fclose(fd);
}


void scrittura2(string filepath,string a1,string a2, string a3,string a4,string a5,string a6,string a7,string a8,string a9,string a10,string a11,string sg){
  FILE *fd;
  int x=-32;

  const char *ppp;
  
  /* apre il file in scrittura */
  ppp=filepath.c_str();
  fd=fopen(ppp, "a");
  if( fd==NULL ) {
    perror("Errore in apertura del file");
    exit(1);
  }

  string time=currentDateTime();


  fprintf(fd, "entusiasmo;interesse;sorpresa;curiosita';concentrazione;attenzione;delusione;noia;perplessita';fastidio;frustrazione;\n");
  fprintf(fd, "%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;\n",a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,sg,time);

		/* chiude il file */
  fclose(fd);
}



std::string ReplaceAll(std::string str, const std::string& from, const std::string& to) {
	size_t start_pos = 0;
	while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
		str.replace(start_pos, from.length(), to);
		start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
	}
	return str;
}



int stampainfo(string per){

	const char *pp;
	pp=per.c_str();
	CvCapture  *capture = cvCaptureFromAVI(pp);
	cvQueryFrame( capture );
	int nFrames = (int) cvGetCaptureProperty( capture , CV_CAP_PROP_FRAME_COUNT );
	return nFrames;
}


std::string GetCurrentWorkingDir( void ) {
  char buff[FILENAME_MAX];
  GetCurrentDir( buff, FILENAME_MAX );
  std::string current_working_dir(buff);
  return current_working_dir;
}
 



std::vector<cv::Point3d> get_3d_model_points()
{
    std::vector<cv::Point3d> modelPoints;

    modelPoints.push_back(cv::Point3d(0.0f, 0.0f, 0.0f)); //The first must be (0,0,0) while using POSIT
    modelPoints.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));
    modelPoints.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));
    modelPoints.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));
    modelPoints.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));
    modelPoints.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));
    
    return modelPoints;
    
}

std::vector<cv::Point2d> get_2d_image_points(full_object_detection &d)
{
    std::vector<cv::Point2d> image_points;
    image_points.push_back( cv::Point2d( d.part(30).x(), d.part(30).y() ) );    // Nose tip
    image_points.push_back( cv::Point2d( d.part(8).x(), d.part(8).y() ) );      // Chin
    image_points.push_back( cv::Point2d( d.part(36).x(), d.part(36).y() ) );    // Left eye left corner
    image_points.push_back( cv::Point2d( d.part(45).x(), d.part(45).y() ) );    // Right eye right corner
    image_points.push_back( cv::Point2d( d.part(48).x(), d.part(48).y() ) );    // Left Mouth corner
    image_points.push_back( cv::Point2d( d.part(54).x(), d.part(54).y() ) );    // Right mouth corner
    return image_points;

}

cv::Mat get_camera_matrix(float focal_length, cv::Point2d center)
{
    cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
    return camera_matrix;
}




void Emotion::real_time(){

	int webcam=0;

	if(ui.comboBox->currentText()=="Webcam Interna"){
		webcam=0;
	}else if(ui.comboBox->currentText()=="Webcam Esterna"){
		webcam=1;
	}

	std::string v=GetCurrentWorkingDir();
	string cc=v+"/secondary_emotions_multiClassSVM.model";
	const char *cstr = cc.c_str();
	svm_model *model= svm_load_model(cstr);

	double predictions;
        double pred;
        double *prob_est=new double[11];   // 11 è il numero di classi
		
			
		std::vector<cv::Point2d> nose_end_point2D;
		std::vector<cv::Point2d> image_points;

		std::string video=salvaFile();
        std::string analisi=salvaRisultati();
	
   //VideoCapture capture;

    // Get a handle to the Video device:
        cv::VideoCapture cap(webcam);
	    
        // Check if we can use this device at all:
        if(!cap.isOpened()) {
            cerr << "Capture Device ID 0 cannot be opened." << endl;
        }


				//namedWindow("MyVideo",CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"

   double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
   double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

   int centro=dWidth/2;

   //cout << "Frame Size = " << dWidth << "x" << dHeight << endl;

   cv::Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));

   cv::VideoWriter oVideoWriter (video, CV_FOURCC('D','I','V','X'), 25, frameSize, true); //initialize the VideoWriter object 

        
        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
		string percorso=v+"/shape_predictor_68_face_landmarks.dat";
        deserialize(percorso) >> pose_model;


		cv::Mat original, img_gray;

	 cv::Mat risultati(dHeight, 400, CV_8UC3);

	 cv::Point pt11;
	 cv::Point pt22;
	 pt11.x = 0;
     pt11.y = dHeight;
     pt22.x = 400;
     pt22.y = 0;
	 cv::rectangle(risultati, pt11, pt22, cv::Scalar(78, 57, 0), -1);

	 cv::Point pt111;
	 cv::Point pt222;
	 pt111.x = 20;
     pt111.y = 440;
     pt222.x = 380;
     pt222.y = 90;

	 cv::rectangle(risultati, pt111, pt222, cv::Scalar(210, 210, 210), -1);
	 
	 cv::Mat logo=cv::imread("Logo2.jpg");
	 
    for(;;){
        // Start time
        //gettimeofday(&start, NULL);

        cap >> original;

		//imwrite("C:/Users/Giuseppe/Desktop/1.jpg",original);


		oVideoWriter.write(original); //writer the frame into the file

        //imshow("MyVideo", frame); //show the frame in "MyVideo" window

		logo.copyTo(risultati(cv::Rect(288,13,logo.cols, logo.rows)));

		cv_image<bgr_pixel> cimg(original);

		std::vector<cv::Point3d> model_points = get_3d_model_points();

            // Detect faces 
            std::vector<rectangle> faces = detector(cimg);
            // Find the pose of each face.
            std::vector<full_object_detection> shapes;
			full_object_detection shape;
            for (unsigned long i = 0; i < faces.size(); ++i)
            {  

               shape = pose_model(cimg, faces[i]);

               //cout << "number of parts: "<< shape.num_parts() << endl;
               //cout << "pixel position of first part:  " << shape.part(0) << endl;
               //cout << "pixel position of second part: " << shape.part(1) << endl;
               shapes.push_back(pose_model(cimg, faces[i]));

			   //cout<< shape.part(0).x() << " " << shape.part(0).y() << endl;
               //cout<< shape.part(1).x() << " " << shape.part(1).y() << endl;

			   image_points = get_2d_image_points(shape);
               double focal_length = original.cols;
               cv::Mat camera_matrix = get_camera_matrix(focal_length, cv::Point2d(original.cols/2,original.rows/2));
               cv::Mat rotation_vector;
               cv::Mat rotation_matrix;
               cv::Mat translation_vector;

			   cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type);
                
               cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

               
			   std::vector<cv::Point3d> nose_end_point3D;
	  		   nose_end_point3D.push_back(cv::Point3d(0,0,1000.0));

			   cv::projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, nose_end_point2D);		
              


              }
       
		
		if(shapes.empty()==false){
			
			int x1=shape.part(0).x();
			int y1=shape.part(0).y();

			int x9=shape.part(8).x();
			int y9=shape.part(8).y();

			int x17=shape.part(16).x();
			int y17=shape.part(16).y();

			int x20=shape.part(19).x();
			int y20=shape.part(19).y();

			int x25=shape.part(24).x();
			int y25=shape.part(24).y();


			int punto_partenza_x=x1-15;

  int a[2];
  a[0]=y20;
  a[1]=y25;

  int minimo=*std::min_element(a,a+2);
  
  int punto_partenza_y=minimo-15;

  int punto_altezza_x=punto_partenza_x;
  int punto_altezza_y=y9+15;

  int altezza=punto_altezza_y-punto_partenza_y;

  int punto_larghezza_x=x17+15;
  int punto_larghezza_y=punto_partenza_y;

  int larghezza=punto_larghezza_x-punto_partenza_x;


  cv::Rect r(punto_partenza_x,punto_partenza_y,larghezza,altezza);


  if(punto_altezza_y>dHeight || punto_partenza_x<0 || punto_larghezza_x>dWidth || y20-15<0 || y25-15<0){

	  mostra(original, risultati);
		

  }else{

 
    //imwrite("C:/Users/Giuseppe/Desktop/1.immagine_originale.jpg",frame);
     cv::Mat faccia = original(r);
	 //imwrite("C:/Users/Giuseppe/Desktop/2.jpg",faccia);
	 
	 cv::Mat vis=faccia.clone();
	 cv::resize(vis, vis, cv::Size(65,65) );
	 vis.copyTo(risultati(cv::Rect(50,13,vis.cols, vis.rows)));
		    
	 cv::Rect rectROI(0,0,faccia.cols,faccia.rows);
	cv::Mat mask(faccia.rows, faccia.cols, CV_8UC1, cv::Scalar(0));

		/***********************************************/

	cv_image<bgr_pixel> cimg2(faccia);

            // Detect faces 
            std::vector<rectangle> faces2 = detector(cimg2);
            // Find the pose of each face.
            std::vector<full_object_detection> shapes2;
			full_object_detection shape2;
            for (unsigned long i = 0; i < faces2.size(); ++i)
            {  

               shape2 = pose_model(cimg2, faces2[i]);

               //cout << "number of parts: "<< shape.num_parts() << endl;
               //cout << "pixel position of first part:  " << shape.part(0) << endl;
               //cout << "pixel position of second part: " << shape.part(1) << endl;
               shapes2.push_back(pose_model(cimg2, faces2[i]));

			   //cout<< shape.part(0).x() << " " << shape.part(0).y() << endl;
               //cout<< shape.part(1).x() << " " << shape.part(1).y() << endl;

              }

		if(shapes2.empty()==false){
		
			int xx1=shape2.part(0).x();
			int yy1=shape2.part(0).y();

			int xx2=shape2.part(1).x();
			int yy2=shape2.part(1).y();

			int xx3=shape2.part(2).x();
			int yy3=shape2.part(2).y();

			int xx4=shape2.part(3).x();
			int yy4=shape2.part(3).y();

			int xx5=shape2.part(4).x();
			int yy5=shape2.part(4).y();

			int xx6=shape2.part(5).x();
			int yy6=shape2.part(5).y();

			int xx7=shape2.part(6).x();
			int yy7=shape2.part(6).y();

			int xx8=shape2.part(7).x();
			int yy8=shape2.part(7).y();

			int xx9=shape2.part(8).x();
			int yy9=shape2.part(8).y();

			int xx10=shape2.part(9).x();
			int yy10=shape2.part(9).y();

			int xx11=shape2.part(10).x();
			int yy11=shape2.part(10).y();

			int xx12=shape2.part(11).x();
			int yy12=shape2.part(11).y();

			int xx13=shape2.part(12).x();
			int yy13=shape2.part(12).y();

			int xx14=shape2.part(13).x();
			int yy14=shape2.part(13).y();

			int xx15=shape2.part(14).x();
			int yy15=shape2.part(14).y();

			int xx16=shape2.part(15).x();
			int yy16=shape2.part(15).y();

			int xx17=shape2.part(16).x();
			int yy17=shape2.part(16).y();

			int xx18=shape2.part(17).x();
			int yy18=shape2.part(17).y();

			int xx19=shape2.part(18).x();
			int yy19=shape2.part(18).y();

			int xx20=shape2.part(19).x();
			int yy20=shape2.part(19).y();

			int xx21=shape2.part(20).x();
			int yy21=shape2.part(20).y();

			int xx22=shape2.part(21).x();
			int yy22=shape2.part(21).y();

			int xx23=shape2.part(22).x();
			int yy23=shape2.part(22).y();

			int xx24=shape2.part(23).x();
			int yy24=shape2.part(23).y();

			int xx25=shape2.part(24).x();
			int yy25=shape2.part(24).y();

			int xx26=shape2.part(25).x();
			int yy26=shape2.part(25).y();

			int xx27=shape2.part(26).x();
			int yy27=shape2.part(26).y();
			

			cv::Point P1(xx1,yy1);
			cv::Point P2(xx2,yy2);
			cv::Point P3(xx3,yy3);
			cv::Point P4(xx4,yy4);
			cv::Point P5(xx5,yy5);
			cv::Point P6(xx6,yy6);
			cv::Point P7(xx7,yy7);
			cv::Point P8(xx8,yy8);
			cv::Point P9(xx9,yy9);
			cv::Point P10(xx10,yy10);
			cv::Point P11(xx11,yy11);
			cv::Point P12(xx12,yy12);
			cv::Point P13(xx13,yy13);
			cv::Point P14(xx14,yy14);
			cv::Point P15(xx15,yy15);
			cv::Point P16(xx16,yy16);
			cv::Point P17(xx17,yy17);
			cv::Point P18(xx18,yy18);
			cv::Point P19(xx19,yy19);
			cv::Point P20(xx20,yy20);
			cv::Point P21(xx21,yy21);
			cv::Point P22(xx22,yy22);
			cv::Point P23(xx23,yy23);
			cv::Point P24(xx24,yy24);
			cv::Point P25(xx25,yy25);
			cv::Point P26(xx26,yy26);
			cv::Point P27(xx27,yy27);

std::vector< std::vector<cv::Point> >  co_ordinates;
   co_ordinates.push_back(std::vector<cv::Point>());
   co_ordinates[0].push_back(P1);
   co_ordinates[0].push_back(P2);
   co_ordinates[0].push_back(P3);
   co_ordinates[0].push_back(P4);
   co_ordinates[0].push_back(P5);
   co_ordinates[0].push_back(P6);
   co_ordinates[0].push_back(P7);
   co_ordinates[0].push_back(P8);
   co_ordinates[0].push_back(P9);
   co_ordinates[0].push_back(P10);
   co_ordinates[0].push_back(P11);
   co_ordinates[0].push_back(P12);
   co_ordinates[0].push_back(P13);
   co_ordinates[0].push_back(P14);
   co_ordinates[0].push_back(P15);
   co_ordinates[0].push_back(P16);
   co_ordinates[0].push_back(P17);
   co_ordinates[0].push_back(P27);
   co_ordinates[0].push_back(P26);
   co_ordinates[0].push_back(P25);
   co_ordinates[0].push_back(P24);
   co_ordinates[0].push_back(P23);
   co_ordinates[0].push_back(P22);
   co_ordinates[0].push_back(P21);
   co_ordinates[0].push_back(P20);
   co_ordinates[0].push_back(P19);
   co_ordinates[0].push_back(P18);
   drawContours( mask,co_ordinates,0, cv::Scalar(255),CV_FILLED, 8 );

   cv::Mat srcROI=faccia(rectROI);
   cv::Mat dst1;

   srcROI.copyTo(dst1,mask);

   cvtColor(dst1, img_gray, CV_RGB2GRAY);
   //imwrite("C:/Users/Giuseppe/Desktop/3.jpg",img_gray);

   
   cv::Mat dst11=dst1.clone();
   cv::Rect rois(0,0,dst11.cols,dst11.rows);

   cv::Mat crop = dst11(rois).clone();      // Crop is color CV_8UC3
        cvtColor(crop, crop, cv::COLOR_BGR2GRAY); // Now crop is grayscale CV_8UC1
        cvtColor(crop, crop, cv::COLOR_GRAY2BGR); // Now crop is grayscale, CV_8UC3
        crop.copyTo(dst11(rois));

    cv::resize(dst11, dst11, cv::Size(65,65) );
    dst11.copyTo(risultati(cv::Rect(135,13,dst11.cols, dst11.rows)));

     cv::resize(img_gray, img_gray, cv::Size(48,64) );

	 //imwrite("C:/Users/Giuseppe/Desktop/4.jpg",img_gray);



	 
	
        //extract feature
        cv::HOGDescriptor d( cv::Size(48,64), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 7);
        std::vector<float> descriptorsValues;
        std::vector<cv::Point> locations;
        d.compute( img_gray, descriptorsValues, cv::Size(0,0), cv::Size(0,0), locations);

	
        //2d vector to Mat
        int row=descriptorsValues.size();
        int col=descriptorsValues.size();
        printf("descript size row=%d, col=%d\n", row, col);

        printf("descript element %f\n", descriptorsValues[12]);


        cv::Mat fm = cv::Mat(descriptorsValues);

        cv::Mat B = fm.t();

		float **mat = new float*[B.rows];
        for(int i=0;i<B.rows;i++){
           mat[i]=new float[B.cols];
        }
		
 
		for(int i=0;i<B.rows;i++){
			for(int j=0;j<B.cols;j++){
				mat[i][j] = B.at<float>(i,j);         
			}
		}

		

		svm_node* testnode = Malloc(svm_node,B.cols+1);
		for (int row=0;row <B.rows; row++){
			for(int col=0;col<B.cols;col++){
				testnode[col].index = col;
				testnode[col].value = mat[row][col];
			}
        testnode[B.cols].index = -1;
		}



		predictions = svm_predict_probability(model, testnode, prob_est);
         printf("\nprediction    %f\n", predictions);
		 printf("\nentusiasmo      %f\n", prob_est[0]);
		 printf("interesse       %f\n", prob_est[1]);
		 printf("sorpresa        %f\n", prob_est[2]);
		 printf("curiosita'      %f\n", prob_est[3]);
		 printf("concentrazione  %f\n", prob_est[4]);
		 printf("attenzione      %f\n", prob_est[5]);
		 printf("delusione       %f\n", prob_est[6]);
		 printf("noia            %f\n", prob_est[7]);
		 printf("perplessita'    %f\n", prob_est[8]);
		 printf("fastidio        %f\n", prob_est[9]);
		 printf("frustrazione    %f\n", prob_est[10]);
		 
		 
        
	

	 pred = svm_predict(model, testnode);
     printf("%f\n", pred);

	 float entusiasmo=prob_est[0]*100;
	 float interesse=prob_est[1]*100;
	 float sorpresa=prob_est[2]*100;
	 float curiosita=prob_est[3]*100;
	 float concentrazione=prob_est[4]*100;
	 float attenzione=prob_est[5]*100;
	 float delusione=prob_est[6]*100;
	 float noia=prob_est[7]*100;
	 float perplessita=prob_est[8]*100;
	 float fastidio=prob_est[9]*100;
	 float frustrazione=prob_est[10]*100;

	 cout << fixed << setprecision(2) << entusiasmo;
	 cout << fixed << setprecision(2) << interesse;
	 cout << fixed << setprecision(2) << sorpresa;
	 cout << fixed << setprecision(2) << curiosita;
	 cout << fixed << setprecision(2) << concentrazione;
	 cout << fixed << setprecision(2) << attenzione;
	 cout << fixed << setprecision(2) << delusione;
	 cout << fixed << setprecision(2) << noia;
	 cout << fixed << setprecision(2) << perplessita;
	 cout << fixed << setprecision(2) << fastidio;
	 cout << fixed << setprecision(2) << frustrazione;

	 cout<<endl<<"percentuale completa: "<<entusiasmo+interesse+sorpresa+curiosita+concentrazione+attenzione+delusione+noia+perplessita+fastidio+frustrazione<<endl;

	 float entu=entusiasmo;
	 int num1=(int) entu;
	 std::string percentuale1;
     std::stringstream oss1;
	 cv::String percentuale11;
	 int num11=num_cifre(num1);
	 if(num11==1){
		 oss1 << std::setprecision(3) << entu;
		 percentuale1=oss1.str();
	     percentuale11=" "+percentuale1+"%";
	 }else{
     oss1 << std::setprecision(4) << entu;
     percentuale1=oss1.str();
	 percentuale11=percentuale1+"%";
	 }
     float perc_entusiasmo=(entu*180)/100;
	 

	 float inter=interesse;
	 int num2=(int) inter;
	 std::string percentuale2;
     std::stringstream oss2;
	 cv::String percentuale22;
	 int num22=num_cifre(num2);
	 if(num22==1){
		 oss2 << std::setprecision(3) << inter;
		 percentuale2=oss2.str();
	     percentuale22=" "+percentuale2+"%";
	 }else{
     oss2 << std::setprecision(4) << inter;
     percentuale2=oss2.str();
	 percentuale22=percentuale2+"%";
	 }
     float perc_interesse=(inter*180)/100;


	 float sorp=sorpresa;
	 int num3=(int) sorp;
	 std::string percentuale3;
     std::stringstream oss3;
	 cv::String percentuale33;
	 int num33=num_cifre(num3);
	 if(num33==1){
		 oss3 << std::setprecision(3) << sorp;
		 percentuale3=oss3.str();
	     percentuale33=" "+percentuale3+"%";
	 }else{
     oss3 << std::setprecision(4) << sorp;
     percentuale3=oss3.str();
	 percentuale33=percentuale3+"%";
	 }
     float perc_sorpresa=(sorp*180)/100;


	 float curio=curiosita;
	 int num4=(int) curio;
	 std::string percentuale4;
     std::stringstream oss4;
	 cv::String percentuale44;
	 int num44=num_cifre(num4);
	 if(num44==1){
		 oss4 << std::setprecision(3) << curio;
		 percentuale4=oss4.str();
	     percentuale44=" "+percentuale4+"%";
	 }else{
     oss4 << std::setprecision(4) << curio;
     percentuale4=oss4.str();
	 percentuale44=percentuale4+"%";
	 }
     float perc_curiosita=(curio*180)/100;


	 float concen=concentrazione;
	 int num5=(int) concen;
	 std::string percentuale5;
     std::stringstream oss5;
	 cv::String percentuale55;
	 int num55=num_cifre(num5);
	 if(num55==1){
		 oss5 << std::setprecision(3) << concen;
		 percentuale5=oss5.str();
	     percentuale55=" "+percentuale5+"%";
	 }else{
     oss5 << std::setprecision(4) << concen;
     percentuale5=oss5.str();
	 percentuale55=percentuale5+"%";
	 }
     float perc_concentrazione=(concen*180)/100;


	 float atten=attenzione;
	 int num6=(int) atten;
	 std::string percentuale6;
     std::stringstream oss6;
	 cv::String percentuale66;
	 int num66=num_cifre(num6);
	 if(num66==1){
		 oss6 << std::setprecision(3) << atten;
		 percentuale6=oss6.str();
	     percentuale66=" "+percentuale6+"%";
	 }else{
     oss6 << std::setprecision(4) << atten;
     percentuale6=oss6.str();
	 percentuale66=percentuale6+"%";
	 }
     float perc_attenzione=(atten*180)/100;


	 float delus=delusione;
	 int num7=(int) delus;
	 std::string percentuale7;
     std::stringstream oss7;
	 cv::String percentuale77;
	 int num77=num_cifre(num7);
	 if(num77==1){
		 oss7 << std::setprecision(3) << delus;
		 percentuale7=oss7.str();
	     percentuale77=" "+percentuale7+"%";
	 }else{
     oss7 << std::setprecision(4) << delus;
     percentuale7=oss7.str();
	 percentuale77=percentuale7+"%";
	 }
     float perc_delusione=(delus*180)/100;


	 float noi=noia;
	 int num8=(int) noi;
	 std::string percentuale8;
     std::stringstream oss8;
	 cv::String percentuale88;
	 int num88=num_cifre(num8);
	 if(num88==1){
		 oss8 << std::setprecision(3) << noi;
		 percentuale8=oss8.str();
	     percentuale88=" "+percentuale8+"%";
	 }else{
     oss8 << std::setprecision(4) << noi;
     percentuale8=oss8.str();
	 percentuale88=percentuale8+"%";
	 }
     float perc_noia=(noi*180)/100;


	 float perpl=perplessita;
	 int num9=(int) perpl;
	 std::string percentuale9;
     std::stringstream oss9;
	 cv::String percentuale99;
	 int num99=num_cifre(num9);
	 if(num99==1){
		 oss9 << std::setprecision(3) << perpl;
		 percentuale9=oss9.str();
	     percentuale99=" "+percentuale9+"%";
	 }else{
     oss9 << std::setprecision(4) << perpl;
     percentuale9=oss9.str();
	 percentuale99=percentuale9+"%";
	 }
     float perc_perplessita=(perpl*180)/100;


	 float fast=fastidio;
	 int num10=(int) fast;
	 std::string percentuale10;
     std::stringstream oss10;
	 cv::String percentuale1010;
	 int num1010=num_cifre(num10);
	 if(num1010==1){
		 oss10 << std::setprecision(3) << fast;
		 percentuale10=oss10.str();
	     percentuale1010=" "+percentuale10+"%";
	 }else{
     oss10 << std::setprecision(4) << fast;
     percentuale10=oss10.str();
	 percentuale1010=percentuale10+"%";
	 }
     float perc_fastidio=(fast*180)/100;


	 float frustr=frustrazione;
	 int num0=(int) frustr;
	 std::string percentuale0;
     std::stringstream oss0;
	 cv::String percentuale00;
	 int num00=num_cifre(num0);
	 if(num00==1){
		 oss0 << std::setprecision(3) << frustr;
		 percentuale0=oss0.str();
	     percentuale00=" "+percentuale0+"%";
	 }else{
     oss0 << std::setprecision(4) << frustr;
     percentuale0=oss0.str();
	 percentuale00=percentuale0+"%";
	 }
     float perc_frustrazione=(frustr*180)/100;


		
	    ////////////////////////
		printf("\n\n");

		// Show the result:
		circle(original, cv::Point(shape.part(0).x(),shape.part(0).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(1).x(),shape.part(1).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(2).x(),shape.part(2).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(3).x(),shape.part(3).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(4).x(),shape.part(4).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(5).x(),shape.part(5).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(6).x(),shape.part(6).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(7).x(),shape.part(7).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(8).x(),shape.part(8).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(9).x(),shape.part(9).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(10).x(),shape.part(10).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(11).x(),shape.part(11).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(12).x(),shape.part(12).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(13).x(),shape.part(13).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(14).x(),shape.part(14).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(15).x(),shape.part(15).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(16).x(),shape.part(16).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(17).x(),shape.part(17).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(18).x(),shape.part(18).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(19).x(),shape.part(19).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(20).x(),shape.part(20).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(21).x(),shape.part(21).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(22).x(),shape.part(22).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(23).x(),shape.part(23).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(24).x(),shape.part(24).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(25).x(),shape.part(25).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(26).x(),shape.part(26).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(27).x(),shape.part(27).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(28).x(),shape.part(28).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(29).x(),shape.part(29).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(30).x(),shape.part(30).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(31).x(),shape.part(31).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(32).x(),shape.part(32).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(33).x(),shape.part(33).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(34).x(),shape.part(34).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(35).x(),shape.part(35).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(36).x(),shape.part(36).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(37).x(),shape.part(37).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(38).x(),shape.part(38).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(39).x(),shape.part(39).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(40).x(),shape.part(40).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(41).x(),shape.part(41).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(42).x(),shape.part(42).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(43).x(),shape.part(43).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(44).x(),shape.part(44).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(45).x(),shape.part(45).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(46).x(),shape.part(46).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(47).x(),shape.part(47).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(48).x(),shape.part(48).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(49).x(),shape.part(49).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(50).x(),shape.part(50).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(51).x(),shape.part(51).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(52).x(),shape.part(52).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(53).x(),shape.part(53).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(54).x(),shape.part(54).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(55).x(),shape.part(55).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(56).x(),shape.part(56).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(57).x(),shape.part(57).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(58).x(),shape.part(58).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(59).x(),shape.part(59).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(60).x(),shape.part(60).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(61).x(),shape.part(61).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(62).x(),shape.part(62).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(63).x(),shape.part(63).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(64).x(),shape.part(64).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(65).x(),shape.part(65).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(66).x(),shape.part(66).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);
		circle(original, cv::Point(shape.part(67).x(),shape.part(67).y()),1, cv::Scalar(0,255,0),CV_FILLED, 8,0);



		cv::rectangle(original, r, cv::Scalar(0,255,0), 1);


		  for(int i=0; i < image_points.size(); i++)
		  {
				circle(original, image_points[i], 3, cv::Scalar(0,0,255), -1);
		  }

		  string sguardo;
		  string sg;

		cv::line(original,image_points[0], nose_end_point2D[0], cv::Scalar(255,0,0), 2);
           
                cout <<  nose_end_point2D[0];

				cout <<" "<<nose_end_point2D[0].x;

				if(nose_end_point2D[0].x >= centro){
					sg="sinistra";
					sguardo= "Sguardo: sinistra";
				}else{
					sg="destra";
					sguardo= "Sguardo: destra";
				}

		cout << " " << sguardo << endl;

		putText(original, "Premi ESC per chiudere", cv::Point(10, 30), cv::FONT_HERSHEY_PLAIN, 1.1, CV_RGB(0,255,0), 1);

		putText(original, sguardo, cv::Point(400, 80), cv::FONT_HERSHEY_PLAIN, 1.3, CV_RGB(0,255,0), 2);

		scrittura2(analisi,percentuale11,percentuale22,percentuale33,percentuale44,percentuale55,percentuale66,percentuale77,percentuale88,percentuale99,percentuale1010,percentuale00,sg);
	

		mostra2(original,risultati,percentuale11,percentuale22,percentuale33,percentuale44,percentuale55,percentuale66,percentuale77,percentuale88,percentuale99,percentuale1010,percentuale00,perc_entusiasmo,perc_interesse,perc_sorpresa,perc_curiosita,perc_concentrazione,perc_attenzione,perc_delusione,perc_noia,perc_perplessita,perc_fastidio,perc_frustrazione);

		
		}
        }

		}else{

		mostra(original,risultati);

		}
		
		
		   char k = cvWaitKey(20);
        if(k==27) break;
    }


}

void Emotion::registra(){

	int webcam=0;

	if(ui.comboBox->currentText()=="Webcam Interna"){
		webcam=0;
	}else if(ui.comboBox->currentText()=="Webcam Esterna"){
		webcam=1;
	}
	
	
	std::string percorso=salvaFile();

    cv::VideoCapture cap(webcam); // open the video camera no. 0

    if (!cap.isOpened())  // if not success, exit program
    {
        cout << "ERROR: Cannot open the video file" << endl;
    }

   cv::namedWindow("Registra",CV_WINDOW_AUTOSIZE); //create a window called "Video"

   double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
   double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

   cout << "Frame Size = " << dWidth << "x" << dHeight << endl;

   cv::Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));

   cv::VideoWriter oVideoWriter (percorso, CV_FOURCC('D','I','V','X'), 25, frameSize, true); //initialize the VideoWriter object 

   if ( !oVideoWriter.isOpened() ) //if not initialize the VideoWriter successfully, exit the program
   {
        cout << "ERROR: Failed to write the video" << endl;
   }

   cv::Mat logo=cv::imread("Logo2.jpg");

   int x=dWidth-93;

    while (1)
    {

        cv::Mat frame;

        bool bSuccess = cap.read(frame); // read a new frame from video

        if (!bSuccess) //if not success, break loop
       {
             cout << "ERROR: Cannot read a frame from video file" << endl;
             break;
        }

         oVideoWriter.write(frame); //writer the frame into the file

		 logo.copyTo(frame(cv::Rect(x,0,logo.cols, logo.rows)));

		 putText(frame, "Registrazione video in corso...", cv::Point(10, 30), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 1.5);

		 putText(frame, "Premi ESC per chiudere", cv::Point(10, 60), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 1.5);

        imshow("Registra", frame); //show the frame in "MyVideo" window

        if (cv::waitKey(10) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
       {
            cout << "esc key is pressed by user" << endl;
            cvDestroyWindow("Registra");
			break;
       }
    }

}







void Emotion::carica_analizza(){

	

	std::string v=GetCurrentWorkingDir();
		std::string cc=v+"/secondary_emotions_multiClassSVM.model";
		const char *cstr = cc.c_str();
		svm_model *model= svm_load_model(cstr);



		double predictions;
        //double pred;
        double *prob_est=new double[11];   // 11 è il numero di classi

		std::vector<cv::Point2d> nose_end_point2D;
		std::vector<cv::Point2d> image_points;

		
		int c=0;
		int fine;
		
		std::string video=apriFile();

		std::string analisi=salvaRisultati();
		
			
			const char *perc;
			perc=video.c_str();

			
			

	
	    CvCapture* cap = cvCreateFileCapture(perc);
      
		IplImage* frame2;

		fine=stampainfo(video);
		c=0;

		frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
		std::string percors=v+"/shape_predictor_68_face_landmarks.dat";
        std::string nn2 = ReplaceAll(percors, "\\", "/");
		
		deserialize(nn2) >> pose_model;

		cv::Mat original, img_gray;

		cv::Mat sfondo=cv::imread("sfondo.jpg");


	while(1){

		cv::Mat visualizza=sfondo.clone();
		
        /* grab frame image, and retrieve */
        frame2 = cvQueryFrame(cap);

		//logo.copyTo(visualizza(cv::Rect(487,0,logo.cols, logo.rows)));

		putText(visualizza, "Premi ESC per interrompere l'analisi", cv::Point(10, 20), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 1);
		

		putText(visualizza, "Analisi in corso...", cv::Point(10, 50), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1);
		
      
        
     cv::Mat frame = cv::cvarrToMat(frame2);

	 int alt=frame.rows;
	 int largh=frame.cols;

	 int centro=largh/2;

	
	 
        original = frame.clone();

		cv_image<bgr_pixel> cimg(original);

		std::vector<cv::Point3d> model_points = get_3d_model_points();

            // Detect faces 
            std::vector<rectangle> faces = detector(cimg);
            // Find the pose of each face.
            std::vector<full_object_detection> shapes;
			full_object_detection shape;
            for (unsigned long i = 0; i < faces.size(); ++i)
            {  

               shape = pose_model(cimg, faces[i]);

               //cout << "number of parts: "<< shape.num_parts() << endl;
               //cout << "pixel position of first part:  " << shape.part(0) << endl;
               //cout << "pixel position of second part: " << shape.part(1) << endl;
               shapes.push_back(pose_model(cimg, faces[i]));

			   //cout<< shape.part(0).x() << " " << shape.part(0).y() << endl;
               //cout<< shape.part(1).x() << " " << shape.part(1).y() << endl;

               image_points = get_2d_image_points(shape);
               double focal_length = original.cols;
               cv::Mat camera_matrix = get_camera_matrix(focal_length, cv::Point2d(original.cols/2,original.rows/2));
               cv::Mat rotation_vector;
               cv::Mat rotation_matrix;
               cv::Mat translation_vector;

			   cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type);
                
               cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

               
			   std::vector<cv::Point3d> nose_end_point3D;
	  		   nose_end_point3D.push_back(cv::Point3d(0,0,1000.0));

			   cv::projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, nose_end_point2D);		
              
			
			}
       

			if(shapes.empty()==false){
			
			int x1=shape.part(0).x();
			int y1=shape.part(0).y();

			int x9=shape.part(8).x();
			int y9=shape.part(8).y();

			int x17=shape.part(16).x();
			int y17=shape.part(16).y();

			int x20=shape.part(19).x();
			int y20=shape.part(19).y();

			int x25=shape.part(24).x();
			int y25=shape.part(24).y();


			int punto_partenza_x=x1-15;

  int a[2];
  a[0]=y20;
  a[1]=y25;

  int minimo=*std::min_element(a,a+2);
  
  int punto_partenza_y=minimo-15;

  int punto_altezza_x=punto_partenza_x;
  int punto_altezza_y=y9+15;

  int altezza=punto_altezza_y-punto_partenza_y;

  int punto_larghezza_x=x17+15;
  int punto_larghezza_y=punto_partenza_y;

  int larghezza=punto_larghezza_x-punto_partenza_x;


  cv::Rect r(punto_partenza_x,punto_partenza_y,larghezza,altezza);


  if(punto_altezza_y>alt || punto_partenza_x<0 || punto_larghezza_x>largh || y20-15<0 || y25-15<0){

	  putText(visualizza, "Volto non rilevato..!!", cv::Point(10, 75), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1);
	  imshow("Carica e Analizza",visualizza);
	  char k = cvWaitKey(20);
      if(k==27) break;

  }else{

 
    //imwrite("C:/Users/Giuseppe/Desktop/1.immagine_originale.jpg",frame);
     cv::Mat faccia = original(r);
	 
		    
	 cv::Rect rectROI(0,0,faccia.cols,faccia.rows);
	cv::Mat mask(faccia.rows, faccia.cols, CV_8UC1, cv::Scalar(0));

		/***********************************************/

	cv_image<bgr_pixel> cimg2(faccia);

            // Detect faces 
            std::vector<rectangle> faces2 = detector(cimg2);
            // Find the pose of each face.
            std::vector<full_object_detection> shapes2;
			full_object_detection shape2;
            for (unsigned long i = 0; i < faces2.size(); ++i)
            {  

               shape2 = pose_model(cimg2, faces2[i]);

               //cout << "number of parts: "<< shape.num_parts() << endl;
               //cout << "pixel position of first part:  " << shape.part(0) << endl;
               //cout << "pixel position of second part: " << shape.part(1) << endl;
               shapes2.push_back(pose_model(cimg2, faces2[i]));

			   //cout<< shape.part(0).x() << " " << shape.part(0).y() << endl;
               //cout<< shape.part(1).x() << " " << shape.part(1).y() << endl;

              }

		if(shapes2.empty()==false){
		
			int xx1=shape2.part(0).x();
			int yy1=shape2.part(0).y();

			int xx2=shape2.part(1).x();
			int yy2=shape2.part(1).y();

			int xx3=shape2.part(2).x();
			int yy3=shape2.part(2).y();

			int xx4=shape2.part(3).x();
			int yy4=shape2.part(3).y();

			int xx5=shape2.part(4).x();
			int yy5=shape2.part(4).y();

			int xx6=shape2.part(5).x();
			int yy6=shape2.part(5).y();

			int xx7=shape2.part(6).x();
			int yy7=shape2.part(6).y();

			int xx8=shape2.part(7).x();
			int yy8=shape2.part(7).y();

			int xx9=shape2.part(8).x();
			int yy9=shape2.part(8).y();

			int xx10=shape2.part(9).x();
			int yy10=shape2.part(9).y();

			int xx11=shape2.part(10).x();
			int yy11=shape2.part(10).y();

			int xx12=shape2.part(11).x();
			int yy12=shape2.part(11).y();

			int xx13=shape2.part(12).x();
			int yy13=shape2.part(12).y();

			int xx14=shape2.part(13).x();
			int yy14=shape2.part(13).y();

			int xx15=shape2.part(14).x();
			int yy15=shape2.part(14).y();

			int xx16=shape2.part(15).x();
			int yy16=shape2.part(15).y();

			int xx17=shape2.part(16).x();
			int yy17=shape2.part(16).y();

			int xx18=shape2.part(17).x();
			int yy18=shape2.part(17).y();

			int xx19=shape2.part(18).x();
			int yy19=shape2.part(18).y();

			int xx20=shape2.part(19).x();
			int yy20=shape2.part(19).y();

			int xx21=shape2.part(20).x();
			int yy21=shape2.part(20).y();

			int xx22=shape2.part(21).x();
			int yy22=shape2.part(21).y();

			int xx23=shape2.part(22).x();
			int yy23=shape2.part(22).y();

			int xx24=shape2.part(23).x();
			int yy24=shape2.part(23).y();

			int xx25=shape2.part(24).x();
			int yy25=shape2.part(24).y();

			int xx26=shape2.part(25).x();
			int yy26=shape2.part(25).y();

			int xx27=shape2.part(26).x();
			int yy27=shape2.part(26).y();
			

			cv::Point P1(xx1,yy1);
			cv::Point P2(xx2,yy2);
			cv::Point P3(xx3,yy3);
			cv::Point P4(xx4,yy4);
			cv::Point P5(xx5,yy5);
			cv::Point P6(xx6,yy6);
			cv::Point P7(xx7,yy7);
			cv::Point P8(xx8,yy8);
			cv::Point P9(xx9,yy9);
			cv::Point P10(xx10,yy10);
			cv::Point P11(xx11,yy11);
			cv::Point P12(xx12,yy12);
			cv::Point P13(xx13,yy13);
			cv::Point P14(xx14,yy14);
			cv::Point P15(xx15,yy15);
			cv::Point P16(xx16,yy16);
			cv::Point P17(xx17,yy17);
			cv::Point P18(xx18,yy18);
			cv::Point P19(xx19,yy19);
			cv::Point P20(xx20,yy20);
			cv::Point P21(xx21,yy21);
			cv::Point P22(xx22,yy22);
			cv::Point P23(xx23,yy23);
			cv::Point P24(xx24,yy24);
			cv::Point P25(xx25,yy25);
			cv::Point P26(xx26,yy26);
			cv::Point P27(xx27,yy27);

std::vector< std::vector<cv::Point> >  co_ordinates;
   co_ordinates.push_back(std::vector<cv::Point>());
   co_ordinates[0].push_back(P1);
   co_ordinates[0].push_back(P2);
   co_ordinates[0].push_back(P3);
   co_ordinates[0].push_back(P4);
   co_ordinates[0].push_back(P5);
   co_ordinates[0].push_back(P6);
   co_ordinates[0].push_back(P7);
   co_ordinates[0].push_back(P8);
   co_ordinates[0].push_back(P9);
   co_ordinates[0].push_back(P10);
   co_ordinates[0].push_back(P11);
   co_ordinates[0].push_back(P12);
   co_ordinates[0].push_back(P13);
   co_ordinates[0].push_back(P14);
   co_ordinates[0].push_back(P15);
   co_ordinates[0].push_back(P16);
   co_ordinates[0].push_back(P17);
   co_ordinates[0].push_back(P27);
   co_ordinates[0].push_back(P26);
   co_ordinates[0].push_back(P25);
   co_ordinates[0].push_back(P24);
   co_ordinates[0].push_back(P23);
   co_ordinates[0].push_back(P22);
   co_ordinates[0].push_back(P21);
   co_ordinates[0].push_back(P20);
   co_ordinates[0].push_back(P19);
   co_ordinates[0].push_back(P18);
   drawContours( mask,co_ordinates,0, cv::Scalar(255),CV_FILLED, 8 );

   cv::Mat srcROI=faccia(rectROI);
   cv::Mat dst1;

   srcROI.copyTo(dst1,mask);

   cvtColor(dst1, img_gray, CV_RGB2GRAY);
   

     cv::resize(img_gray, img_gray, cv::Size(48,64) );
	 
	
        //extract feature
        cv::HOGDescriptor d( cv::Size(48,64), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 7);
        std::vector<float> descriptorsValues;
        std::vector<cv::Point> locations;
        d.compute( img_gray, descriptorsValues, cv::Size(0,0), cv::Size(0,0), locations);

	
        //2d vector to Mat
        int row=descriptorsValues.size();
        int col=descriptorsValues.size();
		/*std::string roww;
		std::stringstream rowww;
		rowww << row;
		roww=rowww.str();

		std::string coll;
		std::stringstream colll;
		colll << col;
		coll=colll.str();
		putText(visualizza, "descript size row="+roww+", col="+coll, cv::Point(10, 50), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1);*/
        
		//printf("descript size row=%d, col=%d\n", row, col);

        //printf("descript element %f\n", descriptorsValues[12]);


        cv::Mat fm = cv::Mat(descriptorsValues);

        cv::Mat B = fm.t();

		float **mat = new float*[B.rows];
        for(int i=0;i<B.rows;i++){
           mat[i]=new float[B.cols];
        }
		
 
		for(int i=0;i<B.rows;i++){
			for(int j=0;j<B.cols;j++){
				mat[i][j] = B.at<float>(i,j);         
			}
		}

		svm_node* testnode = Malloc(svm_node,B.cols+1);
		for (int row=0;row <B.rows; row++){
			for(int col=0;col<B.cols;col++){
				testnode[col].index = col;
				testnode[col].value = mat[row][col];
			}
        testnode[B.cols].index = -1;
		}



		 predictions = svm_predict_probability(model, testnode, prob_est);
         //printf("%f \n %f %f %f %f %f %f\n\n", predictions, prob_est[0], prob_est[1],prob_est[2], prob_est[3],prob_est[4], prob_est[5]);
		 /*printf("\nprediction    %f\n", predictions);
		 printf("\nentusiasmo      %f\n", prob_est[0]);
		 printf("interesse       %f\n", prob_est[1]);
		 printf("sorpresa        %f\n", prob_est[2]);
		 printf("curiosita'      %f\n", prob_est[3]);
		 printf("concentrazione  %f\n", prob_est[4]);
		 printf("attenzione      %f\n", prob_est[5]);
		 printf("delusione       %f\n", prob_est[6]);
		 printf("noia            %f\n", prob_est[7]);
		 printf("perplessita'    %f\n", prob_est[8]);
		 printf("fastidio        %f\n", prob_est[9]);
		 printf("frustrazione    %f\n", prob_est[10]);*/

		 std::string p1 = std::to_string(predictions);
		 putText(visualizza, "Prediction: "+p1, cv::Point(10, 75), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1);
		 std::string e1 = std::to_string(prob_est[0]);
		 putText(visualizza, "Entusiasmo: "+e1, cv::Point(10, 100), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1);
		 std::string i1 = std::to_string(prob_est[1]);
		 putText(visualizza, "Interesse: "+i1, cv::Point(10, 120), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1);
	     std::string s1 = std::to_string(prob_est[2]);
		 putText(visualizza, "Sorpresa: "+s1, cv::Point(10, 140), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1);
	     std::string c1 = std::to_string(prob_est[3]);
		 putText(visualizza, "Curiosita': "+c1, cv::Point(10, 160), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1);
	     std::string cc1 = std::to_string(prob_est[4]);
		 putText(visualizza, "Concentrazione: "+cc1, cv::Point(10, 180), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1);
	     std::string a1 = std::to_string(prob_est[5]);
		 putText(visualizza, "Attenzione: "+a1, cv::Point(10, 200), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1);
	     std::string d1 = std::to_string(prob_est[6]);
		 putText(visualizza, "Delusione: "+d1, cv::Point(10, 220), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1);
	     std::string n1 = std::to_string(prob_est[7]);
		 putText(visualizza, "Noia: "+n1, cv::Point(10, 240), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1);
	     std::string pp1 = std::to_string(prob_est[8]);
		 putText(visualizza, "Perplessita': "+pp1, cv::Point(10, 260), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1);
	     std::string f1 = std::to_string(prob_est[9]);
		 putText(visualizza, "Fastidio: "+f1, cv::Point(10, 280), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1);
	     std::string ff1 = std::to_string(prob_est[10]);
		 putText(visualizza, "Frustrazione: "+ff1, cv::Point(10, 300), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1);
		 
		 
        
	

	 //pred = svm_predict(model, testnode);
     //printf("%f\n", pred);

	 float entusiasmo=prob_est[0]*100;
	 float interesse=prob_est[1]*100;
	 float sorpresa=prob_est[2]*100;
	 float curiosita=prob_est[3]*100;
	 float concentrazione=prob_est[4]*100;
	 float attenzione=prob_est[5]*100;
	 float delusione=prob_est[6]*100;
	 float noia=prob_est[7]*100;
	 float perplessita=prob_est[8]*100;
	 float fastidio=prob_est[9]*100;
	 float frustrazione=prob_est[10]*100;

	 cout << fixed << setprecision(2) << entusiasmo;
	 cout << fixed << setprecision(2) << interesse;
	 cout << fixed << setprecision(2) << sorpresa;
	 cout << fixed << setprecision(2) << curiosita;
	 cout << fixed << setprecision(2) << concentrazione;
	 cout << fixed << setprecision(2) << attenzione;
	 cout << fixed << setprecision(2) << delusione;
	 cout << fixed << setprecision(2) << noia;
	 cout << fixed << setprecision(2) << perplessita;
	 cout << fixed << setprecision(2) << fastidio;
	 cout << fixed << setprecision(2) << frustrazione;

	 //cout<<endl<<"percentuale completa: "<<entusiasmo+interesse+sorpresa+curiosita+concentrazione+attenzione+delusione+noia+perplessita+fastidio+frustrazione<<endl;

	 float entu=entusiasmo;
	 int num1=(int) entu;
	 std::string percentuale1;
     std::stringstream oss1;
	 cv::String percentuale11;
	 int num11=num_cifre(num1);
	 if(num11==1){
		 oss1 << std::setprecision(3) << entu;
		 percentuale1=oss1.str();
	     percentuale11=" "+percentuale1+"%";
	 }else{
     oss1 << std::setprecision(4) << entu;
     percentuale1=oss1.str();
	 percentuale11=percentuale1+"%";
	 }
     float perc_entusiasmo=(entu*150)/100;
	 

	 float inter=interesse;
	 int num2=(int) inter;
	 std::string percentuale2;
     std::stringstream oss2;
	 cv::String percentuale22;
	 int num22=num_cifre(num2);
	 if(num22==1){
		 oss2 << std::setprecision(3) << inter;
		 percentuale2=oss2.str();
	     percentuale22=" "+percentuale2+"%";
	 }else{
     oss2 << std::setprecision(4) << inter;
     percentuale2=oss2.str();
	 percentuale22=percentuale2+"%";
	 }
     float perc_interesse=(inter*150)/100;


	 float sorp=sorpresa;
	 int num3=(int) sorp;
	 std::string percentuale3;
     std::stringstream oss3;
	 cv::String percentuale33;
	 int num33=num_cifre(num3);
	 if(num33==1){
		 oss3 << std::setprecision(3) << sorp;
		 percentuale3=oss3.str();
	     percentuale33=" "+percentuale3+"%";
	 }else{
     oss3 << std::setprecision(4) << sorp;
     percentuale3=oss3.str();
	 percentuale33=percentuale3+"%";
	 }
     float perc_sorpresa=(sorp*150)/100;


	 float curio=curiosita;
	 int num4=(int) curio;
	 std::string percentuale4;
     std::stringstream oss4;
	 cv::String percentuale44;
	 int num44=num_cifre(num4);
	 if(num44==1){
		 oss4 << std::setprecision(3) << curio;
		 percentuale4=oss4.str();
	     percentuale44=" "+percentuale4+"%";
	 }else{
     oss4 << std::setprecision(4) << curio;
     percentuale4=oss4.str();
	 percentuale44=percentuale4+"%";
	 }
     float perc_curiosita=(curio*150)/100;


	 float concen=concentrazione;
	 int num5=(int) concen;
	 std::string percentuale5;
     std::stringstream oss5;
	 cv::String percentuale55;
	 int num55=num_cifre(num5);
	 if(num55==1){
		 oss5 << std::setprecision(3) << concen;
		 percentuale5=oss5.str();
	     percentuale55=" "+percentuale5+"%";
	 }else{
     oss5 << std::setprecision(4) << concen;
     percentuale5=oss5.str();
	 percentuale55=percentuale5+"%";
	 }
     float perc_concentrazione=(concen*150)/100;


	 float atten=attenzione;
	 int num6=(int) atten;
	 std::string percentuale6;
     std::stringstream oss6;
	 cv::String percentuale66;
	 int num66=num_cifre(num6);
	 if(num66==1){
		 oss6 << std::setprecision(3) << atten;
		 percentuale6=oss6.str();
	     percentuale66=" "+percentuale6+"%";
	 }else{
     oss6 << std::setprecision(4) << atten;
     percentuale6=oss6.str();
	 percentuale66=percentuale6+"%";
	 }
     float perc_attenzione=(atten*150)/100;


	 float delus=delusione;
	 int num7=(int) delus;
	 std::string percentuale7;
     std::stringstream oss7;
	 cv::String percentuale77;
	 int num77=num_cifre(num7);
	 if(num77==1){
		 oss7 << std::setprecision(3) << delus;
		 percentuale7=oss7.str();
	     percentuale77=" "+percentuale7+"%";
	 }else{
     oss7 << std::setprecision(4) << delus;
     percentuale7=oss7.str();
	 percentuale77=percentuale7+"%";
	 }
     float perc_delusione=(delus*150)/100;


	 float noi=noia;
	 int num8=(int) noi;
	 std::string percentuale8;
     std::stringstream oss8;
	 cv::String percentuale88;
	 int num88=num_cifre(num8);
	 if(num88==1){
		 oss8 << std::setprecision(3) << noi;
		 percentuale8=oss8.str();
	     percentuale88=" "+percentuale8+"%";
	 }else{
     oss8 << std::setprecision(4) << noi;
     percentuale8=oss8.str();
	 percentuale88=percentuale8+"%";
	 }
     float perc_noia=(noi*150)/100;


	 float perpl=perplessita;
	 int num9=(int) perpl;
	 std::string percentuale9;
     std::stringstream oss9;
	 cv::String percentuale99;
	 int num99=num_cifre(num9);
	 if(num99==1){
		 oss9 << std::setprecision(3) << perpl;
		 percentuale9=oss9.str();
	     percentuale99=" "+percentuale9+"%";
	 }else{
     oss9 << std::setprecision(4) << perpl;
     percentuale9=oss9.str();
	 percentuale99=percentuale9+"%";
	 }
     float perc_perplessita=(perpl*150)/100;


	 float fast=fastidio;
	 int num10=(int) fast;
	 std::string percentuale10;
     std::stringstream oss10;
	 cv::String percentuale1010;
	 int num1010=num_cifre(num10);
	 if(num1010==1){
		 oss10 << std::setprecision(3) << fast;
		 percentuale10=oss10.str();
	     percentuale1010=" "+percentuale10+"%";
	 }else{
     oss10 << std::setprecision(4) << fast;
     percentuale10=oss10.str();
	 percentuale1010=percentuale10+"%";
	 }
     float perc_fastidio=(fast*150)/100;


	 float frustr=frustrazione;
	 int num0=(int) frustr;
	 std::string percentuale0;
     std::stringstream oss0;
	 cv::String percentuale00;
	 int num00=num_cifre(num0);
	 if(num00==1){
		 oss0 << std::setprecision(3) << frustr;
		 percentuale0=oss0.str();
	     percentuale00=" "+percentuale0+"%";
	 }else{
     oss0 << std::setprecision(4) << frustr;
     percentuale0=oss0.str();
	 percentuale00=percentuale0+"%";
	 }
     float perc_frustrazione=(frustr*150)/100;
		 

		  string sg;

	       //cout <<  nose_end_point2D[0];

				//cout <<" "<<nose_end_point2D[0].x;

				if(nose_end_point2D[0].x >= centro){
					sg="sinistra";
				}else{
					sg="destra";
				}

				putText(visualizza, "Sguardo: "+sg, cv::Point(10, 330), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1);
		 
				//cout << " " << sg << endl;


	 scrittura(analisi,percentuale11,percentuale22,percentuale33,percentuale44,percentuale55,percentuale66,percentuale77,percentuale88,percentuale99,percentuale1010,percentuale00,sg);
		

        //imshow("face_recognizer", original);
       
		//scrittura(analisi,label,percentuale2);
	
		
		}
        }

		}else{
			
		
			putText(visualizza, "Volto non rilevato..!!", cv::Point(10, 75), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1);
			imshow("Carica e Analizza",visualizza);
			char k = cvWaitKey(20);
			if(k==27) break;

		}

		

		
		//cout<<"Frame (" <<c;
		c++;
		
		//cout<<"/"<<fine;

		//cout<<")"<<endl;

		std::string cccc = std::to_string(c);
		std::string finee = std::to_string(fine);

		putText(visualizza, "Frame ("+cccc+"/"+finee+")", cv::Point(10, 370), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1);
		 	

		
		imshow("Carica e Analizza",visualizza);
		char k = cvWaitKey(20);
        if(k==27) break;
		
		if(c==fine){
			
			putText(visualizza, "Fine Analisi..!!", cv::Point(10, 410), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1);
			imshow("Carica e Analizza",visualizza);
			cv::waitKey(20);
			//cout<<endl<<"fineeee";
			break;

		}
		
		   
    }




}









std::string Emotion::apriFile(){
	QString filename = QFileDialog::getOpenFileName( 
        this, 
        tr("Carica Video"), 
        QDir::currentPath(), 
        tr("AVI (*.avi*)") );
	std::string risultato=filename.toStdString();
	return risultato;
}


std::string Emotion::salvaFile(){
	QString filename = QFileDialog::getSaveFileName( 
        this, 
        tr("Salva Video"), 
        QDir::currentPath(), 
        tr("AVI (*.avi)") );
	std::string risultato=filename.toStdString();
	return risultato;
}

std::string Emotion::salvaRisultati(){
	QString filename = QFileDialog::getSaveFileName( 
        this, 
        tr("Salva Risultati"), 
        QDir::currentPath(), 
        tr("TXT (*.txt)") );
	std::string risultato=filename.toStdString();
	return risultato;
}
